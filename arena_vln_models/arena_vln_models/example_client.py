"""Example native FollowPath producer; no model or velocity controller.

Use --points '[[0,0,0],[1,0,0]]' for forward motion, or
--points '[[0,0,0],[0,0,0.523599]]' for a 30 degree goal rotation.
Points are demo offsets from the first odometry sample of each released episode.
"""
import argparse
import json
import math
import time


def main(argv=None):
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry, Path
    from nav2_msgs.action import FollowPath
    from rclpy.action import ActionClient
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='/task_generator_node/Ai2_Bot2')
    parser.add_argument('--task-namespace', default='/task_generator_node')
    parser.add_argument('--points', default='[[0,0,0],[1,0,0]]')
    parser.add_argument('--replacement-points', default='')
    parser.add_argument('--replace-after', type=float, default=1.0)
    args, ros_args = parser.parse_known_args(argv)
    points = json.loads(args.points)
    replacement = json.loads(args.replacement_points) if args.replacement_points else None
    for seq in (points, replacement):
        if seq is not None and (not isinstance(seq, list) or not seq or any(
                not isinstance(p, list) or len(p) != 3 or
                any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in p) for p in seq)):
            parser.error('points must be a nonempty list of finite [x, y, yaw] values')
    # Keep the context alive during Ctrl-C cleanup so native cancellation and
    # the terminal result can be awaited before destroying the action client.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)

    class Client(Node):
        def __init__(self):
            super().__init__('waypoint_example_client', namespace=args.namespace)
            self.action = ActionClient(self, FollowPath, 'follow_path')
            self.pub = self.create_publisher(String, 'model/status', 10)
            self.identity = None
            self.released = self.sent = self.stopped = self.replaced = False
            self.pose = self.origin = self.goal_handle = None
            self.sequence = 0
            self.status = 'ready'
            self.started = 0.0
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, args.task_namespace.rstrip('/') + '/eval_ready', self.on_episode, latched)
            self.create_subscription(Odometry, 'odom', lambda msg: setattr(self, 'pose', msg), qos_profile_sensor_data)
            self.create_timer(.1, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.create_timer(1., self.heartbeat, clock=Clock(clock_type=ClockType.STEADY_TIME))

        def on_episode(self, msg):
            data = json.loads(msg.data)
            if data.get('stage') != 'episode' or not data.get('session_id'):
                return
            identity = (data['session_id'], data['episode'])
            if identity != self.identity:
                self.sequence += 1
                if self.goal_handle:
                    self.goal_handle.cancel_goal_async()
                self.identity, self.pose, self.origin, self.goal_handle = identity, None, None, None
                self.sent = self.stopped = self.replaced = False
                self.status = 'ready'
            self.released = bool(data['ready'])
            if not self.released and self.goal_handle:
                self.goal_handle.cancel_goal_async()

        def heartbeat(self):
            if self.identity and self.pose is not None:
                self.pub.publish(String(data=json.dumps(dict(
                    session_id=self.identity[0], episode=self.identity[1], status=self.status))))

        def submit(self, offsets):
            odom = self.origin
            p, q = odom.pose.pose.position, odom.pose.pose.orientation
            yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
            path = Path()
            path.header = odom.header
            for x, y, angle in offsets:
                point = PoseStamped()
                point.header = path.header
                point.pose.position.x = p.x + math.cos(yaw)*x - math.sin(yaw)*y
                point.pose.position.y = p.y + math.sin(yaw)*x + math.cos(yaw)*y
                point.pose.orientation.z = math.sin((yaw + angle) / 2)
                point.pose.orientation.w = math.cos((yaw + angle) / 2)
                path.poses.append(point)
            goal = FollowPath.Goal(path=path, controller_id='FollowPath', goal_checker_id='goal_checker',
                                   progress_checker_id='progress_checker')
            self.sequence += 1
            identity, sequence = self.identity, self.sequence
            self.status, self.sent = 'running', True
            future = self.action.send_goal_async(goal)

            def accepted(future):
                handle = future.result()
                if identity != self.identity or sequence != self.sequence or not self.released:
                    if handle.accepted:
                        handle.cancel_goal_async()
                    return
                if not handle.accepted:
                    self.sent = False
                    return
                self.goal_handle = handle

                def finished(result):
                    if identity != self.identity or sequence != self.sequence or not self.released:
                        return
                    state = result.result().status
                    if state == GoalStatus.STATUS_SUCCEEDED:
                        self.status, self.stopped = 'stop', True
                    elif state == GoalStatus.STATUS_CANCELED:
                        self.status, self.sent = 'ready', False
                    else:
                        self.status, self.stopped = 'error', True
                    self.heartbeat()
                handle.get_result_async().add_done_callback(finished)
            future.add_done_callback(accepted)

        def tick(self):
            if not self.released or self.stopped or self.pose is None or not self.action.server_is_ready():
                return
            if not self.sent:
                self.origin = self.pose
                self.started = time.monotonic()
                self.submit(points)
            elif replacement and not self.replaced and time.monotonic() - self.started >= args.replace_after:
                self.replaced = True
                self.submit(replacement)

    node = Client()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.goal_handle:
            future = node.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, future, timeout_sec=2.)
            rclpy.spin_until_future_complete(node, node.goal_handle.get_result_async(), timeout_sec=3.)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
