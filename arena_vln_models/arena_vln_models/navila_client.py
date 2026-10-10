"""Arena lifecycle and external timed-action conversion for the unchanged NaVILA service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

from arena_vln_models.navila_control import FrameHistory, TCPModel, TimedAction, check_velocity_limits, parse_action


def main(argv=None):
    import numpy as np
    from PIL import Image as PILImage
    import yaml
    import rclpy
    from rclpy.node import Node
    from rclpy.clock import Clock, ClockType
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from ament_index_python.packages import get_package_share_directory
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock as ClockMessage
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Int16, Empty
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--robot', default='Ai2_Bot2')
    parser.add_argument('--run-token', default='manual')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--ready-probe', action='store_true')
    args = parser.parse_args(argv)
    profile = yaml.safe_load(Path(args.profile).read_text())
    robot = yaml.safe_load((Path(get_package_share_directory('arena_robots')) /
                           'robots' / args.robot / 'model_params.yaml').read_text())
    check_velocity_limits(robot)
    if args.smoke:
        assert parse_action('The next action is move forward 75 cm.').duration_sec == 1.5
        print('NaVILA client dependencies, native action parser and direct robot limits: PASS')
        return 0
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    namespace = f'/task_generator_node/{args.robot}'
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    if args.ready_probe:
        probe = Node('navila_ready_probe')
        seen = []
        probe.create_subscription(String, namespace + '/navila/status', lambda m: seen.append(json.loads(m.data)), latched)
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                rclpy.spin_once(probe, timeout_sec=.1)
                if any(r.get('backend') == 'navila' and r.get('backend_ready')
                       and r.get('run_token') == args.run_token and 0 <= time.time() - r.get('wall_time', 0) < 2.5 for r in seen):
                    print('NaVILA ROS client ready')
                    return 0
            raise RuntimeError('fresh owned NaVILA client status was not observed')
        finally:
            probe.destroy_node(); rclpy.shutdown()

    def stamp(msg):
        return msg.sec * 1_000_000_000 + msg.nanosec

    class NaVILAClient(Node):
        def __init__(self):
            super().__init__('navila_client', namespace=namespace)
            address = urlparse(profile['server']['endpoint'])
            self.model = TCPModel(address.hostname, address.port, profile['client']['request_timeout_sec'])
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='navila-tcp')
            self.future = None
            self.pending = None
            self.history = FrameHistory()
            self.generation = self.request = 0
            self.episode = self.ready_episode = None
            self.active = self.terminal = False
            self.termination_reason = ''
            self.instruction = ''
            self.sim = 0.
            self.clock_ns = None
            self.clock_wall = self.odom_wall = self.rgb_wall = float('-inf')
            self.pose = self.last_odom_stamp = None
            self.action = None
            self.action_distance = self.action_yaw = 0.
            self.action_start_pose = None
            self.last_request_stamp = -1
            self.next_frame_after = -1
            self.last_heartbeat = self.last_zero = 0.
            self.last_tick = time.monotonic()
            self.last_command = (0., 0.)
            self.timeout = float(profile['client']['command_timeout_sec'])
            self.base_frame = args.robot + '/' + robot.get('robot_base_frame', 'base_link')
            self.odom_frame = args.robot + '/' + robot.get('robot_odom_frame', 'odom')
            self.cmd = self.create_publisher(Twist, 'cmd_vel', 10)
            qos = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.status = self.create_publisher(String, 'navila/status', qos)
            self.output = self.create_publisher(String, 'navila/model_output', 10)
            self.debug = self.create_publisher(Image, 'navila/debug_image', qos_profile_sensor_data)
            self.create_subscription(Image, profile['client']['rgb_topic'], self.on_image, qos_profile_sensor_data)
            self.create_subscription(Odometry, profile['client']['odom_topic'], self.on_odom, qos_profile_sensor_data)
            self.create_subscription(ClockMessage, '/clock', self.on_clock, qos_profile_sensor_data)
            self.create_subscription(Int16, '/task_generator_node/task_reset', self.on_reset, 10)
            self.create_subscription(String, '/task_generator_node/eval_ready', self.on_ready, latched)
            self.create_subscription(String, '/task_generator_node/vln_instruction', self.on_instruction, latched)
            self.create_subscription(String, '/task_generator_node/episode_outcome', self.on_outcome, latched)
            self.create_subscription(Empty, '/task_generator_node/finished', lambda _: self.clear('task_finished', terminal=True), 10)
            self.create_timer(.025, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

        def emit(self, status, **fields):
            now = time.monotonic()
            record = dict(status=status, backend='navila', backend_ready=True, ready=True,
                controller='navila_action_velocity', execution_mode='external_direct_cmd_vel',
                episode=self.episode, episode_started=self.active, generation=self.generation,
                request_id=self.request, run_token=args.run_token, sim_time=self.sim, wall_time=time.time(),
                debug={'topics': {'rgb': profile['client']['rgb_topic'], 'odom': profile['client']['odom_topic']},
                       'sensor_ages_sec': {'rgb': min(1e9, now-self.rgb_wall), 'odom': min(1e9, now-self.odom_wall)},
                       'stale_after_sec': 2., 'tf_ready': self.pose is not None})
            record.update(fields)
            self.status.publish(String(data=json.dumps(record, allow_nan=False)))

        def velocity(self, v, w):
            msg = Twist(); msg.linear.x = float(v); msg.angular.z = float(w)
            self.cmd.publish(msg)
            self.last_command = (v, w)
            self.emit('navila_command', linear_x=v, angular_z=w)

        def zero(self):
            now = time.monotonic()
            if self.last_command != (0., 0.) or now - self.last_zero >= .05:
                self.velocity(0., 0.)
                self.last_zero = now

        def cancel(self, reason):
            if self.action is not None:
                self.emit('action_aborted', reason=reason, action=self.action.action.record(),
                          actual_distance_m=self.action_distance, actual_yaw_rad=self.action_yaw)
            self.action = None
            self.model.abort()
            self.generation += 1
            self.zero()

        def clear(self, reason, terminal=False):
            self.cancel(reason)
            self.history.reset()
            self.last_request_stamp = self.next_frame_after = -1
            self.active = False
            self.terminal = terminal
            self.termination_reason = reason if terminal else ''

        def reset_episode(self, episode):
            if self.episode is not None and episode <= self.episode:
                return False
            self.clear('task_reset')
            self.episode, self.ready_episode = episode, None
            self.instruction = ''
            self.pose = self.last_odom_stamp = None
            self.odom_wall = self.rgb_wall = float('-inf')
            self.emit('resetting', ready=False, episode_started=False)
            return True

        def on_reset(self, msg):
            self.reset_episode(msg.data)

        def on_ready(self, msg):
            try:
                value = json.loads(msg.data)
            except ValueError:
                return
            episode = value.get('episode')
            if value.get('stage') != 'episode' or type(episode) is not int:
                return
            if self.episode is not None and episode < self.episode:
                return
            self.reset_episode(episode)
            if value.get('ready'):
                self.ready_episode = episode
                self.maybe_start()
            else:
                self.clear('episode_not_ready', terminal=self.terminal or (value.get('details') or {}).get('reason') == 'episode_finished')
                self.ready_episode = None

        def on_instruction(self, msg):
            instruction = msg.data.strip()
            if instruction != self.instruction:
                if self.active:
                    self.clear('instruction_changed')
                self.instruction = instruction
            self.maybe_start()

        def maybe_start(self):
            if (not self.active and not self.terminal and self.episode is not None
                    and self.ready_episode == self.episode and self.instruction):
                self.history.reset()
                self.active = True
                self.emit('episode_ready', instruction=self.instruction)

        def on_outcome(self, msg):
            try:
                value = json.loads(msg.data)
            except ValueError:
                return
            if value.get('episode_index') == self.episode:
                self.clear('episode_outcome', terminal=True)

        def on_clock(self, msg):
            ns = stamp(msg.clock)
            if self.clock_ns is not None and ns < self.clock_ns:
                self.clear('clock_rewind')
                self.episode = self.ready_episode = None
                self.instruction = ''
                self.pose = self.last_odom_stamp = None
                self.odom_wall = self.rgb_wall = float('-inf')
            if self.clock_ns != ns:
                self.clock_wall = time.monotonic()
            self.clock_ns, self.sim = ns, ns / 1e9
            self.tick()

        def on_image(self, msg):
            if msg.encoding not in ('rgb8', 'rgba8', 'bgr8', 'bgra8'):
                return
            ns = stamp(msg.header.stamp)
            if self.clock_ns is not None and (ns > self.clock_ns + 100_000_000 or ns < self.clock_ns - 2_000_000_000):
                return
            channels = 4 if 'a' in msg.encoding else 3
            rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
            rgb = rows[:, :msg.width*channels].reshape(msg.height, msg.width, channels)[:, :, :3]
            if msg.encoding.startswith('bgr'):
                rgb = rgb[:, :, ::-1]
            if self.active and not self.history.offer(ns, PILImage.fromarray(rgb)):
                return
            self.rgb_wall = time.monotonic()
            self.debug.publish(msg)

        def on_odom(self, msg):
            ns = stamp(msg.header.stamp)
            if msg.header.frame_id != self.odom_frame or msg.child_frame_id != self.base_frame:
                return
            if self.last_odom_stamp is not None and ns <= self.last_odom_stamp:
                return
            q = msg.pose.pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
            if self.action is not None and self.pose is not None:
                self.action_distance += math.hypot(pose[0]-self.pose[0], pose[1]-self.pose[1])
                delta = pose[2]-self.pose[2]
                self.action_yaw += math.atan2(math.sin(delta), math.cos(delta))
            self.pose, self.last_odom_stamp = pose, ns
            self.odom_wall = time.monotonic()

        def receive_result(self):
            if self.future is None or not self.future.done():
                return
            future, pending = self.future, self.pending
            self.future = self.pending = None
            if pending['generation'] != self.generation or not self.active or self.terminal:
                self.emit('response_discarded', reason='old_generation')
                return
            try:
                text = future.result()
                latency = (time.monotonic()-pending['started']) * 1000
                self.emit('navila_inference', text=text, latency_ms=latency,
                          capture_stamp_ns=pending['capture'], input_stamps=pending['stamps'])
                self.output.publish(String(data=json.dumps({'text': text, 'request_id': self.request})))
                action = parse_action(text)
                if latency > 2000 or self.sim - pending['capture']/1e9 > 2.:
                    self.emit('action_rejected', reason='expired_observation')
                    self.next_frame_after = self.clock_ns or 0
                    return
                if action.kind == 'stop':
                    self.emit('model_stop', text=text)
                    self.clear('model_stop', terminal=True)
                    self.emit('stop', terminal=True, termination_reason='model_stop')
                    return
                self.action = TimedAction(action, self.sim)
                self.action_start_pose = self.pose
                self.action_distance = self.action_yaw = 0.
                self.emit('action_accepted', action=action.record(), capture_stamp_ns=pending['capture'],
                          start_pose=self.pose, end_sim=self.action.end)
            except Exception as exc:
                self.emit('error', reason='model_request_or_parse', error=str(exc))
                self.clear('model_error', terminal=True)
                self.emit('error', terminal=True, termination_reason='model_error', ready=False)

        def tick(self):
            try:
                self.control_tick()
            finally:
                now = time.monotonic()
                if now - self.last_heartbeat >= .5:
                    # RobotManager keeps only its latest status sample. Repeat
                    # lifecycle state, after command diagnostics, like LightNav.
                    if self.terminal:
                        self.emit('stop', terminal=True, termination_reason=self.termination_reason)
                    elif self.active:
                        self.emit('episode_ready', episode_started=True)
                    else:
                        self.emit('backend_ready')
                    self.last_heartbeat = now

        def control_tick(self):
            now = time.monotonic()
            stalled = now - self.last_tick > self.timeout
            self.last_tick = now
            if now - self.last_heartbeat >= .5:
                if self.count_publishers('cmd_vel') > 1:
                    self.clear('competing_cmd_vel_publisher', terminal=True)
                    raise RuntimeError('another cmd_vel publisher appeared')
            fresh = (not stalled and now-self.clock_wall < self.timeout
                     and now-self.odom_wall < self.timeout and now-self.rgb_wall < 2.)
            if not self.active or self.terminal or not fresh:
                if self.action is not None or (self.pending is not None and self.pending['generation'] == self.generation):
                    self.cancel('watchdog_or_episode_inactive')
                    self.next_frame_after = self.clock_ns or 0
                self.zero()
                self.receive_result()
                return
            self.receive_result()
            if self.terminal:
                self.zero()
                return
            if self.action is not None:
                velocity = self.action.command(self.sim)
                if velocity == (0., 0.):
                    self.zero()
                    self.emit('action_completed', action=self.action.action.record(),
                        start_sim=self.action.start, elapsed_sim=self.sim-self.action.start,
                        actual_distance_m=self.action_distance, actual_yaw_rad=self.action_yaw,
                        start_pose=self.action_start_pose, end_pose=self.pose)
                    self.action = None
                    self.next_frame_after = self.clock_ns or 0
                elif velocity is not None:
                    self.velocity(*velocity)
                return
            self.zero()
            if self.future is None and self.history.latest is not None:
                capture = self.history.latest[0]
                if capture > max(self.last_request_stamp, self.next_frame_after):
                    images, stamps = self.history.sample()
                    self.request += 1
                    self.last_request_stamp = capture
                    self.pending = dict(generation=self.generation, capture=capture, stamps=stamps, started=now)
                    self.emit('request_started', capture_stamp_ns=capture, input_stamps=stamps)
                    self.future = self.pool.submit(self.model.generate, images, self.instruction)

    node = NaVILAClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.clear('shutdown', terminal=True)
        node.pool.shutdown(wait=True, cancel_futures=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
