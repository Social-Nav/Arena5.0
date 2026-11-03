#!/usr/bin/env python3
"""Versioned ProxMPC diagnostic recorder for the A-track.

Records only real ROS topics and real sim-time stamps.  It never synthesizes
predictions from tracker scalars and never treats callback-arrival age as
control-cycle freshness.  Start this node before controller activation so the
pinned Prox marker publisher sees a subscriber.
"""
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PolygonStamped, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from nav2_msgs.msg import CollisionMonitorState
from prox_mpc_msgs.msg import ObstacleArray, SolverDiagnostics
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped


class ProxDiagRecorder(Node):
    def __init__(self):
        super().__init__('prox_diag_recorder')
        self.declare_parameter('robot_namespace', '/task_generator_node/Ai2_Bot2')
        self.declare_parameter('record_dir', '/tmp/prox_diag')
        self.declare_parameter('costmap_hz', 5.0)
        self.declare_parameter('ego_hz', 5.0)
        ns = str(self.get_parameter('robot_namespace').value).rstrip('/')
        rec = str(self.get_parameter('record_dir').value)
        os.makedirs(rec, exist_ok=True)
        self.record_dir = rec
        self.ego_dir = os.path.join(rec, 'ego_frames')
        os.makedirs(self.ego_dir, exist_ok=True)
        self.jsonl = open(os.path.join(rec, 'PROX_DIAG.jsonl'), 'a', encoding='utf-8')
        self.costmap_hz = float(self.get_parameter('costmap_hz').value)
        self.ego_hz = float(self.get_parameter('ego_hz').value)
        self.last_costmap_write = 0.0
        self.last_ego_write = 0.0
        self.counts = {k: 0 for k in (
            'scan', 'costmap', 'local_plan', 'marker', 'tracker', 'diagnostics',
            'grounded_goal', 'cmd', 'odom', 'footprint', 'collision', 'ego',
            'planner_path', 'follow_path_status', 'model_status')}
        self.latest = {}
        self.last_tracker_stamp = 0
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        scan_topic = f'{ns}/laser_scan'
        costmap_topic = f'{ns}/local_costmap/costmap'
        local_plan_topic = f'{ns}/prox_mpc_local_plan'
        marker_topic = f'{ns}/prox_mpc_predicted_obstacles'
        diag_topic = f'{ns}/FollowPath/diagnostics'
        tracker_topic = f'{ns}/tracked_obstacles'
        goal_topic = f'{ns}/lisn_slow_loop/grounded_goal'
        cmd_topic = f'{ns}/cmd_vel'
        odom_topic = f'{ns}/odom'
        footprint_topic = f'{ns}/local_costmap/published_footprint'
        collision_topic = f'{ns}/collision_monitor_state'
        ego_topic = f'{ns}/head_camera/image'
        planner_topic = f'{ns}/received_global_plan'
        status_topic = f'{ns}/follow_path/_action/status'
        model_status_topic = f'{ns}/model/status'
        self.topic_names = dict(scan=scan_topic, costmap=costmap_topic,
            local_plan=local_plan_topic, marker=marker_topic, diagnostics=diag_topic,
            tracker=tracker_topic, grounded_goal=goal_topic, cmd=cmd_topic,
            odom=odom_topic, footprint=footprint_topic, collision=collision_topic,
            ego=ego_topic, planner_path=planner_topic,
            follow_path_status=status_topic, model_status=model_status_topic)

        self.create_subscription(LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data)
        cm_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, costmap_topic, self._on_costmap, cm_qos)
        self.create_subscription(Path, local_plan_topic, self._on_local_plan, 1)
        self.create_subscription(Path, planner_topic, self._on_planner_path, 1)
        self.create_subscription(Image, ego_topic, self._on_ego, qos_profile_sensor_data)
        # The marker publisher only emits while a subscriber exists; this
        # subscription is created before the controller is activated.
        self.create_subscription(MarkerArray, marker_topic, self._on_marker, 1)
        self.create_subscription(SolverDiagnostics, diag_topic, self._on_diagnostics, 10)
        self.create_subscription(ObstacleArray, tracker_topic, self._on_tracker, 10)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)
        self.create_subscription(Twist, cmd_topic, self._on_cmd, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(PolygonStamped, footprint_topic, self._on_footprint, 10)
        self.create_subscription(CollisionMonitorState, collision_topic, self._on_collision, 10)
        self.create_subscription(GoalStatusArray, status_topic, self._on_status, 10)
        self.create_subscription(String, model_status_topic, self._on_model_status, 10)
        self.create_timer(0.2, self._state_tick)
        self.create_timer(1.0, self._write_meta)
        self.get_logger().info(f'prox diag recorder ns={ns} dir={rec}')

    def _sim_now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _emit(self, event, **fields):
        self.jsonl.write(json.dumps({'event': event, 'sim_now': self._sim_now(),
                                     'wall_now': time.monotonic(), **fields},
                                    sort_keys=True) + '\n')
        self.jsonl.flush()

    def _on_scan(self, msg):
        self.counts['scan'] += 1
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        finite = [float(r) for r in msg.ranges if math.isfinite(float(r))]
        self.latest['scan'] = {'stamp': stamp, 'frame': msg.header.frame_id}
        self._emit('scan', source_stamp_ns=stamp, frame_id=msg.header.frame_id,
                   angle_min=float(msg.angle_min), angle_increment=float(msg.angle_increment),
                   range_min=float(msg.range_min), range_max=float(msg.range_max),
                   ranges=[float(r) for r in msg.ranges],
                   finite_count=len(finite),
                   finite_min=min(finite) if finite else None,
                   finite_max=max(finite) if finite else None)

    def _on_costmap(self, msg):
        self.counts['costmap'] += 1
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        self.latest['costmap'] = {'stamp': stamp, 'frame': msg.header.frame_id,
                                  'info': {'resolution': msg.info.resolution,
                                           'width': msg.info.width, 'height': msg.info.height,
                                           'origin': {'x': msg.info.origin.position.x,
                                                      'y': msg.info.origin.position.y}}}
        if self._sim_now() - self.last_costmap_write < 1.0 / self.costmap_hz:
            return
        self.last_costmap_write = self._sim_now()
        self._emit('costmap', source_stamp_ns=stamp, frame_id=msg.header.frame_id,
                   info=self.latest['costmap']['info'],
                   data=[int(v) for v in msg.data])

    def _poses(self, msg):
        return [{'x': float(p.pose.position.x), 'y': float(p.pose.position.y),
                 'yaw': _yaw(p.pose.orientation)} for p in msg.poses]

    def _on_local_plan(self, msg):
        self.counts['local_plan'] += 1
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        self.latest['local_plan'] = {'stamp': stamp, 'frame': msg.header.frame_id}
        self._emit('local_plan', source_stamp_ns=stamp, frame_id=msg.header.frame_id,
                   poses=self._poses(msg))

    def _on_planner_path(self, msg):
        self.counts['planner_path'] += 1
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        poses = self._poses(msg)
        transformed = None
        tf_error = None
        if msg.header.frame_id != 'Ai2_Bot2/odom':
            try:
                transformed = []
                for p in msg.poses:
                    tp = do_transform_pose_stamped(p, self.tf_buffer.lookup_transform(
                        'Ai2_Bot2/odom', msg.header.frame_id,
                        Time(nanoseconds=stamp)))
                    transformed.append({'x': float(tp.pose.position.x),
                                        'y': float(tp.pose.position.y),
                                        'yaw': _yaw(tp.pose.orientation)})
            except Exception as exc:
                tf_error = repr(exc)
        self.latest['planner_path'] = {'stamp': stamp, 'frame': msg.header.frame_id}
        self._emit('planner_path', source_stamp_ns=stamp, frame_id=msg.header.frame_id,
                   poses=poses, odom_poses=transformed, tf_error=tf_error)

    def _on_ego(self, msg):
        self.counts['ego'] += 1
        now = self._sim_now()
        if now - self.last_ego_write < 1.0 / self.ego_hz:
            return
        self.last_ego_write = now
        frame = _image_to_bgr(msg)
        if frame is None:
            return
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        path = os.path.join(self.ego_dir, f'{stamp}.jpg')
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            buf.tofile(path)
            self._emit('ego_frame', source_stamp_ns=stamp, frame_id=msg.header.frame_id,
                       path=path, width=int(msg.width), height=int(msg.height),
                       encoding=str(msg.encoding))

    def _on_marker(self, msg):
        self.counts['marker'] += 1
        markers = []
        for m in msg.markers:
            markers.append({
                'ns': str(m.ns), 'id': int(m.id), 'type': int(m.type),
                'action': int(m.action),
                'pose': {'x': float(m.pose.position.x), 'y': float(m.pose.position.y)},
                'points': [{'x': float(p.x), 'y': float(p.y), 'z': float(p.z)}
                           for p in m.points],
                'scale': {'x': float(m.scale.x), 'y': float(m.scale.y), 'z': float(m.scale.z)},
                'color': {'r': float(m.color.r), 'g': float(m.color.g),
                          'b': float(m.color.b), 'a': float(m.color.a)},
                'text': str(m.text),
            })
        stamp = int(msg.markers[0].header.stamp.sec) * 10**9 + int(msg.markers[0].header.stamp.nanosec) if msg.markers else 0
        self.latest['marker'] = {'stamp': stamp, 'frame': msg.markers[0].header.frame_id if msg.markers else ''}
        self._emit('predicted_markers', source_stamp_ns=stamp,
                   frame_id=msg.markers[0].header.frame_id if msg.markers else '',
                   markers=markers)

    def _on_tracker(self, msg):
        self.counts['tracker'] += 1
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        self.last_tracker_stamp = stamp
        age = self._sim_now() - stamp / 1e9
        obstacles = [{
            'id': int(o.id), 'x': float(o.position.x), 'y': float(o.position.y),
            'vx': float(o.velocity.x), 'vy': float(o.velocity.y),
            'radius': float(o.radius),
            'predicted_samples': len(o.predicted_positions),
        } for o in msg.obstacles]
        self.latest['tracker'] = {'stamp': stamp, 'frame': msg.header.frame_id,
                                  'count': len(obstacles), 'age_s': age,
                                  'obstacles': obstacles}
        self._emit('tracker', source_stamp_ns=stamp, frame_id=msg.header.frame_id,
                   age_s=age, obstacles=obstacles)

    def _on_diagnostics(self, msg):
        self.counts['diagnostics'] += 1
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        control_source_age = (stamp - self.last_tracker_stamp) / 1e9 if self.last_tracker_stamp else None
        d = {'status': int(msg.status), 'converged': bool(msg.converged),
             'num_active_obstacles': int(msg.num_active_obstacles),
             'max_obstacle_slack': float(msg.max_obstacle_slack),
             'solve_time_ms': float(msg.solve_time_ms),
             'control_period_ms': float(msg.control_period_ms),
             'deadline_missed': bool(msg.deadline_missed),
             'control_source_age_s': control_source_age}
        self.latest['diagnostics'] = {'stamp': stamp, **d}
        self._emit('diagnostics', source_stamp_ns=stamp, frame_id=msg.header.frame_id, **d)

    def _on_goal(self, msg):
        self.counts['grounded_goal'] += 1
        stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        goal_odom = None
        tf_error = None
        tf_status = 'TF_ERROR'
        try:
            tp = transform_pose_stamped_to_odom(msg, self.tf_buffer.lookup_transform)
            # ``do_transform_pose_stamped`` must preserve the source stamp and
            # return a PoseStamped whose frame is the lookup target.
            goal_odom = {'x': float(tp.pose.position.x), 'y': float(tp.pose.position.y),
                         'yaw': _yaw(tp.pose.orientation),
                         'frame': str(tp.header.frame_id),
                         'stamp_ns': int(tp.header.stamp.sec) * 10**9
                         + int(tp.header.stamp.nanosec),
                         'result_type': type(tp).__name__}
            tf_status = 'OK'
        except Exception as exc:
            tf_error = repr(exc)
        self.latest['goal'] = {'stamp': stamp, 'frame': msg.header.frame_id,
                               'x': float(msg.pose.position.x), 'y': float(msg.pose.position.y),
                               'odom': goal_odom, 'tf_error': tf_error,
                               'tf_status': tf_status}
        self._emit('grounded_goal', source_stamp_ns=stamp, frame_id=msg.header.frame_id,
                   x=float(msg.pose.position.x), y=float(msg.pose.position.y),
                   odom=goal_odom, tf_error=tf_error, tf_status=tf_status)

    def _on_cmd(self, msg):
        self.counts['cmd'] += 1
        self.latest['cmd'] = {'vx': float(msg.linear.x), 'wz': float(msg.angular.z)}

    def _on_odom(self, msg):
        self.counts['odom'] += 1
        p = msg.pose.pose
        self.latest['odom'] = {'x': float(p.position.x), 'y': float(p.position.y),
                               'yaw': _yaw(p.orientation),
                               'stamp': int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec),
                               'frame': msg.header.frame_id}

    def _on_footprint(self, msg):
        self.counts['footprint'] += 1
        self.latest['footprint'] = {'frame': msg.header.frame_id,
                                    'points': [{'x': float(p.x), 'y': float(p.y)}
                                               for p in msg.polygon.points]}

    def _on_collision(self, msg):
        self.counts['collision'] += 1
        self.latest['collision'] = {'action_type': int(msg.action_type),
                                    'polygon_name': str(msg.polygon_name)}
        self._emit('collision_monitor', action_type=int(msg.action_type),
                   polygon_name=str(msg.polygon_name))

    def _on_status(self, msg):
        self.counts['follow_path_status'] += 1
        self.latest['follow_path_status'] = {
            'statuses': [int(s.status) for s in msg.status_list],
        }

    def _on_model_status(self, msg):
        self.counts['model_status'] += 1
        self.latest['model_status'] = str(msg.data)

    def _state_tick(self):
        self._emit('state', robot_pose=self.latest.get('odom'),
                   cmd=self.latest.get('cmd'), footprint=self.latest.get('footprint'),
                   costmap_stamp=self.latest.get('costmap', {}).get('stamp'),
                   local_plan_stamp=self.latest.get('local_plan', {}).get('stamp'),
                   marker_stamp=self.latest.get('marker', {}).get('stamp'),
                   tracker=self.latest.get('tracker'),
                   diagnostics=self.latest.get('diagnostics'),
                   collision=self.latest.get('collision'),
                   goal=self.latest.get('goal'),
                   planner_path_stamp=self.latest.get('planner_path', {}).get('stamp'),
                   follow_path_status=self.latest.get('follow_path_status'),
                   model_status=self.latest.get('model_status'))

    def _write_meta(self):
        topics = {}
        for key, name in self.topic_names.items():
            topics[key] = {
                'topic': name,
                'publisher_count': self.count_publishers(name),
                'subscriber_count': self.count_subscribers(name),
                'messages': self.counts.get(key, 0),
            }
        vlm_trace = os.path.join(self.record_dir, 'VLM_TRACE.jsonl')
        vlm_lines = 0
        if os.path.exists(vlm_trace):
            with open(vlm_trace, encoding='utf-8') as f:
                vlm_lines = sum(1 for _ in f)
        meta = {
            'schema': 'arena.prox_diag_meta.v1',
            'record_dir': self.record_dir,
            'sim_now': self._sim_now(),
            'topics': topics,
            'counts': self.counts,
            'marker_subscriber_present': topics['marker']['subscriber_count'] > 0,
            'marker_messages': self.counts['marker'],
            'local_plan_messages': self.counts['local_plan'],
            'diagnostics_messages': self.counts['diagnostics'],
            'costmap_frames': self.counts['costmap'],
            'scan_messages': self.counts['scan'],
            'tracker_messages': self.counts['tracker'],
            'ego_frames': self.counts['ego'],
            'vlm_trace_lines': vlm_lines,
            'latest_diagnostics': self.latest.get('diagnostics'),
            'latest_tracker': self.latest.get('tracker'),
            'latest_goal': self.latest.get('goal'),
            'latest_collision': self.latest.get('collision'),
        }
        with open(os.path.join(self.record_dir, 'PROX_DIAG_META.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(meta, f, indent=2, sort_keys=True)


def transform_pose_stamped_to_odom(msg, lookup_transform):
    """Transform a ``geometry_msgs/PoseStamped`` at its own source stamp.

    The lookup is pinned to ``msg.header.stamp``: never ``Time()``/latest, and
    never a fresh re-bind of an older source stamp.  ``lookup_transform`` is
    the buffer method ``(target_frame, source_frame, time)`` so this helper is
    testable without a live ROS graph.
    """
    stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
    transform = lookup_transform(
        'Ai2_Bot2/odom', msg.header.frame_id, Time(nanoseconds=stamp))
    transformed = do_transform_pose_stamped(msg, transform)
    if not isinstance(transformed, PoseStamped):
        raise TypeError(
            f'do_transform_pose_stamped returned {type(transformed).__name__}, '
            'expected PoseStamped')
    return transformed


def _yaw(q):
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _image_to_bgr(msg):
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        return None
    enc = str(msg.encoding or 'rgb8').lower()
    step = int(msg.step or 0) or w * (4 if enc in ('rgba8', 'bgra8') else 3)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if raw.size < step * h:
        return None
    channels = 4 if enc in ('rgba8', 'bgra8') else 3
    row = w * channels
    if step < row:
        return None
    arr = raw.reshape(h, step)[:, :row].reshape(h, w, channels)
    if enc == 'rgb8':
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if enc == 'rgba8':
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if enc == 'bgra8':
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if enc == 'bgr8':
        return arr
    return None


def main():
    rclpy.init()
    node = ProxDiagRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._write_meta()
            node.jsonl.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
