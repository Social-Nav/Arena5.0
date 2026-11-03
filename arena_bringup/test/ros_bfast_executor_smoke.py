#!/usr/bin/env python3
"""No-GPU real ArenaModelExecutor + real bfast_guard ROS smoke.

The fake adapter is deterministic and only supplies a fixed forward path; the
guard, the executor, the FollowPath action server, and the veto topic are real
ROS entities.  This proves the actual executor (not a controlled consumer)
turns a clear guard into a common FollowPath request and turns direct/invalid/
required-stale vetoes into STOP.

Run in the eval container:

  ROS_DOMAIN_ID=98 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    /opt/venv/bin/python3 arena_bringup/test/ros_bfast_executor_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import FollowPath
from rclpy.action import ActionServer
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int16, String
from nav_msgs.msg import Odometry

from arena_bringup.bfast_guard_node import BfastGuardNode
from arena_vln_models.core import ModelPrediction, ModelObservation
from arena_vln_models.executor import ArenaModelExecutor
from arena_vln_models.model_adapter import (
    ModelCapability,
    ModelRawOutput,
    ModelSpec,
    prediction_from_raw,
    register_adapter,
    register_model_spec,
)


@register_adapter('fake_bfast')
class FakeBfastAdapter:
    name = 'fake_bfast'
    capability = ModelCapability(
        name='fake_bfast', raw_output_kind='path', required_inputs=('rgb_image',))

    def __init__(self, *, params=None, logger=None, **kwargs):
        self.params = params or {}
        self.logger = logger

    def reset(self, *, session_id='', episode=-1):
        return None

    def infer(self, observation: ModelObservation) -> ModelRawOutput:
        return ModelRawOutput(
            adapter='fake_bfast', kind='path',
            local_waypoints=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            source='native',
        )

    def to_prediction(self, raw: ModelRawOutput,
                      observation: ModelObservation) -> ModelPrediction:
        return prediction_from_raw(raw, observation, adapter_name='fake_bfast')


register_model_spec(ModelSpec(name='fake_bfast', adapter='fake_bfast',
                              profile='fake_bfast_smoke.yaml'))


class Harness(Node):
    def __init__(self, *, rgb_topic: str, odom_topic: str, veto_topic: str,
                 decision_topic: str, eval_ready_topic: str, task_reset_topic: str,
                 instruction_topic: str, status_topic: str) -> None:
        super().__init__('bfast_executor_harness')
        self.status = None
        self.status_reason = ''
        self.guard_decision = None
        self.goals = []
        self.rgb_pub = self.create_publisher(Image, rgb_topic, qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.veto_pub = self.create_publisher(Bool, veto_topic, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.eval_pub = self.create_publisher(String, eval_ready_topic, latched)
        self.reset_pub = self.create_publisher(Int16, task_reset_topic, latched)
        self.instruction_pub = self.create_publisher(String, instruction_topic, 10)
        self.create_subscription(String, status_topic, self._on_status, 10)
        self.create_subscription(String, decision_topic, self._on_decision, 10)
        self.action_server = ActionServer(
            self, FollowPath, 'follow_path', execute_callback=self._execute,
            goal_callback=self._goal,
        )

    def _on_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        self.status = payload.get('status')
        self.status_reason = payload.get('reason', '')

    def _on_decision(self, msg: String) -> None:
        try:
            self.guard_decision = json.loads(msg.data)
        except ValueError:
            self.guard_decision = {'raw': msg.data}

    def _goal(self, goal_request):
        return rclpy.action.GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        self.goals.append(goal_handle.request.path)
        goal_handle.succeed()
        return FollowPath.Result()

    def publish_episode(self, session_id='smoke-session', episode=0):
        self.eval_pub.publish(String(data=json.dumps({
            'stage': 'episode', 'session_id': session_id,
            'episode': episode, 'ready': True,
        })))

    def publish_inputs(self, stamp_ns, session_id='smoke-session', episode=0):
        self.publish_episode(session_id, episode)
        self.instruction_pub.publish(String(data='walk forward'))
        self.publish_rgb_odom(stamp_ns)

    def publish_rgb_odom(self, stamp_ns: int, frame=None):
        frame = _textured_frame() if frame is None else frame
        image = Image()
        image.height = int(frame.shape[0])
        image.width = int(frame.shape[1])
        image.encoding = 'rgb8'
        image.is_bigendian = 0
        image.step = int(frame.shape[1]) * 3
        image.data = frame.astype(np.uint8).tobytes()
        image.header.stamp.sec = stamp_ns // 1_000_000_000
        image.header.stamp.nanosec = stamp_ns % 1_000_000_000
        image.header.frame_id = 'Ai2_Bot2/base_footprint/head_camera'
        self.rgb_pub.publish(image)

        odom = Odometry()
        odom.header.stamp = image.header.stamp
        odom.header.frame_id = 'Ai2_Bot2/odom'
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(odom)

    def publish_invalid(self, stamp_ns: int):
        image = Image()
        image.height = 2
        image.width = 2
        image.encoding = '32FC1'
        image.is_bigendian = 0
        image.step = 8
        image.data = b'\x00' * 16
        image.header.stamp.sec = stamp_ns // 1_000_000_000
        image.header.stamp.nanosec = stamp_ns % 1_000_000_000
        image.header.frame_id = 'Ai2_Bot2/base_footprint/head_camera'
        self.rgb_pub.publish(image)


def _textured_frame(height=48, width=64):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(width)[None, :] % 60
    frame[:, :, 1] = np.arange(height)[:, None] % 60
    frame[:, :, 2] = 40
    return frame


def _spin(executor, seconds):
    deadline = time.monotonic() + float(seconds)
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)


def _trace_rows(path) -> list:
    if not path or not os.path.exists(str(path)):
        return []
    rows = []
    with open(str(path), encoding='utf-8') as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def main() -> int:
    topics = {
        'rgb': '/task_generator_node/Ai2_Bot2/head_camera/image',
        'odom': '/task_generator_node/Ai2_Bot2/odom',
        'veto': '/task_generator_node/Ai2_Bot2/bfast/veto',
        'decision': '/task_generator_node/Ai2_Bot2/bfast/decision',
        'task_ns': '/task_generator_node',
        'eval_ready': '/task_generator_node/eval_ready',
        'task_reset': '/task_generator_node/task_reset',
        'instruction': '/task_generator_node/vln_instruction',
        'status': '/task_generator_node/Ai2_Bot2/model/status',
        'trace': '/tmp/bfast_executor_smoke',
    }
    sys.argv = [
        sys.argv[0], '--ros-args',
        '-p', 'model_adapter:=fake_bfast',
        '-p', f'rgb_topic:={topics["rgb"]}',
        '-p', f'odom_topic:={topics["odom"]}',
        '-p', f'task_namespace:={topics["task_ns"]}',
        '-p', f'eval_ready_topic:={topics["eval_ready"]}',
        '-p', f'task_reset_topic:={topics["task_reset"]}',
        '-p', f'instruction_topic:={topics["instruction"]}',
        '-p', f'model_status_topic:={topics["status"]}',
        '-p', f'fast_guard_veto_topic:={topics["veto"]}',
        '-p', 'fast_guard_veto_timeout_sec:=1.0',
        '-p', 'fast_guard_required:=true',
        '-p', f'trace_dir:={topics["trace"]}',
        '-p', 'use_tf_exact_stamp:=false',
        '-p', 'inference_hz:=10.0',
        '-p', f'veto_topic:={topics["veto"]}',
        '-p', f'decision_topic:={topics["decision"]}',
        '-p', 'guard_enabled:=true',
        '-p', 'risk_threshold:=0.5',
        '-p', 'max_frame_gap_sec:=1.0',
    ]
    rclpy.init(args=sys.argv)
    guard = BfastGuardNode()
    executor_node = ArenaModelExecutor()
    harness = Harness(
        rgb_topic=topics['rgb'], odom_topic=topics['odom'],
        veto_topic=topics['veto'], decision_topic=topics['decision'],
        eval_ready_topic=topics['eval_ready'],
        task_reset_topic=topics['task_reset'],
        instruction_topic=topics['instruction'], status_topic=topics['status'])
    executor = SingleThreadedExecutor()
    for node in (guard, executor_node, harness):
        executor.add_node(node)
    receipt = {'schema': 'arena.bfast.executor_ros_smoke.v1', 'scenarios': []}
    receipt['executor_params'] = {
        key: executor_node.get_parameter(key).value
        for key in ('model_adapter', 'fast_guard_veto_topic',
                    'fast_guard_required', 'fast_guard_veto_timeout_sec',
                    'use_tf_exact_stamp')
    }
    ok = True
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not executor_node.action.server_is_ready():
            _spin(executor, 0.1)
        if not executor_node.action.server_is_ready():
            receipt['error'] = 'follow_path_server_unavailable'
            ok = False
            raise RuntimeError('follow_path_server_unavailable')
        harness.instruction_pub.publish(String(data='walk forward'))
        stamp = 1_000_000_000
        for _ in range(20):
            harness.publish_inputs(stamp)
            stamp += 100_000_000
            _spin(executor, 0.1)
        rows = _trace_rows(executor_node.trace_path)
        clear_goals = [row for row in rows if row.get('event') == 'follow_path_goal']
        clear = {'name': 'clear_follow_path', 'status': harness.status,
                 'reason': harness.status_reason,
                 'follow_path_goals': len(harness.goals),
                 'trace_follow_path_goals': len(clear_goals)}
        receipt['scenarios'].append(clear)
        ok &= len(clear_goals) >= 1

        # Stop the guard so its clear heartbeat cannot overwrite the direct
        # veto under test.  The guard node remains alive, just not spun.
        executor.remove_node(guard)
        harness.veto_pub.publish(Bool(data=True))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not executor_node._fast_guard_veto:
            _spin(executor, 0.1)
        before = len(harness.goals)
        for _ in range(12):
            harness.publish_inputs(stamp)
            stamp += 100_000_000
            _spin(executor, 0.1)
        rows = _trace_rows(executor_node.trace_path)
        direct_veto = [row for row in rows
                       if row.get('event') == 'bfast_veto'
                       and row.get('reason') == 'bfast_veto']
        direct = {'name': 'direct_veto', 'status': harness.status,
                  'reason': harness.status_reason,
                  'follow_path_goals': len(harness.goals),
                  'trace_bfast_veto': len(direct_veto)}
        receipt['scenarios'].append(direct)
        ok &= len(direct_veto) >= 1
        ok &= direct['follow_path_goals'] == before

        # Re-add the guard and freeze its INVALID veto before the new session
        # releases, so the executor's first decision sees a fresh veto.
        executor.add_node(guard)
        harness.publish_invalid(stamp)
        stamp += 100_000_000
        _spin(executor, 0.2)
        executor.remove_node(guard)
        before = len(harness.goals)
        # New session so the terminal direct-veto state does not mask the next
        # scenario.
        harness.publish_inputs(stamp, 'smoke-session-1', 1)
        for _ in range(5):
            harness.publish_inputs(stamp, 'smoke-session-1', 1)
            stamp += 100_000_000
            _spin(executor, 0.1)
        rows = _trace_rows(executor_node.trace_path)
        invalid_veto = [row for row in rows
                        if row.get('event') == 'bfast_veto'
                        and row.get('reason') == 'bfast_veto'
                        and row.get('session_id') == 'smoke-session-1']
        invalid = {'name': 'invalid_input', 'status': harness.status,
                   'reason': harness.status_reason,
                   'guard_decision': (harness.guard_decision or {}).get('status'),
                   'follow_path_goals': len(harness.goals),
                   'trace_bfast_veto': len(invalid_veto),
                   'executor_veto': bool(executor_node._fast_guard_veto),
                   'executor_veto_age_sec': (
                       time.monotonic() - executor_node._fast_guard_veto_at)}
        receipt['scenarios'].append(invalid)
        ok &= len(invalid_veto) >= 1
        ok &= invalid['guard_decision'] in ('INVALID', 'INVALID_HOLD')
        ok &= invalid['follow_path_goals'] == before

        # Let the frozen veto age past the timeout while no new guard message
        # can refresh it; the guard is already removed.
        _spin(executor, 1.2)
        before_stale = len(harness.goals)
        # New session; the executor remains required and must become typed
        # stale after the timeout.
        harness.publish_inputs(stamp, 'smoke-session-2', 2)
        for _ in range(16):
            harness.publish_inputs(stamp, 'smoke-session-2', 2)
            stamp += 100_000_000
            _spin(executor, 0.1)
        rows = _trace_rows(executor_node.trace_path)
        stale_veto = [row for row in rows
                      if row.get('event') == 'bfast_veto'
                      and row.get('reason') == 'bfast_guard_stale']
        stale = {'name': 'required_stale', 'status': harness.status,
                 'reason': harness.status_reason,
                 'follow_path_goals': len(harness.goals),
                 'trace_bfast_veto': len(stale_veto)}
        receipt['scenarios'].append(stale)
        ok &= len(stale_veto) >= 1
        ok &= stale['follow_path_goals'] == before_stale
    finally:
        executor.shutdown()
        for node in (guard, executor_node, harness):
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()
    receipt['ok'] = bool(ok)
    receipt['ros_domain_id'] = os.environ.get('ROS_DOMAIN_ID')
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
