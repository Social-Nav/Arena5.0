#!/usr/bin/env python3
"""Real-ROS B fast-guard integration smoke (no GPU/Isaac/model).

Run in the eval container with an isolated domain:

  ROS_DOMAIN_ID=97 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    /opt/venv/bin/python3 arena_bringup/test/ros_bfast_guard_smoke.py

It starts the real ``bfast_guard`` node and a controlled executor-compatible
veto consumer on real ROS topics/QoS, then checks:

- valid frame pair -> veto=false -> follow_path allowed
- invalid 32FC1 frame -> veto=true -> follow_path becomes STOP
- direct veto Bool -> follow_path becomes STOP
- stale guard with required=true -> STOP reason bfast_guard_stale
- stale guard with required=false -> no veto

Exit rc=0 only if every scenario matches; prints a JSON receipt.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from arena_bringup.bfast_guard_node import BfastGuardNode


class VetoConsumer(Node):
    """Controlled executor-compatible consumer of the guard veto topic."""

    def __init__(self, *, rgb_topic: str, veto_topic: str, decision_topic: str,
                 timeout_sec: float, required: bool) -> None:
        super().__init__('bfast_veto_consumer')
        self.timeout_sec = float(timeout_sec)
        self.required = bool(required)
        self.veto = False
        self.last_veto_at = float('-inf')
        self.decision = None
        self.create_subscription(Bool, veto_topic, self._on_veto, 10)
        self.create_subscription(String, decision_topic, self._on_decision, 10)
        self.rgb_pub = self.create_publisher(Image, rgb_topic, qos_profile_sensor_data)
        self.veto_pub = self.create_publisher(Bool, veto_topic, 10)

    def _on_veto(self, msg: Bool) -> None:
        self.veto = bool(msg.data)
        self.last_veto_at = time.monotonic()

    def _on_decision(self, msg: String) -> None:
        try:
            self.decision = json.loads(msg.data)
        except ValueError:
            self.decision = {'raw': msg.data}

    def veto_reason(self) -> str:
        age = time.monotonic() - self.last_veto_at
        if self.veto and age <= self.timeout_sec:
            return 'bfast_veto'
        if self.required and age > self.timeout_sec:
            return 'bfast_guard_stale'
        return ''

    def executor_action(self, action: str = 'follow_path') -> str:
        reason = self.veto_reason()
        return 'stop' if reason else action

    def publish_rgb(self, frame: np.ndarray) -> None:
        msg = Image()
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = int(frame.shape[1]) * 3
        msg.data = frame.astype(np.uint8).tobytes()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_pub.publish(msg)

    def publish_invalid_float_image(self) -> None:
        msg = Image()
        msg.height = 2
        msg.width = 2
        msg.encoding = '32FC1'
        msg.is_bigendian = 0
        msg.step = 8
        msg.data = (b'\x00' * 16)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_pub.publish(msg)


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


def main() -> int:
    rgb_topic = '/task_generator_node/Ai2_Bot2/head_camera/image'
    veto_topic = '/task_generator_node/Ai2_Bot2/bfast/veto'
    decision_topic = '/task_generator_node/Ai2_Bot2/bfast/decision'
    sys.argv = [
        sys.argv[0], '--ros-args',
        '-p', f'rgb_topic:={rgb_topic}',
        '-p', f'veto_topic:={veto_topic}',
        '-p', f'decision_topic:={decision_topic}',
        '-p', 'guard_enabled:=true',
        '-p', 'risk_threshold:=0.5',
        '-p', 'max_frame_gap_sec:=1.0',
    ]
    rclpy.init(args=sys.argv)
    guard = BfastGuardNode()
    consumer = VetoConsumer(rgb_topic=rgb_topic, veto_topic=veto_topic,
                            decision_topic=decision_topic,
                            timeout_sec=1.0, required=True)
    executor = SingleThreadedExecutor()
    executor.add_node(guard)
    executor.add_node(consumer)
    frame = _textured_frame()
    receipt = {'schema': 'arena.bfast.ros_smoke.v1', 'scenarios': []}
    ok = True

    try:
        _spin(executor, 0.5)
        consumer.publish_rgb(frame)
        _spin(executor, 0.5)
        warmup = {'name': 'warmup', 'veto': consumer.veto,
                  'decision_status': (consumer.decision or {}).get('status'),
                  'executor_action': consumer.executor_action()}
        receipt['scenarios'].append(warmup)
        ok &= warmup['veto'] is False and warmup['executor_action'] == 'follow_path'

        consumer.publish_rgb(frame)
        _spin(executor, 0.5)
        clear = {'name': 'clear', 'veto': consumer.veto,
                 'decision_status': (consumer.decision or {}).get('status'),
                 'executor_action': consumer.executor_action()}
        receipt['scenarios'].append(clear)
        ok &= clear['veto'] is False and clear['executor_action'] == 'follow_path'

        consumer.publish_invalid_float_image()
        _spin(executor, 0.5)
        invalid = {'name': 'invalid_32fc1', 'veto': consumer.veto,
                   'decision_status': (consumer.decision or {}).get('status'),
                   'executor_action': consumer.executor_action()}
        receipt['scenarios'].append(invalid)
        ok &= invalid['veto'] is True and invalid['executor_action'] == 'stop'

        consumer.veto_pub.publish(Bool(data=True))
        _spin(executor, 0.3)
        direct = {'name': 'direct_veto', 'veto': consumer.veto,
                  'reason': consumer.veto_reason(),
                  'executor_action': consumer.executor_action()}
        receipt['scenarios'].append(direct)
        ok &= direct['veto'] is True and direct['executor_action'] == 'stop'

        _spin(executor, 1.2)
        stale_required = {'name': 'stale_required', 'veto': consumer.veto,
                          'reason': consumer.veto_reason(),
                          'executor_action': consumer.executor_action()}
        receipt['scenarios'].append(stale_required)
        ok &= stale_required['reason'] == 'bfast_guard_stale'
        ok &= stale_required['executor_action'] == 'stop'

        consumer.required = False
        stale_optional = {'name': 'stale_optional', 'veto': consumer.veto,
                          'reason': consumer.veto_reason(),
                          'executor_action': consumer.executor_action()}
        receipt['scenarios'].append(stale_optional)
        ok &= stale_optional['reason'] == ''
        ok &= stale_optional['executor_action'] == 'follow_path'
    finally:
        executor.shutdown()
        guard.destroy_node()
        consumer.destroy_node()
        rclpy.shutdown()

    receipt['ok'] = bool(ok)
    receipt['ros_domain_id'] = os.environ.get('ROS_DOMAIN_ID')
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
