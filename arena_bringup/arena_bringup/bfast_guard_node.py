#!/usr/bin/env python3
"""Strict RGB-only B fast guard node.

Subscribes only to a real RGB ``sensor_msgs/Image`` topic, computes the
``paper_pipeline.compute_flow_risk_proxy`` score, and publishes a Bool veto
that the common Arena executor consumes before sending FollowPath.  It never
subscribes to scan, depth, point cloud, GT map, HuNav People, or eval goals.

Runtime status: IMPLEMENTED_UNVERIFIED.  The pure scoring contract is
CPU-tested; this node has not been executed in an episode.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from arena_bringup.paper_pipeline import (
    PipelineRejected,
    compute_flow_risk_proxy,
    validate_bfast_topics,
)


def decode_rgb(msg: Image) -> np.ndarray:
    """Decode a ROS Image into HxWx3 uint8, respecting step and encoding."""
    height, width = int(msg.height), int(msg.width)
    if height <= 0 or width <= 0:
        raise PipelineRejected('rgb_geometry')
    encoding = str(msg.encoding or 'rgb8').lower()
    step = int(msg.step or 0)
    if encoding in ('rgb8', 'bgr8'):
        channels = 3
    elif encoding in ('rgba8', 'bgra8'):
        channels = 4
    elif encoding in ('mono8', '8uc1'):
        channels = 1
    else:
        raise PipelineRejected(f'rgb_encoding:{encoding}')
    row_bytes = width * channels
    stride = step or row_bytes
    if stride < row_bytes:
        raise PipelineRejected('rgb_step')
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8, count=stride * height)
    rows = raw.reshape(height, stride)[:, :row_bytes]
    if channels == 1:
        return np.repeat(rows.reshape(height, width, 1), 3, axis=2)
    image = rows.reshape(height, width, channels)[:, :, :3]
    if encoding in ('bgr8', 'bgra8'):
        image = image[:, :, ::-1]
    return image.copy()


class BfastGuardNode(Node):
    def __init__(self) -> None:
        super().__init__('bfast_guard')
        self.declare_parameter('rgb_topic', '')
        self.declare_parameter('veto_topic', '')
        self.declare_parameter('decision_topic', '')
        self.declare_parameter('guard_enabled', True)
        self.declare_parameter('risk_threshold', 0.5)
        self.declare_parameter('max_frame_gap_sec', 1.0)
        self.declare_parameter('invalid_hold_sec', 0.5)
        rgb_topic = str(self.get_parameter('rgb_topic').value or '').strip()
        if not rgb_topic:
            raise PipelineRejected('rgb_topic_required')
        check = validate_bfast_topics([rgb_topic])
        if not check['valid']:
            raise PipelineRejected(f'forbidden_rgb_topic:{check["forbidden_topics"]}')
        self.rgb_topic = rgb_topic
        self.veto_topic = str(self.get_parameter('veto_topic').value or '').strip()
        self.decision_topic = str(self.get_parameter('decision_topic').value or '').strip()
        if not self.veto_topic or not self.decision_topic:
            raise PipelineRejected('veto_topic_and_decision_topic_required')
        self.enabled = bool(self.get_parameter('guard_enabled').value)
        self.risk_threshold = float(self.get_parameter('risk_threshold').value)
        self.max_frame_gap_sec = float(self.get_parameter('max_frame_gap_sec').value)
        self.invalid_hold_sec = float(self.get_parameter('invalid_hold_sec').value)
        self._invalid_until = 0.0
        self.prev_frame = None
        self.prev_stamp_ns = None
        self.create_subscription(Image, self.rgb_topic, self._on_image,
                                 qos_profile_sensor_data)
        self.veto_pub = self.create_publisher(Bool, self.veto_topic, 10)
        self.decision_pub = self.create_publisher(String, self.decision_topic, 10)
        self.get_logger().info(
            f'bfast guard rgb={self.rgb_topic} veto={self.veto_topic} '
            f'decision={self.decision_topic} enabled={self.enabled}')

    def _on_image(self, msg: Image) -> None:
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        try:
            frame = decode_rgb(msg)
        except PipelineRejected as exc:
            self._invalid_until = time.monotonic() + self.invalid_hold_sec
            self._publish(True, {'status': 'INVALID', 'reason': str(exc), 'stop': True})
            return
        if self.prev_frame is None or self.prev_stamp_ns is None:
            self.prev_frame = frame
            self.prev_stamp_ns = stamp_ns
            self._publish(False, {'status': 'WARMUP', 'reason': 'first_frame',
                                  'texture_state': 'UNKNOWN', 'stop': False})
            return
        dt_sec = (stamp_ns - self.prev_stamp_ns) / 1e9
        if dt_sec <= 0.0 or dt_sec > self.max_frame_gap_sec:
            self.prev_frame = frame
            self.prev_stamp_ns = stamp_ns
            self._publish(False, {'status': 'UNKNOWN', 'reason': 'frame_gap',
                                  'texture_state': 'UNKNOWN', 'stop': False})
            return
        try:
            risk = compute_flow_risk_proxy(
                self.prev_frame, frame, dt_sec=dt_sec,
                risk_threshold=self.risk_threshold)
        except PipelineRejected as exc:
            self.prev_frame = frame
            self.prev_stamp_ns = stamp_ns
            self._publish(True, {'status': 'INVALID', 'reason': str(exc), 'stop': True})
            return
        self.prev_frame = frame
        self.prev_stamp_ns = stamp_ns
        if time.monotonic() < self._invalid_until:
            self._publish(bool(self.enabled), {
                'schema': 'arena.bfast.runtime_decision.v1',
                'enabled': self.enabled,
                'veto': bool(self.enabled),
                'source_stamp_ns': stamp_ns,
                'risk': risk,
                'status': 'INVALID_HOLD',
                'stop': bool(self.enabled),
                'reason': 'invalid_frame_hold',
            })
            return
        veto = bool(self.enabled and risk.get('stop'))
        payload = {
            'schema': 'arena.bfast.runtime_decision.v1',
            'enabled': self.enabled,
            'veto': veto,
            'source_stamp_ns': stamp_ns,
            'risk': risk,
            'status': ('VETO' if veto else
                       'UNKNOWN' if risk.get('texture_state') == 'UNKNOWN'
                       else 'PASS'),
        }
        self._publish(veto, payload)

    def _publish(self, veto: bool, payload: dict[str, Any]) -> None:
        self.veto_pub.publish(Bool(data=bool(veto)))
        self.decision_pub.publish(String(data=json.dumps(payload, sort_keys=True)))


def main(argv=None) -> int:
    rclpy.init(args=argv)
    node = BfastGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
