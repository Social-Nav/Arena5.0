from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Int16, String


def _stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def _twist_to_dict(msg: Twist) -> dict[str, float]:
    return {
        'linear_x': float(msg.linear.x),
        'linear_y': float(msg.linear.y),
        'linear_z': float(msg.linear.z),
        'angular_x': float(msg.angular.x),
        'angular_y': float(msg.angular.y),
        'angular_z': float(msg.angular.z),
    }


def _twist_from_dict(data: dict[str, float]) -> Twist:
    msg = Twist()
    msg.linear.x = float(data.get('linear_x', 0.0) or 0.0)
    msg.linear.y = float(data.get('linear_y', 0.0) or 0.0)
    msg.linear.z = float(data.get('linear_z', 0.0) or 0.0)
    msg.angular.x = float(data.get('angular_x', 0.0) or 0.0)
    msg.angular.y = float(data.get('angular_y', 0.0) or 0.0)
    msg.angular.z = float(data.get('angular_z', 0.0) or 0.0)
    return msg


@dataclass
class PendingCommand:
    seq: int
    command: dict[str, float]
    receive_wall_time: float
    receive_sim_time: float
    eligible_sim_time: float
    delay_sec: float


class InternNavTimingManager(Node):
    """Delay direct InternNav commands in simulation time and record RTF.

    The upstream InternNav realworld client still runs with wall-clock model
    latency.  This node prevents low-RTF simulation from applying that command
    too early in simulated time by queueing raw cmd_vel messages until
    ``raw_obs_sim_time + latency_sec``.
    """

    def __init__(self, *, parameter_overrides: list[Parameter] | None = None) -> None:
        super().__init__('internnav_timing_manager', parameter_overrides=parameter_overrides or [])

        self.declare_parameter('timing_mode', 'wall')
        self.declare_parameter('latency_policy', 'fixed')
        self.declare_parameter('model_latency_sec', 0.3)
        self.declare_parameter('planning_period_sec', 0.3)
        self.declare_parameter('input_cmd_vel_topic', 'internnav/raw_cmd_vel')
        self.declare_parameter('output_cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('status_topic', 'internnav/status')
        self.declare_parameter('task_reset_topic', '/task_generator_node/task_reset')
        self.declare_parameter('eval_ready_topic', '/task_generator_node/eval_ready')
        self.declare_parameter('record_data_dir', '')
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('action_hold_policy', 'hold_last')
        self.declare_parameter('max_queue_sec', 10.0)

        self.timing_mode = str(self.get_parameter('timing_mode').value or 'wall').strip().lower()
        self.latency_policy = str(self.get_parameter('latency_policy').value or 'fixed').strip().lower()
        self.model_latency_sec = max(float(self.get_parameter('model_latency_sec').value or 0.0), 0.0)
        self.planning_period_sec = max(float(self.get_parameter('planning_period_sec').value or 0.0), 0.0)
        self.action_hold_policy = str(self.get_parameter('action_hold_policy').value or 'hold_last').strip().lower()
        self.max_queue_sec = max(float(self.get_parameter('max_queue_sec').value or 10.0), 0.1)
        publish_rate_hz = max(float(self.get_parameter('publish_rate_hz').value or 20.0), 1.0)

        record_data_dir = str(self.get_parameter('record_data_dir').value or '').strip()
        self.record_dir = Path(record_data_dir) if record_data_dir else None
        self.trace_path = self.record_dir / 'internnav_timing_trace.jsonl' if self.record_dir else None
        self.rtf_path = self.record_dir / 'rtf.csv' if self.record_dir else None
        self.summary_path = self.record_dir / 'internnav_timing_summary.json' if self.record_dir else None

        self.current_sim_time: float | None = None
        self.last_clock_wall_time: float | None = None
        self.last_clock_sim_time: float | None = None
        self.rtf_samples: list[float] = []
        self.pending: list[PendingCommand] = []
        self.last_released_command: dict[str, float] | None = None
        self.latest_measured_latency_sec: float | None = None
        self.episode_started = False
        self.reset_episode: int | None = None
        self.seq = 0
        self.released_count = 0
        self.raw_count = 0

        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            if self.rtf_path and not self.rtf_path.exists():
                with self.rtf_path.open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=['wall_time', 'sim_time', 'dt_wall_sec', 'dt_sim_sec', 'rtf'],
                    )
                    writer.writeheader()

        input_topic = str(self.get_parameter('input_cmd_vel_topic').value or 'internnav/raw_cmd_vel')
        output_topic = str(self.get_parameter('output_cmd_vel_topic').value or 'cmd_vel')
        status_topic = str(self.get_parameter('status_topic').value or 'internnav/status')
        task_reset_topic = str(self.get_parameter('task_reset_topic').value or '/task_generator_node/task_reset')
        eval_ready_topic = str(self.get_parameter('eval_ready_topic').value or '/task_generator_node/eval_ready')

        self.cmd_pub = self.create_publisher(Twist, output_topic, 10)
        self.create_subscription(Twist, input_topic, self._on_raw_cmd, 50)

        clock_qos = QoSProfile(depth=50)
        clock_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Clock, '/clock', self._on_clock, clock_qos)

        status_qos = QoSProfile(depth=10)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.VOLATILE
        self.create_subscription(String, status_topic, self._on_status, status_qos)
        self.create_subscription(Int16, task_reset_topic, self._on_task_reset, 10)

        ready_qos = QoSProfile(depth=1)
        ready_qos.reliability = ReliabilityPolicy.RELIABLE
        ready_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(String, eval_ready_topic, self._on_eval_ready, ready_qos)

        self.timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)
        self.summary_timer = self.create_timer(2.0, self._write_summary)

        self._record_event(
            'timing_manager_started',
            timing_mode=self.timing_mode,
            latency_policy=self.latency_policy,
            model_latency_sec=self.model_latency_sec,
            planning_period_sec=self.planning_period_sec,
            input_cmd_vel_topic=input_topic,
            output_cmd_vel_topic=output_topic,
            status_topic=status_topic,
        )

    def _wall_time(self) -> float:
        return time.time()

    def _monotonic(self) -> float:
        return time.monotonic()

    def _latency_sec(self) -> float:
        if self.latency_policy == 'measured' and self.latest_measured_latency_sec is not None:
            return max(float(self.latest_measured_latency_sec), 0.0)
        return self.model_latency_sec

    def _record_event(self, event: str, **fields: Any) -> None:
        if not self.trace_path:
            return
        record = {
            'event': event,
            'wall_time': self._wall_time(),
            'sim_time': self.current_sim_time,
            **fields,
        }
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        except Exception as exc:
            self.get_logger().warn(f'failed to write timing trace: {exc!r}')

    def _on_clock(self, msg: Clock) -> None:
        sim_time = _stamp_to_sec(msg.clock)
        wall_now = self._monotonic()
        wall_epoch = self._wall_time()
        if self.last_clock_wall_time is not None and self.last_clock_sim_time is not None:
            dt_wall = wall_now - self.last_clock_wall_time
            dt_sim = sim_time - self.last_clock_sim_time
            if dt_wall > 1e-6 and dt_sim >= 0.0:
                rtf = dt_sim / dt_wall
                if math.isfinite(rtf):
                    self.rtf_samples.append(rtf)
                    if self.rtf_path:
                        with self.rtf_path.open('a', newline='', encoding='utf-8') as handle:
                            writer = csv.DictWriter(
                                handle,
                                fieldnames=['wall_time', 'sim_time', 'dt_wall_sec', 'dt_sim_sec', 'rtf'],
                            )
                            writer.writerow(
                                {
                                    'wall_time': wall_epoch,
                                    'sim_time': sim_time,
                                    'dt_wall_sec': dt_wall,
                                    'dt_sim_sec': dt_sim,
                                    'rtf': rtf,
                                }
                            )
        self.current_sim_time = sim_time
        self.last_clock_wall_time = wall_now
        self.last_clock_sim_time = sim_time

    def _on_raw_cmd(self, msg: Twist) -> None:
        self.raw_count += 1
        self.seq += 1
        sim_now = self.current_sim_time
        if sim_now is None:
            self._record_event('raw_cmd_dropped_no_clock', seq=self.seq, command=_twist_to_dict(msg))
            return

        delay = self._latency_sec() if self.timing_mode == 'sim_time_realworld' else 0.0
        command = PendingCommand(
            seq=self.seq,
            command=_twist_to_dict(msg),
            receive_wall_time=self._wall_time(),
            receive_sim_time=sim_now,
            eligible_sim_time=sim_now + delay,
            delay_sec=delay,
        )
        self.pending.append(command)
        self._prune_queue(sim_now)
        self._record_event(
            'raw_cmd_received',
            seq=command.seq,
            command=command.command,
            receive_sim_time=command.receive_sim_time,
            eligible_sim_time=command.eligible_sim_time,
            delay_sec=command.delay_sec,
            latency_policy=self.latency_policy,
        )

    def _on_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data or '{}')
        except Exception:
            payload = {'raw': msg.data}
        status = str(payload.get('status') or payload.get('event') or 'status')
        debug = payload.get('debug') if isinstance(payload.get('debug'), dict) else {}
        elapsed = debug.get('elapsed_sec')
        if elapsed is None:
            elapsed = payload.get('elapsed_sec')
        try:
            if elapsed is not None:
                self.latest_measured_latency_sec = max(float(elapsed), 0.0)
        except Exception:
            pass
        self._record_event(
            'internnav_status',
            status=status,
            latest_measured_latency_sec=self.latest_measured_latency_sec,
            payload=payload,
        )

    def _on_task_reset(self, msg: Int16) -> None:
        self.reset_episode = int(msg.data)
        self.pending.clear()
        self.last_released_command = None
        self.episode_started = False
        self._publish_zero()
        self._record_event('task_reset', episode=self.reset_episode)

    def _on_eval_ready(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data or '{}')
        except Exception:
            return
        if payload.get('stage') != 'episode':
            return
        self.episode_started = bool(payload.get('ready'))
        if not self.episode_started:
            self.pending.clear()
            self.last_released_command = None
            self._publish_zero()
        self._record_event('eval_ready', payload=payload, episode_started=self.episode_started)

    def _prune_queue(self, sim_now: float) -> None:
        cutoff = sim_now - self.max_queue_sec
        before = len(self.pending)
        self.pending = [cmd for cmd in self.pending if cmd.eligible_sim_time >= cutoff]
        dropped = before - len(self.pending)
        if dropped:
            self._record_event('pending_cmd_pruned', dropped=dropped, queue_len=len(self.pending))

    def _publish_zero(self) -> None:
        self.cmd_pub.publish(Twist())

    def _on_timer(self) -> None:
        sim_now = self.current_sim_time
        if sim_now is None:
            return
        if not self.episode_started:
            self._publish_zero()
            return

        eligible = [cmd for cmd in self.pending if cmd.eligible_sim_time <= sim_now]
        if eligible:
            cmd = eligible[-1]
            self.pending = [item for item in self.pending if item.eligible_sim_time > sim_now]
            self.last_released_command = cmd.command
            self.released_count += 1
            self.cmd_pub.publish(_twist_from_dict(cmd.command))
            self._record_event(
                'cmd_released',
                seq=cmd.seq,
                command=cmd.command,
                receive_sim_time=cmd.receive_sim_time,
                eligible_sim_time=cmd.eligible_sim_time,
                release_sim_time=sim_now,
                delay_sec=cmd.delay_sec,
                queue_len=len(self.pending),
            )
            return

        if self.action_hold_policy == 'zero' or self.last_released_command is None:
            self._publish_zero()
        else:
            self.cmd_pub.publish(_twist_from_dict(self.last_released_command))

    def _summary(self) -> dict[str, Any]:
        samples = list(self.rtf_samples)
        summary: dict[str, Any] = {
            'timing_mode': self.timing_mode,
            'latency_policy': self.latency_policy,
            'model_latency_sec': self.model_latency_sec,
            'planning_period_sec': self.planning_period_sec,
            'latest_measured_latency_sec': self.latest_measured_latency_sec,
            'raw_cmd_count': self.raw_count,
            'released_cmd_count': self.released_count,
            'pending_cmd_count': len(self.pending),
            'rtf_sample_count': len(samples),
            'timing_valid_for_realworld': self.timing_mode == 'sim_time_realworld',
        }
        if samples:
            sorted_samples = sorted(samples)
            p50_idx = int(0.50 * (len(sorted_samples) - 1))
            p95_idx = int(0.95 * (len(sorted_samples) - 1))
            summary.update(
                {
                    'rtf_mean': statistics.fmean(samples),
                    'rtf_min': min(samples),
                    'rtf_max': max(samples),
                    'rtf_p50': sorted_samples[p50_idx],
                    'rtf_p95': sorted_samples[p95_idx],
                }
            )
        return summary

    def _write_summary(self) -> None:
        if not self.summary_path:
            return
        try:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(
                json.dumps(self._summary(), indent=2, sort_keys=True),
                encoding='utf-8',
            )
        except Exception as exc:
            self.get_logger().warn(f'failed to write timing summary: {exc!r}')

    def destroy_node(self) -> bool:
        self._record_event('timing_manager_stopped', summary=self._summary())
        self._write_summary()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = InternNavTimingManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
