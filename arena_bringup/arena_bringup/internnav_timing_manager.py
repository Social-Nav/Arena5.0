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


def relay_enabled_for_timing_mode(timing_mode: str | None) -> bool:
    """Is this node the robot's command source for ``timing_mode``?

    THIS IS THE SINGLE PYTHON-SIDE EXPRESSION OF A PREDICATE THAT IS ALSO WRITTEN IN BASH.
    Its peer lives in ``_meta/docker/features/internnav/main`` (the ``cmd_vel_topic_explicit``
    / ``timing_mode != wall`` test that redirects the InternNav client onto the raw topic).
    Both sides answer the same question -- *who publishes the robot's ``cmd_vel``?* -- and
    they must agree:

    * ``timing_mode == 'wall'``: the client is **not** redirected, so it publishes straight
      onto the robot's ``cmd_vel``.  This node's input topic then has no publisher at all, it
      can never forward anything, and any output it produces contends with the client on the
      robot's own command topic.  It is **not** a command source; return ``False``.
    * anything else (``sim_time_realworld``): the client **is** redirected onto the raw topic,
      so relaying it -- delayed, with ``hold_last`` republication -- is this node's whole job
      and the robot has no other command source.  Return ``True``.

    Only the *relay* responsibility is governed by this predicate.  Clock observation and every
    timing artifact (``rtf.csv``, ``internnav_timing_summary.json``,
    ``internnav_timing_trace.jsonl``) are produced in **every** mode and are deliberately not
    gated -- this node is their sole producer, so gating the node itself (for instance with a
    launch-level ``condition=``) would silently destroy the RTF record and the wall->sim
    conversion bridge that the timing analysis depends on.

    Because the two expressions can still drift, the node does not merely trust this predicate:
    it also *observes* whether its input topic actually has a publisher and reports any
    disagreement loudly.  See :meth:`InternNavTimingManager._check_relay_topology`.
    """
    return str(timing_mode or '').strip().lower() != 'wall'


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

    The node has **two independent responsibilities** and they are gated differently:

    1. **Observe** ``/clock`` and write the timing artifacts (``rtf.csv``,
       ``internnav_timing_summary.json``, ``internnav_timing_trace.jsonl``).  Wanted in every
       mode, and this node is their **sole producer**.  Never gated.
    2. **Relay** the raw cmd_vel topic onto the robot's ``cmd_vel``.  Wanted only when the
       InternNav client has been redirected onto the raw topic, i.e. only when
       ``timing_mode != 'wall'``.  Gated by :func:`relay_enabled_for_timing_mode`.
    """

    #: Wall-clock grace period before an input-topology mismatch is reported.  The InternNav
    #: client's publisher appears only once its model server is up, which took 12-58 WALL s in
    #: the runs on record, so a shorter grace period would manufacture false reports.
    INPUT_TOPOLOGY_GRACE_SEC = 30.0

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

        # Is this node the robot's command source in this mode?  See
        # relay_enabled_for_timing_mode() -- the single Python-side expression of the predicate.
        self.relay_enabled = relay_enabled_for_timing_mode(self.timing_mode)
        self.emitted_cmd_count = 0
        self.suppressed_cmd_count = 0
        self.input_publisher_count: int | None = None
        self.input_publisher_seen = False
        self.topology_mismatch: str | None = None
        self._reported_topology_mismatch: str | None = None
        self._start_monotonic = self._monotonic()

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

        self.input_cmd_vel_topic = input_topic
        self.output_cmd_vel_topic = output_topic

        # Create the output publisher ONLY when this node is a command source.  Not creating it
        # is deliberate and stronger than guarding each publish: a registered publisher is itself
        # observable (`ros2 topic info -v` counts it), so leaving one behind would keep telling a
        # live publisher census that this node can drive the robot when it must not.
        if self.relay_enabled:
            self.cmd_pub = self.create_publisher(Twist, output_topic, 10)
        else:
            self.cmd_pub = None
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
        self.summary_timer = self.create_timer(2.0, self._on_periodic)

        self._record_event(
            'timing_manager_started',
            timing_mode=self.timing_mode,
            latency_policy=self.latency_policy,
            model_latency_sec=self.model_latency_sec,
            planning_period_sec=self.planning_period_sec,
            input_cmd_vel_topic=input_topic,
            output_cmd_vel_topic=output_topic,
            status_topic=status_topic,
            relay_enabled=self.relay_enabled,
        )
        if self.relay_enabled:
            self.get_logger().info(
                f'timing_mode={self.timing_mode!r}: relaying {input_topic!r} -> {output_topic!r} '
                f'(this node is the robot command source)'
            )
        else:
            self.get_logger().info(
                f'timing_mode={self.timing_mode!r}: command relay DISABLED, no publisher created on '
                f'{output_topic!r} (the InternNav client publishes there directly in this mode). '
                f'Clock observation and timing artifacts remain active.'
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

    def _emit_cmd(self, msg: Twist) -> None:
        """The single choke point for every emission on the output cmd_vel topic.

        Both counters are always maintained, whichever branch is taken, so the node's own
        telemetry states positively how many commands it put on the robot's command topic and
        how many it withheld.  ``suppressed_cmd_count`` is what makes "emitted nothing" provable
        rather than merely unobserved -- a test or a run that shows 0 emitted and 0 suppressed
        did not exercise the path at all.
        """
        if not self.relay_enabled or self.cmd_pub is None:
            self.suppressed_cmd_count += 1
            return
        self.emitted_cmd_count += 1
        self.cmd_pub.publish(msg)

    def _publish_zero(self) -> None:
        self._emit_cmd(Twist())

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
            self._emit_cmd(_twist_from_dict(cmd.command))
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
            self._emit_cmd(_twist_from_dict(self.last_released_command))

    def _check_relay_topology(self) -> None:
        """Observe whether the input topic really has a publisher, and report disagreement.

        This is a **reporter, never a gate.**  ``relay_enabled`` alone decides whether anything is
        emitted; this method only compares the declared role against the observed topology and
        makes a mismatch loud.  Three properties are deliberate:

        * **Never latched.**  The count is re-read every time.  A once-only check taken at startup
          would see the client's publisher missing (it appears only after the model server is up)
          and would permanently record a mismatch that does not exist.
        * **Never used to suppress an emission.**  If this were allowed to gate, a transient
          discovery gap in ``sim_time_realworld`` would silently drop a real command.
        * **Loud in both directions.**  Declaring a relay role with no input publisher, and *not*
          declaring one while something publishes on the raw topic, are both misconfigurations:
          the first floods the robot's command topic with hold/zero output that nobody asked for,
          the second leaves the robot with no command source at all.

        It exists because the predicate is split across two languages (see
        :func:`relay_enabled_for_timing_mode`), so it can drift.  Rather than re-deriving the
        other side's decision, this observes its consequence.
        """
        try:
            count = int(self.count_publishers(self.input_cmd_vel_topic))
        except Exception as exc:  # pragma: no cover - depends on live graph state
            # Reported, not swallowed: a probe that fails silently is indistinguishable from a
            # probe that ran and found nothing.
            self.get_logger().warn(
                f'could not count publishers on {self.input_cmd_vel_topic!r}: {exc!r}'
            )
            return
        self.input_publisher_count = count
        if count > 0:
            self.input_publisher_seen = True

        elapsed = self._monotonic() - self._start_monotonic
        mismatch: str | None = None
        if self.relay_enabled and not self.input_publisher_seen and elapsed >= self.INPUT_TOPOLOGY_GRACE_SEC:
            mismatch = 'relay_enabled_but_input_topic_has_no_publisher'
        elif not self.relay_enabled and count > 0:
            mismatch = 'relay_disabled_but_input_topic_has_a_publisher'

        self.topology_mismatch = mismatch
        if mismatch != self._reported_topology_mismatch:
            self._reported_topology_mismatch = mismatch
            if mismatch:
                self.get_logger().error(
                    f'timing topology mismatch: {mismatch} '
                    f'(timing_mode={self.timing_mode!r}, relay_enabled={self.relay_enabled}, '
                    f'input={self.input_cmd_vel_topic!r} publishers={count}, '
                    f'output={self.output_cmd_vel_topic!r}). The InternNav client redirect in '
                    f'_meta/docker/features/internnav/main and this node disagree about who '
                    f'publishes the robot cmd_vel.'
                )
                self._record_event(
                    'timing_topology_mismatch',
                    mismatch=mismatch,
                    relay_enabled=self.relay_enabled,
                    input_cmd_vel_topic=self.input_cmd_vel_topic,
                    input_publisher_count=count,
                    output_cmd_vel_topic=self.output_cmd_vel_topic,
                )
            else:
                self.get_logger().info(
                    f'timing topology mismatch resolved '
                    f'(input={self.input_cmd_vel_topic!r} publishers={count})'
                )
                self._record_event('timing_topology_mismatch_resolved', input_publisher_count=count)

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
            # Relay state, so a run's own artifact records whether this node was a command
            # source and what it actually put on the robot's command topic.  Additive: no
            # pre-existing key changes meaning.
            'relay_enabled': self.relay_enabled,
            'input_cmd_vel_topic': self.input_cmd_vel_topic,
            'output_cmd_vel_topic': self.output_cmd_vel_topic,
            'output_publisher_created': self.cmd_pub is not None,
            'emitted_cmd_count': self.emitted_cmd_count,
            'suppressed_cmd_count': self.suppressed_cmd_count,
            'input_publisher_count': self.input_publisher_count,
            'input_publisher_seen': self.input_publisher_seen,
            'topology_mismatch': self.topology_mismatch,
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

    def _on_periodic(self) -> None:
        """2 s housekeeping: observe the relay topology, then persist the summary.

        The topology observation runs first and **unconditionally**, because ``_write_summary``
        returns early when no ``record_data_dir`` is configured; folding the check into the
        writer would silently disable it for any run without an output directory.
        """
        self._check_relay_topology()
        self._write_summary()

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
