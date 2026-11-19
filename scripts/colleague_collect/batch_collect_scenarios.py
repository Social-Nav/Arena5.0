#!/usr/bin/env python3
"""Batch collect Arena scenario episodes.

This script does not launch Arena/Isaac. Start Arena manually with
``save_data:=true`` first, wait until RViz is open, then run this script. For a
``headless:=2`` launch, pass ``--skip-rviz-check`` after the ROS/Nav2 nodes are ready.

For each scenario it:
  1. sets /task_generator_node parameter ``task.scenario.file``
  2. calls /task_generator_node/reset_task
  3. waits for the Nav2 NavigateToPose terminal status
  4. waits for /data_logger/episode_saved when navigation succeeds

Failed / canceled episodes are discarded by the Isaac data logger; this script
can retry or skip them.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import Empty, Int16
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import Trigger
from goal_epoch import GoalEpoch  # Bundled next to this script; no Arena patch required.


TERMINAL = {
    GoalStatus.STATUS_SUCCEEDED,
    GoalStatus.STATUS_ABORTED,
    GoalStatus.STATUS_CANCELED,
}

RUNNING = {
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
    GoalStatus.STATUS_CANCELING,
}


@dataclass
class EpisodeResult:
    status: int
    saved: bool
    saved_before: int = 0
    error: Optional[str] = None
    attempt_started_at: Optional[float] = None
    nav_started_at: Optional[float] = None
    nav_terminal_at: Optional[float] = None
    episode_saved_at: Optional[float] = None
    attempt_finished_at: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.status == GoalStatus.STATUS_SUCCEEDED and self.saved

    @staticmethod
    def _elapsed(start: Optional[float], end: Optional[float]) -> Optional[float]:
        if start is None or end is None:
            return None
        return max(0.0, end - start)

    @property
    def setup_seconds(self) -> Optional[float]:
        return self._elapsed(self.attempt_started_at, self.nav_started_at)

    @property
    def navigation_seconds(self) -> Optional[float]:
        return self._elapsed(self.nav_started_at, self.nav_terminal_at)

    @property
    def save_seconds(self) -> Optional[float]:
        return self._elapsed(self.nav_terminal_at, self.episode_saved_at)

    @property
    def collection_seconds(self) -> Optional[float]:
        """Navigation start through durable episode save; excludes setup/loading."""
        if not self.ok:
            return None
        return self._elapsed(self.nav_started_at, self.episode_saved_at)

    @property
    def attempt_total_seconds(self) -> Optional[float]:
        return self._elapsed(self.attempt_started_at, self.attempt_finished_at)


class BatchCollector:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.node = rclpy.create_node("arena_batch_collect_scenarios")
        self.goal_epoch = GoalEpoch()
        self.last_nav_status: Optional[int] = None
        self.seen_running_after_reset = False
        self.episode_saved_count = 0
        self.task_reset_count = -1
        self.attempt_started_at: Optional[float] = None
        self.nav_started_at: Optional[float] = None
        self.nav_terminal_at: Optional[float] = None
        self.episode_saved_at: Optional[float] = None
        self.timings_file = Path(args.timings_file) if args.timings_file else None
        self.episode_map_file = (
            Path(args.episode_map_file) if args.episode_map_file else None
        )
        self.successful_collection_seconds: list[float] = []
        self.completed_scenarios_file = (
            Path(args.completed_scenarios_file)
            if args.completed_scenarios_file is not None
            else None
        )
        self.completed_scenarios = self._load_completed_scenarios()

        nav_status_topic = (
            f"{args.task_node}/{args.robot}/navigate_to_pose/_action/status"
        )
        self.node.create_subscription(
            GoalStatusArray,
            nav_status_topic,
            self._on_nav_status,
            10,
        )
        self.node.create_subscription(
            Empty,
            "/data_logger/episode_saved",
            self._on_episode_saved,
            10,
        )
        self.node.create_subscription(
            Int16,
            f"{args.task_node}/task_reset",
            self._on_task_reset,
            10,
        )

        self.param_client = self.node.create_client(
            SetParameters, f"{args.task_node}/set_parameters"
        )
        self.reset_client = self.node.create_client(
            EmptySrv, f"{args.task_node}/reset_task"
        )
        self.checked_reset_client = self.node.create_client(
            Trigger, f"{args.task_node}/reset_task_checked"
        )

    def _load_completed_scenarios(self) -> set[str]:
        if self.completed_scenarios_file is None:
            return set()
        try:
            return {
                line.strip()
                for line in self.completed_scenarios_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            }
        except FileNotFoundError:
            return set()

    def _mark_scenario_completed(self, scenario: str) -> None:
        if scenario in self.completed_scenarios:
            return
        if self.completed_scenarios_file is not None:
            self.completed_scenarios_file.parent.mkdir(parents=True, exist_ok=True)
            with self.completed_scenarios_file.open("a", encoding="utf-8") as stream:
                stream.write(f"{scenario}\n")
                stream.flush()
                os.fsync(stream.fileno())
        self.completed_scenarios.add(scenario)
        print(f"  recorded completed scenario: {scenario}")

    def _on_nav_status(self, msg: GoalStatusArray) -> None:
        current = self.goal_epoch.observe(msg.status_list)
        if current is None:
            return
        status = current.status
        self.last_nav_status = status
        if status in RUNNING:
            if not self.seen_running_after_reset:
                self.nav_started_at = time.monotonic()
            self.seen_running_after_reset = True
        elif (
            status in TERMINAL
            and self.seen_running_after_reset
            and self.nav_terminal_at is None
        ):
            self.nav_terminal_at = time.monotonic()

    def _on_episode_saved(self, _msg: Empty) -> None:
        self.episode_saved_count += 1
        self.episode_saved_at = time.monotonic()

    def _on_task_reset(self, msg: Int16) -> None:
        self.task_reset_count = msg.data

    def spin_until(self, predicate, timeout_s: float, label: str) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if predicate():
                return True
        print(f"[TIMEOUT] {label}", file=sys.stderr)
        return False

    def wait_ready(self) -> None:
        print("[Ready] Waiting for task_generator services...")
        if not self.param_client.wait_for_service(timeout_sec=self.args.ready_timeout):
            raise RuntimeError(f"{self.args.task_node}/set_parameters not available")
        if not self.reset_client.wait_for_service(timeout_sec=self.args.ready_timeout):
            raise RuntimeError(f"{self.args.task_node}/reset_task not available")

        print("[Ready] Waiting for ROS nodes...")
        required_nodes = [
            self.args.task_node,
            f"{self.args.task_node}/{self.args.robot}/bt_navigator",
            f"{self.args.task_node}/{self.args.robot}/controller_server",
            f"{self.args.task_node}/{self.args.robot}/planner_server",
        ]
        if not self.args.skip_rviz_check:
            required_nodes.append(f"{self.args.task_node}/rviz_config_generator")
        for node_name in required_nodes:
            self._wait_ros_node(node_name, self.args.ready_timeout)

        if not self.args.skip_rviz_check:
            print("[Ready] Waiting for rviz2 process...")
            self._wait_process("rviz2", self.args.ready_timeout)

        if self.args.check_lifecycle:
            print("[Ready] Waiting for Nav2 lifecycle active...")
            for name in ("bt_navigator", "controller_server", "planner_server"):
                self._wait_lifecycle_active(
                    f"{self.args.task_node}/{self.args.robot}/{name}",
                    self.args.ready_timeout,
                )

        if self.args.settle_s > 0:
            print(f"[Ready] Settling for {self.args.settle_s:.1f}s...")
            end = time.monotonic() + self.args.settle_s
            while time.monotonic() < end:
                rclpy.spin_once(self.node, timeout_sec=0.1)

        print("[Ready] OK. Starting scenario batch.")

    def _wait_ros_node(self, node_name: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            nodes = {name for name, _ns in self.node.get_node_names_and_namespaces()}
            # get_node_names_and_namespaces returns name without namespace, so also
            # use ros2 node list for exact fully-qualified checks.
            if self._ros_node_list_contains(node_name):
                print(f"  node ready: {node_name}")
                return
            rclpy.spin_once(self.node, timeout_sec=0.2)
        raise RuntimeError(f"node not found: {node_name}")

    @staticmethod
    def _ros_node_list_contains(node_name: str) -> bool:
        try:
            out = subprocess.run(
                ["ros2", "node", "list"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).stdout
        except Exception:
            return False
        return node_name in set(line.strip() for line in out.splitlines())

    @staticmethod
    def _wait_process(pattern: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            res = subprocess.run(
                ["pgrep", "-f", pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if res.returncode == 0:
                print(f"  process ready: {pattern}")
                return
            time.sleep(0.5)
        raise RuntimeError(f"process not found: {pattern}")

    @staticmethod
    def _wait_lifecycle_active(node_name: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                res = subprocess.run(
                    ["ros2", "lifecycle", "get", node_name],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                if "active [3]" in res.stdout:
                    print(f"  lifecycle active: {node_name}")
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError(f"lifecycle not active: {node_name}")

    def set_scenario(self, scenario: str) -> None:
        req = SetParameters.Request()
        req.parameters = [
            Parameter(
                name="task.scenario.file",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=scenario,
                ),
            )
        ]
        future = self.param_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        if future.result() is None:
            raise RuntimeError("set_parameters timed out")
        result = future.result().results[0]
        if not result.successful:
            raise RuntimeError(f"failed to set scenario {scenario}: {result.reason}")

    def reset_task(self) -> None:
        checked = self.checked_reset_client.service_is_ready()
        client = self.checked_reset_client if checked else self.reset_client
        request = Trigger.Request() if checked else EmptySrv.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=self.args.reset_service_timeout,
        )
        if future.result() is None:
            raise RuntimeError(
                f"reset_task timed out after {self.args.reset_service_timeout:.1f}s"
            )
        if checked and not future.result().success:
            raise RuntimeError(future.result().message)

    def collect_one(self, scenario: str, attempt: int) -> EpisodeResult:
        print(f"\n[Scenario] {scenario} attempt {attempt}")
        self.attempt_started_at = time.monotonic()
        self.seen_running_after_reset = False
        self.last_nav_status = None
        self.nav_started_at = None
        self.nav_terminal_at = None
        self.episode_saved_at = None
        saved_before = self.episode_saved_count
        reset_before = self.task_reset_count

        print(f"  setting task.scenario.file={scenario}")
        self.set_scenario(scenario)
        self.goal_epoch.begin()

        print(
            "  calling reset_task "
            f"(service timeout={self.args.reset_service_timeout:.1f}s)"
        )
        self.reset_task()

        reset_acknowledged = self.spin_until(
            lambda: self.task_reset_count != reset_before,
            self.args.reset_timeout,
            "waiting for task_reset message",
        )
        if not reset_acknowledged:
            raise RuntimeError("Reset did not publish a successful task_reset event")

        print("  waiting for Nav2 terminal status...")

        def terminal_after_running() -> bool:
            if self.last_nav_status is None:
                return False
            if self.last_nav_status in RUNNING:
                return False
            if self.last_nav_status in TERMINAL:
                # Avoid immediately consuming a stale terminal status from the
                # previous goal. Prefer seeing running first, but allow terminal
                # after a short grace period for very short scenarios.
                return self.seen_running_after_reset
            return False

        ok = self.spin_until(
            terminal_after_running,
            self.args.nav_timeout,
            "waiting for Nav2 succeeded/aborted/canceled",
        )
        status = self.last_nav_status if ok and self.last_nav_status is not None else -1

        if status != GoalStatus.STATUS_SUCCEEDED:
            print(f"  result: not saved, nav status={status}")
        return self._episode_result(status=status, saved=False, saved_before=saved_before)

        print("  Nav2 succeeded. Waiting for /data_logger/episode_saved...")
        saved = self.spin_until(
            lambda: self.episode_saved_count > saved_before,
            self.args.save_timeout,
            "waiting for /data_logger/episode_saved",
        )
        print(f"  result: status=SUCCEEDED saved={saved}")
        return self._episode_result(status=status, saved=saved, saved_before=saved_before)

    def _episode_result(
        self,
        status: int,
        saved: bool,
        saved_before: int = 0,
        error: Optional[str] = None,
    ) -> EpisodeResult:
        return EpisodeResult(
            status=status,
            saved=saved,
            saved_before=saved_before,
            error=error,
            attempt_started_at=self.attempt_started_at,
            nav_started_at=self.nav_started_at,
            nav_terminal_at=self.nav_terminal_at,
            episode_saved_at=self.episode_saved_at,
            attempt_finished_at=time.monotonic(),
        )

    def _record_episode_map(self, scenario: str, result: EpisodeResult) -> None:
        if self.episode_map_file is None:
            return
        episode_index = self.args.episode_base + result.saved_before
        self.episode_map_file.parent.mkdir(parents=True, exist_ok=True)
        with self.episode_map_file.open("a", encoding="utf-8") as stream:
            stream.write(f"{scenario}\t{episode_index}\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _format_seconds(value: Optional[float]) -> str:
        return "" if value is None else f"{value:.3f}"

    def _record_timing(
        self,
        scenario: str,
        attempt: int,
        result: EpisodeResult,
    ) -> None:
        collection = result.collection_seconds
        if collection is not None:
            self.successful_collection_seconds.append(collection)

        print(
            "  [Timing] "
            f"collection={self._format_seconds(collection) or '-'}s "
            f"setup={self._format_seconds(result.setup_seconds) or '-'}s "
            f"navigation={self._format_seconds(result.navigation_seconds) or '-'}s "
            f"save={self._format_seconds(result.save_seconds) or '-'}s "
            f"attempt_total={self._format_seconds(result.attempt_total_seconds) or '-'}s"
        )

        if self.timings_file is None:
            return
        self.timings_file.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "timestamp_utc",
            "world",
            "scenario",
            "attempt",
            "outcome",
            "nav_status",
            "saved",
            "collection_seconds",
            "setup_seconds",
            "navigation_seconds",
            "save_seconds",
            "attempt_total_seconds",
            "error",
        ]
        needs_header = not self.timings_file.exists() or self.timings_file.stat().st_size == 0
        with self.timings_file.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "world": self.args.world,
                    "scenario": scenario,
                    "attempt": attempt,
                    "outcome": "success" if result.ok else "failed",
                    "nav_status": result.status,
                    "saved": str(result.saved).lower(),
                    "collection_seconds": self._format_seconds(collection),
                    "setup_seconds": self._format_seconds(result.setup_seconds),
                    "navigation_seconds": self._format_seconds(result.navigation_seconds),
                    "save_seconds": self._format_seconds(result.save_seconds),
                    "attempt_total_seconds": self._format_seconds(result.attempt_total_seconds),
                    "error": result.error or "",
                }
            )
            stream.flush()
            os.fsync(stream.fileno())

    def _print_timing_summary(self) -> None:
        values = self.successful_collection_seconds
        if not values:
            return
        print(
            "[Timing summary] "
            f"successful={len(values)} "
            f"mean={statistics.fmean(values):.3f}s "
            f"median={statistics.median(values):.3f}s "
            f"min={min(values):.3f}s "
            f"max={max(values):.3f}s"
        )

    def run(self) -> int:
        pending = [
            scenario
            for scenario in self.args.scenarios
            if scenario not in self.completed_scenarios
        ]
        skipped = len(self.args.scenarios) - len(pending)
        if skipped:
            print(f"[Resume] Skipping {skipped} already completed scenario(s).")
        if not pending:
            if self.args.failed_scenarios_file is not None:
                failure_file = Path(self.args.failed_scenarios_file)
                failure_file.parent.mkdir(parents=True, exist_ok=True)
                failure_file.write_text("", encoding="utf-8")
            print("[DONE] All requested scenarios were already completed.")
            return 0

        self.wait_ready()
        failed: list[tuple[str, str]] = []
        for scenario in pending:
            success = False
            failure_reason = "unknown failure"
            for attempt in range(1, self.args.retries + 2):
                try:
                    result = self.collect_one(scenario, attempt)
                except Exception as exc:
                    # A reset service can itself time out (or a ROS service can
                    # disappear). Treat that exactly like a failed navigation
                    # attempt so one unstable scenario does not abort a batch.
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    print(f"  attempt failed: {failure_reason}", file=sys.stderr)
                    result = self._episode_result(
                        status=-1,
                        saved=False,
                        error=failure_reason,
                    )

                self._record_timing(scenario, attempt, result)

                if result.ok:
                    self._record_episode_map(scenario, result)
                    self._mark_scenario_completed(scenario)
                    success = True
                    break
                if result.error is None:
                    failure_reason = f"nav status={result.status}, saved={result.saved}"
                else:
                    failure_reason = result.error
                if attempt <= self.args.retries:
                    print(f"  retrying {scenario}...")
            if not success:
                failed.append((scenario, failure_reason))
                print(f"  skipping {scenario} after {self.args.retries + 1} failed attempt(s).",
                      file=sys.stderr)
                if not self.args.continue_on_fail:
                    break

        if self.args.failed_scenarios_file is not None:
            failure_file = Path(self.args.failed_scenarios_file)
            failure_file.parent.mkdir(parents=True, exist_ok=True)
            failure_file.write_text(
                "".join(f"{scenario}\t{reason}\n" for scenario, reason in failed),
                encoding="utf-8",
            )

        if failed:
            self._print_timing_summary()
            print(
                f"\n[DONE] Failed scenarios: {', '.join(scenario for scenario, _ in failed)}",
                file=sys.stderr,
            )
            if self.args.continue_on_fail:
                print("[DONE] Continuing after partial collection as requested.", file=sys.stderr)
                return 0
            return 1
        self._print_timing_summary()
        print("\n[DONE] All scenarios collected successfully.")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="", help="World name written to timing records.")
    parser.add_argument(
        "--robot",
        default="Ai2_Bot2",
        help="Robot namespace under /task_generator_node.",
    )
    parser.add_argument(
        "--task-node",
        default="/task_generator_node",
        help="Task generator node namespace.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["default", "default_1", "default_2", "default_3", "default_4"],
        help="Scenario names to collect in order.",
    )
    parser.add_argument("--ready-timeout", type=float, default=1500.0)
    parser.add_argument(
        "--reset-service-timeout", type=float,
        default=float(os.environ.get("ARENA_RESET_SERVICE_TIMEOUT", "1500")),
    )
    parser.add_argument("--reset-timeout", type=float, default=1500.0)
    parser.add_argument("--nav-timeout", type=float, default=1500.0)
    parser.add_argument("--save-timeout", type=float, default=1500.0)
    parser.add_argument("--settle-s", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=0,
                        help="Retries after the initial attempt (2 means three attempts total).")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="After all attempts fail, record and skip that scenario. The process exits 0 "
             "after completing the remaining scenarios.",
    )
    parser.add_argument(
        "--failed-scenarios-file",
        help="Optional TSV report path for scenarios skipped after all attempts.",
    )
    parser.add_argument(
        "--completed-scenarios-file",
        help="Optional newline-delimited resume state. Successful scenarios are "
             "appended immediately and skipped on later runs.",
    )
    parser.add_argument(
        "--episode-base",
        type=int,
        default=0,
        help="Episode index of the first episode saved during this batch run.",
    )
    parser.add_argument(
        "--episode-map-file",
        help="Optional TSV file recording scenario<TAB>episode_index for successful saves.",
    )
    parser.add_argument(
        "--timings-file",
        help="Optional append-only CSV containing one timing row per attempt.",
    )
    parser.add_argument(
        "--skip-rviz-check",
        action="store_true",
        help="Do not wait for rviz_config_generator or the rviz2 process. Use this with "
             "headless:=2; task-generator, Nav2, lifecycle and service checks remain enabled.",
    )
    parser.add_argument(
        "--no-lifecycle-check",
        dest="check_lifecycle",
        action="store_false",
        help="Skip waiting for Nav2 lifecycle active.",
    )
    parser.set_defaults(check_lifecycle=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    try:
        collector = BatchCollector(args)
        return collector.run()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
