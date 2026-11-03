"""Deterministic, ROS-free contract for the proposed LISN slow loop.

The module intentionally contains no ROS imports, network clients, model code,
publishers, services, launch entry points, or controller implementation.  It
models the P3 scheduling and atomic apply/readback/rollback rules with injected
adapters so failure behavior can be proven offline before any runtime wiring is
authorized.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from arena_bringup.hierarchical_modulation_contract import (
    CONTROLLER_PARAMETER_RANGES,
    ContractValidationError,
    validate_modulation_snapshot,
)


POLICY_SCHEMA = "arena.lisn_slow_loop_contract"
POLICY_VERSION = 1
EVIDENCE_SCHEMA = "arena.lisn_slow_loop_evidence"
EVIDENCE_VERSION = 1
ACTIVE_CONTROL_PATH = "lisn_style_social_mpc"
ACTIVE_EXECUTION_INTERFACE = "nav2_follow_path_social_mpc"
ISOLATED_CONTROL_INTERFACES = frozenset(
    {"vln_direct_cmd_vel", "nav2_follow_path_rpp"}
)
STAGE_ORDER = (
    "proposal",
    "validation",
    "apply",
    "readback",
    "rollback",
    "evidence",
    "hard_wall",
)
LOGICAL_PARAMETERS = tuple(CONTROLLER_PARAMETER_RANGES)


class SlowLoopPolicyError(ValueError):
    """A checked-in slow-loop policy is internally inconsistent."""


class AdapterError(RuntimeError):
    """The injected parameter adapter could not complete an operation."""


@dataclass(frozen=True)
class ApplyResult:
    successful: bool
    reason: str = ""


class ParameterAdapter(Protocol):
    """Minimal offline stand-in for a future ROS parameter transaction."""

    def read(self, names: tuple[str, ...]) -> Mapping[str, float]: ...

    def set_atomically(self, values: Mapping[str, float]) -> ApplyResult: ...


@dataclass(frozen=True)
class ObservationIdentity:
    episode_id: str
    observation_id: str
    frame_id: str
    ros_stamp_ns: int
    received_monotonic_ns: int
    content_sha256: str


@dataclass(frozen=True)
class SlowLoopTimeline:
    proposal_completed_ns: int
    validation_completed_ns: int
    apply_completed_ns: int
    readback_completed_ns: int
    evidence_completed_ns: int
    rollback_completed_ns: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "proposal_completed_ns": self.proposal_completed_ns,
            "validation_completed_ns": self.validation_completed_ns,
            "apply_completed_ns": self.apply_completed_ns,
            "readback_completed_ns": self.readback_completed_ns,
            "evidence_completed_ns": self.evidence_completed_ns,
            "rollback_completed_ns": self.rollback_completed_ns,
        }


@dataclass(frozen=True)
class SlowLoopPolicy:
    policy_sha256: str
    period_ns: int
    max_observation_age_ns: int
    stage_deadline_offsets_ns: Mapping[str, int]
    active_control_path: str
    active_execution_interface: str
    isolated_control_interfaces: frozenset[str]
    logical_parameters: tuple[str, ...]
    runtime_ready: bool
    runtime_blockers: tuple[str, ...]
    parameter_bindings: Mapping[str, Mapping[str, Any]]
    enforce_observation_age: bool = True

    def deadline_ns(self, scheduled_for_ns: int, stage: str) -> int:
        return scheduled_for_ns + int(self.stage_deadline_offsets_ns[stage])


@dataclass
class SlowLoopState:
    episode_id: str
    epoch_monotonic_ns: int
    next_tick_index: int = 0
    accepted_snapshot: dict[str, Any] | None = None
    accepted_snapshot_sha256: str | None = None
    accepted_update_id: str | None = None
    used_observation_ids: set[str] = field(default_factory=set)
    update_ledger: dict[str, str] = field(default_factory=dict)
    faulted: bool = False


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise SlowLoopPolicyError(
            f"{path} fields differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON used for update and evidence identity.

    ``allow_nan=False`` rejects NaN and infinities rather than assigning a
    platform-specific or non-JSON identity.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_slow_loop_policy(path: Path) -> SlowLoopPolicy:
    raw = path.read_bytes()
    value = json.loads(raw)
    required = {
        "schema",
        "schema_version",
        "runtime_authorized",
        "schedule",
        "budgets",
        "isolation",
        "atomic_apply",
        "evidence",
    }
    _require_exact_fields(value, required, "root")
    if value["schema"] != POLICY_SCHEMA or value["schema_version"] != POLICY_VERSION:
        raise SlowLoopPolicyError("unsupported slow-loop policy schema/version")
    if value["runtime_authorized"] is not False:
        raise SlowLoopPolicyError("slow-loop runtime must remain unauthorized in Git")

    schedule = value["schedule"]
    _require_exact_fields(
        schedule,
        {
            "clock",
            "period_ns",
            "first_tick_offset_ns",
            "catch_up",
            "max_observation_age_ns",
            "one_observation_per_tick",
        },
        "schedule",
    )
    if schedule["clock"] != "time.monotonic_ns":
        raise SlowLoopPolicyError("scheduler must use time.monotonic_ns")
    period_ns = int(schedule["period_ns"])
    if period_ns != 10_000_000_000:
        raise SlowLoopPolicyError("LISN slow-loop period must remain 10 seconds")
    if int(schedule["first_tick_offset_ns"]) != period_ns:
        raise SlowLoopPolicyError("first tick must be one full period after arming")
    if schedule["catch_up"] is not False:
        raise SlowLoopPolicyError("catch-up bursts are forbidden")
    if schedule["one_observation_per_tick"] is not True:
        raise SlowLoopPolicyError("each tick must bind exactly one observation")

    budgets = value["budgets"]
    _require_exact_fields(
        budgets,
        {"clock", "stage_deadline_offsets_ns", "minimum_final_reserve_ns"},
        "budgets",
    )
    if budgets["clock"] != "time.monotonic_ns":
        raise SlowLoopPolicyError("budget clock must be time.monotonic_ns")
    offsets = budgets["stage_deadline_offsets_ns"]
    if tuple(offsets) != STAGE_ORDER:
        raise SlowLoopPolicyError("stage deadline order changed")
    ordered = [int(offsets[name]) for name in STAGE_ORDER]
    if ordered != sorted(ordered) or ordered[0] <= 0:
        raise SlowLoopPolicyError("stage deadlines must be positive and monotonic")
    if ordered[-1] != period_ns:
        raise SlowLoopPolicyError("hard wall must equal one slow-loop period")
    reserve_ns = int(budgets["minimum_final_reserve_ns"])
    if reserve_ns <= 0 or ordered[-1] - ordered[-2] < reserve_ns:
        raise SlowLoopPolicyError("final reserve is not protected")

    isolation = value["isolation"]
    _require_exact_fields(
        isolation,
        {
            "active_control_path",
            "active_execution_interface",
            "isolated_control_interfaces",
            "cross_path_aggregation_allowed",
        },
        "isolation",
    )
    isolated = frozenset(isolation["isolated_control_interfaces"])
    if isolation["active_control_path"] != ACTIVE_CONTROL_PATH:
        raise SlowLoopPolicyError("active control path changed")
    if isolation["active_execution_interface"] != ACTIVE_EXECUTION_INTERFACE:
        raise SlowLoopPolicyError("active execution interface changed")
    if not ISOLATED_CONTROL_INTERFACES.issubset(isolated):
        raise SlowLoopPolicyError("direct/RPP isolation is incomplete")
    if isolation["cross_path_aggregation_allowed"] is not False:
        raise SlowLoopPolicyError("cross-path aggregation must remain forbidden")

    atomic = value["atomic_apply"]
    _require_exact_fields(
        atomic,
        {
            "logical_parameter_allowlist",
            "parameter_bindings",
            "set_parameters_atomically_required",
            "exact_readback_required",
            "rollback_on_any_mismatch",
            "fault_on_rollback_failure",
            "runtime_ready",
            "runtime_blockers",
        },
        "atomic_apply",
    )
    allowlist = tuple(atomic["logical_parameter_allowlist"])
    if allowlist != LOGICAL_PARAMETERS:
        raise SlowLoopPolicyError("logical parameter allowlist changed")
    if set(atomic["parameter_bindings"]) != set(LOGICAL_PARAMETERS):
        raise SlowLoopPolicyError("parameter binding coverage is incomplete")
    for flag in (
        "set_parameters_atomically_required",
        "exact_readback_required",
        "rollback_on_any_mismatch",
        "fault_on_rollback_failure",
    ):
        if atomic[flag] is not True:
            raise SlowLoopPolicyError(f"atomic apply requirement disabled: {flag}")
    if atomic["runtime_ready"] is not False:
        raise SlowLoopPolicyError("audited runtime mapping is not ready")
    if not atomic["runtime_blockers"]:
        raise SlowLoopPolicyError("runtime blockers must be explicit")

    evidence = value["evidence"]
    _require_exact_fields(
        evidence,
        {
            "schema",
            "schema_version",
            "canonical_json",
            "previous_state_hash_required",
            "readback_required",
            "rollback_evidence_required",
        },
        "evidence",
    )
    if evidence["schema"] != EVIDENCE_SCHEMA or evidence["schema_version"] != EVIDENCE_VERSION:
        raise SlowLoopPolicyError("evidence schema/version changed")
    if evidence["canonical_json"] != "RFC8259-sort-keys-no-NaN":
        raise SlowLoopPolicyError("canonical evidence identity changed")
    if any(
        evidence[name] is not True
        for name in (
            "previous_state_hash_required",
            "readback_required",
            "rollback_evidence_required",
        )
    ):
        raise SlowLoopPolicyError("required evidence was disabled")

    return SlowLoopPolicy(
        policy_sha256=hashlib.sha256(raw).hexdigest(),
        period_ns=period_ns,
        max_observation_age_ns=int(schedule["max_observation_age_ns"]),
        stage_deadline_offsets_ns={name: int(offsets[name]) for name in STAGE_ORDER},
        active_control_path=isolation["active_control_path"],
        active_execution_interface=isolation["active_execution_interface"],
        isolated_control_interfaces=isolated,
        logical_parameters=allowlist,
        runtime_ready=atomic["runtime_ready"],
        runtime_blockers=tuple(atomic["runtime_blockers"]),
        parameter_bindings=copy.deepcopy(atomic["parameter_bindings"]),
    )


def expected_update_id(
    *,
    episode_id: str,
    tick_index: int,
    observation_id: str,
    snapshot: Mapping[str, Any],
) -> str:
    identity_payload = copy.deepcopy(dict(snapshot))
    identity_payload.pop("update_id", None)
    digest = canonical_sha256(
        {
            "episode_id": episode_id,
            "tick_index": tick_index,
            "observation_id": observation_id,
            "snapshot": identity_payload,
        }
    )
    return f"lisn-{tick_index:06d}-{digest[:20]}"


def validate_evidence_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "schema_version",
        "policy_sha256",
        "episode_id",
        "tick_index",
        "scheduled_for_monotonic_ns",
        "decision",
        "reason_codes",
        "control_path",
        "execution_interface",
        "observation",
        "update_id",
        "snapshot_sha256",
        "previous_state_sha256",
        "requested_parameters",
        "before_parameters",
        "readback_parameters",
        "rollback_readback_parameters",
        "accepted_state_sha256",
        "timeline",
        "stage_deadlines_ns",
        "runtime_executed",
    }
    _require_exact_fields(record, required, "evidence")
    if record["schema"] != EVIDENCE_SCHEMA or record["schema_version"] != EVIDENCE_VERSION:
        raise ValueError("unsupported evidence schema/version")
    if record["runtime_executed"] is not False:
        raise ValueError("offline evidence cannot claim runtime execution")
    canonical_json_bytes(record)


class DeterministicSlowLoop:
    """Pure scheduler and atomic transaction consumer for offline fixtures."""

    def __init__(
        self,
        *,
        policy: SlowLoopPolicy,
        episode_id: str,
        epoch_monotonic_ns: int,
        target_frame: str,
    ) -> None:
        if not episode_id.strip() or not target_frame.strip():
            raise ValueError("episode_id and target_frame must be non-empty")
        self.policy = policy
        self.target_frame = target_frame
        self.state = SlowLoopState(
            episode_id=episode_id,
            epoch_monotonic_ns=int(epoch_monotonic_ns),
        )

    def scheduled_for_ns(self, tick_index: int | None = None) -> int:
        index = self.state.next_tick_index if tick_index is None else tick_index
        return self.state.epoch_monotonic_ns + (index + 1) * self.policy.period_ns

    def _base_evidence(
        self,
        *,
        tick_index: int,
        scheduled_for_ns: int,
        decision: str,
        reason_codes: list[str],
        observation: ObservationIdentity | None = None,
        snapshot: Mapping[str, Any] | None = None,
        timeline: SlowLoopTimeline | None = None,
    ) -> dict[str, Any]:
        snapshot_hash = canonical_sha256(snapshot) if snapshot is not None else None
        record = {
            "schema": EVIDENCE_SCHEMA,
            "schema_version": EVIDENCE_VERSION,
            "policy_sha256": self.policy.policy_sha256,
            "episode_id": self.state.episode_id,
            "tick_index": tick_index,
            "scheduled_for_monotonic_ns": scheduled_for_ns,
            "decision": decision,
            "reason_codes": list(reason_codes),
            "control_path": snapshot.get("control_path") if snapshot else None,
            "execution_interface": self.policy.active_execution_interface,
            "observation": (
                {
                    "episode_id": observation.episode_id,
                    "id": observation.observation_id,
                    "frame_id": observation.frame_id,
                    "ros_stamp_ns": observation.ros_stamp_ns,
                    "received_monotonic_ns": observation.received_monotonic_ns,
                    "content_sha256": observation.content_sha256,
                }
                if observation
                else None
            ),
            "update_id": snapshot.get("update_id") if snapshot else None,
            "snapshot_sha256": snapshot_hash,
            "previous_state_sha256": self.state.accepted_snapshot_sha256,
            "requested_parameters": (
                copy.deepcopy(snapshot.get("controller_parameters"))
                if snapshot
                else None
            ),
            "before_parameters": None,
            "readback_parameters": None,
            "rollback_readback_parameters": None,
            "accepted_state_sha256": self.state.accepted_snapshot_sha256,
            "timeline": timeline.as_dict() if timeline else None,
            "stage_deadlines_ns": {
                stage: self.policy.deadline_ns(scheduled_for_ns, stage)
                for stage in STAGE_ORDER
            },
            "runtime_executed": False,
        }
        return record

    def _finish(self, record: dict[str, Any], *, advance: bool = True) -> dict[str, Any]:
        if advance:
            self.state.next_tick_index += 1
        record["accepted_state_sha256"] = self.state.accepted_snapshot_sha256
        validate_evidence_record(record)
        return record

    def _reject(
        self,
        record: dict[str, Any],
        *reason_codes: str,
        decision: str = "rejected",
        advance: bool = True,
    ) -> dict[str, Any]:
        record["decision"] = decision
        record["reason_codes"] = list(reason_codes)
        return self._finish(record, advance=advance)

    def process(
        self,
        *,
        snapshot: Mapping[str, Any],
        observation: ObservationIdentity,
        now_monotonic_ns: int,
        now_ros_ns: int,
        timeline: SlowLoopTimeline,
        adapter: ParameterAdapter,
    ) -> dict[str, Any]:
        """Process one proposed tick without reading any ambient clock."""

        tick_index = self.state.next_tick_index
        scheduled = self.scheduled_for_ns(tick_index)
        record = self._base_evidence(
            tick_index=tick_index,
            scheduled_for_ns=scheduled,
            decision="pending",
            reason_codes=[],
            observation=observation,
            snapshot=snapshot,
            timeline=timeline,
        )
        if self.state.faulted:
            return self._reject(record, "STATE_FAULTED", advance=False)
        if now_monotonic_ns < scheduled:
            return self._reject(record, "NOT_DUE", decision="not_due", advance=False)
        if now_monotonic_ns >= scheduled + self.policy.period_ns:
            missed = (now_monotonic_ns - scheduled) // self.policy.period_ns
            self.state.next_tick_index += int(missed)
            record["decision"] = "missed_tick"
            record["reason_codes"] = ["MISSED_TICK_NO_CATCH_UP"]
            record["accepted_state_sha256"] = self.state.accepted_snapshot_sha256
            validate_evidence_record(record)
            return record

        ordered_completions = (
            timeline.proposal_completed_ns,
            timeline.validation_completed_ns,
            timeline.apply_completed_ns,
            timeline.readback_completed_ns,
            timeline.evidence_completed_ns,
        )
        if ordered_completions != tuple(sorted(ordered_completions)):
            return self._reject(record, "TIMELINE_NON_MONOTONIC")
        if timeline.evidence_completed_ns > self.policy.deadline_ns(
            scheduled, "evidence"
        ):
            return self._reject(record, "EVIDENCE_BUDGET_EXCEEDED")

        snapshot_copy = copy.deepcopy(dict(snapshot))
        snapshot_hash = canonical_sha256(snapshot_copy)
        update_id = snapshot_copy.get("update_id")
        if isinstance(update_id, str) and update_id in self.state.update_ledger:
            previous_hash = self.state.update_ledger[update_id]
            if previous_hash == snapshot_hash:
                return self._reject(
                    record, "DUPLICATE_UPDATE_NOOP", decision="duplicate_noop"
                )
            return self._reject(record, "UPDATE_ID_REPLAY_CONFLICT")

        if observation.episode_id != self.state.episode_id:
            return self._reject(record, "OBSERVATION_EPISODE_MISMATCH")
        if observation.received_monotonic_ns > now_monotonic_ns:
            return self._reject(record, "OBSERVATION_FROM_FUTURE")
        if (
            self.policy.enforce_observation_age
            and now_monotonic_ns - observation.received_monotonic_ns
            > self.policy.max_observation_age_ns
        ):
            return self._reject(record, "OBSERVATION_STALE")
        if observation.observation_id in self.state.used_observation_ids:
            return self._reject(record, "OBSERVATION_REUSED")
        self.state.used_observation_ids.add(observation.observation_id)

        if timeline.proposal_completed_ns > self.policy.deadline_ns(scheduled, "proposal"):
            return self._reject(record, "PROPOSAL_BUDGET_EXCEEDED")
        if snapshot_copy.get("control_path") != self.policy.active_control_path:
            return self._reject(record, "CONTROL_PATH_ISOLATION_VIOLATION")
        if self.policy.active_execution_interface in self.policy.isolated_control_interfaces:
            return self._reject(record, "EXECUTION_INTERFACE_ISOLATION_VIOLATION")

        source = snapshot_copy.get("source_observation")
        if not isinstance(source, Mapping) or (
            source.get("id") != observation.observation_id
            or source.get("frame_id") != observation.frame_id
            or source.get("stamp_ns") != observation.ros_stamp_ns
        ):
            return self._reject(record, "SOURCE_OBSERVATION_IDENTITY_MISMATCH")
        expected_id = expected_update_id(
            episode_id=self.state.episode_id,
            tick_index=tick_index,
            observation_id=observation.observation_id,
            snapshot=snapshot_copy,
        )
        if update_id != expected_id:
            return self._reject(record, "UPDATE_ID_MISMATCH")
        try:
            validate_modulation_snapshot(
                snapshot_copy,
                now_ns=now_ros_ns,
                expected_target_frame=self.target_frame,
                max_observation_age_ns=self.policy.max_observation_age_ns,
            )
        except ContractValidationError as exc:
            record["reason_codes"] = ["SNAPSHOT_INVALID", *exc.errors]
            record["decision"] = "rejected"
            return self._finish(record)
        if timeline.validation_completed_ns > self.policy.deadline_ns(
            scheduled, "validation"
        ):
            return self._reject(record, "VALIDATION_BUDGET_EXCEEDED")

        requested = copy.deepcopy(snapshot_copy["controller_parameters"])
        before: dict[str, float] | None = None
        applied = False
        rollback_needed = False
        reason = ""
        try:
            before = dict(adapter.read(self.policy.logical_parameters))
            record["before_parameters"] = copy.deepcopy(before)
            if set(before) != set(self.policy.logical_parameters):
                return self._reject(record, "BEFORE_READBACK_FIELD_MISMATCH")
            if timeline.apply_completed_ns > self.policy.deadline_ns(scheduled, "apply"):
                return self._reject(record, "APPLY_BUDGET_EXCEEDED")
            apply_result = adapter.set_atomically(requested)
            applied = apply_result.successful
            reason = apply_result.reason
            readback = dict(adapter.read(self.policy.logical_parameters))
            record["readback_parameters"] = copy.deepcopy(readback)
            if timeline.readback_completed_ns > self.policy.deadline_ns(
                scheduled, "readback"
            ):
                rollback_needed = readback != before
                reason = "READBACK_BUDGET_EXCEEDED"
            elif not apply_result.successful:
                rollback_needed = readback != before
                reason = "APPLY_REJECTED"
            elif readback != requested:
                rollback_needed = True
                reason = "READBACK_MISMATCH"
            else:
                self.state.accepted_snapshot = snapshot_copy
                self.state.accepted_snapshot_sha256 = snapshot_hash
                self.state.accepted_update_id = str(update_id)
                self.state.update_ledger[str(update_id)] = snapshot_hash
                record["decision"] = "accepted"
                record["reason_codes"] = ["ACCEPTED_ATOMIC_READBACK_MATCH"]
                return self._finish(record)
        except BaseException as exc:
            rollback_needed = before is not None
            reason = f"ADAPTER_EXCEPTION:{type(exc).__name__}:{exc}"

        if not rollback_needed:
            suffix = f":{reason}" if reason else ""
            return self._reject(record, f"APPLY_REJECTED_NO_STATE_CHANGE{suffix}")

        rollback_complete = timeline.rollback_completed_ns
        rollback_deadline = self.policy.deadline_ns(scheduled, "rollback")
        if (
            before is None
            or rollback_complete is None
            or rollback_complete < timeline.readback_completed_ns
            or rollback_complete > rollback_deadline
            or timeline.evidence_completed_ns < rollback_complete
        ):
            self.state.faulted = True
            return self._reject(
                record,
                reason or "TRANSACTION_FAILED",
                "ROLLBACK_BUDGET_OR_TIMELINE_INVALID",
                decision="faulted",
            )
        try:
            rollback_result = adapter.set_atomically(before)
            rollback_readback = dict(adapter.read(self.policy.logical_parameters))
            record["rollback_readback_parameters"] = copy.deepcopy(
                rollback_readback
            )
            if not rollback_result.successful or rollback_readback != before:
                self.state.faulted = True
                return self._reject(
                    record,
                    reason or "TRANSACTION_FAILED",
                    "ROLLBACK_FAILED",
                    decision="faulted",
                )
        except BaseException as exc:
            self.state.faulted = True
            return self._reject(
                record,
                reason or "TRANSACTION_FAILED",
                f"ROLLBACK_EXCEPTION:{type(exc).__name__}:{exc}",
                decision="faulted",
            )
        return self._reject(
            record,
            reason or ("APPLY_REJECTED" if not applied else "TRANSACTION_FAILED"),
            "ROLLBACK_SUCCEEDED_PREVIOUS_STATE_RETAINED",
            decision="rolled_back",
        )
