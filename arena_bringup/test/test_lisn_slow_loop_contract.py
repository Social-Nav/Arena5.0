from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from arena_bringup.lisn_slow_loop_contract import (
    ACTIVE_CONTROL_PATH,
    ACTIVE_EXECUTION_INTERFACE,
    EVIDENCE_SCHEMA,
    ISOLATED_CONTROL_INTERFACES,
    ApplyResult,
    DeterministicSlowLoop,
    ObservationIdentity,
    SlowLoopPolicyError,
    SlowLoopTimeline,
    canonical_json_bytes,
    canonical_sha256,
    expected_update_id,
    load_slow_loop_policy,
    validate_evidence_record,
)


PACKAGE = Path(__file__).parents[1]
POLICY_PATH = PACKAGE / "configs/lisn/slow_loop_contract_v1.json"
EVIDENCE_SCHEMA_PATH = PACKAGE / "configs/lisn/slow_loop_evidence.schema.json"
MODULATION_FIXTURE = PACKAGE / "test/fixtures/modulation_contract/valid.json"
MATRIX_PATH = PACKAGE / "test/fixtures/lisn_slow_loop/fixture_matrix.json"
EPOCH_NS = 100_000_000_000
ROS_NOW_NS = 1_500_000_000


class FakeAdapter:
    def __init__(self, initial, mode="success"):
        self.state = copy.deepcopy(initial)
        self.mode = mode
        self.mode_start_call = 0
        self.set_calls = []
        self.read_calls = 0

    def read(self, names):
        self.read_calls += 1
        return {name: self.state[name] for name in names if name in self.state}

    def set_atomically(self, values):
        self.set_calls.append(copy.deepcopy(dict(values)))
        call = len(self.set_calls)
        mode_call = call - self.mode_start_call
        if self.mode == "reject" and mode_call == 1:
            return ApplyResult(False, "fixture reject")
        if self.mode == "reject_partial" and mode_call == 1:
            first = next(iter(values))
            self.state[first] = values[first]
            return ApplyResult(False, "fixture partial reject")
        if self.mode in {"mismatch", "rollback_fail"} and mode_call == 1:
            self.state.update(values)
            first = next(iter(values))
            self.state[first] = float(values[first]) + 0.01
            return ApplyResult(True, "fixture mismatched readback")
        if self.mode == "rollback_fail" and mode_call >= 2:
            return ApplyResult(False, "fixture rollback rejected")
        if self.mode == "exception_partial" and mode_call == 1:
            first = next(iter(values))
            self.state[first] = values[first]
            raise RuntimeError("fixture apply exception")
        self.state = copy.deepcopy(dict(values))
        return ApplyResult(True, "fixture success")

    def set_mode(self, mode):
        self.mode = mode
        self.mode_start_call = len(self.set_calls)


def load_policy():
    return load_slow_loop_policy(POLICY_PATH)


def make_loop():
    return DeterministicSlowLoop(
        policy=load_policy(),
        episode_id="episode-fixture",
        epoch_monotonic_ns=EPOCH_NS,
        target_frame="map",
    )


def make_observation(loop, *, name="obs-001", stale=False):
    due = loop.scheduled_for_ns()
    return ObservationIdentity(
        episode_id=loop.state.episode_id,
        observation_id=name,
        frame_id="Ai2_Bot2/head_camera",
        ros_stamp_ns=1_400_000_000,
        received_monotonic_ns=due - (6_000_000_000 if stale else 1_000_000_000),
        content_sha256="a" * 64,
    )


def make_snapshot(loop, observation, *, mutation="none", tick_index=None):
    value = json.loads(MODULATION_FIXTURE.read_text(encoding="utf-8"))
    value["generated_at_ns"] = 1_450_000_000
    value["valid_until_ns"] = 2_500_000_000
    value["source_observation"] = {
        "id": observation.observation_id,
        "frame_id": observation.frame_id,
        "stamp_ns": observation.ros_stamp_ns,
    }
    if mutation == "direct_path":
        value["control_path"] = "vln_direct_cmd_vel"
    elif mutation == "partial_snapshot":
        value["controller_parameters"].pop("people_weight")
    elif mutation == "source_mismatch":
        value["source_observation"]["id"] = "different-observation"
    value["update_id"] = expected_update_id(
        episode_id=loop.state.episode_id,
        tick_index=loop.state.next_tick_index if tick_index is None else tick_index,
        observation_id=observation.observation_id,
        snapshot=value,
    )
    if mutation == "update_id_mismatch":
        value["update_id"] = "not-the-deterministic-id"
    return value


def make_timeline(loop, kind="normal"):
    due = loop.scheduled_for_ns()
    values = {
        "proposal_completed_ns": due + 1_000_000_000,
        "validation_completed_ns": due + 1_100_000_000,
        "apply_completed_ns": due + 1_200_000_000,
        "readback_completed_ns": due + 1_300_000_000,
        "rollback_completed_ns": None,
        "evidence_completed_ns": due + 1_700_000_000,
    }
    if kind == "proposal_late":
        values["proposal_completed_ns"] = due + 7_100_000_000
        values["validation_completed_ns"] = due + 7_150_000_000
        values["apply_completed_ns"] = due + 7_200_000_000
        values["readback_completed_ns"] = due + 7_300_000_000
        values["evidence_completed_ns"] = due + 7_500_000_000
    elif kind == "evidence_late":
        values["evidence_completed_ns"] = due + 9_600_000_000
    elif kind == "nonmonotonic":
        values["validation_completed_ns"] = due + 900_000_000
    elif kind == "rollback":
        values["rollback_completed_ns"] = due + 1_500_000_000
    elif kind == "readback_late":
        values["readback_completed_ns"] = due + 8_300_000_000
        values["rollback_completed_ns"] = due + 8_600_000_000
        values["evidence_completed_ns"] = due + 9_000_000_000
    elif kind == "rollback_late":
        values["rollback_completed_ns"] = due + 9_300_000_000
        values["evidence_completed_ns"] = due + 9_400_000_000
    return SlowLoopTimeline(**values)


def execute_fixture(spec):
    loop = make_loop()
    observation = make_observation(
        loop,
        name=f"obs-{spec['name']}",
        stale=spec["mutation"] == "stale_observation",
    )
    snapshot = make_snapshot(loop, observation, mutation=spec["mutation"])
    timeline_kind = spec["timeline"]
    timeline = make_timeline(
        loop,
        "normal" if timeline_kind in {"not_due", "missed"} else timeline_kind,
    )
    due = loop.scheduled_for_ns()
    if timeline_kind == "not_due":
        now = due - 1
    elif timeline_kind == "missed":
        now = due + loop.policy.period_ns
    else:
        now = due
    adapter = FakeAdapter(
        {
            "max_linear_velocity_mps": 0.8,
            "max_angular_velocity_rps": 1.0,
            "goal_weight": 2.0,
            "obstacle_weight": 5.0,
            "people_weight": 4.0,
        },
        spec["adapter"],
    )
    before = copy.deepcopy(adapter.state)
    record = loop.process(
        snapshot=snapshot,
        observation=observation,
        now_monotonic_ns=now,
        now_ros_ns=ROS_NOW_NS,
        timeline=timeline,
        adapter=adapter,
    )
    return loop, adapter, before, record


def test_policy_is_default_disabled_bounded_and_path_isolated():
    policy = load_policy()
    assert policy.period_ns == 10_000_000_000
    assert policy.active_control_path == ACTIVE_CONTROL_PATH
    assert policy.active_execution_interface == ACTIVE_EXECUTION_INTERFACE
    assert ISOLATED_CONTROL_INTERFACES <= policy.isolated_control_interfaces
    assert policy.runtime_ready is False
    assert len(policy.runtime_blockers) >= 5
    assert policy.deadline_ns(0, "hard_wall") == policy.period_ns
    assert policy.deadline_ns(0, "evidence") == 9_500_000_000


def test_actual_parameter_binding_audit_remains_fail_closed():
    bindings = load_policy().parameter_bindings
    assert bindings["max_linear_velocity_mps"]["status"].startswith("exact_dynamic")
    assert bindings["max_angular_velocity_rps"]["status"] == "declared_but_runtime_callback_ignores"
    assert bindings["obstacle_weight"]["status"] == "explicitly_excluded_from_runtime_callback"
    assert bindings["goal_weight"]["status"] == "ambiguous_no_one_to_one_mapping"
    assert bindings["people_weight"]["status"] == "ambiguous_no_one_to_one_mapping"


def test_evidence_json_schema_and_python_validator_have_same_top_level_fields():
    schema = json.loads(EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema"]["const"] == EVIDENCE_SCHEMA
    assert schema["properties"]["runtime_executed"]["const"] is False
    assert set(schema["required"]) == set(schema["properties"])


@pytest.mark.parametrize(
    "spec",
    json.loads(MATRIX_PATH.read_text(encoding="utf-8"))["fixtures"],
    ids=lambda value: value["name"],
)
def test_failure_fixture_matrix_is_deterministic_and_fail_closed(spec):
    loop, adapter, before, record = execute_fixture(spec)
    assert record["decision"] == spec["expected_decision"]
    assert any(spec["expected_reason"] in reason for reason in record["reason_codes"])
    validate_evidence_record(record)
    assert record["runtime_executed"] is False
    if record["decision"] == "accepted":
        assert loop.state.accepted_snapshot_sha256 == record["snapshot_sha256"]
        assert adapter.state == record["requested_parameters"]
    elif record["decision"] == "rolled_back":
        assert adapter.state == before
        assert loop.state.accepted_snapshot is None
    elif record["decision"] == "faulted":
        assert loop.state.faulted is True
    elif record["decision"] == "missed_tick":
        assert loop.state.next_tick_index == 1
        assert loop.scheduled_for_ns() == EPOCH_NS + 2 * loop.policy.period_ns
    else:
        assert loop.state.accepted_snapshot is None
    if spec["expected_reason"] in {
        "NOT_DUE",
        "MISSED_TICK_NO_CATCH_UP",
        "OBSERVATION_STALE",
        "CONTROL_PATH_ISOLATION_VIOLATION",
        "SNAPSHOT_INVALID",
        "SOURCE_OBSERVATION_IDENTITY_MISMATCH",
        "UPDATE_ID_MISMATCH",
        "PROPOSAL_BUDGET_EXCEEDED",
        "EVIDENCE_BUDGET_EXCEEDED",
        "TIMELINE_NON_MONOTONIC",
    }:
        assert adapter.set_calls == []


def test_atomic_rollback_retains_previous_accepted_snapshot():
    loop = make_loop()
    first_observation = make_observation(loop, name="obs-first")
    first_snapshot = make_snapshot(loop, first_observation)
    adapter = FakeAdapter(first_snapshot["controller_parameters"])
    accepted = loop.process(
        snapshot=first_snapshot,
        observation=first_observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=make_timeline(loop),
        adapter=adapter,
    )
    accepted_hash = accepted["accepted_state_sha256"]
    before_second = copy.deepcopy(adapter.state)

    second_observation = make_observation(loop, name="obs-second")
    second_snapshot = make_snapshot(loop, second_observation)
    second_snapshot["controller_parameters"]["people_weight"] = 9.0
    second_snapshot["update_id"] = expected_update_id(
        episode_id=loop.state.episode_id,
        tick_index=loop.state.next_tick_index,
        observation_id=second_observation.observation_id,
        snapshot=second_snapshot,
    )
    adapter.set_mode("mismatch")
    rolled_back = loop.process(
        snapshot=second_snapshot,
        observation=second_observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=make_timeline(loop, "rollback"),
        adapter=adapter,
    )
    assert rolled_back["decision"] == "rolled_back"
    assert rolled_back["accepted_state_sha256"] == accepted_hash
    assert loop.state.accepted_snapshot_sha256 == accepted_hash
    assert adapter.state == before_second


def test_duplicate_is_noop_and_conflicting_replay_is_rejected():
    loop = make_loop()
    observation = make_observation(loop)
    snapshot = make_snapshot(loop, observation)
    adapter = FakeAdapter(snapshot["controller_parameters"])
    accepted = loop.process(
        snapshot=snapshot,
        observation=observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=make_timeline(loop),
        adapter=adapter,
    )
    accepted_state = accepted["accepted_state_sha256"]
    second_observation = make_observation(loop, name="obs-retry")
    duplicate = loop.process(
        snapshot=snapshot,
        observation=second_observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=make_timeline(loop),
        adapter=adapter,
    )
    assert duplicate["decision"] == "duplicate_noop"
    assert len(adapter.set_calls) == 1
    assert duplicate["accepted_state_sha256"] == accepted_state

    conflict = copy.deepcopy(snapshot)
    conflict["reason"] = "same id, different body"
    third_observation = make_observation(loop, name="obs-conflict")
    replay = loop.process(
        snapshot=conflict,
        observation=third_observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=make_timeline(loop),
        adapter=adapter,
    )
    assert replay["decision"] == "rejected"
    assert replay["reason_codes"] == ["UPDATE_ID_REPLAY_CONFLICT"]
    assert len(adapter.set_calls) == 1


def test_faulted_state_rejects_future_updates_without_adapter_calls():
    spec = next(
        item
        for item in json.loads(MATRIX_PATH.read_text(encoding="utf-8"))["fixtures"]
        if item["name"] == "rollback_failure_fault"
    )
    loop, adapter, _before, fault = execute_fixture(spec)
    assert fault["decision"] == "faulted"
    calls = len(adapter.set_calls)
    observation = make_observation(loop, name="obs-after-fault")
    snapshot = make_snapshot(loop, observation)
    rejected = loop.process(
        snapshot=snapshot,
        observation=observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=make_timeline(loop),
        adapter=adapter,
    )
    assert rejected["reason_codes"] == ["STATE_FAULTED"]
    assert len(adapter.set_calls) == calls


def test_same_inputs_generate_byte_identical_evidence_and_do_not_mutate_snapshot():
    records = []
    snapshots = []
    for _ in range(2):
        loop = make_loop()
        observation = make_observation(loop)
        snapshot = make_snapshot(loop, observation)
        original = copy.deepcopy(snapshot)
        adapter = FakeAdapter(snapshot["controller_parameters"])
        record = loop.process(
            snapshot=snapshot,
            observation=observation,
            now_monotonic_ns=loop.scheduled_for_ns(),
            now_ros_ns=ROS_NOW_NS,
            timeline=make_timeline(loop),
            adapter=adapter,
        )
        assert snapshot == original
        records.append(canonical_json_bytes(record))
        snapshots.append(canonical_sha256(snapshot))
    assert records[0] == records[1]
    assert snapshots[0] == snapshots[1]


def test_canonical_identity_rejects_non_json_nan():
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})


def test_policy_loader_rejects_runtime_authorization_or_path_drift(tmp_path):
    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    value["runtime_authorized"] = True
    path = tmp_path / "authorized.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SlowLoopPolicyError, match="unauthorized"):
        load_slow_loop_policy(path)

    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    value["isolation"]["isolated_control_interfaces"].remove("nav2_follow_path_rpp")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SlowLoopPolicyError, match="isolation"):
        load_slow_loop_policy(path)
