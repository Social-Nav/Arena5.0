"""Focused tests for the grounding-live profile: no hard 5s age gate, identity kept."""
import copy
import json
from pathlib import Path

from arena_bringup.lisn_slow_loop_contract import (
    ApplyResult,
    DeterministicSlowLoop,
    ObservationIdentity,
    SlowLoopTimeline,
    expected_update_id,
)
from arena_bringup.lisn_slow_loop_ros import load_policy

PACKAGE = Path(__file__).parents[1]
PROFILE = PACKAGE / "configs/lisn/slow_loop_grounding_live_v1.json"
FROZEN = PACKAGE / "configs/lisn/slow_loop_contract_v1.json"
MODULATION = PACKAGE / "test/fixtures/modulation_contract/valid.json"
EPOCH_NS = 100_000_000_000
ROS_NOW_NS = 1_500_000_000


class FakeAdapter:
    def __init__(self):
        self.state = {
            "max_linear_velocity_mps": 0.8,
            "max_angular_velocity_rps": 1.0,
            "goal_weight": 2.0,
            "obstacle_weight": 5.0,
            "people_weight": 4.0,
        }
        self.set_calls = []

    def read(self, names):
        return {name: self.state[name] for name in names if name in self.state}

    def set_atomically(self, values):
        self.set_calls.append(dict(values))
        self.state.update(values)
        return ApplyResult(True, "fixture success")


def _loop(path):
    return DeterministicSlowLoop(
        policy=load_policy(path, engineering=True),
        episode_id="grounding-live-fixture",
        epoch_monotonic_ns=EPOCH_NS,
        target_frame="map",
    )


def _observation(loop, *, age_ns, name="obs-1"):
    due = loop.scheduled_for_ns()
    return ObservationIdentity(
        episode_id=loop.state.episode_id,
        observation_id=name,
        frame_id="Ai2_Bot2/head_camera",
        ros_stamp_ns=1_400_000_000,
        received_monotonic_ns=due - age_ns,
        content_sha256="a" * 64,
    )


def _snapshot(loop, observation, *, mismatch=False):
    value = json.loads(MODULATION.read_text())
    value["generated_at_ns"] = 1_450_000_000
    value["valid_until_ns"] = 2_500_000_000
    value["source_observation"] = {
        "id": "different-observation" if mismatch else observation.observation_id,
        "frame_id": observation.frame_id,
        "stamp_ns": observation.ros_stamp_ns,
    }
    value["update_id"] = expected_update_id(
        episode_id=loop.state.episode_id,
        tick_index=loop.state.next_tick_index,
        observation_id=observation.observation_id,
        snapshot=value,
    )
    return value


def _timeline(loop):
    due = loop.scheduled_for_ns()
    return SlowLoopTimeline(
        proposal_completed_ns=due + 6_200_000_000,
        validation_completed_ns=due + 6_300_000_000,
        apply_completed_ns=due + 6_400_000_000,
        readback_completed_ns=due + 6_500_000_000,
        evidence_completed_ns=due + 6_700_000_000,
    )


def test_grounding_live_profile_disables_only_hard_age_gate():
    policy = load_policy(PROFILE, engineering=True)
    assert policy.period_ns == 10_000_000_000
    assert policy.enforce_observation_age is False
    assert policy.max_observation_age_ns == 30_000_000_000
    assert policy.runtime_ready is True
    assert policy.active_control_path == "lisn_style_social_mpc"


def test_grounding_live_accepts_six_second_old_observation():
    loop = _loop(PROFILE)
    observation = _observation(loop, age_ns=6_200_000_000)
    snapshot = _snapshot(loop, observation)
    record = loop.process(
        snapshot=snapshot,
        observation=observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=_timeline(loop),
        adapter=FakeAdapter(),
    )
    assert record["decision"] == "accepted", record
    assert "OBSERVATION_STALE" not in record.get("reason_codes", [])


def test_grounding_live_still_rejects_source_identity_mismatch():
    loop = _loop(PROFILE)
    observation = _observation(loop, age_ns=6_200_000_000)
    snapshot = _snapshot(loop, observation, mismatch=True)
    record = loop.process(
        snapshot=snapshot,
        observation=observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=_timeline(loop),
        adapter=FakeAdapter(),
    )
    assert record["decision"] == "rejected", record
    assert "SOURCE_OBSERVATION_IDENTITY_MISMATCH" in record["reason_codes"]


def test_grounding_live_still_rejects_future_observation():
    loop = _loop(PROFILE)
    observation = _observation(loop, age_ns=-1_000_000_000)
    snapshot = _snapshot(loop, observation)
    record = loop.process(
        snapshot=snapshot,
        observation=observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=_timeline(loop),
        adapter=FakeAdapter(),
    )
    assert record["decision"] == "rejected", record
    assert "OBSERVATION_FROM_FUTURE" in record["reason_codes"]


def test_frozen_profile_still_rejects_six_second_old_observation():
    loop = _loop(FROZEN)
    observation = _observation(loop, age_ns=6_200_000_000)
    snapshot = _snapshot(loop, observation)
    record = loop.process(
        snapshot=snapshot,
        observation=observation,
        now_monotonic_ns=loop.scheduled_for_ns(),
        now_ros_ns=ROS_NOW_NS,
        timeline=_timeline(loop),
        adapter=FakeAdapter(),
    )
    assert record["decision"] == "rejected", record
    assert "OBSERVATION_STALE" in record["reason_codes"]
