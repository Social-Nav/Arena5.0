"""CPU-only tests for the Task34 three-segment cmd_vel trace dictionary."""
import json
import math

from arena_vln_models.cmd_vel_trace import TRACE_SCHEMA, trace_record, write_trace


def _record(**overrides):
    values = dict(
        wall_time=1000.0, monotonic_ns=123, episode=0,
        raw_linear_x=0.8, raw_angular_z=0.1,
        delivered_linear_x=0.7, delivered_angular_z=0.09,
        odom_linear_x=0.68, odom_angular_z=0.08,
        odom_pose_xy=(0.0, 0.0), goal_xy=(2.0, 0.0), agents=[],
    )
    values.update(overrides)
    return trace_record(**values)


def test_three_segments_and_deltas():
    rec = _record()
    assert rec['schema'] == TRACE_SCHEMA
    assert rec['raw']['linear_x'] == 0.8
    assert rec['delivered']['linear_x'] == 0.7
    assert rec['odom']['linear_x'] == 0.68
    assert rec['smoother_delta_linear_x'] == round(0.7 - 0.8, 6)
    assert rec['plant_tracking_delta_linear_x'] == round(0.68 - 0.7, 6)
    assert rec['condition']['safe_slowdown'] is False
    assert rec['condition']['stop'] is False


def test_human_person_speed_and_min_distance_only_person():
    agents = [
        {'type': 1, 'position': {'position': {'x': 1.0, 'y': 0.0}},
         'velocity': {'linear': {'x': 0.5, 'y': 0.0}}},
        {'type': 1, 'position': {'position': {'x': 2.0, 'y': 0.0}},
         'velocity': {'linear': {'x': 0.2, 'y': 0.1}}},
        {'type': 2, 'position': {'position': {'x': 9.0, 'y': 9.0}},
         'velocity': {'linear': {'x': 9.0, 'y': 9.0}}},  # ROBOT ignored
    ]
    rec = _record(agents=agents)
    assert rec['human']['count'] == 2
    assert rec['human']['max_speed_mps'] == 0.5
    assert rec['human']['min_distance_m'] == 1.0


def test_goal_turn_condition_and_distance():
    rec = _record(goal_xy=(0.0, 2.0))
    assert math.isclose(rec['goal']['distance_m'], 2.0)
    assert rec['goal']['bearing_rad'] == round(math.pi / 2, 6)
    assert rec['goal']['turn_condition'] is True
    straight = _record(goal_xy=(2.0, 0.0))
    assert straight['goal']['turn_condition'] is False


def test_safe_slowdown_and_stop_flags():
    slow = _record(delivered_linear_x=0.2)
    assert slow['condition']['safe_slowdown'] is True
    assert slow['condition']['stop'] is False
    stopped = _record(delivered_linear_x=0.0)
    assert stopped['condition']['safe_slowdown'] is True
    assert stopped['condition']['stop'] is True


def test_missing_pose_or_goal_is_null_not_guessed():
    rec = _record(odom_pose_xy=None)
    assert rec['goal']['distance_m'] is None
    assert rec['goal']['bearing_rad'] is None
    assert rec['goal']['turn_condition'] is None


def test_write_trace_appends_jsonl(tmp_path):
    path = tmp_path / 'cmd_vel_trace.jsonl'
    write_trace(path, _record())
    write_trace(path, _record(delivered_linear_x=0.2))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]['condition']['safe_slowdown'] is True
