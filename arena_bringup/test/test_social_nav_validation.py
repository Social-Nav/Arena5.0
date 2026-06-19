import json

from arena_bringup.social_nav_validation import _check_dynamic_scene, _check_model_control, _trace_events


def test_trace_events_accepts_official_client_event_field(tmp_path):
    trace_path = tmp_path / 'internnav_trace.jsonl'
    trace_path.write_text(
        '\n'.join(
            [
                json.dumps({"event": "planning_response_received"}),
                json.dumps({"event_type": "model_result"}),
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    total, counts = _trace_events(trace_path)

    assert total == 2
    assert counts["planning_response_received"] == 1
    assert counts["model_result"] == 1


def test_direct_client_model_control_accepts_trace_without_status(tmp_path):
    trace_path = tmp_path / 'internnav_trace.jsonl'
    trace_path.write_text(json.dumps({"event": "trajectory"}) + '\n', encoding='utf-8')
    manifest = {
        "parameters": {
            "internnav_direct_cmd_vel": True,
            "internnav_external_server": True,
        },
        "artifacts": {
            "internnav_trace_path": str(trace_path),
            "dual_vln_status_path": str(tmp_path / 'internnav_status.json'),
        },
    }

    result = _check_model_control(tmp_path, manifest)

    assert result["pass"] is True
    assert result["status_present"] is False
    assert result["status_required"] is False
    assert result["direct_control_event_count"] == 1


def test_wrapper_model_control_still_requires_status(tmp_path):
    trace_path = tmp_path / 'internnav_trace.jsonl'
    trace_path.write_text(json.dumps({"event_type": "model_result"}) + '\n', encoding='utf-8')
    manifest = {
        "parameters": {
            "internnav_direct_cmd_vel": False,
        },
        "artifacts": {
            "internnav_trace_path": str(trace_path),
            "dual_vln_status_path": str(tmp_path / 'internnav_status.json'),
        },
    }

    result = _check_model_control(tmp_path, manifest)

    assert result["pass"] is False
    assert result["status_present"] is False
    assert result["status_required"] is True


def test_dynamic_scene_check_requires_motion_overlap_and_interaction():
    result = _check_dynamic_scene(
        {
            "moving_human_count": 1,
            "human_motion_total_m": 2.0,
            "human_motion_time_sec": 8.0,
            "robot_motion_time_sec": 6.0,
            "human_robot_motion_overlap_time_sec": 5.0,
            "human_robot_interaction_time_sec": 2.0,
            "dynamic_scene_success": True,
            "thresholds": {
                "min_moving_human_count": 1,
                "min_human_motion_time_sec": 5.0,
                "min_human_robot_motion_overlap_time_sec": 3.0,
                "min_human_robot_interaction_time_sec": 1.0,
            },
        }
    )

    assert result["pass"] is True
    assert result["required_fields_present"] is True
    assert result["failures"] == []


def test_dynamic_scene_check_fails_static_humans():
    result = _check_dynamic_scene(
        {
            "moving_human_count": 0,
            "human_motion_total_m": 0.0,
            "human_motion_time_sec": 0.0,
            "robot_motion_time_sec": 6.0,
            "human_robot_motion_overlap_time_sec": 0.0,
            "human_robot_interaction_time_sec": 0.0,
            "dynamic_scene_success": False,
            "thresholds": {
                "min_moving_human_count": 1,
                "min_human_motion_time_sec": 5.0,
                "min_human_robot_motion_overlap_time_sec": 3.0,
                "min_human_robot_interaction_time_sec": 1.0,
            },
        }
    )

    assert result["pass"] is False
    assert "moving_human_count_below_threshold" in result["failures"]
    assert "human_robot_motion_overlap_below_threshold" in result["failures"]
