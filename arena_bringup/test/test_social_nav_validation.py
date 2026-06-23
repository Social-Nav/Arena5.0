import csv
import json

from arena_bringup.social_nav_validation import (
    _check_dynamic_scene,
    _check_metrics,
    _check_model_control,
    _check_videos,
    _trace_events,
)


def _write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def test_video_check_reports_debug_overlay_fallback(tmp_path):
    video_path = tmp_path / "ego_debug_overlay.mp4"
    video_path.write_bytes(b"not-empty")
    result = _check_videos(
        tmp_path,
        {
            "episodes": [
                {
                    "ego_video": str(video_path),
                    "ego_frames": 1,
                    "debug_overlay_video": str(video_path),
                    "debug_overlay_frames": 1,
                    "debug_overlay_fallback": True,
                    "sim_top_down_video": str(video_path),
                    "sim_top_down_frames": 1,
                    "map_top_down_video": str(video_path),
                    "top_down_frames": 1,
                }
            ]
        },
    )

    assert result["pass"] is True
    assert result["videos"]["ego_debug_overlay"]["fallback"] is True


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


def test_metrics_check_requires_strict_task_and_social_success(tmp_path):
    _write_rows(tmp_path / "metrics.csv", ["result"], [{"result": "GOAL_REACHED"}])
    (tmp_path / "vln_task_metrics.json").write_text(
        json.dumps({"strict_task_success": True, "strict_task_failure_reasons": []}),
        encoding="utf-8",
    )
    social_metrics = {
        "humans_present": True,
        "social_success": True,
        "strict_social_success": True,
        "strict_social_failure_reasons": [],
        "path_length_m": 1.0,
        "base_metrics": {"first": {"result": "GOAL_REACHED"}},
        "min_human_distance_m": 2.0,
        "personal_space_violation_time_sec": 0.0,
        "near_miss_count": 0,
        "human_collision_count": 0,
        "crowd_freezing_time_sec": 0.0,
    }

    result = _check_metrics(tmp_path, social_metrics)

    assert result["pass"] is True
    assert result["legacy_task_success"] is True
    assert result["task_success"] is True
    assert result["legacy_social_success"] is True
    assert result["social_success"] is True
    assert result["strict_task_success"] is True
    assert result["strict_social_success"] is True


def test_metrics_check_fails_legacy_goal_reached_when_strict_task_failed(tmp_path):
    _write_rows(tmp_path / "metrics.csv", ["result"], [{"result": "GOAL_REACHED"}])
    (tmp_path / "vln_task_metrics.json").write_text(
        json.dumps({"strict_task_success": False, "strict_task_failure_reasons": ["goal_not_reached"]}),
        encoding="utf-8",
    )
    social_metrics = {
        "humans_present": True,
        "social_success": True,
        "strict_social_success": True,
        "strict_social_failure_reasons": [],
        "path_length_m": 1.0,
        "base_metrics": {"first": {"result": "GOAL_REACHED"}},
        "min_human_distance_m": 2.0,
        "personal_space_violation_time_sec": 0.0,
        "near_miss_count": 0,
        "human_collision_count": 0,
        "crowd_freezing_time_sec": 0.0,
    }

    result = _check_metrics(tmp_path, social_metrics)

    assert result["pass"] is False
    assert result["legacy_task_success"] is True
    assert result["task_success"] is False
    assert result["legacy_social_success"] is True
    assert result["social_success"] is True
    assert result["strict_task_success"] is False
    assert result["strict_task_failure_reasons"] == ["goal_not_reached"]


def test_metrics_check_reports_strict_social_as_default_social_success(tmp_path):
    _write_rows(tmp_path / "metrics.csv", ["result"], [{"result": "GOAL_REACHED"}])
    (tmp_path / "vln_task_metrics.json").write_text(
        json.dumps({"strict_task_success": True, "strict_task_failure_reasons": []}),
        encoding="utf-8",
    )
    social_metrics = {
        "humans_present": True,
        "legacy_social_success": True,
        "social_success": False,
        "strict_social_success": False,
        "strict_social_failure_reasons": ["footprint_near_miss"],
        "path_length_m": 1.0,
        "base_metrics": {"first": {"result": "GOAL_REACHED"}},
        "min_human_distance_m": 2.0,
        "personal_space_violation_time_sec": 0.0,
        "near_miss_count": 0,
        "human_collision_count": 0,
        "crowd_freezing_time_sec": 0.0,
    }

    result = _check_metrics(tmp_path, social_metrics)

    assert result["pass"] is False
    assert result["legacy_social_success"] is True
    assert result["social_success"] is False
    assert result["strict_social_success"] is False
