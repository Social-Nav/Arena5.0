import csv
import json

from arena_bringup.social_nav_validation import (
    _check_dynamic_scene,
    _check_humans,
    _check_metrics,
    _check_model_control,
    _check_videos,
    _diagnostic_warnings,
    _trace_events,
    generate_artifact_validation,
)


def _write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metric_bundle(tmp_path, *, task_success=True, social_success=True):
    _write_rows(
        tmp_path / "metrics.csv",
        ["episode", "result", "time", "path_length"],
        [{
            "episode": 0,
            "result": "GOAL_REACHED" if task_success else "MODEL_STOP",
            "time": 4.0,
            "path_length": 1.0,
        }],
    )
    task = {
        "schema_version": 3,
        "strict_task_success": task_success,
        "strict_task_failure_reasons": [] if task_success else ["goal_not_reached"],
        "goal": {"navigation_error_m": 0.2 if task_success else 2.0},
        "vln": {
            "spl": 1.0 if task_success else 0.0,
            "ndtw": 1.0 if task_success else 0.2,
            "sdtw": 1.0 if task_success else 0.0,
            "trajectory_length_m": 1.0,
            "reference_path": {"available": True},
        },
        "episode_timing": {"duration_sec": 4.0},
    }
    (tmp_path / "vln_task_metrics.json").write_text(
        json.dumps(task), encoding="utf-8"
    )
    social = {
        "schema_version": 2,
        "humans_present": True,
        "social_success": social_success,
        "strict_social_success": social_success,
        "strict_social_failure_reasons": [] if social_success else ["footprint_near_miss"],
        "path_length_m": 1.0,
        "max_humans_observed": 1,
        "moving_human_count": 1,
        "human_motion_time_sec": 4.0,
        "robot_motion_time_sec": 4.0,
        "base_metrics": {"first": {"result": "GOAL_REACHED" if task_success else "MODEL_STOP"}},
        "min_human_distance_m": 2.0,
        "min_footprint_clearance_m": 1.45,
        "personal_space_violation_time_sec": 0.0,
        "footprint_personal_space_violation_time_sec": 0.0,
        "near_miss_count": 0,
        "human_collision_count": 0,
        "footprint_near_miss_count": 0,
        "footprint_human_collision_count": 0,
        "crowd_freezing_time_sec": 0.0,
    }
    (tmp_path / "social_metrics.json").write_text(
        json.dumps(social), encoding="utf-8"
    )
    return social


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
                    "debug_overlay_source": {
                        "status": "no_post_reset_model_debug_image",
                        "topic": "/task_generator_node/Ai2_Bot2/internnav/debug_image",
                        "received_count": 0,
                        "post_reset_received_count": 0,
                        "model_frame_count": 0,
                        "fallback_frame_count": 1,
                    },
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
    assert result["videos"]["ego_debug_overlay"]["source_status"] == "no_post_reset_model_debug_image"
    assert result["videos"]["ego_debug_overlay"]["source"]["fallback_frame_count"] == 1


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


def test_zero_human_control_is_valid_when_explicitly_expected(tmp_path):
    _write_rows(tmp_path / 'human_states.csv', ['time', 'data'], [])
    social = {
        'humans_present': False,
        'max_humans_observed': 0,
        'moving_human_count': 0,
        'human_motion_total_m': 0.0,
        'human_motion_time_sec': 0.0,
        'robot_motion_time_sec': 1.0,
        'human_robot_motion_overlap_time_sec': 0.0,
        'human_robot_interaction_time_sec': 0.0,
        'dynamic_scene_success': False,
        'thresholds': {},
    }

    assert _check_humans(tmp_path, social, expected_human_count=0)['pass'] is True
    dynamic = _check_dynamic_scene(social, expected_human_count=0)
    assert dynamic['pass'] is True
    assert dynamic['max_humans_observed'] == 0


def test_exact_human_count_mismatch_is_invalid(tmp_path):
    _write_rows(tmp_path / 'human_states.csv', ['time', 'data'], [
        {'time': '1', 'data': json.dumps([{'id': index} for index in range(4)])},
    ])
    social = {
        'humans_present': True,
        'max_humans_observed': 4,
        'moving_human_count': 4,
        'human_motion_total_m': 10.0,
        'human_motion_time_sec': 10.0,
        'robot_motion_time_sec': 1.0,
        'human_robot_motion_overlap_time_sec': 1.0,
        'human_robot_interaction_time_sec': 0.0,
        'dynamic_scene_success': False,
        'thresholds': {},
    }

    assert _check_humans(tmp_path, social, expected_human_count=5)['pass'] is False
    assert 'human_count_mismatch' in _check_dynamic_scene(
        social, expected_human_count=5
    )['failures']


def test_dynamic_scene_validity_does_not_require_policy_to_enter_human_interaction_radius():
    result = _check_dynamic_scene(
        {
            "moving_human_count": 2,
            "human_motion_total_m": 20.0,
            "human_motion_time_sec": 30.0,
            "robot_motion_time_sec": 10.0,
            "human_robot_motion_overlap_time_sec": 10.0,
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

    assert result["pass"] is True
    assert result["dynamic_scene_success"] is False
    assert result["human_robot_interaction_time_sec"] == 0.0


def test_metrics_check_accepts_valid_successful_episode(tmp_path):
    social_metrics = _write_metric_bundle(tmp_path)

    result = _check_metrics(tmp_path, social_metrics)

    assert result["pass"] is True
    assert result["task_success"] is True
    assert result["social_success"] is True
    assert result["strict_task_success"] is True
    assert result["strict_social_success"] is True


def test_metrics_check_accepts_valid_unsuccessful_episode(tmp_path):
    social_metrics = _write_metric_bundle(tmp_path, task_success=False)

    result = _check_metrics(tmp_path, social_metrics)

    assert result["pass"] is True
    assert result["task_success"] is False
    assert result["social_success"] is True
    assert result["strict_task_success"] is False
    assert result["strict_task_failure_reasons"] == ["goal_not_reached"]


def test_metrics_check_preserves_social_failure_as_outcome(tmp_path):
    social_metrics = _write_metric_bundle(tmp_path, social_success=False)

    result = _check_metrics(tmp_path, social_metrics)

    assert result["pass"] is True
    assert result["social_success"] is False
    assert result["strict_social_success"] is False


def test_metrics_check_rejects_legacy_social_schema_without_footprint_fields(tmp_path):
    social_metrics = _write_metric_bundle(tmp_path)
    del social_metrics["min_footprint_clearance_m"]
    del social_metrics["footprint_human_collision_count"]

    result = _check_metrics(tmp_path, social_metrics)

    assert result["pass"] is False
    assert result["required_social_fields_present"] is False
    assert "min_footprint_clearance_m" in result["missing_social_fields"]
    assert "footprint_human_collision_count" in result["missing_social_fields"]


def test_metrics_check_rejects_missing_or_corrupt_metrics(tmp_path):
    social_metrics = _write_metric_bundle(tmp_path)
    (tmp_path / "metrics.csv").unlink()
    assert _check_metrics(tmp_path, social_metrics)["pass"] is False

    (tmp_path / "metrics.csv").write_text(
        'episode,result,time,path_length\n0,MODEL_STOP,4.0,1.0\n',
        encoding='utf-8',
    )
    (tmp_path / "vln_task_metrics.json").write_text('{not-json', encoding='utf-8')
    result = _check_metrics(tmp_path, social_metrics)
    assert result["pass"] is False
    assert result["vln_task_schema_valid"] is False


def test_versioned_validation_output_does_not_overwrite_original(tmp_path, monkeypatch):
    monkeypatch.setenv('ARENA_POSTPROCESS_SOURCE_COMMIT', 'parser123')
    original = tmp_path / 'artifact_validation.json'
    original.write_text('{"schema_version": 1, "sentinel": true}', encoding='utf-8')
    (tmp_path / 'run_manifest.yaml').write_text(
        'parameters:\n  expected_human_count: 0\n', encoding='utf-8'
    )
    derived = tmp_path / 'artifact_validation.semantic_v2.json'

    report = generate_artifact_validation(tmp_path, derived)

    assert json.loads(original.read_text())['sentinel'] is True
    assert derived.is_file()
    assert report['schema_version'] == 2
    assert report['semantics'] == 'artifact_evidence_validity_separate_from_task_outcome'
    assert report['parser']['git_commit'] == 'parser123'


def test_diagnostic_warnings_report_unsupported_language_predicates(tmp_path):
    strict_task = {
        "strict_task_success": True,
        "language_task_contract": {
            "unsupported_predicates_present": True,
            "unsupported_predicates": ["bddl_semantic_predicates", "orientation_at_goal"],
        },
    }
    task_path = tmp_path / "vln_task_metrics.json"
    task_path.write_text(json.dumps(strict_task), encoding="utf-8")
    manifest = {
        "artifacts": {"vln_task_metrics_path": str(task_path)},
        "parameters": {},
    }

    warnings = _diagnostic_warnings({}, manifest, {"base_metrics": {"first": {"result": "GOAL_REACHED"}}})

    assert any("unsupported predicates" in warning for warning in warnings)
    assert any("bddl_semantic_predicates" in warning for warning in warnings)
