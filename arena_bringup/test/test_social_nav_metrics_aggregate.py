import json

import yaml

from arena_bringup.social_nav_metrics_aggregate import aggregate_summary, summarize_run


def test_aggregate_reports_legacy_and_strict_rates_separately(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "parameters": {
                    "world": "grscenes_5",
                    "robot": "Ai2_Bot2",
                    "local_planner": "dual_vln",
                    "human": "hunav",
                    "scenario_file": "default",
                },
                "result": {"end_reason": "finished"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "vln_task_metrics.json").write_text(
        json.dumps(
            {
                "strict_task_success": False,
                "strict_task_failure_reasons": ["goal_not_reached"],
                "episode_timing": {"duration_sec": 120.0, "timed_out": True},
                "goal": {
                    "start_xy": [0.0, 0.0],
                    "goal_xy": [0.0, 5.0],
                    "final_xy": [0.0, 3.5],
                    "navigation_error_m": 1.5,
                    "oracle_error_m": 1.0,
                },
                "vln": {"spl": 0.0, "ndtw": 0.2},
                "static_occupancy": {"collision_sample_count": 2},
                "commanded_stuck": {"commanded_stuck_time_sec": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "social_metrics.json").write_text(
        json.dumps(
            {
                "humans_present": True,
                "social_success": True,
                "strict_social_success": False,
                "strict_social_failure_reasons": ["static_occupancy_collision"],
                "path_length_m": 2.0,
                "min_human_distance_m": 1.0,
                "min_footprint_clearance_m": 0.4,
                "min_footprint_clearance_sample": {
                    "time_sec": 12.5,
                    "human_id": "7",
                    "distance_m": 0.95,
                    "footprint_clearance_m": 0.4,
                },
                "near_miss_count": 0,
                "human_collision_count": 0,
                "footprint_near_miss_count": 0,
                "footprint_human_collision_count": 0,
                "personal_space_violation_time_sec": 0.0,
                "footprint_personal_space_violation_time_sec": 0.0,
                "crowd_freezing_time_sec": 0.0,
                "max_humans_observed": 2,
                "base_metrics": {"first": {"result": "GOAL_REACHED"}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifact_validation.json").write_text(
        json.dumps(
            {
                "overall_pass": False,
                "social_nav_ready": False,
                "failed_checks": ["metrics"],
                "warnings": [
                    "legacy GOAL_REACHED is true but strict_task_success is false",
                    "legacy social_success is true but strict_social_success is false",
                ],
                "checks": {
                    "metrics": {
                        "episode_result": "GOAL_REACHED",
                        "legacy_task_success": True,
                        "strict_task_success": False,
                        "strict_social_success": False,
                        "robot_moved": True,
                    },
                    "videos": {
                        "videos": {
                            "ego_debug_overlay": {
                                "fallback": True,
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    row = summarize_run(run_dir)
    summary = aggregate_summary([row])

    assert row["legacy_task_success"] is True
    assert row["task_success"] is False
    assert row["strict_task_success"] is False
    assert row["episode_timeout"] is True
    assert row["episode_duration_sec"] == 120.0
    assert row["goal_progress_m"] == 3.5
    assert row["min_footprint_clearance_time_sec"] == 12.5
    assert row["min_footprint_clearance_human_id"] == "7"
    assert row["debug_overlay_fallback"] is True
    assert row["legacy_social_success"] is True
    assert row["social_success"] is False
    assert row["strict_social_success"] is False
    assert "legacy_task_false_positive" in row["failure_tags"]
    assert "timeout" in row["failure_tags"]
    assert "debug_overlay_fallback" in row["failure_tags"]
    assert summary["task_success_rate"] == 0.0
    assert summary["social_success_rate"] == 0.0
    assert summary["legacy_task_success_rate"] == 1.0
    assert summary["strict_task_success_rate"] == 0.0
    assert summary["legacy_social_success_rate"] == 1.0
    assert summary["strict_social_success_rate"] == 0.0
    assert summary["benchmark_ready_rate"] == 0.0


def test_aggregate_falls_back_to_internnav_odom_goal_progress(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "parameters": {
                    "world": "grscenes_5",
                    "robot": "Ai2_Bot2",
                    "local_planner": "dual_vln",
                    "human": "hunav",
                    "scenario_file": "default",
                },
                "result": {"end_reason": "finished"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "vln_task_metrics.json").write_text(
        json.dumps(
            {
                "strict_task_success": False,
                "strict_task_failure_reasons": ["goal_not_reached"],
                "episode_timing": {"duration_sec": 30.0, "timed_out": False},
                "goal": {"navigation_error_m": 2.0, "oracle_error_m": 2.0},
                "vln": {"spl": 0.0, "ndtw": 0.1},
                "static_occupancy": {"collision_sample_count": 0},
                "commanded_stuck": {"commanded_stuck_time_sec": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "social_metrics.json").write_text(
        json.dumps(
            {
                "humans_present": True,
                "social_success": True,
                "strict_social_success": True,
                "strict_social_failure_reasons": [],
                "path_length_m": 1.0,
                "min_human_distance_m": 2.0,
                "min_footprint_clearance_m": 1.4,
                "near_miss_count": 0,
                "human_collision_count": 0,
                "footprint_near_miss_count": 0,
                "footprint_human_collision_count": 0,
                "personal_space_violation_time_sec": 0.0,
                "footprint_personal_space_violation_time_sec": 0.0,
                "crowd_freezing_time_sec": 0.0,
                "max_humans_observed": 2,
                "base_metrics": {"first": {"result": "TIMEOUT"}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifact_validation.json").write_text(
        json.dumps(
            {
                "overall_pass": False,
                "social_nav_ready": False,
                "failed_checks": ["metrics"],
                "warnings": [],
                "checks": {
                    "metrics": {
                        "episode_result": "TIMEOUT",
                        "legacy_task_success": False,
                        "strict_task_success": False,
                        "strict_social_success": True,
                        "robot_moved": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "internnav_diagnostic_summary.json").write_text(
        json.dumps(
            {
                "odom_goal_distance": {
                    "first": 4.8,
                    "last": 1.7,
                    "min": 1.6,
                    "progress_first_minus_last": 3.1,
                    "sample_count": 10,
                },
                "command_stats": {"forward_count": 4, "rotate_count": 6, "stop_count": 0},
                "fault_candidates": {},
            }
        ),
        encoding="utf-8",
    )

    row = summarize_run(run_dir)

    assert row["goal_progress_m"] == 3.1
    assert row["diagnostic_goal_progress_m"] == 3.1
    assert row["diagnostic_goal_distance_min_m"] == 1.6
