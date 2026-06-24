import json

import yaml

from arena_bringup.benchmark_failure_report import generate_report


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_benchmark_failure_report_classifies_strict_failure(tmp_path):
    run_dir = tmp_path
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "parameters": {
                    "world": "grscenes_5",
                    "scenario_file": "default",
                    "robot": "Ai2_Bot2",
                },
                "result": {
                    "finished_observed": True,
                    "end_reason": "episode_sim_timeout",
                    "launch_returncode": 0,
                    "artifact_validation_returncode": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "vln_task_metrics.json",
        {
            "world": "grscenes_5",
            "scenario": "default",
            "instruction": "Go to the counter.",
            "strict_task_success": False,
            "strict_task_failure_reasons": [
                "episode_timeout",
                "goal_not_reached",
                "commanded_stuck",
                "static_occupancy_collision",
            ],
            "goal": {
                "navigation_error_m": 1.8,
                "goal_tolerance_m": 0.75,
            },
            "language_task_contract": {
                "contract_type": "native_scenario_goal",
                "evaluated_predicates": ["goal_reached(robot, native_scenario_goal)"],
                "unsupported_predicates": ["bddl_semantic_predicates"],
            },
            "episode_timing": {"timed_out": True},
            "commanded_stuck": {
                "commanded_stuck_time_sec": 6.0,
                "commanded_stuck_intervals": [{"start_sec": 91.0, "end_sec": 97.0}],
            },
            "static_occupancy": {
                "map_available": False,
                "collision_sample_count": 3,
                "collision_samples": [
                    {"time_sec": 77.0, "position": [-7.7, 6.4], "pixel": [1, 2]},
                ],
                "intervals": [{"start_sec": 77.0, "end_sec": 122.0}],
            },
            "legacy_goal_reached_false_positive": True,
        },
    )
    _write_json(
        run_dir / "social_metrics.json",
        {
            "strict_social_success": False,
            "strict_social_failure_reasons": [
                "footprint_human_collision",
                "footprint_near_miss",
                "static_occupancy_collision",
                "commanded_stuck",
            ],
            "dynamic_scene_success": True,
            "moving_human_count": 2,
            "human_motion_total_m": 26.9,
            "min_footprint_clearance_m": -0.01,
            "min_footprint_clearance_sample": {
                "time_sec": 37.0,
                "robot_position": [-7.2, 3.8],
                "human_id": "1",
                "human_position": [-6.9, 3.4],
            },
            "footprint_human_collision_events": [{"time_sec": 37.0}],
            "footprint_near_miss_events": [{"time_sec": 12.0}],
            "review_intervals": {
                "static_occupancy": [{"start_sec": 77.0, "end_sec": 122.0}],
                "commanded_stuck": [{"start_sec": 91.0, "end_sec": 97.0}],
            },
        },
    )
    _write_json(
        run_dir / "artifact_validation.json",
        {
            "overall_pass": False,
            "social_nav_ready": False,
            "failed_checks": ["metrics"],
            "warnings": ["debug overlay uses ego-camera fallback; model debug image was unavailable"],
        },
    )
    _write_json(
        run_dir / "internnav_diagnostic_summary.json",
        {
            "event_counts": {"trajectory": 3},
            "command_stats": {"forward_count": 2},
            "odom_goal_distance": {"progress_first_minus_last": 3.0},
        },
    )
    _write_json(
        run_dir / "video_index.json",
        {
            "episodes": [
                {
                    "ego_video": str(run_dir / "videos" / "episode_0000" / "ego_observation.mp4"),
                    "debug_overlay_video": str(run_dir / "videos" / "episode_0000" / "ego_debug_overlay.mp4"),
                    "sim_top_down_video": str(run_dir / "videos" / "episode_0000" / "sim_top_down.mp4"),
                    "map_top_down_video": str(run_dir / "videos" / "episode_0000" / "map_top_down_follow.mp4"),
                    "debug_overlay_fallback": True,
                    "debug_overlay_source": {
                        "status": "no_post_reset_model_debug_image",
                        "topic": "/task_generator_node/Ai2_Bot2/internnav/debug_image",
                        "received_count": 0,
                        "post_reset_received_count": 0,
                        "model_frame_count": 0,
                        "fallback_frame_count": 10,
                    },
                    "ego_frames": 10,
                    "debug_overlay_frames": 10,
                    "top_down_frames": 10,
                    "sim_top_down_frames": 10,
                }
            ]
        },
    )

    report = generate_report(run_dir)

    assert (run_dir / "benchmark_failure_report.json").exists()
    assert report["strict_task"]["success"] is False
    assert report["strict_task"]["language_task_contract"]["contract_type"] == "native_scenario_goal"
    assert report["strict_social"]["dynamic_scene_success"] is True
    assert report["videos"]["debug_overlay_fallback"] is True
    assert report["videos"]["debug_overlay_source"]["status"] == "no_post_reset_model_debug_image"
    assert report["strict_social"]["min_footprint_clearance_sample"]["time_sec"] == 37.0
    assert "dynamic_scene_valid" in report["classification"]["tags"]
    assert "commanded_stuck_behavior" in report["classification"]["tags"]
    assert "social_footprint_safety_failure" in report["classification"]["tags"]
    assert "debug_overlay_fallback" in report["classification"]["tags"]
    assert "debug_overlay_source_no_post_reset_model_debug_image" in report["classification"]["tags"]
