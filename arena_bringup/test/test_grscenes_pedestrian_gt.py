import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena_bringup.grscenes_pedestrian_gt import (  # noqa: E402
    batch_main,
    compare_executed_to_gt,
    compare_scenario_to_gt,
    draw_overlay,
    load_human_states_csv,
    load_scenario_tracks,
    load_tracks_from_parquet,
    scenario_gt_params_path,
    write_gt_scenario,
    write_tracks_csv,
    write_tracks_json,
)
from arena_bringup.grscenes_pedestrian_replay import sample_tracks  # noqa: E402


def _matrix(x, y, yaw=0.0, z=0.0):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [c, -s, 0.0, x, s, c, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0]


def test_load_tracks_filters_non_body_transforms(tmp_path):
    parquet = tmp_path / "params.parquet"
    pd.DataFrame(
        [
            {
                "frame_index": 0,
                "observation.peds_state": {
                    "p_1": _matrix(1.0, 2.0, 0.1),
                    "p_1_head": _matrix(1.0, 2.0, 0.1, z=0.875),
                    "p_2": _matrix(4.0, 5.0, -0.2),
                },
            },
            {
                "frame_index": 1,
                "observation.peds_state": {
                    "p_1": _matrix(2.0, 2.5, 0.2),
                    "p_1_head": _matrix(2.0, 2.5, 0.2, z=0.875),
                    "p_2": _matrix(4.5, 5.5, -0.1),
                },
            },
        ]
    ).to_parquet(parquet)

    tracks = load_tracks_from_parquet(parquet, frame_dt_sec=0.4)

    assert [track.source_id for track in tracks] == ["p_1", "p_2"]
    assert tracks[0].name == "hunav_01"
    assert tracks[0].frames[1].time_sec == 0.4
    assert tracks[0].frames[1].x == 2.0
    assert tracks[1].frames[0].y == 5.0


def test_scenario_comparison_reports_large_end_error(tmp_path):
    parquet = tmp_path / "params.parquet"
    scenario = tmp_path / "scenario.yaml"
    pd.DataFrame(
        [
            {"frame_index": 0, "observation.peds_state": {"p_1": _matrix(0.0, 0.0)}},
            {"frame_index": 1, "observation.peds_state": {"p_1": _matrix(1.0, 0.0)}},
        ]
    ).to_parquet(parquet)
    scenario.write_text(
        yaml.safe_dump(
            {
                "dynamic": [
                    {
                        "name": "hunav_1",
                        "model": "female_adult_business_02",
                        "pose": [0.0, 0.0, 0.0],
                        "waypoints": [[5.0, 0.0, 0.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = compare_scenario_to_gt(load_scenario_tracks(scenario), load_tracks_from_parquet(parquet))

    assert report["scenario_track_count"] == 1
    assert report["gt_track_count"] == 1
    assert report["tracks"][0]["start_error_m"] == 0.0
    assert report["tracks"][0]["end_error_m"] == 4.0


def test_writers_overlay_and_replay_sampling(tmp_path):
    parquet = tmp_path / "params.parquet"
    pd.DataFrame(
        [
            {"frame_index": 0, "observation.peds_state": {"p_1": _matrix(0.0, 0.0, 0.0)}},
            {"frame_index": 1, "observation.peds_state": {"p_1": _matrix(2.0, 0.0, 0.0)}},
        ]
    ).to_parquet(parquet)
    tracks = load_tracks_from_parquet(parquet, frame_dt_sec=1.0)

    write_tracks_json(tracks, tmp_path / "tracks.json")
    write_tracks_csv(tracks, tmp_path / "tracks.csv")
    draw_overlay(gt_tracks=tracks, output_path=tmp_path / "overlay.png")
    sample = sample_tracks(tracks, 0.25)[0][1]

    assert json.loads((tmp_path / "tracks.json").read_text())["tracks"][0]["source_id"] == "p_1"
    assert "hunav_01" in (tmp_path / "tracks.csv").read_text()
    assert (tmp_path / "overlay.png").stat().st_size > 0
    assert sample.x == 0.5


def test_write_gt_scenario_preserves_metadata_and_replaces_waypoints(tmp_path):
    parquet = tmp_path / "params.parquet"
    scenario = tmp_path / "scenario.yaml"
    output = tmp_path / "gt_scenario.yaml"
    pd.DataFrame(
        [
            {"frame_index": 0, "observation.peds_state": {"p_1": _matrix(0.0, 0.0, 0.0)}},
            {"frame_index": 1, "observation.peds_state": {"p_1": _matrix(1.0, 0.0, 0.0)}},
            {"frame_index": 2, "observation.peds_state": {"p_1": _matrix(2.0, 0.0, 0.0)}},
        ]
    ).to_parquet(parquet)
    scenario.write_text(
        yaml.safe_dump(
            {
                "dynamic": [
                    {
                        "name": "hunav_1",
                        "model": "female_adult_business_02",
                        "pose": [9.0, 9.0, 0.0],
                        "behavior_tree": "BTRegularNav.xml",
                        "velocity": 0.8,
                        "waypoints": [[10.0, 10.0, 0.0]],
                    }
                ],
                "robots": [{"start": [0, 0, 0], "goal": [1, 1, 0]}],
            }
        ),
        encoding="utf-8",
    )

    write_gt_scenario(
        scenario,
        load_tracks_from_parquet(parquet, frame_dt_sec=1.0),
        output,
        min_spacing_m=0.5,
    )
    data = yaml.safe_load(output.read_text())

    assert data["dynamic"][0]["model"] == "female_adult_business_02"
    assert data["dynamic"][0]["velocity"] == 0.8
    assert data["dynamic"][0]["behavior_tree"] == "./BTRegularNav.xml"
    assert data["dynamic"][0]["pose"] == [0.0, 0.0, 0.0]
    assert data["dynamic"][0]["waypoints"][-1] == [2.0, 0.0, 0.0]
    assert data["robots"][0]["goal"] == [1, 1, 0]


def test_human_states_csv_compares_to_gt_with_normalized_names(tmp_path):
    parquet = tmp_path / "params.parquet"
    human_states = tmp_path / "human_states.csv"
    pd.DataFrame(
        [
            {"frame_index": 0, "observation.peds_state": {"p_1": _matrix(0.0, 0.0, 0.0)}},
            {"frame_index": 1, "observation.peds_state": {"p_1": _matrix(1.0, 0.0, 0.0)}},
        ]
    ).to_parquet(parquet)
    human_states.write_text(
        "time,data\n"
        "0,\"[{'id': '1', 'name': 'hunav_1', 'position': [0.0, 0.0], 'theta': 0.0}]\"\n"
        "10000000000,\"[{'id': '1', 'name': 'hunav_1', 'position': [1.0, 0.0], 'theta': 0.0}]\"\n",
        encoding="utf-8",
    )

    report = compare_executed_to_gt(
        load_human_states_csv(human_states),
        load_tracks_from_parquet(parquet, frame_dt_sec=1.0),
    )

    assert report["executed_track_count"] == 1
    assert report["tracks"][0]["executed_name"] == "hunav_01"
    assert report["tracks"][0]["end_error_m"] == 0.0


def test_batch_main_writes_summary(tmp_path):
    grscenes_root = tmp_path / "grscenes_"
    worlds_dir = tmp_path / "worlds"
    params_dir = grscenes_root / "grscenes_1" / "default" / "stamp" / "episode_00" / "data"
    scenario_dir = worlds_dir / "grscenes_1" / "scenarios" / "default"
    params_dir.mkdir(parents=True)
    scenario_dir.mkdir(parents=True)
    (worlds_dir / "grscenes_1" / "map").mkdir(parents=True)
    pd.DataFrame(
        [
            {"frame_index": 0, "observation.peds_state": {"p_1": _matrix(0.0, 0.0, 0.0)}},
            {"frame_index": 1, "observation.peds_state": {"p_1": _matrix(1.0, 0.0, 0.0)}},
        ]
    ).to_parquet(params_dir / "params.parquet")
    (scenario_dir / "scenario.yaml").write_text(
        yaml.safe_dump(
            {
                "dynamic": [
                    {
                        "name": "hunav_1",
                        "model": "female_adult_business_02",
                        "pose": [0.0, 0.0, 0.0],
                        "waypoints": [[1.0, 0.0, 0.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    batch_main(
        [
            "--grscenes-root",
            str(grscenes_root),
            "--scenario-worlds-dir",
            str(worlds_dir),
            "--output-dir",
            str(tmp_path / "report"),
            "--no-overlay",
        ]
    )

    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["cases"][0]["world"] == "grscenes_1"
    assert summary["cases"][0]["max_end_error_m"] == 0.0


def test_scenario_gt_params_path_prefers_bound_metadata(tmp_path):
    grscenes_root = tmp_path / "grscenes_"
    worlds_dir = tmp_path / "worlds"
    old_params = grscenes_root / "grscenes_1" / "default" / "old" / "episode_00" / "data" / "params.parquet"
    bound_params = grscenes_root / "grscenes_1" / "default" / "bound" / "episode_00" / "data" / "params.parquet"
    scenario_dir = worlds_dir / "grscenes_1" / "scenarios" / "default"
    old_params.parent.mkdir(parents=True)
    bound_params.parent.mkdir(parents=True)
    scenario_dir.mkdir(parents=True)
    pd.DataFrame([{"frame_index": 0, "observation.peds_state": {"p_1": _matrix(0.0, 0.0)}}]).to_parquet(old_params)
    pd.DataFrame([{"frame_index": 0, "observation.peds_state": {"p_1": _matrix(1.0, 0.0)}}]).to_parquet(bound_params)
    (scenario_dir / "scenario.yaml").write_text(
        yaml.safe_dump({"metadata": {"gt_params_path": str(bound_params)}, "dynamic": []}),
        encoding="utf-8",
    )

    resolved = scenario_gt_params_path(
        world="grscenes_1",
        scenario="default",
        grscenes_root=grscenes_root,
        scenario_worlds_dir=worlds_dir,
    )

    assert resolved == bound_params
