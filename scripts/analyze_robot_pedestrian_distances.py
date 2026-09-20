#!/usr/bin/env python3
"""Compute per-episode robot--pedestrian nearest distances from social_gen parquet.

Distances are 2-D distances between the recovered robot-footprint origin and
each pedestrian root transform.  They are geometric proximity metrics, not
Isaac physics-contact labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from extract_reference_trajectories import (
    EPISODE_RE,
    camera_to_robot,
    discover_scenes,
    pose_column,
    read_robot_scale,
)


BINS = ((0.4, "<0.4 m"), (0.6, "0.4–0.6 m"), (0.8, "0.6–0.8 m"),
        (1.0, "0.8–1.0 m"), (float("inf"), "≥1.0 m"))


def classify(distance: float | None) -> str:
    if distance is None:
        return "no pedestrian trajectory"
    for upper, name in BINS:
        if distance < upper:
            return name
    raise AssertionError("unreachable")


def pedestrian_xy(row: dict | None) -> list[np.ndarray]:
    """Extract valid x/y translations from one pose_peds parquet row."""
    result = []
    if not isinstance(row, dict):
        return result
    for name, value in row.items():
        if name == "none" or value is None:
            continue
        flat = np.asarray(value, dtype=np.float64)
        if flat.size != 16 or not np.isfinite(flat).all():
            continue
        matrix = flat.reshape(4, 4)
        result.append(matrix[:2, 3])
    return result


def analyze_episode(path: Path, c2r: np.ndarray, requested_column: str | None) -> dict:
    names = pq.ParquetFile(path).schema_arrow.names
    ep_id = int(EPISODE_RE.search(path.name).group(1))
    if "pose_peds" not in names:
        return {"episode_index": ep_id, "n_frames": 0, "min_center_distance_m": None,
                "distance_bin": "no pedestrian trajectory", "pedestrian_trajectory": False}

    pose_name = pose_column(path, requested_column)
    columns = [pose_name, "pose_peds"]
    if "timestamp" in names:
        columns.append("timestamp")
    table = pq.read_table(path, columns=columns)
    cameras = np.asarray(table.column(pose_name).to_pylist(), dtype=np.float64).reshape(-1, 4, 4)
    robots = (cameras @ c2r)[:, :2, 3]
    peds_rows = table.column("pose_peds").to_pylist()
    frame_min = np.full(len(robots), np.nan, dtype=np.float64)
    max_peds = 0
    for i, row in enumerate(peds_rows):
        peds = pedestrian_xy(row)
        max_peds = max(max_peds, len(peds))
        if peds:
            frame_min[i] = np.linalg.norm(np.asarray(peds) - robots[i], axis=1).min()

    valid = np.isfinite(frame_min)
    result = {
        "episode_index": ep_id,
        "n_frames": int(len(robots)),
        "frames_with_pedestrians": int(valid.sum()),
        "max_pedestrians_per_frame": max_peds,
        "min_center_distance_m": float(frame_min[valid].min()) if valid.any() else None,
        "mean_nearest_distance_m": float(frame_min[valid].mean()) if valid.any() else None,
        "distance_bin": classify(float(frame_min[valid].min())) if valid.any() else classify(None),
        "pedestrian_trajectory": bool(valid.any()),
        "frames_below_0_4m": int((frame_min < 0.4).sum()),
        "frames_below_0_6m": int((frame_min < 0.6).sum()),
        "frames_below_0_8m": int((frame_min < 0.8).sum()),
        "frames_below_1_0m": int((frame_min < 1.0).sum()),
        "pose_column": pose_name,
    }
    if "timestamp" in table.column_names and len(robots) > 1:
        timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        result["duration_s"] = float(timestamps[-1] - timestamps[0])
    else:
        result["duration_s"] = (len(robots) - 1) / 30.0
    return result


def main() -> int:
    arena_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pose-column")
    parser.add_argument(
        "--model-params", type=Path,
        default=arena_root / "arena_robots/arena_robots/robots/Ai2_Bot2/model_params.yaml",
    )
    args = parser.parse_args()
    scale = read_robot_scale(args.model_params)
    c2r = camera_to_robot(scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for scene in discover_scenes(args.data_path):
        for parquet in sorted(scene.glob("data/chunk-*/episode_*.parquet"),
                              key=lambda p: int(EPISODE_RE.search(p.name).group(1))):
            record = analyze_episode(parquet, c2r, args.pose_column)
            record["world"] = scene.name
            record["parquet_file"] = str(parquet.relative_to(args.data_path))
            record["robot_scale"] = scale
            records.append(record)

    fields = ["world", "episode_index", "parquet_file", "n_frames", "duration_s",
              "frames_with_pedestrians", "max_pedestrians_per_frame", "min_center_distance_m",
              "mean_nearest_distance_m", "distance_bin", "frames_below_0_4m",
              "frames_below_0_6m", "frames_below_0_8m", "frames_below_1_0m",
              "pedestrian_trajectory", "pose_column", "robot_scale"]
    csv_path = args.output_dir / "robot_pedestrian_min_distances.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    counts = {name: sum(r["distance_bin"] == name for r in records)
              for _, name in BINS}
    no_ped = sum(not r["pedestrian_trajectory"] for r in records)
    valid = [r["min_center_distance_m"] for r in records if r["min_center_distance_m"] is not None]
    summary = {
        "metric": "2-D robot-footprint to pedestrian-root center distance",
        "robot_scale": scale,
        "episodes_total": len(records),
        "episodes_with_pedestrian_trajectory": len(valid),
        "episodes_without_pedestrian_trajectory": no_ped,
        "minimum_distance_overall_m": min(valid) if valid else None,
        "median_episode_minimum_distance_m": float(np.median(valid)) if valid else None,
        "counts_by_episode_minimum_distance_bin": counts,
    }
    summary_path = args.output_dir / "robot_pedestrian_distance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Per-episode details: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
