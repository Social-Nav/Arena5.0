#!/usr/bin/env python3
"""Extract robot demonstration trajectories and path lengths from social_gen parquet.

The recorded ``pose.<camera_setting>`` is the camera-to-world transform.  This
tool applies the same fixed camera-mount inverse used by
``process_raw_to_dataset.py`` to recover the robot footprint pose for every
recorded frame.  It writes one compact ``.npz`` per episode and JSONL/CSV
summaries of the resulting path lengths.

This is an *executed* (demonstration) trajectory, not Nav2's unrecorded global
plan.  It therefore includes any detours made around dynamic pedestrians.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


ROBOT_MODEL = "Ai2_Bot2"
CAMERA_FORWARD_M = 0.3506
CAMERA_LATERAL_M = 0.0
CAMERA_HEIGHT_M = 1.6453
CAMERA_PITCH_DEG = 30
T_ROBOT2CAMERA = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def read_robot_scale(model_params: Path) -> float:
    with model_params.open(encoding="utf-8") as f:
        value = float((yaml.safe_load(f) or {})["scale"])
    if value <= 0.0:
        raise ValueError(f"robot scale must be positive, got {value}: {model_params}")
    return value


def camera_to_robot(scale: float) -> np.ndarray:
    """Camera-to-world -> robot-footprint-to-world right-hand transform."""
    mount = np.eye(4, dtype=np.float64)
    mount[:3, 3] = [
        CAMERA_FORWARD_M * scale,
        CAMERA_LATERAL_M * scale,
        CAMERA_HEIGHT_M * scale,
    ]
    rad = np.radians(CAMERA_PITCH_DEG)
    pitch = np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, np.cos(-rad), -np.sin(-rad), 0.0],
         [0.0, np.sin(-rad), np.cos(-rad), 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return np.linalg.inv(mount @ T_ROBOT2CAMERA @ pitch)


def discover_scenes(data_path: Path) -> list[Path]:
    if (data_path / "data").is_dir():
        return [data_path]
    return sorted(
        (p for p in data_path.glob("grscenes_*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("_", 1)[1]),
    )


def pose_column(path: Path, requested: str | None) -> str:
    names = pq.ParquetFile(path).schema_arrow.names
    if requested:
        if requested not in names:
            raise KeyError(f"{path}: missing requested column {requested}")
        return requested
    candidates = [name for name in names if name.startswith("pose.")]
    if len(candidates) != 1:
        raise ValueError(f"{path}: expected exactly one pose.* column, got {candidates}")
    return candidates[0]


def trajectory_from_parquet(path: Path, c2r: np.ndarray, requested_column: str | None):
    column = pose_column(path, requested_column)
    columns = [column]
    names = pq.ParquetFile(path).schema_arrow.names
    if "timestamp" in names:
        columns.append("timestamp")
    table = pq.read_table(path, columns=columns)
    camera = np.asarray(table.column(column).to_pylist(), dtype=np.float64).reshape(-1, 4, 4)
    robot = camera @ c2r
    xyyaw = np.column_stack((
        robot[:, 0, 3],
        robot[:, 1, 3],
        np.arctan2(robot[:, 1, 0], robot[:, 0, 0]),
    )).astype(np.float32)
    if "timestamp" in table.column_names:
        timestamp = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float32)
    else:
        timestamp = np.arange(len(xyyaw), dtype=np.float32) / 30.0
    return xyyaw, timestamp, column


def lengths(xyyaw: np.ndarray, min_step: float) -> tuple[float, float]:
    if len(xyyaw) < 2:
        return 0.0, 0.0
    steps = np.linalg.norm(np.diff(xyyaw[:, :2], axis=0), axis=1)
    return float(steps.sum()), float(steps[steps >= min_step].sum())


def main() -> int:
    arena_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, type=Path,
                        help="grscenes root or one grscenes_N directory")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="separate directory for extracted trajectories and summaries")
    parser.add_argument("--pose-column", help="override auto-detected pose.<setting> column")
    parser.add_argument("--min-step-m", type=float, default=0.01,
                        help="ignore smaller per-frame displacements for filtered length (default: 0.01)")
    parser.add_argument("--model-params", type=Path,
                        default=arena_root / "arena_robots/arena_robots/robots" / ROBOT_MODEL / "model_params.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.min_step_m < 0:
        parser.error("--min-step-m must be non-negative")
    scale = read_robot_scale(args.model_params)
    c2r = camera_to_robot(scale)
    scenes = discover_scenes(args.data_path)
    if not scenes:
        parser.error(f"no grscenes_* worlds found below {args.data_path}")

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    for scene in scenes:
        episode_files = sorted(
            scene.glob("data/chunk-*/episode_*.parquet"),
            key=lambda p: int(EPISODE_RE.search(p.name).group(1)),
        )
        if not episode_files:
            print(f"[SKIP] {scene.name}: no parquet episodes")
            continue
        out_dir = output_root / scene.name / "trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        scene_records = []
        for parquet in episode_files:
            ep_id = int(EPISODE_RE.search(parquet.name).group(1))
            trajectory, timestamps, column = trajectory_from_parquet(parquet, c2r, args.pose_column)
            npz_path = out_dir / f"episode_{ep_id:06d}.npz"
            if args.overwrite or not npz_path.exists():
                np.savez_compressed(npz_path, xyyaw=trajectory, timestamp=timestamps)
            raw_length, filtered_length = lengths(trajectory, args.min_step_m)
            record = {
                "world": scene.name,
                "episode_index": ep_id,
                "trajectory_file": str(npz_path.relative_to(output_root)),
                "n_frames": int(len(trajectory)),
                "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
                "length_m_raw": raw_length,
                "length_m": filtered_length,
                "length_min_step_m": args.min_step_m,
                "pose_column": column,
                "robot_scale": scale,
            }
            scene_records.append(record)
            all_records.append(record)

        index_path = output_root / scene.name / "reference_trajectories.jsonl"
        with index_path.open("w", encoding="utf-8") as f:
            for record in scene_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[OK] {scene.name}: {len(scene_records)} trajectories -> {index_path}")

    csv_path = output_root / "reference_trajectory_lengths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0]) if all_records else ["world"])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[OK] {len(all_records)} total trajectories; summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
