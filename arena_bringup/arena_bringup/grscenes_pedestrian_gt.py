"""GRScenes pedestrian GT extraction, overlay, and replay helpers."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class PedestrianFrame:
    frame_index: int
    time_sec: float
    x: float
    y: float
    yaw_rad: float


@dataclass(frozen=True)
class PedestrianTrack:
    source_id: str
    name: str
    frames: tuple[PedestrianFrame, ...]

    @property
    def points_xy(self) -> list[tuple[float, float]]:
        return [(frame.x, frame.y) for frame in self.frames]


def _matrix_values(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        if len(value) != 16:
            return None
        return [float(item) for item in value]
    except Exception:
        return None


def yaw_from_matrix(matrix: list[float]) -> float:
    return math.atan2(float(matrix[4]), float(matrix[0]))


def load_tracks_from_parquet(
    params_path: str | Path,
    *,
    frame_dt_sec: float = 0.1,
    z_filter_abs_max: float = 0.25,
    name_prefix: str = "hunav",
) -> list[PedestrianTrack]:
    """Load body-level pedestrian GT tracks from a GRScenes params.parquet file."""

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("pandas and a parquet engine are required to read GRScenes GT") from exc

    path = Path(params_path).expanduser()
    df = pd.read_parquet(path, columns=["observation.peds_state"])
    if df.empty:
        return []

    first = df.iloc[0]["observation.peds_state"] or {}
    source_ids = []
    for source_id, raw_matrix in first.items():
        matrix = _matrix_values(raw_matrix)
        if matrix is None:
            continue
        # GRScenes can include auxiliary/head transforms. Keep body transforms.
        if abs(float(matrix[11])) <= z_filter_abs_max:
            source_ids.append(str(source_id))
    source_ids = sorted(source_ids)

    tracks: dict[str, list[PedestrianFrame]] = {source_id: [] for source_id in source_ids}
    for row_index, row in df.iterrows():
        states = row["observation.peds_state"] or {}
        try:
            frame_index = int(row.get("frame_index", row_index))
        except Exception:
            frame_index = int(row_index)
        time_sec = float(frame_index) * float(frame_dt_sec)
        for source_id in source_ids:
            matrix = _matrix_values(states.get(source_id))
            if matrix is None:
                continue
            tracks[source_id].append(
                PedestrianFrame(
                    frame_index=frame_index,
                    time_sec=time_sec,
                    x=float(matrix[3]),
                    y=float(matrix[7]),
                    yaw_rad=yaw_from_matrix(matrix),
                )
            )

    return [
        PedestrianTrack(
            source_id=source_id,
            name=f"{name_prefix}_{index + 1:02d}",
            frames=tuple(frames),
        )
        for index, (source_id, frames) in enumerate(tracks.items())
        if frames
    ]


def load_scenario_tracks(scenario_path: str | Path) -> list[PedestrianTrack]:
    data = yaml.safe_load(Path(scenario_path).read_text(encoding="utf-8")) or {}
    dynamic = data.get("dynamic") or []
    tracks = []
    for index, agent in enumerate(dynamic):
        if not isinstance(agent, dict):
            continue
        pose = agent.get("pose")
        if not _is_xy(pose):
            continue
        points = [pose] + list(agent.get("waypoints") or [])
        frames = []
        for frame_index, point in enumerate(points):
            if not _is_xy(point):
                continue
            yaw = math.radians(float(point[2])) if len(point) >= 3 else 0.0
            frames.append(
                PedestrianFrame(
                    frame_index=frame_index,
                    time_sec=float(frame_index),
                    x=float(point[0]),
                    y=float(point[1]),
                    yaw_rad=yaw,
                )
            )
        if frames:
            tracks.append(
                PedestrianTrack(
                    source_id=str(agent.get("name") or f"scenario_{index + 1}"),
                    name=str(agent.get("name") or f"hunav_{index + 1:02d}"),
                    frames=tuple(frames),
                )
            )
    return tracks


def _is_xy(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except Exception:
        return False
    return True


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(points, points[1:]))


def normalize_human_name(name: str) -> str:
    prefix, sep, suffix = str(name or "").rpartition("_")
    if sep and suffix.isdigit():
        return f"{prefix}_{int(suffix):02d}"
    return str(name or "")


def load_human_states_csv(human_states_path: str | Path) -> list[PedestrianTrack]:
    """Load Arena recorder human_states.csv into PedestrianTrack objects."""

    tracks: dict[str, list[PedestrianFrame]] = {}
    with Path(human_states_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            try:
                time_raw = float(row.get("time", 0.0))
            except Exception:
                time_raw = 0.0
            time_sec = time_raw / 1e10
            try:
                agents = ast.literal_eval(row.get("data", "") or "[]")
            except Exception:
                continue
            if not isinstance(agents, list):
                continue
            for agent in agents:
                if not isinstance(agent, dict):
                    continue
                pos = agent.get("position")
                if not _is_xy(pos):
                    continue
                name = normalize_human_name(str(agent.get("name") or agent.get("id") or "human"))
                yaw = float(agent.get("theta", 0.0) or 0.0)
                tracks.setdefault(name, []).append(
                    PedestrianFrame(
                        frame_index=row_index,
                        time_sec=time_sec,
                        x=float(pos[0]),
                        y=float(pos[1]),
                        yaw_rad=yaw,
                    )
                )
    return [
        PedestrianTrack(source_id=name, name=name, frames=tuple(frames))
        for name, frames in sorted(tracks.items())
        if frames
    ]


def compare_executed_to_gt(
    executed_tracks: Iterable[PedestrianTrack],
    gt_tracks: Iterable[PedestrianTrack],
) -> dict[str, Any]:
    executed_list = list(executed_tracks)
    gt_list = list(gt_tracks)
    gt_by_name = {normalize_human_name(track.name): track for track in gt_list}
    rows = []
    for index, executed in enumerate(executed_list):
        gt = gt_by_name.get(normalize_human_name(executed.name))
        if gt is None and index < len(gt_list):
            gt = gt_list[index]
        if gt is None or not executed.frames:
            rows.append(
                {
                    "executed_name": executed.name,
                    "matched": False,
                    "reason": "missing_gt_track",
                }
            )
            continue
        executed_points = executed.points_xy
        gt_points = gt.points_xy
        nearest = [min(_distance(point, gt_point) for gt_point in gt_points) for point in executed_points]
        rows.append(
            {
                "executed_name": executed.name,
                "gt_source_id": gt.source_id,
                "gt_name": gt.name,
                "matched": True,
                "executed_frame_count": len(executed.frames),
                "gt_frame_count": len(gt.frames),
                "start_error_m": _distance(executed_points[0], gt_points[0]),
                "end_error_m": _distance(executed_points[-1], gt_points[-1]),
                "executed_path_length_m": _path_length(executed_points),
                "gt_path_length_m": _path_length(gt_points),
                "path_length_ratio": (
                    _path_length(executed_points) / _path_length(gt_points)
                    if _path_length(gt_points) > 1e-9
                    else None
                ),
                "max_executed_to_gt_m": max(nearest),
                "mean_executed_to_gt_m": sum(nearest) / len(nearest),
            }
        )
    return {
        "executed_track_count": len(executed_list),
        "gt_track_count": len(gt_list),
        "tracks": rows,
    }


def compare_scenario_to_gt(
    scenario_tracks: Iterable[PedestrianTrack],
    gt_tracks: Iterable[PedestrianTrack],
) -> dict[str, Any]:
    scenario_list = list(scenario_tracks)
    gt_list = list(gt_tracks)
    rows = []
    for index, scenario_track in enumerate(scenario_list):
        gt_track = gt_list[index] if index < len(gt_list) else None
        scenario_points = scenario_track.points_xy
        if gt_track is None or not scenario_points:
            rows.append(
                {
                    "scenario_name": scenario_track.name,
                    "gt_source_id": None,
                    "matched": False,
                    "reason": "missing_gt_track",
                }
            )
            continue
        gt_points = gt_track.points_xy
        nearest = [min(_distance(point, gt) for gt in gt_points) for point in scenario_points]
        rows.append(
            {
                "scenario_name": scenario_track.name,
                "gt_source_id": gt_track.source_id,
                "gt_name": gt_track.name,
                "matched": True,
                "scenario_point_count": len(scenario_points),
                "gt_frame_count": len(gt_points),
                "start_error_m": _distance(scenario_points[0], gt_points[0]),
                "end_error_m": _distance(scenario_points[-1], gt_points[-1]),
                "scenario_path_length_m": _path_length(scenario_points),
                "gt_path_length_m": _path_length(gt_points),
                "max_scenario_point_to_gt_m": max(nearest),
                "mean_scenario_point_to_gt_m": sum(nearest) / len(nearest),
            }
        )
    return {
        "scenario_track_count": len(scenario_list),
        "gt_track_count": len(gt_list),
        "tracks": rows,
    }


def sample_track_waypoints(
    track: PedestrianTrack,
    *,
    min_spacing_m: float = 0.5,
    max_waypoints: int = 64,
) -> list[list[float]]:
    """Return [x, y, yaw_deg] points sampled from a dense GT track."""

    if not track.frames:
        return []
    selected = [track.frames[0]]
    last_xy = (track.frames[0].x, track.frames[0].y)
    for frame in track.frames[1:-1]:
        xy = (frame.x, frame.y)
        if _distance(last_xy, xy) >= min_spacing_m:
            selected.append(frame)
            last_xy = xy
    if track.frames[-1] is not selected[-1]:
        selected.append(track.frames[-1])

    if max_waypoints > 0 and len(selected) > max_waypoints + 1:
        # Keep the first pose plus at most max_waypoints following points.
        keep_count = max_waypoints + 1
        indices = [
            round(i * (len(selected) - 1) / (keep_count - 1))
            for i in range(keep_count)
        ]
        selected = [selected[index] for index in indices]

    return [
        [round(frame.x, 5), round(frame.y, 5), round(math.degrees(frame.yaw_rad), 5)]
        for frame in selected
    ]


def build_gt_scenario(
    scenario_path: str | Path,
    gt_tracks: Iterable[PedestrianTrack],
    *,
    params_path: str | Path | None = None,
    min_spacing_m: float = 0.5,
    max_waypoints: int = 64,
) -> dict[str, Any]:
    """Build a scenario.yaml structure whose dynamic tracks come from GT."""

    path = Path(scenario_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dynamic = list(data.get("dynamic") or [])
    gt_list = list(gt_tracks)
    rebuilt_dynamic = []
    for index, agent in enumerate(dynamic):
        if not isinstance(agent, dict):
            continue
        if index >= len(gt_list):
            rebuilt_dynamic.append(agent)
            continue
        sampled = sample_track_waypoints(
            gt_list[index],
            min_spacing_m=min_spacing_m,
            max_waypoints=max_waypoints,
        )
        if not sampled:
            rebuilt_dynamic.append(agent)
            continue
        updated = dict(agent)
        updated["pose"] = sampled[0]
        updated["waypoints"] = sampled[1:]
        behavior_tree = str(updated.get("behavior_tree") or "").strip()
        if behavior_tree and not behavior_tree.startswith(("/", "./")):
            updated["behavior_tree"] = f"./{behavior_tree}"
        updated.setdefault("gt_source_id", gt_list[index].source_id)
        updated.setdefault("gt_source", "observation.peds_state")
        rebuilt_dynamic.append(updated)
    data["dynamic"] = rebuilt_dynamic
    if params_path is not None:
        metadata = dict(data.get("metadata") or {})
        metadata["gt_source"] = "observation.peds_state"
        metadata["gt_params_path"] = str(params_path)
        data["metadata"] = metadata
    return data


def write_gt_scenario(
    scenario_path: str | Path,
    gt_tracks: Iterable[PedestrianTrack],
    output_path: str | Path,
    *,
    params_path: str | Path | None = None,
    min_spacing_m: float = 0.5,
    max_waypoints: int = 64,
) -> None:
    scenario = build_gt_scenario(
        scenario_path,
        gt_tracks,
        params_path=params_path,
        min_spacing_m=min_spacing_m,
        max_waypoints=max_waypoints,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Generated from GRScenes observation.peds_state GT trajectories\n"
        + yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_tracks_json(tracks: Iterable[PedestrianTrack], output_path: str | Path) -> None:
    output = {
        "tracks": [
            {
                "source_id": track.source_id,
                "name": track.name,
                "frames": [
                    {
                        "frame_index": frame.frame_index,
                        "time_sec": frame.time_sec,
                        "x": frame.x,
                        "y": frame.y,
                        "yaw_rad": frame.yaw_rad,
                    }
                    for frame in track.frames
                ],
            }
            for track in tracks
        ]
    }
    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")


def write_tracks_csv(tracks: Iterable[PedestrianTrack], output_path: str | Path) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "source_id", "frame_index", "time_sec", "x", "y", "yaw_rad"],
        )
        writer.writeheader()
        for track in tracks:
            for frame in track.frames:
                writer.writerow(
                    {
                        "name": track.name,
                        "source_id": track.source_id,
                        "frame_index": frame.frame_index,
                        "time_sec": f"{frame.time_sec:.6f}",
                        "x": f"{frame.x:.6f}",
                        "y": f"{frame.y:.6f}",
                        "yaw_rad": f"{frame.yaw_rad:.6f}",
                    }
                )


def draw_overlay(
    *,
    gt_tracks: Iterable[PedestrianTrack],
    scenario_tracks: Iterable[PedestrianTrack] = (),
    map_yaml: str | Path | None = None,
    output_path: str | Path,
    size: tuple[int, int] = (1200, 1000),
) -> None:
    from PIL import Image, ImageDraw

    gt_tracks = list(gt_tracks)
    scenario_tracks = list(scenario_tracks)
    background = None
    origin = None
    resolution = None
    if map_yaml is not None:
        map_path = Path(map_yaml)
        map_data = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}
        image_path = Path(str(map_data.get("image", "")))
        if not image_path.is_absolute():
            image_path = map_path.parent / image_path
        if image_path.is_file():
            background = Image.open(image_path).convert("RGBA")
            origin = map_data.get("origin")
            resolution = float(map_data.get("resolution", 0.0) or 0.0)

    if background is not None and origin is not None and resolution > 0.0:
        image = background.resize(size)
        scale_x = size[0] / background.size[0]
        scale_y = size[1] / background.size[1]

        def to_px(point: tuple[float, float]) -> tuple[float, float]:
            px = (point[0] - float(origin[0])) / resolution * scale_x
            py = background.size[1] - (point[1] - float(origin[1])) / resolution
            return px, py * scale_y
    else:
        all_points = [point for track in [*gt_tracks, *scenario_tracks] for point in track.points_xy]
        if not all_points:
            all_points = [(0.0, 0.0), (1.0, 1.0)]
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        pad = max(1.0, 0.1 * max(max_x - min_x, max_y - min_y, 1.0))
        min_x -= pad
        max_x += pad
        min_y -= pad
        max_y += pad
        image = Image.new("RGBA", size, (245, 245, 245, 255))

        def to_px(point: tuple[float, float]) -> tuple[float, float]:
            x = (point[0] - min_x) / (max_x - min_x) * (size[0] - 40) + 20
            y = size[1] - ((point[1] - min_y) / (max_y - min_y) * (size[1] - 40) + 20)
            return x, y

    draw = ImageDraw.Draw(image, "RGBA")
    gt_colors = [(220, 20, 60, 255), (0, 120, 220, 255), (30, 150, 80, 255), (160, 80, 200, 255)]
    scenario_color = (255, 170, 0, 220)
    for index, track in enumerate(gt_tracks):
        points = [to_px(point) for point in track.points_xy]
        if len(points) >= 2:
            draw.line(points, fill=gt_colors[index % len(gt_colors)], width=4)
        for point in points[:1]:
            _draw_dot(draw, point, gt_colors[index % len(gt_colors)], radius=6)
        for point in points[-1:]:
            _draw_dot(draw, point, (0, 0, 0, 255), radius=5)
    for track in scenario_tracks:
        points = [to_px(point) for point in track.points_xy]
        if len(points) >= 2:
            draw.line(points, fill=scenario_color, width=2)
        for point in points:
            _draw_dot(draw, point, scenario_color, radius=3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_dot(draw: Any, point: tuple[float, float], fill: tuple[int, int, int, int], *, radius: int) -> None:
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _case_key_from_path(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"grscenes_\d+", part) and index + 1 < len(parts):
            scenario = parts[index + 1]
            if scenario != "scenarios":
                return part, scenario
            if index + 2 < len(parts):
                return part, parts[index + 2]
    return None


def _discover_cases(grscenes_root: Path, scenario_worlds_dir: Path) -> list[dict[str, Path | str]]:
    cases_by_key: dict[tuple[str, str], dict[str, Path | str]] = {}
    fallback_cases = []
    for params_path in sorted(grscenes_root.glob("grscenes_*/*/*/episode_*/data/params.parquet")):
        key = _case_key_from_path(params_path)
        if key is None:
            continue
        world, scenario = key
        scenario_path = scenario_worlds_dir / world / "scenarios" / scenario / "scenario.yaml"
        if not scenario_path.is_file():
            continue
        map_yaml = scenario_worlds_dir / world / "map" / "map.yaml"
        case = {
            "world": world,
            "scenario": scenario,
            "params_path": params_path,
            "scenario_path": scenario_path,
            "map_yaml": map_yaml,
        }
        bound = _bound_gt_params_path(scenario_path)
        if bound is not None:
            case["params_path"] = bound
            cases_by_key[(world, scenario)] = case
        else:
            fallback_cases.append(case)
    for case in fallback_cases:
        key = (str(case["world"]), str(case["scenario"]))
        cases_by_key.setdefault(key, case)
    return [cases_by_key[key] for key in sorted(cases_by_key)]


def _bound_gt_params_path(scenario_path: str | Path) -> Path | None:
    try:
        data = yaml.safe_load(Path(scenario_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if not isinstance(metadata, dict):
        return None
    value = str(metadata.get("gt_params_path") or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def scenario_gt_params_path(
    *,
    world: str,
    scenario: str = "default",
    grscenes_root: str | Path = "/home/ubuntu/arena_jazzy_ws/grscenes_",
    scenario_worlds_dir: str | Path = "/home/ubuntu/arena_jazzy_ws/src/Arena/arena_simulation_setup/worlds",
) -> Path | None:
    """Resolve the params.parquet bound to a GRScenes world/scenario pair."""

    selected = _select_case_for_run(
        _discover_cases(Path(grscenes_root), Path(scenario_worlds_dir)),
        world,
        scenario,
    )
    if selected is None:
        return None
    path = Path(selected["params_path"])
    return path if path.is_file() else None


def _select_case_for_run(cases: list[dict[str, Path | str]], world: str, scenario: str) -> dict[str, Path | str] | None:
    matching = [
        case for case in cases
        if case["world"] == world and case["scenario"] == scenario
    ]
    if not matching:
        return None
    for case in matching:
        bound = _bound_gt_params_path(case["scenario_path"])
        if bound is not None:
            selected = dict(case)
            selected["params_path"] = bound
            return selected
    return matching[0]


def batch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-audit GRScenes scenario and optional eval human tracks.")
    parser.add_argument("--grscenes-root", default="/home/ubuntu/arena_jazzy_ws/grscenes_")
    parser.add_argument("--scenario-worlds-dir", default="/home/ubuntu/arena_jazzy_ws/src/Arena/arena_simulation_setup/worlds")
    parser.add_argument("--eval-output-root", help="Optional output tree containing run_manifest.yaml + human_states.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-dt-sec", type=float, default=0.1)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _discover_cases(Path(args.grscenes_root), Path(args.scenario_worlds_dir))
    summary_rows = []
    for case in cases:
        gt_tracks = load_tracks_from_parquet(case["params_path"], frame_dt_sec=args.frame_dt_sec)
        scenario_tracks = load_scenario_tracks(case["scenario_path"])
        report = compare_scenario_to_gt(scenario_tracks, gt_tracks)
        case_name = f"{case['world']}_{case['scenario']}"
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "scenario_gt_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.no_overlay:
            draw_overlay(
                gt_tracks=gt_tracks,
                scenario_tracks=scenario_tracks,
                map_yaml=case["map_yaml"],
                output_path=case_dir / "gt_overlay.png",
            )
        matched = [row for row in report["tracks"] if row.get("matched")]
        summary_rows.append(
            {
                "world": case["world"],
                "scenario": case["scenario"],
                "params_path": str(case["params_path"]),
                "scenario_path": str(case["scenario_path"]),
                "scenario_track_count": report["scenario_track_count"],
                "gt_track_count": report["gt_track_count"],
                "max_start_error_m": max((row["start_error_m"] for row in matched), default=None),
                "max_end_error_m": max((row["end_error_m"] for row in matched), default=None),
                "max_scenario_point_to_gt_m": max((row["max_scenario_point_to_gt_m"] for row in matched), default=None),
            }
        )

    if args.eval_output_root:
        eval_root = Path(args.eval_output_root)
        for human_states_path in sorted(eval_root.glob("**/human_states.csv")):
            manifest_path = human_states_path.parent / "run_manifest.yaml"
            try:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            except Exception:
                manifest = {}
            params = manifest.get("parameters", {}) if isinstance(manifest, dict) else {}
            world = str(params.get("world") or "")
            scenario = str(params.get("scenario_file") or "")
            selected_case = _select_case_for_run(cases, world, scenario)
            if selected_case is None:
                continue
            gt_tracks = load_tracks_from_parquet(selected_case["params_path"], frame_dt_sec=args.frame_dt_sec)
            executed_tracks = load_human_states_csv(human_states_path)
            report = compare_executed_to_gt(executed_tracks, gt_tracks)
            report["gt_params_path"] = str(selected_case["params_path"])
            run_name = human_states_path.parent.name
            run_dir = output_dir / "eval_runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "human_states_gt_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    (output_dir / "summary.json").write_text(json.dumps({"cases": summary_rows}, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "world",
            "scenario",
            "scenario_track_count",
            "gt_track_count",
            "max_start_error_m",
            "max_end_error_m",
            "max_scenario_point_to_gt_m",
            "params_path",
            "scenario_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract and visualize GRScenes pedestrian GT tracks.")
    parser.add_argument("--params", required=True, help="Path to GRScenes data/params.parquet")
    parser.add_argument("--scenario", help="Optional Arena scenario.yaml to compare/overlay")
    parser.add_argument("--map-yaml", help="Optional map.yaml used as overlay background")
    parser.add_argument("--output-dir", required=True, help="Directory for gt_tracks.json/csv and overlay.png")
    parser.add_argument("--frame-dt-sec", type=float, default=0.1)
    parser.add_argument("--write-scenario", help="Optional output scenario.yaml with dynamic tracks rebuilt from GT")
    parser.add_argument("--sample-spacing-m", type=float, default=0.5)
    parser.add_argument("--max-waypoints", type=int, default=64)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_tracks = load_tracks_from_parquet(args.params, frame_dt_sec=args.frame_dt_sec)
    scenario_tracks = load_scenario_tracks(args.scenario) if args.scenario else []
    write_tracks_json(gt_tracks, output_dir / "gt_pedestrian_tracks.json")
    write_tracks_csv(gt_tracks, output_dir / "gt_pedestrian_tracks.csv")
    if scenario_tracks:
        report = compare_scenario_to_gt(scenario_tracks, gt_tracks)
        (output_dir / "scenario_gt_comparison.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    if args.write_scenario:
        if not args.scenario:
            raise ValueError("--write-scenario requires --scenario")
        write_gt_scenario(
            args.scenario,
            gt_tracks,
            args.write_scenario,
            params_path=args.params,
            min_spacing_m=args.sample_spacing_m,
            max_waypoints=args.max_waypoints,
        )
    if not args.no_overlay:
        draw_overlay(
            gt_tracks=gt_tracks,
            scenario_tracks=scenario_tracks,
            map_yaml=args.map_yaml,
            output_path=output_dir / "gt_overlay.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
