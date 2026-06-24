"""Generate an offline benchmark failure review report for one Arena run."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _as_float(value[0])
    y = _as_float(value[1])
    if x is None or y is None:
        return None
    return x, y


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _video_paths(run_dir: Path, video_index: dict[str, Any]) -> dict[str, Any]:
    episodes = video_index.get("episodes") if isinstance(video_index, dict) else None
    first = episodes[0] if isinstance(episodes, list) and episodes and isinstance(episodes[0], dict) else {}
    videos = {
        "ego_observation": first.get("ego_video"),
        "ego_debug_overlay": first.get("debug_overlay_video"),
        "map_top_down_follow": first.get("map_top_down_video") or first.get("top_down_video"),
        "sim_top_down": first.get("sim_top_down_video"),
    }
    host_videos = {
        key: str(host_path)
        for key, value in videos.items()
        if (host_path := _host_equivalent(value)) is not None
    }
    return {
        **videos,
        "host_equivalent": host_videos,
        "debug_overlay_fallback": bool(first.get("debug_overlay_fallback")),
        "debug_overlay_source": first.get("debug_overlay_source", {}) if isinstance(first.get("debug_overlay_source"), dict) else {},
        "frames": {
            "ego": _as_int(first.get("ego_frames")),
            "debug_overlay": _as_int(first.get("debug_overlay_frames")),
            "map_top_down": _as_int(first.get("top_down_frames")),
            "sim_top_down": _as_int(first.get("sim_top_down_frames")),
        },
        "video_index": str(run_dir / "video_index.json"),
    }


def _host_equivalent(path_value: Any) -> Path | None:
    if not path_value:
        return None
    text = str(path_value)
    replacements = [
        ("/opt/arena_ws/src/Arena", "/home/ubuntu/arena_jazzy_ws/src/Arena"),
        ("/opt/arena_ws/install/arena_simulation_setup/share/arena_simulation_setup", "/home/ubuntu/arena_jazzy_ws/src/Arena/arena_simulation_setup"),
    ]
    for old, new in replacements:
        if text.startswith(old):
            return Path(new + text[len(old):])
    return None


def _resolve_existing_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.exists():
        return path
    host_path = _host_equivalent(path_value)
    if host_path and host_path.exists():
        return host_path
    return path


class _MapReview:
    def __init__(self, static_occupancy: dict[str, Any]) -> None:
        self.map_yaml = _resolve_existing_path(static_occupancy.get("map_yaml"))
        metadata = _read_yaml(self.map_yaml) if self.map_yaml else {}
        self.metadata = metadata if isinstance(metadata, dict) else {}
        image_name = self.metadata.get("image")
        self.image_path = self.map_yaml.parent / str(image_name) if self.map_yaml and image_name else _resolve_existing_path(static_occupancy.get("image"))
        self.resolution = _as_float(self.metadata.get("resolution"), _as_float(static_occupancy.get("resolution"), 0.0)) or 0.0
        origin = self.metadata.get("origin") or static_occupancy.get("origin") or [0.0, 0.0, 0.0]
        self.origin_x = _as_float(origin[0], 0.0) if isinstance(origin, list) and origin else 0.0
        self.origin_y = _as_float(origin[1], 0.0) if isinstance(origin, list) and len(origin) > 1 else 0.0
        self.negate = _as_int(self.metadata.get("negate"), 0)
        self.occupied_thresh = _as_float(self.metadata.get("occupied_thresh"), 0.65) or 0.65
        self.width = 0
        self.height = 0
        self.available = bool(self.image_path and self.image_path.exists() and self.resolution > 0)
        self._occupied: set[tuple[int, int]] = set()
        if self.available and self.image_path is not None:
            self._load(self.image_path)

    def _load(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("L")
        self.width, self.height = image.size
        pix = image.load()
        occupied: set[tuple[int, int]] = set()
        for y in range(self.height):
            for x in range(self.width):
                value = pix[x, y] / 255.0
                occ = value if self.negate else 1.0 - value
                if occ >= self.occupied_thresh:
                    occupied.add((x, y))
        self._occupied = occupied

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int] | None:
        if not self.available:
            return None
        px = int(math.floor((x - self.origin_x) / self.resolution))
        py_from_bottom = int(math.floor((y - self.origin_y) / self.resolution))
        return px, self.height - 1 - py_from_bottom

    def nearest_occupied(self, center_px: tuple[int, int], search_radius_px: int) -> dict[str, Any] | None:
        if not self.available:
            return None
        cx, cy = center_px
        best: tuple[float, tuple[int, int]] | None = None
        for oy in range(-search_radius_px, search_radius_px + 1):
            for ox in range(-search_radius_px, search_radius_px + 1):
                candidate = (cx + ox, cy + oy)
                if candidate not in self._occupied:
                    continue
                dist_px = math.hypot(ox, oy)
                if best is None or dist_px < best[0]:
                    best = (dist_px, candidate)
        if best is None:
            return None
        dist_px, pixel = best
        return {
            "pixel": [pixel[0], pixel[1]],
            "distance_px": dist_px,
            "distance_m": dist_px * self.resolution,
        }


def _map_evidence(vln_task: dict[str, Any]) -> dict[str, Any]:
    static = vln_task.get("static_occupancy") if isinstance(vln_task, dict) else {}
    if not isinstance(static, dict):
        return {}
    samples = static.get("collision_samples")
    first_sample = samples[0] if isinstance(samples, list) and samples and isinstance(samples[0], dict) else None
    review = _MapReview(static)
    robot_radius_m = _as_float((vln_task.get("thresholds") or {}).get("robot_radius_m"), 0.30) if isinstance(vln_task.get("thresholds"), dict) else 0.30
    center_px = None
    nearest = None
    clearance_m = None
    if first_sample:
        pos = _xy(first_sample.get("position"))
        if pos:
            center_px = review.world_to_pixel(pos[0], pos[1])
            if center_px is not None:
                search_px = max(1, int(math.ceil((float(robot_radius_m or 0.30) + 0.50) / review.resolution))) if review.resolution > 0 else 0
                nearest = review.nearest_occupied(center_px, search_px)
                if nearest and nearest.get("distance_m") is not None:
                    clearance_m = float(nearest["distance_m"]) - float(robot_radius_m or 0.30)
    return {
        "map_available": bool(static.get("map_available")),
        "map_yaml": static.get("map_yaml"),
        "host_map_yaml": str(review.map_yaml) if review.map_yaml else None,
        "host_equivalent_map_yaml": str(_host_equivalent(static.get("map_yaml"))) if _host_equivalent(static.get("map_yaml")) else None,
        "image": static.get("image"),
        "host_image": str(review.image_path) if review.image_path else None,
        "host_equivalent_image": str(_host_equivalent(static.get("image"))) if _host_equivalent(static.get("image")) else None,
        "resolution": static.get("resolution"),
        "origin": static.get("origin"),
        "robot_radius_m": robot_radius_m,
        "collision_sample_count": static.get("collision_sample_count", 0),
        "first_collision_sample": first_sample,
        "first_collision_center_pixel": list(center_px) if center_px else None,
        "nearest_occupied_to_first_collision": nearest,
        "estimated_static_clearance_m": clearance_m,
        "intervals": static.get("intervals", []),
    }


def _classification(vln_task: dict[str, Any], social: dict[str, Any], validation: dict[str, Any], video: dict[str, Any], map_review: dict[str, Any]) -> dict[str, Any]:
    task_failures = vln_task.get("strict_task_failure_reasons", []) if isinstance(vln_task, dict) else []
    social_failures = social.get("strict_social_failure_reasons", []) if isinstance(social, dict) else []
    tags: list[str] = []
    notes: list[str] = []

    if "dynamic_scene_failed" in social_failures or not social.get("dynamic_scene_success", False):
        tags.append("instrumentation_or_hunav_failure")
        notes.append("Dynamic scene did not meet HuNav motion thresholds.")
    elif social.get("moving_human_count", 0):
        tags.append("dynamic_scene_valid")
        notes.append("HuNav motion is present; pedestrian absence is not the root cause.")

    if "static_occupancy_collision" in task_failures:
        clearance = _as_float(map_review.get("estimated_static_clearance_m"))
        if clearance is not None and clearance < 0.0:
            tags.append("static_map_footprint_overlap")
            notes.append("Robot footprint overlaps occupied map cells; review whether this is real obstacle contact or map/asset registration.")
        else:
            tags.append("static_occupancy_requires_review")
            notes.append("Static occupancy collision was reported but map clearance estimate is inconclusive.")

    if "commanded_stuck" in task_failures:
        tags.append("commanded_stuck_behavior")
        notes.append("Commands requested motion while odom progress was below threshold.")

    if "goal_not_reached" in task_failures:
        tags.append("task_goal_not_reached")
        notes.append("Final pose remained outside the native scenario goal tolerance.")

    if any(reason in social_failures for reason in ("footprint_human_collision", "footprint_near_miss")):
        tags.append("social_footprint_safety_failure")
        notes.append("Footprint-aware robot-human clearance violated strict social thresholds.")

    if video.get("debug_overlay_fallback"):
        tags.append("debug_overlay_fallback")
        source = video.get("debug_overlay_source", {}) if isinstance(video.get("debug_overlay_source"), dict) else {}
        status = source.get("status") or "unknown"
        if status:
            tags.append(f"debug_overlay_source_{status}")
        notes.append(f"Debug overlay video used fallback imagery; model debug image source={status}.")

    failed_checks = validation.get("failed_checks", []) if isinstance(validation, dict) else []
    if failed_checks:
        tags.append("artifact_validation_failed")

    return {
        "primary_classification": tags[0] if tags else "no_strict_failure_detected",
        "tags": _dedupe(tags),
        "notes": notes,
    }


def generate_report(run_dir: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    manifest = _read_yaml(run_path / "run_manifest.yaml") or {}
    vln_task = _read_json(run_path / "vln_task_metrics.json") or {}
    social = _read_json(run_path / "social_metrics.json") or {}
    validation = _read_json(run_path / "artifact_validation.json") or {}
    diagnostics = _read_json(run_path / "internnav_diagnostic_summary.json") or {}
    video_index = _read_json(run_path / "video_index.json") or {}

    params = manifest.get("parameters", {}) if isinstance(manifest, dict) else {}
    result = manifest.get("result", {}) if isinstance(manifest, dict) else {}
    video = _video_paths(run_path, video_index if isinstance(video_index, dict) else {})
    map_review = _map_evidence(vln_task if isinstance(vln_task, dict) else {})
    classification = _classification(
        vln_task if isinstance(vln_task, dict) else {},
        social if isinstance(social, dict) else {},
        validation if isinstance(validation, dict) else {},
        video,
        map_review,
    )
    min_footprint = social.get("min_footprint_clearance_sample") if isinstance(social, dict) else None
    report = {
        "schema_version": 1,
        "run_dir": str(run_path),
        "world": params.get("world") or vln_task.get("world"),
        "scenario_file": params.get("scenario_file") or vln_task.get("scenario"),
        "robot": params.get("robot"),
        "instruction": vln_task.get("instruction") or params.get("vln_instruction"),
        "run_result": {
            "finished_observed": result.get("finished_observed"),
            "end_reason": result.get("end_reason"),
            "launch_returncode": result.get("launch_returncode"),
            "artifact_validation_returncode": result.get("artifact_validation_returncode"),
        },
        "strict_task": {
            "success": vln_task.get("strict_task_success"),
            "failure_reasons": vln_task.get("strict_task_failure_reasons", []),
            "language_task_contract": vln_task.get("language_task_contract", {}),
            "goal": vln_task.get("goal", {}),
            "episode_timing": vln_task.get("episode_timing", {}),
            "commanded_stuck": vln_task.get("commanded_stuck", {}),
            "static_occupancy": map_review,
        },
        "strict_social": {
            "success": social.get("strict_social_success"),
            "failure_reasons": social.get("strict_social_failure_reasons", []),
            "dynamic_scene_success": social.get("dynamic_scene_success"),
            "moving_human_count": social.get("moving_human_count"),
            "human_motion_total_m": social.get("human_motion_total_m"),
            "min_human_distance_m": social.get("min_human_distance_m"),
            "min_footprint_clearance_m": social.get("min_footprint_clearance_m"),
            "min_footprint_clearance_sample": min_footprint,
            "footprint_near_miss_events": social.get("footprint_near_miss_events", []),
            "footprint_human_collision_events": social.get("footprint_human_collision_events", []),
            "review_intervals": social.get("review_intervals", {}),
        },
        "legacy_diagnostics": {
            "goal_reached_false_positive": vln_task.get("legacy_goal_reached_false_positive"),
        },
        "internnav": {
            "event_counts": diagnostics.get("event_counts", {}),
            "command_stats": diagnostics.get("command_stats", {}),
            "fault_candidates": diagnostics.get("fault_candidates", {}),
            "odom_goal_distance": diagnostics.get("odom_goal_distance", {}),
        },
        "videos": video,
        "validation": {
            "overall_pass": validation.get("overall_pass"),
            "social_nav_ready": validation.get("social_nav_ready"),
            "failed_checks": validation.get("failed_checks", []),
            "warnings": validation.get("warnings", []),
        },
        "classification": classification,
    }
    output_path = Path(output) if output else run_path / "benchmark_failure_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an offline benchmark failure review report.")
    parser.add_argument("--dir", required=True, help="Eval run directory")
    parser.add_argument("--output", help="Output JSON path; defaults to benchmark_failure_report.json in the run directory")
    args = parser.parse_args()
    report = generate_report(args.dir, args.output)
    print(json.dumps({"output": str(Path(args.output) if args.output else Path(args.dir) / "benchmark_failure_report.json"), "classification": report["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
