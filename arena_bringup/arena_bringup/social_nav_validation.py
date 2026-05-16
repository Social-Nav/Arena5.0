"""Acceptance validation for Arena social-navigation eval artifacts."""

from __future__ import annotations

import ast
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


REQUIRED_ENVIRONMENT = {
    "world": "hospital_1",
    "robot": "Ai2_Bot2",
    "human": "hunav",
    "tm_obstacles": "scenario",
}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _parse_value(value: str) -> Any:
    value = str(value or '').strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def _check_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    params = manifest.get('parameters', {}) if isinstance(manifest, dict) else {}
    mismatches = {}
    for key, expected in REQUIRED_ENVIRONMENT.items():
        actual = params.get(key)
        if str(actual) != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    return {
        "pass": not mismatches,
        "required": REQUIRED_ENVIRONMENT,
        "mismatches": mismatches,
    }


def _human_rows(run_dir: Path) -> tuple[int, int, int]:
    rows = _read_csv(run_dir / 'human_states.csv')
    nonempty = 0
    max_humans = 0
    for row in rows:
        data = _parse_value(row.get('data', ''))
        if isinstance(data, str):
            data = _parse_value(data)
        if isinstance(data, list) and data:
            nonempty += 1
            max_humans = max(max_humans, len(data))
    return len(rows), nonempty, max_humans


def _check_humans(run_dir: Path, social_metrics: dict[str, Any] | None) -> dict[str, Any]:
    rows, nonempty, max_humans = _human_rows(run_dir)
    metrics_present = bool(social_metrics and social_metrics.get('humans_present'))
    passed = max_humans > 0 and nonempty > 0 and metrics_present
    return {
        "pass": passed,
        "human_states_csv_present": (run_dir / 'human_states.csv').exists(),
        "human_states_rows": rows,
        "human_states_nonempty_rows": nonempty,
        "max_humans_observed": max_humans,
        "social_metrics_humans_present": metrics_present,
    }


def _episode_videos(video_index: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(video_index, dict):
        return {}
    episodes = video_index.get('episodes') or []
    return episodes[0] if episodes else {}


def _check_videos(run_dir: Path, video_index: dict[str, Any] | None) -> dict[str, Any]:
    ep = _episode_videos(video_index)
    required = {
        "ego_observation": ("ego_video", "ego_frames"),
        "ego_debug_overlay": ("debug_overlay_video", "debug_overlay_frames"),
        "sim_top_down": ("sim_top_down_video", "sim_top_down_frames"),
        "map_top_down_follow": ("map_top_down_video", "top_down_frames"),
    }
    results = {}
    for label, (path_key, frame_key) in required.items():
        path = Path(str(ep.get(path_key) or '')) if ep else Path('')
        if path and not path.is_absolute():
            path = run_dir / path
        frames = int(ep.get(frame_key) or 0) if ep else 0
        codec = ep.get(f'{path_key}_codec_detected') or ep.get(f'{label}_codec_detected')
        results[label] = {
            "path": str(path) if str(path) != '.' else '',
            "exists": bool(path and path.exists()),
            "frames": frames,
            "codec": codec,
            "pass": bool(path and path.exists() and frames > 0),
        }
    return {
        "pass": all(item["pass"] for item in results.values()),
        "video_index_present": isinstance(video_index, dict),
        "videos": results,
    }


def _trace_events(trace_path: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    if not trace_path.exists():
        return 0, counts
    for line in trace_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        total += 1
        event_type = str(record.get('event_type') or '')
        counts[event_type] = counts.get(event_type, 0) + 1
    return total, counts


def _odom_teleports(run_dir: Path, threshold_m: float = 5.0) -> list[dict[str, Any]]:
    positions = []
    for row in _read_csv(run_dir / 'odom.csv'):
        data = _parse_value(row.get('data', ''))
        position = data.get('position') if isinstance(data, dict) else None
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            positions.append((float(position[0]), float(position[1])))
    teleports = []
    for idx in range(1, len(positions)):
        distance = math.hypot(positions[idx][0] - positions[idx - 1][0], positions[idx][1] - positions[idx - 1][1])
        if distance > threshold_m:
            teleports.append({"index": idx, "distance_m": distance})
    return teleports


def _check_model_control(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get('artifacts', {}) if isinstance(manifest, dict) else {}
    trace_path = Path(str(artifacts.get('internnav_trace_path') or run_dir / 'internnav_trace.jsonl'))
    status_path = Path(str(artifacts.get('dual_vln_status_path') or run_dir / 'internnav_status.json'))
    total, event_counts = _trace_events(trace_path)
    teleports = _odom_teleports(run_dir)
    status = _read_json(status_path)
    model_results = event_counts.get('model_result', 0)
    return {
        "pass": trace_path.exists() and model_results > 0 and status is not None and not teleports,
        "trace_present": trace_path.exists(),
        "trace_record_count": total,
        "trace_event_counts": event_counts,
        "model_result_count": model_results,
        "status_present": status is not None,
        "status": status,
        "odom_present": bool(_read_csv(run_dir / 'odom.csv')),
        "large_teleports": teleports,
    }


def _check_metrics(run_dir: Path, social_metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics_path = run_dir / 'metrics.csv'
    social_path = run_dir / 'social_metrics.json'
    social_present = isinstance(social_metrics, dict)
    base_metrics = social_metrics.get('base_metrics') if social_present else {}
    base_first = base_metrics.get('first') if isinstance(base_metrics, dict) else {}
    episode_result = str(base_first.get('result') or '') if isinstance(base_first, dict) else ''
    task_success = episode_result == 'GOAL_REACHED'
    try:
        path_length_m = float(social_metrics.get('path_length_m') or 0.0) if social_present else 0.0
    except Exception:
        path_length_m = 0.0
    return {
        "pass": (
            metrics_path.exists()
            and social_present
            and bool(social_metrics.get('humans_present'))
            and bool(social_metrics.get('social_success'))
        ),
        "metrics_csv_present": metrics_path.exists(),
        "social_metrics_present": social_path.exists(),
        "social_success": social_metrics.get('social_success') if social_present else None,
        "task_success": task_success,
        "episode_result": episode_result or None,
        "path_length_m": path_length_m,
        "required_social_fields_present": all(
            key in social_metrics for key in (
                'min_human_distance_m',
                'personal_space_violation_time_sec',
                'near_miss_count',
                'human_collision_count',
                'crowd_freezing_time_sec',
                'social_success',
            )
        ) if social_present else False,
    }


def _frame_analysis(run_dir: Path) -> dict[str, Any]:
    analysis = _read_json(run_dir / 'frame_analysis' / 'video_frame_analysis.json')
    if analysis is None:
        return {"present": False, "warning": "frame analysis not found; manual video review required"}
    return {"present": True, "videos": analysis}


def generate_artifact_validation(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    run_path = Path(run_dir)
    manifest = _read_yaml(run_path / 'run_manifest.yaml') or {}
    social_metrics = _read_json(run_path / 'social_metrics.json')
    video_index = _read_json(run_path / 'video_index.json')

    checks = {
        "environment": _check_environment(manifest),
        "humans": _check_humans(run_path, social_metrics),
        "model_control": _check_model_control(run_path, manifest),
        "videos": _check_videos(run_path, video_index),
        "metrics": _check_metrics(run_path, social_metrics),
    }
    warnings = []
    frame_analysis = _frame_analysis(run_path)
    if not frame_analysis.get('present'):
        warnings.append(frame_analysis.get('warning'))

    failed = [name for name, result in checks.items() if not result.get('pass')]
    report = {
        "schema_version": 1,
        "run_dir": str(run_path),
        "overall_pass": not failed,
        "social_nav_ready": not failed,
        "failed_checks": failed,
        "warnings": [warning for warning in warnings if warning],
        "checks": checks,
        "frame_analysis": frame_analysis,
    }
    (run_path / 'artifact_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate Arena social-navigation eval artifacts.')
    parser.add_argument('--dir', required=True, help='Eval run directory')
    args = parser.parse_args()
    report = generate_artifact_validation(args.dir)
    return 0 if report.get('overall_pass') else 1


if __name__ == '__main__':
    raise SystemExit(main())
