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
    "human": "hunav",
    "tm_obstacles": "scenario",
}

REQUIRED_DYNAMIC_SCENE_FIELDS = (
    'moving_human_count',
    'human_motion_total_m',
    'human_motion_time_sec',
    'robot_motion_time_sec',
    'human_robot_motion_overlap_time_sec',
    'human_robot_interaction_time_sec',
    'dynamic_scene_success',
)

DIRECT_CLIENT_CONTROL_EVENTS = {
    "planning_response_received",
    "trajectory",
    "discrete_action",
    "stop",
    "unknown_response",
    "internnav_command",
    "control_tick",
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
    expectations = params.get('social_eval_expectations') if isinstance(params.get('social_eval_expectations'), dict) else {}
    required = {
        **REQUIRED_ENVIRONMENT,
        **{
            key: expectations[key]
            for key in ('world', 'robot')
            if expectations.get(key)
        },
    }
    mismatches = {}
    for key, expected in required.items():
        actual = params.get(key)
        if str(actual) != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    return {
        "pass": not mismatches,
        "required": required,
        "mismatches": mismatches,
    }


def _human_csv_path(run_dir: Path) -> Path:
    human_states = run_dir / 'human_states.csv'
    if human_states.exists():
        return human_states
    return run_dir / 'pedsim_agents_data.csv'


def _human_rows(run_dir: Path) -> tuple[Path, int, int, int]:
    path = _human_csv_path(run_dir)
    rows = _read_csv(path)
    nonempty = 0
    max_humans = 0
    for row in rows:
        data = _parse_value(row.get('data', ''))
        if isinstance(data, str):
            data = _parse_value(data)
        if isinstance(data, list) and data:
            nonempty += 1
            max_humans = max(max_humans, len(data))
    return path, len(rows), nonempty, max_humans


def _check_humans(run_dir: Path, social_metrics: dict[str, Any] | None) -> dict[str, Any]:
    path, rows, nonempty, max_humans = _human_rows(run_dir)
    metrics_present = bool(social_metrics and social_metrics.get('humans_present'))
    passed = max_humans > 0 and nonempty > 0 and metrics_present
    return {
        "pass": passed,
        "human_states_csv_present": (run_dir / 'human_states.csv').exists(),
        "pedsim_agents_data_csv_present": (run_dir / 'pedsim_agents_data.csv').exists(),
        "human_source_csv": path.name,
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
        if label == 'ego_debug_overlay':
            results[label]["fallback"] = bool(ep.get('debug_overlay_fallback')) if ep else False
            source = ep.get('debug_overlay_source') if isinstance(ep, dict) else None
            results[label]["source"] = source if isinstance(source, dict) else {}
            source_status = results[label]["source"].get("status")
            if source_status:
                results[label]["source_status"] = source_status
        if not results[label]["pass"]:
            if label == 'ego_debug_overlay':
                results[label]["diagnostic"] = "debug_overlay_missing_or_empty"
            elif not results[label]["exists"]:
                results[label]["diagnostic"] = "video_missing"
            else:
                results[label]["diagnostic"] = "video_empty"
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
        event_type = str(record.get('event_type') or record.get('event') or '')
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
    result = manifest.get('result', {}) if isinstance(manifest, dict) else {}
    params = manifest.get('parameters', {}) if isinstance(manifest, dict) else {}
    trace_path = Path(str(artifacts.get('internnav_trace_path') or run_dir / 'internnav_trace.jsonl'))
    status_path = Path(str(artifacts.get('dual_vln_status_path') or run_dir / 'internnav_status.json'))
    total, event_counts = _trace_events(trace_path)
    teleports = _odom_teleports(run_dir)
    status = _read_json(status_path)
    direct_cmd_vel = bool(params.get('internnav_direct_cmd_vel') or params.get('dual_vln_direct_cmd_vel'))
    model_results = event_counts.get('model_result', 0)
    direct_control_events = sum(event_counts.get(event, 0) for event in DIRECT_CLIENT_CONTROL_EVENTS)
    has_trace_evidence = model_results > 0 or direct_control_events > 0
    status_required = not direct_cmd_vel
    missing_model_control_loop = (
        not trace_path.exists()
        or not has_trace_evidence
        or (status_required and status is None)
    )
    return {
        "pass": trace_path.exists() and has_trace_evidence and (status is not None or not status_required) and not teleports,
        "trace_present": trace_path.exists(),
        "trace_record_count": total,
        "trace_event_counts": event_counts,
        "model_result_count": model_results,
        "direct_control_event_count": direct_control_events,
        "status_present": status is not None,
        "status_required": status_required,
        "status": status,
        "odom_present": bool(_read_csv(run_dir / 'odom.csv')),
        "large_teleports": teleports,
        "missing_model_control_loop": missing_model_control_loop,
        "external_server": bool(params.get('internnav_external_server')),
        "direct_cmd_vel": direct_cmd_vel,
        "external_server_preflight": result.get('external_server_preflight'),
    }


def _check_metrics(run_dir: Path, social_metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics_path = run_dir / 'metrics.csv'
    vln_task_metrics_path = run_dir / 'vln_task_metrics.json'
    social_path = run_dir / 'social_metrics.json'
    social_present = isinstance(social_metrics, dict)
    vln_task_metrics = _read_json(vln_task_metrics_path)
    strict_task_present = isinstance(vln_task_metrics, dict)
    base_metrics = social_metrics.get('base_metrics') if social_present else {}
    base_first = base_metrics.get('first') if isinstance(base_metrics, dict) else {}
    episode_result = str(base_first.get('result') or '') if isinstance(base_first, dict) else ''
    strict_task_success = bool(vln_task_metrics.get('strict_task_success')) if strict_task_present else False
    strict_social_success = bool(social_metrics.get('strict_social_success', social_metrics.get('social_success'))) if social_present else False
    try:
        path_length_m = float(social_metrics.get('path_length_m') or 0.0) if social_present else 0.0
    except Exception:
        path_length_m = 0.0
    is_dual_vln = False
    try:
        manifest = _read_yaml(run_dir / 'run_manifest.yaml') or {}
        params = manifest.get('parameters', {}) if isinstance(manifest, dict) else {}
        is_dual_vln = str(params.get('local_planner', '')).lower() == 'dual_vln'
    except Exception:
        is_dual_vln = False
    robot_moved = (not is_dual_vln) or path_length_m >= 0.1
    return {
        "pass": (
            metrics_path.exists()
            and strict_task_present
            and social_present
            and bool(social_metrics.get('humans_present'))
            and strict_task_success
            and strict_social_success
            and robot_moved
        ),
        "metrics_csv_present": metrics_path.exists(),
        "vln_task_metrics_present": strict_task_present,
        "social_metrics_present": social_path.exists(),
        "social_success": strict_social_success if social_present else None,
        "strict_social_success": social_metrics.get('strict_social_success') if social_present else None,
        "strict_task_success": strict_task_success if strict_task_present else None,
        "strict_task_failure_reasons": vln_task_metrics.get('strict_task_failure_reasons') if strict_task_present else [],
        "strict_social_failure_reasons": social_metrics.get('strict_social_failure_reasons') if social_present else [],
        "task_success": strict_task_success if strict_task_present else None,
        "episode_result": episode_result or None,
        "path_length_m": path_length_m,
        "robot_moved": robot_moved,
        "required_social_fields_present": all(
            key in social_metrics for key in (
                'min_human_distance_m',
                'personal_space_violation_time_sec',
                'near_miss_count',
                'human_collision_count',
                'crowd_freezing_time_sec',
                'social_success',
                'strict_social_success',
            )
        ) if social_present else False,
    }


def _check_dynamic_scene(social_metrics: dict[str, Any] | None) -> dict[str, Any]:
    social_present = isinstance(social_metrics, dict)
    thresholds = social_metrics.get('thresholds') if social_present and isinstance(social_metrics.get('thresholds'), dict) else {}

    def number(key: str) -> float:
        try:
            return float(social_metrics.get(key) or 0.0) if social_present else 0.0
        except Exception:
            return 0.0

    moving_human_count = int(number('moving_human_count'))
    min_moving_humans = int(float(thresholds.get('min_moving_human_count', 1) or 1))
    human_motion_time_sec = number('human_motion_time_sec')
    min_human_motion_time_sec = float(thresholds.get('min_human_motion_time_sec', 5.0) or 5.0)
    overlap_time_sec = number('human_robot_motion_overlap_time_sec')
    min_overlap_time_sec = float(thresholds.get('min_human_robot_motion_overlap_time_sec', 3.0) or 3.0)
    interaction_time_sec = number('human_robot_interaction_time_sec')
    min_interaction_time_sec = float(thresholds.get('min_human_robot_interaction_time_sec', 1.0) or 1.0)
    dynamic_scene_success = bool(social_metrics.get('dynamic_scene_success')) if social_present else False

    failures: list[str] = []
    if not social_present:
        failures.append('social_metrics_missing')
    elif not all(field in social_metrics for field in REQUIRED_DYNAMIC_SCENE_FIELDS):
        failures.append('dynamic_scene_fields_missing')
    if moving_human_count < min_moving_humans:
        failures.append('moving_human_count_below_threshold')
    if human_motion_time_sec < min_human_motion_time_sec:
        failures.append('human_motion_time_below_threshold')
    if overlap_time_sec < min_overlap_time_sec:
        failures.append('human_robot_motion_overlap_below_threshold')
    if interaction_time_sec < min_interaction_time_sec:
        failures.append('human_robot_interaction_time_below_threshold')
    if social_present and not dynamic_scene_success:
        failures.append('dynamic_scene_success_false')

    return {
        "pass": not failures,
        "failures": failures,
        "required_fields_present": all(field in social_metrics for field in REQUIRED_DYNAMIC_SCENE_FIELDS) if social_present else False,
        "moving_human_count": moving_human_count,
        "min_moving_human_count": min_moving_humans,
        "human_motion_total_m": number('human_motion_total_m'),
        "human_motion_time_sec": human_motion_time_sec,
        "min_human_motion_time_sec": min_human_motion_time_sec,
        "robot_motion_time_sec": number('robot_motion_time_sec'),
        "human_robot_motion_overlap_time_sec": overlap_time_sec,
        "min_human_robot_motion_overlap_time_sec": min_overlap_time_sec,
        "human_robot_interaction_time_sec": interaction_time_sec,
        "min_human_robot_interaction_time_sec": min_interaction_time_sec,
        "dynamic_scene_success": dynamic_scene_success,
    }


def _frame_analysis(run_dir: Path) -> dict[str, Any]:
    analysis = _read_json(run_dir / 'frame_analysis' / 'video_frame_analysis.json')
    if analysis is None:
        return {"present": False, "warning": "frame analysis not found; manual video review required"}
    return {"present": True, "videos": analysis}


def _diagnostic_warnings(checks: dict[str, Any], manifest: dict[str, Any], social_metrics: dict[str, Any] | None) -> list[str]:
    warnings: list[str] = []
    params = manifest.get('parameters', {}) if isinstance(manifest, dict) else {}
    if isinstance(social_metrics, dict):
        try:
            path_length_m = float(social_metrics.get('path_length_m') or 0.0)
        except Exception:
            path_length_m = 0.0
        if str(params.get('local_planner', '')).lower() == 'dual_vln' and path_length_m < 0.1:
            warnings.append(f'robot appears stationary: path_length_m={path_length_m:.3f}')
        legacy_social_success = social_metrics.get('legacy_social_success', social_metrics.get('social_success'))
        if legacy_social_success and not social_metrics.get('strict_social_success'):
            warnings.append('legacy social_success is true but strict_social_success is false')

    strict_task_metrics = _read_json(Path(str((manifest.get('artifacts') or {}).get('vln_task_metrics_path') or '')))
    if not isinstance(strict_task_metrics, dict):
        strict_task_metrics = _read_json(Path(str(manifest.get('result_dir_absolute') or '')) / 'vln_task_metrics.json')
    if isinstance(strict_task_metrics, dict):
        legacy_first = ((social_metrics or {}).get('base_metrics') or {}).get('first') if isinstance(social_metrics, dict) else {}
        if isinstance(legacy_first, dict) and legacy_first.get('result') == 'GOAL_REACHED' and not strict_task_metrics.get('strict_task_success'):
            warnings.append('legacy GOAL_REACHED is true but strict_task_success is false')
        task_contract = strict_task_metrics.get('language_task_contract')
        if isinstance(task_contract, dict) and task_contract.get('unsupported_predicates_present'):
            unsupported = task_contract.get('unsupported_predicates')
            if isinstance(unsupported, list) and unsupported:
                warnings.append('language task contract uses native scenario goal; unsupported predicates: ' + ', '.join(str(item) for item in unsupported))
            else:
                warnings.append('language task contract uses native scenario goal; richer semantic predicates are not scored')

    model_control = checks.get('model_control') if isinstance(checks.get('model_control'), dict) else {}
    if model_control.get('missing_model_control_loop'):
        warnings.append('model-control loop missing: InternNav trace/status artifacts are incomplete')

    videos = checks.get('videos') if isinstance(checks.get('videos'), dict) else {}
    overlay = (videos.get('videos') or {}).get('ego_debug_overlay') if isinstance(videos.get('videos'), dict) else {}
    if isinstance(overlay, dict) and overlay.get('diagnostic') == 'debug_overlay_missing_or_empty':
        warnings.append('debug overlay video missing or empty')
    if isinstance(overlay, dict) and overlay.get('fallback'):
        source = overlay.get('source') if isinstance(overlay.get('source'), dict) else {}
        source_status = source.get('status') or 'unknown'
        warnings.append(f'debug overlay uses ego-camera fallback; model debug image source={source_status}')
    return warnings


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
        "dynamic_scene": _check_dynamic_scene(social_metrics),
    }
    warnings = []
    frame_analysis = _frame_analysis(run_path)
    if not frame_analysis.get('present'):
        warnings.append(frame_analysis.get('warning'))
    warnings.extend(_diagnostic_warnings(checks, manifest, social_metrics))

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
