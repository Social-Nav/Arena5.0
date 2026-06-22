"""Aggregate Dynamic Social VLN eval outputs across run directories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SUMMARY_FIELDS = [
    'run_id',
    'run_dir',
    'scenario_id',
    'world',
    'robot',
    'planner',
    'human',
    'scenario_file',
    'episode_result',
    'task_success',
    'legacy_task_success',
    'strict_task_success',
    'social_success',
    'legacy_social_success',
    'strict_social_success',
    'artifact_validation_pass',
    'social_nav_ready',
    'benchmark_ready',
    'path_length_m',
    'navigation_error_m',
    'oracle_error_m',
    'spl',
    'ndtw',
    'robot_moved',
    'goal_progress_m',
    'min_human_distance_m',
    'min_footprint_clearance_m',
    'near_miss_count',
    'human_collision_count',
    'footprint_near_miss_count',
    'footprint_human_collision_count',
    'static_occupancy_collision_samples',
    'commanded_stuck_time_sec',
    'personal_space_violation_time_sec',
    'footprint_personal_space_violation_time_sec',
    'crowd_freezing_time_sec',
    'humans_present',
    'max_humans_observed',
    'stale_camera_count',
    'forward_count',
    'rotate_count',
    'stop_count',
    'strict_task_failure_reasons',
    'strict_social_failure_reasons',
    'validation_failed_checks',
    'validation_warnings',
    'primary_failure',
    'failure_tags',
]


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def read_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def discover_run_dirs(roots: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if (root / 'run_manifest.yaml').exists() or (root / 'social_metrics.json').exists():
            found.add(root.resolve())
            continue
        for marker in root.rglob('social_metrics.json'):
            found.add(marker.parent.resolve())
        for marker in root.rglob('artifact_validation.json'):
            found.add(marker.parent.resolve())
    return sorted(found)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_yaml(run_dir / 'run_manifest.yaml') or {}
    vln_task = read_json(run_dir / 'vln_task_metrics.json') or {}
    social = read_json(run_dir / 'social_metrics.json') or {}
    validation = read_json(run_dir / 'artifact_validation.json') or {}
    diagnostics = read_json(run_dir / 'internnav_diagnostic_summary.json') or {}

    params = manifest.get('parameters', {}) if isinstance(manifest, dict) else {}
    result = manifest.get('result', {}) if isinstance(manifest, dict) else {}
    checks = validation.get('checks', {}) if isinstance(validation, dict) else {}
    metrics_check = checks.get('metrics', {}) if isinstance(checks, dict) else {}

    base_metrics = social.get('base_metrics', {}) if isinstance(social, dict) else {}
    base_first = base_metrics.get('first', {}) if isinstance(base_metrics, dict) else {}
    episode_result = str(metrics_check.get('episode_result') or base_first.get('result') or '')
    legacy_task_success = bool(metrics_check.get('legacy_task_success')) if 'legacy_task_success' in metrics_check else episode_result == 'GOAL_REACHED'
    strict_task_success = bool(metrics_check.get('strict_task_success')) if 'strict_task_success' in metrics_check else _as_bool(vln_task.get('strict_task_success'))
    legacy_social_success = _as_bool(social.get('social_success')) if isinstance(social, dict) else False
    strict_social_success = _as_bool(social.get('strict_social_success')) if isinstance(social, dict) else False
    artifact_pass = _as_bool(validation.get('overall_pass')) if isinstance(validation, dict) else False
    social_nav_ready = _as_bool(validation.get('social_nav_ready')) if isinstance(validation, dict) else False

    command_stats = diagnostics.get('command_stats', {}) if isinstance(diagnostics, dict) else {}
    fault = diagnostics.get('fault_candidates', {}) if isinstance(diagnostics, dict) else {}
    stale_camera_count = fault.get('stale_record_count', 0) if isinstance(fault, dict) else 0
    goal_distance = diagnostics.get('goal_distance', {}) if isinstance(diagnostics, dict) else {}
    goal_progress = goal_distance.get('progress_first_minus_last') if isinstance(goal_distance, dict) else None
    goal_metrics = vln_task.get('goal', {}) if isinstance(vln_task, dict) else {}
    vln_metrics = vln_task.get('vln', {}) if isinstance(vln_task, dict) else {}
    static_occupancy = vln_task.get('static_occupancy', {}) if isinstance(vln_task, dict) else {}
    commanded_stuck = vln_task.get('commanded_stuck', {}) if isinstance(vln_task, dict) else {}
    strict_task_failures = vln_task.get('strict_task_failure_reasons', []) if isinstance(vln_task, dict) else []
    strict_social_failures = social.get('strict_social_failure_reasons', []) if isinstance(social, dict) else []
    failed_checks = validation.get('failed_checks', []) if isinstance(validation, dict) else []
    warnings = validation.get('warnings', []) if isinstance(validation, dict) else []
    benchmark_ready = strict_task_success and strict_social_success and artifact_pass and social_nav_ready

    row = {
        'run_id': run_dir.name,
        'run_dir': str(run_dir),
        'scenario_id': params.get('scenario_config_id') or params.get('output_prefix') or '',
        'world': params.get('world') or '',
        'robot': params.get('robot') or '',
        'planner': params.get('local_planner') or '',
        'human': params.get('human') or '',
        'scenario_file': params.get('scenario_file') or '',
        'episode_result': episode_result,
        'task_success': legacy_task_success,
        'legacy_task_success': legacy_task_success,
        'strict_task_success': strict_task_success,
        'social_success': legacy_social_success,
        'legacy_social_success': legacy_social_success,
        'strict_social_success': strict_social_success,
        'artifact_validation_pass': artifact_pass,
        'social_nav_ready': social_nav_ready,
        'benchmark_ready': benchmark_ready,
        'path_length_m': _float_or_none(social.get('path_length_m') if isinstance(social, dict) else None),
        'navigation_error_m': _float_or_none(goal_metrics.get('navigation_error_m') if isinstance(goal_metrics, dict) else None),
        'oracle_error_m': _float_or_none(goal_metrics.get('oracle_error_m') if isinstance(goal_metrics, dict) else None),
        'spl': _float_or_none(vln_metrics.get('spl') if isinstance(vln_metrics, dict) else None),
        'ndtw': _float_or_none(vln_metrics.get('ndtw') if isinstance(vln_metrics, dict) else None),
        'robot_moved': metrics_check.get('robot_moved'),
        'goal_progress_m': _float_or_none(goal_progress),
        'min_human_distance_m': _float_or_none(social.get('min_human_distance_m') if isinstance(social, dict) else None),
        'min_footprint_clearance_m': _float_or_none(social.get('min_footprint_clearance_m') if isinstance(social, dict) else None),
        'near_miss_count': _int_or_zero(social.get('near_miss_count') if isinstance(social, dict) else None),
        'human_collision_count': _int_or_zero(social.get('human_collision_count') if isinstance(social, dict) else None),
        'footprint_near_miss_count': _int_or_zero(social.get('footprint_near_miss_count') if isinstance(social, dict) else None),
        'footprint_human_collision_count': _int_or_zero(social.get('footprint_human_collision_count') if isinstance(social, dict) else None),
        'static_occupancy_collision_samples': _int_or_zero(static_occupancy.get('collision_sample_count') if isinstance(static_occupancy, dict) else None),
        'commanded_stuck_time_sec': _float_or_none(commanded_stuck.get('commanded_stuck_time_sec') if isinstance(commanded_stuck, dict) else None),
        'personal_space_violation_time_sec': _float_or_none(social.get('personal_space_violation_time_sec') if isinstance(social, dict) else None),
        'footprint_personal_space_violation_time_sec': _float_or_none(social.get('footprint_personal_space_violation_time_sec') if isinstance(social, dict) else None),
        'crowd_freezing_time_sec': _float_or_none(social.get('crowd_freezing_time_sec') if isinstance(social, dict) else None),
        'humans_present': _as_bool(social.get('humans_present')) if isinstance(social, dict) else False,
        'max_humans_observed': _int_or_zero(social.get('max_humans_observed') if isinstance(social, dict) else None),
        'stale_camera_count': _int_or_zero(stale_camera_count),
        'forward_count': _int_or_zero(command_stats.get('forward_count') if isinstance(command_stats, dict) else None),
        'rotate_count': _int_or_zero(command_stats.get('rotate_count') if isinstance(command_stats, dict) else None),
        'stop_count': _int_or_zero(command_stats.get('stop_count') if isinstance(command_stats, dict) else None),
        'strict_task_failure_reasons': ';'.join(str(item) for item in strict_task_failures),
        'strict_social_failure_reasons': ';'.join(str(item) for item in strict_social_failures),
        'validation_failed_checks': ';'.join(str(item) for item in failed_checks),
        'validation_warnings': ';'.join(str(item) for item in warnings),
    }
    failures = failure_tags(row, manifest, validation)
    row['primary_failure'] = failures[0] if failures else 'success'
    row['failure_tags'] = ';'.join(failures) if failures else ''
    return row


def failure_tags(row: dict[str, Any], manifest: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if not row.get('artifact_validation_pass'):
        tags.append('artifact_failure')
    if row.get('legacy_task_success') and not row.get('strict_task_success'):
        tags.append('legacy_task_false_positive')
    if row.get('legacy_social_success') and not row.get('strict_social_success'):
        tags.append('legacy_social_false_positive')
    if not row.get('humans_present'):
        tags.append('missing_humans')
    path_length = row.get('path_length_m')
    if path_length is not None and float(path_length) < 0.1:
        tags.append('no_motion')
    if row.get('human_collision_count', 0) > 0:
        tags.append('collision')
    if row.get('near_miss_count', 0) > 0:
        tags.append('near_miss')
    if row.get('footprint_human_collision_count', 0) > 0:
        tags.append('footprint_collision')
    if row.get('footprint_near_miss_count', 0) > 0:
        tags.append('footprint_near_miss')
    if row.get('static_occupancy_collision_samples', 0) > 0:
        tags.append('static_occupancy_collision')
    if (row.get('commanded_stuck_time_sec') or 0.0) > 0.0:
        tags.append('commanded_stuck')
    if (row.get('personal_space_violation_time_sec') or 0.0) > 0.0:
        tags.append('personal_space_violation')
    if not row.get('strict_task_success'):
        result = manifest.get('result', {}) if isinstance(manifest, dict) else {}
        if result.get('timed_out') or result.get('end_reason') == 'timeout':
            tags.append('timeout')
        else:
            tags.append('task_failure')
    if row.get('stale_camera_count', 0) > 0 and not (row.get('social_success') and row.get('task_success')):
        tags.append('stale_observation_candidate')
    if not tags and row.get('strict_social_success') and row.get('strict_task_success'):
        return []
    return _dedupe(tags)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counter: Counter[str] = Counter()
    for row in rows:
        tags = [tag for tag in str(row.get('failure_tags') or '').split(';') if tag]
        if not tags:
            failure_counter['success'] += 1
        else:
            failure_counter.update(tags)
    count = len(rows)
    return {
        'run_count': count,
        'task_success_rate': _rate(rows, 'task_success'),
        'social_success_rate': _rate(rows, 'social_success'),
        'legacy_task_success_rate': _rate(rows, 'legacy_task_success'),
        'legacy_social_success_rate': _rate(rows, 'legacy_social_success'),
        'strict_task_success_rate': _rate(rows, 'strict_task_success'),
        'strict_social_success_rate': _rate(rows, 'strict_social_success'),
        'benchmark_ready_rate': _rate(rows, 'benchmark_ready'),
        'artifact_validation_pass_rate': _rate(rows, 'artifact_validation_pass'),
        'mean_path_length_m': _mean(row.get('path_length_m') for row in rows),
        'mean_navigation_error_m': _mean(row.get('navigation_error_m') for row in rows),
        'mean_min_human_distance_m': _mean(row.get('min_human_distance_m') for row in rows),
        'mean_min_footprint_clearance_m': _mean(row.get('min_footprint_clearance_m') for row in rows),
        'failure_counts': dict(sorted(failure_counter.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Aggregate Dynamic Social VLN run metrics.')
    parser.add_argument('--root', action='append', default=[], help='Root directory to recursively search for run dirs')
    parser.add_argument('--run-dir', action='append', default=[], help='Explicit run directory')
    parser.add_argument('--output-csv', required=True, help='Output aggregate CSV path')
    parser.add_argument('--summary-json', default='', help='Optional output summary JSON path')
    parser.add_argument('--failure-csv', default='', help='Optional output failure-count CSV path')
    args = parser.parse_args(argv)

    roots = [Path(path).expanduser() for path in args.root] + [Path(path).expanduser() for path in args.run_dir]
    run_dirs = discover_run_dirs(roots)
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    write_csv(Path(args.output_csv).expanduser(), rows, SUMMARY_FIELDS)
    summary = aggregate_summary(rows)
    if args.summary_json:
        path = Path(args.summary_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    if args.failure_csv:
        failure_rows = [
            {'failure_tag': tag, 'count': count, 'rate': (count / len(rows) if rows else 0.0)}
            for tag, count in sorted(summary['failure_counts'].items())
        ]
        write_csv(Path(args.failure_csv).expanduser(), failure_rows, ['failure_tag', 'count', 'rate'])
    print(f'aggregated {len(rows)} run(s) -> {args.output_csv}')
    return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _mean(values) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if _as_bool(row.get(key))) / len(rows) if rows else 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
