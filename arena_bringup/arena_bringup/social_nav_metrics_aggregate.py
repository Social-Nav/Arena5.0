"""Aggregate Dynamic Social VLN eval outputs across run directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


BENCHMARK_RESULT_SCHEMA = 'arena.benchmark_result'
BENCHMARK_RESULT_SCHEMA_VERSION = 1

SUMMARY_FIELDS = [
    'schema_version',
    'run_id',
    'run_dir',
    'timestamp',
    'source_commit',
    'source_branch',
    'scenario_id',
    'sim',
    'world',
    'robot',
    'planner',
    'human',
    'scenario_file',
    'episodes_requested',
    'single_episode',
    'episode_result',
    'end_reason',
    'evaluator_returncode',
    'execution_pass',
    'task_success',
    'strict_task_success',
    'social_success',
    'strict_social_success',
    'artifact_validation_pass',
    'social_nav_ready',
    'valid_run',
    'result_status',
    'benchmark_ready',
    'debug_overlay_fallback',
    'debug_overlay_source_status',
    'debug_overlay_model_frames',
    'debug_overlay_fallback_frames',
    'debug_overlay_received_count',
    'path_length_m',
    'episode_duration_sec',
    'episode_timeout',
    'navigation_error_m',
    'oracle_error_m',
    'spl',
    'ndtw',
    'sdtw',
    'trajectory_length_m',
    'shortest_path_length_m',
    'reference_path_source',
    'reference_path_available',
    'goal_reached',
    'robot_moved',
    'goal_progress_m',
    'diagnostic_goal_progress_m',
    'diagnostic_goal_distance_min_m',
    'min_human_distance_m',
    'min_footprint_clearance_m',
    'min_footprint_clearance_time_sec',
    'min_footprint_clearance_human_id',
    'near_miss_count',
    'human_collision_count',
    'footprint_near_miss_count',
    'footprint_human_collision_count',
    'static_occupancy_collision_samples',
    'commanded_stuck_time_sec',
    'personal_space_violation_time_sec',
    'footprint_personal_space_violation_time_sec',
    'crowd_freezing_time_sec',
    'dynamic_scene_success',
    'moving_human_count',
    'human_motion_time_sec',
    'human_robot_motion_overlap_time_sec',
    'human_robot_interaction_time_sec',
    'humans_present',
    'max_humans_observed',
    'stale_camera_count',
    'forward_count',
    'rotate_count',
    'stop_count',
    'model_trace_present',
    'model_control_pass',
    'model_trace_record_count',
    'model_control_event_count',
    'planning_request_count',
    'trajectory_event_count',
    'discrete_action_event_count',
    'video_pass',
    'metrics_complete',
    'metrics_pass',
    'rtf_mean',
    'rtf_p50',
    'rtf_p95',
    'rtf_sample_count',
    'strict_task_failure_reasons',
    'strict_social_failure_reasons',
    'validation_failed_checks',
    'validation_warnings',
    'primary_failure',
    'failure_tags',
    'diagnostic_tags',
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


def _canonical_result_is_current(run_dir: Path, canonical: dict[str, Any]) -> bool:
    files = (canonical.get('artifacts') or {}).get('files')
    if not isinstance(files, dict):
        return False
    for metadata in files.values():
        if not isinstance(metadata, dict):
            return False
        path = run_dir / str(metadata.get('path') or '')
        expected_present = _as_bool(metadata.get('present'))
        if path.exists() != expected_present:
            return False
        if expected_present and (
            path.stat().st_size != _int_or_zero(metadata.get('size_bytes'))
            or path.stat().st_mtime_ns != _int_or_zero(metadata.get('modified_time_ns'))
        ):
            return False
    return True


def discover_run_dirs(roots: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if any((root / marker).exists() for marker in (
            'benchmark_result.json',
            'run_manifest.yaml',
            'social_metrics.json',
        )):
            found.add(root.resolve())
            continue
        for marker in root.rglob('social_metrics.json'):
            found.add(marker.parent.resolve())
        for marker in root.rglob('artifact_validation.json'):
            found.add(marker.parent.resolve())
        for marker in root.rglob('benchmark_result.json'):
            found.add(marker.parent.resolve())
    return sorted(found)


def summarize_run(run_dir: Path, *, prefer_canonical: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if prefer_canonical:
        canonical = read_json(run_dir / 'benchmark_result.json')
        canonical_summary = canonical.get('summary') if isinstance(canonical, dict) else None
        if (
            isinstance(canonical, dict)
            and canonical.get('schema') == BENCHMARK_RESULT_SCHEMA
            and canonical.get('schema_version') == BENCHMARK_RESULT_SCHEMA_VERSION
            and isinstance(canonical_summary, dict)
            and _canonical_result_is_current(run_dir, canonical)
        ):
            row = {field: canonical_summary.get(field, '') for field in SUMMARY_FIELDS}
            row['run_id'] = run_dir.name
            row['run_dir'] = str(run_dir.resolve())
            return row

    manifest = read_yaml(run_dir / 'run_manifest.yaml') or {}
    vln_task = read_json(run_dir / 'vln_task_metrics.json') or {}
    social = read_json(run_dir / 'social_metrics.json') or {}
    validation = read_json(run_dir / 'artifact_validation.json') or {}
    diagnostics = read_json(run_dir / 'internnav_diagnostic_summary.json') or {}
    timing = read_json(run_dir / 'internnav_timing_summary.json') or {}

    params = manifest.get('parameters', {}) if isinstance(manifest, dict) else {}
    result = manifest.get('result', {}) if isinstance(manifest, dict) else {}
    checks = validation.get('checks', {}) if isinstance(validation, dict) else {}
    metrics_check = checks.get('metrics', {}) if isinstance(checks, dict) else {}
    video_check = checks.get('videos', {}) if isinstance(checks, dict) else {}
    video_results = video_check.get('videos', {}) if isinstance(video_check, dict) else {}
    debug_overlay = video_results.get('ego_debug_overlay', {}) if isinstance(video_results, dict) else {}
    debug_overlay_source = debug_overlay.get('source', {}) if isinstance(debug_overlay, dict) else {}

    base_metrics = social.get('base_metrics', {}) if isinstance(social, dict) else {}
    base_first = base_metrics.get('first', {}) if isinstance(base_metrics, dict) else {}
    episode_result = str(metrics_check.get('episode_result') or base_first.get('result') or '')
    strict_task_success = _as_bool(metrics_check.get('strict_task_success')) if 'strict_task_success' in metrics_check else _as_bool(vln_task.get('strict_task_success'))
    strict_social_success = _as_bool(social.get('strict_social_success', social.get('social_success'))) if isinstance(social, dict) else False
    artifact_pass = _as_bool(validation.get('overall_pass')) if isinstance(validation, dict) else False
    social_nav_ready = _as_bool(validation.get('social_nav_ready')) if isinstance(validation, dict) else False
    evaluator_returncode = result.get('evaluator_returncode')
    end_reason = str(result.get('end_reason') or '')
    execution_pass = (
        bool(result)
        and _as_bool(result.get('finished_observed'))
        and result.get('launch_returncode') == 0
        and result.get('metrics_returncode') in (None, 0)
        and result.get('vln_task_metrics_returncode') in (None, 0)
        and result.get('social_metrics_returncode') in (None, 0)
        and result.get('video_recorder_returncode') in (None, 0)
        and end_reason not in {
            'external_preflight_failed',
            'pedestrian_traversal_preflight_failed',
            'vln_instruction_manifest_lookup_failed',
            'infrastructure_exception',
            'camera_not_ready',
            'required_inputs_not_ready',
        }
    )
    episodes_requested = _int_or_zero(params.get('episodes'))
    single_episode = episodes_requested == 1
    metrics_complete = bool(
        metrics_check.get('metrics_csv_present')
        and metrics_check.get('vln_task_metrics_present')
        and metrics_check.get('social_metrics_present')
        and metrics_check.get('required_social_fields_present')
        and metrics_check.get('reference_path_ready', True)
    )
    validity_checks = ('environment', 'humans', 'model_control', 'videos', 'dynamic_scene')
    valid_run = execution_pass and single_episode and metrics_complete and all(
        _as_bool((checks.get(name) or {}).get('pass'))
        for name in validity_checks
    )

    command_stats = diagnostics.get('command_stats', {}) if isinstance(diagnostics, dict) else {}
    event_counts = diagnostics.get('event_counts', {}) if isinstance(diagnostics, dict) else {}
    fault = diagnostics.get('fault_candidates', {}) if isinstance(diagnostics, dict) else {}
    stale_camera_count = fault.get('stale_record_count', 0) if isinstance(fault, dict) else 0
    goal_distance = diagnostics.get('goal_distance', {}) if isinstance(diagnostics, dict) else {}
    odom_goal_distance = diagnostics.get('odom_goal_distance', {}) if isinstance(diagnostics, dict) else {}
    diagnostic_goal_progress = goal_distance.get('progress_first_minus_last') if isinstance(goal_distance, dict) else None
    if diagnostic_goal_progress is None and isinstance(odom_goal_distance, dict):
        diagnostic_goal_progress = odom_goal_distance.get('progress_first_minus_last')
    diagnostic_goal_distance_min = goal_distance.get('min') if isinstance(goal_distance, dict) else None
    if diagnostic_goal_distance_min is None and isinstance(odom_goal_distance, dict):
        diagnostic_goal_distance_min = odom_goal_distance.get('min')
    goal_metrics = vln_task.get('goal', {}) if isinstance(vln_task, dict) else {}
    start_distance_m = None
    final_distance_m = None
    if isinstance(goal_metrics, dict):
        start_xy = _xy(goal_metrics.get('start_xy'))
        goal_xy = _xy(goal_metrics.get('goal_xy'))
        final_xy = _xy(goal_metrics.get('final_xy'))
        if start_xy is not None and goal_xy is not None:
            start_distance_m = _distance(start_xy, goal_xy)
        if final_xy is not None and goal_xy is not None:
            final_distance_m = _distance(final_xy, goal_xy)
    goal_progress_m = (
        start_distance_m - final_distance_m
        if start_distance_m is not None and final_distance_m is not None
        else diagnostic_goal_progress
    )
    vln_metrics = vln_task.get('vln', {}) if isinstance(vln_task, dict) else {}
    reference_path = vln_metrics.get('reference_path', {}) if isinstance(vln_metrics, dict) else {}
    timing_metrics = vln_task.get('episode_timing', {}) if isinstance(vln_task, dict) else {}
    static_occupancy = vln_task.get('static_occupancy', {}) if isinstance(vln_task, dict) else {}
    commanded_stuck = vln_task.get('commanded_stuck', {}) if isinstance(vln_task, dict) else {}
    strict_task_failures = vln_task.get('strict_task_failure_reasons', []) if isinstance(vln_task, dict) else []
    strict_social_failures = social.get('strict_social_failure_reasons', []) if isinstance(social, dict) else []
    min_footprint_sample = social.get('min_footprint_clearance_sample', {}) if isinstance(social, dict) else {}
    failed_checks = validation.get('failed_checks', []) if isinstance(validation, dict) else []
    warnings = validation.get('warnings', []) if isinstance(validation, dict) else []
    benchmark_ready = (
        valid_run
        and strict_task_success
        and strict_social_success
        and artifact_pass
        and social_nav_ready
    )

    row = {
        'schema_version': BENCHMARK_RESULT_SCHEMA_VERSION,
        'run_id': run_dir.name,
        'run_dir': str(run_dir),
        'timestamp': manifest.get('timestamp') or '',
        'source_commit': ((manifest.get('provenance') or {}).get('git_commit') if isinstance(manifest, dict) else '') or '',
        'source_branch': ((manifest.get('provenance') or {}).get('git_branch') if isinstance(manifest, dict) else '') or '',
        'scenario_id': params.get('scenario_config_id') or params.get('output_prefix') or '',
        'sim': params.get('sim') or '',
        'world': params.get('world') or '',
        'robot': params.get('robot') or '',
        'planner': params.get('local_planner') or '',
        'human': params.get('human') or '',
        'scenario_file': params.get('scenario_file') or '',
        'episodes_requested': episodes_requested,
        'single_episode': single_episode,
        'episode_result': episode_result,
        'end_reason': end_reason,
        'evaluator_returncode': evaluator_returncode,
        'execution_pass': execution_pass,
        'task_success': strict_task_success,
        'strict_task_success': strict_task_success,
        'social_success': strict_social_success,
        'strict_social_success': strict_social_success,
        'artifact_validation_pass': artifact_pass,
        'social_nav_ready': social_nav_ready,
        'valid_run': valid_run,
        'result_status': 'passed' if benchmark_ready else ('failed' if valid_run else 'invalid'),
        'benchmark_ready': benchmark_ready,
        'debug_overlay_fallback': _as_bool(debug_overlay.get('fallback')) if isinstance(debug_overlay, dict) else False,
        'debug_overlay_source_status': debug_overlay_source.get('status') if isinstance(debug_overlay_source, dict) else '',
        'debug_overlay_model_frames': _int_or_zero(debug_overlay_source.get('model_frame_count') if isinstance(debug_overlay_source, dict) else None),
        'debug_overlay_fallback_frames': _int_or_zero(debug_overlay_source.get('fallback_frame_count') if isinstance(debug_overlay_source, dict) else None),
        'debug_overlay_received_count': _int_or_zero(debug_overlay_source.get('received_count') if isinstance(debug_overlay_source, dict) else None),
        'path_length_m': _float_or_none(social.get('path_length_m') if isinstance(social, dict) else None),
        'episode_duration_sec': _float_or_none(timing_metrics.get('duration_sec') if isinstance(timing_metrics, dict) else None),
        'episode_timeout': _as_bool(timing_metrics.get('timed_out')) if isinstance(timing_metrics, dict) else False,
        'navigation_error_m': _float_or_none(goal_metrics.get('navigation_error_m') if isinstance(goal_metrics, dict) else None),
        'oracle_error_m': _float_or_none(goal_metrics.get('oracle_error_m') if isinstance(goal_metrics, dict) else None),
        'spl': _float_or_none(vln_metrics.get('spl') if isinstance(vln_metrics, dict) else None),
        'ndtw': _float_or_none(vln_metrics.get('ndtw') if isinstance(vln_metrics, dict) else None),
        'sdtw': _float_or_none(vln_metrics.get('sdtw') if isinstance(vln_metrics, dict) else None),
        'trajectory_length_m': _float_or_none(vln_metrics.get('trajectory_length_m') if isinstance(vln_metrics, dict) else None),
        'shortest_path_length_m': _float_or_none(vln_metrics.get('shortest_path_length_m') if isinstance(vln_metrics, dict) else None),
        'reference_path_source': reference_path.get('source') if isinstance(reference_path, dict) else '',
        'reference_path_available': _as_bool(reference_path.get('available')) if isinstance(reference_path, dict) else False,
        'goal_reached': _as_bool(goal_metrics.get('goal_reached')) if isinstance(goal_metrics, dict) else False,
        'robot_moved': metrics_check.get('robot_moved'),
        'goal_progress_m': _float_or_none(goal_progress_m),
        'diagnostic_goal_progress_m': _float_or_none(diagnostic_goal_progress),
        'diagnostic_goal_distance_min_m': _float_or_none(diagnostic_goal_distance_min),
        'min_human_distance_m': _float_or_none(social.get('min_human_distance_m') if isinstance(social, dict) else None),
        'min_footprint_clearance_m': _float_or_none(social.get('min_footprint_clearance_m') if isinstance(social, dict) else None),
        'min_footprint_clearance_time_sec': _float_or_none(
            min_footprint_sample.get('time_sec') if isinstance(min_footprint_sample, dict) else None
        ),
        'min_footprint_clearance_human_id': (
            min_footprint_sample.get('human_id') if isinstance(min_footprint_sample, dict) else ''
        ),
        'near_miss_count': _int_or_zero(social.get('near_miss_count') if isinstance(social, dict) else None),
        'human_collision_count': _int_or_zero(social.get('human_collision_count') if isinstance(social, dict) else None),
        'footprint_near_miss_count': _int_or_zero(social.get('footprint_near_miss_count') if isinstance(social, dict) else None),
        'footprint_human_collision_count': _int_or_zero(social.get('footprint_human_collision_count') if isinstance(social, dict) else None),
        'static_occupancy_collision_samples': _int_or_zero(static_occupancy.get('collision_sample_count') if isinstance(static_occupancy, dict) else None),
        'commanded_stuck_time_sec': _float_or_none(commanded_stuck.get('commanded_stuck_time_sec') if isinstance(commanded_stuck, dict) else None),
        'personal_space_violation_time_sec': _float_or_none(social.get('personal_space_violation_time_sec') if isinstance(social, dict) else None),
        'footprint_personal_space_violation_time_sec': _float_or_none(social.get('footprint_personal_space_violation_time_sec') if isinstance(social, dict) else None),
        'crowd_freezing_time_sec': _float_or_none(social.get('crowd_freezing_time_sec') if isinstance(social, dict) else None),
        'dynamic_scene_success': _as_bool(social.get('dynamic_scene_success')) if isinstance(social, dict) else False,
        'moving_human_count': _int_or_zero(social.get('moving_human_count') if isinstance(social, dict) else None),
        'human_motion_time_sec': _float_or_none(social.get('human_motion_time_sec') if isinstance(social, dict) else None),
        'human_robot_motion_overlap_time_sec': _float_or_none(social.get('human_robot_motion_overlap_time_sec') if isinstance(social, dict) else None),
        'human_robot_interaction_time_sec': _float_or_none(social.get('human_robot_interaction_time_sec') if isinstance(social, dict) else None),
        'humans_present': _as_bool(social.get('humans_present')) if isinstance(social, dict) else False,
        'max_humans_observed': _int_or_zero(social.get('max_humans_observed') if isinstance(social, dict) else None),
        'stale_camera_count': _int_or_zero(stale_camera_count),
        'forward_count': _int_or_zero(command_stats.get('forward_count') if isinstance(command_stats, dict) else None),
        'rotate_count': _int_or_zero(command_stats.get('rotate_count') if isinstance(command_stats, dict) else None),
        'stop_count': _int_or_zero(command_stats.get('stop_count') if isinstance(command_stats, dict) else None),
        'model_trace_present': _as_bool((checks.get('model_control') or {}).get('trace_present')) if isinstance(checks, dict) else False,
        'model_control_pass': _as_bool((checks.get('model_control') or {}).get('pass')) if isinstance(checks, dict) else False,
        'model_trace_record_count': _int_or_zero((checks.get('model_control') or {}).get('trace_record_count')) if isinstance(checks, dict) else 0,
        'model_control_event_count': _int_or_zero((checks.get('model_control') or {}).get('direct_control_event_count')) if isinstance(checks, dict) else 0,
        'planning_request_count': _int_or_zero(event_counts.get('planning_request_started') if isinstance(event_counts, dict) else None),
        'trajectory_event_count': _int_or_zero(event_counts.get('trajectory') if isinstance(event_counts, dict) else None),
        'discrete_action_event_count': _int_or_zero(event_counts.get('discrete_action') if isinstance(event_counts, dict) else None),
        'video_pass': _as_bool(video_check.get('pass')) if isinstance(video_check, dict) else False,
        'metrics_complete': metrics_complete,
        'metrics_pass': _as_bool(metrics_check.get('pass')) if isinstance(metrics_check, dict) else False,
        'rtf_mean': _float_or_none(timing.get('rtf_mean') if isinstance(timing, dict) else None),
        'rtf_p50': _float_or_none(timing.get('rtf_p50') if isinstance(timing, dict) else None),
        'rtf_p95': _float_or_none(timing.get('rtf_p95') if isinstance(timing, dict) else None),
        'rtf_sample_count': _int_or_zero(timing.get('rtf_sample_count') if isinstance(timing, dict) else None),
        'strict_task_failure_reasons': ';'.join(str(item) for item in strict_task_failures),
        'strict_social_failure_reasons': ';'.join(str(item) for item in strict_social_failures),
        'validation_failed_checks': ';'.join(str(item) for item in failed_checks),
        'validation_warnings': ';'.join(str(item) for item in warnings),
    }
    failures = failure_tags(row, manifest, validation)
    row['primary_failure'] = failures[0] if failures else 'success'
    row['failure_tags'] = ';'.join(failures) if failures else ''
    row['diagnostic_tags'] = ';'.join(diagnostic_tags(row))
    return row


def failure_tags(row: dict[str, Any], manifest: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    result = manifest.get('result', {}) if isinstance(manifest, dict) else {}
    end_reason = str(result.get('end_reason') or row.get('end_reason') or '').lower()
    if end_reason in {'external_preflight_failed', 'pedestrian_traversal_preflight_failed'}:
        tags.extend(['preflight_failure', end_reason])
    elif end_reason == 'vln_instruction_manifest_lookup_failed':
        tags.extend(['input_contract_failure', end_reason])
    elif end_reason == 'benchmark_result_failed':
        tags.extend(['artifact_generation_failure', end_reason])
    elif end_reason in {'infrastructure_exception', 'camera_not_ready', 'required_inputs_not_ready'}:
        tags.extend(['infrastructure_failure', end_reason])
    evaluator_returncode = row.get('evaluator_returncode')
    if (
        evaluator_returncode not in (None, '', 0, '0')
        and not row.get('valid_run')
        and not tags
    ):
        tags.append('execution_failure')
    if not row.get('single_episode'):
        tags.append('unsupported_multi_episode_run')
    if result and not _as_bool(result.get('finished_observed')):
        tags.append('finished_signal_missing')
    if row.get('episode_timeout') or result.get('timed_out') or 'timeout' in end_reason:
        tags.append('timeout')
    if not row.get('humans_present'):
        tags.append('missing_humans')
    if not row.get('model_control_pass'):
        tags.append('model_control_failure')
    if not row.get('video_pass'):
        tags.append('video_failure')
    if not row.get('metrics_complete'):
        tags.append('metrics_incomplete')
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
    if (row.get('personal_space_violation_time_sec') or 0.0) > 0.0:
        tags.append('personal_space_violation')
    if row.get('static_occupancy_collision_samples', 0) > 0:
        tags.append('static_occupancy_collision')
    if (row.get('commanded_stuck_time_sec') or 0.0) > 0.0:
        tags.append('commanded_stuck')
    if not row.get('strict_task_success'):
        tags.append('task_failure')
    if not row.get('artifact_validation_pass'):
        tags.append('artifact_failure')
    return _dedupe(tags)


def diagnostic_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if row.get('stale_camera_count', 0) > 0:
        tags.append('stale_observation_candidate')
    if row.get('debug_overlay_fallback'):
        tags.append('debug_overlay_fallback')
        source_status = str(row.get('debug_overlay_source_status') or '')
        if source_status:
            tags.append(f'debug_overlay_source_{source_status}')
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
    primary_failure_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    for row in rows:
        primary_failure_counter[str(row.get('primary_failure') or 'unknown')] += 1
        status_counter[str(row.get('result_status') or 'unknown')] += 1
        tags = [tag for tag in str(row.get('failure_tags') or '').split(';') if tag]
        if not tags:
            failure_counter['success'] += 1
        else:
            failure_counter.update(tags)
    count = len(rows)
    valid_rows = [row for row in rows if _as_bool(row.get('valid_run'))]
    return {
        'schema': 'arena.benchmark_aggregate',
        'schema_version': BENCHMARK_RESULT_SCHEMA_VERSION,
        'run_count': count,
        'benchmark_ready_count': _count(rows, 'benchmark_ready'),
        'valid_run_count': _count(rows, 'valid_run'),
        'invalid_run_count': count - _count(rows, 'valid_run'),
        'strict_task_success_count': _count(rows, 'strict_task_success'),
        'strict_social_success_count': _count(rows, 'strict_social_success'),
        'task_success_rate': _rate(rows, 'task_success'),
        'social_success_rate': _rate(rows, 'social_success'),
        'strict_task_success_rate': _rate(rows, 'strict_task_success'),
        'strict_social_success_rate': _rate(rows, 'strict_social_success'),
        'task_success_rate_valid_runs': _rate(valid_rows, 'strict_task_success'),
        'social_success_rate_valid_runs': _rate(valid_rows, 'strict_social_success'),
        'benchmark_ready_rate_valid_runs': _rate(valid_rows, 'benchmark_ready'),
        'benchmark_ready_rate': _rate(rows, 'benchmark_ready'),
        'valid_run_rate': _rate(rows, 'valid_run'),
        'artifact_validation_pass_rate': _rate(rows, 'artifact_validation_pass'),
        'execution_pass_rate': _rate(rows, 'execution_pass'),
        'dynamic_scene_success_rate': _rate(rows, 'dynamic_scene_success'),
        'model_control_pass_rate': _rate(rows, 'model_control_pass'),
        'video_pass_rate': _rate(rows, 'video_pass'),
        'metrics_complete_rate': _rate(rows, 'metrics_complete'),
        'metrics_pass_rate': _rate(rows, 'metrics_pass'),
        'collision_run_rate': _positive_rate(rows, 'footprint_human_collision_count'),
        'near_miss_run_rate': _positive_rate(rows, 'footprint_near_miss_count'),
        'mean_path_length_m': _mean(row.get('path_length_m') for row in rows),
        'mean_path_length_m_valid_runs': _mean(row.get('path_length_m') for row in valid_rows),
        'mean_episode_duration_sec': _mean(row.get('episode_duration_sec') for row in rows),
        'mean_navigation_error_m': _mean(row.get('navigation_error_m') for row in rows),
        'mean_spl': _mean(row.get('spl') for row in rows),
        'mean_ndtw': _mean(row.get('ndtw') for row in rows),
        'mean_sdtw': _mean(row.get('sdtw') for row in rows),
        'mean_spl_valid_runs': _mean(row.get('spl') for row in valid_rows),
        'mean_ndtw_valid_runs': _mean(row.get('ndtw') for row in valid_rows),
        'mean_sdtw_valid_runs': _mean(row.get('sdtw') for row in valid_rows),
        'mean_goal_progress_m': _mean(row.get('goal_progress_m') for row in rows),
        'mean_rtf': _mean(row.get('rtf_mean') for row in rows),
        'mean_min_human_distance_m': _mean(row.get('min_human_distance_m') for row in rows),
        'mean_min_footprint_clearance_m': _mean(row.get('min_footprint_clearance_m') for row in rows),
        'statistics': {
            key: _distribution(rows, key)
            for key in (
                'episode_duration_sec',
                'path_length_m',
                'navigation_error_m',
                'oracle_error_m',
                'spl',
                'ndtw',
                'sdtw',
                'goal_progress_m',
                'min_human_distance_m',
                'min_footprint_clearance_m',
                'personal_space_violation_time_sec',
                'human_robot_motion_overlap_time_sec',
                'human_robot_interaction_time_sec',
                'rtf_mean',
            )
        },
        'total_footprint_human_collisions': sum(
            _int_or_zero(row.get('footprint_human_collision_count')) for row in rows
        ),
        'total_footprint_near_misses': sum(
            _int_or_zero(row.get('footprint_near_miss_count')) for row in rows
        ),
        'failure_counts': dict(sorted(failure_counter.items())),
        'primary_failure_counts': dict(sorted(primary_failure_counter.items())),
        'result_status_counts': dict(sorted(status_counter.items())),
        'by_world': _group_summary(rows, ('world',)),
        'by_scenario': _group_summary(rows, ('world', 'scenario_file')),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Aggregate Dynamic Social VLN run metrics.')
    parser.add_argument('--root', action='append', default=[], help='Root directory to recursively search for run dirs')
    parser.add_argument('--run-dir', action='append', default=[], help='Explicit run directory')
    parser.add_argument('--output-csv', required=True, help='Output aggregate CSV path')
    parser.add_argument('--summary-json', default='', help='Optional output summary JSON path')
    parser.add_argument('--failure-csv', default='', help='Optional output failure-count CSV path')
    parser.add_argument('--min-runs', type=int, default=0, help='Fail if fewer than this many runs are discovered')
    parser.add_argument('--require-ready', action='store_true', help='Fail if any discovered run is not benchmark-ready')
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
    if len(rows) < max(args.min_runs, 0):
        print(f'expected at least {args.min_runs} run(s), found {len(rows)}')
        return 2
    if args.require_ready and any(not _as_bool(row.get('benchmark_ready')) for row in rows):
        print('one or more runs are not benchmark-ready')
        return 1
    return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except Exception:
        return None
    return converted if math.isfinite(converted) else None


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except Exception:
        return None


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _mean(values) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [
        value
        for row in rows
        if _as_bool(row.get('valid_run'))
        for value in [_float_or_none(row.get(key))]
        if value is not None
    ]
    count = len(values)
    mean = statistics.fmean(values) if values else None
    stddev = statistics.stdev(values) if count >= 2 else (0.0 if count == 1 else None)
    half_width = 1.96 * stddev / math.sqrt(count) if count >= 2 else (0.0 if count == 1 else None)
    return {
        'count': count,
        'mean': mean,
        'stddev': stddev,
        'ci95_low': mean - half_width if mean is not None and half_width is not None else None,
        'ci95_high': mean + half_width if mean is not None and half_width is not None else None,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if _as_bool(row.get(key))) / len(rows) if rows else 0.0


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if _as_bool(row.get(key)))


def _positive_rate(rows: list[dict[str, Any]], key: str) -> float:
    return (
        sum(1 for row in rows if _int_or_zero(row.get(key)) > 0) / len(rows)
        if rows
        else 0.0
    )


def _group_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = '/'.join(str(row.get(key) or '<unknown>') for key in keys)
        groups.setdefault(label, []).append(row)
    return {
        label: {
            'run_count': len(group),
            'benchmark_ready_count': _count(group, 'benchmark_ready'),
            'valid_run_count': _count(group, 'valid_run'),
            'benchmark_ready_rate': _rate(group, 'benchmark_ready'),
            'valid_run_rate': _rate(group, 'valid_run'),
            'strict_task_success_rate': _rate(group, 'strict_task_success'),
            'strict_social_success_rate': _rate(group, 'strict_social_success'),
        }
        for label, group in sorted(groups.items())
    }


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
