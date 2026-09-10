"""Generate the canonical, compact result record for one Arena benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arena_bringup.social_nav_metrics_aggregate import (
    BENCHMARK_RESULT_SCHEMA,
    BENCHMARK_RESULT_SCHEMA_VERSION,
    read_json,
    read_yaml,
    summarize_run,
)


SOURCE_ARTIFACTS = (
    'run_manifest.yaml',
    'metrics.csv',
    'vln_task_metrics.json',
    'social_metrics.json',
    'artifact_validation.json',
    'internnav_status.json',
    'internnav_trace.jsonl',
    'internnav_diagnostic_summary.json',
    'video_index.json',
    'internnav_timing_summary.json',
    'episode_outcome.json',
    'params.yaml',
    'start_goal.csv',
    'odom.csv',
    'cmd_vel.csv',
    'human_states.csv',
    'pedsim_agents_data.csv',
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _split_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [item for item in str(value or '').split(';') if item]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def generate_benchmark_result(
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Write a stable result schema derived from the detailed run artifacts."""
    run_path = Path(run_dir).expanduser().resolve()
    manifest = read_yaml(run_path / 'run_manifest.yaml') or {}
    if not manifest:
        raise ValueError(f'missing or invalid run manifest: {run_path / "run_manifest.yaml"}')
    validation = read_json(run_path / 'artifact_validation.json') or {}
    vln_task = read_json(run_path / 'vln_task_metrics.json') or {}
    social = read_json(run_path / 'social_metrics.json') or {}
    summary = summarize_run(run_path, prefer_canonical=False)
    source_files = {}
    for filename in SOURCE_ARTIFACTS:
        path = run_path / filename
        source_files[filename] = {
            'path': filename,
            'present': path.exists(),
            'size_bytes': path.stat().st_size if path.exists() else 0,
            'modified_time_ns': path.stat().st_mtime_ns if path.exists() else None,
            'sha256': _sha256(path) if path.exists() else None,
        }

    manifest_result = manifest.get('result', {}) if isinstance(manifest, dict) else {}
    payload = {
        'schema': BENCHMARK_RESULT_SCHEMA,
        'schema_version': BENCHMARK_RESULT_SCHEMA_VERSION,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'run': {
            'id': summary['run_id'],
            'timestamp': summary['timestamp'],
            'sim': summary['sim'],
            'world': summary['world'],
            'scenario': summary['scenario_file'],
            'scenario_id': summary['scenario_id'],
            'robot': summary['robot'],
            'planner': summary['planner'],
            'human': summary['human'],
            'episodes_requested': summary['episodes_requested'],
            'single_episode': summary['single_episode'],
        },
        'provenance': {
            'git_commit': summary['source_commit'],
            'git_branch': summary['source_branch'],
            'git_dirty': ((manifest.get('provenance') or {}).get('git_dirty') if isinstance(manifest, dict) else None),
            'git_submodules': ((manifest.get('provenance') or {}).get('git_submodules') if isinstance(manifest, dict) else None),
            'launch_command': manifest.get('launch_command') if isinstance(manifest, dict) else None,
            'runtime_adjustments': manifest.get('runtime_adjustments') if isinstance(manifest, dict) else None,
            'runtime_environment': manifest.get('runtime_environment') if isinstance(manifest, dict) else None,
            'instruction': vln_task.get('instruction'),
            'language_task_contract': vln_task.get('language_task_contract'),
            'metric_schema_versions': {
                'vln_task_metrics': vln_task.get('schema_version'),
                'social_metrics': social.get('schema_version'),
                'artifact_validation': validation.get('schema_version'),
            },
        },
        'model': {
            'mode': (manifest.get('parameters') or {}).get('dual_vln_mode'),
            'path': (manifest.get('parameters') or {}).get('dual_vln_model_path'),
            'device': (manifest.get('parameters') or {}).get('dual_vln_device'),
            'output_policy': (manifest.get('parameters') or {}).get('dual_vln_model_output_policy'),
            'external_server': _as_bool((manifest.get('parameters') or {}).get('internnav_external_server')),
            'direct_cmd_vel': _as_bool((manifest.get('parameters') or {}).get('internnav_direct_cmd_vel')),
        },
        'verdict': {
            'status': summary['result_status'],
            'valid_run': summary['valid_run'],
            'benchmark_ready': summary['benchmark_ready'],
            'execution_pass': summary['execution_pass'],
            'task_success': summary['strict_task_success'],
            'social_success': summary['strict_social_success'],
            'artifact_validation_pass': summary['artifact_validation_pass'],
            'social_nav_ready': summary['social_nav_ready'],
            'primary_failure': summary['primary_failure'],
            'failure_tags': _split_list(summary['failure_tags']),
            'diagnostic_tags': _split_list(summary['diagnostic_tags']),
            'task_failure_reasons': _split_list(summary['strict_task_failure_reasons']),
            'social_failure_reasons': _split_list(summary['strict_social_failure_reasons']),
            'failed_checks': _split_list(validation.get('failed_checks')),
            'warnings': _split_list(validation.get('warnings')),
        },
        'execution': {
            'episode_result': summary['episode_result'],
            'end_reason': summary['end_reason'],
            'evaluator_returncode': summary['evaluator_returncode'],
            'launch_returncode': manifest_result.get('launch_returncode'),
            'metrics_returncode': manifest_result.get('metrics_returncode'),
            'vln_task_metrics_returncode': manifest_result.get('vln_task_metrics_returncode'),
            'social_metrics_returncode': manifest_result.get('social_metrics_returncode'),
            'artifact_validation_returncode': manifest_result.get('artifact_validation_returncode'),
            'video_recorder_returncode': manifest_result.get('video_recorder_returncode'),
            'finished_observed': _as_bool(manifest_result.get('finished_observed')),
            'timed_out': _as_bool(manifest_result.get('timed_out')),
            'episode_outcome': manifest_result.get('episode_outcome'),
        },
        'metrics': {
            'task': {key: summary[key] for key in (
                'goal_reached', 'episode_duration_sec', 'navigation_error_m',
                'oracle_error_m', 'trajectory_length_m', 'shortest_path_length_m',
                'reference_path_source', 'reference_path_available',
                'spl', 'ndtw', 'sdtw', 'goal_progress_m',
                'static_occupancy_collision_samples', 'commanded_stuck_time_sec',
            )},
            'social': {key: summary[key] for key in (
                'humans_present', 'max_humans_observed', 'dynamic_scene_success',
                'moving_human_count', 'human_motion_time_sec',
                'human_robot_motion_overlap_time_sec', 'human_robot_interaction_time_sec',
                'min_human_distance_m', 'min_footprint_clearance_m',
                'personal_space_violation_time_sec',
                'footprint_personal_space_violation_time_sec', 'crowd_freezing_time_sec',
                'near_miss_count', 'human_collision_count',
                'footprint_near_miss_count', 'footprint_human_collision_count',
            )},
            'model': {key: summary[key] for key in (
                'model_trace_present', 'model_control_pass', 'stale_camera_count',
                'model_trace_record_count', 'model_control_event_count',
                'planning_request_count', 'trajectory_event_count',
                'discrete_action_event_count', 'forward_count', 'rotate_count',
                'stop_count', 'rtf_mean', 'rtf_p50', 'rtf_p95',
                'rtf_sample_count',
            )},
            'thresholds': {
                'task': vln_task.get('thresholds'),
                'social': social.get('thresholds'),
            },
        },
        'artifacts': {
            'video_pass': summary['video_pass'],
            'metrics_complete': summary['metrics_complete'],
            'metrics_pass': summary['metrics_pass'],
            'files': source_files,
            'validation_checks': {
                name: _as_bool(check.get('pass'))
                for name, check in (validation.get('checks') or {}).items()
                if isinstance(check, dict)
            } if isinstance(validation, dict) else {},
        },
        # Flat summary is deliberately retained for CSV export and backwards-
        # compatible batch aggregation without duplicating field derivation.
        'summary': summary,
    }
    output_path = Path(output).expanduser() if output else run_path / 'benchmark_result.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'.{output_path.name}.tmp.{os.getpid()}')
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
            encoding='utf-8',
        )
        os.replace(temporary_path, output_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate one canonical Arena benchmark result.')
    parser.add_argument('--dir', required=True, help='Eval run directory')
    parser.add_argument('--output', default='', help='Output path (default: <dir>/benchmark_result.json)')
    parser.add_argument('--require-valid', action='store_true', help='Exit non-zero unless the run is complete and scoreable')
    parser.add_argument('--require-ready', action='store_true', help='Exit non-zero unless benchmark_ready is true')
    args = parser.parse_args(argv)
    payload = generate_benchmark_result(args.dir, args.output or None)
    output_path = Path(args.output).expanduser() if args.output else Path(args.dir).expanduser() / 'benchmark_result.json'
    print(json.dumps({
        'output': str(output_path),
        'benchmark_ready': payload['verdict']['benchmark_ready'],
        'primary_failure': payload['verdict']['primary_failure'],
    }, indent=2))
    if args.require_ready and not payload['verdict']['benchmark_ready']:
        return 1
    if args.require_valid and not payload['verdict']['valid_run']:
        return 1
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
