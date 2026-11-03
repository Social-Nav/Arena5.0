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
    CONTROL_PATH_FOLLOW_PATH,
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
    'model_status.json',
    'model_trace.jsonl',
    'model_diagnostic_summary.json',
    'path_execution.jsonl',
    'timing_summary.json',
    'episode_outcome.json',
    'params.yaml',
    'start_goal.csv',
    'odom.csv',
    'cmd_vel.csv',
    'human_states.csv',
    'pedsim_agents_data.csv',
)
DIRECT_TIMING_CLOCK = 'python.time.monotonic_ns'
DIRECT_TIMING_WARMUP_SAMPLES = 1
DIRECT_TIMING_MIN_VALID_SAMPLES = 3
DIRECT_TIMING_MIN_PAIRING_COVERAGE = 0.8


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


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _event_rate_hz(times: list[float]) -> float | None:
    ordered = sorted(times)
    if len(ordered) < 2:
        return None
    duration = ordered[-1] - ordered[0]
    return (len(ordered) - 1) / duration if duration > 0.0 else None


def _direct_policy_timing_metrics(trace_path: Path) -> dict[str, Any]:
    """Measure direct-policy request/action timing on one monotonic clock."""
    result = {
        'measurement_source': 'internnav_trace.status_watcher_receipt_monotonic',
        'measurement_semantics': (
            'planning_request_started receipt -> first matching '
            'trajectory/discrete_action/stop receipt'
        ),
        'start_event': 'planning_request_started',
        'end_events': ['trajectory', 'discrete_action', 'stop'],
        'clock_source': DIRECT_TIMING_CLOCK,
        'clock_scope': 'single_status_watcher_process',
        'time_basis': 'monotonic',
        'latency_unit': 'seconds',
        'rate_unit': 'hertz',
        'measurement_valid': False,
        'invalid_reasons': [],
        'request_count': 0,
        'action_event_count': 0,
        'raw_paired_sample_count': 0,
        'valid_sample_count': 0,
        'pairing_coverage': 0.0,
        'unmatched_request_count': 0,
        'warmup_exclusion_policy': 'exclude_first_matched_pair',
        'warmup_exclusion_samples_requested': DIRECT_TIMING_WARMUP_SAMPLES,
        'warmup_exclusion_samples_applied': 0,
        'minimum_valid_samples': DIRECT_TIMING_MIN_VALID_SAMPLES,
        'minimum_pairing_coverage': DIRECT_TIMING_MIN_PAIRING_COVERAGE,
        'latency_mean_sec': None,
        'latency_p50_sec': None,
        'latency_p95_sec': None,
        'effective_planning_rate_hz': None,
        'effective_action_rate_hz': None,
        'action_event_types': {},
    }
    if not trace_path.is_file():
        result['invalid_reasons'] = ['trace_missing']
        return result

    request_ids = set()
    action_ids = set()
    requests: dict[int, tuple[float, str, int]] = {}
    responses: dict[int, tuple[float, str, str, int]] = {}
    missing_monotonic_count = 0
    action_types = {'trajectory', 'discrete_action', 'stop'}
    with trace_path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            parsed = record.get('parsed') if isinstance(record.get('parsed'), dict) else {}
            debug = parsed.get('debug') if isinstance(parsed.get('debug'), dict) else {}
            event_type = str(record.get('event_type') or record.get('event') or '')
            if event_type not in action_types and event_type != 'planning_request_started':
                continue
            try:
                monotonic_sec = int(record.get('monotonic_time_ns')) / 1_000_000_000.0
                clock_source = str(record.get('monotonic_clock') or '')
                recorder_pid = int(record.get('recorder_pid'))
            except (TypeError, ValueError):
                monotonic_sec = None
                clock_source = ''
                recorder_pid = -1
                missing_monotonic_count += 1
            if event_type == 'planning_request_started':
                request_id = debug.get('request_id', record.get('request_id'))
                try:
                    request_id = int(request_id)
                except (TypeError, ValueError):
                    continue
                request_ids.add(request_id)
                if monotonic_sec is not None:
                    requests.setdefault(
                        request_id,
                        (monotonic_sec, clock_source, recorder_pid),
                    )
            else:
                response_id = parsed.get('http_idx', record.get('http_idx'))
                try:
                    response_id = int(response_id)
                except (TypeError, ValueError):
                    continue
                if response_id >= 0:
                    action_ids.add(response_id)
                    if monotonic_sec is not None and response_id not in responses:
                        responses[response_id] = (
                            monotonic_sec,
                            event_type,
                            clock_source,
                            recorder_pid,
                        )

    paired = []
    for request_id, (request_time, request_clock, request_pid) in requests.items():
        response = responses.get(request_id)
        if response is None:
            continue
        response_time, event_type, response_clock, response_pid = response
        if response_time < request_time:
            continue
        paired.append({
            'request_id': request_id,
            'request_time': request_time,
            'response_time': response_time,
            'latency': response_time - request_time,
            'event_type': event_type,
            'request_clock': request_clock,
            'response_clock': response_clock,
            'request_pid': request_pid,
            'response_pid': response_pid,
        })
    paired.sort(key=lambda item: (item['request_time'], item['request_id']))
    excluded_count = min(DIRECT_TIMING_WARMUP_SAMPLES, len(paired))
    retained = paired[excluded_count:]
    latency_values = [item['latency'] for item in retained]
    request_times = [item['request_time'] for item in retained]
    response_times = [item['response_time'] for item in retained]
    event_type_counts: dict[str, int] = {}
    for item in retained:
        event_type = item['event_type']
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

    raw_sample_count = len(paired)
    sample_count = len(retained)
    pairing_coverage = raw_sample_count / len(request_ids) if request_ids else 0.0
    observed_clocks = {
        value
        for item in paired
        for value in (item['request_clock'], item['response_clock'])
        if value
    }
    observed_pids = {
        value
        for item in paired
        for value in (item['request_pid'], item['response_pid'])
        if value >= 0
    }
    invalid_reasons = []
    if missing_monotonic_count:
        invalid_reasons.append('monotonic_timestamp_missing')
    if observed_clocks != {DIRECT_TIMING_CLOCK}:
        invalid_reasons.append('monotonic_clock_inconsistent')
    if len(observed_pids) != 1:
        invalid_reasons.append('recorder_process_inconsistent')
    if pairing_coverage < DIRECT_TIMING_MIN_PAIRING_COVERAGE:
        invalid_reasons.append('pairing_coverage_below_threshold')
    if sample_count < DIRECT_TIMING_MIN_VALID_SAMPLES:
        invalid_reasons.append('insufficient_post_warmup_samples')
    planning_rate = _event_rate_hz(request_times)
    action_rate = _event_rate_hz(response_times)
    if sample_count >= DIRECT_TIMING_MIN_VALID_SAMPLES and planning_rate is None:
        invalid_reasons.append('effective_planning_rate_unavailable')
    if sample_count >= DIRECT_TIMING_MIN_VALID_SAMPLES and action_rate is None:
        invalid_reasons.append('effective_action_rate_unavailable')

    result.update({
        'measurement_valid': not invalid_reasons,
        'invalid_reasons': invalid_reasons,
        'request_count': len(request_ids),
        'action_event_count': len(action_ids),
        'raw_paired_sample_count': raw_sample_count,
        'valid_sample_count': sample_count,
        'pairing_coverage': pairing_coverage,
        'unmatched_request_count': max(0, len(request_ids) - raw_sample_count),
        'warmup_exclusion_samples_applied': excluded_count,
        'latency_mean_sec': sum(latency_values) / sample_count if sample_count else None,
        'latency_p50_sec': _percentile(latency_values, 0.50),
        'latency_p95_sec': _percentile(latency_values, 0.95),
        'effective_planning_rate_hz': planning_rate,
        'effective_action_rate_hz': action_rate,
        'action_event_types': event_type_counts,
    })
    return result


def _read_sidecar(run_dir, name: str) -> dict[str, Any]:
    if run_dir is None:
        return {}
    try:
        payload = read_json(Path(run_dir) / name)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _overlay_summary(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        'schema', 'source_worktree', 'source_branch', 'source_commit', 'source_dirty',
        'validator_source_commit', 'install_root', 'install_present', 'install_entry_count',
        'install_file_count', 'install_symlink_count', 'install_inventory_sha256',
        'collected_at',
    )
    return {key: record.get(key) for key in keys if key in record}


def _qualification_evidence(
    manifest: dict[str, Any],
    direct_policy_timing: dict[str, Any],
    run_dir=None,
) -> dict[str, Any]:
    parameters = manifest.get('parameters') or {}
    provenance = manifest.get('provenance') or {}
    overlay = manifest.get('provenance_overlay') or _read_sidecar(run_dir, 'overlay_provenance.json')
    model_provenance = manifest.get('model_provenance') or {}
    if not (model_provenance.get('base_checkpoint') or {}).get('fingerprint_complete'):
        sidecar_model = _read_sidecar(run_dir, 'model_provenance.json')
        if sidecar_model:
            model_provenance = sidecar_model
    containers = (manifest.get('runtime_environment') or {}).get('containers') or {}
    sidecar_containers = _read_sidecar(run_dir, 'container_provenance.json')
    required_services = ('arena', 'isaac', 'internnav')
    if sidecar_containers and not all(
        str((containers.get(service) or {}).get('image_id') or '').startswith('sha256:')
        for service in required_services
    ):
        containers = sidecar_containers
    base_checkpoint = model_provenance.get('base_checkpoint') or {}
    direct_cmd_vel = _as_bool(parameters.get('internnav_direct_cmd_vel'))
    checks = {
        'git_commit_present': bool(str(provenance.get('git_commit') or '').strip()),
        'recursive_submodule_shas_present': bool(
            str(provenance.get('git_submodules') or '').strip()
        ),
        'repository_clean': provenance.get('git_dirty') is False,
        'overlay_source_present': bool(
            str(overlay.get('source_worktree') or '').strip()
            and str(overlay.get('source_commit') or '').strip()
            and str(overlay.get('install_root') or '').strip()
            and str(overlay.get('install_inventory_sha256') or '').strip()
        ),
        'base_checkpoint_fingerprint_complete': _as_bool(
            base_checkpoint.get('fingerprint_complete')
        ),
        'container_image_digests_complete': all(
            str((containers.get(service) or {}).get('image_id') or '').startswith('sha256:')
            for service in required_services
        ),
        'direct_policy_timing_valid': (
            _as_bool(direct_policy_timing.get('measurement_valid'))
            if direct_cmd_vel
            else True
        ),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        'valid': not reasons,
        'fail_closed': True,
        'checks': checks,
        'failure_reasons': reasons,
    }


def _append_delimited(value: Any, item: str) -> str:
    items = _split_list(value)
    if item not in items:
        items.append(item)
    return ';'.join(items)


def generate_benchmark_result(
    run_dir: str | Path,
    output: str | Path | None = None,
    artifact_validation: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a stable result schema derived from the detailed run artifacts."""
    run_path = Path(run_dir).expanduser().resolve()
    manifest = read_yaml(run_path / 'run_manifest.yaml') or {}
    if not manifest:
        raise ValueError(f'missing or invalid run manifest: {run_path / "run_manifest.yaml"}')
    validation_path = (
        Path(artifact_validation).expanduser().resolve()
        if artifact_validation
        else run_path / 'artifact_validation.json'
    )
    validation = read_json(validation_path) or {}
    vln_task = read_json(run_path / 'vln_task_metrics.json') or {}
    social = read_json(run_path / 'social_metrics.json') or {}
    summary = summarize_run(
        run_path,
        prefer_canonical=False,
        validation_path=validation_path,
    )
    parameters = manifest.get('parameters') or {}
    runtime_environment = manifest.get('runtime_environment') or {}
    container_provenance = runtime_environment.get('containers') or {}
    sidecar_containers = _read_sidecar(run_path, 'container_provenance.json')
    required_services = ('arena', 'isaac', 'internnav')
    if sidecar_containers and not all(
        str((container_provenance.get(service) or {}).get('image_id') or '').startswith('sha256:')
        for service in required_services
    ):
        container_provenance = sidecar_containers
    model_provenance = manifest.get('model_provenance') or {}
    if not (model_provenance.get('base_checkpoint') or {}).get('fingerprint_complete'):
        sidecar_model = _read_sidecar(run_path, 'model_provenance.json')
        if sidecar_model:
            model_provenance = sidecar_model
    overlay_source = manifest.get('provenance_overlay') or _read_sidecar(
        run_path, 'overlay_provenance.json'
    )
    direct_policy_timing = _direct_policy_timing_metrics(
        run_path / 'internnav_trace.jsonl'
    )
    direct_policy_timing['configured_planning_rate_hz'] = parameters.get(
        'internnav_planning_rate_hz',
        parameters.get('dual_vln_inference_rate_hz'),
    )
    qualification_evidence = _qualification_evidence(manifest, direct_policy_timing, run_path)
    if not qualification_evidence['valid'] and summary.get('valid_run'):
        summary = dict(summary)
        summary['valid_run'] = False
        summary['benchmark_ready'] = False
        summary['result_status'] = 'failed'
        summary['primary_failure'] = 'qualification_evidence_invalid'
        summary['failure_tags'] = _append_delimited(
            summary.get('failure_tags'),
            'qualification_evidence_invalid',
        )
        summary['benchmark_success'] = False
        summary['success'] = False
        for reason in qualification_evidence['failure_reasons']:
            summary['benchmark_success_failure_reasons'] = _append_delimited(
                summary.get('benchmark_success_failure_reasons'),
                reason,
            )
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
            'expected_human_count': summary['expected_human_count'],
            'density_contract': (
                manifest.get('density_contract')
                or parameters.get('density_contract')
            ),
            'episodes_requested': summary['episodes_requested'],
            'single_episode': summary['single_episode'],
        },
        'provenance': {
            'git_commit': summary['source_commit'],
            'git_branch': summary['source_branch'],
            'git_dirty': ((manifest.get('provenance') or {}).get('git_dirty') if isinstance(manifest, dict) else None),
            'git_submodules': ((manifest.get('provenance') or {}).get('git_submodules') if isinstance(manifest, dict) else None),
            'base_runtime': {
                'git_commit': summary['source_commit'],
                'git_branch': summary['source_branch'],
                'git_dirty': ((manifest.get('provenance') or {}).get('git_dirty') if isinstance(manifest, dict) else None),
                'git_submodules': ((manifest.get('provenance') or {}).get('git_submodules') if isinstance(manifest, dict) else None),
            },
            'overlay_source': _overlay_summary(overlay_source),
            'launch_command': manifest.get('launch_command') if isinstance(manifest, dict) else None,
            'runtime_adjustments': manifest.get('runtime_adjustments') if isinstance(manifest, dict) else None,
            'runtime_environment': runtime_environment,
            'container_images': container_provenance,
            'simulator': {
                'backend': summary['sim'],
                'container': container_provenance.get('isaac'),
            },
            'qualification_evidence': qualification_evidence,
            'artifact_validation_source': {
                'path': str(validation_path),
                'present': validation_path.is_file(),
                'sha256': _sha256(validation_path) if validation_path.is_file() else None,
                'schema_version': validation.get('schema_version'),
                'semantics': validation.get('semantics'),
                'parser': validation.get('parser'),
            },
            'derivation': {
                'parser_implementation': 'arena_bringup.benchmark_result',
                'parser_git_commit': os.environ.get('ARENA_POSTPROCESS_SOURCE_COMMIT'),
                'source_run_git_commit': summary['source_commit'],
            },
            'instruction': vln_task.get('instruction'),
            'language_task_contract': vln_task.get('language_task_contract'),
            'metric_schema_versions': {
                'vln_task_metrics': vln_task.get('schema_version'),
                'social_metrics': social.get('schema_version'),
                'artifact_validation': validation.get('schema_version'),
            },
        },
        'model': {
            'label': parameters.get('model_backend') or manifest.get('model_backend') or 'internnav',
            'mode': parameters.get('dual_vln_mode'),
            'path': parameters.get('dual_vln_model_path'),
            'base_checkpoint': model_provenance.get('base_checkpoint'),
            'system2_path': parameters.get('dual_vln_system2_model_path'),
            'system2_checkpoint': model_provenance.get('system2_checkpoint'),
            'device': parameters.get('dual_vln_device'),
            'output_policy': parameters.get('dual_vln_model_output_policy'),
            'external_server': _as_bool(parameters.get('internnav_external_server')),
            'direct_cmd_vel': _as_bool(parameters.get('internnav_direct_cmd_vel')),
            'control_path': summary['control_path'],
            'interface': (
                'nav2_msgs/action/FollowPath'
                if summary['control_path'] == CONTROL_PATH_FOLLOW_PATH
                else 'geometry_msgs/msg/Twist'
            ),
            'external_client': summary['control_path'] == CONTROL_PATH_FOLLOW_PATH,
        },
        'verdict': {
            'status': summary['result_status'],
            'valid_run': summary['valid_run'],
            'benchmark_ready': summary['benchmark_ready'],
            'execution_pass': summary['execution_pass'],
            'success': summary['benchmark_success'],
            'benchmark_success': summary['benchmark_success'],
            'instruction_following_success': summary['instruction_following_success'],
            'human_safety_gate_applied': summary['human_safety_gate_applied'],
            'safety_evidence_complete': summary['safety_evidence_complete'],
            'collision_free': summary['collision_free'],
            'social_distance_success': summary['social_distance_success'],
            'task_success': summary['strict_task_success'],
            'social_success': summary['strict_social_success'],
            'artifact_validation_pass': summary['artifact_validation_pass'],
            'social_nav_ready': summary['social_nav_ready'],
            'primary_failure': summary['primary_failure'],
            'failure_tags': _split_list(summary['failure_tags']),
            'diagnostic_tags': _split_list(summary['diagnostic_tags']),
            'task_failure_reasons': _split_list(summary['strict_task_failure_reasons']),
            'social_failure_reasons': _split_list(summary['strict_social_failure_reasons']),
            'benchmark_success_failure_reasons': _split_list(
                summary['benchmark_success_failure_reasons']
            ),
            'failed_checks': _split_list(validation.get('failed_checks')),
            'warnings': _split_list(validation.get('warnings')),
        },
        'execution': {
            'model_backend': parameters.get('model_backend') or manifest.get('model_backend') or 'internnav',
            'controller': parameters.get('controller'),
            'execution_mode': parameters.get('execution_mode'),
            'control_path': summary['control_path'],
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
            'success_criteria': {
                'policy': 'arena.task_and_social_safety.v1',
                'formula': (
                    'instruction_following_success AND '
                    '(NOT human_safety_gate_applied OR '
                    '(collision_free AND social_distance_success))'
                ),
                'instruction_following_metric': 'strict_task_success',
                'human_safety_gate_applied': summary['human_safety_gate_applied'],
                'collision_free': summary['collision_free'],
                'social_distance_success': summary['social_distance_success'],
                'social_distance_metric': summary['social_distance_metric'],
                'social_distance_operator': summary['social_distance_operator'],
                'social_distance_threshold_m': summary['social_distance_threshold_m'],
                'safety_evidence_complete': summary['safety_evidence_complete'],
            },
            'task': {key: summary[key] for key in (
                'goal_reached', 'episode_duration_sec', 'navigation_error_m',
                'oracle_error_m', 'trajectory_length_m', 'shortest_path_length_m',
                'shortest_path_length_source', 'reference_path_source',
                'reference_path_geometry_source', 'reference_path_available',
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
                'model_trace_record_count', 'model_control_event_count', 'path_count',
                'planning_request_count', 'trajectory_event_count',
                'discrete_action_event_count', 'forward_count', 'rotate_count',
                'stop_count', 'rtf_mean', 'rtf_p50', 'rtf_p95',
                'rtf_sample_count',
            )},
            'system': {
                'direct_policy_timing': direct_policy_timing,
                'rtf': {
                    'mean': summary['rtf_mean'],
                    'p50': summary['rtf_p50'],
                    'p95': summary['rtf_p95'],
                    'sample_count': summary['rtf_sample_count'],
                },
            },
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
    if provenance is not None:
        payload['provenance'] = provenance
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
    parser.add_argument('--artifact-validation', default='', help='Versioned artifact validation input')
    parser.add_argument('--require-valid', action='store_true', help='Exit non-zero unless the run is complete and scoreable')
    parser.add_argument('--require-ready', action='store_true', help='Exit non-zero unless benchmark_ready is true')
    parser.add_argument('--provenance-file', default='',
                        help='Optional JSON provenance envelope to embed')
    args = parser.parse_args(argv)
    provenance = None
    if args.provenance_file:
        provenance = json.loads(
            Path(args.provenance_file).expanduser().read_text(encoding='utf-8'))
        from arena_bringup.paper_pipeline import (
            PipelineRejected, require_provenance_authenticity)
        try:
            require_provenance_authenticity(
                provenance, expected_run_dir=args.dir)
        except PipelineRejected as exc:
            print(json.dumps({'valid': False, 'errors': [str(exc)]},
                             indent=2))
            return 2
    payload = generate_benchmark_result(
        args.dir,
        args.output or None,
        args.artifact_validation or None,
        provenance=provenance,
    )
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
