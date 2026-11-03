import json
import shutil
from pathlib import Path

import pytest
import yaml

from arena_bringup.benchmark_result import (
    BENCHMARK_RESULT_SCHEMA,
    generate_benchmark_result,
    main as benchmark_result_main,
)
from arena_bringup.social_nav_metrics_aggregate import (
    aggregate_summary,
    derive_benchmark_success,
    main as aggregate_main,
    summarize_run,
)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding='utf-8')


def _write_complete_run(run_dir: Path) -> None:
    run_dir.mkdir()
    (run_dir / 'metrics.csv').write_text('result\nGOAL_REACHED\n', encoding='utf-8')
    (run_dir / 'internnav_trace.jsonl').write_text(
        '\n'.join(json.dumps(record) for record in (
            {
                'wall_time': 10.0,
                'monotonic_time_ns': 10_000_000_000,
                'monotonic_clock': 'python.time.monotonic_ns',
                'recorder_pid': 42,
                'event_type': 'planning_request_started',
                'parsed': {'debug': {'request_id': 0}},
            },
            {
                'wall_time': 10.3,
                'monotonic_time_ns': 10_300_000_000,
                'monotonic_clock': 'python.time.monotonic_ns',
                'recorder_pid': 42,
                'event_type': 'trajectory',
                'parsed': {'http_idx': 0},
            },
            {
                'wall_time': 11.0,
                'monotonic_time_ns': 11_000_000_000,
                'monotonic_clock': 'python.time.monotonic_ns',
                'recorder_pid': 42,
                'event_type': 'planning_request_started',
                'parsed': {'debug': {'request_id': 1}},
            },
            {
                'wall_time': 11.4,
                'monotonic_time_ns': 11_400_000_000,
                'monotonic_clock': 'python.time.monotonic_ns',
                'recorder_pid': 42,
                'event_type': 'stop',
                'parsed': {'http_idx': 1},
            },
            {
                'wall_time': 12.0,
                'monotonic_time_ns': 12_000_000_000,
                'monotonic_clock': 'python.time.monotonic_ns',
                'recorder_pid': 42,
                'event_type': 'planning_request_started',
                'parsed': {'debug': {'request_id': 2}},
            },
            {
                'wall_time': 12.2,
                'monotonic_time_ns': 12_200_000_000,
                'monotonic_clock': 'python.time.monotonic_ns',
                'recorder_pid': 42,
                'event_type': 'discrete_action',
                'parsed': {'http_idx': 2},
            },
            {
                'wall_time': 13.0,
                'monotonic_time_ns': 13_000_000_000,
                'monotonic_clock': 'python.time.monotonic_ns',
                'recorder_pid': 42,
                'event_type': 'planning_request_started',
                'parsed': {'debug': {'request_id': 3}},
            },
            {
                'wall_time': 13.5,
                'monotonic_time_ns': 13_500_000_000,
                'monotonic_clock': 'python.time.monotonic_ns',
                'recorder_pid': 42,
                'event_type': 'trajectory',
                'parsed': {'http_idx': 3},
            },
        )) + '\n',
        encoding='utf-8',
    )
    (run_dir / 'run_manifest.yaml').write_text(
        yaml.safe_dump({
            'schema_version': 1,
            'timestamp': '20260909_120000',
            'launch_command': ['ros2', 'launch', 'arena_bringup', 'arena.launch.py'],
            'parameters': {
                'sim': 'isaac_eval',
                'world': 'grscenes_20_v1',
                'scenario_file': 'default_2',
                'scenario_config_id': 'case-03',
                'robot': 'Ai2_Bot2',
                'local_planner': 'dual_vln',
                'human': 'hunav',
                'episodes': 1,
                'dual_vln_model_path': '/models/base',
                'dual_vln_system2_model_path': '/models/socialgen',
                'dual_vln_inference_rate_hz': 3.3333333333,
                'internnav_direct_cmd_vel': True,
            },
            'runtime_adjustments': {'dual_vln_status_topic': '/status'},
            'runtime_environment': {
                'ros_discovery': {'ROS_DOMAIN_ID': '1'},
                'containers': {
                    'arena': {
                        'container_id': 'container-arena',
                        'image_id': 'sha256:arena',
                        'image_ref': 'arena:dev',
                    },
                    'isaac': {
                        'container_id': 'container-isaac',
                        'image_id': 'sha256:isaac',
                        'image_ref': 'arena:dev',
                    },
                    'internnav': {
                        'container_id': 'container-internnav',
                        'image_id': 'sha256:internnav',
                        'image_ref': 'arena:dev',
                    },
                },
            },
            'model_provenance': {
                'base_checkpoint': {
                    'path': '/models/base',
                    'index_sha256': 'base-index',
                    'fingerprint_complete': True,
                    'fingerprint_manifest_sha256': 'base-manifest',
                },
                'system2_checkpoint': {'path': '/models/socialgen', 'revision': {'revision': 'abc'}},
            },
            'provenance': {
                'git_commit': 'abc123',
                'git_branch': 'feat/internnav-eval-progress',
                'git_dirty': False,
                'git_submodules': ' deadbeef arena_evaluation',
            },
            'provenance_overlay': {
                'schema': 'arena.vln.overlay_provenance.v1',
                'source_worktree': '/worktrees/vln-target-merged',
                'source_branch': 'feat/internnav-eval-progress',
                'source_commit': 'b6500c8',
                'source_dirty': False,
                'install_root': '/tmp/unified_install_1001c7a',
                'install_present': True,
                'install_entry_count': 212,
                'install_inventory_sha256': 'f' * 64,
            },
            'result': {
                'finished_observed': True,
                'timed_out': False,
                'end_reason': 'episode_goal_reached',
                'launch_returncode': 0,
                'evaluator_returncode': 0,
            },
        }),
        encoding='utf-8',
    )
    _write_json(run_dir / 'vln_task_metrics.json', {
        'strict_task_success': True,
        'schema_version': 2,
        'strict_task_failure_reasons': [],
        'episode_timing': {'duration_sec': 42.0, 'timed_out': False},
        'goal': {
            'start_xy': [0.0, 0.0],
            'goal_xy': [4.0, 0.0],
            'final_xy': [3.8, 0.0],
            'goal_reached': True,
            'navigation_error_m': 0.2,
            'oracle_error_m': 0.1,
        },
        'vln': {
            'trajectory_length_m': 4.5,
            'shortest_path_length_m': 4.0,
            'spl': 0.88,
            'ndtw': 0.91,
            'sdtw': 0.91,
            'reference_path': {
                'source': 'occupancy_grid_astar',
                'available': True,
            },
        },
        'static_occupancy': {'collision_sample_count': 0},
        'commanded_stuck': {'commanded_stuck_time_sec': 0.0},
    })
    _write_json(run_dir / 'social_metrics.json', {
        'humans_present': True,
        'max_humans_observed': 3,
        'moving_human_count': 3,
        'human_motion_time_sec': 30.0,
        'human_robot_motion_overlap_time_sec': 20.0,
        'human_robot_interaction_time_sec': 8.0,
        'dynamic_scene_success': True,
        'strict_social_success': True,
        'strict_social_failure_reasons': [],
        'path_length_m': 4.5,
        'min_human_distance_m': 1.5,
        'min_footprint_clearance_m': 0.95,
        'near_miss_count': 0,
        'human_collision_count': 0,
        'footprint_near_miss_count': 0,
        'footprint_human_collision_count': 0,
        'personal_space_violation_time_sec': 0.0,
        'footprint_personal_space_violation_time_sec': 0.0,
        'crowd_freezing_time_sec': 0.0,
        'base_metrics': {'first': {'result': 'GOAL_REACHED'}},
    })
    _write_json(run_dir / 'internnav_diagnostic_summary.json', {
        'command_stats': {'forward_count': 10, 'rotate_count': 2, 'stop_count': 1},
        'fault_candidates': {'stale_record_count': 0},
    })
    _write_json(run_dir / 'artifact_validation.json', {
        'overall_pass': True,
        'social_nav_ready': True,
        'failed_checks': [],
        'warnings': [],
        'checks': {
            'metrics': {
                'pass': True,
                'metrics_csv_present': True,
                'vln_task_metrics_present': True,
                'social_metrics_present': True,
                'required_social_fields_present': True,
                'episode_result': 'GOAL_REACHED',
                'strict_task_success': True,
                'robot_moved': True,
            },
            'model_control': {'pass': True, 'trace_present': True},
            'videos': {'pass': True, 'videos': {}},
            'environment': {'pass': True},
            'humans': {'pass': True},
            'dynamic_scene': {'pass': True},
        },
    })


def test_generate_benchmark_result_writes_complete_canonical_schema(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)

    result = generate_benchmark_result(run_dir)

    assert result['schema'] == BENCHMARK_RESULT_SCHEMA
    assert result['schema_version'] == 1
    assert result['run']['sim'] == 'isaac_eval'
    assert result['provenance']['git_commit'] == 'abc123'
    assert result['verdict']['benchmark_ready'] is True
    assert result['verdict']['status'] == 'passed'
    assert result['verdict']['valid_run'] is True
    assert result['verdict']['success'] is True
    assert result['verdict']['benchmark_success'] is True
    assert result['verdict']['instruction_following_success'] is True
    assert result['verdict']['collision_free'] is True
    assert result['verdict']['social_distance_success'] is True
    assert result['metrics']['success_criteria']['social_distance_threshold_m'] == 1.0
    assert result['verdict']['primary_failure'] == 'success'
    assert result['metrics']['task']['spl'] == 0.88
    assert result['metrics']['social']['min_footprint_clearance_m'] == 0.95
    assert result['metrics']['model']['forward_count'] == 10
    timing = result['metrics']['system']['direct_policy_timing']
    assert timing['measurement_valid'] is True
    assert timing['raw_paired_sample_count'] == 4
    assert timing['valid_sample_count'] == 3
    assert timing['warmup_exclusion_samples_applied'] == 1
    assert timing['latency_mean_sec'] == pytest.approx(0.3666666667)
    assert timing['latency_p50_sec'] == pytest.approx(0.4)
    assert timing['effective_planning_rate_hz'] == pytest.approx(1.0)
    assert timing['effective_action_rate_hz'] == pytest.approx(0.9523809524)
    assert timing['configured_planning_rate_hz'] == 3.3333333333
    assert result['model']['base_checkpoint']['index_sha256'] == 'base-index'
    assert result['model']['control_path'] == 'vln_direct_cmd_vel'
    assert result['model']['system2_path'] == '/models/socialgen'
    assert result['model']['system2_checkpoint']['revision']['revision'] == 'abc'
    assert result['provenance']['simulator']['container']['image_id'] == 'sha256:isaac'
    artifact = result['artifacts']['files']['run_manifest.yaml']
    assert artifact['present'] is True
    assert artifact['size_bytes'] > 0
    assert len(artifact['sha256']) == 64
    assert json.loads((run_dir / 'benchmark_result.json').read_text()) == result


def test_invalid_legacy_wall_timing_fails_closed_without_breaking_old_result(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    (run_dir / 'internnav_trace.jsonl').write_text(
        '\n'.join((
            json.dumps({
                'wall_time': 10.0,
                'event_type': 'planning_request_started',
                'parsed': {'debug': {'request_id': 0}},
            }),
            json.dumps({
                'wall_time': 10.3,
                'event_type': 'trajectory',
                'parsed': {'http_idx': 0},
            }),
        )) + '\n',
        encoding='utf-8',
    )

    result = generate_benchmark_result(run_dir)

    timing = result['metrics']['system']['direct_policy_timing']
    assert timing['measurement_valid'] is False
    assert 'monotonic_timestamp_missing' in timing['invalid_reasons']
    assert result['provenance']['qualification_evidence']['valid'] is False
    assert result['verdict']['valid_run'] is False
    assert result['verdict']['primary_failure'] == 'qualification_evidence_invalid'


def test_qualification_reads_overlay_model_and_container_sidecars(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    manifest = yaml.safe_load((run_dir / 'run_manifest.yaml').read_text())
    manifest.pop('model_provenance', None)
    manifest.pop('provenance_overlay', None)
    manifest['runtime_environment'].pop('containers', None)
    (run_dir / 'run_manifest.yaml').write_text(yaml.safe_dump(manifest), encoding='utf-8')
    _write_json(run_dir / 'overlay_provenance.json', {
        'schema': 'arena.vln.overlay_provenance.v1',
        'source_worktree': '/worktrees/vln-target-merged',
        'source_commit': 'b6500c8',
        'install_root': '/tmp/unified_install_1001c7a',
        'install_inventory_sha256': 'e' * 64,
    })
    _write_json(run_dir / 'model_provenance.json', {
        'model': 'internnav',
        'base_checkpoint': {
            'path': '/opt/arena_ws/deps/models/InternVLA-N1-DualVLN',
            'fingerprint_complete': True,
            'fingerprint_manifest_sha256': 'base-manifest',
            'fingerprint_source': 'nfs_registry',
        },
    })
    _write_json(run_dir / 'container_provenance.json', {
        'arena': {'container_id': 'c-arena', 'image_id': 'sha256:arena'},
        'isaac': {'container_id': 'c-isaac', 'image_id': 'sha256:isaac'},
        'internnav': {'container_id': 'c-internnav', 'image_id': 'sha256:internnav'},
    })

    result = generate_benchmark_result(run_dir)

    qualification = result['provenance']['qualification_evidence']
    assert qualification['valid'] is True, qualification
    assert qualification['checks']['overlay_source_present'] is True
    assert qualification['checks']['base_checkpoint_fingerprint_complete'] is True
    assert qualification['checks']['container_image_digests_complete'] is True
    assert result['provenance']['overlay_source']['source_commit'] == 'b6500c8'
    assert result['model']['base_checkpoint']['fingerprint_source'] == 'nfs_registry'
    assert result['verdict']['valid_run'] is True


def test_missing_overlay_source_fails_qualification_closed(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    manifest = yaml.safe_load((run_dir / 'run_manifest.yaml').read_text())
    manifest.pop('provenance_overlay', None)
    (run_dir / 'run_manifest.yaml').write_text(yaml.safe_dump(manifest), encoding='utf-8')

    result = generate_benchmark_result(run_dir)

    qualification = result['provenance']['qualification_evidence']
    assert qualification['valid'] is False
    assert 'overlay_source_present' in qualification['failure_reasons']
    assert result['verdict']['valid_run'] is False


def test_canonical_result_preserves_warning_boundaries(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    validation = json.loads((run_dir / 'artifact_validation.json').read_text())
    validation['warnings'] = ['frame analysis missing; manual review required']
    _write_json(run_dir / 'artifact_validation.json', validation)

    result = generate_benchmark_result(run_dir)

    assert result['verdict']['warnings'] == [
        'frame analysis missing; manual review required'
    ]


def test_summarize_run_rejects_stale_canonical_result(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    generated = generate_benchmark_result(run_dir)
    _write_json(run_dir / 'vln_task_metrics.json', {'strict_task_success': False})
    validation = json.loads((run_dir / 'artifact_validation.json').read_text())
    validation['checks']['metrics']['strict_task_success'] = False
    _write_json(run_dir / 'artifact_validation.json', validation)

    assert summarize_run(run_dir) != generated['summary']
    assert summarize_run(run_dir, prefer_canonical=False)['strict_task_success'] is False


def test_canonical_summary_rebases_run_path_after_directory_move(tmp_path):
    original = tmp_path / 'original'
    moved = tmp_path / 'moved'
    _write_complete_run(original)
    generate_benchmark_result(original)
    shutil.move(original, moved)

    summary = summarize_run(moved)

    assert summary['run_id'] == 'moved'
    assert summary['run_dir'] == str(moved.resolve())
    assert summary['benchmark_ready'] is True


def test_preflight_failure_is_primary_and_not_ready(tmp_path):
    run_dir = tmp_path / 'failed'
    run_dir.mkdir()
    (run_dir / 'run_manifest.yaml').write_text(
        yaml.safe_dump({
            'parameters': {'sim': 'isaac_eval', 'world': 'grscenes_20_v1'},
            'result': {'end_reason': 'external_preflight_failed', 'evaluator_returncode': 2},
        }),
        encoding='utf-8',
    )

    result = generate_benchmark_result(run_dir)

    assert result['verdict']['benchmark_ready'] is False
    assert result['verdict']['status'] == 'invalid'
    assert result['verdict']['primary_failure'] == 'preflight_failure'
    assert result['verdict']['failure_tags'][:2] == [
        'preflight_failure',
        'external_preflight_failed',
    ]


def test_valid_task_failure_is_distinct_from_invalid_run(tmp_path):
    run_dir = tmp_path / 'failed'
    _write_complete_run(run_dir)
    task = json.loads((run_dir / 'vln_task_metrics.json').read_text())
    task['strict_task_success'] = False
    task['strict_task_failure_reasons'] = ['goal_not_reached']
    task['goal']['goal_reached'] = False
    _write_json(run_dir / 'vln_task_metrics.json', task)
    manifest = yaml.safe_load((run_dir / 'run_manifest.yaml').read_text())
    manifest['result']['end_reason'] = 'episode_model_stop'
    manifest['result']['evaluator_returncode'] = 0
    manifest['result']['artifact_validation_returncode'] = 0
    (run_dir / 'run_manifest.yaml').write_text(yaml.safe_dump(manifest), encoding='utf-8')
    validation = json.loads((run_dir / 'artifact_validation.json').read_text())
    validation['overall_pass'] = True
    validation['social_nav_ready'] = True
    validation['failed_checks'] = []
    validation['checks']['metrics']['pass'] = True
    validation['checks']['metrics']['strict_task_success'] = False
    _write_json(run_dir / 'artifact_validation.json', validation)

    result = generate_benchmark_result(run_dir)

    assert result['verdict']['valid_run'] is True
    assert result['verdict']['benchmark_ready'] is True
    assert result['verdict']['status'] == 'failed'
    assert result['verdict']['primary_failure'] == 'task_failure'
    assert benchmark_result_main([
        '--dir', str(run_dir),
        '--require-valid',
    ]) == 0
    assert benchmark_result_main([
        '--dir', str(run_dir),
        '--require-ready',
    ]) == 0


def test_postprocess_failure_is_invalid_and_not_benchmark_ready(tmp_path):
    run_dir = tmp_path / 'postprocess_failed'
    _write_complete_run(run_dir)
    manifest = yaml.safe_load((run_dir / 'run_manifest.yaml').read_text())
    manifest['result']['metrics_returncode'] = 2
    (run_dir / 'run_manifest.yaml').write_text(
        yaml.safe_dump(manifest), encoding='utf-8'
    )

    result = generate_benchmark_result(run_dir)

    assert result['verdict']['execution_pass'] is False
    assert result['verdict']['valid_run'] is False
    assert result['verdict']['benchmark_ready'] is False
    assert result['verdict']['status'] == 'invalid'


def test_versioned_validation_can_drive_derived_canonical_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv('ARENA_POSTPROCESS_SOURCE_COMMIT', 'parser123')
    run_dir = tmp_path / 'versioned'
    _write_complete_run(run_dir)
    original = run_dir / 'artifact_validation.json'
    original_payload = json.loads(original.read_text())
    original_payload.update({
        'overall_pass': False,
        'social_nav_ready': False,
        'failed_checks': ['metrics'],
    })
    original.write_text(json.dumps(original_payload), encoding='utf-8')
    derived_payload = dict(original_payload)
    derived_payload.update({
        'schema_version': 2,
        'semantics': 'artifact_evidence_validity_separate_from_task_outcome',
        'overall_pass': True,
        'social_nav_ready': True,
        'failed_checks': [],
    })
    derived_payload['checks'] = dict(original_payload['checks'])
    derived_payload['checks']['metrics'] = dict(original_payload['checks']['metrics'])
    derived_payload['checks']['metrics']['pass'] = True
    derived = run_dir / 'artifact_validation.semantic_v2.json'
    derived.write_text(json.dumps(derived_payload), encoding='utf-8')
    output = run_dir / 'benchmark_result.semantic_v2.json'

    result = generate_benchmark_result(
        run_dir,
        output,
        artifact_validation=derived,
    )

    assert original_payload == json.loads(original.read_text())
    assert output.is_file()
    assert result['verdict']['benchmark_ready'] is True
    assert result['provenance']['artifact_validation_source']['path'] == str(derived)
    assert result['provenance']['artifact_validation_source']['schema_version'] == 2
    assert result['provenance']['derivation']['parser_git_commit'] == 'parser123'
    assert result['provenance']['derivation']['source_run_git_commit'] == 'abc123'


def test_zero_human_control_does_not_get_missing_humans_failure_tag(tmp_path):
    run_dir = tmp_path / 'zero_humans'
    _write_complete_run(run_dir)
    manifest = yaml.safe_load((run_dir / 'run_manifest.yaml').read_text())
    manifest['parameters']['expected_human_count'] = 0
    (run_dir / 'run_manifest.yaml').write_text(
        yaml.safe_dump(manifest), encoding='utf-8'
    )
    social = json.loads((run_dir / 'social_metrics.json').read_text())
    social['humans_present'] = False
    social['max_humans_observed'] = 0
    social['moving_human_count'] = 0
    _write_json(run_dir / 'social_metrics.json', social)
    validation = json.loads((run_dir / 'artifact_validation.json').read_text())
    validation['checks']['humans']['expected_human_count'] = 0
    _write_json(run_dir / 'artifact_validation.json', validation)

    result = generate_benchmark_result(run_dir)

    assert 'missing_humans' not in result['verdict']['failure_tags']
    assert result['verdict']['human_safety_gate_applied'] is False
    assert result['verdict']['success'] is result['verdict']['instruction_following_success']
    assert result['verdict']['collision_free'] is None
    assert result['verdict']['social_distance_success'] is None


def test_canonical_result_preserves_resolved_density_contract(tmp_path):
    run_dir = tmp_path / 'density'
    _write_complete_run(run_dir)
    manifest = yaml.safe_load((run_dir / 'run_manifest.yaml').read_text())
    contract = {
        'schema': 'arena.density_consumer_contract.v1',
        'applicable': True,
        'valid': True,
        'config_path_resolved': '/configs/task_generator_n05.yaml',
        'config_sha256': 'a' * 64,
        'configured_human_count': 5,
        'expected_human_count': 5,
        'consumer': {
            'requested': 'random',
            'resolved': 'random',
            'required': 'random',
        },
        'failure_reasons': [],
    }
    manifest['density_contract'] = contract
    manifest['parameters']['density_contract'] = contract
    (run_dir / 'run_manifest.yaml').write_text(
        yaml.safe_dump(manifest), encoding='utf-8'
    )

    result = generate_benchmark_result(run_dir)

    assert result['run']['density_contract'] == contract


def _derive_human_success(**social_overrides):
    social = {
        'humans_present': True,
        'min_human_distance_m': 1.01,
        'human_collision_count': 0,
        'footprint_human_collision_count': 0,
        'thresholds': {'personal_space_radius_m': 1.0},
    }
    social.update(social_overrides)
    return derive_benchmark_success(
        expected_human_count=5,
        strict_task_success=True,
        social_metrics=social,
        static_occupancy={'collision_sample_count': 0},
    )


def test_human_episode_success_requires_collision_free_and_social_distance():
    result = _derive_human_success()

    assert result['instruction_following_success'] is True
    assert result['collision_free'] is True
    assert result['social_distance_success'] is True
    assert result['benchmark_success'] is True


def test_human_episode_collision_overrides_instruction_success():
    result = _derive_human_success(footprint_human_collision_count=1)

    assert result['collision_free'] is False
    assert result['benchmark_success'] is False
    assert 'collision_detected' in result['benchmark_success_failure_reasons']


def test_human_episode_social_distance_violation_overrides_instruction_success():
    result = _derive_human_success(min_human_distance_m=0.99)

    assert result['collision_free'] is True
    assert result['social_distance_success'] is False
    assert result['benchmark_success'] is False
    assert 'social_distance_threshold_failed' in result['benchmark_success_failure_reasons']


def test_social_distance_threshold_is_strictly_greater_than():
    result = _derive_human_success(min_human_distance_m=1.0)

    assert result['social_distance_operator'] == '>'
    assert result['social_distance_success'] is False
    assert result['benchmark_success'] is False


def test_human_episode_missing_safety_evidence_fails_closed():
    result = _derive_human_success(min_human_distance_m=None)

    assert result['safety_evidence_complete'] is False
    assert result['social_distance_success'] is False
    assert result['benchmark_success'] is False
    assert 'social_distance_evidence_missing' in result['benchmark_success_failure_reasons']


def test_human_episode_missing_collision_evidence_fails_closed():
    result = _derive_human_success(human_collision_count=None)

    assert result['safety_evidence_complete'] is False
    assert result['collision_free'] is False
    assert result['benchmark_success'] is False
    assert 'collision_evidence_missing' in result['benchmark_success_failure_reasons']


def test_unspecified_expected_count_uses_observed_human_presence():
    result = derive_benchmark_success(
        expected_human_count=-1,
        strict_task_success=True,
        social_metrics={
            'humans_present': True,
            'min_human_distance_m': 1.2,
            'human_collision_count': 0,
            'footprint_human_collision_count': 0,
            'thresholds': {'personal_space_radius_m': 1.0},
        },
        static_occupancy={'collision_sample_count': 0},
    )

    assert result['human_safety_gate_applied'] is True
    assert result['human_count_source'] == 'social_metrics.humans_present'
    assert result['benchmark_success'] is True


def test_aggregate_summary_has_counts_rates_and_groups(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    row = generate_benchmark_result(run_dir)['summary']

    summary = aggregate_summary([row])

    assert summary['schema'] == 'arena.benchmark_aggregate'
    assert summary['run_count'] == 1
    assert summary['benchmark_ready_count'] == 1
    assert summary['valid_run_count'] == 1
    assert summary['invalid_run_count'] == 0
    assert summary['benchmark_ready_rate'] == 1.0
    assert summary['benchmark_success_count'] == 1
    assert summary['benchmark_success_rate'] == 1.0
    assert summary['benchmark_success_rate_valid_runs'] == 1.0
    assert summary['instruction_following_success_rate_valid_runs'] == 1.0
    assert summary['task_success_rate_valid_runs'] == 1.0
    assert summary['mean_spl_valid_runs'] == 0.88
    assert summary['statistics']['spl'] == {
        'count': 1,
        'mean': 0.88,
        'stddev': 0.0,
        'ci95_low': 0.88,
        'ci95_high': 0.88,
    }
    assert summary['mean_spl'] == 0.88
    assert summary['by_world']['grscenes_20_v1']['benchmark_ready_rate'] == 1.0
    assert summary['by_scenario']['grscenes_20_v1/default_2']['run_count'] == 1


def test_follow_path_result_uses_canonical_control_path(tmp_path):
    run_dir = tmp_path / 'follow_path'
    _write_complete_run(run_dir)
    manifest = yaml.safe_load((run_dir / 'run_manifest.yaml').read_text())
    manifest['model_backend'] = 'internnav'
    manifest['parameters'].update({
        'model_backend': 'internnav',
        'execution_mode': 'nav2_follow_path',
        'control_path': 'nav2_follow_path_rpp',
        'internnav_direct_cmd_vel': False,
        'controller': 'regulated_pure_pursuit',
    })
    (run_dir / 'run_manifest.yaml').write_text(
        yaml.safe_dump(manifest), encoding='utf-8'
    )

    result = generate_benchmark_result(run_dir)

    assert result['model']['control_path'] == 'nav2_follow_path_rpp'
    assert result['model']['interface'] == 'nav2_msgs/action/FollowPath'
    assert result['execution']['control_path'] == 'nav2_follow_path_rpp'
    assert result['summary']['control_path'] == 'nav2_follow_path_rpp'


def test_aggregate_rejects_cross_control_path_rows(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    direct = generate_benchmark_result(run_dir)['summary']
    follow_path = {**direct, 'run_id': 'follow', 'control_path': 'nav2_follow_path_rpp'}

    with pytest.raises(ValueError, match='cross-path aggregation is forbidden'):
        aggregate_summary([direct, follow_path])


def test_aggregate_cli_rejects_cross_control_path_without_writing_csv(tmp_path):
    direct_dir = tmp_path / 'direct'
    follow_dir = tmp_path / 'follow'
    _write_complete_run(direct_dir)
    _write_complete_run(follow_dir)
    follow_manifest = yaml.safe_load((follow_dir / 'run_manifest.yaml').read_text())
    follow_manifest['parameters'].update({
        'execution_mode': 'nav2_follow_path',
        'control_path': 'nav2_follow_path_rpp',
        'internnav_direct_cmd_vel': False,
    })
    (follow_dir / 'run_manifest.yaml').write_text(
        yaml.safe_dump(follow_manifest), encoding='utf-8'
    )
    output = tmp_path / 'mixed.csv'

    assert aggregate_main([
        '--run-dir', str(direct_dir),
        '--run-dir', str(follow_dir),
        '--output-csv', str(output),
    ]) == 2
    assert not output.exists()


def test_density_provenance_is_exported_to_aggregate_rows(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    manifest = yaml.safe_load((run_dir / 'run_manifest.yaml').read_text())
    manifest['parameters'].update({
        'expected_human_count': 5,
        'pedestrian_seed': 20260923,
        'pedestrian_pool_size': 10,
    })
    (run_dir / 'run_manifest.yaml').write_text(
        yaml.safe_dump(manifest), encoding='utf-8'
    )

    row = summarize_run(run_dir, prefer_canonical=False)

    assert row['expected_human_count'] == 5
    assert row['pedestrian_seed'] == 20260923
    assert row['pedestrian_pool_size'] == 10


def test_aggregate_cli_can_gate_on_ready_runs(tmp_path):
    run_dir = tmp_path / 'run'
    _write_complete_run(run_dir)
    generate_benchmark_result(run_dir)
    csv_path = tmp_path / 'summary.csv'

    assert aggregate_main([
        '--run-dir', str(run_dir),
        '--output-csv', str(csv_path),
        '--min-runs', '1',
        '--require-ready',
    ]) == 0
    assert csv_path.exists()
    assert aggregate_main([
        '--run-dir', str(run_dir),
        '--output-csv', str(csv_path),
        '--min-runs', '2',
    ]) == 2
