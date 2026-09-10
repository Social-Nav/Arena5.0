import json
import shutil
from pathlib import Path

import yaml

from arena_bringup.benchmark_result import (
    BENCHMARK_RESULT_SCHEMA,
    generate_benchmark_result,
    main as benchmark_result_main,
)
from arena_bringup.social_nav_metrics_aggregate import (
    aggregate_summary,
    main as aggregate_main,
    summarize_run,
)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding='utf-8')


def _write_complete_run(run_dir: Path) -> None:
    run_dir.mkdir()
    (run_dir / 'metrics.csv').write_text('result\nGOAL_REACHED\n', encoding='utf-8')
    (run_dir / 'internnav_trace.jsonl').write_text(
        json.dumps({'event': 'trajectory'}) + '\n',
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
            },
            'runtime_adjustments': {'dual_vln_status_topic': '/status'},
            'runtime_environment': {'ros_discovery': {'ROS_DOMAIN_ID': '1'}},
            'provenance': {
                'git_commit': 'abc123',
                'git_branch': 'feat/internnav-eval-progress',
                'git_dirty': False,
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
    assert result['verdict']['primary_failure'] == 'success'
    assert result['metrics']['task']['spl'] == 0.88
    assert result['metrics']['social']['min_footprint_clearance_m'] == 0.95
    assert result['metrics']['model']['forward_count'] == 10
    artifact = result['artifacts']['files']['run_manifest.yaml']
    assert artifact['present'] is True
    assert artifact['size_bytes'] > 0
    assert len(artifact['sha256']) == 64
    assert json.loads((run_dir / 'benchmark_result.json').read_text()) == result


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
    manifest['result']['evaluator_returncode'] = 1
    manifest['result']['artifact_validation_returncode'] = 1
    (run_dir / 'run_manifest.yaml').write_text(yaml.safe_dump(manifest), encoding='utf-8')
    validation = json.loads((run_dir / 'artifact_validation.json').read_text())
    validation['overall_pass'] = False
    validation['social_nav_ready'] = False
    validation['failed_checks'] = ['metrics']
    validation['checks']['metrics']['pass'] = False
    validation['checks']['metrics']['strict_task_success'] = False
    _write_json(run_dir / 'artifact_validation.json', validation)

    result = generate_benchmark_result(run_dir)

    assert result['verdict']['valid_run'] is True
    assert result['verdict']['benchmark_ready'] is False
    assert result['verdict']['status'] == 'failed'
    assert result['verdict']['primary_failure'] == 'task_failure'
    assert benchmark_result_main([
        '--dir', str(run_dir),
        '--require-valid',
    ]) == 0
    assert benchmark_result_main([
        '--dir', str(run_dir),
        '--require-ready',
    ]) == 1


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
