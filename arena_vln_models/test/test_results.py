"""Current-session execution evidence and generic result generation."""
import json

import pytest
import yaml

from arena_vln_models.diagnostics import execution_evidence


def test_evidence_ignores_old_goals_and_accepts_current_no_motion_stop(tmp_path):
    (tmp_path / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id='new', episode_index=1, reason='model_stop')))
    (tmp_path / 'path_execution.jsonl').write_text(json.dumps(
        dict(session_id='old', event='goal_status', status=4, goal_id='123')))
    trace = tmp_path / 'model_trace.jsonl'
    trace.write_text(json.dumps(dict(parsed=dict(session_id='old', status='stop'))))
    assert not execution_evidence(tmp_path)['pass']
    trace.write_text(json.dumps(dict(parsed=dict(session_id='new', status='stop'))))
    evidence = execution_evidence(tmp_path)
    assert evidence['pass'] and evidence['model_stop'] and evidence['path_count'] == 0


def _write_jsonl(path, records):
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))


def _unified_records(session, **overrides):
    records = [
        dict(event='executor_started', session_id=''),
        dict(event='session_started', session_id=session, episode=0),
        dict(event='inference_submitted', session_id=session, episode=0, request_id=1),
        dict(event='decision', session_id=session, episode=0,
             decision=dict(action='follow_path', status='running', pose_count=1)),
        dict(event='follow_path_goal', session_id=session, episode=0, serial=1),
        dict(event='goal_accepted', session_id=session, episode=0, serial=1, goal_id='g1'),
    ]
    for index, record in overrides.items():
        records[index] = dict(records[index], **record)
    return records


def test_unified_executor_evidence_passes_with_real_control_loop(tmp_path):
    (tmp_path / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id='s1', episode_index=0, reason='sim_timeout')))
    _write_jsonl(tmp_path / 'executor_1_arena_model_executor.jsonl', _unified_records('s1'))
    evidence = execution_evidence(tmp_path)
    assert evidence['pass']
    assert evidence['unified_control_loop']
    assert evidence['executor_trace_present']
    assert evidence['path_count'] == 1
    assert evidence['inference_submitted_count'] == 1
    assert evidence['trajectory_event_count'] == 1
    assert evidence['model_control_event_count'] == 4
    assert not evidence['model_stop']


def test_lisn_grounded_goal_loop_qualifies_with_social_mpc(tmp_path):
    (tmp_path / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id='s1', episode_index=0, reason='sim_timeout')))
    _write_jsonl(tmp_path / 'executor_1_arena_model_executor.jsonl', [
        dict(event='session_started', session_id='s1', episode=0),
        dict(event='lisn_grounded_goal_received', session_id='s1', episode=0,
             poses=[[1.0, 2.0, 0.0]], stamp_ns=10),
        dict(event='decision', session_id='s1', episode=0,
             decision=dict(action='follow_path', status='running', pose_count=2)),
        dict(event='follow_path_goal', session_id='s1', episode=0, goal_serial=1),
        dict(event='goal_accepted', session_id='s1', episode=0, serial=1),
    ])
    evidence = execution_evidence(tmp_path)
    assert evidence['pass']
    assert evidence['unified_control_loop']
    assert evidence['control_path'] == 'nav2_follow_path_social_mpc'
    assert evidence['lisn_grounded_goal_count'] == 1
    assert evidence['inference_submitted_count'] == 0


def test_heartbeat_only_does_not_prove_model_control(tmp_path):
    (tmp_path / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id='s1', episode_index=0, reason='sim_timeout')))
    _write_jsonl(tmp_path / 'model_trace.jsonl', [
        dict(parsed=dict(session_id='s1', status='running')),
        dict(parsed=dict(session_id='s1', status='running')),
    ])
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['unified_control_loop']


def test_session_mismatch_does_not_qualify(tmp_path):
    (tmp_path / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id='s1', episode_index=0, reason='sim_timeout')))
    _write_jsonl(tmp_path / 'executor_1_arena_model_executor.jsonl',
                 [dict(event='session_started', session_id='s1')] + _unified_records('other'))
    _write_jsonl(tmp_path / 'model_trace.jsonl', [dict(parsed=dict(session_id='other', status='running'))])
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['unified_control_loop']
    assert evidence['executor_trace_present']
    assert evidence['path_count'] == 0


def test_error_only_decision_does_not_qualify(tmp_path):
    (tmp_path / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id='s1', episode_index=0, reason='episode_model_error')))
    records = _unified_records('s1')
    records[3] = dict(records[3], decision=dict(action='error', status='error', error_type='model_error'))
    records.pop(4)
    records.pop(4)
    _write_jsonl(tmp_path / 'executor_1_arena_model_executor.jsonl', records)
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['unified_control_loop']
    assert evidence['trajectory_event_count'] == 0


def test_unified_evidence_requires_all_four_events(tmp_path):
    (tmp_path / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id='s1', episode_index=0, reason='sim_timeout')))
    for missing in ('inference_submitted', 'follow_path_goal', 'goal_accepted'):
        records = [record for record in _unified_records('s1') if record['event'] != missing]
        _write_jsonl(tmp_path / 'executor_1_arena_model_executor.jsonl', records)
        evidence = execution_evidence(tmp_path)
        assert not evidence['pass'], missing
        assert not evidence['unified_control_loop'], missing


@pytest.mark.parametrize('reason', ['episode_model_error', 'episode_control_error'])
def test_error_cannot_be_classified_as_successful_execution(tmp_path, reason):
    from arena_bringup.benchmark_result import generate_benchmark_result
    manifest = dict(model_backend='custom_experiment',
                    parameters=dict(execution_mode='nav2_follow_path', episodes=1),
                    result=dict(finished_observed=True, launch_returncode=0,
                                evaluator_returncode=2, end_reason=reason))
    (tmp_path / 'run_manifest.yaml').write_text(yaml.safe_dump(manifest))
    result = generate_benchmark_result(tmp_path)
    assert result['model']['label'] == 'custom_experiment'
    assert result['model']['interface'] == 'nav2_msgs/action/FollowPath'
    assert not result['verdict']['execution_pass']
    assert not result['verdict']['valid_run']


# --- external FollowPath client (LightNav/NaVILA) evidence -----------------

def _external_client_records(session='s1', goal='g1', events=None):
    events = events or ('inference_started', 'inference_result', 'path_submitted',
                        'goal_accepted', 'episode_released')
    records = []
    for event in events:
        record = dict(event=event, session_id=session, episode=0)
        if event in ('path_submitted', 'goal_accepted'):
            record['goal_uuid'] = goal
        records.append(record)
    return records


def _write_external_fixture(tmp_path, session='s1', goal='g1', client_events=None,
                            client_session=None, path_goals=None, path_sessions=('life',),
                            controller_active=True):
    (tmp_path / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id=session, episode_index=0, reason='sim_timeout')))
    trace_dir = tmp_path / 'external_client_trace'
    trace_dir.mkdir()
    _write_jsonl(trace_dir / 'client_1.jsonl',
                 _external_client_records(client_session or session, goal, client_events))
    path_records = []
    for index, path_session in enumerate(path_sessions):
        path_records.append(dict(event='session_started', session_id=path_session, episode=0))
        for path_goal in (path_goals if path_goals is not None else [goal]):
            path_records.append(dict(event='goal_status', session_id=path_session,
                                     goal_id=path_goal, status=2))
    if controller_active:
        path_records.append(dict(event='controller_active',
                                 session_id=path_sessions[0] if path_sessions else ''))
    _write_jsonl(tmp_path / 'path_execution.jsonl', path_records)


def test_external_client_evidence_passes_with_joined_goals(tmp_path):
    _write_external_fixture(tmp_path)
    evidence = execution_evidence(tmp_path)
    assert evidence['pass']
    assert evidence['external_control_loop']
    assert evidence['external_joined_goal_count'] == 1
    assert evidence['path_count'] == 1
    assert evidence['external_inference_count'] == 1
    assert evidence['model_control_event_count'] == 4


def test_external_client_heartbeat_only_fails(tmp_path):
    _write_external_fixture(tmp_path, client_events=('status', 'model_reachable'))
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['external_control_loop']


def test_external_client_missing_accepted_goal_fails(tmp_path):
    _write_external_fixture(
        tmp_path,
        client_events=('inference_started', 'inference_result', 'path_submitted', 'episode_released'),
    )
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['external_control_loop']


def test_external_client_session_mismatch_fails(tmp_path):
    _write_external_fixture(tmp_path, client_session='other')
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['external_control_loop']
    assert evidence['external_client_trace_present']


def test_external_client_mixed_old_run_goal_fails(tmp_path):
    _write_external_fixture(tmp_path, path_goals=['g1', 'old'])
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['external_control_loop']
    assert evidence['external_joined_goal_count'] == 1


def test_external_client_multiple_lifecycle_sessions_fails(tmp_path):
    _write_external_fixture(tmp_path, path_sessions=('life-a', 'life-b'))
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['external_control_loop']


def test_external_client_requires_controller_active(tmp_path):
    _write_external_fixture(tmp_path, controller_active=False)
    evidence = execution_evidence(tmp_path)
    assert not evidence['pass']
    assert not evidence['external_control_loop']


def test_external_client_trace_is_discovered_from_session_model_runtime(tmp_path):
    run_dir = tmp_path / 'evaluation' / 'nav2_follow_path_rpp' / 'case' / 'run'
    run_dir.mkdir(parents=True)
    (run_dir / 'episode_outcome.json').write_text(json.dumps(
        dict(session_id='s1', episode_index=0, reason='sim_timeout')))
    _write_jsonl(run_dir / 'path_execution.jsonl', [
        dict(event='session_started', session_id='life'),
        dict(event='goal_status', session_id='life', goal_id='g1', status=2),
        dict(event='controller_active', session_id='life'),
    ])
    client_dir = tmp_path / 'model_runtime' / 'client'
    client_dir.mkdir(parents=True)
    _write_jsonl(client_dir / 'client_1.jsonl', _external_client_records())
    evidence = execution_evidence(run_dir)
    assert evidence['pass']
    assert evidence['external_client_trace_files'] == ['client_1.jsonl']

