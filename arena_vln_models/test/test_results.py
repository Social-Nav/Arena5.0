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
