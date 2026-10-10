from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'arena_bringup'))
from arena_bringup.benchmark_profile import load_profile, validate_profile, evaluation_defaults, BenchmarkProfileError
from arena_bringup.benchmark_external import Deployment, installed_features
from arena_bringup.navila_support import diagnostic_summary

PROFILE = ROOT / 'arena_bringup/configs/benchmark/profiles/navila_grscenes.yaml'


def test_profile_and_public_plan_isolate_the_selected_backend():
    defaults = evaluation_defaults(load_profile(PROFILE))
    assert defaults['model_backend'] == defaults['local_planner'] == 'navila'
    assert defaults['dual_vln_status_topic'] == 'navila/status'
    assert not defaults['internnav_direct_cmd_vel']
    result = subprocess.run([str(ROOT / '_meta/docker/features/benchmark/main'), 'plan', 'run',
        '--profile', str(PROFILE)], capture_output=True, text=True, check=True)
    assert 'arena-navila' in result.stdout and 'navila_action_velocity' in result.stdout
    assert 'arena-lightnav' not in result.stdout and 'features/internnav/main launch' not in result.stdout


@pytest.mark.parametrize('section,key,value', [('server', 'endpoint', 'ws://127.0.0.1:54321'),
    ('client', 'kind', 'lightnav_native_mpc'), ('client', 'request_timeout_sec', 0),
    ('evaluation', 'local_planner', 'lightnav')])
def test_wrong_protocol_or_controller_is_rejected(section, key, value):
    data = deepcopy(load_profile(PROFILE)); data[section][key] = value
    with pytest.raises(BenchmarkProfileError): validate_profile(data)


def test_three_algorithms_coexist_but_duplicates_and_unknown_are_rejected(tmp_path):
    p = tmp_path / '.installed'; p.write_text('isaac\ninternnav\nlightnav\nnavila\n')
    for backend in ['internnav', 'lightnav', 'navila']:
        assert backend in installed_features(p, backend)
    for content in ['isaac\nnavila\nnavila\n', 'isaac\nnavila\nunknown\n', 'navila\n']:
        p.write_text(content)
        with pytest.raises(ValueError): installed_features(p, 'navila')


def test_velocity_evidence_requires_matching_model_request_and_action(tmp_path):
    path = tmp_path / 'trace.jsonl'
    identity = dict(episode=0, generation=2, request_id=1, run_token='owned')
    records = [dict(identity, status='navila_command', linear_x=.5, angular_z=0)]
    def write(): path.write_text('\n'.join(json.dumps(r) for r in records))
    write(); assert not diagnostic_summary(path)['commands_linked_to_inference']
    records.extend([dict(identity, status='navila_inference', capture_stamp_ns=1, latency_ms=500),
                    dict(identity, status='action_accepted')])
    write(); assert diagnostic_summary(path)['commands_linked_to_inference']
    records[-1]['generation'] = 1
    write(); assert not diagnostic_summary(path)['commands_linked_to_inference']


def test_owner_check_rejects_other_profiles_and_reused_pids():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)',
                               'benchmark_external.py', '--profile', str(PROFILE)])
    try:
        deployment = object.__new__(Deployment)
        deployment.backend = 'navila'; deployment.profile_path = PROFILE.resolve()
        state = {'owner_pid': process.pid, 'model_backend': 'navila'}
        assert deployment.owner_active(state)
        assert not deployment.owner_active(dict(state, model_backend='lightnav'))
        assert not deployment.owner_active(dict(state, owner_start_ticks='stale_pid'))
        deployment.profile_path = PROFILE.with_name('lightnav_grscenes.yaml')
        assert not deployment.owner_active(state)
    finally:
        process.terminate(); process.wait(timeout=3)


def test_result_identity_uses_selected_backend_not_legacy_internnav_flags(tmp_path):
    import yaml
    from arena_bringup.benchmark_result import generate_benchmark_result
    (tmp_path / 'run_manifest.yaml').write_text(yaml.safe_dump({
        'model_backend': 'navila', 'parameters': {'execution_mode': 'external_direct_cmd_vel',
            'internnav_external_server': False, 'internnav_direct_cmd_vel': False,
            'dual_vln_model_output_policy': 'trajectory', 'dual_vln_mode': 'navila'}}))
    model = generate_benchmark_result(tmp_path)['model']
    assert model['external_server'] and model['direct_cmd_vel']
    assert model['output_policy'] == 'discrete_action'
