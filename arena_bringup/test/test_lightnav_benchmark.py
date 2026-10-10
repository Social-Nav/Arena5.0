from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'arena_bringup'))
from arena_bringup.benchmark_profile import load_profile, validate_profile, evaluation_defaults, BenchmarkProfileError
from arena_bringup.benchmark_lightnav import installed_features
from arena_bringup.lightnav_support import diagnostic_summary

PROFILE = ROOT / 'arena_bringup/configs/benchmark/profiles/lightnav_grscenes.yaml'


def test_native_backend_keeps_own_identity_and_no_internnav_flags():
    profile = load_profile(PROFILE)
    defaults = evaluation_defaults(profile)
    assert defaults['model_backend'] == defaults['dual_vln_mode'] == 'lightnav'
    assert defaults['execution_mode'] == 'external_direct_cmd_vel'
    assert not defaults['internnav_direct_cmd_vel']
    assert defaults['local_planner'] == 'lightnav'
    assert defaults['dual_vln_depth_topic'] == ''


@pytest.mark.parametrize('section,key,value', [
    ('server', 'mode', 'internnav'), ('server', 'endpoint', 'http://127.0.0.1:8050'),
    ('client', 'kind', 'nav2'), ('execution', 'timing_mode', 'sim_time_realworld'),
    ('evaluation', 'local_planner', 'regulated_pure_pursuit'),
    ('client', 'command_timeout_sec', 0),
    ('client', 'path_timeout_sec', float('inf')),
    ('video', 'fps', float('nan')),
])
def test_incompatible_profile_fails_closed(section, key, value):
    data = deepcopy(load_profile(PROFILE)); data[section][key] = value
    with pytest.raises(BenchmarkProfileError):
        validate_profile(data)


def test_feature_coexistence_does_not_enable_unknown_backends(tmp_path):
    p = tmp_path / '.installed'
    p.write_text('isaac\ninternnav\nlightnav\n')
    assert installed_features(p, 'lightnav') == ['internnav', 'isaac', 'lightnav']
    p.write_text('isaac\ninternnav\n')
    with pytest.raises(ValueError): installed_features(p, 'lightnav')
    p.write_text('isaac\nlightnav\nlightnav\n')
    with pytest.raises(ValueError): installed_features(p, 'lightnav')


def test_public_plan_selects_native_mpc_and_separate_model_project():
    result = subprocess.run([str(ROOT / '_meta/docker/features/benchmark/main'), 'plan', 'run',
                             '--profile', str(PROFILE)], capture_output=True, text=True, check=True)
    assert 'arena-lightnav' in result.stdout
    assert 'lightnav_mpc' in result.stdout
    assert 'features/internnav/main launch' not in result.stdout
    assert '--remove-orphans' not in result.stdout


def test_control_requires_a_real_matching_prediction_and_accepted_path(tmp_path):
    trace = tmp_path / 'lightnav_trace.jsonl'
    identity = {'episode': 0, 'native_episode': 1, 'request_id': 7}
    records = [dict(identity, status='lightnav_command', linear_x=.1, angular_z=0.)]
    trace.write_text('\n'.join(json.dumps({'parsed': r}) for r in records))
    assert not diagnostic_summary(trace)['commands_linked_to_inference']
    records.extend([dict(identity, status='lightnav_inference', response={'latency_ms': 50}),
                    dict(identity, status='path_accepted')])
    trace.write_text('\n'.join(json.dumps({'parsed': r}) for r in records))
    assert diagnostic_summary(trace)['commands_linked_to_inference']
    records[-1]['episode'] = 1
    trace.write_text('\n'.join(json.dumps({'parsed': r}) for r in records))
    assert not diagnostic_summary(trace)['commands_linked_to_inference']


def test_duplicate_capture_counts_are_scoped_to_model_session(tmp_path):
    trace = tmp_path / 'frames.jsonl'
    trace.write_text('\n'.join(json.dumps(dict(status='lightnav_inference',
        episode=episode, native_episode=session, response={'latency_ms': 10, 'capture_stamp_ns': stamp}))
        for episode, session, stamp in [(0, 1, 100), (0, 1, 100), (0, 1, 101),
                                        (0, 2, 100), (1, 3, 100)]))
    summary = diagnostic_summary(trace)
    assert summary['unique_image_count'] == 4
    assert summary['duplicate_capture_count'] == 1
    assert summary['missing_capture_stamp_count'] == 0
    assert len(summary['episode_frame_stats']) == 3
