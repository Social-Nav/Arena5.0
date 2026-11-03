"""CPU-only contracts and negative tests for the paper pipeline modules.

No ROS graph, simulator, model, or GPU is used.
"""
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from arena_bringup import paper_pipeline as pp


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _settings(track='A', **overrides):
    payload = {
        'schema': pp.SETTINGS_SCHEMA,
        'track': track,
        'scenario_mode': 'closedloop',
        'world': 'grscenes_20_v1',
        'scenario': 'default_2',
        'robot': 'Ai2_Bot2',
        'episodes': 1,
        'pedestrian_seed': 0,
        'sensor_rgb_topic': '/task_generator_node/Ai2_Bot2/head_camera/image',
        'ego_odom_topic': '/task_generator_node/Ai2_Bot2/odom',
        'sensor_only': track == 'B',
        'latency_ms': 0.0,
        'output_prefix': 'outputs/paper/a',
    }
    if track == 'A':
        payload['sensor_scan_topic'] = '/task_generator_node/Ai2_Bot2/laser_scan'
    if track == 'C':
        payload['teacher_oracle'] = True
    payload.update(overrides)
    return payload


def test_settings_a_valid():
    result = pp.validate_settings(_settings('A'))
    assert result['valid'] is True
    assert result['errors'] == []


def test_settings_b_rejects_scan_and_gt():
    result = pp.validate_settings(_settings('B', sensor_scan_topic='/scan', gt_map='x'))
    assert result['valid'] is False
    assert 'B:forbidden:sensor_scan_topic' in result['errors']
    assert 'B:forbidden:gt_map' in result['errors']


def test_settings_b_requires_sensor_only():
    result = pp.validate_settings(_settings('B', sensor_only=False))
    assert result['valid'] is False
    assert 'B:sensor_only_must_be_true' in result['errors']


def test_settings_c_requires_teacher_oracle():
    result = pp.validate_settings(_settings('C', teacher_oracle=False))
    assert result['valid'] is False
    assert 'C:teacher_oracle_must_be_true' in result['errors']


def test_settings_missing_and_bad_latency_fail_close():
    result = pp.validate_settings(_settings('A', world='', latency_ms=-1.0))
    assert result['valid'] is False
    assert 'missing:world' in result['errors']
    assert 'latency_ms' in result['errors']


# ---------------------------------------------------------------------------
# B strict RGB-only fast guard
# ---------------------------------------------------------------------------

def _proposal(action='goal', **overrides):
    payload = {
        'schema': pp.BFAST_SCHEMA,
        'action': action,
        'point': {
            'u': 0.5, 'v': 0.5, 'units': 'normalized', 'confidence': 0.9,
        },
    }
    if action == 'bearing':
        payload['bearing_rad'] = 0.1
    payload.update(overrides)
    return payload


def test_bfast_on_off_share_the_same_common_action():
    proposal = _proposal()
    executor = {'action': 'follow_path', 'poses': [[1.0, 2.0, 0.0], [2.0, 3.0, 0.0]]}
    on = pp.apply_bfast(proposal, enabled=True, executor_input=executor)
    off = pp.apply_bfast(proposal, enabled=False, executor_input=executor)
    assert on['executor_input'] == off['executor_input'] == executor
    assert on['proposal'] == off['proposal']
    assert on['fast_enabled'] is True and off['fast_enabled'] is False
    assert on['source'] == 'rgb_only_fast'
    assert off['source'] == 'rgb_only_passthrough'


def test_bfast_bare_point_and_bearing_are_unsupported_stop():
    bare = pp.apply_bfast(_proposal('goal'), enabled=True)
    assert bare['unsupported'] is True
    assert bare['unsupported_reason'] == 'UNSUPPORTED_NO_COMMON_ACTION'
    assert bare['executor_input'] == {'action': 'stop'}
    bearing = pp.apply_bfast(_proposal('bearing'), enabled=True,
                             executor_input={'action': 'bearing', 'bearing_rad': 0.1})
    assert bearing['unsupported'] is True
    assert bearing['unsupported_reason'] == 'UNSUPPORTED_BEARING_ACTION'
    assert bearing['executor_input'] == {'action': 'stop'}


def test_bfast_follow_path_requires_real_metric_poses():
    bad = pp.apply_bfast(_proposal(), enabled=True,
                         executor_input={'action': 'follow_path'})
    assert bad['unsupported_reason'] == 'UNSUPPORTED_NO_METRIC_PATH'
    assert bad['executor_input'] == {'action': 'stop'}
    nan_pose = pp.apply_bfast(_proposal(), enabled=True,
                              executor_input={'action': 'follow_path',
                                              'poses': [[float('nan'), 1.0]]})
    assert nan_pose['unsupported_reason'] == 'UNSUPPORTED_PATH_POSE_NOT_FINITE'


def test_bfast_unknown_action_schema_units_fail_close():
    for proposal in (
        _proposal(action='teleport'),
        _proposal(schema='wrong'),
        _proposal(point={'u': 0.5, 'v': 0.5, 'units': 'meters', 'confidence': 0.9}),
        _proposal(point={'u': float('nan'), 'v': 0.5, 'units': 'normalized', 'confidence': 0.9}),
    ):
        with pytest.raises(pp.PipelineRejected):
            pp.apply_bfast(proposal, enabled=True)


def test_bfast_forbids_scan_depth_gt_people():
    for key in ('scan', 'depth', 'gt_map', 'people', 'eval_goal'):
        with pytest.raises(pp.PipelineRejected):
            pp.apply_bfast(_proposal(**{key: {}}), enabled=True)


def test_bfast_stop_hold_error_map_to_common_executor():
    stop = pp.apply_bfast({'schema': pp.BFAST_SCHEMA, 'action': 'stop'}, enabled=True)
    hold = pp.apply_bfast({'schema': pp.BFAST_SCHEMA, 'action': 'hold'}, enabled=False)
    error = pp.apply_bfast({'schema': pp.BFAST_SCHEMA, 'action': 'error'}, enabled=True)
    assert stop['executor_input'] == {'action': 'stop'}
    assert hold['executor_input'] == {'action': 'hold'}
    assert error['executor_input'] == {'action': 'error'}


def test_bfast_topic_denylist_rejects_scan_depth_gt():
    result = pp.validate_bfast_topics([
        '/task_generator_node/Ai2_Bot2/head_camera/image',
        '/task_generator_node/Ai2_Bot2/laser_scan',
        '/task_generator_node/Ai2_Bot2/head_camera/depth',
        '/task_generator_node/Ai2_Bot2/people',
    ])
    assert result['valid'] is False
    assert len(result['forbidden_topics']) == 3


def _textured_frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(160)[None, :] % 50
    frame[:, :, 1] = np.arange(120)[:, None] % 50
    return frame


def test_flow_risk_approach_vs_lateral_vs_rotation():
    frame = _textured_frame()
    ys, xs = np.mgrid[0:120, 0:160]
    nx = xs - 79.5
    ny = ys - 59.5
    norm = np.sqrt(nx * nx + ny * ny)
    norm[norm < 1e-9] = 1.0

    def approach(prev, curr):
        return np.stack([nx / norm, ny / norm], axis=2) * 20.0

    def lateral(prev, curr):
        return np.stack([np.full((120, 160), 20.0),
                         np.zeros((120, 160))], axis=2)

    def rotation(prev, curr):
        return np.stack([-ny / norm, nx / norm], axis=2) * 20.0

    approach_result = pp.compute_flow_risk_proxy(
        frame, frame, dt_sec=0.1, risk_threshold=0.5, flow_backend=approach)
    lateral_result = pp.compute_flow_risk_proxy(
        frame, frame, dt_sec=0.1, risk_threshold=0.5, flow_backend=lateral)
    rotation_result = pp.compute_flow_risk_proxy(
        frame, frame, dt_sec=0.1, risk_threshold=0.5, flow_backend=rotation)
    assert approach_result['stop'] is True
    assert approach_result['radial_expansion'] > 0
    assert lateral_result['stop'] is False
    assert rotation_result['stop'] is False


def test_flow_risk_zero_texture_is_unknown_not_stop():
    flat = np.zeros((120, 160, 3), dtype=np.uint8)
    result = pp.compute_flow_risk_proxy(
        flat, flat, dt_sec=0.1,
        flow_backend=lambda a, b: np.zeros((120, 160, 2)))
    assert result['texture_state'] == 'UNKNOWN'
    assert result['stop'] is False
    assert result['reason'] == 'low_texture_unknown'


def test_flow_risk_fails_closed_on_bad_flow_or_shape():
    frame = _textured_frame()
    with pytest.raises(pp.PipelineRejected):
        pp.compute_flow_risk_proxy(frame, frame, dt_sec=0.0,
                                  flow_backend=lambda a, b: None)
    with pytest.raises(pp.PipelineRejected):
        pp.compute_flow_risk_proxy(frame, frame[:80], dt_sec=0.1,
                                  flow_backend=lambda a, b: np.zeros((120, 160, 2)))
    with pytest.raises(pp.PipelineRejected):
        pp.compute_flow_risk_proxy(frame, frame, dt_sec=0.1,
                                   flow_backend=lambda a, b: None)
    with pytest.raises(pp.PipelineRejected):
        pp.compute_flow_risk_proxy(frame, frame, dt_sec=0.1,
                                   flow_backend=lambda a, b: np.full((120, 160, 2), np.nan))


def test_flow_stop_only_vetoes_and_never_rewrites_the_goal():
    proposal = _proposal()
    executor = {'action': 'follow_path', 'poses': [[1.0, 2.0, 0.0]]}
    flow = {'stop': True, 'risk_score': 2.0}
    decision = pp.apply_bfast(proposal, enabled=True,
                              executor_input=executor, flow_risk=flow)
    assert decision['executor_input'] == {'action': 'stop'}
    assert decision['proposal']['point'] == proposal['point']
    off = pp.apply_bfast(proposal, enabled=False,
                         executor_input=executor, flow_risk=flow)
    assert off['executor_input'] == executor
    assert off['fast_veto'] is False


def _profile(arm='on'):
    on = arm == 'on'
    return {
        'schema': pp.BFAST_PROFILE_SCHEMA,
        'arm': arm,
        'guard_enabled': on,
        'guard_required': on,
        'rgb_topic': '/task_generator_node/Ai2_Bot2/head_camera/image',
        'veto_topic': '/task_generator_node/Ai2_Bot2/bfast/veto' if on else '',
        'decision_topic': '/task_generator_node/Ai2_Bot2/bfast/decision',
        'executor_veto_topic': '/task_generator_node/Ai2_Bot2/bfast/veto' if on else '',
        'executor_veto_timeout_sec': 1.0 if on else 0.0,
        'model': 'Qwen/Qwen3-VL-8B-Instruct-FP8',
        'model_revision': '9cdc6310a8cb770ce18efaf4e9935334512aee45',
        'prompt_sha256': 'c' * 64,
        'action_mapping': 'common_follow_path_stop_hold_error',
        'topic_allowlist': ['/task_generator_node/Ai2_Bot2/head_camera/image',
                            '/task_generator_node/Ai2_Bot2/odom'],
        'topic_denylist': ['/task_generator_node/Ai2_Bot2/laser_scan',
                           '/task_generator_node/Ai2_Bot2/head_camera/depth',
                           '/task_generator_node/Ai2_Bot2/people',
                           '/task_generator_node/Ai2_Bot2/global_costmap/costmap',
                           'gt_map'],
    }


def test_bfast_profile_off_on_identity_and_denylist():
    off = pp.validate_bfast_profile(_profile('off'))
    on = pp.validate_bfast_profile(_profile('on'))
    assert off['valid'] is True and on['valid'] is True
    assert on['identity_source'] == 'manifest'
    assert on['model'] == 'Qwen/Qwen3-VL-8B-Instruct-FP8'
    bad = _profile('off')
    bad['executor_veto_topic'] = '/veto'
    assert pp.validate_bfast_profile(bad)['valid'] is False
    no_prompt = _profile('on')
    no_prompt['prompt_sha256'] = ''
    assert pp.validate_bfast_profile(no_prompt)['valid'] is False
    leak = _profile('on')
    leak['topic_allowlist'].append('/task_generator_node/Ai2_Bot2/laser_scan')
    assert pp.validate_bfast_profile(leak)['valid'] is False


def test_bfast_plan_on_requires_thresholds_or_overrides():
    on = _profile('on')
    result = pp.build_bfast_plan(on)
    assert result['valid'] is False
    assert 'on:risk_threshold_required' in result['errors']
    assert 'on:max_frame_gap_sec_required' in result['errors']
    overridden = pp.build_bfast_plan(on, risk_threshold=0.5, max_frame_gap_sec=1.0)
    assert overridden['valid'] is True
    assert 'fast_guard_enabled:=true' in overridden['identity']['launch_arguments']
    assert 'bfast_risk_threshold:=0.5' in overridden['identity']['launch_arguments']
    off = pp.build_bfast_plan(_profile('off'))
    assert off['valid'] is True
    assert 'fast_guard_enabled:=false' in off['identity']['launch_arguments']


# ---------------------------------------------------------------------------
# Latency injection
# ---------------------------------------------------------------------------

SOURCE = {'image_sha256': 'a' * 64, 'request_id': 'r1'}


def test_latency_zero_and_delay_preserve_source_identity():
    zero = pp.inject_latency(
        source_stamp_ns=1000, source_frame_id='map', source_identity=SOURCE,
        latency_ms=0.0, now_sim_ns=2000, mode='passthrough')
    delay = pp.inject_latency(
        source_stamp_ns=1000, source_frame_id='map', source_identity=SOURCE,
        latency_ms=500.0, now_sim_ns=2000, mode='delay')
    assert zero['source_stamp_ns'] == 1000
    assert zero['source_frame_id'] == 'map'
    assert zero['source_identity'] == SOURCE
    assert zero['refreshed'] is False
    assert zero['release_time_sim_ns'] == 2000
    assert zero['queue_wait_ms'] == 0.0
    assert delay['source_age_ns'] == (2000 + 500_000_000) - 1000
    assert delay['source_unchanged'] is True
    assert delay['queue_wait_ms'] == 500.0
    with pytest.raises(pp.PipelineRejected):
        pp.inject_latency(source_stamp_ns=1000, source_frame_id='map',
                          source_identity=SOURCE, latency_ms=500.0,
                          now_sim_ns=2000, mode='passthrough')


def test_latency_drop_and_out_of_order_are_fail_closed():
    drop = pp.inject_latency(
        source_stamp_ns=1000, source_frame_id='map', source_identity=SOURCE,
        latency_ms=500.0, now_sim_ns=2000, mode='drop')
    ooo = pp.inject_latency(
        source_stamp_ns=1000, source_frame_id='map', source_identity=SOURCE,
        latency_ms=500.0, now_sim_ns=2000, mode='out_of_order',
        previous_response_sim_ns=2000)
    assert drop['dropped'] is True and drop['applied'] is False
    assert drop['response_sim_ns'] is None
    assert drop['release_time_sim_ns'] is None
    assert drop['dropped_reason'] == 'injected_drop'
    assert ooo['out_of_order'] is True and ooo['applied'] is False
    assert ooo['response_sim_ns'] < 2000
    assert ooo['release_policy'] == 'buffered_release_order'
    assert drop['source_stamp_ns'] == 1000 and ooo['source_stamp_ns'] == 1000


def test_latency_requires_frozen_source_and_rejects_negative():
    with pytest.raises(pp.PipelineRejected):
        pp.inject_latency(source_stamp_ns=None, source_frame_id='map',
                          source_identity=SOURCE, latency_ms=0.0, now_sim_ns=1)
    with pytest.raises(pp.PipelineRejected):
        pp.inject_latency(source_stamp_ns=1, source_frame_id='map',
                          source_identity={}, latency_ms=0.0, now_sim_ns=1)
    with pytest.raises(pp.PipelineRejected):
        pp.inject_latency(source_stamp_ns=1, source_frame_id='map',
                          source_identity=SOURCE, latency_ms=-1.0, now_sim_ns=1)


def test_release_queue_delay_drop_order_passthrough():
    second = 1_000_000_000
    clock = {'now': second}
    queue = pp.ReleaseQueue(now_sim_ns_fn=lambda: clock['now'],
                            now_monotonic_ns_fn=lambda: 0)
    passthrough = queue.submit(source_stamp_ns=500, source_frame_id='map',
                               source_identity=SOURCE, mode='passthrough',
                               latency_ms=0.0)
    assert queue.poll(second) == [passthrough]
    assert passthrough['source_stamp_ns'] == 500
    assert passthrough['refreshed'] is False

    delayed = queue.submit(source_stamp_ns=500, source_frame_id='map',
                           source_identity=SOURCE, mode='delay', latency_ms=500.0)
    assert queue.poll(second) == []
    released = queue.poll(second + 500_000_000)
    assert released == [delayed]
    assert delayed['queue_wait_at_release_ms'] == 500.0
    assert delayed['source_stamp_ns'] == 500

    dropped = queue.submit(source_stamp_ns=500, source_frame_id='map',
                           source_identity=SOURCE, mode='drop', latency_ms=500.0)
    assert dropped['dropped'] is True
    assert queue.pending() == 0
    assert queue.poll(second + 1_000_000_000) == []


def test_release_queue_out_of_order_keeps_buffered_order():
    second = 1_000_000_000
    clock = {'now': second}
    queue = pp.ReleaseQueue(now_sim_ns_fn=lambda: clock['now'],
                            now_monotonic_ns_fn=lambda: 0)
    first = queue.submit(source_stamp_ns=500, source_frame_id='map',
                         source_identity=SOURCE, mode='delay', latency_ms=100.0)
    ooo = queue.submit(source_stamp_ns=500, source_frame_id='map',
                       source_identity=SOURCE, mode='out_of_order',
                       latency_ms=500.0, previous_response_sim_ns=second + 500_000_000)
    released = queue.poll(second + 500_000_000)
    assert [item['queue_id'] for item in released] == [first['queue_id'],
                                                       ooo['queue_id']]
    assert ooo['out_of_order'] is True
    assert ooo['applied'] is False
    assert ooo['release_policy'] == 'buffered_release_order'


def test_latency_commit_gate_disabled_is_immediate_exact_once():
    calls = []
    gate = pp.LatencyCommitGate(mode='disabled')
    result, receipt = gate.commit(
        source_stamp_ns=1000, source_frame_id='map', source_identity=SOURCE,
        payload={'proposal': {'goal': 1}}, commit_fn=lambda payload: calls.append(payload) or {'ok': True})
    assert result == {'ok': True}
    assert calls == [{'proposal': {'goal': 1}}]
    assert receipt['commit_called'] is True and receipt['exact_once'] is True
    assert receipt['source_stamp_ns'] == 1000 and receipt['refreshed'] is False


def test_latency_gate_two_responses_arrival_order_can_differ_from_release_order():
    clock = {'now': 1_000_000_000}
    commits = []

    def commit_fn(payload):
        commits.append(payload)
        return {'ok': True}

    gate = pp.LatencyCommitGate(
        mode='delay', latency_ms=500.0,
        now_sim_ns_fn=lambda: clock['now'],
        now_monotonic_ns_fn=lambda: 0,
        sleep_fn=lambda seconds: None,
    )
    # First response has a long delay; second response has no delay and is
    # enqueued later but released first.  Arrival order != release order.
    gate.mode = 'delay'
    gate.latency_ms = 500.0
    first_item, first_submit = gate.enqueue(
        source_stamp_ns=100, source_frame_id='map', source_identity=SOURCE,
        payload={'proposal': {'id': 'first'}})
    gate.latency_ms = 0.0
    second_item, second_submit = gate.enqueue(
        source_stamp_ns=200, source_frame_id='map', source_identity=SOURCE,
        payload={'proposal': {'id': 'second'}})
    assert first_submit['phase'] == 'submit'
    assert second_submit['phase'] == 'submit'
    clock['now'] = 1_100_000_000
    released = gate.poll_ready(clock['now'], commit_fn)
    assert [item['queue_id'] for item, _result, _receipt in released] == [
        second_item['queue_id']]
    assert commits == [{'proposal': {'id': 'second'}}]
    clock['now'] = 1_600_000_000
    released = gate.poll_ready(clock['now'], commit_fn)
    assert [item['queue_id'] for item, _result, _receipt in released] == [
        first_item['queue_id']]
    assert commits == [{'proposal': {'id': 'second'}},
                       {'proposal': {'id': 'first'}}]
    assert all(item['source_unchanged'] for item, _result, _receipt in released)
    assert all(item['refreshed'] is False for item, _result, _receipt in released)
    assert len(commits) == 2


def test_latency_commit_gate_delay_drop_and_exact_once():
    clock = {'now': 1_000_000_000}
    calls = []
    gate = pp.LatencyCommitGate(
        mode='delay', latency_ms=500.0,
        now_sim_ns_fn=lambda: clock['now'],
        now_monotonic_ns_fn=lambda: 0,
        sleep_fn=lambda seconds: clock.__setitem__('now', clock['now'] + 100_000_000),
    )
    result, receipt = gate.commit(
        source_stamp_ns=500_000_000, source_frame_id='map', source_identity=SOURCE,
        payload={'proposal': {'goal': 2}},
        commit_fn=lambda payload: calls.append(payload) or {'ok': True})
    assert result == {'ok': True}
    assert calls == [{'proposal': {'goal': 2}}]
    assert receipt['commit_called'] is True and receipt['exact_once'] is True
    assert receipt['source_stamp_ns'] == 500_000_000
    assert receipt['refreshed'] is False

    drop_calls = []
    drop_gate = pp.LatencyCommitGate(
        mode='drop', latency_ms=500.0,
        now_sim_ns_fn=lambda: clock['now'],
        now_monotonic_ns_fn=lambda: 0,
        sleep_fn=lambda seconds: None,
    )
    result, receipt = drop_gate.commit(
        source_stamp_ns=500_000_000, source_frame_id='map', source_identity=SOURCE,
        payload={'proposal': {'goal': 3}},
        commit_fn=lambda payload: drop_calls.append(payload) or {'ok': True})
    assert result['decision'] == 'dropped'
    assert drop_calls == []
    assert receipt['commit_called'] is False
    assert receipt['dropped'] is True
    assert receipt['source_stamp_ns'] == 500_000_000


def test_derived_run_leaves_source_hashes_unchanged(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'odom.csv').write_text('odom')
    (source / 'metrics.csv').write_text('old metrics')
    (source / 'videos').mkdir()
    (source / 'videos' / 'x.mp4').write_text('video')
    derived = pp.prepare_derived_run(source, tmp_path / 'derived')
    assert (derived / 'odom.csv').read_text() == 'odom'
    assert not (derived / 'metrics.csv').exists()
    assert (derived / 'videos').is_dir()
    assert not (derived / 'videos').is_symlink()
    assert (derived / 'videos' / 'x.mp4').read_text() == 'video'
    before = pp.source_hash_manifest(source)
    (derived / 'metrics.csv').write_text('new metrics')
    after = pp.source_hash_manifest(source)
    assert before == after
    incident = pp.write_metrics_overwrite_incident(source, 'in-place command')
    payload = json.loads(incident.read_text())
    assert payload['exact_backup_available'] is False
    assert payload['restore_claimed'] is False
    assert 'metrics.csv' in payload['overwritten_files']


# ---------------------------------------------------------------------------
# Scenario profiles
# ---------------------------------------------------------------------------

def _scenario(mode='closedloop', **overrides):
    payload = {
        'schema': pp.SCENARIO_PROFILE_SCHEMA,
        'mode': mode,
        'world': 'grscenes_20_v1',
        'scenario': 'default_2',
        'robot': 'Ai2_Bot2',
        'episodes': 1,
        'pedestrian_seed': 0,
        'human_source': 'none' if mode == 'static' else 'hunav',
        'dynamic_humans': mode != 'static',
        'robot_human_feedback': mode == 'closedloop',
    }
    if mode in ('openloop', 'closedloop'):
        payload['hunav_config'] = 'default_sf60.yaml'
    payload.update(overrides)
    return payload


def test_scenario_static_closedloop_supported_openloop_unsupported():
    static = pp.validate_scenario_profile(_scenario('static'))
    closed = pp.validate_scenario_profile(_scenario('closedloop'))
    openloop = pp.validate_scenario_profile(_scenario('openloop'))
    assert static['valid'] is True and static['runtime_status'] == 'SUPPORTED'
    assert closed['valid'] is True and closed['runtime_status'] == 'SUPPORTED'
    assert openloop['valid'] is True and openloop['runtime_status'] == 'UNSUPPORTED'
    assert 'scripted-human' in openloop['runtime_reason']


def test_scenario_feedback_mismatch_and_static_hunav_fail_close():
    bad = pp.validate_scenario_profile(_scenario('openloop', robot_human_feedback=True))
    assert bad['valid'] is False
    assert 'openloop:robot_human_feedback_must_be_False' in bad['errors']
    static_hunav = pp.validate_scenario_profile(_scenario('static', hunav_config='x'))
    assert static_hunav['valid'] is False
    assert 'static:hunav_config_must_be_absent' in static_hunav['errors']


def test_scenario_closedloop_pins_hunav_config_path_and_sha():
    profile = _scenario('closedloop')
    profile['hunav_config'] = 'arena_bringup/configs/hunav/default_sf60.yaml'
    profile['hunav_config_sha256'] = (
        'a834b6bb42e6c8f14ea4734aa5ab1fd03791a1897f9445d24c2fb7ccade9eeb1')
    result = pp.validate_scenario_profile(profile)
    assert result['valid'] is True
    assert result['resolved_hunav_config'].endswith(
        'arena_bringup/configs/hunav/default_sf60.yaml')
    assert Path(result['resolved_hunav_config']).is_file()
    command = pp.scenario_command(profile)
    assert command['env']['ARENA_HUNAV_CONFIG'] == result['resolved_hunav_config']
    bad = dict(profile)
    bad['hunav_config_sha256'] = '0' * 64
    assert pp.validate_scenario_profile(bad)['valid'] is False
    assert 'hunav_config_sha256' in pp.validate_scenario_profile(bad)['errors']


def test_scenario_command_uses_env_for_hunav_and_dummy_for_static():
    closed = pp.scenario_command(_scenario('closedloop'))
    assert closed['argv'][:4] == ['ros2', 'run', 'arena_bringup', 'internnav_eval']
    assert '--human' in closed['argv'] and 'hunav' in closed['argv']
    assert closed['env']['ARENA_HUNAV_CONFIG'].endswith(
        'arena_bringup/configs/hunav/default_sf60.yaml')
    assert Path(closed['env']['ARENA_HUNAV_CONFIG']).is_file()
    assert closed['social_eval_is_metrics_only'] is True
    assert closed['feedback_source'] == 'hunav_sfm'
    static = pp.scenario_command(_scenario('static'))
    assert 'dummy' in static['argv']
    assert '--expected-human-count' in static['argv']
    assert static['env'] == {}
    with pytest.raises(pp.PipelineRejected):
        pp.scenario_command(_scenario('openloop'))


# ---------------------------------------------------------------------------
# Dataset and teacher manifests
# ---------------------------------------------------------------------------

def _dataset():
    return {
        'schema': pp.DATASET_SCHEMA,
        'split': 'paper_v1',
        'cases': [{
            'case_id': 'grscenes_20_v1/default_2',
            'world': 'grscenes_20_v1',
            'scenario': 'default_2',
            'robot': 'Ai2_Bot2',
            'instruction': 'walk to the second table',
            'pedestrian_seed': 0,
        }],
    }


def test_dataset_valid_and_missing_duplicate_gt_fail_close():
    assert pp.validate_dataset_manifest(_dataset())['valid'] is True
    missing = _dataset()
    missing['cases'][0]['world'] = ''
    assert pp.validate_dataset_manifest(missing)['valid'] is False
    duplicate = _dataset()
    duplicate['cases'].append(dict(duplicate['cases'][0]))
    assert any('duplicate' in e for e in pp.validate_dataset_manifest(duplicate)['errors'])
    gt = _dataset()
    gt['cases'][0]['gt_map'] = 'x'
    assert any('forbidden:gt_map' in e for e in pp.validate_dataset_manifest(gt)['errors'])


def test_dataset_semantic_split_leakage():
    payload = _dataset()
    payload['cases'][0]['semantics'] = {
        'scene_id': 'scene_1', 'task_id': 'task_1', 'assets': ['asset_1']}
    payload['cases'].append({
        'case_id': 'grscenes_20_v1/default_3',
        'world': 'grscenes_20_v1', 'scenario': 'default_3', 'robot': 'Ai2_Bot2',
        'instruction': 'walk to the third table', 'pedestrian_seed': 0,
        'semantics': {'scene_id': 'scene_1', 'task_id': 'task_2', 'assets': ['asset_1']},
    })
    payload['splits'] = {'train': ['grscenes_20_v1/default_2'],
                         'val': ['grscenes_20_v1/default_3'],
                         'test': []}
    result = pp.validate_dataset_manifest(payload)
    assert result['valid'] is False
    assert 'scene_id:scene_1:train/val' in result['leakage']
    assert 'asset:asset_1:train/val' in result['leakage']


def _teacher():
    return {
        'schema': pp.TEACHER_SCHEMA,
        'source_track': 'C',
        'source_runs': [{
            'run_id': 'gt_socialmpc_run_1',
            'run_label': 'GT_C',
            'manifest_sha256': 'b' * 64,
            'source_files': [
                {'path': 'human_states.csv', 'sha256': 'c' * 64},
            ],
        }],
        'inputs': [
            {'name': 'global_map', 'source': 'gt_map', 'track': 'C'},
            {'name': 'people', 'source': 'gt_people', 'track': 'C'},
        ],
        'splits': {'train': ['scene_1'], 'val': ['scene_2'], 'test': ['scene_3']},
        'semantic_splits': {
            'assets': {'train': ['asset_1'], 'val': ['asset_2'], 'test': ['asset_3']},
            'tasks': {'train': ['task_1'], 'val': ['task_2'], 'test': ['task_3']},
        },
        'instruction_qc': {'status': 'required', 'min_chars': 3,
                           'max_chars': 200, 'language': 'en'},
        'scene_count': None,
        'episode_count': None,
    }


def test_teacher_valid_and_counts_not_frozen():
    result = pp.validate_teacher_manifest(_teacher())
    assert result['valid'] is True
    assert result['scene_count'] is None
    assert result['episode_count'] is None
    assert result['provenance_status'] == 'OK'


def test_teacher_missing_provenance_is_typed_fail_close():
    missing = _teacher()
    missing['source_runs'] = ['gt_socialmpc_run_1']
    result = pp.validate_teacher_manifest(missing)
    assert result['valid'] is False
    assert result['missing_source_sample'] is True
    assert any('missing_provenance' in error for error in result['errors'])


def test_teacher_leakage_and_qc_not_set_fail_close():
    leaked = _teacher()
    leaked['splits']['val'] = ['scene_1']
    result = pp.validate_teacher_manifest(leaked)
    assert result['valid'] is False
    assert 'scene_1:train/val' in result['leakage']
    semantic = _teacher()
    semantic['semantic_splits']['assets']['val'] = ['asset_1']
    result = pp.validate_teacher_manifest(semantic)
    assert result['valid'] is False
    assert 'assets:asset_1:train/val' in result['leakage']
    not_set = _teacher()
    not_set['instruction_qc'] = {'status': 'not_set'}
    result = pp.validate_teacher_manifest(not_set)
    assert result['valid'] is False
    assert result['qc_not_set'] is True
    assert 'instruction_qc:not_set' in result['errors']


def _source_run(tmp_path, *, controller_people, model_adapter, model_server):
    source = tmp_path / 'source_run'
    source.mkdir()
    (source / 'run_manifest.yaml').write_text(
        'model_backend: lisn_qwen3vl_fp8\n'
        'launch_command:\n'
        f'  - "model_adapter:={model_adapter}"\n'
        f'  - "model_server_url:={model_server}"\n')
    (source / 'CONTROLLER_PEOPLE.json').write_text(json.dumps({
        'schema': 'arena.lisn.controller_people.v1', 'people': controller_people}))
    (source / 'HUMAN_PEOPLE_BRIDGE.json').write_text(json.dumps({
        'schema': 'arena.lisn.hunav_people_bridge.v1',
        'people': [{'name': 'hunav_1'}, {'name': 'hunav_2'}],
        'published_count': 8331}))
    for name in ('human_states.csv', 'pedsim_agents_data.csv', 'odom.csv', 'start_goal.csv'):
        (source / name).write_text('t,value\n0,1\n')
    return source


def test_source_run_audit_detects_controller_zero_and_internnav(tmp_path):
    source = _source_run(tmp_path, controller_people=0,
                         model_adapter='internnav',
                         model_server='http://127.0.0.1:5801/eval_dual')
    receipt, path = pp.audit_source_run(source, tmp_path / 'derived')
    assert receipt['blockers'] == ['CONTROLLER_PEOPLE_ZERO',
                                   'EXECUTOR_IDENTITY_INTERNNAV']
    assert receipt['controller_people'] == 0
    assert receipt['bridge_people'] == 2
    assert receipt['published_count'] == 8331
    assert receipt['executor_identity'] == 'InternNav'
    assert receipt['gt_available_for_evaluation'] is True
    assert receipt['gt_consumed_by_controller'] is False
    assert receipt['source_eligible'] is False
    assert receipt['teacher_export_created'] is False
    assert receipt['old_gt_labels_preserved'] is True
    assert receipt['source_unchanged'] is True
    assert path.is_file()


def test_source_run_audit_empty_source_is_ineligible(tmp_path):
    source = tmp_path / 'empty_source'
    source.mkdir()
    receipt, _path = pp.audit_source_run(source, tmp_path / 'derived')
    assert 'MISSING_SOURCE_FILES' in receipt['blockers']
    assert 'UNKNOWN_CONTROLLER_PEOPLE' in receipt['blockers']
    assert 'UNKNOWN_EXECUTOR_IDENTITY' in receipt['blockers']
    assert receipt['source_eligible'] is False


def test_source_run_audit_unknown_controller_and_executor(tmp_path):
    source = _source_run(tmp_path, controller_people=None,
                         model_adapter='', model_server='')
    (source / 'CONTROLLER_PEOPLE.json').write_text(json.dumps({
        'schema': 'arena.lisn.controller_people.v1', 'people': None}))
    (source / 'run_manifest.yaml').write_text('model_backend: unknown\n')
    receipt, _path = pp.audit_source_run(source, tmp_path / 'derived')
    assert 'UNKNOWN_CONTROLLER_PEOPLE' in receipt['blockers']
    assert 'UNKNOWN_EXECUTOR_IDENTITY' in receipt['blockers']
    assert receipt['source_eligible'] is False


def test_source_run_audit_hash_mismatch_is_ineligible(tmp_path, monkeypatch):
    source = _source_run(tmp_path, controller_people=2,
                         model_adapter='lisn_qwen3vl_fp8', model_server='')
    original = pp._file_digest
    counter = {'n': 0}

    def flaky(path):
        digest = original(path)
        counter['n'] += 1
        digest['sha256'] = f'{counter["n"]:064x}'
        return digest

    monkeypatch.setattr(pp, '_file_digest', flaky)
    receipt, _path = pp.audit_source_run(source, tmp_path / 'derived')
    assert 'SOURCE_HASH_MISMATCH' in receipt['blockers']
    assert receipt['source_unchanged'] is False
    assert receipt['source_eligible'] is False


def test_source_run_audit_eligible_when_controller_people_and_qwen_executor(tmp_path):
    source = _source_run(tmp_path, controller_people=2,
                         model_adapter='lisn_qwen3vl_fp8',
                         model_server='')
    receipt, _path = pp.audit_source_run(source, tmp_path / 'derived')
    assert receipt['blockers'] == []
    assert receipt['executor_identity'] == 'LISN/Qwen'
    assert receipt['gt_consumed_by_controller'] is True
    assert receipt['source_eligible'] is True


def test_teacher_audit_typed_blocker():
    missing = _teacher()
    missing['source_runs'] = ['gt_run_without_provenance']
    result = pp.audit_teacher_manifest(missing)
    assert result['typed_blocker'] == 'TEACHER_SOURCE_SAMPLE_MISSING'
    assert result['old_gt_labels_preserved'] is True
    assert result['missing_source_sample'] is True
    valid = pp.audit_teacher_manifest(_teacher())
    assert valid['valid'] is True
    assert valid['typed_blocker'] is None
    relabeled = _teacher()
    relabeled['source_track'] = 'B'
    result = pp.audit_teacher_manifest(relabeled)
    assert 'NON_GT_MIXING' in result['typed_blocker']


def test_teacher_rejects_gt_mixed_into_b_and_non_gt_label():
    mixed = _teacher()
    mixed['source_track'] = 'B'
    result = pp.validate_teacher_manifest(mixed)
    assert result['valid'] is False
    assert 'non_gt_mixing' in result['errors']
    relabeled = _teacher()
    relabeled['source_runs'][0]['run_label'] = 'non_gt'
    result = pp.validate_teacher_manifest(relabeled)
    assert result['valid'] is False
    assert any('non_gt_label' in error for error in result['errors'])


# ---------------------------------------------------------------------------
# Metrics wrapper fail-close contract
# ---------------------------------------------------------------------------

def _plan():
    return [
        ('stage_a_metrics', ['a']),
        ('stage_b_vln_task_metrics', ['b']),
        ('stage_c_social_metrics', ['c']),
        ('validation', ['validation']),
    ]


def _run(codes):
    calls = []

    def runner(command):
        calls.append(tuple(command))
        return codes[command[0]]

    result = pp.run_metrics_pipeline(run_dir='/tmp/run', stage_runner=runner,
                                     commands=_plan())
    return result, calls


def test_metrics_stage_a_rc1_stops_immediately():
    result, calls = _run({'a': 1, 'b': 0, 'c': 0, 'validation': 0})
    assert result['status'] == 'fault'
    assert result['failed_stage'] == 'stage_a_metrics'
    assert result['validation_executed'] is False
    assert calls == [('a',)]


def test_metrics_stage_b_rc2_stops_immediately():
    result, calls = _run({'a': 0, 'b': 2, 'c': 0, 'validation': 0})
    assert result['failed_stage'] == 'stage_b_vln_task_metrics'
    assert result['fault'] is True
    assert calls == [('a',), ('b',)]


def test_metrics_stage_c_rc1_stops_immediately():
    result, calls = _run({'a': 0, 'b': 0, 'c': 1, 'validation': 0})
    assert result['failed_stage'] == 'stage_c_social_metrics'
    assert calls == [('a',), ('b',), ('c',)]


def test_metrics_validation_not_executed_is_fault():
    result = pp.run_metrics_pipeline(
        run_dir='/tmp/run', stage_runner=lambda c: 0,
        commands=[('stage_a_metrics', ['a']), ('stage_b_vln_task_metrics', ['b'])]
    )
    assert result['status'] == 'fault'
    assert result['validation_executed'] is False
    assert result['reason'] == 'validation_not_executed'
    assert pp.metrics_exit_code(result) == 2


def test_metrics_validation_rc1_is_engineering_closure():
    result, _ = _run({'a': 0, 'b': 0, 'c': 0, 'validation': 1})
    assert result['status'] == 'engineering_closed_scientific_fail'
    assert result['validation_executed'] is True
    assert result['validation_rc'] == 1
    assert result['scientific_fail'] is True
    assert result['fault'] is False
    assert pp.metrics_exit_code(result) == 1


def test_metrics_validation_rc2_is_fault():
    result, _ = _run({'a': 0, 'b': 0, 'c': 0, 'validation': 2})
    assert result['status'] == 'fault'
    assert result['failed_stage'] == 'validation'
    assert result['validation_rc'] == 2
    assert pp.metrics_exit_code(result) == 2


def test_metrics_all_zero_is_pass():
    result, _ = _run({'a': 0, 'b': 0, 'c': 0, 'validation': 0})
    assert result['status'] == 'pass'
    assert result['validation_executed'] is True
    assert pp.metrics_exit_code(result) == 0


# ---------------------------------------------------------------------------
# Unique prefix / Docker endpoint
# ---------------------------------------------------------------------------

def test_unique_output_prefix_does_not_overwrite(tmp_path):
    target = tmp_path / 'run'
    assert pp.unique_output_prefix(str(target), now=1.0) == str(target)
    target.mkdir()
    first = pp.unique_output_prefix(str(target), now=1.0)
    assert first != str(target)
    Path(first).mkdir()
    second = pp.unique_output_prefix(str(target), now=1.0)
    assert second not in (str(target), first)


def _agentic_example(name):
    return json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval' / 'examples' / name)
        .read_text())


def _make_run_manifest(run_dir, *, model_backend='internnav',
                       profile_id='internnav_grscenes',
                       profile_sha256=None,
                       timeout=300, control_path='nav2_follow_path_rpp',
                       controller='', seed=0, episodes=1):
    if profile_sha256 is None:
        profile_sha256 = hashlib.sha256(
            (Path(__file__).parents[2] / 'arena_bringup' / 'configs'
             / 'benchmark' / 'profiles' / 'internnav_grscenes.yaml')
            .read_bytes()).hexdigest()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'run_manifest.yaml').write_text(
        'model_backend: ' + model_backend + '\n'
        'model_provenance:\n'
        f'  model: {model_backend}\n'
        'parameters:\n'
        '  world: grscenes_20_v1\n'
        '  scenario_file: default_2\n'
        '  robot: Ai2_Bot2\n'
        f'  pedestrian_seed: {seed}\n'
        f'  episodes: {episodes}\n'
        f'  timeout: {timeout}\n'
        f'  control_path: {control_path}\n'
        f'  controller: {controller}\n'
        '  vln_rgb_topic: head_camera/image\n'
        '  eval_video_scan_topic: /task_generator_node/Ai2_Bot2/scan\n'
        'benchmark_profile:\n'
        f'  id: {profile_id}\n'
        f'  sha256: {profile_sha256}\n')
    (run_dir / 'odom.csv').write_text('odom')


def _qwen_a_request():
    return json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval' / 'examples'
         / 'qwen_a_single_case.json').read_text())


def test_agentic_request_generic_executable():
    result = pp.validate_agentic_request(_agentic_example('generic_internnav_single_case.json'))
    assert result['valid'] is True
    assert result['executable'] is True
    assert result['commands']
    assert result['blockers'] == []
    assert result['setting_status'] == 'GENERIC_INTERNAV_SMOKE_ONLY'


def test_agentic_request_strict_b_blocked():
    result = pp.validate_agentic_request(_agentic_example('strict_b_single_case.json'))
    assert result['valid'] is True
    assert result['executable'] is False
    assert result['commands'] == []
    assert 'QWEN_LISN_COMMON_METRIC_ACTION_BRIDGE_MISSING' in result['blockers']
    assert result['setting_status'] == 'SMOKE_ONLY_NOT_PAPER_SETTING'


def test_agentic_request_missing_field_and_oracle_blocked():
    payload = _agentic_example('generic_internnav_single_case.json')
    payload.pop('case')
    assert pp.validate_agentic_request(payload)['valid'] is False
    oracle = _agentic_example('generic_internnav_single_case.json')
    oracle['track'] = 'oracle'
    result = pp.validate_agentic_request(oracle)
    assert result['executable'] is False
    assert 'ORACLE_NOT_OPERATOR_EXECUTABLE' in result['blockers']


def _make_bag(run_dir, topics):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'recording').mkdir()
    db3 = run_dir / 'recording' / 'recording_0.db3'
    connection = sqlite3.connect(db3)
    connection.execute('CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT)')
    connection.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, data BLOB)')
    for topic_id, name, type_name, count in topics:
        connection.execute('INSERT INTO topics VALUES (?, ?, ?)', (topic_id, name, type_name))
        for index in range(count):
            connection.execute('INSERT INTO messages (topic_id, data) VALUES (?, ?)',
                               (topic_id, b'x'))
    connection.commit()
    connection.close()
    return db3


def test_evidence_index_bag_playback_blocked_and_ready(tmp_path):
    run = tmp_path / 'run'
    _make_bag(run, [
        (1, '/task_generator_node/Ai2_Bot2/odom', 'nav_msgs/msg/Odometry', 3),
        (2, '/task_generator_node/Ai2_Bot2/cmd_vel', 'geometry_msgs/msg/Twist', 2),
        (3, '/task_generator_node/human_states', 'hunav_msgs/msg/Agents', 3),
    ])
    (run / 'benchmark_result.json').write_text('{}')
    (run / 'metrics.csv').write_text('t,v\n0,1\n')
    index = pp.build_evidence_index(run, tmp_path / 'blocked.json')
    assert index['playback_status'] == 'BLOCKED'
    assert 'RGB_NOT_IN_BAG' in index['blockers']
    assert 'CAMERA_INFO_NOT_IN_BAG' in index['blockers']
    assert 'LASER_SCAN_ZERO_MESSAGES' in index['blockers']
    assert index['bag']['topics'][1]['messages'] == 3

    ready_run = tmp_path / 'ready_run'
    _make_bag(ready_run, [
        (1, '/task_generator_node/Ai2_Bot2/head_camera/image', 'sensor_msgs/msg/Image', 2),
        (2, '/task_generator_node/Ai2_Bot2/head_camera/camera_info', 'sensor_msgs/msg/CameraInfo', 2),
        (3, '/task_generator_node/Ai2_Bot2/laser_scan', 'sensor_msgs/msg/LaserScan', 2),
    ])
    ready = pp.build_evidence_index(ready_run, tmp_path / 'ready.json')
    assert ready['playback_status'] == 'READY'
    assert ready['blockers'] == []


def test_evidence_index_refuses_existing_output(tmp_path):
    run = tmp_path / 'run'
    _make_bag(run, [])
    output = tmp_path / 'out.json'
    output.write_text('{}')
    with pytest.raises(pp.PipelineRejected):
        pp.build_evidence_index(run, output)


def test_evidence_index_refuses_output_inside_run(tmp_path):
    run = tmp_path / 'run'
    _make_bag(run, [])
    with pytest.raises(pp.PipelineRejected):
        pp.build_evidence_index(run, run / 'evidence.json')


def _make_episode_videos(run, *, external=None):
    run.mkdir(parents=True, exist_ok=True)
    episode = run / 'videos' / 'episode_0000'
    episode.mkdir(parents=True)
    ego = episode / 'ego_observation.mp4'
    top_down = episode / 'map_top_down_follow.mp4'
    sim = episode / 'sim_top_down.mp4'
    for path in (ego, top_down, sim):
        path.write_bytes(b'video')
    pts = episode / 'video_frame_pts.json'
    pts.write_text(json.dumps({
        'schema': 'arena.eval.video_frame_pts.v1',
        'views': {'ego': [{'frame': 0}], 'sim_top_down': [{'frame': 0}]},
    }))
    entry = {
        'episode': 0,
        'directory': str(episode),
        'ego_video': str(ego),
        'top_down_video': str(top_down),
        'map_top_down_video': str(top_down),
        'debug_overlay_video': None,
        'sim_top_down_video': str(sim),
        'video_frame_pts': 'video_frame_pts.json',
    }
    if external is not None:
        entry['ego_video'] = str(external)
    (run / 'video_index.json').write_text(json.dumps({
        'schema': 'arena.eval.video_index.v1',
        'episodes': [entry],
    }))
    return ego, top_down, sim


def test_evidence_index_episodes_structure_aliases_and_rgb_only(tmp_path):
    run = tmp_path / 'run'
    ego, top_down, sim = _make_episode_videos(run)
    _make_bag(run, [
        (1, '/task_generator_node/Ai2_Bot2/head_camera/image',
         'sensor_msgs/msg/Image', 2),
        (2, '/task_generator_node/Ai2_Bot2/head_camera/camera_info',
         'sensor_msgs/msg/CameraInfo', 2),
        (3, '/task_generator_node/Ai2_Bot2/scan',
         'sensor_msgs/msg/LaserScan', 0),
    ])
    index = pp.build_evidence_index(
        run, tmp_path / 'out.json', sensor_profile='rgb_only')
    assert index['playback_status'] == 'READY'
    assert 'LASER_SCAN_ZERO_MESSAGES' not in index['blockers']
    assert index['videos']['episode_0/ego_video']['present'] is True
    assert index['videos']['episode_0/top_down_video']['aliases'] == [
        'episode_0/map_top_down_video']
    assert index['videos']['episode_0/top_down_video']['video_frame_pts']['present'] is True
    assert index['videos']['episode_0/ego_video']['ownership'] == 'RUN_RELATIVE'
    assert index['video_frame_pts']['episode_0']['schema'] == (
        'arena.eval.video_frame_pts.v1')

    strict = pp.build_evidence_index(
        run, tmp_path / 'strict.json', sensor_profile='rgb_lidar')
    assert strict['playback_status'] == 'BLOCKED'
    assert 'LASER_SCAN_ZERO_MESSAGES' in strict['blockers']
    assert strict['bag']['laser_scan_messages'] == 0


def test_evidence_index_external_video_and_parent_decode(tmp_path):
    run = tmp_path / 'run'
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(b'video')
    _make_episode_videos(run, external=outside)
    _make_bag(run, [])
    (tmp_path / 'DECODE_v2.json').write_text(json.dumps({
        'schema': 'arena.prox_diag_decode.v2',
        'ego': {'path': str(run / 'videos' / 'episode_0000' / 'ego_observation.mp4')},
    }))
    index = pp.build_evidence_index(run, tmp_path / 'out.json')
    assert any(blocker.startswith('VIDEO_PATH_OUTSIDE_RUN:')
               for blocker in index['artifact_blockers'])
    assert index['videos']['episode_0/ego_video']['ownership'] == 'EXTERNAL'
    assert index['decode']['status'] == 'AVAILABLE'
    assert index['decode']['records']['PARENT:DECODE_v2.json']['references_run'] is True


def test_agentic_request_binds_robot_seed_and_profile():
    result = pp.validate_agentic_request(_agentic_example('generic_internnav_single_case.json'))
    assert result['executable'] is True
    command = result['commands'][0]
    assert command[1] == 'run'
    assert command[command.index('--profile') + 1].endswith(
        'internnav_grscenes.yaml')
    assert command[command.index('--episodes') + 1] == '1'
    assert result['identity']['runtime']['http_port'] == 5801
    assert result['identity']['profile']['profile_id'] == 'internnav_grscenes'
    assert result['env']['ARENA_WS_DIR'] == '/home/ubuntu/arena_jazzy_ws'
    assert result['env']['ARENA_DIR'] == result['identity']['arena_dir']
    assert result['env']['ARENA_HUNAV_CONFIG'].endswith('default_sf60.yaml')
    assert result['env']['ARENA_HUNAV_CONFIG_SHA256'] == (
        'a834b6bb42e6c8f14ea4734aa5ab1fd03791a1897f9445d24c2fb7ccade9eeb1')

    changed = _agentic_example('generic_internnav_single_case.json')
    changed['case'] = dict(changed['case'])
    changed['case']['robot'] = 'Go2'
    changed['seed'] = 42
    changed_cmd = pp.validate_agentic_request(changed)['commands'][0]
    assert changed_cmd[changed_cmd.index('--robot') + 1] == 'Go2'
    assert changed_cmd[changed_cmd.index('--pedestrian-seed') + 1] == '42'


def test_agentic_request_model_mismatch_and_unsupported_modes():
    payload = _agentic_example('generic_internnav_single_case.json')
    payload['model'] = 'not_internnav'
    result = pp.validate_agentic_request(payload)
    assert result['executable'] is False
    assert result['commands'] == []
    assert 'MODEL_PROFILE_MISMATCH' in result['blockers']

    payload = _agentic_example('generic_internnav_single_case.json')
    payload['model_revision'] = 'wrong'
    result = pp.validate_agentic_request(payload)
    assert 'MODEL_REVISION_PROFILE_MISMATCH' in result['blockers']

    for mode in ('audit', 'full_eval'):
        payload = _agentic_example('generic_internnav_single_case.json')
        payload['mode'] = mode
        result = pp.validate_agentic_request(payload)
        assert result['executable'] is False
        assert result['commands'] == []
        assert f'UNSUPPORTED_MODE:{mode}' in result['blockers']


def test_agentic_request_rejects_bad_episodes_and_unsafe_paths():
    payload = _agentic_example('generic_internnav_single_case.json')
    payload['episodes'] = 2
    result = pp.validate_agentic_request(payload)
    assert result['valid'] is False
    assert 'episodes:must_be_one' in result['errors']

    payload = _agentic_example('generic_internnav_single_case.json')
    payload['episodes'] = True
    assert pp.validate_agentic_request(payload)['valid'] is False

    payload = _agentic_example('generic_internnav_single_case.json')
    payload['request_id'] = '../evil'
    assert pp.validate_agentic_request(payload)['valid'] is False

    payload = _agentic_example('generic_internnav_single_case.json')
    payload['profile'] = '../etc/passwd'
    result = pp.validate_agentic_request(payload)
    assert result['valid'] is False
    assert 'profile:path_traversal' in result['errors']


def test_agentic_generated_command_parses_without_running(tmp_path):
    payload = _agentic_example('generic_internnav_single_case.json')
    payload['arena_dir'] = str(pp.REPO_ROOT)
    payload['source_binding'] = str(pp.REPO_ROOT)
    result = pp.validate_agentic_request(payload)
    assert result['executable'] is True
    command = result['commands'][0]
    plan = [command[0], 'plan', 'run', *command[2:]]
    environment = {
        'PATH': '/usr/local/bin:/usr/bin:/bin',
        'HOST_ARENA_WS_DIR': '/home/ubuntu/arena_jazzy_ws',
    }
    completed = pp.subprocess.run(
        plan, capture_output=True, text=True, env=environment, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert 'nothing above was executed' in completed.stdout
    assert 'robot=Go2' not in completed.stdout
    assert 'robot=Ai2_Bot2' in completed.stdout


def test_agentic_schema_rejects_bad_integer_and_paths():
    jsonschema = pytest.importorskip('jsonschema')
    schema = json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval'
         / 'operator_request.v1.schema.json').read_text())
    for name in ('generic_internnav_single_case.json', 'strict_b_single_case.json'):
        jsonschema.validate(_agentic_example(name), schema)
    bad = _agentic_example('generic_internnav_single_case.json')
    bad['episodes'] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
    bad = _agentic_example('generic_internnav_single_case.json')
    bad['episodes'] = 2
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
    bad = _agentic_example('generic_internnav_single_case.json')
    bad['request_id'] = '../evil'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
    bad = _agentic_example('generic_internnav_single_case.json')
    bad['profile'] = '../etc/passwd'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_metrics_derived_root_inside_source_rejected(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'odom.csv').write_text('odom')
    with pytest.raises(pp.PipelineRejected):
        pp.prepare_derived_run(source, source / 'derived')


def test_metrics_source_tree_guard_detects_new_file(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'odom.csv').write_text('odom')
    before = pp.source_hash_manifest(source)
    (source / 'unexpected.txt').write_text('x')
    after = pp.source_hash_manifest(source)
    assert pp._tree_changes(before, after) == ['unexpected.txt']


def test_metrics_cli_never_writes_source_incident(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'odom.csv').write_text('odom')
    incident = source / 'METRICS_OVERWRITE_INCIDENT.json'
    incident.write_text('{"historical": true}')
    incident_hash = pp._file_digest(incident)['sha256']

    def fake_pipeline(*, run_dir, **_kwargs):
        return {
            'schema': pp.METRICS_SCHEMA,
            'status': 'pass',
            'failed_stage': None,
            'validation_executed': True,
            'validation_rc': 0,
            'scientific_fail': False,
            'fault': False,
            'stages': [],
        }

    monkeypatch.setattr(pp, 'run_metrics_pipeline', fake_pipeline)
    rc = pp._cmd_metrics(str(source), str(tmp_path / 'derived'))
    assert rc == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted['source_unchanged'] is True
    assert emitted['historical_metrics_incident']['read_only'] is True
    assert pp._file_digest(incident)['sha256'] == incident_hash
    assert not (source / 'metrics.csv').exists()
    derived = Path(emitted['derived_run'])
    assert (derived / 'odom.csv').read_text() == 'odom'
    assert (derived / 'PAPER_METRICS_RECEIPT.json').is_file()


def test_agentic_lifecycle_cli_all_actions(tmp_path, capsys):
    payload = _agentic_example('generic_internnav_single_case.json')
    request_file = tmp_path / 'request.json'
    request_file.write_text(json.dumps(payload))
    for action in ('plan', 'start', 'cancel', 'finalize'):
        capsys.readouterr()
        rc = pp.main(['agentic-lifecycle', '--file', str(request_file),
                      '--action', action])
        assert rc in (0, 2), action
        assert json.loads(capsys.readouterr().out)['action'] == action

    run_dir = tmp_path / 'run'
    _make_run_manifest(run_dir)
    payload['run_dir'] = str(run_dir)
    payload['output_root'] = str(tmp_path / 'out')
    request_file.write_text(json.dumps(payload))
    for action in ('read', 'audit'):
        capsys.readouterr()
        rc = pp.main(['agentic-lifecycle', '--file', str(request_file),
                      '--action', action])
        assert rc == 0, action
        result = json.loads(capsys.readouterr().out)
        assert result['action'] == action
        assert result['executable'] is True
        assert result['run_identity']['model'] == 'internnav'


def test_agentic_lifecycle_bound_actions_and_blockers(tmp_path):
    payload = _agentic_example('generic_internnav_single_case.json')
    plan = pp.build_operator_lifecycle(payload, 'plan')
    assert plan['state'] == 'planned'
    assert plan['executable'] is True
    assert plan['commands'][0][1] == 'plan'
    assert plan['allowed_next_actions'] == ['start', 'cancel', 'read', 'audit']
    assert plan['exit_taxonomy'] == {
        'ok': 0, 'scientific_fail': 1, 'contract_or_evidence_blocked': 2}
    assert plan['idempotency_key'].startswith('sha256:')

    start = pp.build_operator_lifecycle(payload, 'start')
    assert start['state'] == 'blocked'
    assert start['executable'] is False
    assert start['commands'] == []
    assert 'START_RUNTIME_IDEMPOTENCY_GUARD_MISSING' in start['blockers']

    cancel = pp.build_operator_lifecycle(payload, 'cancel')
    assert cancel['state'] == 'cancelled'
    assert cancel['executable'] is True
    assert cancel['commands'][0][1] == 'stop'

    run_dir = tmp_path / 'run'
    _make_run_manifest(run_dir)
    for action, state in (('read', 'read_ready'), ('audit', 'audit_ready')):
        request = dict(payload)
        request['run_dir'] = str(run_dir)
        request['output_root'] = str(tmp_path / f'out_{action}')
        result = pp.build_operator_lifecycle(request, action)
        assert result['state'] == state
        assert result['executable'] is True
        assert result['commands']
        assert result['run_identity']['model'] == 'internnav'
        assert result['provenance']['run_identity_digest'].startswith('sha256:')
        assert result['provenance_path'] == str(
            tmp_path / f'out_{action}' / 'provenance.json')
        if action == 'audit':
            assert result['audit_manifest']['schema'] == (
                'arena.agentic.audit_manifest.v1')
            assert result['audit_manifest']['provenance'] == result['provenance']
            assert result['audit_manifest']['run_dir'] == str(run_dir)

    finalize = pp.build_operator_lifecycle(payload, 'finalize')
    assert finalize['state'] == 'blocked'
    assert finalize['executable'] is False
    assert finalize['commands'] == []
    assert 'ACTION_NOT_BOUND:finalize' in finalize['blockers']

    invalid = pp.build_operator_lifecycle(payload, 'plan', from_state='started')
    assert invalid['executable'] is False
    assert 'INVALID_STATE_TRANSITION:started->plan' in invalid['blockers']

    strict_b = pp.build_operator_lifecycle(
        _agentic_example('strict_b_single_case.json'), 'plan')
    assert strict_b['executable'] is False
    assert strict_b['commands'] == []
    assert 'QWEN_LISN_COMMON_METRIC_ACTION_BRIDGE_MISSING' in strict_b['blockers']


def test_lifecycle_qwen_a_positive_and_internnav_negative(tmp_path):
    qwen_run = tmp_path / 'qwen_run'
    _make_run_manifest(
        qwen_run, model_backend='lisn_qwen3vl_fp8',
        profile_id='internnav_follow_path_grscenes',
        profile_sha256='f2ef7d2b506092a03fed3dbc9432468fef0ee537eb93d83a133b4bb6dcf32b89',
        timeout=59, control_path='nav2_follow_path_prox_mpc',
        controller='prox_mpc')
    qwen_request = _qwen_a_request()
    qwen_request['run_dir'] = str(qwen_run)
    qwen_request['output_root'] = str(tmp_path / 'qwen_out')
    plan = pp.build_operator_lifecycle(qwen_request, 'plan')
    assert plan['executable'] is False
    assert plan['commands'] == []
    assert 'AUDIT_ONLY_TRACK_NOT_LAUNCHABLE' in plan['blockers']
    assert 'PROFILE_INVALID' not in ' '.join(plan['blockers'])
    assert 'MODEL_PROFILE_MISMATCH' not in ' '.join(plan['blockers'])

    positive = pp.build_operator_lifecycle(qwen_request, 'read')
    assert positive['executable'] is True
    assert positive['blockers'] == []
    assert positive['run_identity']['track'] == 'qwen_prox_a'
    assert positive['run_identity']['model'] == 'Qwen/Qwen3-VL-8B-Instruct-FP8'
    assert positive['run_identity']['model_revision'] == (
        '9cdc6310a8cb770ce18efaf4e9935334512aee45')
    assert positive['run_identity']['model_backend'] == 'lisn_qwen3vl_fp8'
    assert positive['run_identity']['controller'] == 'prox_mpc'

    internnav_request = _agentic_example('generic_internnav_single_case.json')
    internnav_request['run_dir'] = str(qwen_run)
    internnav_request['output_root'] = str(tmp_path / 'internnav_out')
    negative = pp.build_operator_lifecycle(internnav_request, 'read')
    assert negative['executable'] is False
    assert negative['commands'] == []
    assert 'RUN_IDENTITY_MISMATCH:track' in negative['blockers']
    assert 'RUN_IDENTITY_MISMATCH:model' in negative['blockers']
    assert 'RUN_IDENTITY_MISMATCH:model_revision' in negative['blockers']
    assert 'RUN_IDENTITY_MISMATCH:profile_sha256' in negative['blockers']

    internnav_profile_as_qwen = _qwen_a_request()
    internnav_profile_as_qwen['track'] = 'generic_internnav'
    internnav_profile_as_qwen['profile'] = (
        'arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml')
    validation = pp.validate_agentic_request(internnav_profile_as_qwen)
    assert validation['valid'] is True
    assert 'MODEL_PROFILE_MISMATCH' in validation['blockers']
    internnav_profile_as_qwen['run_dir'] = str(qwen_run)
    internnav_profile_as_qwen['output_root'] = str(tmp_path / 'mixed_out')
    blocked = pp.build_operator_lifecycle(internnav_profile_as_qwen, 'read')
    assert blocked['executable'] is False
    assert blocked['commands'] == []
    assert 'MODEL_PROFILE_MISMATCH' in blocked['blockers']


def test_agentic_lifecycle_idempotency_and_output_root_guards(tmp_path):
    payload = _agentic_example('generic_internnav_single_case.json')
    run_dir = tmp_path / 'run'
    _make_run_manifest(run_dir)
    payload['run_dir'] = str(run_dir)
    payload['output_root'] = str(tmp_path / 'out')
    first = pp.build_operator_lifecycle(payload, 'read')
    second = pp.build_operator_lifecycle(payload, 'read')
    assert first == second
    assert first['idempotency_key'] == second['idempotency_key']
    assert first['commands'] == second['commands']
    assert first['output_root'] == second['output_root']

    existing = tmp_path / 'existing'
    existing.mkdir()
    payload['output_root'] = str(existing)
    result = pp.build_operator_lifecycle(payload, 'read')
    assert result['executable'] is False
    assert result['commands'] == []
    assert any('output_root:exists' in item for item in result['errors'])

    payload['output_root'] = str(run_dir / 'inside')
    result = pp.build_operator_lifecycle(payload, 'read')
    assert result['executable'] is False
    assert result['commands'] == []
    assert any('output_root:inside_run' in item for item in result['errors'])

    start_one = pp.build_operator_lifecycle(payload, 'start')
    start_two = pp.build_operator_lifecycle(payload, 'start')
    assert start_one['commands'] == [] == start_two['commands']
    assert start_one['blockers'] == start_two['blockers']


def _audit_provenance(run_dir, *, request_id='audit-test'):
    identity = pp.extract_run_identity(run_dir)
    return pp.build_run_provenance(request_id, identity), identity


def _request_identity_from(identity):
    return {
        key: identity.get(key) for key in (
            'track', 'model', 'model_revision', 'case', 'seed', 'episodes',
            'action_control', 'sensor_profile')
    }


def test_audit_consolidate_requires_provenance_and_classifies(tmp_path):
    run_dir = tmp_path / 'run'
    _make_run_manifest(run_dir)
    provenance, identity = _audit_provenance(run_dir)
    receipts = tmp_path / 'receipts'
    receipts.mkdir()
    benchmark = receipts / 'benchmark_result.json'
    benchmark.write_text(json.dumps({
        'provenance': provenance,
        'verdict': {'benchmark_ready': True, 'benchmark_success': True},
    }))
    evidence = receipts / 'evidence_verdict.json'
    evidence.write_text(json.dumps({
        'provenance': provenance,
        'verdict': 'artifact_complete',
    }))
    metrics = receipts / 'metrics_receipt.json'
    metrics.write_text(json.dumps({
        'provenance': provenance,
        'status': 'pass',
    }))
    manifest = receipts / 'manifest.json'
    manifest.write_text(json.dumps({
        'schema': 'arena.agentic.audit_manifest.v1',
        'request_id': 'audit-test',
        'track': 'generic_internnav',
        'run_dir': str(run_dir),
        'output_root': str(receipts),
        'provenance': provenance,
        'request_identity': _request_identity_from(identity),
        'receipts': [
            {'name': 'benchmark_result', 'path': str(benchmark), 'required': True},
            {'name': 'evidence_verdict', 'path': str(evidence), 'required': True},
            {'name': 'metrics_receipt', 'path': str(metrics), 'required': True},
        ],
    }))
    result = pp.build_audit_consolidated(json.loads(manifest.read_text()))
    assert result['verdict'] == 'artifact_complete'
    assert result['exit_code'] == 0
    assert result['blockers'] == []

    evidence.write_text(json.dumps({
        'provenance': provenance, 'verdict': 'playback_blocked'}))
    result = pp.build_audit_consolidated(json.loads(manifest.read_text()))
    assert result['verdict'] == 'playback_blocked'
    assert result['exit_code'] == 2

    evidence.write_text(json.dumps({
        'provenance': provenance, 'verdict': 'scientific_fail'}))
    result = pp.build_audit_consolidated(json.loads(manifest.read_text()))
    assert result['verdict'] == 'scientific_fail'
    assert result['exit_code'] == 1

    metrics.write_text(json.dumps({
        'provenance': provenance, 'status': 'fault'}))
    evidence.write_text(json.dumps({
        'provenance': provenance, 'verdict': 'artifact_complete'}))
    result = pp.build_audit_consolidated(json.loads(manifest.read_text()))
    assert result['verdict'] == 'contract_blocked'
    assert result['exit_code'] == 2
    assert 'METRICS_STAGE_FAULT' in result['reasons']

    manifest_payload = json.loads(manifest.read_text())
    manifest_payload['receipts'] = manifest_payload['receipts'][:1]
    result = pp.build_audit_consolidated(manifest_payload)
    assert result['verdict'] == 'contract_blocked'
    assert any(blocker.startswith('RECEIPT_MISSING:')
               for blocker in result['blockers'])


def test_audit_consolidate_minimal_and_forged_cross_run_fail(tmp_path):
    receipts = tmp_path / 'receipts'
    receipts.mkdir()
    benchmark = receipts / 'benchmark_result.json'
    benchmark.write_text(json.dumps({
        'verdict': {'benchmark_ready': True, 'benchmark_success': True},
    }))
    evidence = receipts / 'evidence_verdict.json'
    evidence.write_text(json.dumps({'verdict': 'artifact_complete'}))
    metrics = receipts / 'metrics_receipt.json'
    metrics.write_text(json.dumps({'status': 'pass'}))
    minimal_manifest = {
        'schema': 'arena.agentic.audit_manifest.v1',
        'request_id': 'audit-minimal',
        'track': 'generic_internnav',
        'receipts': [
            {'name': 'benchmark_result', 'path': str(benchmark), 'required': True},
            {'name': 'evidence_verdict', 'path': str(evidence), 'required': True},
            {'name': 'metrics_receipt', 'path': str(metrics), 'required': True},
        ],
    }
    result = pp.build_audit_consolidated(minimal_manifest)
    assert result['verdict'] == 'contract_blocked'
    assert 'AUDIT_MANIFEST_PROVENANCE_MISSING' in result['blockers']
    assert 'AUDIT_MANIFEST_RUN_DIR_MISSING' in result['blockers']
    assert 'AUDIT_MANIFEST_REQUEST_IDENTITY_MISSING' in result['blockers']

    run_dir = tmp_path / 'run'
    _make_run_manifest(run_dir)
    provenance, identity = _audit_provenance(run_dir)
    other_run = tmp_path / 'other_run'
    _make_run_manifest(other_run, seed=7)
    other_provenance, _ = _audit_provenance(other_run, request_id='audit-test')
    benchmark.write_text(json.dumps({
        'provenance': provenance,
        'verdict': {'benchmark_ready': True, 'benchmark_success': True},
    }))
    evidence.write_text(json.dumps({
        'provenance': other_provenance,
        'verdict': 'artifact_complete',
    }))
    metrics.write_text(json.dumps({
        'provenance': provenance, 'status': 'pass'}))
    manifest = {
        'schema': 'arena.agentic.audit_manifest.v1',
        'request_id': 'audit-cross-run',
        'track': 'generic_internnav',
        'run_dir': str(run_dir),
        'output_root': str(receipts),
        'provenance': provenance,
        'request_identity': _request_identity_from(identity),
        'receipts': [
            {'name': 'benchmark_result', 'path': str(benchmark), 'required': True},
            {'name': 'evidence_verdict', 'path': str(evidence), 'required': True},
            {'name': 'metrics_receipt', 'path': str(metrics), 'required': True},
        ],
    }
    result = pp.build_audit_consolidated(manifest)
    assert result['verdict'] == 'contract_blocked'
    assert any(
        blocker.startswith('RECEIPT_evidence_verdict:PROVENANCE_AUTHENTICITY_MISMATCH')
        for blocker in result['blockers'])

    forged = dict(provenance)
    forged['run_identity'] = dict(provenance['run_identity'])
    forged['run_identity']['model'] = 'forged-model'
    forged['run_identity_digest'] = 'sha256:' + 'f' * 64
    benchmark.write_text(json.dumps({
        'provenance': forged,
        'verdict': {'benchmark_ready': True, 'benchmark_success': True},
    }))
    evidence.write_text(json.dumps({
        'provenance': forged, 'verdict': 'artifact_complete'}))
    metrics.write_text(json.dumps({
        'provenance': forged, 'status': 'pass'}))
    forged_manifest = dict(manifest)
    forged_manifest['provenance'] = forged
    result = pp.build_audit_consolidated(forged_manifest)
    assert result['verdict'] == 'contract_blocked'
    assert any(
        blocker.startswith('AUDIT_MANIFEST:PROVENANCE_AUTHENTICITY_MISMATCH')
        for blocker in result['blockers'])


def test_producers_reject_forged_provenance(tmp_path):
    run_dir = tmp_path / 'run'
    _make_run_manifest(run_dir)
    provenance, _identity = _audit_provenance(run_dir)
    forged = dict(provenance)
    forged['run_identity'] = dict(provenance['run_identity'])
    forged['run_identity']['model'] = 'forged-model'
    forged['run_identity_digest'] = 'sha256:' + 'f' * 64

    authenticity = pp.verify_provenance_authenticity(forged)
    assert authenticity['valid'] is False
    assert any(
        error.startswith('PROVENANCE_AUTHENTICITY_MISMATCH')
        for error in authenticity['errors'])

    with pytest.raises(pp.PipelineRejected):
        pp.build_evidence_index(
            run_dir, tmp_path / 'index.json', provenance=forged)

    real_index = pp.build_evidence_index(
        run_dir, tmp_path / 'real_index.json', provenance=provenance)
    assert real_index['provenance']['run_identity_digest'] == (
        provenance['run_identity_digest'])
    with pytest.raises(pp.PipelineRejected):
        pp.write_evidence_verdict(
            tmp_path / 'real_index.json', tmp_path / 'verdict.json',
            provenance=forged)

    with pytest.raises(pp.PipelineRejected):
        pp.audit_source_run(run_dir, tmp_path / 'source_audit',
                            provenance=forged)

    forged_file = tmp_path / 'forged_provenance.json'
    forged_file.write_text(json.dumps(forged))
    rc = pp._cmd_metrics(str(run_dir), str(tmp_path / 'derived'),
                         str(forged_file))
    assert rc == 2


def test_audit_consolidated_schema_validates(tmp_path):
    jsonschema = pytest.importorskip('jsonschema')
    schema = json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval'
         / 'audit_consolidated.v1.schema.json').read_text())
    evidence = tmp_path / 'evidence_verdict.json'
    evidence.write_text(json.dumps({'verdict': 'artifact_complete'}))
    manifest = {
        'schema': 'arena.agentic.audit_manifest.v1',
        'request_id': 'audit-schema',
        'track': 'generic_internnav',
        'receipts': [
            {'name': 'evidence_verdict', 'path': str(evidence), 'required': True},
        ],
    }
    result = pp.build_audit_consolidated(manifest)
    jsonschema.validate(result, schema)


def _recording_contract(name):
    return json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval' / 'recording'
         / name).read_text())


def test_recording_contract_track_sensor_split():
    a = pp.validate_recording_contract(_recording_contract('a_rgb_lidar.json'))
    b = pp.validate_recording_contract(
        _recording_contract('strict_b_rgb_only.json'))
    assert a['valid'] is True
    assert b['valid'] is True
    assert 'laser_scan' in a['identity']['required_types']
    assert 'laser_scan' not in b['identity']['required_types']

    bad = _recording_contract('strict_b_rgb_only.json')
    bad['required_topics'] = dict(bad['required_topics'])
    bad['required_topics']['rgb'] = ['/task_generator_node/Ai2_Bot2/laser_scan']
    result = pp.validate_recording_contract(bad)
    assert result['valid'] is False
    assert any('strict_b_forbidden_topic' in item for item in result['errors'])

    bad = _recording_contract('a_rgb_lidar.json')
    bad['open_mode'] = 'rw'
    result = pp.validate_recording_contract(bad)
    assert 'open_mode:must_be_ro' in result['errors']


def test_recording_preflight_a_blocked_and_b_rgb_only(tmp_path):
    a_contract = _recording_contract('a_rgb_lidar.json')
    a_run = tmp_path / 'a_run'
    _make_bag(a_run, [
        (1, '/task_generator_node/Ai2_Bot2/head_camera/image',
         'sensor_msgs/msg/Image', 2),
        (2, '/task_generator_node/Ai2_Bot2/head_camera/camera_info',
         'sensor_msgs/msg/CameraInfo', 2),
        (3, '/task_generator_node/Ai2_Bot2/laser_scan',
         'sensor_msgs/msg/LaserScan', 0),
    ])
    result = pp.recording_preflight(
        a_contract, a_run, tmp_path / 'a_preflight.json')
    assert result['playback_status'] == 'BLOCKED'
    assert result['current_run_status'] == 'BLOCKED'
    assert 'RECORDING_MESSAGES_BELOW_MIN:/task_generator_node/Ai2_Bot2/laser_scan' in result['blockers']

    b_contract = _recording_contract('strict_b_rgb_only.json')
    b_run = tmp_path / 'b_run'
    _make_bag(b_run, [
        (1, '/task_generator_node/Ai2_Bot2/head_camera/image',
         'sensor_msgs/msg/Image', 2),
        (2, '/task_generator_node/Ai2_Bot2/head_camera/camera_info',
         'sensor_msgs/msg/CameraInfo', 2),
    ])
    result = pp.recording_preflight(
        b_contract, b_run, tmp_path / 'b_preflight.json')
    assert result['playback_status'] == 'READY'
    assert result['blockers'] == []

    with pytest.raises(pp.PipelineRejected):
        pp.recording_preflight(a_contract, a_run, a_run / 'inside.json')


def test_recording_contract_and_preflight_schemas():
    jsonschema = pytest.importorskip('jsonschema')
    base = Path(__file__).parents[2] / 'configs' / 'agentic_eval'
    contract_schema = json.loads(
        (base / 'recording_contract.v1.schema.json').read_text())
    preflight_schema = json.loads(
        (base / 'recording_preflight.v1.schema.json').read_text())
    for name in ('a_rgb_lidar.json', 'strict_b_rgb_only.json'):
        jsonschema.validate(_recording_contract(name), contract_schema)
    # preflight schema is exercised by the CLI receipt below; validate a
    # synthetic result shape without touching a real run.
    synthetic = {
        'schema': 'arena.agentic.recording_preflight.v1',
        'contract_schema': 'arena.agentic.recording_contract.v1',
        'track': 'A',
        'sensor_profile': 'rgb_lidar',
        'run_dir': '/tmp/run',
        'bag_path': '/tmp/run/recording/recording_0.db3',
        'bag_present': True,
        'bag_sha256': 'a' * 64,
        'open_mode': 'ro',
        'checks': [{
            'sensor': 'rgb', 'topic': '/image', 'required_types': [
                'sensor_msgs/msg/Image'], 'min_messages': 1,
            'present': True, 'type': 'sensor_msgs/msg/Image', 'messages': 2,
            'status': 'PASS',
        }],
        'playback_status': 'READY',
        'blockers': [],
        'contract_status': 'CONTRACT_ONLY',
        'current_run_status': 'READY',
        'note': 'Preflight only; it never creates missing recording evidence.',
    }
    jsonschema.validate(synthetic, preflight_schema)


def test_agentic_lifecycle_schema_validates_response():
    jsonschema = pytest.importorskip('jsonschema')
    schema = json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval'
         / 'operator_lifecycle.v1.schema.json').read_text())
    example = json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval' / 'examples'
         / 'operator_lifecycle_plan.example.json').read_text())
    jsonschema.validate(example, schema)
    bad = dict(example)
    bad['state'] = 'ready'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def _write_metrics(run, *, task_reached, social_success, timed_out=False):
    (run / 'benchmark_result.json').write_text('{}')
    (run / 'run_manifest.yaml').write_text('model: test\n')
    (run / 'metrics.csv').write_text('t,v\n0,1\n')
    (run / 'vln_task_metrics.json').write_text(json.dumps({
        'goal': {'goal_reached': task_reached},
        'episode_timing': {'timed_out': timed_out},
    }))
    (run / 'social_metrics.json').write_text(json.dumps({
        'social_success': social_success,
    }))
    (run / 'artifact_validation.json').write_text(json.dumps({
        'overall_pass': True,
    }))


def _ready_bag(run):
    _make_bag(run, [
        (1, '/task_generator_node/Ai2_Bot2/head_camera/image',
         'sensor_msgs/msg/Image', 2),
        (2, '/task_generator_node/Ai2_Bot2/head_camera/camera_info',
         'sensor_msgs/msg/CameraInfo', 2),
        (3, '/task_generator_node/Ai2_Bot2/laser_scan',
         'sensor_msgs/msg/LaserScan', 2),
    ])


def test_evidence_verdict_classes_and_unknown_never_pass(tmp_path):
    run = tmp_path / 'complete'
    _make_episode_videos(run)
    _ready_bag(run)
    _write_metrics(run, task_reached=True, social_success=True)
    index = pp.build_evidence_index(run, tmp_path / 'complete.json')
    verdict = index['verdict']
    assert verdict['verdict'] == 'artifact_complete'
    assert verdict['flags']['artifact_complete'] is True
    assert pp.evidence_verdict_exit_code(verdict) == 0

    failing = tmp_path / 'failing'
    _make_episode_videos(failing)
    _ready_bag(failing)
    _write_metrics(failing, task_reached=False, social_success=True,
                   timed_out=True)
    fail_index = pp.build_evidence_index(failing, tmp_path / 'failing.json')
    fail_verdict = fail_index['verdict']
    assert fail_verdict['verdict'] == 'scientific_fail'
    assert fail_verdict['flags']['scientific_fail'] is True
    assert pp.evidence_verdict_exit_code(fail_verdict) == 1

    blocked = tmp_path / 'blocked'
    _make_episode_videos(blocked)
    _make_bag(blocked, [])
    _write_metrics(blocked, task_reached=True, social_success=True)
    blocked_index = pp.build_evidence_index(blocked, tmp_path / 'blocked.json')
    assert blocked_index['verdict']['verdict'] == 'playback_blocked'
    assert pp.evidence_verdict_exit_code(blocked_index['verdict']) == 2

    unknown = tmp_path / 'unknown'
    _make_episode_videos(unknown)
    _ready_bag(unknown)
    unknown_index = pp.build_evidence_index(unknown, tmp_path / 'unknown.json')
    assert unknown_index['verdict']['verdict'] == 'contract_blocked'
    assert 'METRICS_EVIDENCE_UNKNOWN' in (
        unknown_index['verdict']['contract_reasons'])
    assert unknown_index['verdict']['flags']['artifact_complete'] is False


def test_evidence_verdict_is_deterministic_and_schema_valid(tmp_path):
    jsonschema = pytest.importorskip('jsonschema')
    run = tmp_path / 'run'
    _make_episode_videos(run)
    _ready_bag(run)
    _write_metrics(run, task_reached=True, social_success=True)
    index_path = tmp_path / 'index.json'
    pp.build_evidence_index(run, index_path)
    first = pp.write_evidence_verdict(index_path, tmp_path / 'v1.json')
    second = pp.write_evidence_verdict(index_path, tmp_path / 'v2.json')
    assert first == second
    schema = json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval'
         / 'evidence_verdict.v1.schema.json').read_text())
    jsonschema.validate(first, schema)
    example = json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval' / 'examples'
         / 'evidence_verdict_playback_blocked.example.json').read_text())
    jsonschema.validate(example, schema)


def _common_contract():
    return json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval' / 'methods'
         / 'common_contract.v1.json').read_text())


def _method_bundle(method_id):
    return json.loads(
        (Path(__file__).parents[2] / 'configs' / 'agentic_eval' / 'methods'
         / f'{method_id}.v1.json').read_text())


def test_method_common_contract_and_bundles_validate():
    common = _common_contract()
    validation = pp.validate_common_contract(common)
    assert validation['valid'] is True
    assert validation['contract_hash'].startswith('sha256:')
    for method_id in pp.METHOD_IDS:
        bundle = pp.validate_method_bundle(_method_bundle(method_id))
        assert bundle['valid'] is True, (method_id, bundle['errors'])
    assert _method_bundle('qwen_agent_proxmpc')['role'] == 'qwen_prox_a'
    assert _method_bundle('qwen_agent_proxmpc')['status'] == 'BLOCKED'
    assert _method_bundle('strict_b_on')['status'] == 'BLOCKED'
    assert _method_bundle('socialmpc_oracle')['status'] == 'BLOCKED'


def test_method_plan_switch_changes_only_private_fields():
    common = _common_contract()
    plans = {}
    for method_id in ('internnav', 'navila', 'lightnav', 'qwen_agent_proxmpc'):
        plans[method_id] = pp.build_method_plan(
            method_id, common, _method_bundle(method_id))
    common_hashes = {plan['common_contract_hash'] for plan in plans.values()}
    assert len(common_hashes) == 1
    private_hashes = {plan['method_private_hash'] for plan in plans.values()}
    assert len(private_hashes) == len(plans)
    for method_id in ('internnav', 'navila', 'lightnav'):
        plan = plans[method_id]
        assert plan['status'] == 'READY'
        assert plan['executable'] is False
        assert plan['commands'] == []
        assert plan['future_launch_command'] == []
        assert plan['plan_only'] is True
        assert plan['common_case_blockers']
    assert plans['internnav']['current_profile_drift'] is True
    assert 'METHOD_BUNDLE_CURRENT_PROFILE_DRIFT' in plans['internnav']['blockers']
    qwen = plans['qwen_agent_proxmpc']
    assert qwen['status'] == 'BLOCKED'
    assert qwen['executable'] is False
    assert qwen['commands'] == []
    assert 'QWEN_AGENT_PROXMPC_RUNTIME_PROFILE_MISSING' in qwen['blockers']
    assert qwen['role'] == 'qwen_prox_a'


def test_current_profile_drift_and_common_case_block_vln_methods():
    common = _common_contract()
    plans = {method_id: pp.build_method_plan(
        method_id, common, _method_bundle(method_id))
        for method_id in pp.METHOD_IDS}
    for method_id in ('internnav', 'navila', 'lightnav'):
        assert plans[method_id]['executable'] is False
        assert plans[method_id]['common_case_blockers']
    assert 'METHOD_BUNDLE_CURRENT_PROFILE_DRIFT' in plans['internnav']['blockers']
    assert plans['internnav']['current_profile_sha256'] != (
        plans['internnav']['pinned_profile_sha256'])
    for method_id in ('qwen_agent_proxmpc', 'strict_b_on', 'strict_b_off',
                      'socialmpc_oracle'):
        assert plans[method_id]['status'] == 'BLOCKED'
        assert plans[method_id]['executable'] is False
        assert plans[method_id]['commands'] == []
    comparison = pp.build_method_plan_comparison(list(plans.values()))
    assert comparison['ok'] is False
    assert set(comparison['ready_methods_blocked']) == {
        'internnav', 'navila', 'lightnav'}
    assert comparison['common_hash_identical'] is True
    assert comparison['private_hashes_distinct'] is True


def test_consistent_current_profile_can_be_ready(tmp_path):
    common = dict(_common_contract())
    common['case'] = {'world': 'grscenes_1', 'scenario': 'eval_1',
                      'robot': 'Ai2_Bot2'}
    bundle = _method_bundle('internnav')
    profile_path = (Path(__file__).parents[2] / 'arena_bringup' / 'configs'
                    / 'benchmark' / 'profiles' / 'internnav_grscenes.yaml')
    current_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    bundle['profile'] = dict(bundle['profile'])
    bundle['profile']['sha256'] = current_sha
    bundle['source_commit'] = '8bb1896'
    plan = pp.build_method_plan('internnav', common, bundle)
    assert plan['executable'] is True
    assert plan['current_profile_drift'] is False
    assert plan['common_case_blockers'] == []
    assert plan['blockers'] == []
    assert plan['commands'] == []


def test_method_plan_comparison_common_hash_and_conflicts():
    common = _common_contract()
    plans = [pp.build_method_plan(method_id, common, _method_bundle(method_id))
             for method_id in pp.METHOD_IDS]
    comparison = pp.build_method_plan_comparison(plans)
    assert comparison['ok'] is False
    assert comparison['ready_methods_blocked'] == [
        'internnav', 'navila', 'lightnav']
    assert comparison['common_hash_identical'] is True
    assert comparison['private_hashes_distinct'] is True
    assert comparison['conflict_blockers'] == []
    assert any(item.startswith('METHOD_SHARED:namespace:')
               for item in comparison['conflict_warnings'])

    conflicting = list(plans)
    for plan in conflicting:
        if plan['method_id'] == 'navila':
            plan = dict(plan)
            plan['ports'] = [5801]
            plan['output_prefix'] = 'internnav_eval'
            plan['topics'] = {
                'status': '/task_generator_node/Ai2_Bot2/internnav/status',
            }
            conflicting[conflicting.index(
                next(item for item in conflicting
                     if item['method_id'] == 'navila'))] = plan
    comparison = pp.build_method_plan_comparison(conflicting)
    assert comparison['ok'] is False
    assert any(item.startswith('METHOD_CONFLICT:port:')
               for item in comparison['conflict_blockers'])
    assert any(item.startswith('METHOD_CONFLICT:topic:')
               for item in comparison['conflict_blockers'])


def test_method_bundle_override_common_fails():
    bundle = _method_bundle('internnav')
    bundle['case'] = {'world': 'other', 'scenario': 'x', 'robot': 'y'}
    result = pp.validate_method_bundle(bundle)
    assert result['valid'] is False
    assert 'METHOD_OVERRIDES_COMMON:case' in result['errors']

    plan = pp.build_method_plan('not_a_method', _common_contract(), bundle)
    assert plan['executable'] is False
    assert plan['commands'] == []
    assert 'METHOD_UNKNOWN:not_a_method' in plan['blockers']


def test_method_plan_schemas_validate():
    jsonschema = pytest.importorskip('jsonschema')
    base = Path(__file__).parents[2] / 'configs' / 'agentic_eval' / 'methods'
    common_schema = json.loads(
        (base / 'common_contract.v1.schema.json').read_text())
    bundle_schema = json.loads(
        (base / 'method_bundle.v1.schema.json').read_text())
    plan_schema = json.loads((base / 'method_plan.v1.schema.json').read_text())
    comparison_schema = json.loads(
        (base / 'method_plan_comparison.v1.schema.json').read_text())
    common = _common_contract()
    jsonschema.validate(common, common_schema)
    plans = []
    for method_id in pp.METHOD_IDS:
        bundle = _method_bundle(method_id)
        jsonschema.validate(bundle, bundle_schema)
        plan = pp.build_method_plan(method_id, common, bundle)
        jsonschema.validate(plan, plan_schema)
        plans.append(plan)
    comparison = pp.build_method_plan_comparison(plans)
    jsonschema.validate(comparison, comparison_schema)


def test_benchmark_main_method_plan_entry(tmp_path):
    main = (Path(__file__).parents[2] / '_meta' / 'docker' / 'features'
            / 'benchmark' / 'main')
    common = (Path(__file__).parents[2] / 'configs' / 'agentic_eval'
              / 'methods' / 'common_contract.v1.json')
    output = tmp_path / 'method_plan.json'
    completed = pp.subprocess.run(
        [str(main), 'plan', '--method', 'internnav',
         '--common', str(common), '--output', str(output)],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 2
    plan = json.loads(output.read_text())
    assert plan['schema'] == 'arena.agentic.method_plan.v1'
    assert plan['method_id'] == 'internnav'
    assert plan['plan_only'] is True
    assert plan['commands'] == []
    assert plan['executable'] is False
    assert ('METHOD_BUNDLE_CURRENT_PROFILE_DRIFT' in plan['blockers']
            or plan['common_case_blockers'])

    blocked = tmp_path / 'blocked.json'
    completed = pp.subprocess.run(
        [str(main), 'plan', '--method', 'qwen_agent_proxmpc',
         '--common', str(common), '--output', str(blocked)],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 2
    plan = json.loads(blocked.read_text())
    assert plan['status'] == 'BLOCKED'
    assert plan['commands'] == []


def test_benchmark_main_method_plan_unknown_and_bundle_override(tmp_path):
    main = (Path(__file__).parents[2] / '_meta' / 'docker' / 'features'
            / 'benchmark' / 'main')
    common = (Path(__file__).parents[2] / 'configs' / 'agentic_eval'
              / 'methods' / 'common_contract.v1.json')
    completed = pp.subprocess.run(
        [str(main), 'plan', '--method', 'not_a_method',
         '--common', str(common)],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 2
    result = json.loads(completed.stdout)
    assert 'METHOD_UNKNOWN:not_a_method' in result['blockers']
    assert result['commands'] == []
    assert result['future_launch_command'] == []

    fake_bundle = tmp_path / 'fake_bundle.json'
    fake_bundle.write_text(json.dumps(_method_bundle('internnav')))
    completed = pp.subprocess.run(
        [str(main), 'plan', '--method', 'internnav',
         '--common', str(common), '--bundle', str(fake_bundle)],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 2
    result = json.loads(completed.stdout)
    assert 'BUNDLE_OVERRIDE_NOT_ALLOWED' in result['blockers']
    assert result['commands'] == []


def test_method_bundle_forged_profile_and_container_rejected(tmp_path):
    common = _common_contract()
    bundle = _method_bundle('internnav')
    forged_profile = dict(bundle)
    forged_profile['profile'] = dict(bundle['profile'])
    forged_profile['profile']['sha256'] = 'f' * 64
    plan = pp.build_method_plan('internnav', common, forged_profile)
    assert plan['executable'] is False
    assert 'METHOD_BUNDLE_PROFILE_SHA_MISMATCH' in plan['blockers']
    assert plan['commands'] == []

    forged_container = dict(bundle)
    forged_container['model'] = dict(bundle['model'])
    forged_container['model']['container'] = 'forged-container'
    forged_container['model']['entrypoint'] = 'forged-entrypoint'
    override = tmp_path / 'forged_bundle.json'
    override.write_text(json.dumps(forged_container))
    rc = pp._cmd_method_plan(
        'internnav', str(Path(__file__).parents[2] / 'configs'
                         / 'agentic_eval' / 'methods'
                         / 'common_contract.v1.json'),
        str(override))
    assert rc == 2
    plan = pp.build_method_plan('internnav', common, forged_container)
    assert plan['bundle_source_digest'] != pp.build_method_plan(
        'internnav', common, bundle)['bundle_source_digest']


def test_method_plans_include_bundle_source_digest():
    common = _common_contract()
    plans = [pp.build_method_plan(method_id, common, _method_bundle(method_id))
             for method_id in pp.METHOD_IDS]
    digests = [plan['bundle_source_digest'] for plan in plans]
    assert all(item.startswith('sha256:') for item in digests)
    assert len(set(digests)) == len(plans)
    for plan in plans:
        assert plan['canonical_bundle_path'].endswith(
            f'{plan["method_id"]}.v1.json')
    comparison = pp.build_method_plan_comparison(plans)
    assert all('bundle_source_digest' in item
               for item in comparison['methods'].values())


def test_endpoint_resolution_and_missing_container():
    assert pp.resolve_container_endpoint(
        'qwen', inspect=lambda name: '172.17.0.2') == (
            'http://172.17.0.2:8000/v1/chat/completions')
    with pytest.raises(pp.PipelineRejected):
        pp.resolve_container_endpoint('qwen', inspect=lambda name: '')
