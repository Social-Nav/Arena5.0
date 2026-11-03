"""Cross-boundary structural and behavior preflight for the B eval chain.

No episode is started: the behavior test uses internnav_eval's
--print-effective-config, which resolves and exits before launch.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from arena_bringup import paper_pipeline as pp


ROOT = Path(__file__).parents[2]
ARENA_LAUNCH = ROOT / 'arena_bringup' / 'launch' / 'arena.launch.py'
TASK_LAUNCH = ROOT / 'task_generator' / 'launch' / 'task_generator.launch.py'
ROBOT_MANAGER = ROOT / 'task_generator' / 'task_generator' / 'manager' / 'robot_manager' / 'robot_manager.py'
INTERNNAV_EVAL = ROOT / 'arena_bringup' / 'arena_bringup' / 'internnav_eval.py'
RUNNER = ROOT / 'arena_vln_models' / 'arena_vln_models' / 'runner.py'
FOLLOW_PATH = ROOT / 'arena_vln_models' / 'launch' / 'follow_path.launch.py'
BENCHMARK_PROFILE = ROOT / 'arena_bringup' / 'configs' / 'benchmark' / 'profiles' / 'base_follow_path_grscenes_task34.yaml'
ON_PROFILE = ROOT / 'arena_bringup' / 'configs' / 'lisn' / 'bfast_on_v1.json'
OFF_PROFILE = ROOT / 'arena_bringup' / 'configs' / 'lisn' / 'bfast_off_v1.json'
STATIC_SMOKE = ROOT / 'arena_bringup' / 'configs' / 'lisn' / 'scenario_static_smoke_v1.json'
CLOSED_SMOKE = ROOT / 'arena_bringup' / 'configs' / 'lisn' / 'scenario_closedloop_smoke_v1.json'


def test_b_eval_parameter_chain_is_structurally_wired():
    arena = ARENA_LAUNCH.read_text()
    task = TASK_LAUNCH.read_text()
    manager = ROBOT_MANAGER.read_text()
    internnav = INTERNNAV_EVAL.read_text()
    runner = RUNNER.read_text()
    follow = FOLLOW_PATH.read_text()
    for token in ('fast_guard_enabled', 'fast_guard_veto_topic', 'fast_guard_required',
                  'fast_guard_veto_timeout_sec', 'bfast_risk_threshold',
                  'bfast_max_frame_gap_sec'):
        assert token in arena, token
        assert token in task, token
        assert token in manager, token
    assert '**fast_guard_enabled.dict' in arena
    assert '**fast_guard_enabled.str_param' in task
    assert "'fast_guard_enabled':" in manager
    assert '--bfast-profile' in internnav
    assert 'fast_guard_enabled:=' in internnav
    assert 'build_bfast_plan' in internnav
    assert '--bfast-profile' in runner
    assert '--bfast-guard-enabled' in runner
    assert 'bfast_profile' in runner
    assert 'fast_guard_enabled' in follow
    assert 'bfast_guard' in follow


def _effective(profile: Path) -> dict:
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f'{ROOT / "arena_vln_models"}:{ROOT / "arena_bringup"}:{ROOT}:'
        + env.get('PYTHONPATH', '')
    )
    command = [
        sys.executable, '-m', 'arena_bringup.internnav_eval',
        '--print-effective-config',
        '--benchmark-profile', str(BENCHMARK_PROFILE),
        '--bfast-profile', str(profile),
        '--world', 'grscenes_20_v1',
        '--scenario-file', 'default_2',
        '--robot', 'Ai2_Bot2',
        '--execution-mode', 'nav2_follow_path',
        '--local-planner', 'regulated_pure_pursuit',
    ]
    if profile == ON_PROFILE:
        command += ['--bfast-risk-threshold', '0.5',
                    '--bfast-max-frame-gap-sec', '1.0']
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_b_on_off_effective_config_differs_only_in_guard_fields():
    on = _effective(ON_PROFILE)
    off = _effective(OFF_PROFILE)
    on_params = on['resolved_parameters']
    off_params = off['resolved_parameters']
    on_bfast = on_params.pop('bfast_profile')
    off_bfast = off_params.pop('bfast_profile')
    assert on_bfast['arm'] == 'on'
    assert off_bfast['arm'] == 'off'
    assert on_bfast['model'] == off_bfast['model']
    assert on_bfast['model_revision'] == off_bfast['model_revision']
    assert on_bfast['prompt_sha256'] == off_bfast['prompt_sha256']
    assert on_bfast['action_mapping'] == off_bfast['action_mapping']
    assert on_bfast['guard_enabled'] is True
    assert off_bfast['guard_enabled'] is False
    assert on_bfast['veto_topic']
    assert off_bfast['veto_topic'] == ''
    assert on_bfast['risk_threshold'] == 0.5
    assert on_bfast['max_frame_gap_sec'] == 1.0
    assert on_bfast['setting_status'] == 'SMOKE_ONLY_NOT_PAPER_SETTING'
    assert off_bfast['risk_threshold'] is None
    assert off_bfast['max_frame_gap_sec'] is None
    assert off_bfast['setting_status'] == 'NOT_APPLICABLE'
    assert on_params == off_params, 'on/off differ outside the guard fields'


def _runner_plan(profile: Path, extra_args: list[str]) -> list:
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f'{ROOT / "arena_vln_models"}:{ROOT / "arena_bringup"}:{ROOT}:'
        + env.get('PYTHONPATH', '')
    )
    result = subprocess.run([
        sys.executable, '-m', 'arena_vln_models.runner', 'plan', '--local',
        '--base-profile', 'base_follow_path_grscenes_task34.yaml',
        '--model', 'internnav', '--robot', 'Ai2_Bot2',
        '--case', 'grscenes_20_v1/default_2',
        '--lisn-slow-loop',
        '--lisn-producer-endpoint', 'http://127.0.0.1:8000/v1/chat/completions',
        '--bfast-profile', str(profile),
        *extra_args,
    ], capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_runner_b_plan_is_blocked_with_no_commands():
    for profile, extra in (
        (ON_PROFILE, ['--bfast-risk-threshold', '0.5',
                      '--bfast-max-frame-gap-sec', '1.0']),
        (OFF_PROFILE, []),
    ):
        plan = _runner_plan(profile, extra)
        assert plan['executable'] is False
        assert plan['commands'] == []
        assert plan['blocked_reason'] == 'qwen_lisn_common_metric_action_bridge_missing'
        assert plan['model'] == 'Qwen/Qwen3-VL-8B-Instruct-FP8'
        assert plan['model_backend'] == 'lisn_qwen3vl_fp8'
        assert plan['runner_entry'] == 'internnav'
        assert plan['model_adapter'] == 'lisn_qwen3vl_fp8'
        assert plan['checkpoint'] == ''


def test_static_closedloop_smoke_profiles_generated_commands_parse():
    for profile_path in (STATIC_SMOKE, CLOSED_SMOKE):
        payload = json.loads(profile_path.read_text())
        assert payload.get('setting_status') == 'SMOKE_ONLY_NOT_PAPER_SETTING'
        validation = pp.validate_scenario_profile(payload)
        assert validation['valid'] is True
        assert validation['runtime_status'] == 'SUPPORTED'
        command = pp.scenario_command(payload)
        env = os.environ.copy()
        env.update(command['env'])
        env['PYTHONPATH'] = (
            f'{ROOT / "arena_vln_models"}:{ROOT / "arena_bringup"}:{ROOT}:'
            + env.get('PYTHONPATH', '')
        )
        result = subprocess.run(
            [*command['argv'], '--print-effective-config'],
            capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['resolved_parameters']['human'] in ('dummy', 'hunav')


def test_b_profile_on_missing_thresholds_fails_closed(tmp_path):
    payload = json.loads(ON_PROFILE.read_text())
    payload.pop('risk_threshold', None)
    payload.pop('max_frame_gap_sec', None)
    path = tmp_path / 'bfast_on_missing.json'
    path.write_text(json.dumps(payload))
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f'{ROOT / "arena_vln_models"}:{ROOT / "arena_bringup"}:{ROOT}:'
        + env.get('PYTHONPATH', '')
    )
    result = subprocess.run([
        sys.executable, '-m', 'arena_bringup.internnav_eval',
        '--print-effective-config',
        '--benchmark-profile', str(BENCHMARK_PROFILE),
        '--bfast-profile', str(path),
        '--world', 'grscenes_20_v1',
        '--scenario-file', 'default_2',
        '--robot', 'Ai2_Bot2',
        '--execution-mode', 'nav2_follow_path',
        '--local-planner', 'regulated_pure_pursuit',
    ], capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert 'risk_threshold_required' in result.stderr
