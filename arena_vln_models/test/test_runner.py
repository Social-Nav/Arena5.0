import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arena_vln_models.runner import Runner, main, owned_process, process_start


@pytest.fixture
def clean_ros_environment(monkeypatch):
    # These command tests inspect benchmark configuration. An enclosing RPP
    # run may deliberately use another domain; keep that environment intact
    # outside each test instead of forcing the entire suite onto domain 1.
    for name in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'FASTDDS_BUILTIN_TRANSPORTS',
                 'ROS_LOCALHOST_ONLY', 'ROS_AUTOMATIC_DISCOVERY_RANGE'):
        monkeypatch.delenv(name, raising=False)


def test_three_entry_points_plan_the_same_interface(capsys, clean_ros_environment):
    root = Path(__file__).resolve().parents[2]
    for model in ('internnav', 'lightnav', 'navila'):
        profile_name = 'internnav_follow_path_grscenes' if model == 'internnav' else f'{model}_grscenes'
        profile = root / f'arena_bringup/configs/benchmark/profiles/{profile_name}.yaml'
        assert main(['plan', 'run', '--profile', str(profile)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload['execution_mode'] == 'nav2_follow_path'
        assert payload['control_path'] == 'nav2_follow_path_rpp'
        assert payload['external_command'] == []
        assert payload['action'].endswith('/follow_path')


def test_serve_requires_model_repository_command(capsys, clean_ros_environment):
    assert main(['serve']) == 2
    assert 'external.command' in capsys.readouterr().err


def test_plan_reports_conflicting_ros_environment(capsys, monkeypatch, clean_ros_environment):
    monkeypatch.setenv('ROS_DOMAIN_ID', '97')
    assert main(['plan', 'run']) == 2
    message = capsys.readouterr().err
    assert 'ambient ROS environment conflicts' in message
    assert "ROS_DOMAIN_ID='97' (profile '1')" in message


def test_invalid_old_control_option_does_not_silently_run():
    with pytest.raises(SystemExit):
        main(['run', '--model-path', '/old/model'])


def test_evaluation_command_carries_overlay_provenance(
    tmp_path, monkeypatch, clean_ros_environment
):
    root = Path(__file__).resolve().parents[2]
    install = tmp_path / 'install'
    install.mkdir()
    (install / 'setup.bash').write_text('source x\n', encoding='utf-8')
    monkeypatch.setenv('ARENA_VLN_OVERLAY_INSTALL_ROOT', str(install))
    profile = root / 'arena_bringup/configs/benchmark/profiles/internnav_follow_path_grscenes.yaml'
    args = argparse.Namespace(
        model='internnav', profile=str(profile), episodes=None, robot=None, local=True,
        case=None, extra=[], world=None, scenario=None, human=None, tm_obstacles=None,
        timeout=None, expected_human_count=None, pedestrian_seed=None,
        pedestrian_pool_size=None, task_generator_parameter_file=None,
    )
    runner = Runner(args)
    command = runner.evaluation_command()
    payload = next(
        item for item in command if item.startswith('ARENA_EVAL_OVERLAY_PROVENANCE_JSON=')
    )
    record = json.loads(payload.split('=', 1)[1])
    assert record['source_worktree'] == str(root.resolve())
    assert record['source_commit']
    assert record['install_root'] == str(install.resolve())
    assert record['install_inventory_sha256']
    assert any(item.startswith('ARENA_EVAL_OVERLAY_SOURCE_COMMIT=') for item in command)


def test_process_identity_never_claims_unrelated_process():
    state = dict(pid=os.getpid(), start=process_start(os.getpid()), token='not-our-process')
    assert not owned_process(state)
    assert not owned_process({})


def test_owned_process_matches_token_and_start_time():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
        env={**os.environ, 'ARENA_VLN_RUN_TOKEN': 'integration-token'}, start_new_session=True)
    try:
        state = dict(pid=child.pid, start=process_start(child.pid), token='integration-token')
        assert owned_process(state)
        assert not owned_process({**state, 'start': 'wrong'})
        assert not owned_process({**state, 'token': 'wrong'})
    finally:
        child.terminate()
        child.wait(timeout=3)


def test_model_params_json_is_forwarded_to_the_evaluator(tmp_path, monkeypatch, clean_ros_environment):
    root = Path(__file__).resolve().parents[2]
    install = tmp_path / 'install'
    install.mkdir()
    (install / 'setup.bash').write_text('source x\n', encoding='utf-8')
    monkeypatch.setenv('ARENA_VLN_OVERLAY_INSTALL_ROOT', str(install))
    profile = root / 'arena_bringup/configs/benchmark/profiles/internnav_follow_path_grscenes.yaml'
    params = '{"host": "arena-navila-fixed-1", "port": 54321}'
    args = argparse.Namespace(
        model='internnav', profile=str(profile), episodes=None, robot=None, local=True,
        case=None, extra=[], world=None, scenario=None, human=None, tm_obstacles=None,
        timeout=None, expected_human_count=None, pedestrian_seed=None,
        pedestrian_pool_size=None, task_generator_parameter_file=None,
        model_params_json=params,
    )
    runner = Runner(args)
    command = runner.evaluation_command()
    index = command.index('--model-params-json')
    assert command[index + 1] == params
