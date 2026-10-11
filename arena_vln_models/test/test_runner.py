import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arena_vln_models.runner import main, owned_process, process_start


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
        profile = root / f'arena_bringup/configs/benchmark/profiles/{model}_grscenes.yaml'
        assert main(['plan', 'run', '--profile', str(profile)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload['execution_mode'] == 'nav2_follow_path'
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
