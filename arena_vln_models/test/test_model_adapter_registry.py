"""Registry, source-invariant, OpenSpec-structure and plan-equivalence tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena_vln_models.core import ModelObservation, Pose2D
from arena_vln_models.model_adapter import (
    CapabilityMismatchError,
    UnknownModelAdapterError,
    available_adapters,
    create_adapter,
)
from arena_vln_models.runner import main

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'arena_vln_models' / 'arena_vln_models'
ADAPTERS = PACKAGE / 'adapters'
CHANGE = ROOT / 'openspec' / 'changes' / 'unify-vln-model-adapters'

PLAN_WHITELIST = {
    'model', 'model_adapter', 'id', 'profile_id', 'path', 'profile_path',
    'sha256', 'profile_sha256', 'base_profile_path', 'base_profile_sha256',
    'overlay_profile_path', 'overlay_profile_sha256', 'resolved_profile_sha256',
    'resolved_profile_snapshot', 'output_prefix', 'external_command',
    'model_server', 'checkpoint', 'commands', 'adapter',
    'model_backend', 'runner_entry', 'actual_model_backend',
}


def _observation(*, depth=True):
    return ModelObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        instruction='go',
        rgb_image=None,
        depth_image=None if not depth else object(),
        camera_intrinsics=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        frame_id='odom',
        stamp_ns=1,
        received_at=1.0,
    )


@pytest.fixture
def clean_ros_environment(monkeypatch):
    for name in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'FASTDDS_BUILTIN_TRANSPORTS',
                 'ROS_LOCALHOST_ONLY', 'ROS_AUTOMATIC_DISCOVERY_RANGE'):
        monkeypatch.delenv(name, raising=False)


def test_registry_resolves_three_models_without_preimport():
    assert set(available_adapters()) >= {'internnav', 'navila', 'lightnav'}
    for name in ('internnav', 'navila', 'lightnav'):
        adapter = create_adapter(name)
        assert adapter.name == name
        assert adapter.capability.name == name
    with pytest.raises(UnknownModelAdapterError):
        create_adapter('not-a-model')


def test_internnav_capability_mismatch_fails_before_transport():
    called = {'value': False}

    def _transport(**kwargs):
        called['value'] = True
        return {}

    adapter = create_adapter('internnav', transport=_transport)
    with pytest.raises(CapabilityMismatchError):
        adapter.infer(_observation(depth=False))
    assert called['value'] is False


def test_adapters_never_publish_velocity_or_use_ros():
    forbidden = ('rclpy', 'create_publisher', 'Twist', 'cmd_vel', 'send_goal_async')
    for path in sorted(ADAPTERS.glob('*.py')):
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f'{path.name} contains forbidden token {token!r}'


def test_only_executor_sends_follow_path():
    senders = []
    for path in sorted(PACKAGE.glob('*.py')):
        text = path.read_text()
        if 'ActionClient(' in text or 'send_goal_async(' in text:
            senders.append(path.name)
    assert senders == ['executor.py']
    executor = (PACKAGE / 'executor.py').read_text()
    assert 'Twist' not in executor
    assert 'cmd_vel' not in executor.replace('cmd_vel publisher', '').replace('cmd_vel;', '')


def test_execution_config_rejects_native_mode():
    from arena_vln_models.execution import ExecutionConfig

    with pytest.raises(ValueError, match='(?i)native velocity'):
        ExecutionConfig(model='internnav', control_mode='native')


def test_three_model_plan_diff_only_model_whitelist(tmp_path, monkeypatch, capsys, clean_ros_environment):
    monkeypatch.setenv('ARENA_WS_DIR', str(tmp_path))
    plans = {}
    for model in ('internnav', 'navila', 'lightnav'):
        assert main(['plan', 'run', '--model', model]) == 0
        plans[model] = json.loads(capsys.readouterr().out)
    base = {key: value for key, value in plans['internnav'].items() if key not in PLAN_WHITELIST}
    for model in ('navila', 'lightnav'):
        other = {key: value for key, value in plans[model].items() if key not in PLAN_WHITELIST}
        assert other == base
        assert plans[model]['control_path'] == 'nav2_follow_path_rpp'
        assert plans[model]['controller'] == 'regulated_pure_pursuit'
        assert plans[model]['model_adapter'] == model
        assert plans[model]['resolved_profile_sha256']
        assert plans[model]['base_profile_sha256']
        assert plans[model]['overlay_profile_sha256']


def test_profile_and_model_conflict_fails_before_plan(tmp_path, monkeypatch, capsys, clean_ros_environment):
    monkeypatch.setenv('ARENA_WS_DIR', str(tmp_path))
    profile = ROOT / 'arena_bringup/configs/benchmark/profiles/internnav_follow_path_grscenes.yaml'
    assert main(['plan', 'run', '--model', 'navila', '--profile', str(profile)]) == 2
    assert 'conflicts with --model' in capsys.readouterr().err


def test_native_control_mode_is_rejected_by_unified_runner(tmp_path, monkeypatch, capsys, clean_ros_environment):
    monkeypatch.setenv('ARENA_WS_DIR', str(tmp_path))
    assert main(['plan', 'run', '--model', 'lightnav', '--control-mode', 'native']) == 2
    assert 'legacy direct entry' in capsys.readouterr().err


def test_openspec_change_structure():
    required = [
        '.openspec.yaml', 'proposal.md', 'design.md', 'tasks.md',
        'specs/unified-vln-model-adapter/spec.md',
        'specs/unified-vln-eval-runner/spec.md',
    ]
    for relative in required:
        assert (CHANGE / relative).is_file(), f'missing OpenSpec file {relative}'
    proposal = (CHANGE / 'proposal.md').read_text()
    for heading in ('## Why', '## What Changes', '## Impact'):
        assert heading in proposal
    design = (CHANGE / 'design.md').read_text()
    for heading in ('## Context', '## Goals / Non-Goals', '## Decisions', '## Risks / Trade-offs', '## Migration Plan'):
        assert heading in design
    tasks = (CHANGE / 'tasks.md').read_text()
    assert '## 1.' in tasks and '## 8.' in tasks
    for spec_name in ('unified-vln-model-adapter', 'unified-vln-eval-runner'):
        spec = (CHANGE / 'specs' / spec_name / 'spec.md').read_text()
        assert spec.count('### Requirement:') >= 1
        assert spec.count('#### Scenario:') >= 1
