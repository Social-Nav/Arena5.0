"""Frozen-parameter observation contract for the LISN engineering adapter."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena_bringup.lisn_slow_loop_ros import (
    ENGINEERING_APPLIED_PARAMETERS,
    ENGINEERING_FROZEN_PARAMETERS,
    ENGINEERING_LOGICAL_PARAMETERS,
    ApplyResult,
    EngineeringParameterAdapter,
    RosParameterAdapter,
    load_grounding_live_policy,
)

PROFILE = Path(__file__).resolve().parents[1] / 'configs/lisn/slow_loop_grounding_live_v1.json'


def _policy():
    return load_grounding_live_policy(PROFILE)


def test_read_binding_uses_exact_candidate_first():
    adapter = RosParameterAdapter.__new__(RosParameterAdapter)
    adapter._policy = _policy()
    assert adapter._read_name('max_linear_velocity_mps') == (
        'FollowPath.optimizer.max_linear_velocity'
    )
    assert adapter._read_name('max_angular_velocity_rps') == (
        'FollowPath.optimizer.max_angular_velocity'
    )
    assert adapter._read_name('obstacle_weight') == (
        'FollowPath.optimizer.weights.obstacle_weight'
    )


def test_read_binding_falls_back_to_first_candidate_for_frozen_ambiguous():
    adapter = RosParameterAdapter.__new__(RosParameterAdapter)
    adapter._policy = _policy()
    assert adapter._read_name('goal_weight') == (
        'FollowPath.optimizer.weights.distance_weight'
    )
    assert adapter._read_name('people_weight') == (
        'FollowPath.optimizer.weights.proxemics_d0'
    )
    binding = adapter._policy.parameter_bindings['people_weight']
    assert binding['observation_candidate'] == (
        'FollowPath.optimizer.weights.proxemics_d0'
    )


def test_write_binding_still_rejects_ambiguous_parameters():
    adapter = RosParameterAdapter.__new__(RosParameterAdapter)
    adapter._policy = _policy()
    with pytest.raises(RuntimeError, match='no exact runtime binding'):
        adapter._actual_name('goal_weight')
    with pytest.raises(RuntimeError, match='no exact runtime binding'):
        adapter._actual_name('people_weight')


class _Base:
    def __init__(self, values):
        self.values = dict(values)
        self.applied = {}
        self.last_read_bindings = {name: name for name in self.values}

    def read(self, names):
        return {name: float(self.values[name]) for name in names}

    def set_atomically(self, values):
        self.applied = dict(values)
        return ApplyResult(True, '')


def test_engineering_adapter_observes_frozen_and_applies_only_qualified():
    base = _Base({name: 1.0 for name in ENGINEERING_LOGICAL_PARAMETERS})
    adapter = EngineeringParameterAdapter(base)
    observed = adapter.read(ENGINEERING_LOGICAL_PARAMETERS)
    assert set(observed) == set(ENGINEERING_LOGICAL_PARAMETERS)
    assert adapter.last_read_bindings == base.last_read_bindings
    result = adapter.set_atomically({**observed, 'max_linear_velocity_mps': 0.8})
    assert result.successful is True
    assert set(base.applied) == set(ENGINEERING_APPLIED_PARAMETERS)


def _verify(plugin, plugins):
    adapter = RosParameterAdapter.__new__(RosParameterAdapter)
    adapter._policy = _policy()
    adapter._timeout = 1.0
    adapter._ParameterType = SimpleNamespace(
        PARAMETER_STRING_ARRAY=4, PARAMETER_STRING=3,
    )

    class _Future:
        def done(self):
            return True

        def result(self):
            return SimpleNamespace(values=[
                SimpleNamespace(type=4, string_array_value=list(plugins), string_value=''),
                SimpleNamespace(type=3, string_array_value=[], string_value=plugin),
            ])

    class _Client:
        srv_name = 'get_parameters'

        def wait_for_service(self, timeout_sec):
            return True

        def call_async(self, request):
            return _Future()

    adapter._get = _Client()
    adapter._node = None
    return adapter.verify_social_mpc()


def test_verify_accepts_direct_social_mpc_plugin():
    assert _verify('nav2_social_mpc_controller::SocialMPCController',
                   ['FollowPath']) is True


def test_verify_rejects_rotation_shim_wrapper():
    with pytest.raises(RuntimeError, match='not social_mpc'):
        _verify('nav2_rotation_shim_controller::RotationShimController', ['FollowPath'])


def test_verify_rejects_missing_follow_path_plugin():
    with pytest.raises(RuntimeError, match='not social_mpc'):
        _verify('nav2_social_mpc_controller::SocialMPCController', ['DWB'])


def test_engineering_adapter_rejects_any_frozen_change():
    base = _Base({name: 1.0 for name in ENGINEERING_LOGICAL_PARAMETERS})
    adapter = EngineeringParameterAdapter(base)
    observed = adapter.read(ENGINEERING_LOGICAL_PARAMETERS)
    for name in ENGINEERING_FROZEN_PARAMETERS:
        changed = dict(observed)
        changed[name] = 2.0
        result = adapter.set_atomically(changed)
        assert result.successful is False
        assert 'UNQUALIFIED_PARAMETER_CHANGE' in result.reason
        assert base.applied == {}
