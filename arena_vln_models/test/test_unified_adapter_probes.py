"""Behavior probes for the unified model adapter contract.

These are deliberately behavior-level tests (not just imports): they exercise
the registry bootstrap, pixel-only fail-close, observation freshness, LightNav
transport persistence, origin-only path rejection, and NaVILA primitive/black
frame history.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from arena_vln_models.adapters.internnav import InternNavAdapter, INTERNNAV_CAPABILITY
from arena_vln_models.adapters.lightnav import LightNavAdapter
from arena_vln_models.adapters.navila import NaVILAAdapter
from arena_vln_models.core import ModelObservation, Pose2D
from arena_vln_models.execution import ExecutionConfig, ExecutionCore
from arena_vln_models.model_adapter import (
    CapabilityMismatchError,
    EmptyModelOutputError,
    ModelRawOutput,
    ModelTimeoutError,
    UnknownModelAdapterError,
    create_adapter,
    prediction_from_raw,
)


def _observation(*, stamp_ns: int = 1, received_at: float = 1.0, rgb=None) -> ModelObservation:
    return ModelObservation(
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        instruction='go to the kitchen',
        rgb_image=np.zeros((48, 64, 3), dtype=np.uint8) if rgb is None else rgb,
        frame_id='odom',
        stamp_ns=stamp_ns,
        received_at=received_at,
    )


def test_create_adapter_without_preimporting_adapters_package():
    from arena_vln_models.model_adapter import create_adapter as direct_create_adapter

    for name in ('internnav', 'lightnav', 'navila'):
        adapter = direct_create_adapter(name)
        assert adapter.name == name
        assert adapter.capability.name == name
    with pytest.raises(UnknownModelAdapterError):
        direct_create_adapter('unknown-model')


def test_internnav_pixel_only_is_explicitly_unsupported():
    assert 'pixel_diagnostic' not in INTERNNAV_CAPABILITY.output_modes
    adapter = InternNavAdapter(transport=lambda *args, **kwargs: {})
    with pytest.raises(CapabilityMismatchError):
        adapter._decode(
            {'output_pixel': [100, 100]},
            elapsed=0.1,
            depth=np.zeros((48, 64), dtype=np.float32),
            reset=False,
            observation=_observation(),
        )


def test_execution_core_fresh_and_stale_observation():
    class _Adapter:
        name = 'fixture'
        capability = INTERNNAV_CAPABILITY

        def reset(self, **kwargs):
            return None

        def infer(self, observation):
            raise AssertionError('stale observation must not reach the model')

        def to_prediction(self, raw, observation):
            raise AssertionError('stale observation must not reach the model')

    core = ExecutionCore(_Adapter(), ExecutionConfig(model='fixture', observation_max_age_sec=2.0))
    core.reset('session', 0)
    core.release()
    core.set_instruction('go')

    fresh = _observation(stamp_ns=10, received_at=100.0)
    assert core.eligible(fresh, now=101.0) == (True, '')

    stale = _observation(stamp_ns=11, received_at=50.0)
    assert core.eligible(stale, now=101.0) == (False, 'stale_observation')

    missing = _observation(stamp_ns=12, received_at=0.0)
    assert core.eligible(missing, now=101.0) == (False, 'stale_observation')


def test_lightnav_transport_is_persisted_and_reset_reaches_it(monkeypatch):
    import arena_vln_models.adapters.lightnav as lightnav

    instances = []

    class _FakeTransport:
        def __init__(self, url, timeout):
            self.url = url
            self.timeout = timeout
            self.connects = 0
            self.resets = 0
            instances.append(self)

        def reset(self):
            self.resets += 1

        def generate(self, *, seq, image, instruction):
            self.connects += 1
            return {
                'action': 'next',
                'data': {
                    'rc': 0,
                    'seq': seq,
                    'stop': False,
                    'actions': {'actions': [[0.5, 0.0, 0.0]]},
                },
            }

    monkeypatch.setattr(lightnav, '_WebSocketTransport', _FakeTransport)
    adapter = LightNavAdapter(params={'server_url': 'ws://fixture'})
    adapter.infer(_observation())
    adapter.infer(_observation(stamp_ns=2))
    assert len(instances) == 1
    assert adapter._transport is instances[0]
    assert instances[0].connects == 2
    adapter.reset(session_id='session', episode=1)
    assert instances[0].resets == 1


def test_origin_only_non_stop_is_empty_but_pure_rotation_is_kept():
    observation = _observation()
    with pytest.raises(EmptyModelOutputError):
        prediction_from_raw(
            ModelRawOutput(adapter='fixture', local_waypoints=[(0.0, 0.0, 0.0)], stop=False),
            observation,
        )
    rotation = prediction_from_raw(
        ModelRawOutput(adapter='fixture', local_waypoints=[(0.0, 0.0, 0.3)], stop=False),
        observation,
    )
    assert rotation.stop is False
    assert len(rotation.poses) >= 1
    assert abs(rotation.poses[-1][2] - 0.3) < 1e-6


def test_timeout_error_type_is_preserved():
    with pytest.raises(ModelTimeoutError):
        prediction_from_raw(
            ModelRawOutput(adapter='fixture', status='timeout', stop=False, debug={'reason': 'timeout'}),
            _observation(),
        )


def test_navila_source_is_primitive_and_short_history_uses_black_frames():
    adapter = NaVILAAdapter(transport=lambda *args, **kwargs: 'move forward 75 cm')
    white = np.full((48, 64, 3), 255, dtype=np.uint8)
    raw = adapter.infer(_observation(rgb=white))
    assert raw.source == 'primitive'
    assert raw.modality == 'primitive'
    assert raw.local_waypoints == [(0.75, 0.0, 0.0)]

    images, stamps = adapter._history.sample()
    assert len(images) == 8
    assert len(stamps) == 8
    first = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[0]))).convert('RGB'))
    last = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[-1]))).convert('RGB'))
    assert first.max() == 0, 'missing history must be black frames'
    assert last.max() > 0, 'current frame must be the real image'
