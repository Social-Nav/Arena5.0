"""Fixture-driven positive and negative contracts for the three adapters."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from arena_vln_models.adapters.internnav import InternNavAdapter
from arena_vln_models.adapters.lightnav import LightNavAdapter
from arena_vln_models.adapters.navila import NaVILAAdapter
from arena_vln_models.core import ModelObservation, Pose2D
from arena_vln_models.model_adapter import (
    CapabilityMismatchError,
    ModelProtocolError,
    ModelTimeoutError,
    coerce_local_waypoints,
)

FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def _observation(*, stamp_ns: int = 1, frame_id: str = 'odom', received_at: float = 1.0,
                 instruction: str = 'go to the kitchen', rgb=None, depth=None,
                 intrinsics=(500.0, 0.0, 32.0, 0.0, 500.0, 24.0, 0.0, 0.0, 1.0)):
    return ModelObservation(
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        instruction=instruction,
        rgb_image=np.zeros((48, 64, 3), dtype=np.uint8) if rgb is None else rgb,
        depth_image=np.full((48, 64), 1.0, dtype=np.float32) if depth is None else depth,
        camera_intrinsics=intrinsics,
        frame_id=frame_id,
        stamp_ns=stamp_ns,
        received_at=received_at,
        session_id='session',
        episode=0,
    )


def _internnav(raw):
    return InternNavAdapter(transport=lambda *, payload, files: raw)


def _navila(text):
    return NaVILAAdapter(transport=lambda *, images, query: text)


def _lightnav(raw):
    class _Transport:
        def reset(self):
            return None

        def generate(self, *, seq, image, instruction):
            return raw

    return LightNavAdapter(transport=_Transport())


@pytest.mark.parametrize(
    'fixture,source,modality,waypoint_count',
    [
        ('internnav_native.json', 'native', 'trajectory', 3),
        ('internnav_synthetic.json', 'synthetic', 'trajectory', 1),
        ('internnav_primitive.json', 'primitive', 'primitive', 3),
        ('internnav_duplicate_origin.json', 'native', 'trajectory', 3),
    ],
)
def test_internnav_positive_provenance_and_waypoints(fixture, source, modality, waypoint_count):
    raw = _internnav(_fixture(fixture)).infer(_observation())
    assert raw.source == source
    assert raw.modality == modality
    assert raw.stop is False
    assert len(raw.local_waypoints) == waypoint_count
    assert raw.raw_hash


def test_internnav_native_coordinate_unit_and_yaw():
    adapter = _internnav(_fixture('internnav_native.json'))
    prediction = adapter.to_prediction(adapter.infer(_observation()), _observation())
    assert prediction.stop is False
    assert prediction.source == 'native'
    assert prediction.poses[0] == pytest.approx((0.0, 0.0, 0.0))
    assert prediction.poses[-1][0] == pytest.approx(0.5, abs=1e-6)
    assert prediction.poses[-1][1] == pytest.approx(0.1, abs=1e-6)
    # Path is sampled at <= 5 cm and remains continuous.
    for start, end in zip(prediction.poses, prediction.poses[1:]):
        assert math.hypot(end[0] - start[0], end[1] - start[1]) <= 0.0500001


def test_internnav_duplicate_origin_is_deduped_and_pure_rotation_kept():
    local = coerce_local_waypoints(_fixture('internnav_duplicate_origin.json')['output_trajectory'])
    assert len(local) == 3
    assert local[0] == (0.0, 0.0, 0.0)
    assert local[-1][2] == pytest.approx(0.3)


def test_internnav_primitive_expands_forward_and_turn():
    raw = _internnav(_fixture('internnav_primitive.json')).infer(_observation())
    assert raw.source == 'primitive'
    # forward 25 cm, left 15 deg, forward 25 cm
    assert raw.local_waypoints[0] == pytest.approx((0.25, 0.0, 0.0))
    assert raw.local_waypoints[1][2] == pytest.approx(math.radians(15.0))
    assert raw.local_waypoints[2][2] == pytest.approx(math.radians(15.0))


def test_internnav_stop_is_terminal_with_no_path():
    adapter = _internnav(_fixture('internnav_stop.json'))
    prediction = adapter.to_prediction(adapter.infer(_observation()), _observation())
    assert prediction.stop is True
    assert prediction.poses == []


@pytest.mark.parametrize(
    'fixture',
    ['internnav_negative_nan.json', 'internnav_negative_empty.json',
     'internnav_negative_discontinuity.json'],
)
def test_internnav_negative_outputs_fail_closed(fixture):
    with pytest.raises(ModelProtocolError):
        _internnav(_fixture(fixture)).infer(_observation())


def test_internnav_pixel_only_is_unsupported_capability():
    with pytest.raises(CapabilityMismatchError):
        _internnav(_fixture('internnav_pixel.json')).infer(_observation())


def test_internnav_missing_frame_and_stamp_fail_before_transport():
    called = {'value': False}

    def _transport(**kwargs):
        called['value'] = True
        return {}

    adapter = InternNavAdapter(transport=_transport)
    with pytest.raises(CapabilityMismatchError):
        adapter.infer(_observation(frame_id=''))
    with pytest.raises(CapabilityMismatchError):
        adapter.infer(_observation(stamp_ns=0))
    assert called['value'] is False


def test_internnav_timeout_and_transport_error_types():
    def _timeout(**kwargs):
        raise TimeoutError('slow model')

    with pytest.raises(ModelTimeoutError):
        InternNavAdapter(transport=_timeout).infer(_observation())

    def _error(**kwargs):
        raise RuntimeError('transport exploded')

    with pytest.raises(ModelProtocolError):
        InternNavAdapter(transport=_error).infer(_observation())


def test_navila_forward_unit_and_yaw():
    adapter = _navila(_fixture('navila_forward.json'))
    prediction = adapter.to_prediction(adapter.infer(_observation()), _observation())
    assert prediction.source == 'primitive'
    assert prediction.poses[-1][0] == pytest.approx(0.75, abs=1e-6)
    assert prediction.poses[-1][1] == pytest.approx(0.0, abs=1e-6)


def test_navila_right_turn_is_negative_yaw():
    adapter = _navila(_fixture('navila_turn.json'))
    prediction = adapter.to_prediction(adapter.infer(_observation()), _observation())
    assert prediction.poses[-1][2] == pytest.approx(-math.radians(45.0))


def test_navila_stop_modality_and_terminal():
    adapter = _navila(_fixture('navila_stop.json'))
    raw = adapter.infer(_observation())
    assert raw.stop is True
    assert raw.modality == 'text_stop'
    assert raw.source == 'primitive'
    prediction = adapter.to_prediction(raw, _observation())
    assert prediction.stop is True
    assert prediction.poses == []


@pytest.mark.parametrize('fixture', ['navila_negative_unrecognized.json', 'navila_negative_server_error.json'])
def test_navila_negative_text_fails_closed(fixture):
    with pytest.raises(ModelProtocolError):
        _navila(_fixture(fixture)).infer(_observation())


def test_lightnav_full_waypoint_path_and_lateral_left():
    adapter = _lightnav(_fixture('lightnav_waypoints.json'))
    raw = adapter.infer(_observation())
    assert raw.source == 'native'
    assert raw.modality == 'waypoints'
    assert len(raw.local_waypoints) == 2
    prediction = adapter.to_prediction(raw, _observation())
    assert prediction.poses[-1][0] == pytest.approx(0.8, abs=1e-6)
    assert prediction.poses[-1][1] == pytest.approx(0.2, abs=1e-6)


def test_lightnav_stop_is_terminal():
    adapter = _lightnav(_fixture('lightnav_stop.json'))
    raw = adapter.infer(_observation())
    assert raw.stop is True
    assert raw.modality == 'stop'
    prediction = adapter.to_prediction(raw, _observation())
    assert prediction.stop is True
    assert prediction.poses == []


@pytest.mark.parametrize(
    'fixture',
    ['lightnav_negative_rc.json', 'lightnav_negative_seq.json',
     'lightnav_negative_empty.json', 'lightnav_negative_nan.json'],
)
def test_lightnav_negative_outputs_fail_closed(fixture):
    with pytest.raises(ModelProtocolError):
        _lightnav(_fixture(fixture)).infer(_observation())


def test_lightnav_timeout_and_transport_error_types():
    class _Timeout:
        def generate(self, **kwargs):
            raise TimeoutError('slow lightnav')

    with pytest.raises(ModelTimeoutError):
        LightNavAdapter(transport=_Timeout()).infer(_observation())

    class _Error:
        def generate(self, **kwargs):
            raise RuntimeError('lightnav exploded')

    with pytest.raises(ModelProtocolError):
        LightNavAdapter(transport=_Error()).infer(_observation())


def test_non_stop_origin_only_is_empty_but_rotation_is_valid():
    from arena_vln_models.model_adapter import EmptyModelOutputError, prediction_from_raw, ModelRawOutput

    with pytest.raises(EmptyModelOutputError):
        prediction_from_raw(
            ModelRawOutput(adapter='fixture', local_waypoints=[(0.0, 0.0, 0.0)], stop=False),
            _observation(),
        )
    rotation = prediction_from_raw(
        ModelRawOutput(adapter='fixture', local_waypoints=[(0.0, 0.0, 0.3)], stop=False),
        _observation(),
    )
    assert rotation.stop is False
    assert rotation.poses[-1][2] == pytest.approx(0.3)
