"""Exact-stamp RGB/depth pairing for the unified executor observation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arena_vln_models.core import Pose2D
from arena_vln_models.executor import ArenaModelExecutor
from arena_vln_models.model_adapter import ModelCapability


class _Fake:
    def __init__(self, required=('rgb', 'depth')):
        self.adapter = SimpleNamespace(
            capability=ModelCapability(name='fixture', raw_output_kind='fixture', required_inputs=required)
        )
        self.core = SimpleNamespace(instruction='go', session_id='session', episode=0)
        self._odom_frame = 'odom'
        self._odom_frame_id = 'odom'
        self._odom_at = 1.0
        self._camera_info = None
        self._camera_info_stamp_ns = 0
        self._rgb_buffer = {}
        self._depth_buffer = {}
        self._sync_hold_reason = ''
        self._image = None
        self._depth = None
        self._image_stamp_ns = 0
        self._depth_stamp_ns = 0
        self._image_frame_id = ''

    def _pose_at_stamp(self, stamp):
        return Pose2D(0.0, 0.0, 0.0)


def _rgb():
    return np.zeros((2, 2, 3), dtype=np.uint8)


def _depth():
    return np.ones((2, 2), dtype=np.float32)


def test_rgb_first_tick_holds_without_submitting_depth_none():
    fake = _Fake()
    fake._rgb_buffer[10] = (_rgb(), 1.0, 'cam')
    observation = ArenaModelExecutor._observation(fake)
    assert observation is None
    assert fake._sync_hold_reason == 'waiting_for_synchronized_rgb_depth'

    # The matching depth makes exactly one exact-stamp pair available.
    fake._depth_buffer[10] = (_depth(), 1.1)
    observation = ArenaModelExecutor._observation(fake)
    assert observation is not None
    assert observation.stamp_ns == 10
    assert observation.depth_image is not None


def test_only_exact_stamp_pair_is_selected():
    fake = _Fake()
    fake._rgb_buffer[1] = (_rgb(), 1.0, 'cam')
    fake._rgb_buffer[2] = (_rgb(), 1.0, 'cam')
    fake._depth_buffer[1] = (_depth(), 1.0)
    observation = ArenaModelExecutor._observation(fake)
    assert observation is not None
    assert observation.stamp_ns == 1


def test_pair_buffer_is_bounded():
    fake = _Fake()
    for stamp in range(20):
        ArenaModelExecutor._remember(fake._rgb_buffer, stamp, stamp)
    assert len(fake._rgb_buffer) == 16
    assert 0 not in fake._rgb_buffer
    assert 19 in fake._rgb_buffer


def test_session_clear_drops_pending_pairs():
    fake = _Fake()
    fake._rgb_buffer[10] = (_rgb(), 1.0, 'cam')
    fake._depth_buffer[10] = (_depth(), 1.0)
    ArenaModelExecutor._clear_observation_buffers(fake)
    assert fake._rgb_buffer == {}
    assert fake._depth_buffer == {}
    assert ArenaModelExecutor._observation(fake) is None


def test_adapter_without_depth_is_not_blocked():
    fake = _Fake(required=('rgb',))
    fake._rgb_buffer[10] = (_rgb(), 1.0, 'cam')
    observation = ArenaModelExecutor._observation(fake)
    assert observation is not None
    assert observation.stamp_ns == 10
    assert observation.depth_image is None


def test_same_stamp_is_not_inferred_twice():
    from arena_vln_models.execution import ExecutionConfig, ExecutionCore

    class _Adapter:
        name = 'fixture'
        capability = ModelCapability(name='fixture', raw_output_kind='fixture')

        def reset(self, **kwargs):
            return None

        def infer(self, observation):
            return SimpleNamespace(local_waypoints=[(1.0, 0.0, 0.0)], stop=False, source='native')

        def to_prediction(self, raw, observation):
            from arena_vln_models.core import ModelPrediction

            return ModelPrediction(
                adapter='fixture', poses=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                stop=False, status='running', source='native',
            )

    core = ExecutionCore(_Adapter(), ExecutionConfig(model='fixture'))
    core.reset('session', 0)
    core.release()
    core.set_instruction('go')
    from arena_vln_models.core import ModelObservation

    first = ModelObservation(
        pose=Pose2D(0.0, 0.0, 0.0), instruction='go', frame_id='odom',
        stamp_ns=10, received_at=100.0,
    )
    assert core.eligible(first, 101.0) == (True, '')
    core.commit(core.plan(first, 101.0), first, 101.0)
    assert core.eligible(first, 102.0) == (False, 'observation_not_new')
