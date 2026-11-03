"""Deterministic replay of the first real Qwen grounding rejection.

The bundle/K/TF values below come from the live
``GROUNDING_FIRST_BUNDLE.json`` receipt captured in the 20261030_193009 run.
No model request is made here; the exact rejected goal values are replayed
through the production resolver.
"""
import numpy as np
import pytest

from arena_bringup.lisn_grounding import GroundingError, resolve_grounded_goal
from arena_bringup.lisn_slow_loop_ros import (
    LISN_GROUNDING_PROMPT,
    build_grounding_text,
)

SOURCE_FRAME = 'Ai2_Bot2/base_footprint/head_camera'
SOURCE_STAMP_NS = 36_516_668_571
WIDTH, HEIGHT = 640, 480
K = (387.8787834656402, 0.0, 320.0, 0.0, 387.8787834656402, 240.0, 0.0, 0.0, 1.0)
TF = (
    (0.016337081777036966, 0.49785697951568975, -0.86710525757057, 0.29083386022908697),
    (0.9998665409599974, -0.008139180712819494, 0.014165239356042247, -6.206130815284729),
    (-5.263108486863732e-06, -0.8672209532090642, -0.4979235064620484, 1.6460177781223873),
    (0.0, 0.0, 0.0, 1.0),
)


def _depth(value=2.0):
    return np.full((HEIGHT, WIDTH), value, dtype=np.float32)


def _resolve(goal, *, depth=None):
    return resolve_grounded_goal(
        goal=goal,
        width=WIDTH,
        height=HEIGHT,
        depth_image=_depth() if depth is None else depth,
        camera_info_k=K,
        tf_map_camera=TF,
        expected_frame_id=SOURCE_FRAME,
        source_stamp_ns=SOURCE_STAMP_NS,
        min_confidence=0.5,
    )


def _goal(*, frame_id='camera', units='pixel', x=350, y=400, confidence=0.95):
    return {
        'kind': 'pixel_goal',
        'frame_id': frame_id,
        'stamp_ns': 1_715_683_200_000_000_000,  # the value the VLM fabricated
        'units': units,
        'value': {'x': x, 'y': y},
        'confidence': confidence,
    }


def test_replay_vlm_camera_frame_is_rejected():
    with pytest.raises(GroundingError, match='camera_frame_mismatch:camera'):
        _resolve(_goal(frame_id='camera', x=350, y=400))


def test_replay_vlm_out_of_image_pixel_is_rejected():
    with pytest.raises(GroundingError, match='point_out_of_image'):
        _resolve(_goal(frame_id=SOURCE_FRAME, x=320, y=480))


def test_replay_missing_confidence_is_rejected():
    goal = _goal(frame_id=SOURCE_FRAME, x=350, y=400)
    goal.pop('confidence')
    with pytest.raises(GroundingError, match='missing_or_invalid_confidence'):
        _resolve(goal)


def test_corrected_normalized_source_frame_resolves():
    result = _resolve(_goal(frame_id=SOURCE_FRAME, units='normalized', x=0.55, y=0.62))
    assert np.isfinite(result['map_pose']['x'])
    assert np.isfinite(result['map_pose']['y'])
    assert result['point']['units'] == 'normalized'
    assert result['point']['frame_id'] == SOURCE_FRAME


def test_invalid_depth_fails_closed():
    depth = _depth()
    depth[297][351] = 0.0
    with pytest.raises(GroundingError, match='invalid_depth'):
        _resolve(_goal(frame_id=SOURCE_FRAME, units='normalized', x=0.55, y=0.62),
                 depth=depth)


def test_grounding_text_carries_host_source_context():
    text = build_grounding_text(
        LISN_GROUNDING_PROMPT, instruction='Walk straight.',
        frame_id=SOURCE_FRAME, stamp_ns=SOURCE_STAMP_NS,
        width=WIDTH, height=HEIGHT,
    )
    assert f'Source frame_id: {SOURCE_FRAME}' in text
    assert f'Source stamp_ns: {SOURCE_STAMP_NS}' in text
    assert f'Image width: {WIDTH}, height: {HEIGHT}' in text
    assert 'normalized' in text
    assert 'Instruction: Walk straight.' in text
