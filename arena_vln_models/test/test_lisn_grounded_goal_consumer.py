"""Focused CPU tests for the LISN grounded-goal -> common FollowPath decision."""
from types import SimpleNamespace

import pytest

from arena_vln_models.executor import grounded_goal_decision, grounded_goal_identity


def _msg(*, frame_id='map', x=1.0, y=2.0, sec=10, nanosec=0):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id, stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)),
    )


def test_grounded_goal_maps_to_follow_path_decision():
    decision = grounded_goal_decision(_msg())
    assert decision.action == 'follow_path'
    assert decision.poses == [(1.0, 2.0, 0.0)]
    assert decision.debug['source'] == 'lisn_grounding_live'
    assert decision.debug['stamp_ns'] == 10_000_000_000


def test_grounded_goal_rejects_non_map_frame():
    with pytest.raises(ValueError, match='map'):
        grounded_goal_decision(_msg(frame_id='camera'))


def test_grounded_goal_identity_is_map_and_original_stamp():
    decision = grounded_goal_decision(_msg(sec=10, nanosec=123))
    assert grounded_goal_identity(decision) == ('map', 10_000_000_123)


def test_grounded_goal_identity_rejects_missing_identity():
    decision = grounded_goal_decision(_msg())
    decision.debug['frame_id'] = 'odom'
    with pytest.raises(ValueError, match='map_identity'):
        grounded_goal_identity(decision)
    decision.debug['frame_id'] = 'map'
    decision.debug['stamp_ns'] = 0
    with pytest.raises(ValueError, match='map_identity'):
        grounded_goal_identity(decision)


def test_grounded_goal_rejects_nan_coordinates():
    with pytest.raises(ValueError, match='not_finite'):
        grounded_goal_decision(_msg(x=float('nan')))
    with pytest.raises(ValueError, match='not_finite'):
        grounded_goal_decision(_msg(y=float('inf')))


def test_grounded_goal_rejects_missing_stamp():
    with pytest.raises(ValueError, match='stamp_missing'):
        grounded_goal_decision(_msg(sec=0, nanosec=0))
