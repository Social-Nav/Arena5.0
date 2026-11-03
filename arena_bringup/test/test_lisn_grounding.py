"""Focused CPU tests for the pure LISN grounding resolver."""
import math

import pytest

from arena_bringup.lisn_grounding import (
    GroundingError,
    normalize_point,
    people_entry_from_grounded_goal,
    project_pixel_to_map,
    resolve_grounded_goal,
)

K = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
IDENTITY_TF = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _goal(**overrides):
    value = {
        'kind': 'pixel_goal',
        'frame_id': 'Ai2_Bot2/head_camera',
        'stamp_ns': 1400000000,
        'value': {'x': 320.0, 'y': 240.0},
        'units': 'pixel',
        'confidence': 0.9,
    }
    value.update(overrides)
    return value


def test_center_pixel_projects_to_optical_axis():
    pose = project_pixel_to_map(u=320, v=240, depth_m=2.0, camera_info_k=K,
                                tf_map_camera=IDENTITY_TF)
    assert pose == {'x': 0.0, 'y': 0.0, 'z': 2.0}


def test_offset_pixel_projects_with_intrinsics():
    pose = project_pixel_to_map(u=370, v=240, depth_m=2.0, camera_info_k=K,
                                tf_map_camera=IDENTITY_TF)
    assert pose['x'] == pytest.approx(0.2)
    assert pose['y'] == pytest.approx(0.0)
    assert pose['z'] == pytest.approx(2.0)


def test_normalized_point_converts_to_pixel():
    point = normalize_point(_goal(value={'x': 0.5, 'y': 0.5}, units='normalized'),
                            width=640, height=480)
    assert point['u'] == 320
    assert point['v'] == 240


def test_map_frame_point_is_rejected():
    with pytest.raises(GroundingError, match='invalid_camera_frame'):
        normalize_point(_goal(frame_id='map'), width=640, height=480)


def test_missing_units_or_confidence_is_rejected():
    with pytest.raises(GroundingError, match='units'):
        normalize_point(_goal(units=None), width=640, height=480)
    with pytest.raises(GroundingError, match='confidence'):
        normalize_point(_goal(confidence=None), width=640, height=480)


def test_low_confidence_is_rejected():
    with pytest.raises(GroundingError, match='confidence_below_min'):
        normalize_point(_goal(confidence=0.2), width=640, height=480,
                        min_confidence=0.5)


def test_invalid_depth_and_intrinsics_fail_closed():
    with pytest.raises(GroundingError, match='invalid_depth'):
        project_pixel_to_map(u=320, v=240, depth_m=0.0, camera_info_k=K,
                             tf_map_camera=IDENTITY_TF)
    with pytest.raises(GroundingError, match='invalid_camera_info_k'):
        project_pixel_to_map(u=320, v=240, depth_m=2.0,
                             camera_info_k=[math.nan] + K[1:],
                             tf_map_camera=IDENTITY_TF)


def test_camera_frame_mismatch_is_rejected():
    depth = [[2.0 for _ in range(480)] for _ in range(480)]
    with pytest.raises(GroundingError, match='camera_frame_mismatch'):
        resolve_grounded_goal(goal=_goal(frame_id='other_camera'),
                              width=640, height=480, depth_image=depth,
                              camera_info_k=K, tf_map_camera=IDENTITY_TF,
                              expected_frame_id='Ai2_Bot2/head_camera',
                              source_stamp_ns=1400000000)


def test_trusted_goal_and_people_entry():
    depth = [[2.0 for _ in range(480)] for _ in range(480)]
    result = resolve_grounded_goal(goal=_goal(), width=640, height=480,
                                   depth_image=depth, camera_info_k=K,
                                   tf_map_camera=IDENTITY_TF,
                                   expected_frame_id='Ai2_Bot2/head_camera',
                                   source_stamp_ns=1400000000)
    assert result['trusted'] is True
    assert result['map_pose']['stamp_ns'] == 1400000000
    entry = people_entry_from_grounded_goal(name='lisn_grounded_1',
                                            map_pose=result['map_pose'],
                                            confidence=result['confidence'])
    assert entry['position']['x'] == pytest.approx(0.0)
    assert entry['source'] == 'lisn_grounding_live_v1'


def test_people_entry_requires_map_frame():
    with pytest.raises(GroundingError, match='map_frame'):
        people_entry_from_grounded_goal(
            name='x', map_pose={'frame_id': 'camera', 'x': 0.0, 'y': 0.0},
            confidence=0.9)
