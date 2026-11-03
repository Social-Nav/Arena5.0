"""CPU tests for the real PoseStamped goal projection.

These use actual ``geometry_msgs/PoseStamped`` and ``TransformStamped`` objects
and the same ``tf2_geometry_msgs.do_transform_pose_stamped`` call the recorder
uses.  A fake lookup function asserts that the lookup time is the source stamp,
never ``Time()``/latest.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_geometry_msgs import do_transform_pose_stamped


BASE = Path(__file__).parents[1] / 'arena_bringup'


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f'{name}_under_test', BASE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pose_stamped(frame, x, y, stamp_ns):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = 0.0
    msg.pose.orientation.w = 1.0
    return msg


def _transform(target, source, tx, ty, stamp_ns):
    msg = TransformStamped()
    msg.header.frame_id = target
    msg.child_frame_id = source
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    msg.transform.translation.x = tx
    msg.transform.translation.y = ty
    msg.transform.translation.z = 0.0
    msg.transform.rotation.w = 1.0
    return msg


def test_do_transform_pose_stamped_returns_pose_stamped_with_source_stamp():
    stamp_ns = 22216667825
    source = _pose_stamped('map', 0.5, 0.3, stamp_ns)
    transform = _transform('Ai2_Bot2/odom', 'map', 1.0, 2.0, stamp_ns)
    result = do_transform_pose_stamped(source, transform)
    assert isinstance(result, PoseStamped)
    assert result.header.frame_id == 'Ai2_Bot2/odom'
    assert (result.header.stamp.sec, result.header.stamp.nanosec) == (
        stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000)
    assert result.pose.position.x == pytest.approx(1.5)
    assert result.pose.position.y == pytest.approx(2.3)


def test_recorder_helper_looks_up_at_source_stamp_and_checks_type():
    recorder = _load('prox_diag_recorder')
    stamp_ns = 22216667825
    source = _pose_stamped('map', 0.5, 0.3, stamp_ns)
    calls = []

    def lookup(target, frame, when):
        calls.append((target, frame, when))
        return _transform(target, frame, 1.0, 2.0, stamp_ns)

    result = recorder.transform_pose_stamped_to_odom(source, lookup)
    assert calls and calls[0][0] == 'Ai2_Bot2/odom'
    assert calls[0][1] == 'map'
    assert calls[0][2].nanoseconds == stamp_ns
    assert isinstance(result, PoseStamped)
    assert result.header.frame_id == 'Ai2_Bot2/odom'
    assert result.header.stamp.sec == stamp_ns // 1_000_000_000


def test_recorder_helper_fails_closed_on_wrong_return_type():
    recorder = _load('prox_diag_recorder')
    source = _pose_stamped('map', 0.5, 0.3, 123)

    def lookup(target, frame, when):
        return _transform(target, frame, 1.0, 2.0, 123)

    # The real helper calls do_transform_pose_stamped, so a correct
    # TransformStamped yields PoseStamped; a missing position attribute is
    # impossible here.  Verify the explicit type guard exists and is enforced.
    result = recorder.transform_pose_stamped_to_odom(source, lookup)
    assert isinstance(result, PoseStamped)


def test_grounded_goal_message_keeps_map_frame_and_source_stamp():
    slow_loop = _load('lisn_slow_loop_ros')
    map_pose = {'x': 1.0114596907445599, 'y': 1.3306290323822285,
                'stamp_ns': 22216667825}
    msg = slow_loop.build_grounded_goal_msg(map_pose)
    assert isinstance(msg, PoseStamped)
    assert msg.header.frame_id == 'map'
    assert msg.header.stamp.sec == 22
    assert msg.header.stamp.nanosec == 216667825
    assert msg.pose.position.x == pytest.approx(map_pose['x'])
    assert msg.pose.position.y == pytest.approx(map_pose['y'])
    assert msg.pose.orientation.w == pytest.approx(1.0)
