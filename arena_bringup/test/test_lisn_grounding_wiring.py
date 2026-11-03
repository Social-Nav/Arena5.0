"""CPU tests for LISN depth decoding (dtype, stride padding, endian, scale)."""
from types import SimpleNamespace

import numpy as np
import pytest

from arena_bringup.lisn_slow_loop_ros import LisnSlowLoopNode, strip_ros_args


def _msg(data, *, encoding, height, width, step, bigendian=0):
    return SimpleNamespace(data=data, encoding=encoding, height=height, width=width,
                           step=step, is_bigendian=bigendian)


def test_32fc1_float32_with_row_padding():
    payload = np.array([[1.0, 2.0, 9.9], [3.0, 4.0, 9.9]], dtype=np.float32)
    depth = LisnSlowLoopNode._depth_metric(
        _msg(payload.tobytes(), encoding='32FC1', height=2, width=2, step=12)
    )
    assert depth.dtype == np.float32
    assert depth.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_64fc1_float64():
    payload = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    depth = LisnSlowLoopNode._depth_metric(
        _msg(payload.tobytes(), encoding='64FC1', height=2, width=2, step=16)
    )
    assert depth.dtype == np.float64
    assert depth.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_16uc1_millimeters_with_padding():
    payload = np.array([[1000, 2000, 0], [3000, 4000, 0]], dtype=np.uint16)
    depth = LisnSlowLoopNode._depth_metric(
        _msg(payload.tobytes(), encoding='16UC1', height=2, width=2, step=6)
    )
    assert depth.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_16uc1_bigendian_is_byteswapped():
    payload = np.array([[1000, 2000], [3000, 4000]], dtype='>u2')
    depth = LisnSlowLoopNode._depth_metric(
        _msg(payload.tobytes(), encoding='16UC1', height=2, width=2, step=4, bigendian=1)
    )
    assert depth.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_strip_ros_args_keeps_node_args():
    argv = ['--evidence', 'x', '--grounding-live', 'true',
            '--ros-args', '-r', '__node:=lisn_slow_loop_bridge', '--log-level', 'warn']
    assert strip_ros_args(argv) == ['--evidence', 'x', '--grounding-live', 'true']
    assert strip_ros_args(['--evidence', 'x']) == ['--evidence', 'x']


def test_unsupported_depth_encoding_fails_closed():
    with pytest.raises(ValueError, match='unsupported depth encoding'):
        LisnSlowLoopNode._depth_metric(
            _msg(b'\x00' * 8, encoding='rgb8', height=2, width=2, step=4)
        )


def test_invalid_stride_fails_closed():
    with pytest.raises(ValueError, match='row stride'):
        LisnSlowLoopNode._depth_metric(
            _msg(b'\x00' * 4, encoding='32FC1', height=1, width=2, step=6)
        )
