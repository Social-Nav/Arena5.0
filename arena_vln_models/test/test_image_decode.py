"""Pure decode tests for ROS Image ``step``/``is_bigendian`` handling."""
from __future__ import annotations

import struct

import numpy as np

from arena_vln_models.executor import _depth_to_metric, _image_to_rgb


class _Msg:
    def __init__(self, *, data, height, width, encoding, step, is_bigendian=0):
        self.data = data
        self.height = height
        self.width = width
        self.encoding = encoding
        self.step = step
        self.is_bigendian = is_bigendian


def test_rgb_padded_rows_are_stripped():
    data = bytes([
        1, 2, 3, 4, 5, 6, 9, 9,
        7, 8, 9, 10, 11, 12, 9, 9,
    ])
    msg = _Msg(data=data, height=2, width=2, encoding='rgb8', step=8)
    array = _image_to_rgb(msg)
    assert array.shape == (2, 2, 3)
    assert array[0, 0].tolist() == [1, 2, 3]
    assert array[1, 1].tolist() == [10, 11, 12]


def test_depth_32fc1_little_endian_padded_rows():
    data = (
        struct.pack('<2f', 1.0, 2.0) + b'\x00' * 4
        + struct.pack('<2f', 3.0, 4.0) + b'\x00' * 4
    )
    msg = _Msg(data=data, height=2, width=2, encoding='32FC1', step=12, is_bigendian=0)
    array = _depth_to_metric(msg)
    np.testing.assert_allclose(array, [[1.0, 2.0], [3.0, 4.0]])


def test_depth_16uc1_big_endian_padded_rows():
    data = (
        struct.pack('>2H', 1000, 2000) + b'\x00' * 4
        + struct.pack('>2H', 3000, 4000) + b'\x00' * 4
    )
    msg = _Msg(data=data, height=2, width=2, encoding='16UC1', step=8, is_bigendian=1)
    array = _depth_to_metric(msg)
    np.testing.assert_allclose(array, [[1.0, 2.0], [3.0, 4.0]])


def test_depth_32fc1_big_endian():
    data = struct.pack('>2f', 1.5, 2.5) + struct.pack('>2f', 3.5, 4.5)
    msg = _Msg(data=data, height=2, width=2, encoding='32FC1', step=8, is_bigendian=1)
    array = _depth_to_metric(msg)
    np.testing.assert_allclose(array, [[1.5, 2.5], [3.5, 4.5]])
