"""Pure ROS Image decoding helpers (no rclpy import).

``sensor_msgs/Image`` rows may be padded (``step > width * bytes_per_pixel``)
and multi-byte depth may be big- or little-endian.  Never reshape the raw byte
buffer blindly.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

_CHANNELS = {
    'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1, '8uc1': 1,
}
_MULTI = {
    '16uc1': ('u2', 2), 'mono16': ('u2', 2), '32fc1': ('f4', 4), '64fc1': ('f8', 8),
}


def _dtype(base: str, is_bigendian: Any) -> np.dtype:
    big = bool(is_bigendian)
    native = np.dtype(base)
    if native.itemsize == 1:
        return native
    return np.dtype(('>' if big else '<') + base)


def _rows(msg: Any, width: int, height: int, bytes_per_pixel: int) -> np.ndarray:
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = int(getattr(msg, 'step', 0) or 0) or width * bytes_per_pixel
    if step < width * bytes_per_pixel:
        raise ValueError(f'Image.step={step} is smaller than width*bpp={width * bytes_per_pixel}')
    if len(data) < step * height:
        raise ValueError(f'Image data has {len(data)} bytes, need {step * height}')
    return data.reshape(height, step)


def decode_rgb(msg: Any) -> Optional[np.ndarray]:
    height, width = int(getattr(msg, 'height', 0) or 0), int(getattr(msg, 'width', 0) or 0)
    encoding = str(getattr(msg, 'encoding', '') or '').lower()
    if height <= 0 or width <= 0:
        return None
    channels = _CHANNELS.get(encoding)
    if channels is None:
        return None
    rows = _rows(msg, width, height, channels)
    pixels = rows[:, : width * channels].reshape(height, width, channels)
    if encoding in {'bgr8', 'bgra8'}:
        pixels = pixels[:, :, ::-1]
    if channels == 4:
        pixels = pixels[:, :, :3]
    if channels == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    return np.ascontiguousarray(pixels)


def decode_depth(msg: Any) -> Optional[np.ndarray]:
    height, width = int(getattr(msg, 'height', 0) or 0), int(getattr(msg, 'width', 0) or 0)
    encoding = str(getattr(msg, 'encoding', '') or '').lower()
    if height <= 0 or width <= 0:
        return None
    spec = _MULTI.get(encoding)
    if spec is None:
        return None
    base, bytes_per_pixel = spec
    rows = _rows(msg, width, height, bytes_per_pixel)
    raw = rows[:, : width * bytes_per_pixel].copy()
    dtype = _dtype(base, getattr(msg, 'is_bigendian', 0))
    values = raw.view(dtype).reshape(height, width)
    depth = values.astype(np.float32)
    if encoding in {'16uc1', 'mono16'}:
        depth /= 1000.0
    depth[~np.isfinite(depth)] = 0.0
    return depth


def valid_depth(depth: Any, row: int, col: int, *, max_depth_m: float = 10.0) -> Optional[float]:
    if depth is None:
        return None
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 2 or not (0 <= row < array.shape[0] and 0 <= col < array.shape[1]):
        return None
    value = float(array[row, col])
    if not math.isfinite(value) or value <= 0.0 or value > float(max_depth_m):
        return None
    return value
