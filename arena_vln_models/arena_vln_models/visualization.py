from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping
from typing import Optional

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

try:
    from sensor_msgs.msg import Image
except ModuleNotFoundError:  # pragma: no cover - exercised in non-ROS unit tests
    class _Header:
        def __init__(self) -> None:
            self.frame_id = ''

    class Image:  # type: ignore[override]
        def __init__(self) -> None:
            self.header = _Header()
            self.height = 0
            self.width = 0
            self.encoding = ''
            self.is_bigendian = 0
            self.step = 0
            self.data = b''

from arena_vln_models.backends import DualVLNDecision, DualVLNObservation


def image_msg_to_numpy(message: Image) -> Optional[np.ndarray]:
    data = np.frombuffer(message.data, dtype=np.uint8)
    if message.encoding in ('rgb8', 'bgr8'):
        image = data.reshape((message.height, message.step // 3, 3))[:, : message.width, :].copy()
        if message.encoding == 'bgr8':
            image = image[:, :, ::-1]
        return image
    if message.encoding in ('rgba8', 'bgra8'):
        image = data.reshape((message.height, message.step // 4, 4))[:, : message.width, :4].copy()
        if message.encoding == 'bgra8':
            image = image[:, :, [2, 1, 0, 3]]
        return image[:, :, :3]
    if message.encoding == 'mono8':
        mono = data.reshape((message.height, message.step))[:, : message.width].copy()
        return np.repeat(mono[:, :, None], 3, axis=2)
    if message.encoding in ('mono16', '16UC1'):
        depth = np.frombuffer(message.data, dtype=np.uint16)
        depth = depth.reshape((message.height, message.step // 2))[:, : message.width].astype(np.float32)
        return depth / 1000.0
    if message.encoding == '32FC1':
        depth = np.frombuffer(message.data, dtype=np.float32)
        return depth.reshape((message.height, message.step // 4))[:, : message.width].copy()
    return None


def numpy_to_image_msg(image: np.ndarray, reference: Optional[Image] = None, *, frame_id: str = '') -> Image:
    msg = Image()
    if reference is not None:
        msg.header = reference.header
    elif frame_id:
        msg.header.frame_id = frame_id
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = int(image.shape[1] * 3)
    msg.data = image.astype(np.uint8).tobytes()
    return msg


def _coerce_point(value: object) -> Optional[tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (int(round(float(value[0]))), int(round(float(value[1]))))
    except (TypeError, ValueError):
        return None


def _coerce_points(value: object) -> list[tuple[int, int]]:
    if not isinstance(value, (list, tuple)):
        return []
    points = []
    for item in value:
        point = _coerce_point(item)
        if point is not None:
            points.append(point)
    return points


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _mapping_get(mapping: object, *keys: str) -> object:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _point_in_image(point: tuple[int, int], width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height


def _coerce_image_point(value: object, width: int, height: int) -> Optional[tuple[int, int]]:
    # Some adapters return normalized [0, 1] coordinates.  Treat them as image
    # pixels only when both axes are in the normalized range.
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError):
            return None
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            normalized = (int(round(x * (width - 1))), int(round(y * (height - 1))))
            if _point_in_image(normalized, width, height):
                return normalized
    point = _coerce_point(value)
    if point is None:
        return None
    if _point_in_image(point, width, height):
        return point
    return None


def _coerce_local_trajectory_points(value: object) -> list[tuple[float, float]]:
    """Convert a trajectory-like object to local (forward, lateral) points.

    InternNav's System-1 trajectory is usually a list of dense local waypoints
    ``[x_forward_m, y_left_m, yaw]``.  Some debug paths contain a single control
    step instead.  Keep the parser permissive so the overlay can still show a
    useful path when the adapter shape changes slightly.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = _first_present(
            value.get('output_trajectory'),
            value.get('trajectory'),
            value.get('trajectory_preview'),
            value.get('points'),
        )
    if not isinstance(value, (list, tuple)) or not value:
        return []

    # Single waypoint/control step: [x, y, yaw].
    if len(value) >= 2 and not isinstance(value[0], (list, tuple, dict, np.ndarray)):
        try:
            return [(float(value[0]), float(value[1]))]
        except (TypeError, ValueError):
            return []

    points: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, Mapping):
            item = _first_present(item.get('point'), item.get('xy'), item.get('position'), item.get('waypoint'))
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return points


def _local_trajectory_to_image_points(
    trajectory: object,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    local_points = _coerce_local_trajectory_points(trajectory)
    if not local_points:
        return []

    max_forward = max(max(point[0] for point in local_points), 0.5)
    lateral_extent = max(max(abs(point[1]) for point in local_points), 0.35)
    # Draw a paper-style egocentric route ribbon on the lower half of the camera
    # image: robot at bottom center, forward direction upwards, lateral left/right
    # mapped to screen left/right.  This is intentionally a visualization of the
    # model's local trajectory, not a calibrated 3D projection.
    origin = (width // 2, height - max(36, height // 12))
    vertical_scale = (height * 0.44) / max_forward
    horizontal_scale = (width * 0.30) / lateral_extent
    pixels = [origin]
    for forward_m, lateral_m in local_points:
        if not math.isfinite(forward_m) or not math.isfinite(lateral_m):
            continue
        x = int(round(origin[0] + lateral_m * horizontal_scale))
        y = int(round(origin[1] - max(forward_m, 0.0) * vertical_scale))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        pixels.append((x, y))
    return pixels


def _short_text(value: object, limit: int = 72) -> str:
    text = str(value or '')
    return text if len(text) <= limit else text[: max(limit - 1, 0)] + '…'


def _draw_marker(draw: ImageDraw.ImageDraw, point: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
    x, y = point
    radius = 8
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line([(x - 12, y), (x + 12, y)], fill=color, width=2)
    draw.line([(x, y - 12), (x, y + 12)], fill=color, width=2)
    draw.text((x + 10, y - 18), label, fill=color)


def _draw_path(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: tuple[int, int, int], label: str) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    for point in points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
    if points:
        draw.text((points[0][0] + 8, points[0][1] + 8), label, fill=color)


def _draw_label_panel(draw: ImageDraw.ImageDraw, lines: list[str], width: int) -> None:
    line_height = 18
    panel_width = min(width - 16, max(260, 18 + max((len(line) for line in lines), default=0) * 8))
    panel_height = 18 + line_height * max(len(lines), 1) + 8
    draw.rectangle([(8, 8), (8 + panel_width, panel_height)], fill=(0, 0, 0), outline=(0, 255, 0), width=2)
    for index, line in enumerate(lines):
        draw.text((18, 18 + index * line_height), line, fill=(255, 255, 255))


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    width: int = 5,
    label: str = '',
) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = max(10, width * 3)
    spread = math.radians(28)
    left = (
        int(end[0] - head * math.cos(angle - spread)),
        int(end[1] - head * math.sin(angle - spread)),
    )
    right = (
        int(end[0] - head * math.cos(angle + spread)),
        int(end[1] - head * math.sin(angle + spread)),
    )
    draw.polygon([end, left, right], fill=color)
    if label:
        draw.text((end[0] + 8, end[1] - 14), label, fill=color)


def _draw_action_glyph(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    action_label: object,
    linear_x: float,
    angular_z: float,
    color: tuple[int, int, int],
) -> None:
    label = str(action_label or '').lower()
    is_stop = 'stop' in label or (abs(linear_x) < 1e-6 and abs(angular_z) < 1e-6)
    is_left = 'left' in label or angular_z > 1e-6
    is_right = 'right' in label or angular_z < -1e-6
    is_forward = 'forward' in label or (linear_x > 1e-6 and not is_left and not is_right)
    if is_stop:
        radius = 28
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline=color, width=5)
        draw.line((center[0] - 18, center[1] - 18, center[0] + 18, center[1] + 18), fill=color, width=5)
        draw.text((center[0] - 22, center[1] + 34), 'model action: STOP', fill=color)
        return
    if is_left or is_right:
        radius = 42
        bbox = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
        if is_left:
            draw.arc(bbox, start=205, end=25, fill=color, width=6)
            end = (center[0] - 28, center[1] - 32)
            points = [end, (end[0] + 18, end[1] - 4), (end[0] + 6, end[1] + 16)]
            text = 'model action: LEFT ARC' if linear_x > 1e-6 else 'model action: LEFT'
        else:
            draw.arc(bbox, start=155, end=335, fill=color, width=6)
            end = (center[0] + 28, center[1] - 32)
            points = [end, (end[0] - 18, end[1] - 4), (end[0] - 6, end[1] + 16)]
            text = 'model action: RIGHT ARC' if linear_x > 1e-6 else 'model action: RIGHT'
        draw.polygon(points, fill=color)
        if linear_x > 1e-6:
            _draw_arrow(draw, center, (center[0], center[1] - 48), color, width=3, label='vx')
        draw.text((center[0] - 52, center[1] + 52), text, fill=color)
        return
    if is_forward:
        length = max(34, min(90, int(130.0 * max(linear_x, 0.05))))
        _draw_arrow(draw, center, (center[0], center[1] - length), color, width=7, label='model action: FWD')
        return
    _draw_arrow(draw, center, (center[0], center[1] - 42), color, width=5, label='model action')


def _format_scalar(name: str, value: object, *, precision: int = 3) -> str:
    if isinstance(value, (float, int)):
        return f'{name}: {float(value):.{precision}f}'
    return f'{name}: {value}'


def _wrap_line(prefix: str, value: str, width: int) -> list[str]:
    chunks = textwrap.wrap(value, width=max(width, 12)) or ['']
    lines = []
    for index, chunk in enumerate(chunks):
        lines.append(f'{prefix}{chunk}' if index == 0 else f'  {chunk}')
    return lines


def render_debug_overlay(
    rgb_image: np.ndarray,
    observation: DualVLNObservation,
    decision: DualVLNDecision,
    *,
    backend_name: str,
) -> np.ndarray:
    if rgb_image.ndim == 2:
        rgb_image = np.repeat(rgb_image[:, :, None], 3, axis=2)
    canvas = PILImage.fromarray(rgb_image.astype(np.uint8), mode='RGB')
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    selected_action = decision.debug.get('selected_action')
    action_label = decision.debug.get('action_label')
    native_action_label = decision.debug.get('native_action_label', action_label)
    effective_action_label = decision.debug.get('effective_action_label', action_label)
    action_parts = []
    if selected_action is not None:
        action_parts.append(str(selected_action))
    display_label = effective_action_label or native_action_label or action_label
    if display_label not in (None, ''):
        action_parts.append(str(display_label))
    elif abs(float(decision.linear_x)) < 1e-6 and abs(float(decision.angular_z)) < 1e-6:
        action_parts.append('stop')
    action_text = 'action: ' + (' | '.join(action_parts) if action_parts else 'n/a')

    debug = decision.debug
    raw_model_output = debug.get('raw_model_output')
    output_mode = _first_present(
        debug.get('model_generation_output_mode'),
        debug.get('selected_output_mode'),
        _mapping_get(raw_model_output, 'model_generation_output_mode', 'selected_output_mode'),
    )
    pixel_goal_raw = _first_present(
        debug.get('pixel_goal'),
        debug.get('target_pixel'),
        debug.get('output_pixel'),
        _mapping_get(raw_model_output, 'pixel_goal', 'target_pixel', 'output_pixel'),
    )
    pixel_goal = _coerce_image_point(pixel_goal_raw, width, height)
    trajectory_raw = _first_present(
        debug.get('trajectory_preview'),
        debug.get('output_trajectory'),
        debug.get('trajectory'),
        _mapping_get(raw_model_output, 'output_trajectory', 'trajectory', 'trajectory_preview'),
    )
    trajectory_pixels_raw = _first_present(
        debug.get('trajectory_pixels'),
        debug.get('image_trajectory'),
        _mapping_get(raw_model_output, 'trajectory_pixels', 'image_trajectory'),
    )
    trajectory_pixels = [point for point in _coerce_points(trajectory_pixels_raw) if _point_in_image(point, width, height)]
    if not trajectory_pixels:
        trajectory_pixels = _local_trajectory_to_image_points(trajectory_raw, width, height)
    trajectory_point_count = max(len(trajectory_pixels) - 1, 0) if trajectory_pixels else 0

    info_lines = [
        action_text,
        f'output: {_short_text(output_mode or decision.status, 56)}',
        f'pixel goal: {list(pixel_goal) if pixel_goal is not None else _short_text(pixel_goal_raw or "n/a", 44)}',
        f'trajectory: {trajectory_point_count} pts | vx={float(decision.linear_x):.2f} wz={float(decision.angular_z):.2f}',
    ]
    raw_llm = _first_present(
        debug.get('raw_output_text'),
        debug.get('subprocess_llm_output'),
        debug.get('adapter_llm_output'),
        debug.get('llm_output'),
    )
    if raw_llm:
        info_lines.append('llm: ' + _short_text(raw_llm, 68))
    _draw_label_panel(draw, info_lines, width)

    center = (width // 2, height - 36)
    draw.ellipse((center[0] - 7, center[1] - 7, center[0] + 7, center[1] + 7), fill=(255, 255, 255))

    if trajectory_pixels:
        _draw_path(draw, trajectory_pixels, (64, 200, 255), 'S1 trajectory')
        if len(trajectory_pixels) >= 2:
            _draw_arrow(draw, trajectory_pixels[-2], trajectory_pixels[-1], (64, 200, 255), width=3)

    if pixel_goal is not None:
        _draw_marker(draw, pixel_goal, (80, 255, 80), 'S2 pixel goal')

    glyph_color = (80, 255, 80) if not decision.degraded else (255, 160, 80)
    glyph_center = (width - 76, height - 70)
    draw.rectangle((glyph_center[0] - 48, glyph_center[1] - 34, glyph_center[0] + 48, glyph_center[1] + 34), fill=(0, 0, 0), outline=glyph_color, width=2)
    glyph = '?'
    if selected_action in (0, '0'):
        glyph = 'STOP'
    elif selected_action in (1, '1'):
        glyph = 'FWD'
    elif selected_action in (2, '2', 3, '3'):
        if decision.angular_z < -1e-6:
            glyph = 'RIGHT'
        elif decision.angular_z > 1e-6:
            glyph = 'LEFT'
        else:
            glyph = 'TURN'
    elif selected_action in (5, '5'):
        glyph = 'LOOK'
    draw.text((glyph_center[0] - 22, glyph_center[1] - 11), glyph, fill=glyph_color)
    if effective_action_label not in (None, ''):
        draw.text((glyph_center[0] - 42, glyph_center[1] + 12), str(effective_action_label)[:12], fill=glyph_color)

    arrow_color = (0, 255, 0) if not decision.degraded else (255, 128, 0)
    _draw_action_glyph(
        draw,
        center,
        effective_action_label or action_label,
        float(decision.linear_x),
        float(decision.angular_z),
        arrow_color,
    )

    return np.asarray(canvas)
