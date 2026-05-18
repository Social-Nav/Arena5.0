from __future__ import annotations

import math
import textwrap
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

    goal_distance = decision.debug.get('goal_distance')
    yaw_error = decision.debug.get('yaw_error')
    infer_time = decision.debug.get('infer_time_sec')
    availability = observation.metadata or {}
    instruction_lines = _wrap_line('instruction: ', observation.instruction[:220], max(width // 12, 28))

    text_lines = [
        f'backend: {backend_name}',
        f'status: {decision.status}',
        f'cmd_vel: vx={decision.linear_x:.3f} wz={decision.angular_z:.3f}',
        f'degraded: {decision.degraded}',
        _format_scalar('goal_distance', goal_distance) if isinstance(goal_distance, (float, int)) else 'goal_distance: n/a',
        _format_scalar('yaw_error', yaw_error) if isinstance(yaw_error, (float, int)) else 'yaw_error: n/a',
        _format_scalar('infer_time_sec', infer_time, precision=4)
        if isinstance(infer_time, (float, int))
        else 'infer_time_sec: n/a',
        (
            'modalities: '
            f"rgb={'yes' if availability.get('rgb_available') else 'no'} "
            f"depth={'yes' if availability.get('depth_available') else 'no'} "
            f"camera_info={'yes' if availability.get('camera_info_available') else 'no'}"
        ),
        f'look_down: {bool(decision.debug.get("look_down", observation.look_down))}',
    ]

    selected_action = decision.debug.get('selected_action')
    action_label = decision.debug.get('action_label')
    native_action_label = decision.debug.get('native_action_label', action_label)
    effective_action_label = decision.debug.get('effective_action_label', action_label)
    if selected_action is not None:
        action_text = f'action: {selected_action} native={native_action_label or "n/a"}'
        if effective_action_label != native_action_label:
            action_text += f' effective={effective_action_label}'
        if decision.debug.get('invert_discrete_turns'):
            action_text += ' invert=true'
        text_lines.append(action_text)
    history_tail = decision.debug.get('action_history_tail')
    if history_tail not in (None, ''):
        text_lines.extend(_wrap_line('action_history: ', str(history_tail), max(width // 12, 24)))
    sensor_ages = decision.debug.get('sensor_ages_sec')
    if isinstance(sensor_ages, dict):
        age_parts = []
        stale_after = decision.debug.get('stale_after_sec') or availability.get('stale_after_sec') or 0.0
        try:
            stale_after = float(stale_after)
        except (TypeError, ValueError):
            stale_after = 0.0
        for key in ('rgb', 'depth', 'camera_info'):
            age = sensor_ages.get(key)
            if isinstance(age, (float, int)):
                marker = ' stale' if stale_after > 0.0 and float(age) > stale_after else ''
                age_parts.append(f'{key}={float(age):.2f}s{marker}')
            else:
                age_parts.append(f'{key}=n/a')
        text_lines.extend(_wrap_line('freshness: ', ' '.join(age_parts), max(width // 12, 24)))

    for key in (
        'adapter_target',
        'shim_mode',
        'camera_frame_id',
        'remaining_action_queue',
        'failure_reason',
        'shim_reason',
    ):
        value = decision.debug.get(key)
        if value is None and key == 'camera_frame_id':
            value = observation.camera_frame_id
        if value not in (None, ''):
            text_lines.extend(_wrap_line(f'{key}: ', str(value), max(width // 12, 24)))

    trajectory_preview = decision.debug.get('trajectory_first_step', decision.debug.get('trajectory_preview'))
    if trajectory_preview not in (None, ''):
        text_lines.extend(_wrap_line('trajectory: ', str(trajectory_preview), max(width // 12, 24)))

    target_pixel_text = decision.debug.get('target_pixel') or decision.debug.get('subgoal_pixel') or decision.debug.get('goal_pixel')
    if target_pixel_text not in (None, ''):
        text_lines.append(f'target_pixel: {target_pixel_text}')

    text_lines.extend(instruction_lines)

    line_height = 18
    panel_height = min(max(140, 18 + line_height * len(text_lines)), max(140, height - 16))
    draw.rectangle([(8, 8), (width - 8, panel_height)], fill=(0, 0, 0), outline=(0, 255, 0), width=2)

    y = 16
    for line in text_lines:
        draw.text((18, y), line, fill=(255, 255, 255))
        y += line_height

    target_pixel = _coerce_point(
        decision.debug.get('target_pixel')
        or decision.debug.get('subgoal_pixel')
        or decision.debug.get('goal_pixel')
    )
    if target_pixel is not None:
        _draw_marker(draw, target_pixel, (255, 64, 64), 'target')

    trajectory_pixels = _coerce_points(decision.debug.get('trajectory_pixels') or decision.debug.get('path_pixels'))
    waypoint_pixels = _coerce_points(decision.debug.get('waypoint_pixels'))
    path_pixels = trajectory_pixels or waypoint_pixels
    if path_pixels:
        _draw_path(draw, path_pixels, (64, 255, 255), 'trajectory' if trajectory_pixels else 'waypoints')

    if trajectory_pixels and waypoint_pixels:
        _draw_path(draw, waypoint_pixels, (64, 128, 255), 'waypoints')

    overlay_points = _coerce_points(decision.debug.get('overlay_points'))
    if overlay_points:
        _draw_path(draw, overlay_points, (255, 255, 64), 'points')

    center = (width // 2, height - 36)
    draw.ellipse((center[0] - 7, center[1] - 7, center[0] + 7, center[1] + 7), fill=(255, 255, 255))

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

    if isinstance(yaw_error, (float, int)):
        # Goal bearing in image coordinates: straight up is aligned with the robot,
        # positive yaw_error is to the robot's left.
        goal_heading = -math.pi / 2.0 - float(yaw_error)
        goal_end = (
            int(center[0] + math.cos(goal_heading) * 58),
            int(center[1] + math.sin(goal_heading) * 58),
        )
        _draw_arrow(draw, center, goal_end, (255, 255, 64), width=3, label='goal')

    return np.asarray(canvas)
