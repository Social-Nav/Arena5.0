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

    for key in (
        'adapter_target',
        'shim_mode',
        'camera_frame_id',
        'selected_action',
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

    heading = 0.0
    if isinstance(yaw_error, (float, int)):
        heading = -float(yaw_error)
    magnitude = max(0.15, min(1.0, abs(decision.linear_x) / 0.3 if 0.3 > 0 else 0.15))
    length = int(70 * magnitude)
    end_point = (
        int(center[0] + math.cos(heading) * length),
        int(center[1] - math.sin(heading) * length),
    )
    arrow_color = (0, 255, 0) if not decision.degraded else (255, 128, 0)
    draw.line([center, end_point], fill=arrow_color, width=5)
    draw.polygon(
        [
            end_point,
            (end_point[0] - 8, end_point[1] - 5),
            (end_point[0] - 8, end_point[1] + 5),
        ],
        fill=arrow_color,
    )

    return np.asarray(canvas)
