from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ArenaVLNAdapter(Protocol):
    """Protocol implemented by Arena VLN model adapters."""

    def compute(self, observation: Any) -> dict[str, Any]:
        """Return a mapping accepted by arena_vln_models' PythonAdapterBackend."""


@dataclass(frozen=True)
class BackendCapability:
    """Static metadata describing a VLN backend adapter."""

    name: str
    required_inputs: tuple[str, ...] = ()
    output_modes: tuple[str, ...] = ()
    supports_batch: bool = False
    notes: str = ''


@dataclass
class ArenaVLNOutput:
    """Canonical Arena-facing VLN output mapping."""

    linear_x: Optional[float] = None
    angular_z: Optional[float] = None
    discrete_action: Optional[int] = None
    output_trajectory: Any = None
    output_pixel: Any = None
    status: str = 'adapter_command'
    degraded: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            'status': self.status,
            'degraded': bool(self.degraded),
            'debug': sanitize_debug(self.debug),
        }
        if self.linear_x is not None:
            result['linear_x'] = float(self.linear_x)
        if self.angular_z is not None:
            result['angular_z'] = float(self.angular_z)
        if self.discrete_action is not None:
            result['discrete_action'] = int(self.discrete_action)
        if self.output_trajectory is not None:
            result['output_trajectory'] = to_jsonable(self.output_trajectory)
        if self.output_pixel is not None:
            result['output_pixel'] = to_jsonable(self.output_pixel)
        return result


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return to_jsonable(value.tolist())
    return str(value)


def sanitize_debug(debug: Any) -> dict[str, Any]:
    if debug is None:
        return {}
    if isinstance(debug, dict):
        return {str(key): to_jsonable(value) for key, value in debug.items()}
    return {'debug_value': to_jsonable(debug)}


def normalize_rgb(image: Any) -> Optional[np.ndarray]:
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        return None
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] != 3:
        return None
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 255.0)
        array = array.astype(np.uint8)
    return array.copy()


def normalize_depth(depth: Any, *, reference_shape: Optional[tuple[int, int]] = None) -> Optional[np.ndarray]:
    if depth is None:
        if reference_shape is None:
            return None
        return np.zeros(reference_shape, dtype=np.float32)
    array = np.asarray(depth)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        return None
    normalized = array.astype(np.float32, copy=True)
    if array.dtype == np.uint16:
        normalized /= 1000.0
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def depth_debug(depth: Optional[np.ndarray]) -> dict[str, Any]:
    if depth is None:
        return {}
    finite = depth[np.isfinite(depth)]
    debug: dict[str, Any] = {'depth_shape': [int(dim) for dim in depth.shape]}
    if finite.size > 0:
        debug.update({'depth_min': float(np.min(finite)), 'depth_max': float(np.max(finite))})
    return debug


def pose_vector(observation: Any) -> np.ndarray:
    pose = getattr(observation, 'pose', None)
    if pose is None:
        return np.zeros(3, dtype=np.float32)
    return np.asarray([
        float(getattr(pose, 'x', 0.0)),
        float(getattr(pose, 'y', 0.0)),
        float(getattr(pose, 'yaw', 0.0)),
    ], dtype=np.float32)


def camera_intrinsic_matrix(observation: Any, rgb: Optional[np.ndarray]) -> np.ndarray:
    intrinsics = getattr(observation, 'camera_intrinsics', None)
    if intrinsics is not None:
        values = [float(value) for value in intrinsics]
        if len(values) >= 16:
            return np.asarray(values[:16], dtype=np.float32).reshape(4, 4)
        if len(values) >= 9:
            return np.asarray(values[:9], dtype=np.float32).reshape(3, 3)

    width = int(rgb.shape[1]) if rgb is not None and rgb.ndim >= 2 else 640
    height = int(rgb.shape[0]) if rgb is not None and rgb.ndim >= 2 else 480
    fx = max(width / 2.0, 1.0)
    fy = max(height / 2.0, 1.0)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.asarray(
        [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def goal_delta(observation: Any) -> Optional[tuple[float, float, float, float]]:
    pose = getattr(observation, 'pose', None)
    goal = getattr(observation, 'goal', None)
    if pose is None or goal is None:
        return None
    import math

    pose_x = float(getattr(pose, 'x', 0.0))
    pose_y = float(getattr(pose, 'y', 0.0))
    pose_yaw = float(getattr(pose, 'yaw', 0.0))
    dx = float(getattr(goal, 'x', 0.0)) - pose_x
    dy = float(getattr(goal, 'y', 0.0)) - pose_y
    local_x = math.cos(-pose_yaw) * dx - math.sin(-pose_yaw) * dy
    local_y = math.sin(-pose_yaw) * dx + math.cos(-pose_yaw) * dy
    dist = math.hypot(local_x, local_y)
    yaw = math.atan2(local_y, local_x)
    return local_x, local_y, dist, yaw


def safe_stop(status: str, reason: str, *, debug: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    merged_debug = sanitize_debug(debug)
    merged_debug.update({'shim_reason': reason, 'safe_stop': True})
    return ArenaVLNOutput(discrete_action=0, status=status, degraded=True, debug=merged_debug).to_mapping()


def normalize_backend_output(output: Any, *, default_status: str = 'adapter_command') -> dict[str, Any]:
    if isinstance(output, ArenaVLNOutput):
        return output.to_mapping()
    if isinstance(output, dict):
        debug = sanitize_debug(output.get('debug', {}))
        result: dict[str, Any] = {
            'status': str(output.get('status', default_status)),
            'degraded': bool(output.get('degraded', False)),
            'debug': debug,
        }
        for source_key, target_key in (
            ('linear_x', 'linear_x'),
            ('angular_z', 'angular_z'),
            ('discrete_action', 'discrete_action'),
            ('output_action', 'discrete_action'),
            ('action', 'discrete_action'),
            ('output_trajectory', 'output_trajectory'),
            ('trajectory', 'output_trajectory'),
            ('output_pixel', 'output_pixel'),
            ('target_pixel', 'output_pixel'),
            ('pixel_goal', 'output_pixel'),
        ):
            if source_key in output and target_key not in result:
                result[target_key] = to_jsonable(output[source_key])
        return result

    action = getattr(output, 'output_action', None)
    trajectory = getattr(output, 'output_trajectory', None)
    pixel = getattr(output, 'output_pixel', None)
    debug: dict[str, Any] = {}
    if hasattr(output, 'llm_output') and getattr(output, 'llm_output'):
        debug['llm_output'] = str(getattr(output, 'llm_output'))[:240]
    result = ArenaVLNOutput(status=default_status, debug=debug)
    if action is not None:
        action_list = np.asarray(action).reshape(-1).tolist()
        if action_list:
            result.discrete_action = int(action_list[0])
            if len(action_list) > 1:
                result.debug['action_sequence_tail'] = [int(item) for item in action_list[1:]]
    if trajectory is not None:
        result.output_trajectory = trajectory
    if pixel is not None:
        result.output_pixel = pixel
    if result.discrete_action is None and result.output_trajectory is None:
        return safe_stop('empty_model_output', 'backend returned neither action nor trajectory', debug=result.debug)
    return result.to_mapping()
