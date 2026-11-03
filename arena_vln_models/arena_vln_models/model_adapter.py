"""Unified VLN model adapter protocol, registry and canonical geometry.

This module is deliberately ROS-free.  Adapters decode raw model responses into
:class:`ModelRawOutput`, then normalise them into :class:`ModelPrediction`
fixed-odom paths.  Only :mod:`arena_vln_models.executor` sends ``FollowPath``
goals; no adapter may publish ``cmd_vel``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .core import ModelObservation, ModelPrediction, Pose2D, sanitize_debug

SAMPLE_STEP_M = 0.05
MAX_SEGMENT_M = 5.0
MAX_PATH_POINTS = 4096
DEFAULT_CONTROL_MODE = 'arena_follow_path'


class ModelAdapterError(RuntimeError):
    """Base error for unified model adapters."""


class UnknownModelAdapterError(ModelAdapterError):
    """Raised when a model name is not registered."""


class CapabilityMismatchError(ModelAdapterError):
    """Raised before inference when the observation lacks required inputs."""


class ModelProtocolError(ModelAdapterError):
    """Raised when a raw model response violates its protocol."""


class ModelTimeoutError(ModelAdapterError):
    """Raised when a model request exceeds its timeout."""


class EmptyModelOutputError(ModelProtocolError):
    """Raised when a model returns neither a path, action nor pixel goal."""


@dataclass(frozen=True)
class ModelCapability:
    """Static metadata describing one model adapter."""

    name: str
    raw_output_kind: str
    required_inputs: tuple[str, ...] = ()
    output_modes: tuple[str, ...] = ('path', 'stop')
    supports_stop: bool = True
    notes: str = ''


@dataclass
class ModelRawOutput:
    """Decoded but not yet projected raw model output."""

    adapter: str
    raw: Any = None
    kind: str = ''
    text: str = ''
    local_waypoints: list[tuple[float, float, float]] = field(default_factory=list)
    stop: bool = False
    status: str = 'running'
    source: str = 'native'
    modality: str = ''
    raw_hash: str = ''
    debug: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            'adapter': self.adapter,
            'kind': self.kind,
            'source': self.source,
            'modality': self.modality,
            'raw_hash': self.raw_hash or raw_response_hash(self.raw),
            'stop': bool(self.stop),
            'status': self.status,
            'waypoint_count': len(self.local_waypoints),
            'text_preview': _bounded_text(self.text),
            'debug': bounded_debug(self.debug),
        }


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry for one model's adapter/profile/runtime defaults."""

    name: str
    adapter: str
    profile: str
    server_default: str = ''
    checkpoint_env: str = ''
    native_only: bool = False
    notes: str = ''


MODEL_SPECS: dict[str, ModelSpec] = {
    'internnav': ModelSpec(
        name='internnav',
        adapter='internnav',
        profile='internnav_follow_path_grscenes.yaml',
        server_default='http://127.0.0.1:5801/eval_dual',
        checkpoint_env='ARENA_INTERNNAV_MODEL_PATH',
        notes='InternVLA-N1 DualVLN realworld HTTP /eval_dual.',
    ),
    'navila': ModelSpec(
        name='navila',
        adapter='navila',
        profile='navila_grscenes.yaml',
        server_default='tcp://127.0.0.1:54321',
        checkpoint_env='NAVILA_MODEL_PATH',
        notes='NaVILA eight-frame TCP text-action service.',
    ),
    'lightnav': ModelSpec(
        name='lightnav',
        adapter='lightnav',
        profile='lightnav_grscenes.yaml',
        server_default='ws://127.0.0.1:8050',
        checkpoint_env='LIGHTNAV_MODEL_PATH',
        notes='LightNav WebSocket cumulative waypoint service.',
    ),
    'lisn_qwen3vl_fp8': ModelSpec(
        name='lisn_qwen3vl_fp8',
        adapter='lisn_qwen3vl_fp8',
        profile='',
        notes=(
            'Qwen/LISN producer identity. The sidecar currently emits pixel/bearing '
            'and has no common metric action/Path bridge; the adapter fails closed.'
        ),
    ),
}


def register_model_spec(spec: ModelSpec) -> ModelSpec:
    MODEL_SPECS[spec.name.strip().lower()] = spec
    return spec


def resolve_model_spec(name: str) -> ModelSpec:
    key = str(name or '').strip().lower()
    if not key:
        raise UnknownModelAdapterError('model name is empty')
    spec = MODEL_SPECS.get(key)
    if spec is None:
        raise UnknownModelAdapterError(
            f"unknown model '{name}'; supported models: {sorted(MODEL_SPECS)}"
        )
    return spec


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol implemented by all unified VLN model adapters."""

    name: str
    capability: ModelCapability

    def reset(self, *, session_id: str = '', episode: int = -1) -> None:
        """Reset per-episode model history."""

    def infer(self, observation: ModelObservation) -> ModelRawOutput:
        """Call the model and return its decoded raw output."""

    def to_prediction(self, raw: ModelRawOutput, observation: ModelObservation) -> ModelPrediction:
        """Validate and project the raw output into fixed-odom poses."""


_ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(*names: str) -> Callable[[type], type]:
    def _decorator(cls: type) -> type:
        for name in names:
            _ADAPTER_REGISTRY[str(name).strip().lower()] = cls
        return cls

    return _decorator


@register_adapter('lisn_qwen3vl_fp8')
class LisnQwenAdapterStub:
    """Typed-blocker adapter for the Qwen/LISN B identity.

    The Qwen/LISN producer sidecar currently emits pixel/bearing/STOP and has
    no bridge to the common metric FollowPath action.  This adapter must never
    fabricate a Path from a pixel; it fails closed with a typed capability
    mismatch until that bridge exists.
    """

    name = 'lisn_qwen3vl_fp8'
    capability = ModelCapability(
        name='lisn_qwen3vl_fp8',
        raw_output_kind='pixel_or_bearing',
        required_inputs=('rgb_image',),
        output_modes=('stop',),
        notes='qwen_lisn_common_metric_action_bridge_missing',
    )

    def __init__(self, *, params=None, logger=None, transport=None, **kwargs):
        self.params = params or {}
        self.logger = logger
        self.transport = transport

    def reset(self, *, session_id: str = '', episode: int = -1) -> None:
        return None

    def infer(self, observation: ModelObservation) -> ModelRawOutput:
        raise CapabilityMismatchError(
            'qwen_lisn_common_metric_action_bridge_missing: '
            'the Qwen/LISN sidecar pixel/bearing output is not a common metric Path'
        )

    def to_prediction(self, raw: ModelRawOutput,
                      observation: ModelObservation) -> ModelPrediction:
        raise CapabilityMismatchError(
            'qwen_lisn_common_metric_action_bridge_missing: '
            'no metric Path can be fabricated from a pixel/bearing'
        )


def create_adapter(
    name: str,
    *,
    params: Optional[dict[str, Any]] = None,
    logger: Any = None,
    transport: Any = None,
) -> ModelAdapter:
    key = str(name or '').strip().lower()
    if not key:
        raise UnknownModelAdapterError('model adapter name is empty')
    _ensure_builtin_adapters()
    adapter_cls = _ADAPTER_REGISTRY.get(key)
    if adapter_cls is None:
        raise UnknownModelAdapterError(
            f"unknown model adapter '{name}'; supported adapters: {sorted(_ADAPTER_REGISTRY)}"
        )
    kwargs = {'params': params or {}, 'logger': logger}
    if transport is not None:
        kwargs['transport'] = transport
    return adapter_cls(**kwargs)


def available_adapters() -> list[str]:
    _ensure_builtin_adapters()
    return sorted(_ADAPTER_REGISTRY)


_BUILTINS_LOADED = False


def _ensure_builtin_adapters() -> None:
    """Import the built-in adapter modules once, without caller-side imports.

    Lazy on purpose: adapter modules import this module for ``register_adapter``,
    so importing them at module load would be circular.
    """

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    try:
        from .adapters import internnav as _internnav  # noqa: F401
        from .adapters import lightnav as _lightnav  # noqa: F401
        from .adapters import navila as _navila  # noqa: F401
    except Exception:
        _BUILTINS_LOADED = False
        raise
    _BUILTINS_LOADED = True


def wrap_angle(angle: float) -> float:
    value = float(angle)
    if not math.isfinite(value):
        raise ModelProtocolError(f'yaw is not finite: {angle!r}')
    return math.atan2(math.sin(value), math.cos(value))


def _finite(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ModelProtocolError(f'{field_name} must be a finite number, not a boolean')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelProtocolError(f'{field_name} is not numeric: {value!r}') from exc
    if not math.isfinite(number):
        raise ModelProtocolError(f'{field_name} is not finite: {value!r}')
    return number


def validate_observation(observation: ModelObservation, capability: ModelCapability) -> None:
    """Reject missing frame/stamp/pose/inputs before any model request."""

    frame = str(getattr(observation, 'frame_id', '') or '').strip()
    if not frame:
        raise CapabilityMismatchError(f'{capability.name}: observation.frame_id is empty')
    stamp = getattr(observation, 'stamp_ns', 0)
    if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
        raise CapabilityMismatchError(f'{capability.name}: observation.stamp_ns must be a positive integer')
    pose = getattr(observation, 'pose', None)
    if pose is None:
        raise CapabilityMismatchError(f'{capability.name}: observation.pose is missing')
    for attr in ('x', 'y', 'yaw'):
        _finite(getattr(pose, attr, None), field_name=f'pose.{attr}')
    required = set(capability.required_inputs)
    if 'rgb' in required and getattr(observation, 'rgb_image', None) is None:
        raise CapabilityMismatchError(f'{capability.name}: rgb_image is required')
    if 'depth' in required and getattr(observation, 'depth_image', None) is None:
        raise CapabilityMismatchError(f'{capability.name}: depth_image is required')
    if 'instruction' in required and not str(getattr(observation, 'instruction', '') or '').strip():
        raise CapabilityMismatchError(f'{capability.name}: instruction is required')
    if 'camera_intrinsics' in required and getattr(observation, 'camera_intrinsics', None) is None:
        raise CapabilityMismatchError(f'{capability.name}: camera_intrinsics is required')


def raw_response_hash(raw: Any) -> str:
    """Stable hash for a raw model response, used in traces/manifests."""

    try:
        payload = json.dumps(raw, sort_keys=True, default=str, separators=(',', ':')).encode('utf-8')
    except Exception:
        payload = repr(raw).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _bounded_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or '')
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()[:16]
    return f'<text len={len(text)} sha256={digest}>'


def bounded_debug(value: Any, *, depth: int = 0, limit: int = 32) -> Any:
    """Bound diagnostics so manifests never contain images/base64/raw responses."""

    if depth > 4:
        return '<max-depth>'
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return f'<bytes len={len(value)}>'
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, dict):
        return {
            str(key): bounded_debug(item, depth=depth + 1, limit=limit)
            for key, item in list(value.items())[:limit]
        }
    if isinstance(value, (list, tuple)):
        return [bounded_debug(item, depth=depth + 1, limit=limit) for item in list(value)[:limit]]
    shape = getattr(value, 'shape', None)
    if shape is not None:
        return {'type': type(value).__name__, 'shape': [int(dim) for dim in shape]}
    if hasattr(value, 'tolist'):
        try:
            return bounded_debug(value.tolist(), depth=depth + 1, limit=limit)
        except Exception:
            pass
    return _bounded_text(value)


def coerce_local_waypoints(
    points: Any,
    *,
    source: str = 'native',
    max_segment_m: float = MAX_SEGMENT_M,
) -> list[tuple[float, float, float]]:
    """Validate cumulative local ``(x, y[, yaw])`` waypoints.

    Two-column points get a heading derived from the segment direction.  The
    first point is interpreted relative to the robot at the observation pose.
    Adjacent identical poses are removed: InternNav trajectories normally start
    at the origin and may repeat a pose.  A repeated position with a different
    yaw is a pure rotation and is retained.
    """

    if points is None:
        raise EmptyModelOutputError('model output contains no waypoints')
    try:
        rows = list(points)
    except TypeError as exc:
        raise ModelProtocolError(f'waypoints must be a sequence, got {type(points).__name__}') from exc
    if not rows:
        raise EmptyModelOutputError('model output contains an empty waypoint list')
    result: list[tuple[float, float, float]] = []
    for index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not hasattr(row, '__iter__'):
            raise ModelProtocolError(f'waypoint {index} is not a sequence: {row!r}')
        values = list(row)
        if len(values) < 2:
            raise ModelProtocolError(f'waypoint {index} must have at least two coordinates')
        x = _finite(values[0], field_name=f'waypoint[{index}].x')
        y = _finite(values[1], field_name=f'waypoint[{index}].y')
        if len(values) >= 3:
            yaw = wrap_angle(_finite(values[2], field_name=f'waypoint[{index}].yaw'))
        elif index == 0:
            yaw = math.atan2(y, x) if math.hypot(x, y) > 1e-9 else 0.0
        else:
            prev_x, prev_y, _ = result[-1] if result else (0.0, 0.0, 0.0)
            dx, dy = x - prev_x, y - prev_y
            yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-9 else (result[-1][2] if result else 0.0)
        candidate = (x, y, yaw)
        if result and _same_pose(result[-1], candidate):
            continue
        if result:
            prev_x, prev_y, _ = result[-1]
            step = math.hypot(x - prev_x, y - prev_y)
            if step > float(max_segment_m):
                raise ModelProtocolError(
                    f'waypoint {index} is discontinuous: {step:.3f} m exceeds {max_segment_m:.3f} m'
                )
        result.append(candidate)
    if not result:
        raise EmptyModelOutputError('model output contains no distinct waypoint')
    return result


def _same_pose(a: tuple[float, float, float], b: tuple[float, float, float], *, eps: float = 1e-9) -> bool:
    return (
        abs(a[0] - b[0]) <= eps
        and abs(a[1] - b[1]) <= eps
        and abs(wrap_angle(a[2] - b[2])) <= eps
    )


def _has_executable_geometry(local: list[tuple[float, float, float]]) -> bool:
    """True when a non-STOP local path has translation or rotation.

    A single forward primitive at ``(x, 0, 0)`` is executable; a single
    ``(0, 0, 0)`` origin is not.  Pure rotations are executable.
    """

    if not local:
        return False
    for x, y, yaw in local:
        if math.hypot(x, y) > 1e-9:
            return True
        if abs(wrap_angle(yaw)) > 1e-9:
            return True
    for previous, current in zip(local, local[1:]):
        if math.hypot(current[0] - previous[0], current[1] - previous[1]) > 1e-9:
            return True
        if abs(wrap_angle(current[2] - previous[2])) > 1e-9:
            return True
    return False


def _sample_segment(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    step_m: float,
) -> list[tuple[float, float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        # Pure rotation: one pose with the terminal orientation.
        return [end]
    count = max(1, int(math.ceil(distance / max(step_m, 1e-6))))
    angle = wrap_angle(end[2] - start[2])
    samples: list[tuple[float, float, float]] = []
    for index in range(1, count):
        fraction = index / count
        samples.append((
            start[0] + fraction * dx,
            start[1] + fraction * dy,
            wrap_angle(start[2] + fraction * angle),
        ))
    samples.append(end)
    return samples


def local_to_odom(
    local_waypoints: Any,
    observation: ModelObservation,
    *,
    sample_step_m: float = SAMPLE_STEP_M,
    max_segment_m: float = MAX_SEGMENT_M,
    max_points: int = MAX_PATH_POINTS,
) -> list[tuple[float, float, float]]:
    """Rigidly project cumulative ego waypoints into the observation frame.

    ``observation.pose`` is the robot pose in the fixed odom frame; the
    observation stamp is already validated by :func:`validate_observation`.
    """

    pose = observation.pose
    origin = (float(pose.x), float(pose.y), wrap_angle(float(pose.yaw)))
    local = coerce_local_waypoints(local_waypoints, max_segment_m=max_segment_m)
    cos_yaw = math.cos(origin[2])
    sin_yaw = math.sin(origin[2])
    projected: list[tuple[float, float, float]] = []
    for x, y, yaw in local:
        projected.append((
            origin[0] + cos_yaw * x - sin_yaw * y,
            origin[1] + sin_yaw * x + cos_yaw * y,
            wrap_angle(origin[2] + yaw),
        ))
    if not projected:
        return []
    if all(math.hypot(x - origin[0], y - origin[1]) <= 1e-12 for x, y, _ in projected):
        return [projected[-1]]
    result = [origin]
    for endpoint in projected:
        result.extend(_sample_segment(result[-1], endpoint, step_m=sample_step_m))
        if len(result) > int(max_points):
            raise ModelProtocolError(f'path exceeds {max_points} sampled points')
    return result


def validate_odom_path(
    poses: Any,
    observation: ModelObservation,
    *,
    max_segment_m: float = MAX_SEGMENT_M,
    max_points: int = MAX_PATH_POINTS,
) -> list[tuple[float, float, float]]:
    """Validate a fixed-odom path before it is sent to FollowPath."""

    if poses is None:
        raise EmptyModelOutputError('prediction contains no fixed-odom poses')
    rows = list(poses)
    if not rows:
        raise EmptyModelOutputError('prediction path is empty')
    if len(rows) > int(max_points):
        raise ModelProtocolError(f'path has {len(rows)} poses, limit is {max_points}')
    validated: list[tuple[float, float, float]] = []
    for index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not hasattr(row, '__iter__'):
            raise ModelProtocolError(f'odom pose {index} is not a sequence')
        values = list(row)
        if len(values) < 2:
            raise ModelProtocolError(f'odom pose {index} must have at least two coordinates')
        x = _finite(values[0], field_name=f'odom_pose[{index}].x')
        y = _finite(values[1], field_name=f'odom_pose[{index}].y')
        yaw = wrap_angle(_finite(values[2], field_name=f'odom_pose[{index}].yaw')) if len(values) >= 3 else 0.0
        candidate = (x, y, yaw)
        if validated and _same_pose(validated[-1], candidate):
            continue
        if validated:
            prev_x, prev_y, _ = validated[-1]
            step = math.hypot(x - prev_x, y - prev_y)
            if step > float(max_segment_m):
                raise ModelProtocolError(
                    f'odom pose {index} is discontinuous: {step:.3f} m exceeds {max_segment_m:.3f} m'
                )
        validated.append(candidate)
    if not validated:
        raise EmptyModelOutputError('prediction has no distinct odom pose')
    # The first pose must be the current robot pose; otherwise RPP would cut a
    # straight line from an unrelated point.
    pose = observation.pose
    start_distance = math.hypot(validated[0][0] - float(pose.x), validated[0][1] - float(pose.y))
    if start_distance > max(0.25, float(max_segment_m)):
        raise ModelProtocolError(
            f'path starts {start_distance:.3f} m away from the observation pose'
        )
    return validated


def prediction_from_raw(
    raw: ModelRawOutput,
    observation: ModelObservation,
    *,
    adapter_name: str = '',
) -> ModelPrediction:
    """Shared raw -> prediction conversion for the three adapters."""

    if not raw.raw_hash:
        raw.raw_hash = raw_response_hash(raw.raw)
    if raw.stop:
        return ModelPrediction(
            adapter=adapter_name or raw.adapter,
            poses=[],
            stop=True,
            status='stop',
            source=raw.source,
            raw=raw.raw,
            debug={
                'modality': raw.modality,
                'raw_hash': raw.raw_hash,
                'raw_output': raw.to_manifest(),
                **bounded_debug(raw.debug),
            },
        )
    if raw.status == 'timeout':
        raise ModelTimeoutError(raw.debug.get('reason') or 'model request timed out')
    if raw.status == 'error':
        raise ModelProtocolError(raw.debug.get('reason') or 'model request failed')
    if not _has_executable_geometry(raw.local_waypoints):
        raise EmptyModelOutputError('non-STOP prediction has no executable geometry')
    poses = local_to_odom(raw.local_waypoints, observation)
    poses = validate_odom_path(poses, observation)
    return ModelPrediction(
        adapter=adapter_name or raw.adapter,
        poses=poses,
        stop=False,
        status=raw.status or 'running',
        source=raw.source,
        raw=raw.raw,
        debug={
            'modality': raw.modality,
            'raw_hash': raw.raw_hash,
            'raw_output': raw.to_manifest(),
            **bounded_debug(raw.debug),
        },
    )


def pose_distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))
