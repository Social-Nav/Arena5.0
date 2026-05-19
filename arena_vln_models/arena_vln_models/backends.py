from __future__ import annotations

import importlib
import inspect
import math
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


INTERNNAV_REFERENCE = (
    'InternRobotics/InternNav@7a5c62400ac45b313d9b709c740b64191556a242 '
    '(checkpoint candidate: InternRobotics/InternVLA-N1-DualVLN)'
)

INTERNNAV_CHECKPOINT_DOWNLOAD = (
    'git clone https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN '
    '<model_dir> && git -C <model_dir> lfs pull'
)

INTERNNAV_NATIVE_IO_CONTRACT = (
    'InternNav reference agent.step(rgb, depth, camera_pose, instruction, intrinsic=..., '
    'look_down=...) -> trajectory/pixel goal or action sequence; the Arena adapter keeps '
    'a minimal ROS-safe contract by mapping observation=(pose, goal, instruction, optional '
    'rgb/depth) to cmd_vel'
)

TORCHSCRIPT_IO_CONTRACT = (
    'forward(obs: Tensor[1,N]) -> Tensor[1,2] where obs=[dx, dy, dist, yaw_err, instr_len] '
    'and output=[linear_x, angular_z]'
)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _shape_of(value: Any) -> Optional[list[int]]:
    shape = getattr(value, 'shape', None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except Exception:
        return None


def _sanitize_debug(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, 'tolist'):
        return _to_jsonable(value.tolist())
    shape = _shape_of(value)
    if shape is not None:
        return {'type': type(value).__name__, 'shape': shape}
    return str(value)


def _trajectory_control_step(trajectory: Any) -> Optional[tuple[float, float, float]]:
    """Return the trajectory waypoint used for continuous velocity control.

    InternNav's real-world continuous controller derives ``(v, w)`` from the
    final predicted subgoal rather than turning the trajectory into a discrete
    action sequence.  Use the last available waypoint to keep Arena eval close
    to that continuous-action interface; when the trajectory contains only x/y,
    fall back to the waypoint bearing as the angular component.
    """
    try:
        if isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
            if not trajectory:
                return None
            selected = trajectory[-1]
            if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
                values = [float(item) for item in selected[:3]]
            else:
                values = [float(item) for item in trajectory[:3]]
        else:
            values = [float(item) for item in trajectory[:3]]
    except Exception:
        return None

    if len(values) < 2:
        return None
    x = values[0]
    y = values[1]
    yaw = values[2] if len(values) >= 3 else math.atan2(y, x)
    return x, y, yaw


def _trajectory_first_step(trajectory: Any) -> Optional[tuple[float, float, float]]:
    """Backward-compatible debug helper for the first trajectory waypoint."""
    try:
        if isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
            if not trajectory:
                return None
            selected = trajectory[0]
            if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
                values = [float(item) for item in selected[:3]]
            else:
                values = [float(item) for item in trajectory[:3]]
        else:
            values = [float(item) for item in trajectory[:3]]
    except Exception:
        return None

    if len(values) < 2:
        return None
    x = values[0]
    y = values[1]
    yaw = values[2] if len(values) >= 3 else math.atan2(y, x)
    return x, y, yaw


def _model_output_policy(params: Mapping[str, Any]) -> str:
    """Resolve how adapter outputs should be converted to cmd_vel.

    ``trajectory`` is the default because InternNav's real-world continuous
    controller consumes S1's generated trajectory directly.  ``discrete`` keeps
    a diagnostic path for action-id evaluation, and ``raw`` preserves the legacy
    adapter precedence where discrete actions won when both fields were present.
    """
    raw = str(params.get('model_output_policy', params.get('output_policy', 'trajectory'))).strip().lower()
    aliases = {
        'continuous': 'trajectory',
        'continuous_trajectory': 'trajectory',
        'output_trajectory': 'trajectory',
        'traj': 'trajectory',
        'action': 'discrete',
        'actions': 'discrete',
        'discrete_action': 'discrete',
        'legacy': 'raw',
        'auto': 'trajectory',
    }
    resolved = aliases.get(raw, raw)
    if resolved not in {'trajectory', 'discrete', 'raw'}:
        return 'trajectory'
    return resolved


def _action_to_command(action: Any, params: dict[str, Any]) -> Optional[tuple[float, float, str, dict[str, Any]]]:
    selected = action
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
        if not action:
            return None
        selected = action[0]

    try:
        selected_int = int(selected)
    except (TypeError, ValueError):
        return None

    max_linear = float(params['max_linear'])
    max_angular = float(params['max_angular'])
    invert_turns = bool(params.get('invert_discrete_turns', False))
    left_sign = -1.0 if invert_turns else 1.0
    right_sign = 1.0 if invert_turns else -1.0
    native_label = {0: 'stop', 1: 'forward', 2: 'turn_left', 3: 'turn_right', 5: 'look_down'}.get(
        selected_int,
        'unsupported',
    )
    effective_label = native_label
    if selected_int == 2 and invert_turns:
        effective_label = 'turn_right'
    elif selected_int == 3 and invert_turns:
        effective_label = 'turn_left'
    debug = {
        'selected_action': selected_int,
        'action_label': native_label,
        'native_action_label': native_label,
        'effective_action_label': effective_label,
        'invert_discrete_turns': invert_turns,
    }
    if selected_int == 0:
        return 0.0, 0.0, 'discrete_stop', debug
    if selected_int == 1:
        return max(max_linear * 0.6, 0.05), 0.0, 'discrete_forward', debug
    if selected_int == 2:
        # In Arena's real robot/Isaac direct bridge, InternNav frequently emits
        # repeated turn actions before it ever produces a forward token.  If
        # turns execute as pure in-place rotation, eval videos can appear
        # stationary even though the policy is actively steering.  Preserve the
        # historical gentle forward arc used by the successful eval runs so the
        # robot keeps making visible progress while turning.
        debug['arc_turn'] = True
        status = 'discrete_turn_right' if invert_turns else 'discrete_turn_left'
        return max(max_linear * 0.6, 0.12), left_sign * max_angular * 0.25, status, debug
    if selected_int == 3:
        debug['arc_turn'] = True
        status = 'discrete_turn_left' if invert_turns else 'discrete_turn_right'
        return max(max_linear * 0.6, 0.12), right_sign * max_angular * 0.25, status, debug
    if selected_int == 5:
        debug['look_down_requested'] = True
        return 0.0, 0.0, 'look_down_requested', debug
    debug['unsupported_action'] = selected_int
    return 0.0, 0.0, 'unsupported_discrete_action', debug


def _goal_debug(observation: ModelSimObservation) -> dict[str, Any]:
    pose = observation.pose
    goal = observation.goal
    debug: dict[str, Any] = {
        'instruction_length': len(observation.instruction),
        'instruction_preview': observation.instruction[:160],
        'look_down': bool(observation.look_down),
        'camera_frame_id': observation.camera_frame_id,
        'camera_intrinsics_available': observation.camera_intrinsics is not None,
    }
    rgb_shape = _shape_of(observation.rgb_image)
    depth_shape = _shape_of(observation.depth_image)
    if rgb_shape is not None:
        debug['rgb_shape'] = rgb_shape
    if depth_shape is not None:
        debug['depth_shape'] = depth_shape
    if observation.metadata:
        for key in ('rgb_available', 'depth_available', 'camera_info_available', 'namespace'):
            if key in observation.metadata:
                debug[key] = observation.metadata[key]
    if pose is None or goal is None:
        return debug

    dx = goal.x - pose.x
    dy = goal.y - pose.y
    dist = math.hypot(dx, dy)
    target_yaw = math.atan2(dy, dx)
    yaw_err = math.atan2(math.sin(target_yaw - pose.yaw), math.cos(target_yaw - pose.yaw))
    debug.update({
        'goal_distance': dist,
        'yaw_error': yaw_err,
        'goal_pose': [goal.x, goal.y, goal.yaw],
    })
    return debug


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class ModelSimObservation:
    pose: Optional[Pose2D]
    goal: Optional[Pose2D]
    instruction: str
    rgb_image: Any = None
    depth_image: Any = None
    subgoal: Optional[Pose2D] = None
    camera_intrinsics: Optional[tuple[float, ...]] = None
    camera_frame_id: str = ''
    look_down: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rgb(self) -> Any:
        return self.rgb_image

    @property
    def depth(self) -> Any:
        return self.depth_image

    @property
    def goal_pose(self) -> Optional[Pose2D]:
        return self.goal

    @property
    def subgoal_pose(self) -> Optional[Pose2D]:
        return self.subgoal


@dataclass
class ModelSimDecision:
    linear_x: float = 0.0
    angular_z: float = 0.0
    status: str = 'safe_stop'
    degraded: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


class ModelSimBackend(ABC):
    backend_type = 'base'
    model_source = 'internal'
    model_download = 'N/A'
    io_contract = 'N/A'
    uses_model_inference = False

    def __init__(self, logger, params: dict[str, Any]) -> None:
        self._logger = logger
        self._params = params

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        log_fn = getattr(self._logger, level, None)
        if callable(log_fn):
            try:
                log_fn(message)
                return
            except Exception:
                pass
        try:
            sys.stderr.write(f'[arena_vln_models:{level}] {message}\n')
            sys.stderr.flush()
        except Exception:
            pass

    def describe(self) -> str:
        return (
            f'backend={self.backend_type} source={self.model_source} '
            f'download="{self.model_download}" '
            f'io_contract="{self.io_contract}"'
        )

    @abstractmethod
    def compute(self, observation: ModelSimObservation) -> ModelSimDecision:
        raise NotImplementedError


_BACKEND_REGISTRY: dict[str, type[ModelSimBackend]] = {}


def register_backend(*modes: str) -> Callable[[type[ModelSimBackend]], type[ModelSimBackend]]:
    def _decorator(cls: type[ModelSimBackend]) -> type[ModelSimBackend]:
        for mode in modes:
            _BACKEND_REGISTRY[mode.strip().lower()] = cls
        return cls

    return _decorator


def _finalize_decision(decision: ModelSimDecision, params: dict[str, Any]) -> ModelSimDecision:
    decision.status = str(decision.status or 'safe_stop')
    decision.debug = _sanitize_debug(decision.debug)
    if not _is_finite_number(decision.linear_x) or not _is_finite_number(decision.angular_z):
        raise ValueError('non-finite command values')
    decision.linear_x = clamp(
        float(decision.linear_x),
        -float(params['max_linear']),
        float(params['max_linear']),
    )
    decision.angular_z = clamp(
        float(decision.angular_z),
        -float(params['max_angular']),
        float(params['max_angular']),
    )
    return decision


def _safe_stop_decision(
    observation: ModelSimObservation,
    *,
    status: str,
    reason: str,
    debug: Optional[Mapping[str, Any]] = None,
) -> ModelSimDecision:
    merged_debug = _goal_debug(observation)
    merged_debug.update(_sanitize_debug(debug or {}))
    merged_debug['failure_reason'] = reason
    merged_debug['safe_stop'] = True
    return ModelSimDecision(status=status, degraded=True, debug=merged_debug)


def _load_adapter_target(target: str) -> Any:
    normalized = target.strip()
    if not normalized:
        raise ValueError('adapter_target is empty')

    module_name: str
    attr_name: str
    if ':' in normalized:
        module_name, attr_name = normalized.split(':', 1)
    else:
        module_name, _, attr_name = normalized.rpartition('.')
        if not module_name:
            raise ValueError(
                f"adapter_target='{target}' must use 'module:attr' or 'module.attr' format"
            )

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _call_adapter_factory(target: Callable[..., Any], logger, params: dict[str, Any]) -> Any:
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((), {'logger': logger, 'params': params}),
        ((), {'params': params}),
        ((), {'logger': logger}),
        ((), {}),
        ((logger, params), {}),
        ((params,), {}),
        ((logger,), {}),
    ]
    last_error: Optional[TypeError] = None
    for args, kwargs in attempts:
        try:
            return target(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return target()


def _looks_like_adapter_factory(target: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return False

    parameters = list(signature.parameters.values())
    if not parameters:
        return True

    recognized_names = {'logger', 'params'}
    for parameter in parameters:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return True
        if parameter.name not in recognized_names:
            return False
    return True


def _resolve_adapter_callable(target: Any, logger, params: dict[str, Any], adapter_target: str) -> Callable[[ModelSimObservation], Any]:
    candidate = target
    if inspect.isclass(candidate):
        candidate = _call_adapter_factory(candidate, logger, params)
    elif callable(candidate) and not hasattr(candidate, 'compute') and not hasattr(candidate, 'predict'):
        if _looks_like_adapter_factory(candidate):
            initialized = _call_adapter_factory(candidate, logger, params)
            if initialized is not candidate:
                candidate = initialized

    if hasattr(candidate, 'compute') and callable(candidate.compute):
        return candidate.compute
    if hasattr(candidate, 'predict') and callable(candidate.predict):
        return candidate.predict
    if callable(candidate):
        return candidate
    raise TypeError(
        f"adapter_target='{adapter_target}' did not resolve to a callable adapter or object exposing compute()/predict()"
    )


@register_backend('heuristic')
class HeuristicBackend(ModelSimBackend):
    backend_type = 'heuristic'
    model_source = 'arena_builtin_safe_controller'
    model_download = 'built-in (no external weights)'
    io_contract = 'pose + goal -> cmd_vel'

    def compute(self, observation: ModelSimObservation) -> ModelSimDecision:
        pose = observation.pose
        # Prefer the Nav2 path lookahead while the final episode goal is still
        # far away.  The final goal can be across walls or around corners in
        # hospital maps; steering directly toward it makes the heuristic rotate
        # in place against an unreachable bearing.  Switch back to the fixed
        # final goal near the destination so completion remains stable and does
        # not chase a moving lookahead at the goal boundary.
        final_goal = observation.goal
        path_goal = observation.subgoal
        goal = final_goal or path_goal
        target_source = 'goal' if final_goal is not None else 'subgoal'
        final_goal_distance = None
        if pose is not None and final_goal is not None:
            final_goal_distance = math.hypot(final_goal.x - pose.x, final_goal.y - pose.y)
        if pose is not None and final_goal is not None and path_goal is not None:
            final_goal_near_radius = max(float(self._params['goal_tolerance']) * 3.0, 1.0)
            path_goal_distance = math.hypot(path_goal.x - pose.x, path_goal.y - pose.y)
            if final_goal_distance > final_goal_near_radius and path_goal_distance > 1.0:
                goal = path_goal
                target_source = 'subgoal'
        if pose is None or goal is None:
            return ModelSimDecision(status='missing_pose_or_goal', degraded=True)

        dx = goal.x - pose.x
        dy = goal.y - pose.y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(target_yaw - pose.yaw), math.cos(target_yaw - pose.yaw))
        goal_yaw_err = math.atan2(math.sin(goal.yaw - pose.yaw), math.cos(goal.yaw - pose.yaw))
        max_lin = float(self._params['max_linear'])
        max_ang = float(self._params['max_angular'])
        # The direct InternNav bridge republishes the most recent command between
        # service responses.  Holding a 1.0-1.5 rad/s pure-rotation command is
        # unstable in Isaac at low effective update rates: the robot repeatedly
        # overshoots the target bearing and appears to spin in place.  Keep the
        # far-goal heuristic controller deliberately underdamped and arc-based so
        # every command either makes translational progress or rotates slowly
        # enough to be corrected by the next odom sample.
        far_goal_max_ang = min(max_ang, 0.6)
        pure_rotate_max_ang = min(max_ang, 0.25)
        arc_yaw_limit = 2.4
        k_lin = float(self._params['k_lin'])
        k_ang = float(self._params['k_ang'])
        goal_tolerance = float(self._params['goal_tolerance'])
        angle_tol = float(self._params['angle_tolerance'])
        min_lin = float(self._params['min_lin_when_aligned'])

        if dist <= goal_tolerance:
            if abs(goal_yaw_err) <= angle_tol:
                return ModelSimDecision(
                    status='goal_reached',
                    debug={
                        'goal_distance': dist,
                        'yaw_error': goal_yaw_err,
                        'target_source': target_source,
                    },
                )

            return ModelSimDecision(
                linear_x=0.0,
                angular_z=clamp(k_ang * goal_yaw_err, -max_ang, max_ang),
                status='align_goal_heading',
                debug={
                    'goal_distance': dist,
                    'yaw_error': goal_yaw_err,
                    'instruction_length': len(observation.instruction),
                    'target_source': target_source,
                },
            )

        angular_z = clamp(k_ang * yaw_err, -pure_rotate_max_ang, pure_rotate_max_ang)
        status = 'rotate_to_goal'
        # Keep fallback control convergent while the slow CPU InternNav adapter
        # is still running.  Pure path-lookahead arcs can orbit near the goal; a
        # fixed final-goal fallback should first align decisively, then drive.
        abs_yaw_err = abs(yaw_err)
        if abs(yaw_err) <= angle_tol:
            linear_x = clamp(max(min_lin, k_lin * dist), 0.0, max_lin)
            angular_z = clamp(k_ang * yaw_err, -far_goal_max_ang, far_goal_max_ang)
            status = 'drive_to_goal'
            arc_turn = False
        elif abs_yaw_err <= arc_yaw_limit:
            yaw_scale = clamp(1.0 - (abs_yaw_err / arc_yaw_limit), 0.0, 1.0)
            linear_x = clamp(max(min_lin, max_lin * (0.15 + 0.55 * yaw_scale)), 0.0, max_lin)
            angular_z = clamp(k_ang * yaw_err, -far_goal_max_ang, far_goal_max_ang)
            status = 'arc_to_goal'
            arc_turn = True
        else:
            linear_x = 0.0
            # Isaac Ai2_Bot2 is prone to lateral drift and bearing overshoot
            # while holding pure angular commands.  Keep pure rotations slower
            # than arc turns so a large initial heading error can settle into
            # the arc/drive bands instead of orbiting around the start pose.
            angular_z = clamp(k_ang * yaw_err, -pure_rotate_max_ang, pure_rotate_max_ang)
            arc_turn = False

        return ModelSimDecision(
            linear_x=linear_x,
            angular_z=angular_z,
            status=status,
            debug={
                'goal_distance': dist,
                'yaw_error': yaw_err,
                'instruction_length': len(observation.instruction),
                'target_source': target_source,
                'arc_turn': arc_turn,
                'effective_max_angular': far_goal_max_ang,
                'pure_rotate_max_angular': pure_rotate_max_ang,
                'arc_yaw_limit': arc_yaw_limit,
                'final_goal_distance': final_goal_distance,
            },
        )


@register_backend('torchscript', 'model')
class TorchScriptBackend(ModelSimBackend):
    backend_type = 'torchscript'
    model_source = INTERNNAV_REFERENCE
    model_download = INTERNNAV_CHECKPOINT_DOWNLOAD
    io_contract = f'{INTERNNAV_NATIVE_IO_CONTRACT}; fallback_torchscript_contract={TORCHSCRIPT_IO_CONTRACT}'
    uses_model_inference = True

    def __init__(self, logger, params: dict[str, Any]) -> None:
        super().__init__(logger, params)
        self._torch = None
        self._model = None

        model_path = str(params['model_path'])
        device = str(params['device'])
        if not model_path:
            self._log(
                'warn',
                "backend='torchscript' but model_path is empty; commands will fall back to safe-stop"
            )
            return

        try:
            import torch  # type: ignore

            self._torch = torch
        except Exception as exc:
            self._log(
                'error',
                f"Failed to import torch for torchscript backend (model_path='{model_path}'): {exc}"
            )
            return

        try:
            torch = self._torch
            assert torch is not None
            self._model = torch.jit.load(model_path, map_location=device)
            self._model.eval()
            self._log('info', f"Loaded torchscript model from '{model_path}' (device={device})")
        except Exception as exc:
            self._log(
                'error',
                f"Failed to load torchscript model from '{model_path}' (device={device}): {exc}"
            )
            self._model = None

    def compute(self, observation: ModelSimObservation) -> ModelSimDecision:
        if observation.pose is None or observation.goal is None:
            return ModelSimDecision(status='missing_pose_or_goal', degraded=True)
        if self._model is None or self._torch is None:
            return ModelSimDecision(status='model_unavailable', degraded=True)

        pose = observation.pose
        goal = observation.goal
        dx = goal.x - pose.x
        dy = goal.y - pose.y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(target_yaw - pose.yaw), math.cos(target_yaw - pose.yaw))
        instr_len = float(len(observation.instruction))

        torch = self._torch
        assert torch is not None
        obs = torch.tensor([[dx, dy, dist, yaw_err, instr_len]], dtype=torch.float32)

        start = time.monotonic()
        with torch.no_grad():
            output = self._model(obs)
        elapsed = time.monotonic() - start
        timeout_sec = float(self._params['inference_timeout_sec'])
        if elapsed > timeout_sec:
            debug = _goal_debug(observation)
            debug['infer_time_sec'] = elapsed
            return ModelSimDecision(
                status='inference_timeout',
                degraded=True,
                debug=debug,
            )

        try:
            values = output.squeeze().detach().cpu().tolist()
            linear_x = float(values[0])
            angular_z = float(values[1])
        except Exception as exc:
            self._log('error', f'Failed to parse torchscript output: {exc}')
            return ModelSimDecision(status='invalid_model_output', degraded=True)

        return _finalize_decision(ModelSimDecision(
            linear_x=clamp(linear_x, -float(self._params['max_linear']), float(self._params['max_linear'])),
            angular_z=clamp(angular_z, -float(self._params['max_angular']), float(self._params['max_angular'])),
            status='model_command',
            debug={
                'goal_distance': dist,
                'yaw_error': yaw_err,
                'instruction_length': instr_len,
                'infer_time_sec': elapsed,
            },
        ), self._params)


@register_backend('adapter', 'python', 'python_adapter', 'internnav')
class PythonAdapterBackend(ModelSimBackend):
    backend_type = 'python_adapter'
    model_source = 'external_python_adapter'
    model_download = 'provided by adapter_target'
    io_contract = (
        'adapter_target resolves to a class / callable / object method that accepts '
        'ModelSimObservation and returns ModelSimDecision or a mapping using either '
        '{linear_x, angular_z, status?, degraded?, debug?} or '
        '{discrete_action|action|output_action, output_pixel?, output_trajectory?, debug?}'
    )
    uses_model_inference = True

    def __init__(self, logger, params: dict[str, Any]) -> None:
        super().__init__(logger, params)
        self._adapter_callable: Optional[Callable[[ModelSimObservation], Any]] = None
        self._adapter_target = str(params.get('adapter_target', '')).strip()
        self._require_real_backend = bool(params.get('require_real_backend', False))
        if not self._adapter_target:
            if self._require_real_backend:
                raise RuntimeError("backend='python_adapter' requires adapter_target when require_real_backend=true")
            self._log(
                'warn',
                "backend='python_adapter' but adapter_target is empty; commands will fall back to safe-stop"
            )
            return

        try:
            target = _load_adapter_target(self._adapter_target)
            self._adapter_callable = _resolve_adapter_callable(target, logger, params, self._adapter_target)

            self.model_source = self._adapter_target
            self._log('info', f"Loaded python adapter backend target '{self._adapter_target}'")
        except Exception as exc:
            self._log('error', f"Failed to load adapter_target='{self._adapter_target}': {exc}")
            self._adapter_callable = None
            if self._require_real_backend:
                raise RuntimeError(
                    f"Failed to load required adapter_target='{self._adapter_target}': {exc}"
                ) from exc

    def _coerce_output(self, output: Any) -> ModelSimDecision:
        if isinstance(output, ModelSimDecision):
            output.debug.setdefault('raw_model_output_type', type(output).__name__)
            return output
        if isinstance(output, Mapping):
            debug = _sanitize_debug(output.get('debug', {}))
            debug.setdefault('raw_model_output', _to_jsonable(output))
            debug.setdefault('raw_model_output_type', type(output).__name__)
            target_pixel = output.get('target_pixel')
            if target_pixel is None:
                target_pixel = output.get('output_pixel', output.get('pixel_goal'))
            if target_pixel is not None:
                debug.setdefault('target_pixel', target_pixel)

            output_policy = _model_output_policy(self._params)
            debug.setdefault('model_output_policy', output_policy)

            trajectory = output.get('output_trajectory', output.get('trajectory'))
            if trajectory is not None:
                debug.setdefault('trajectory_preview', trajectory)

            discrete_action = output.get('discrete_action', output.get('output_action', output.get('action')))

            def _trajectory_decision() -> Optional[ModelSimDecision]:
                if trajectory is None:
                    return None
                control_step = _trajectory_control_step(trajectory)
                if control_step is None:
                    return None
                x, y, yaw = control_step
                linear_x = clamp(math.hypot(x, y), 0.0, float(self._params['max_linear']))
                angular_z = clamp(yaw, -float(self._params['max_angular']), float(self._params['max_angular']))
                debug.setdefault('trajectory_control_step', [x, y, yaw])
                first_step = _trajectory_first_step(trajectory)
                if first_step is not None:
                    debug.setdefault('trajectory_first_step', list(first_step))
                debug.setdefault('trajectory_command_interface', 'internnav_continuous_subgoal_vw')
                debug.setdefault('selected_output_mode', 'trajectory')
                return ModelSimDecision(
                    linear_x=linear_x,
                    angular_z=angular_z,
                    status=str(output.get('status', 'trajectory_command')),
                    degraded=bool(output.get('degraded', False)),
                    debug=debug,
                )

            def _discrete_decision() -> Optional[ModelSimDecision]:
                if discrete_action is None or 'linear_x' in output or 'angular_z' in output:
                    return None
                command = _action_to_command(discrete_action, self._params)
                if command is None:
                    return None
                linear_x, angular_z, status, action_debug = command
                debug.update(action_debug)
                debug.setdefault('converted_status', status)
                debug.setdefault('selected_output_mode', 'discrete')
                if trajectory is not None:
                    first_step = _trajectory_first_step(trajectory)
                    if first_step is not None:
                        debug.setdefault('trajectory_first_step', list(first_step))
                return ModelSimDecision(
                    linear_x=linear_x,
                    angular_z=angular_z,
                    status=str(output.get('status', status)),
                    degraded=bool(output.get('degraded', False)),
                    debug=debug,
                )

            if output_policy == 'trajectory':
                decision = _trajectory_decision() or _discrete_decision()
                if decision is not None:
                    return decision
            elif output_policy == 'discrete':
                decision = _discrete_decision() or _trajectory_decision()
                if decision is not None:
                    return decision
            else:
                decision = _discrete_decision() or _trajectory_decision()
                if decision is not None:
                    return decision

            debug.setdefault('selected_output_mode', 'cmd_vel')
            return ModelSimDecision(
                linear_x=float(output.get('linear_x', 0.0)),
                angular_z=float(output.get('angular_z', 0.0)),
                status=str(output.get('status', 'adapter_command')),
                degraded=bool(output.get('degraded', False)),
                debug=debug,
            )
        raise TypeError(
            f'Adapter output must be ModelSimDecision or dict, got {type(output).__name__}'
        )

    def compute(self, observation: ModelSimObservation) -> ModelSimDecision:
        if self._adapter_callable is None:
            return _safe_stop_decision(
                observation,
                status='model_unavailable',
                reason=f"adapter_target '{self._adapter_target}' is unavailable",
                debug={'adapter_target': self._adapter_target},
            )

        start = time.monotonic()
        try:
            output = self._adapter_callable(observation)
        except Exception as exc:
            self._log('error', f"Adapter compute failed for '{self._adapter_target}': {exc}")
            return _safe_stop_decision(
                observation,
                status='adapter_exception',
                reason=str(exc),
                debug={'adapter_target': self._adapter_target},
            )

        elapsed = time.monotonic() - start
        if elapsed > float(self._params['inference_timeout_sec']):
            return _safe_stop_decision(
                observation,
                status='inference_timeout',
                reason=f'adapter inference exceeded {self._params["inference_timeout_sec"]}s',
                debug={'adapter_target': self._adapter_target, 'infer_time_sec': elapsed},
            )

        try:
            decision = self._coerce_output(output)
        except Exception as exc:
            self._log('error', f"Adapter output validation failed for '{self._adapter_target}': {exc}")
            return _safe_stop_decision(
                observation,
                status='invalid_adapter_output',
                reason=str(exc),
                debug={'adapter_target': self._adapter_target, 'raw_output_type': type(output).__name__},
            )

        decision.debug.update(_goal_debug(observation))
        if 'adapter_target' in decision.debug and decision.debug['adapter_target'] != self._adapter_target:
            decision.debug.setdefault('native_adapter_target', decision.debug['adapter_target'])
        decision.debug['adapter_target'] = self._adapter_target
        decision.debug['infer_time_sec'] = elapsed
        try:
            return _finalize_decision(decision, self._params)
        except Exception as exc:
            self._log('error', f"Adapter decision finalization failed for '{self._adapter_target}': {exc}")
            return _safe_stop_decision(
                observation,
                status='invalid_adapter_output',
                reason=str(exc),
                debug={'adapter_target': self._adapter_target, 'infer_time_sec': elapsed},
            )


def create_model_backend(mode: str, logger, params: dict[str, Any]) -> ModelSimBackend:
    normalized_mode = mode.strip().lower()
    backend_cls = _BACKEND_REGISTRY.get(normalized_mode)
    if backend_cls is None:
        raise ValueError(
            f"Unsupported dual_vln backend mode '{mode}'. Supported modes: {sorted(_BACKEND_REGISTRY)}"
        )
    return backend_cls(logger, params)


# Backward-compatible aliases while the repository migrates from the older
# Dual-VLN naming to the more general model-sim wrapper vocabulary.
DualVLNObservation = ModelSimObservation
DualVLNDecision = ModelSimDecision
DualVLNBackend = ModelSimBackend
create_backend = create_model_backend
