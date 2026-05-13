from __future__ import annotations

import importlib
import os
import site
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from arena_vln_models.core import (
    BackendCapability,
    camera_intrinsic_matrix,
    depth_debug,
    goal_delta,
    normalize_backend_output,
    normalize_depth,
    normalize_rgb,
    pose_vector,
    safe_stop,
    to_jsonable,
)


INTERNNAV_REALWORLD_CAPABILITY = BackendCapability(
    name='internnav_realworld_internvla_n1',
    required_inputs=('rgb_image', 'depth_image', 'instruction', 'camera_intrinsics'),
    output_modes=('discrete_action', 'output_trajectory', 'output_pixel'),
    supports_batch=False,
    notes='Uses InternNav InternVLAN1AsyncAgent realworld step(rgb, depth, pose, instruction, intrinsic, look_down).',
)

INTERNNAV_MODEL_PATH_ENV_VARS = (
    'ARENA_INTERNNAV_MODEL_PATH',
    'INTERNNAV_MODEL_PATH',
    'ARENA_VLN_MODEL_PATH',
)

INTERNNAV_MODEL_PYTHON_ENV_VARS = (
    'ARENA_VLN_MODEL_PYTHON',
    'ARENA_INTERNNAV_PYTHON',
    'ARENA_PYTHON',
)


def _workspace_root() -> Path:
    explicit = os.environ.get('ARENA_WS') or os.environ.get('ARENA_WORKSPACE')
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.exists():
            return candidate

    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / 'deps' / 'InternNav').exists():
            return parent
    # Installed package fallback: /workspace/install/pkg/lib/python.../site-packages/...
    for parent in current.parents:
        workspace = parent.parent if parent.name in {'install', 'build'} else parent
        if (workspace / 'deps' / 'InternNav').exists():
            return workspace
    return current.parents[4]


def _internnav_root() -> Path:
    return _workspace_root() / 'deps' / 'InternNav'


def _recommended_requirement_files() -> list[str]:
    requirements_dir = _internnav_root() / 'requirements'
    files = [
        requirements_dir / 'core_requirements.txt',
        requirements_dir / 'internvla_n1.txt',
    ]
    return [str(path) for path in files if path.is_file()]


def _ensure_internnav_sys_paths() -> list[str]:
    root = _internnav_root()
    candidates: list[Path] = []
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = ''
    if user_site:
        candidates.append(Path(user_site))
    candidates.extend([root, root / 'third_party' / 'diffusion-policy'])

    added: list[str] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if not candidate.exists():
            continue
        if candidate_str in sys.path:
            sys.path.remove(candidate_str)
        sys.path.insert(0, candidate_str)
        added.append(candidate_str)
    return added


def _configure_internnav_runtime_env() -> dict[str, Any]:
    debug: dict[str, Any] = {}
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    debug['recommended_requirement_files'] = _recommended_requirement_files()
    debug['recommended_model_python_env_vars'] = list(INTERNNAV_MODEL_PYTHON_ENV_VARS)
    debug['recommended_model_path_env_vars'] = list(INTERNNAV_MODEL_PATH_ENV_VARS)
    if not os.environ.get('INTERNNAV_DEPTH_ANYTHING_CKPT', '').strip():
        workspace_root = _workspace_root()
        candidates = [
            _internnav_root() / 'checkpoints' / 'depth_anything_v2_metric_hypersim_vits.pth',
            workspace_root / 'deps' / 'models' / 'depth-anything-v2-metric-hypersim-small' / 'depth_anything_v2_metric_hypersim_vits.pth',
            workspace_root / 'deps' / 'models' / 'depth_anything_v2_metric_hypersim_vits.pth',
        ]
        for checkpoint in candidates:
            if not checkpoint.is_file():
                continue
            os.environ['INTERNNAV_DEPTH_ANYTHING_CKPT'] = str(checkpoint)
            debug['depth_anything_checkpoint'] = str(checkpoint)
            debug['depth_anything_checkpoint_source'] = 'local_default'
            break
    return debug


def _ensure_legacy_module_aliases() -> list[str]:
    aliases: list[tuple[str, str]] = [
        ('diffusion_policy', 'internnav.model.encoder.diffusion_policy'),
    ]
    added: list[str] = []
    for alias, target in aliases:
        if alias in sys.modules:
            continue
        try:
            sys.modules[alias] = importlib.import_module(target)
            added.append(alias)
        except Exception:
            continue
    return added


def _resolve_model_path(model_path: str) -> tuple[str, dict[str, Any]]:
    raw = str(model_path or '').strip()
    debug: dict[str, Any] = {'requested_model_path': raw}
    if not raw:
        for env_name in INTERNNAV_MODEL_PATH_ENV_VARS:
            env_value = str(os.environ.get(env_name, '')).strip()
            if not env_value:
                continue
            raw = env_value
            debug['requested_model_path'] = raw
            debug['requested_model_path_source'] = f'env:{env_name}'
            break
    if not raw:
        return '', debug

    workspace_root = _workspace_root()
    requested = Path(raw).expanduser()
    candidate_roots = [
        workspace_root,
        workspace_root / 'models',
        workspace_root / 'deps',
        workspace_root / 'deps' / 'models',
        _internnav_root(),
        _internnav_root().parent,
        _internnav_root().parent / 'models',
    ]
    candidates: list[Path] = []

    def _add_candidate(path: Path) -> None:
        try:
            normalized = path.expanduser().resolve(strict=False)
        except Exception:
            normalized = path.expanduser()
        if normalized not in candidates:
            candidates.append(normalized)

    if requested.is_absolute():
        _add_candidate(requested)
    else:
        _add_candidate(Path.cwd() / requested)
        for root in candidate_roots:
            _add_candidate(root / requested)

    requested_name = requested.name
    if requested_name:
        for root in candidate_roots:
            _add_candidate(root / requested_name)

    debug['model_path_candidates'] = [str(path) for path in candidates[:12]]
    for candidate in candidates:
        config_candidate = candidate / 'config.json'
        if candidate.is_dir() and config_candidate.is_file():
            resolved = str(candidate)
            debug['resolved_model_path'] = resolved
            debug['resolved_model_path_source'] = 'local_directory'
            if resolved != raw:
                debug['resolved_model_path_via_fallback'] = True
            return resolved, debug

    debug['resolved_model_path'] = raw
    debug['resolved_model_path_source'] = 'unresolved'
    return raw, debug


def _resolve_runtime_device(requested_device: str, *, strict_device: bool = False) -> tuple[str, str]:
    requested = str(requested_device or 'cpu').strip() or 'cpu'
    try:
        import torch  # type: ignore

        if requested.startswith('cuda') and not torch.cuda.is_available():
            if strict_device:
                raise RuntimeError(f"Requested device '{requested}' but CUDA is unavailable")
            return 'cpu', f"Requested device '{requested}' but CUDA is unavailable; falling back to cpu"
        torch.device(requested)
        return requested, ''
    except Exception as exc:
        if requested == 'cpu':
            return 'cpu', f'Failed to validate torch runtime: {exc}'
        if strict_device:
            raise RuntimeError(f"Failed to validate requested device '{requested}': {exc}") from exc
        return 'cpu', f"Failed to validate requested device '{requested}': {exc}; falling back to cpu"


class InternNavAdapter:
    """Arena adapter_target wrapper for InternNav backends.

    The class intentionally accepts duck-typed observations so it can be loaded
    by the consolidated Arena VLN backend without creating model-specific ROS
    type dependencies.
    """

    capability = INTERNNAV_REALWORLD_CAPABILITY

    def __init__(self, logger=None, params: Optional[dict[str, Any]] = None) -> None:
        self._logger = logger
        self._params = params or {}
        self._adapter: Any = None
        self._mode = 'mock'
        self._fallback_reason = ''
        self._load_error = ''
        self._load_debug: dict[str, Any] = {'wrapper_package': 'arena_vln_models'}
        self._pending_actions: list[int] = []
        self._session_key: Optional[tuple[str, Optional[tuple[float, float, float]]]] = None
        self._require_real_backend = bool(self._params.get('require_real_backend', False))
        self._strict_device = bool(self._params.get('strict_device', False))
        self._load_backend()

    def _log(self, level: str, message: str) -> None:
        logger = self._logger
        if logger is None:
            return
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(message)

    def _load_backend(self) -> None:
        requested_model_path = str(self._params.get('model_path', '')).strip()
        model_path, path_debug = _resolve_model_path(requested_model_path)
        requested_device = str(self._params.get('device', 'cpu')).strip() or 'cpu'
        if self._strict_device:
            os.environ['INTERNNAV_STRICT_DEVICE'] = '1'
        self._load_debug.update({
            'model_path': model_path,
            'requested_device': requested_device,
            'adapter_target': 'internnav.agent.internvla_n1_agent_realworld.InternVLAN1AsyncAgent',
            'capability': self.capability.name,
            **path_debug,
        })
        if not requested_model_path or requested_model_path.lower() in {'mock', 'dryrun', 'dummy', 'stub'}:
            self._fallback_reason = 'InternNav wrapper running in deterministic mock mode (no model_path provided)'
            self._log('warn', self._fallback_reason)
            self._load_debug['load_mode'] = 'mock'
            if self._require_real_backend:
                raise RuntimeError(self._fallback_reason)
            return

        root = _internnav_root()
        if not root.exists():
            self._mode = 'unavailable'
            self._load_error = f'InternNav repo not found at {root}'
            self._load_debug['load_mode'] = 'unavailable'
            self._log('error', self._load_error)
            if self._require_real_backend:
                raise RuntimeError(self._load_error)
            return

        added_paths = _ensure_internnav_sys_paths()
        runtime_env_debug = _configure_internnav_runtime_env()
        added_aliases = _ensure_legacy_module_aliases()
        if added_paths:
            self._load_debug['added_sys_paths'] = added_paths
        if added_aliases:
            self._load_debug['added_module_aliases'] = added_aliases
        self._load_debug.update(runtime_env_debug)

        try:
            runtime_device, runtime_note = _resolve_runtime_device(requested_device, strict_device=self._strict_device)
            self._load_debug.update({'runtime_device': runtime_device})
            if runtime_note:
                self._load_debug['runtime_note'] = runtime_note
            module = importlib.import_module('internnav.agent.internvla_n1_agent_realworld')
            agent_cls = getattr(module, 'InternVLAN1AsyncAgent')
            args = SimpleNamespace(
                device=runtime_device,
                model_path=model_path,
                resize_w=int(self._params.get('internnav_resize_w', 448)),
                resize_h=int(self._params.get('internnav_resize_h', 448)),
                num_history=int(self._params.get('internnav_num_history', 4)),
                plan_step_gap=int(self._params.get('internnav_plan_step_gap', 4)),
            )
            self._adapter = agent_cls(args)
            self._mode = 'internnav_async_agent'
            self._fallback_reason = ''
            self._load_error = ''
            self._load_debug.update({'load_mode': 'real'})
            if runtime_note:
                self._log('warn', runtime_note)
            if model_path != requested_model_path:
                self._log('warn', f"InternNav wrapper resolved model_path '{requested_model_path}' -> '{model_path}'")
            self._log('info', f"InternNav wrapper loaded real backend from '{model_path}' on device='{runtime_device}'")
        except Exception as exc:
            self._adapter = None
            self._mode = 'unavailable'
            self._load_error = str(exc)
            self._load_debug['load_mode'] = 'unavailable'
            self._load_debug['load_exception_type'] = type(exc).__name__
            self._log('error', f"InternNav wrapper failed to load real backend from '{model_path}': {exc}")
            if self._require_real_backend:
                raise RuntimeError(
                    f"InternNav required real backend failed to load from '{model_path}': {exc}"
                ) from exc

    def _unavailable_response(self) -> dict[str, Any]:
        return safe_stop(
            'model_unavailable',
            self._load_error or 'InternNav backend is unavailable',
            debug={'shim_mode': self._mode, **self._load_debug},
        )

    def _reset_if_needed(self, observation: Any) -> None:
        goal = getattr(observation, 'goal', None)
        goal_key = None
        if goal is not None:
            goal_key = (
                round(float(getattr(goal, 'x', 0.0)), 2),
                round(float(getattr(goal, 'y', 0.0)), 2),
                round(float(getattr(goal, 'yaw', 0.0)), 2),
            )
        session_key = (str(getattr(observation, 'instruction', '')), goal_key)
        if session_key == self._session_key:
            return

        self._pending_actions.clear()
        self._session_key = session_key
        if self._adapter is not None and hasattr(self._adapter, 'reset'):
            try:
                self._adapter.reset()
            except TypeError:
                try:
                    self._adapter.reset(None)
                except Exception:
                    pass
            except Exception as exc:
                self._log('warn', f'InternNav wrapper reset failed: {exc}')

    def _queued_action_response(self) -> Optional[dict[str, Any]]:
        if not self._pending_actions:
            return None
        action = self._pending_actions.pop(0)
        return {
            'discrete_action': action,
            'status': 'internnav_action_queue',
            'debug': {
                'shim_mode': self._mode,
                'queued_action': True,
                'remaining_action_queue': len(self._pending_actions),
                **self._load_debug,
            },
        }

    def _mock_response(self, observation: Any) -> dict[str, Any]:
        delta = goal_delta(observation)
        if delta is None:
            return safe_stop(
                'mock_missing_pose_or_goal',
                self._fallback_reason or 'missing pose/goal',
                debug={'shim_mode': self._mode, **self._load_debug},
            )

        local_x, local_y, dist, yaw = delta
        target_pixel = None
        rgb = getattr(observation, 'rgb_image', None)
        if rgb is not None:
            shape = np.asarray(rgb).shape
            if len(shape) >= 2:
                height, width = int(shape[0]), int(shape[1])
                target_pixel = [width // 2 + int(max(min(yaw, 1.2), -1.2) / 1.2 * width * 0.2), height // 2]

        if dist <= float(self._params.get('goal_tolerance', 0.35)):
            action = 0
        elif abs(yaw) > float(self._params.get('angle_tolerance', 0.25)):
            action = 2 if yaw > 0.0 else 3
        else:
            action = 1

        return {
            'discrete_action': action,
            'output_trajectory': [[local_x, local_y, yaw]],
            'output_pixel': target_pixel,
            'status': 'mock_internnav_command',
            'debug': {
                'shim_mode': self._mode,
                'shim_reason': self._fallback_reason or 'deterministic fallback',
                'goal_local': [local_x, local_y, yaw],
                'remaining_action_queue': 0,
                **self._load_debug,
            },
        }

    def _real_response(self, observation: Any) -> dict[str, Any]:
        rgb = normalize_rgb(getattr(observation, 'rgb_image', None))
        if rgb is None:
            return safe_stop(
                'internnav_missing_rgb',
                'rgb_image is required',
                debug={'shim_mode': self._mode, **self._load_debug},
            )

        depth = normalize_depth(getattr(observation, 'depth_image', None), reference_shape=(rgb.shape[0], rgb.shape[1]))
        if depth is None:
            return safe_stop(
                'internnav_missing_depth',
                'depth_image must be HxW or HxWx1',
                debug={'shim_mode': self._mode, **self._load_debug},
            )

        output = self._adapter.step(
            rgb,
            depth,
            pose_vector(observation),
            str(getattr(observation, 'instruction', '')),
            camera_intrinsic_matrix(observation, rgb),
            bool(getattr(observation, 'look_down', False)),
        )

        response = normalize_backend_output(output, default_status='internnav_command')
        response.setdefault('debug', {})
        response['debug'].update({'shim_mode': self._mode, **self._load_debug, **depth_debug(depth)})

        action_tail = response.get('debug', {}).pop('action_sequence_tail', None)
        if action_tail:
            self._pending_actions.extend([int(item) for item in action_tail])
            response['debug']['remaining_action_queue'] = len(self._pending_actions)
        return response

    def compute(self, observation: Any) -> dict[str, Any]:
        self._reset_if_needed(observation)

        queued = self._queued_action_response()
        if queued is not None:
            return queued

        if self._adapter is None:
            if self._mode == 'mock':
                return self._mock_response(observation)
            return self._unavailable_response()
        return self._real_response(observation)

    def predict(self, observation: Any) -> dict[str, Any]:
        return self.compute(observation)


def load_internnav_adapter(logger=None, params: Optional[dict[str, Any]] = None) -> InternNavAdapter:
    return InternNavAdapter(logger=logger, params=params)


def available_backends() -> list[dict[str, Any]]:
    return [to_jsonable(INTERNNAV_REALWORLD_CAPABILITY.__dict__)]
