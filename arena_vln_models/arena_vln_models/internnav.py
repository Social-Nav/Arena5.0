from __future__ import annotations

import importlib
import io
import json
import os
import site
import subprocess
import sys
import tempfile
import threading
import time
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
            reasons: list[str] = []
            torch_cuda_version = getattr(getattr(torch, 'version', None), 'cuda', None)
            if not torch_cuda_version:
                reasons.append('installed torch build has no CUDA support (torch.version.cuda is None)')
            visible_gpu_nodes = sorted(str(path) for path in Path('/dev').glob('nvidia*'))
            if not visible_gpu_nodes:
                reasons.append('no /dev/nvidia* devices are visible in the current process environment')
            nvidia_visible_devices = os.environ.get('NVIDIA_VISIBLE_DEVICES', '').strip()
            if nvidia_visible_devices.lower() in {'void', 'none'}:
                reasons.append(f'NVIDIA_VISIBLE_DEVICES={nvidia_visible_devices}')
            detail = '; '.join(reasons) or 'CUDA is unavailable'
            if strict_device:
                raise RuntimeError(f"Requested device '{requested}' but CUDA is unavailable: {detail}")
            return 'cpu', f"Requested device '{requested}' but CUDA is unavailable: {detail}; falling back to cpu"
        torch.device(requested)
        return requested, ''
    except Exception as exc:
        if requested == 'cpu':
            return 'cpu', f'Failed to validate torch runtime: {exc}'
        if strict_device:
            raise RuntimeError(f"Failed to validate requested device '{requested}': {exc}") from exc
        return 'cpu', f"Failed to validate requested device '{requested}': {exc}; falling back to cpu"


def _pose_matrix_from_observation(observation: Any) -> list[list[float]]:
    pose = pose_vector(observation)
    x = float(pose[0]) if len(pose) > 0 else 0.0
    y = float(pose[1]) if len(pose) > 1 else 0.0
    yaw = float(pose[2]) if len(pose) > 2 else 0.0
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return [
        [c, -s, 0.0, x],
        [s, c, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


class InternVLARealworldHttpAdapter:
    """Arena adapter for InternNav's official realworld HTTP `/eval_dual` service.

    This keeps the Arena side as a ROS2 `get_command` server/client while the
    heavy InternVLA-N1 System-2/System-1 model lives behind the upstream
    realworld HTTP API.  `internnav_server` already invokes adapter compute in a
    background worker, so the simulator/control loop returns cached/fallback
    commands while slow HTTP model inference is running.
    """

    capability = INTERNNAV_REALWORLD_CAPABILITY

    def __init__(self, logger=None, params: Optional[dict[str, Any]] = None) -> None:
        self._logger = logger
        self._params = params or {}
        self._url = str(
            self._params.get('internnav_http_url')
            or os.environ.get('ARENA_EVAL_INTERNNAV_HTTP_URL')
            or os.environ.get('ARENA_INTERNNAV_HTTP_URL')
            or 'http://127.0.0.1:5801/eval_dual'
        ).strip()
        timeout_value = (
            self._params.get('internnav_http_timeout_sec')
            or os.environ.get('ARENA_EVAL_INTERNNAV_HTTP_TIMEOUT_SEC')
            or os.environ.get('ARENA_INTERNNAV_HTTP_TIMEOUT_SEC')
            or self._params.get('inference_timeout_sec', 120.0)
        )
        try:
            self._timeout = float(timeout_value)
        except (TypeError, ValueError):
            self._log('warn', f'Invalid internnav_http_timeout_sec={timeout_value!r}; using inference_timeout_sec')
            self._timeout = float(self._params.get('inference_timeout_sec', 120.0) or 120.0)
        if self._timeout <= 0.0:
            self._timeout = float(self._params.get('inference_timeout_sec', 120.0) or 120.0)
        self._session_key: Optional[tuple[str, Optional[tuple[float, float, float]]]] = None
        self._seq = 0

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        if level == 'debug':
            self._logger.debug(message)
        elif level == 'info':
            self._logger.info(message)
        elif level in ('warn', 'warning'):
            self._logger.warn(message)
        elif level == 'error':
            self._logger.error(message)

    def _session_reset_required(self, observation: Any) -> bool:
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
            return False
        self._session_key = session_key
        return True

    @staticmethod
    def _image_payloads(rgb: np.ndarray, depth: np.ndarray) -> tuple[io.BytesIO, io.BytesIO]:
        from PIL import Image as PILImage

        rgb_bytes = io.BytesIO()
        PILImage.fromarray(rgb).save(rgb_bytes, format='JPEG')
        rgb_bytes.seek(0)

        depth_m = np.asarray(depth, dtype=np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_u16 = np.clip(depth_m * 10000.0, 0.0, 65535.0).astype(np.uint16)
        depth_bytes = io.BytesIO()
        PILImage.fromarray(depth_u16).save(depth_bytes, format='PNG')
        depth_bytes.seek(0)
        return rgb_bytes, depth_bytes

    @staticmethod
    def _post_eval_dual(url: str, *, files: dict[str, tuple[str, io.BytesIO, str]], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            requests = importlib.import_module('requests')
        except ModuleNotFoundError:
            requests = None

        if requests is not None:
            response = requests.post(url, files=files, data={'json': json.dumps(payload)}, timeout=timeout)
            response.raise_for_status()
            return response.json()

        import urllib.request
        import uuid

        boundary = f'----arena-internvla-{uuid.uuid4().hex}'
        body = bytearray()

        def _add_field(name: str, value: str) -> None:
            body.extend(f'--{boundary}\r\n'.encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(value.encode())
            body.extend(b'\r\n')

        def _add_file(name: str, filename: str, fileobj: io.BytesIO, content_type: str) -> None:
            body.extend(f'--{boundary}\r\n'.encode())
            body.extend(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f'Content-Type: {content_type}\r\n\r\n'.encode()
            )
            body.extend(fileobj.getvalue())
            body.extend(b'\r\n')

        _add_field('json', json.dumps(payload))
        for field_name, (filename, fileobj, content_type) in files.items():
            _add_file(field_name, filename, fileobj, content_type)
        body.extend(f'--{boundary}--\r\n'.encode())
        request = urllib.request.Request(
            url,
            data=bytes(body),
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    def compute(self, observation: Any) -> dict[str, Any]:
        rgb = normalize_rgb(getattr(observation, 'rgb_image', None))
        if rgb is None:
            return safe_stop(
                'internnav_http_missing_rgb',
                'rgb_image is required for InternVLA realworld HTTP client',
                debug={'http_url': self._url, 'realworld_http_client': True},
            )
        depth = normalize_depth(getattr(observation, 'depth_image', None), reference_shape=(rgb.shape[0], rgb.shape[1]))
        if depth is None:
            return safe_stop(
                'internnav_http_missing_depth',
                'depth_image must be HxW or HxWx1 for InternVLA realworld HTTP client',
                debug={'http_url': self._url, 'realworld_http_client': True},
            )

        self._seq += 1
        request_id = self._seq
        reset = self._session_reset_required(observation)
        rgb_bytes, depth_bytes = self._image_payloads(rgb, depth)
        payload = {
            'reset': reset,
            'idx': request_id - 1,
            'request_id': request_id,
            'instruction': str(getattr(observation, 'instruction', '')),
            'pose': to_jsonable(pose_vector(observation)),
            'camera_pose': _pose_matrix_from_observation(observation),
            'intrinsic': to_jsonable(camera_intrinsic_matrix(observation, rgb)),
            'look_down': bool(getattr(observation, 'look_down', False)),
            'model_output_policy': str(self._params.get('model_output_policy', 'trajectory')),
            'client': 'arena_ros2_get_command_async_worker',
        }
        files = {
            'image': ('rgb_image.jpg', rgb_bytes, 'image/jpeg'),
            'depth': ('depth_image.png', depth_bytes, 'image/png'),
        }

        start = time.monotonic()
        try:
            raw = self._post_eval_dual(self._url, files=files, payload=payload, timeout=self._timeout)
        except Exception as exc:
            return safe_stop(
                'internnav_http_request_failed',
                str(exc),
                debug={
                    'http_url': self._url,
                    'http_timeout_sec': self._timeout,
                    'request_id': request_id,
                    'realworld_http_client': True,
                },
            )

        elapsed = time.monotonic() - start
        result = normalize_backend_output(raw, default_status='internnav_http_command')
        result.setdefault('debug', {})
        result['debug'].update({
            'http_url': self._url,
            'http_timeout_sec': self._timeout,
            'http_request_id': request_id,
            'http_reset': reset,
            'http_elapsed_sec': elapsed,
            'realworld_http_client': True,
            'arena_async_boundary': 'ros2_get_command_returns_cached_command_while_http_compute_runs',
            'system2_http_server': True,
            'system1_http_server': True,
            **depth_debug(depth),
        })
        return result

    def predict(self, observation: Any) -> dict[str, Any]:
        return self.compute(observation)


class InternNavSubprocessAdapter:
    """Run the heavy InternNav model in a separate Python environment.

    ROS 2 nodes must run in the Python interpreter that matches the installed
    rclpy/type-support ABI, while the InternNav checkpoint often needs a separate
    torch/transformers/flash-attn environment.  This adapter keeps ROS in the eval
    interpreter and exchanges only JSON + temporary NumPy files with the model
    interpreter selected by ARENA_VLN_MODEL_PYTHON / --internnav-python-executable.
    """

    def __init__(
        self,
        python_executable: str,
        model_path: str,
        requested_device: str,
        params: dict[str, Any],
        internnav_root: Path,
        logger=None,
    ) -> None:
        self._python = python_executable
        self._logger = logger
        self._timeout = float(params.get('inference_timeout_sec', 120.0))
        self._tmpdir = tempfile.TemporaryDirectory(prefix='arena_internnav_ipc_')
        self._seq = 0
        self._request_files: dict[int, tuple[str, str]] = {}
        self._stderr_lines: list[str] = []
        env = os.environ.copy()
        output_policy = str(params.get('model_output_policy', 'trajectory')).strip().lower()
        default_max_new_tokens = 32 if output_policy in {'trajectory', 'continuous', 'output_trajectory', 'traj'} else 10
        env.setdefault(
            'ARENA_INTERNNAV_MAX_NEW_TOKENS',
            str(int(params.get('internnav_max_new_tokens', default_max_new_tokens))),
        )
        work_dir = Path(env.get('ARENA_INTERNNAV_WORK_DIR', '/tmp/arena_internnav_work'))
        work_dir.mkdir(parents=True, exist_ok=True)
        pythonpath_parts = [str(internnav_root), str(internnav_root / 'third_party' / 'diffusion-policy')]
        if env.get('PYTHONPATH'):
            pythonpath_parts.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
        worker_code = self._worker_code()
        command = [
            python_executable,
            '-u',
            '-c',
            worker_code,
            '--model-path',
            model_path,
            '--device',
            requested_device,
            '--resize-w',
            str(int(params.get('internnav_resize_w', os.environ.get('ARENA_INTERNNAV_RESIZE_W', 336)))),
            '--resize-h',
            str(int(params.get('internnav_resize_h', os.environ.get('ARENA_INTERNNAV_RESIZE_H', 336)))),
            '--num-history',
            str(int(params.get('internnav_num_history', os.environ.get('ARENA_INTERNNAV_NUM_HISTORY', 0)))),
            '--plan-step-gap',
            str(int(params.get('internnav_plan_step_gap', os.environ.get('ARENA_INTERNNAV_PLAN_STEP_GAP', 12)))),
            '--model-output-policy',
            output_policy,
        ]
        if bool(params.get('strict_device', False)):
            command.append('--strict-device')
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(work_dir),
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name='arena_internnav_stderr_drain',
            daemon=True,
        )
        self._stderr_thread.start()
        ready = self._read_response(timeout_sec=max(self._timeout, 30.0))
        if ready.get('status') != 'ready':
            raise RuntimeError(f'InternNav subprocess did not become ready: {ready}')

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        if level == 'debug':
            self._logger.debug(message)
        elif level == 'info':
            self._logger.info(message)
        elif level in ('warn', 'warning'):
            self._logger.warn(message)
        elif level == 'error':
            self._logger.error(message)

    def _drain_stderr(self) -> None:
        stream = getattr(self._proc, 'stderr', None)
        if stream is None:
            return
        try:
            for line in stream:
                line = line.rstrip('\n')
                if not line:
                    continue
                self._stderr_lines.append(line)
                if len(self._stderr_lines) > 40:
                    del self._stderr_lines[: len(self._stderr_lines) - 40]
        except Exception:
            return

    @staticmethod
    def _worker_code() -> str:
        return r'''
import argparse, json, math, sys, traceback
from types import SimpleNamespace
import numpy as np

def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return _jsonable(value.tolist())
    return str(value)

def _action_to_synthetic_trajectory(action):
    """Convert a discrete action id to a synthetic trajectory waypoint.

    ROS coordinate convention (REP 103): +yaw = counter-clockwise = LEFT turn.
    This mapping is intentionally hard-coded to match the trajectory policy's
    passthrough yaw sign — a mismatched sign between discrete and trajectory
    policies was the root cause of yaw_sign_mismatch_ratio reaching 0.73–0.96.
    """
    try:
        action_id = int(action)
    except Exception:
        return None
    # Local [x, y, yaw] subgoal matching InternNav's real-world continuous
    # trajectory interface closely enough for Arena's trajectory cmd_vel path.
    # This is only used when the model answers with symbolic arrows despite a
    # trajectory-policy prompt, so trajectory policy still exercises the
    # output_trajectory -> cmd_vel converter instead of silently falling back to
    # action_list -> cmd_vel.
    turn = math.radians(15.0)
    if action_id == 1:
        return [[0.25, 0.0, 0.0]]
    if action_id == 2:
        # action 2 = native "turn_left" → +yaw (ROS CCW / left)
        return [[0.25, 0.0, turn]]
    if action_id == 3:
        # action 3 = native "turn_right" → -yaw (ROS CW / right)
        return [[0.25, 0.0, -turn]]
    if action_id == 0:
        return [[0.0, 0.0, 0.0]]
    if action_id == 5:
        return [[0.0, 0.0, 0.0]]
    return None

def _normalize_output(output, policy='trajectory'):
    if isinstance(output, dict):
        result = dict(output)
    elif isinstance(output, (int, np.integer)):
        result = {'discrete_action': int(output)}
    elif isinstance(output, (list, tuple, np.ndarray)):
        data = output.tolist() if hasattr(output, 'tolist') else list(output)
        if data and isinstance(data[0], (int, float, np.integer, np.floating)):
            result = {'discrete_action': int(data[0])}
        else:
            result = {'output_trajectory': data}
    else:
        action = getattr(output, 'output_action', None)
        trajectory = getattr(output, 'output_trajectory', None)
        pixel = getattr(output, 'output_pixel', None)
        result = {'debug': {'raw_output': str(output)}}
        if action is not None:
            action_list = np.asarray(action).reshape(-1).tolist()
            if action_list:
                result['discrete_action'] = int(action_list[0])
                if len(action_list) > 1:
                    result['debug']['action_sequence_tail'] = [int(item) for item in action_list[1:]]
        if trajectory is not None:
            result['output_trajectory'] = trajectory
        if pixel is not None:
            result['output_pixel'] = pixel
        if 'discrete_action' not in result and 'output_trajectory' not in result:
            result['status'] = 'internnav_unknown_output'
    result.setdefault('status', 'internnav_command')
    result.setdefault('debug', {})
    normalized_policy = str(policy or 'trajectory').strip().lower()
    if normalized_policy in {'trajectory', 'continuous', 'output_trajectory', 'traj'}:
        if result.get('output_trajectory') is None and result.get('discrete_action') is not None:
            synthetic = _action_to_synthetic_trajectory(
                result.get('discrete_action'),
            )
            if synthetic is not None:
                result['output_trajectory'] = synthetic
                result['debug']['trajectory_synthesized_from_discrete_action'] = True
                result['debug']['trajectory_synthesis_reason'] = 'model_returned_symbolic_action_without_output_trajectory'
    return _jsonable(result)

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
parser.add_argument('--device', default='cpu')
parser.add_argument('--resize-w', type=int, default=448)
parser.add_argument('--resize-h', type=int, default=448)
parser.add_argument('--num-history', type=int, default=4)
parser.add_argument('--plan-step-gap', type=int, default=4)
parser.add_argument('--model-output-policy', default='trajectory')
parser.add_argument('--strict-device', action='store_true')
args = parser.parse_args()

try:
    import torch
    runtime_device = args.device
    if runtime_device.startswith('cuda') and not torch.cuda.is_available():
        if args.strict_device:
            raise RuntimeError(
                f"Requested device '{runtime_device}' but CUDA is unavailable in InternNav subprocess"
            )
        runtime_device = 'cpu'
    torch.device(runtime_device)
    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
    agent_args = SimpleNamespace(
        device=runtime_device,
        model_path=args.model_path,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        num_history=args.num_history,
        plan_step_gap=args.plan_step_gap,
    )
    agent = InternVLAN1AsyncAgent(agent_args)
    policy = str(args.model_output_policy or 'trajectory').strip().lower()
    if policy in {'trajectory', 'continuous', 'output_trajectory', 'traj'}:
        # InternVLA-N1 can emit either symbolic arrows or image coordinates.
        # For Arena's trajectory policy, bias S2 toward numeric image waypoints
        # so S1 can decode the latent into output_trajectory instead of falling
        # back to an action list.  Keep the normal prompt for discrete policy.
        prompt = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. "
            "Where should you go next to stay on track? Output ONLY the next waypoint's "
            "numeric image coordinates as two integers in row column order, for example "
            "240 320. Do not output arrow actions such as ↑, ←, →, ↓. Output STOP only "
            "when you have successfully completed the task."
        )
        try:
            agent.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
        except Exception:
            pass
    print(json.dumps({
        'status': 'ready',
        'runtime_device': runtime_device,
        'resize_w': args.resize_w,
        'resize_h': args.resize_h,
        'num_history': args.num_history,
        'plan_step_gap': args.plan_step_gap,
        'max_new_tokens': getattr(agent, 'max_new_tokens', None),
    }), flush=True)
except Exception as exc:
    print(json.dumps({'status': 'error', 'error': str(exc), 'traceback': traceback.format_exc()}), flush=True)
    sys.exit(1)

for line in sys.stdin:
    request_id = None
    try:
        req = json.loads(line)
        request_id = req.get('request_id')
        if req.get('cmd') == 'reset':
            if hasattr(agent, 'reset'):
                try:
                    agent.reset()
                except TypeError:
                    agent.reset(None)
            print(json.dumps({'status': 'reset_ok', 'request_id': request_id}), flush=True)
            continue
        rgb = np.load(req['rgb_path'])
        depth = np.load(req['depth_path'])
        import time
        t0 = time.time()
        output = agent.step(
            rgb,
            depth,
            req.get('pose', [0.0, 0.0, 0.0]),
            req.get('instruction', ''),
            req.get('intrinsic'),
            bool(req.get('look_down', False)),
        )
        elapsed = time.time() - t0
        result = _normalize_output(
            output,
            policy=args.model_output_policy,
        )
        result.setdefault('debug', {})
        result['debug']['subprocess_runtime_device'] = getattr(agent_args, 'device', '')
        result['debug']['subprocess_compute_sec'] = elapsed
        result['debug']['subprocess_episode_idx'] = getattr(agent, 'episode_idx', None)
        result['debug']['subprocess_model_output_policy'] = str(args.model_output_policy or '')
        result['debug']['subprocess_llm_output'] = getattr(agent, 'llm_output', '')
        result['debug']['subprocess_output_latent_pending'] = getattr(agent, 'output_latent', None) is not None
        result['debug']['subprocess_output_action_pending'] = getattr(agent, 'output_action', None) is not None
        print(json.dumps({'status': 'ok', 'request_id': request_id, 'result': result}), flush=True)
    except Exception as exc:
        print(json.dumps({'status': 'error', 'request_id': request_id, 'error': str(exc), 'traceback': traceback.format_exc()}), flush=True)
'''

    def _cleanup_request_files(self, request_id: Any) -> None:
        try:
            key = int(request_id)
        except Exception:
            return
        paths = self._request_files.pop(key, None)
        if not paths:
            return
        for path in paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _read_response(self, timeout_sec: float, expected_request_id: Optional[int] = None) -> dict[str, Any]:
        import select
        if self._proc.stdout is None:
            return {'status': 'error', 'error': 'subprocess stdout is unavailable'}
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        skipped_stdout: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return {
                    'status': 'error',
                    'error': f'timed out after {timeout_sec:.1f}s waiting for model subprocess',
                    'stdout_tail': skipped_stdout[-10:],
                    'stderr_tail': list(getattr(self, '_stderr_lines', [])[-20:]),
                }
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            line = self._proc.stdout.readline()
            if not line:
                return {
                    'status': 'error',
                    'error': f'subprocess exited rc={self._proc.poll()}',
                    'stdout_tail': skipped_stdout[-10:],
                    'stderr_tail': list(getattr(self, '_stderr_lines', [])[-20:]),
                }
            try:
                response = json.loads(line)
            except Exception:
                skipped_stdout.append(line.strip()[:400])
                if len(skipped_stdout) > 20:
                    del skipped_stdout[: len(skipped_stdout) - 20]
                continue
            if expected_request_id is None:
                return response
            response_request_id = response.get('request_id')
            if response_request_id == expected_request_id:
                return response
            self._cleanup_request_files(response_request_id)
            skipped_stdout.append(
                f'skipped stale subprocess response request_id={response_request_id!r} '
                f'while waiting for request_id={expected_request_id!r}: '
                f'status={response.get("status")!r}'
            )
            if len(skipped_stdout) > 20:
                del skipped_stdout[: len(skipped_stdout) - 20]

    def reset(self) -> None:
        if self._proc.poll() is not None:
            return
        if self._proc.stdin is None:
            return
        self._proc.stdin.write(json.dumps({'cmd': 'reset'}) + '\n')
        self._proc.stdin.flush()
        self._read_response(timeout_sec=5.0)

    def compute(self, observation: Any) -> dict[str, Any]:
        if self._proc.poll() is not None:
            raise RuntimeError(f'InternNav subprocess exited rc={self._proc.returncode}')
        rgb = normalize_rgb(getattr(observation, 'rgb_image', None))
        depth = normalize_depth(getattr(observation, 'depth_image', None), reference_shape=(rgb.shape[0], rgb.shape[1]) if rgb is not None else None)
        if rgb is None or depth is None:
            return safe_stop('internnav_missing_camera', 'rgb/depth image is required for subprocess InternNav')
        self._seq += 1
        rgb_path = os.path.join(self._tmpdir.name, f'rgb_{self._seq}.npy')
        depth_path = os.path.join(self._tmpdir.name, f'depth_{self._seq}.npy')
        np.save(rgb_path, rgb)
        np.save(depth_path, depth)
        self._request_files[self._seq] = (rgb_path, depth_path)
        payload = {
            'cmd': 'compute',
            'request_id': self._seq,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'pose': to_jsonable(pose_vector(observation)),
            'instruction': str(getattr(observation, 'instruction', '')),
            'intrinsic': camera_intrinsic_matrix(observation, rgb).tolist(),
            'look_down': bool(getattr(observation, 'look_down', False)),
        }
        response: dict[str, Any] = {}
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(payload) + '\n')
            self._proc.stdin.flush()
            response = self._read_response(timeout_sec=self._timeout, expected_request_id=self._seq)
        finally:
            # Do not remove IPC arrays after a timeout: the worker subprocess may
            # still be computing and has not necessarily opened the files yet.
            # Removing them races the worker and turns a slow inference into a
            # misleading FileNotFoundError on the next status update.
            timed_out = response.get('status') == 'error' and 'timed out' in str(response.get('error', ''))
            if not timed_out:
                self._cleanup_request_files(self._seq)
        if response.get('status') != 'ok':
            raise RuntimeError(response.get('error') or str(response))
        result = response.get('result')
        if not isinstance(result, dict):
            raise RuntimeError(f'Invalid subprocess result: {result!r}')
        return result

    def __del__(self):
        try:
            if getattr(self, '_proc', None) is not None and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass


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
        self._subprocess_adapter: Optional['InternNavSubprocessAdapter'] = None
        self._mode = 'mock'
        self._fallback_reason = ''
        self._load_error = ''
        self._load_debug: dict[str, Any] = {'wrapper_package': 'arena_vln_models'}
        self._pending_actions: list[int] = []
        self._session_key: Optional[tuple[str, Optional[tuple[float, float, float]]]] = None
        self._require_real_backend = bool(self._params.get('require_real_backend', False))
        self._strict_device = bool(self._params.get('strict_device', False))
        self._loading = False
        self._load_thread: Optional[threading.Thread] = None

        # Loading InternVLA-N1 can take longer than a short Arena episode on CPU.
        # Keep ROS services/status responsive and allow the wrapper to use its
        # deterministic InternNav-style command shim until the real subprocess is
        # ready.  When a caller explicitly requires the real backend, keep the old
        # synchronous fail-fast behavior.
        if self._require_real_backend:
            self._load_backend()
        else:
            self._mode = 'mock'
            self._fallback_reason = 'InternNav real backend is loading asynchronously; using deterministic command shim'
            self._load_debug['load_mode'] = 'loading_async'
            self._loading = True
            self._load_thread = threading.Thread(
                target=self._load_backend_async,
                name='arena_internnav_backend_loader',
                daemon=True,
            )
            self._load_thread.start()

    def _log(self, level: str, message: str) -> None:
        logger = self._logger
        if logger is None:
            return
        if level == 'debug':
            logger.debug(message)
        elif level == 'info':
            logger.info(message)
        elif level in ('warn', 'warning'):
            logger.warn(message)
        elif level == 'error':
            logger.error(message)

    def _load_backend_async(self) -> None:
        try:
            self._load_backend()
        finally:
            self._loading = False

    def _internnav_work_dir(self) -> Path:
        return Path(os.environ.get('ARENA_INTERNNAV_WORK_DIR', '/tmp/arena_internnav_work')).resolve()

    def _normalize_adapter_save_dir(self) -> None:
        if self._adapter is None:
            return
        save_dir = getattr(self._adapter, 'save_dir', None)
        if not isinstance(save_dir, str) or not save_dir.strip():
            return
        work_dir = self._internnav_work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        save_dir_path = Path(save_dir)
        if not save_dir_path.is_absolute():
            save_dir_path = (work_dir / save_dir_path).resolve()
        save_dir_path.mkdir(parents=True, exist_ok=True)
        self._adapter.save_dir = str(save_dir_path)
        self._load_debug['work_dir'] = str(work_dir)
        self._load_debug['save_dir'] = str(save_dir_path)

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

        external_python = ''
        for env_name in INTERNNAV_MODEL_PYTHON_ENV_VARS:
            value = os.environ.get(env_name, '').strip()
            if value:
                external_python = value
                self._load_debug['model_python'] = value
                self._load_debug['model_python_source'] = env_name
                break
        if external_python and Path(external_python).resolve() != Path(sys.executable).resolve():
            try:
                self._subprocess_adapter = InternNavSubprocessAdapter(
                    external_python,
                    model_path,
                    requested_device,
                    self._params,
                    root,
                    logger=self._logger,
                )
                self._mode = 'internnav_subprocess_agent'
                self._fallback_reason = ''
                self._load_error = ''
                self._load_debug.update({'load_mode': 'real_subprocess'})
                self._log('info', f"InternNav wrapper loaded subprocess backend via '{external_python}'")
                return
            except Exception as exc:
                self._subprocess_adapter = None
                self._load_debug['subprocess_load_exception_type'] = type(exc).__name__
                self._load_debug['subprocess_load_error'] = str(exc)
                self._log('error', f"InternNav subprocess backend failed via '{external_python}': {exc}")
                if self._require_real_backend:
                    raise RuntimeError(
                        f"InternNav required subprocess backend failed via '{external_python}': {exc}"
                    ) from exc

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
                resize_w=int(self._params.get('internnav_resize_w', os.environ.get('ARENA_INTERNNAV_RESIZE_W', 336))),
                resize_h=int(self._params.get('internnav_resize_h', os.environ.get('ARENA_INTERNNAV_RESIZE_H', 336))),
                num_history=int(self._params.get('internnav_num_history', os.environ.get('ARENA_INTERNNAV_NUM_HISTORY', 0))),
                plan_step_gap=int(self._params.get('internnav_plan_step_gap', os.environ.get('ARENA_INTERNNAV_PLAN_STEP_GAP', 12))),
            )
            work_dir = self._internnav_work_dir()
            work_dir.mkdir(parents=True, exist_ok=True)
            previous_cwd = Path.cwd()
            try:
                os.chdir(work_dir)
                self._adapter = agent_cls(args)
            finally:
                os.chdir(previous_cwd)
            self._normalize_adapter_save_dir()
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
            self._subprocess_adapter = None
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
        if self._subprocess_adapter is not None:
            try:
                self._subprocess_adapter.reset()
            except Exception as exc:
                self._log('warn', f'InternNav subprocess reset failed: {exc}')
        if self._adapter is not None and hasattr(self._adapter, 'reset'):
            try:
                self._adapter.reset()
                self._normalize_adapter_save_dir()
            except TypeError:
                try:
                    self._adapter.reset(None)
                    self._normalize_adapter_save_dir()
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
                'real_backend_loading': bool(self._loading),
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
        output_policy = str(self._params.get('model_output_policy', 'trajectory')).strip().lower()
        if action_tail and output_policy in {'discrete', 'raw', 'legacy'}:
            self._pending_actions.extend([int(item) for item in action_tail])
            response['debug']['remaining_action_queue'] = len(self._pending_actions)
        elif action_tail:
            response['debug']['ignored_action_sequence_tail'] = [int(item) for item in action_tail]
            response['debug']['ignored_action_sequence_tail_reason'] = 'model_output_policy=trajectory'
        return response

    def _subprocess_response(self, observation: Any) -> dict[str, Any]:
        if self._subprocess_adapter is None:
            return self._unavailable_response()
        response = self._subprocess_adapter.compute(observation)
        response.setdefault('debug', {})
        response['debug'].update({'shim_mode': self._mode, **self._load_debug})
        return response

    def compute(self, observation: Any) -> dict[str, Any]:
        self._reset_if_needed(observation)

        queued = self._queued_action_response()
        if queued is not None:
            return queued

        if self._subprocess_adapter is not None:
            return self._subprocess_response(observation)
        if self._adapter is None:
            if self._mode == 'mock':
                return self._mock_response(observation)
            return self._unavailable_response()
        return self._real_response(observation)

    def predict(self, observation: Any) -> dict[str, Any]:
        return self.compute(observation)


def load_internnav_adapter(logger=None, params: Optional[dict[str, Any]] = None) -> InternNavAdapter:
    return InternNavAdapter(logger=logger, params=params)


def load_internvla_realworld_http_adapter(logger=None, params: Optional[dict[str, Any]] = None) -> InternVLARealworldHttpAdapter:
    return InternVLARealworldHttpAdapter(logger=logger, params=params)


def available_backends() -> list[dict[str, Any]]:
    http_capability = BackendCapability(
        name='internvla_n1_realworld_http',
        required_inputs=INTERNNAV_REALWORLD_CAPABILITY.required_inputs,
        output_modes=INTERNNAV_REALWORLD_CAPABILITY.output_modes,
        supports_batch=False,
        notes='Posts Arena RGB/depth/instruction/pose observations to InternNav realworld HTTP /eval_dual.',
    )
    return [
        to_jsonable(INTERNNAV_REALWORLD_CAPABILITY.__dict__),
        to_jsonable(http_capability.__dict__),
    ]
