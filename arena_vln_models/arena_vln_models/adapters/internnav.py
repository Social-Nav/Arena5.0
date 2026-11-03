"""InternNav / InternVLA-N1 raw-output adapter.

The original inference return is ``InternVLAN1AsyncAgent.step(...) ->
S2Output``.  Its native path is ``output_trajectory``: cumulative local
``[T+1, 2]`` metres after ``/4.0``.  This adapter preserves that output, labels
wrapper-synthesised trajectories as ``synthetic``, and expands symbolic action
ids only when no native trajectory exists.
"""
from __future__ import annotations

import io
import json
import math
from typing import Any, Optional

import numpy as np

from ..core import (
    ModelObservation,
    camera_intrinsic_matrix,
    depth_debug,
    normalize_depth,
    normalize_rgb,
    pose_vector,
    to_jsonable,
)
from ..model_adapter import (
    CapabilityMismatchError,
    ModelCapability,
    ModelProtocolError,
    ModelRawOutput,
    coerce_local_waypoints,
    local_to_odom,
    raw_response_hash,
    register_adapter,
    validate_odom_path,
    wrap_angle,
)
from .base import BaseModelAdapter

INTERNNAV_CAPABILITY = ModelCapability(
    name='internnav',
    raw_output_kind='S2Output.output_trajectory | output_action | output_pixel',
    required_inputs=('rgb', 'depth', 'instruction', 'camera_intrinsics'),
    output_modes=('path', 'stop'),
    supports_stop=True,
    notes=(
        'InternNav native trajectory is local metres with no yaw column; '
        'segment heading is derived. Symbolic wrapper trajectories are synthetic.'
    ),
)

ACTION_FORWARD_M = 0.25
ACTION_TURN_RAD = math.radians(15.0)


def _action_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return [int(value)]
    try:
        rows = np.asarray(value).reshape(-1).tolist()
    except Exception as exc:
        raise ModelProtocolError(f'discrete action is not a sequence: {value!r}') from exc
    result: list[int] = []
    for item in rows:
        if isinstance(item, bool):
            raise ModelProtocolError(f'discrete action contains a boolean: {value!r}')
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ModelProtocolError(f'discrete action contains a non-integer: {value!r}') from exc
    return result


def expand_actions(actions: list[int]) -> tuple[list[tuple[float, float, float]], bool]:
    """Expand InternNav symbolic ids into a cumulative local path.

    ids: 0 STOP, 1 forward, 2 turn left, 3 turn right, 5 look-down.
    Returns ``(waypoints, stop)``.  A leading STOP is terminal; look-down is a
    no-motion diagnostic and is skipped.
    """

    if not actions:
        raise ModelProtocolError('empty discrete action sequence')
    if actions[0] == 0:
        return [], True
    x = y = yaw = 0.0
    waypoints: list[tuple[float, float, float]] = []
    for action in actions:
        if action == 0:
            return waypoints, True
        if action == 1:
            x += ACTION_FORWARD_M * math.cos(yaw)
            y += ACTION_FORWARD_M * math.sin(yaw)
            waypoints.append((x, y, wrap_angle(yaw)))
        elif action == 2:
            yaw = wrap_angle(yaw + ACTION_TURN_RAD)
            waypoints.append((x, y, yaw))
        elif action == 3:
            yaw = wrap_angle(yaw - ACTION_TURN_RAD)
            waypoints.append((x, y, yaw))
        elif action == 5:
            continue
        else:
            raise ModelProtocolError(f'unsupported InternNav action id {action}')
    if not waypoints:
        raise ModelProtocolError(f'action sequence produced no motion: {actions}')
    return waypoints, False


@register_adapter('internnav', 'internvla', 'internvla_n1')
class InternNavAdapter(BaseModelAdapter):
    name = 'internnav'
    capability = INTERNNAV_CAPABILITY

    def __init__(self, *, params: Optional[dict[str, Any]] = None, logger: Any = None, transport: Any = None) -> None:
        super().__init__(params=params, logger=logger, transport=transport)
        self._url = str(
            self._params.get('server_url')
            or self._params.get('internnav_http_url')
            or 'http://127.0.0.1:5801/eval_dual'
        ).strip()
        self._session_key: Optional[tuple[str, int]] = None

    def _reset_required(self, observation: ModelObservation) -> bool:
        key = (str(observation.session_id), int(observation.episode))
        if key == self._session_key:
            return False
        self._session_key = key
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

    def _post(self, payload: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            response = self._transport(payload=payload, files=files)
            if not isinstance(response, dict):
                raise ModelProtocolError(f'InternNav transport returned {type(response).__name__}, expected dict')
            return response
        # Reuse the proven legacy multipart implementation; it has a urllib
        # fallback when requests is absent.
        from ..internnav import InternVLARealworldHttpAdapter

        return InternVLARealworldHttpAdapter._post_eval_dual(
            self._url, files=files, payload=payload, timeout=self._timeout()
        )

    def infer(self, observation: ModelObservation) -> ModelRawOutput:
        self._validate(observation)
        rgb = normalize_rgb(getattr(observation, 'rgb_image', None))
        depth = normalize_depth(
            getattr(observation, 'depth_image', None),
            reference_shape=(rgb.shape[0], rgb.shape[1]) if rgb is not None else None,
        )
        if rgb is None or depth is None:
            raise ModelProtocolError('InternNav requires RGB and depth images')
        self._request_id += 1
        reset = self._reset_required(observation)
        rgb_bytes, depth_bytes = self._image_payloads(rgb, depth)
        payload = {
            'reset': reset,
            'idx': self._request_id - 1,
            'request_id': self._request_id,
            'instruction': str(getattr(observation, 'instruction', '')),
            'pose': to_jsonable(pose_vector(observation)),
            'camera_pose': _pose_matrix(observation),
            'intrinsic': to_jsonable(camera_intrinsic_matrix(observation, rgb)),
            'look_down': bool(getattr(observation, 'look_down', False)),
            'model_output_policy': str(self._params.get('model_output_policy', 'trajectory')),
            'client': 'arena_unified_model_adapter',
        }
        files = {
            'image': ('rgb_image.jpg', rgb_bytes, 'image/jpeg'),
            'depth': ('depth_image.png', depth_bytes, 'image/png'),
        }
        raw, elapsed = self._timed(lambda: self._post(payload, files))
        output = self._decode(raw, elapsed=elapsed, depth=depth, reset=reset, observation=observation)
        return output

    def _decode(
        self,
        raw: dict[str, Any],
        *,
        elapsed: float,
        depth: np.ndarray,
        reset: bool,
        observation: ModelObservation,
    ) -> ModelRawOutput:
        debug: dict[str, Any] = {
            'http_url': self._url,
            'http_request_id': self._request_id,
            'http_reset': reset,
            'http_elapsed_sec': elapsed,
            'realworld_http_client': True,
            **depth_debug(depth),
        }
        status = str(raw.get('status', 'internnav_command'))
        if status in {'error', 'timeout', 'internnav_http_request_failed'}:
            return self._error_raw(status='timeout' if status == 'timeout' else 'error', reason=str(raw.get('error', status)), raw=raw, debug=debug)
        raw_debug = raw.get('debug') if isinstance(raw.get('debug'), dict) else {}
        debug.update(raw_debug)
        trajectory = raw.get('output_trajectory', raw.get('trajectory'))
        discrete = raw.get('output_action', raw.get('discrete_action', raw.get('action')))
        pixel = raw.get('output_pixel', raw.get('target_pixel', raw.get('pixel_goal')))
        synthesized = bool(raw_debug.get('trajectory_synthesized_from_discrete_action', False))
        actions = _action_ids(discrete)
        stop = bool(actions and actions[0] == 0) or status in {'stop', 'model_stop'} or bool(raw.get('stop', False))
        if trajectory is not None:
            local = coerce_local_waypoints(trajectory, max_segment_m=float(self._params.get('max_segment_m', 5.0)))
            source = 'synthetic' if synthesized else 'native'
            if synthesized:
                debug['synthetic_source'] = 'internnav_symbolic_trajectory'
                debug['synthetic_reason'] = raw_debug.get(
                    'trajectory_synthesis_reason', 'symbolic action without native trajectory'
                )
            return ModelRawOutput(
                adapter=self.name,
                raw=raw,
                kind='local_trajectory',
                text=str(raw_debug.get('llm_output', raw.get('raw_output_text', '')) or ''),
                local_waypoints=local,
                stop=stop,
                status='stop' if stop else 'running',
                source=source,
                modality='trajectory',
                raw_hash=raw_response_hash(raw),
                debug={**debug, 'output_pixel': to_jsonable(pixel) if pixel is not None else None},
            )
        if actions:
            local, action_stop = expand_actions(actions)
            return ModelRawOutput(
                adapter=self.name,
                raw=raw,
                kind='primitive_actions',
                text=str(raw_debug.get('llm_output', '') or ''),
                local_waypoints=local,
                stop=stop or action_stop,
                status='stop' if (stop or action_stop) else 'running',
                source='primitive',
                modality='primitive',
                raw_hash=raw_response_hash(raw),
                debug={**debug, 'actions': actions},
            )
        if pixel is not None:
            # No verified camera optical-axis/depth projection contract exists in
            # the current source.  Refuse pixel-only mode explicitly before any
            # runtime guess instead of leaving an undefined projection symbol.
            raise CapabilityMismatchError(
                'InternNav pixel-only output is not a supported capability: no verified '
                'camera optical-axis/depth projection contract; refusing to guess depth'
            )
        raise ModelProtocolError('InternNav returned no trajectory, action or pixel goal')

    def to_prediction(self, raw: ModelRawOutput, observation: ModelObservation):
        from ..model_adapter import prediction_from_raw

        return prediction_from_raw(raw, observation, adapter_name=self.name)


def _pose_matrix(observation: Any) -> list[list[float]]:
    pose = pose_vector(observation)
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        [c, -s, 0.0, x],
        [s, c, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
