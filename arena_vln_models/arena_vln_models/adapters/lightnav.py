"""LightNav raw-output adapter.

The original service returns a JSON object
``{"action": "next", "data": {"rc": 0, "seq": N, "stop": bool,
"actions": {"actions": [[forward_m, lateral_m, yaw_rad], ...]}}}`` over a
WebSocket.  The actions are cumulative ego poses at the image timestamp.  This
adapter validates the full response, retains every finite waypoint, and projects
the path into fixed odom; it never publishes velocity.
"""
from __future__ import annotations

import base64
import io
import json
import math
from typing import Any, Optional

import numpy as np

from ..core import ModelObservation, ModelPrediction
from ..model_adapter import (
    ModelCapability,
    ModelProtocolError,
    ModelRawOutput,
    prediction_from_raw,
    raw_response_hash,
    register_adapter,
)
from .base import BaseModelAdapter

LIGHTNAV_CAPABILITY = ModelCapability(
    name='lightnav',
    raw_output_kind='WebSocket JSON -> cumulative [forward_m, lateral_m, yaw_rad]',
    required_inputs=('rgb', 'instruction'),
    output_modes=('path', 'stop'),
    supports_stop=True,
    notes='LightNav service owns image history; Arena sends the latest image and instruction.',
)


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelProtocolError(f'{field_name} must be a finite number, got {value!r}')
    number = float(value)
    if not math.isfinite(number):
        raise ModelProtocolError(f'{field_name} is not finite: {value!r}')
    return number


def parse_prediction(response: Any, expected_seq: int) -> dict[str, Any]:
    """Validate the LightNav served protocol before geometry conversion."""

    if not isinstance(response, dict) or response.get('action') != 'next':
        raise ModelProtocolError('LightNav prediction response action must be next')
    data = response.get('data')
    if not isinstance(data, dict):
        raise ModelProtocolError('LightNav prediction data must be an object')
    rc = data.get('rc')
    if isinstance(rc, bool) or not isinstance(rc, int) or rc != 0:
        raise ModelProtocolError(f'LightNav prediction failed: {data!r}')
    seq = data.get('seq')
    if isinstance(seq, bool) or not isinstance(seq, int) or seq != expected_seq:
        raise ModelProtocolError(f'LightNav prediction sequence mismatch: expected {expected_seq}, got {seq!r}')
    stop = data.get('stop')
    if not isinstance(stop, bool):
        raise ModelProtocolError('LightNav prediction data.stop must be a boolean')
    actions = data.get('actions')
    if actions is None and stop:
        points: list[Any] = []
    else:
        if not isinstance(actions, dict) or 'actions' not in actions:
            raise ModelProtocolError('LightNav prediction data.actions must contain an actions array')
        points = actions['actions']
        if points is None and stop:
            points = []
        if not isinstance(points, list):
            raise ModelProtocolError('LightNav actions.actions must be an array of three-column poses')
        for index, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ModelProtocolError(f'LightNav waypoint {index} must have exactly three columns')
            for column, value in enumerate(point):
                _finite_number(value, f'waypoint[{index}][{column}]')
    if not stop and not points:
        raise ModelProtocolError('non-STOP LightNav prediction has no waypoints')
    return {
        **response,
        'stop': stop,
        'waypoints': [tuple(float(value) for value in point) for point in points],
    }


def websocket_transport_available() -> bool:
    """Check the approved runtime dependency at startup, not mid-episode."""

    try:
        from websockets.sync.client import connect  # noqa: F401
        return True
    except Exception:
        return False


class _WebSocketTransport:
    """Minimal synchronous WebSocket transport using the Arena runtime client."""

    def __init__(self, server_url: str, timeout: float) -> None:
        self.server_url = str(server_url)
        self.timeout = float(timeout)
        self._connection = None
        if not websocket_transport_available():
            raise ModelProtocolError(
                'LightNav WebSocket transport requires the Arena runtime websockets package; '
                'the approved arena:dev image provides websockets 15.x. Inject a transport for fixtures.'
            )

    def _connect(self):
        if self._connection is not None:
            return self._connection
        from websockets.sync.client import connect as ws_connect

        self._connection = ws_connect(
            self.server_url,
            open_timeout=min(3.0, self.timeout),
            close_timeout=min(1.0, self.timeout),
        )
        for action, data in (('login', {'clientId': 'arena_lightnav'}), ('reset', {})):
            self._connection.send(json.dumps({'action': action, 'data': data}))
            response = json.loads(self._connection.recv(timeout=self.timeout))
            if response.get('action') != action or (response.get('data') or {}).get('rc') != 0:
                raise ModelProtocolError(f'invalid LightNav {action} response: {response!r}')
        return self._connection

    def reset(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    def generate(self, *, seq: int, image: str, instruction: str) -> dict[str, Any]:
        connection = self._connect()
        connection.send(json.dumps({
            'action': 'next',
            'data': {'seq': seq, 'image': image, 'instruction': instruction},
        }))
        response = json.loads(connection.recv(timeout=self.timeout))
        if not isinstance(response, dict):
            raise ModelProtocolError('LightNav response must be a JSON object')
        return response


@register_adapter('lightnav', 'light_nav')
class LightNavAdapter(BaseModelAdapter):
    name = 'lightnav'
    capability = LIGHTNAV_CAPABILITY

    def __init__(self, *, params: Optional[dict[str, Any]] = None, logger: Any = None, transport: Any = None) -> None:
        super().__init__(params=params, logger=logger, transport=transport)
        self._server_url = str(self._params.get('server_url', 'ws://127.0.0.1:8050') or 'ws://127.0.0.1:8050')
        self._seq = 0
        # Keep construction side-effect free for fixtures/registry.  doctor()
        # is the explicit startup dependency check; _generate() creates the
        # default transport lazily and still fails closed if it is absent.

    def doctor(self) -> dict[str, Any]:
        """Fail before launch when the approved runtime dependency is absent."""

        if self._transport is None and not websocket_transport_available():
            raise ModelProtocolError(
                'LightNav requires the Arena runtime websockets package; '
                'the approved arena:dev image provides websockets 15.x.'
            )
        return {
            'adapter': self.name,
            'transport': 'injected' if self._transport is not None else 'websockets.sync',
            'available': True,
        }

    def reset(self, *, session_id: str = '', episode: int = -1) -> None:
        super().reset(session_id=session_id, episode=episode)
        self._seq = 0
        reset = getattr(self._transport, 'reset', None)
        if callable(reset):
            reset()

    @staticmethod
    def _encode_rgb(rgb: Any) -> str:
        from PIL import Image as PILImage

        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ModelProtocolError('LightNav requires an HxWx3 RGB image')
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        PILImage.fromarray(array[:, :, :3]).convert('RGB').save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('ascii')

    def _generate(self, *, seq: int, image: str, instruction: str) -> dict[str, Any]:
        transport = self._transport
        if transport is None:
            transport = _WebSocketTransport(self._server_url, self._timeout())
            # Persist the owned transport so connect/login happen once per
            # episode and reset()/close() remain reachable.
            self._transport = transport
        generate = getattr(transport, 'generate', None)
        if callable(generate):
            return generate(seq=seq, image=image, instruction=instruction)
        try:
            return transport(seq=seq, image=image, instruction=instruction)
        except TypeError:
            return transport(seq, image, instruction)

    def infer(self, observation: ModelObservation) -> ModelRawOutput:
        self._validate(observation)
        self._seq += 1
        encoded = self._encode_rgb(getattr(observation, 'rgb_image', None))
        raw, elapsed = self._timed(lambda: self._generate(
            seq=self._seq,
            image=encoded,
            instruction=str(getattr(observation, 'instruction', '')),
        ))
        prediction = parse_prediction(raw, self._seq)
        debug = {
            'server': self._server_url,
            'seq': self._seq,
            'elapsed_sec': elapsed,
            'waypoint_count': len(prediction['waypoints']),
            'raw_preview': {
                'action': raw.get('action'),
                'data_keys': sorted((raw.get('data') or {}).keys()),
            },
        }
        return ModelRawOutput(
            adapter=self.name,
            raw=raw,
            kind='cumulative_waypoints',
            text=str(raw.get('data', {}).get('message', '') or ''),
            local_waypoints=prediction['waypoints'],
            stop=bool(prediction['stop']),
            status='stop' if prediction['stop'] else 'running',
            source='native',
            modality='waypoints' if not prediction['stop'] else 'stop',
            raw_hash=raw_response_hash(raw),
            debug=debug,
        )

    def to_prediction(self, raw: ModelRawOutput, observation: ModelObservation) -> ModelPrediction:
        return prediction_from_raw(raw, observation, adapter_name=self.name)
