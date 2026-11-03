"""NaVILA raw-output adapter.

The original model service returns a JSON string such as ``move forward 75 cm``,
``turn left 15 degrees`` or an affirmative STOP over a length-prefixed TCP
socket.  This adapter preserves the strict primitive vocabulary and expands one
primitive into a fixed-odom path; it never emits velocity.
"""
from __future__ import annotations

import json
import math
import re
import socket
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
from .navila_history import FrameHistory

NAVILA_CAPABILITY = ModelCapability(
    name='navila',
    raw_output_kind='TCP JSON string -> primitive text action',
    required_inputs=('rgb', 'instruction'),
    output_modes=('path', 'stop'),
    supports_stop=True,
    notes='NaVILA eight-frame TCP service; forward 25/50/75 cm and turns 15/30/45 degrees.',
)

_STOP_PATTERN = re.compile(
    r'(?:(?:i (?:think i )?should )?stop'
    r'(?: because i have (?:finished|completed) (?:the )?(?:instruction|task))?'
    r'|the task is (?:completed|finished),?\s+stop)',
    re.I,
)
_FORWARD_PATTERN = re.compile(r'move forward\s+(25|50|75)\s+cm', re.I)
_TURN_PATTERN = re.compile(r'turn (left|right)\s+(15|30|45)\s+degrees?', re.I)


def parse_action(text: Any) -> tuple[str, int]:
    """Strict NaVILA primitive parser: (kind, magnitude) or raises."""

    if not isinstance(text, str) or text.lower().startswith('server error:'):
        raise ModelProtocolError(f'NaVILA server error or invalid response: {text!r}')
    value = re.sub(r'^the next action is\s+', '', text.strip(), flags=re.I).strip(' .\n')
    if _STOP_PATTERN.fullmatch(value):
        return 'stop', 0
    match = _FORWARD_PATTERN.fullmatch(value)
    if match:
        return 'forward', int(match.group(1))
    match = _TURN_PATTERN.fullmatch(value)
    if match:
        return match.group(1).lower(), int(match.group(2))
    raise ModelProtocolError(f'unrecognized or conflicting NaVILA action: {text!r}')


class _TcpTransport:
    """Length-prefixed JSON TCP transport used by the NaVILA service."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    @staticmethod
    def _receive(connection, size: int, deadline: float) -> bytes:
        import time

        data = bytearray()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('NaVILA request timed out')
            connection.settimeout(remaining)
            chunk = connection.recv(min(size - len(data), 1048576))
            if not chunk:
                raise ConnectionError('NaVILA disconnected before completing response')
            data.extend(chunk)
        return bytes(data)

    def __call__(self, *, images: list[str], query: str) -> str:
        import time

        payload = json.dumps({'images': images, 'query': query}).encode()
        if len(payload) > 128 * 1024 * 1024:
            raise ModelProtocolError('NaVILA request exceeds the 128 MiB server limit')
        deadline = time.monotonic() + self.timeout
        with socket.create_connection((self.host, self.port), timeout=min(3.0, self.timeout)) as connection:
            connection.settimeout(max(0.001, deadline - time.monotonic()))
            connection.sendall(len(payload).to_bytes(8, 'big') + payload)
            size = int.from_bytes(self._receive(connection, 8, deadline), 'big')
            if not 0 < size <= 65536:
                raise ModelProtocolError('invalid NaVILA response size')
            result = json.loads(self._receive(connection, size, deadline).decode())
        if not isinstance(result, str):
            raise ModelProtocolError(f'NaVILA response must be a JSON string, got {type(result).__name__}')
        if result.lower().startswith('server error:'):
            raise ModelProtocolError(f'NaVILA server error: {result!r}')
        return result


@register_adapter('navila', 'navila_tcp')
class NaVILAAdapter(BaseModelAdapter):
    name = 'navila'
    capability = NAVILA_CAPABILITY

    def __init__(self, *, params: Optional[dict[str, Any]] = None, logger: Any = None, transport: Any = None) -> None:
        super().__init__(params=params, logger=logger, transport=transport)
        self._history = FrameHistory()
        self._host = str(self._params.get('host', '127.0.0.1'))
        self._port = int(self._params.get('port', 54321))
        self._server_url = str(self._params.get('server_url', '') or '')

    def reset(self, *, session_id: str = '', episode: int = -1) -> None:
        super().reset(session_id=session_id, episode=episode)
        self._history.reset()

    @staticmethod
    def _rgb_image(rgb: Any):
        from PIL import Image as PILImage

        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ModelProtocolError('NaVILA requires an HxWx3 RGB image')
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return PILImage.fromarray(array[:, :, :3]).convert('RGB')

    def _transport_call(self, images: list[str], query: str) -> str:
        transport = self._transport
        if transport is None:
            transport = _TcpTransport(self._host, self._port, self._timeout())
        try:
            return transport(images=images, query=query)
        except TypeError:
            return transport(images, query)

    def infer(self, observation: ModelObservation) -> ModelRawOutput:
        self._validate(observation)
        image = self._rgb_image(getattr(observation, 'rgb_image', None))
        self._history.offer(observation.stamp_ns, image)
        frames, stamps = self._history.sample()
        text, elapsed = self._timed(
            lambda: self._transport_call(frames, str(getattr(observation, 'instruction', '')))
        )
        kind, magnitude = parse_action(text)
        self._request_id += 1
        debug = {
            'server': self._server_url or f'tcp://{self._host}:{self._port}',
            'request_id': self._request_id,
            'elapsed_sec': elapsed,
            'history_frames': len(frames),
            'history_stamps': stamps,
            'history_duplicates': self._history.duplicates,
            'primitive_kind': kind,
            'primitive_magnitude': magnitude,
            'raw_text': text,
        }
        if kind == 'stop':
            return ModelRawOutput(
                adapter=self.name, raw=text, kind='text_action', text=text,
                local_waypoints=[], stop=True, status='stop', source='primitive',
                modality='text_stop', raw_hash=raw_response_hash(text), debug=debug,
            )
        if kind == 'forward':
            distance = magnitude / 100.0
            local = [(distance, 0.0, 0.0)]
        elif kind in {'left', 'right'}:
            angle = math.radians(magnitude) * (1.0 if kind == 'left' else -1.0)
            local = [(0.0, 0.0, angle)]
        else:  # pragma: no cover - parse_action already restricts the vocabulary
            raise ModelProtocolError(f'unsupported NaVILA primitive: {kind}')
        return ModelRawOutput(
            adapter=self.name, raw=text, kind='text_action', text=text,
            local_waypoints=local, stop=False, status='running', source='primitive',
            modality='primitive', raw_hash=raw_response_hash(text), debug=debug,
        )

    def to_prediction(self, raw: ModelRawOutput, observation: ModelObservation) -> ModelPrediction:
        return prediction_from_raw(raw, observation, adapter_name=self.name)
