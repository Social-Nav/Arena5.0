"""Shared, ROS-free base class for unified model adapters."""
from __future__ import annotations

import time
from typing import Any, Optional

from ..core import ModelObservation, ModelPrediction
from ..model_adapter import (
    CapabilityMismatchError,
    ModelAdapterError,
    ModelCapability,
    ModelProtocolError,
    ModelRawOutput,
    ModelTimeoutError,
    prediction_from_raw,
    validate_observation,
)


class BaseModelAdapter:
    """Common lifecycle, logging and error mapping for all adapters."""

    name = 'base'
    capability = ModelCapability(name='base', raw_output_kind='none')

    def __init__(
        self,
        *,
        params: Optional[dict[str, Any]] = None,
        logger: Any = None,
        transport: Any = None,
    ) -> None:
        self._params = dict(params or {})
        self._logger = logger
        self._transport = transport
        self._session_id = ''
        self._episode = -1
        self._request_id = 0
        self._last_infer_at = 0.0
        self._last_error = ''

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    def _log(self, level: str, message: str) -> None:
        logger = self._logger
        if logger is None:
            return
        try:
            getattr(logger, level, logger.info)(message)
        except Exception:
            return

    def _timeout(self) -> float:
        try:
            value = float(self._params.get('request_timeout_sec', self._params.get('inference_timeout_sec', 10.0)))
        except (TypeError, ValueError):
            value = 10.0
        return value if value > 0 else 10.0

    def _validate(self, observation: ModelObservation) -> None:
        validate_observation(observation, self.capability)

    def reset(self, *, session_id: str = '', episode: int = -1) -> None:
        self._session_id = str(session_id or '')
        self._episode = int(episode)
        self._request_id = 0
        self._last_infer_at = 0.0
        self._last_error = ''

    def infer(self, observation: ModelObservation) -> ModelRawOutput:
        raise NotImplementedError

    def to_prediction(self, raw: ModelRawOutput, observation: ModelObservation) -> ModelPrediction:
        return prediction_from_raw(raw, observation, adapter_name=self.name)

    def _timed(self, operation):
        start = time.monotonic()
        try:
            result = operation()
        except ModelAdapterError:
            raise
        except TimeoutError as exc:
            raise ModelTimeoutError(str(exc) or 'model request timed out') from exc
        except Exception as exc:
            raise ModelProtocolError(str(exc)) from exc
        elapsed = time.monotonic() - start
        if elapsed > self._timeout():
            raise ModelTimeoutError(
                f'model request took {elapsed:.3f}s, timeout is {self._timeout():.3f}s'
            )
        self._last_infer_at = elapsed
        return result, elapsed

    def _error_raw(
        self,
        *,
        status: str,
        reason: str,
        raw: Any = None,
        debug: Optional[dict[str, Any]] = None,
    ) -> ModelRawOutput:
        self._last_error = str(reason)
        return ModelRawOutput(
            adapter=self.name,
            raw=raw,
            kind='error',
            stop=status == 'stop',
            status=status,
            source='native',
            modality='stop' if status == 'stop' else 'error',
            debug={**(debug or {}), 'reason': str(reason)},
        )

    def capability_manifest(self) -> dict[str, Any]:
        return {
            'name': self.capability.name,
            'raw_output_kind': self.capability.raw_output_kind,
            'required_inputs': list(self.capability.required_inputs),
            'output_modes': list(self.capability.output_modes),
            'supports_stop': self.capability.supports_stop,
            'notes': self.capability.notes,
        }


__all__ = ['BaseModelAdapter', 'CapabilityMismatchError']
