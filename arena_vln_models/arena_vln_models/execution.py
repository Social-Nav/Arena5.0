"""Pure execution core shared by the ROS executor and fixture tests.

No ROS imports.  The core decides whether an observation is eligible, calls the
adapter, and maps adapter errors to explicit terminal/hold decisions.  Only the
ROS executor turns a ``follow_path`` decision into a FollowPath goal.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from .core import ModelObservation, ModelPrediction
from .model_adapter import (
    DEFAULT_CONTROL_MODE,
    CapabilityMismatchError,
    EmptyModelOutputError,
    ModelAdapterError,
    ModelProtocolError,
    ModelTimeoutError,
    validate_odom_path,
)

SUPPORTED_CONTROL_MODES = frozenset({DEFAULT_CONTROL_MODE})


@dataclass
class ExecutionConfig:
    model: str = ''
    control_mode: str = DEFAULT_CONTROL_MODE
    inference_hz: float = 4.0
    request_timeout_sec: float = 10.0
    observation_max_age_sec: float = 2.0
    max_segment_m: float = 5.0
    max_path_points: int = 4096
    min_path_interval_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.control_mode not in SUPPORTED_CONTROL_MODES:
            raise ValueError(
                f"unsupported unified control_mode '{self.control_mode}'; "
                f"supported: {sorted(SUPPORTED_CONTROL_MODES)}. Native velocity "
                "belongs to the isolated legacy entry."
            )
        if not (self.inference_hz > 0 and self.request_timeout_sec > 0 and self.observation_max_age_sec > 0):
            raise ValueError('inference_hz, request_timeout_sec and observation_max_age_sec must be positive')


@dataclass
class ExecutionDecision:
    action: str
    status: str
    reason: str = ''
    poses: list[tuple[float, float, float]] = field(default_factory=list)
    prediction: Optional[ModelPrediction] = None
    error_type: str = ''
    terminal: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'status': self.status,
            'reason': self.reason,
            'pose_count': len(self.poses),
            'error_type': self.error_type,
            'terminal': bool(self.terminal),
            'prediction': self.prediction.to_manifest() if self.prediction is not None else None,
            'debug': self.debug,
        }


@dataclass
class SessionTransition:
    """Gate a new session until the old goal and inference are quiescent.

    The executor must not reset the GoalStateMachine or the adapter while an
    old goal may still execute or an old inference is still using the shared
    transport/history.  A pending identity is recorded and activated only after
    the worker is idle and the goal state machine is back to ``idle``.
    """

    pending: Optional[tuple[str, int]] = None

    def request(self, identity: tuple[str, int]) -> tuple[str, int]:
        if self.pending is None:
            self.pending = (str(identity[0]), int(identity[1]))
        return self.pending

    def can_complete(self, *, worker_idle: bool, goals_state: str) -> bool:
        return self.pending is not None and bool(worker_idle) and goals_state == 'idle'

    def complete(self) -> Optional[tuple[str, int]]:
        identity = self.pending
        self.pending = None
        return identity


class ExecutionCore:
    """Episode-aware, ROS-free adapter execution."""

    def __init__(self, adapter: Any, config: Optional[ExecutionConfig] = None) -> None:
        self.adapter = adapter
        self.config = config or ExecutionConfig(model=getattr(adapter, 'name', ''))
        self.session_id = ''
        self.episode = -1
        self.released = False
        self.terminal = False
        self.status = 'hold'
        self.reason = 'waiting_for_session'
        self.instruction = ''
        self.last_infer_at = float('-inf')
        self.last_decision: Optional[ExecutionDecision] = None
        self.last_observation_stamp = -1

    def reset(self, session_id: str, episode: int) -> None:
        self.session_id = str(session_id)
        self.episode = int(episode)
        self.released = False
        self.terminal = False
        self.status = 'hold'
        self.reason = 'waiting_for_release'
        self.instruction = ''
        self.last_infer_at = float('-inf')
        self.last_decision = None
        self.last_observation_stamp = -1
        reset = getattr(self.adapter, 'reset', None)
        if callable(reset):
            reset(session_id=self.session_id, episode=self.episode)

    def release(self) -> None:
        if self.terminal:
            return
        self.released = True
        self.status = 'ready'
        self.reason = ''

    def set_instruction(self, instruction: str) -> None:
        self.instruction = str(instruction or '')

    def set_hold(self, reason: str) -> ExecutionDecision:
        self.status = 'hold'
        self.reason = str(reason)
        decision = ExecutionDecision(action='hold', status='hold', reason=str(reason))
        self.last_decision = decision
        return decision

    def set_stop(self, reason: str = 'model_stop') -> ExecutionDecision:
        self.terminal = True
        self.status = 'stop'
        self.reason = str(reason)
        decision = ExecutionDecision(action='stop', status='stop', reason=str(reason), terminal=True)
        self.last_decision = decision
        return decision

    def set_error(self, reason: str, *, error_type: str = 'error') -> ExecutionDecision:
        self.terminal = True
        self.status = 'error'
        self.reason = str(reason)
        decision = ExecutionDecision(
            action='error', status='error', reason=str(reason), error_type=error_type, terminal=True
        )
        self.last_decision = decision
        return decision

    def eligible(self, observation: Optional[ModelObservation], now: float) -> tuple[bool, str]:
        if self.terminal:
            return False, 'terminal'
        if not self.released:
            return False, 'not_released'
        if not self.instruction.strip():
            return False, 'waiting_for_instruction'
        if observation is None:
            return False, 'waiting_for_observation'
        stamp = int(getattr(observation, 'stamp_ns', 0) or 0)
        if stamp <= 0:
            return False, 'invalid_observation_stamp'
        received = float(getattr(observation, 'received_at', 0.0) or 0.0)
        if received <= 0.0 or now - received > float(self.config.observation_max_age_sec):
            return False, 'stale_observation'
        if stamp == self.last_observation_stamp:
            return False, 'observation_not_new'
        if now - self.last_infer_at < 1.0 / float(self.config.inference_hz):
            return False, 'rate_limited'
        return True, ''

    def plan(self, observation: ModelObservation, now: float) -> ExecutionDecision:
        """Run the adapter and return a decision without mutating core state.

        Safe to call from the worker thread: the main thread commits the
        decision only after generation/session/episode still match.
        """

        eligible, reason = self.eligible(observation, now)
        if not eligible:
            if reason in {'waiting_for_observation', 'stale_observation', 'waiting_for_instruction'}:
                return ExecutionDecision(action='hold', status='hold', reason=reason)
            return ExecutionDecision(action='skip', status=self.status, reason=reason)
        try:
            raw = self.adapter.infer(observation)
            prediction = self.adapter.to_prediction(raw, observation)
            if prediction.stop:
                return ExecutionDecision(
                    action='stop', status='stop', reason='model_stop', terminal=True,
                    prediction=prediction, debug=prediction.debug,
                )
            if not prediction.poses:
                return ExecutionDecision(
                    action='error', status='error', reason='empty_path',
                    error_type='empty_model_output', terminal=True,
                )
            poses = validate_odom_path(
                prediction.poses,
                observation,
                max_segment_m=float(self.config.max_segment_m),
                max_points=int(self.config.max_path_points),
            )
            return ExecutionDecision(
                action='follow_path',
                status='running',
                reason='',
                poses=poses,
                prediction=prediction,
                debug=prediction.debug,
            )
        except ModelTimeoutError as exc:
            return ExecutionDecision(action='error', status='error', reason=str(exc), error_type='timeout', terminal=True)
        except CapabilityMismatchError as exc:
            return ExecutionDecision(action='error', status='error', reason=str(exc), error_type='capability_mismatch', terminal=True)
        except EmptyModelOutputError as exc:
            return ExecutionDecision(action='error', status='error', reason=str(exc), error_type='empty_model_output', terminal=True)
        except ModelProtocolError as exc:
            return ExecutionDecision(action='error', status='error', reason=str(exc), error_type='protocol', terminal=True)
        except ModelAdapterError as exc:
            return ExecutionDecision(action='error', status='error', reason=str(exc), error_type='adapter_error', terminal=True)
        except Exception as exc:  # defensive: never let a model exception escape
            return ExecutionDecision(action='error', status='error', reason=f'{type(exc).__name__}: {exc}', error_type='unexpected', terminal=True)

    def commit(self, decision: ExecutionDecision, observation: ModelObservation, now: float) -> ExecutionDecision:
        """Apply a still-current decision to the shared core (main thread only)."""

        if decision.action in {'follow_path', 'stop', 'error'}:
            self.last_infer_at = float(now)
            self.last_observation_stamp = int(getattr(observation, 'stamp_ns', 0) or 0)
        if decision.action == 'follow_path':
            self.status = 'running'
            self.reason = ''
        elif decision.action == 'stop':
            self.terminal = True
            self.status = 'stop'
            self.reason = decision.reason or 'model_stop'
        elif decision.action == 'error':
            self.terminal = True
            self.status = 'error'
            self.reason = decision.reason
        elif decision.action == 'hold':
            self.status = 'hold'
            self.reason = decision.reason
        self.last_decision = decision
        return decision

    def step(self, observation: ModelObservation, now: Optional[float] = None) -> ExecutionDecision:
        now = time.monotonic() if now is None else float(now)
        return self.commit(self.plan(observation, now), observation, now)


class AsyncInferenceWorker:
    """One-in-flight inference worker with generation-based result discard.

    ROS callbacks must never block on model inference.  The executor submits a
    job from a timer, harvests completed futures from another timer, and can
    invalidate all outstanding results on reset/STOP/error.  A generation token
    prevents an old blocked inference from mutating the new episode.
    """

    def __init__(self, core: ExecutionCore, *, max_workers: int = 1) -> None:
        self.core = core
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='arena-model')
        self._lock = threading.Lock()
        self._future: Optional[Future] = None
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def in_flight(self) -> bool:
        with self._lock:
            return self._future is not None and not self._future.done()

    def submit(self, observation: ModelObservation, now: float) -> bool:
        with self._lock:
            if self._future is not None and not self._future.done():
                return False
            generation = self._generation
            session_id = self.core.session_id
            episode = self.core.episode
            self._future = self._executor.submit(
                self._run, generation, session_id, episode, observation, now
            )
            return True

    def _run(self, generation, session_id, episode, observation, now):
        # Pure: never mutate the shared core from this thread.
        decision = self.core.plan(observation, float(now))
        return generation, session_id, episode, observation, now, decision

    def harvest(self) -> Optional[ExecutionDecision]:
        with self._lock:
            future = self._future
            if future is None or not future.done():
                return None
            self._future = None
            generation = self._generation
        try:
            result_generation, session_id, episode, observation, now, decision = future.result()
        except Exception as exc:
            return ExecutionDecision(
                action='error', status='error', reason=f'{type(exc).__name__}: {exc}',
                error_type='unexpected', terminal=True,
            )
        if (
            result_generation != generation
            or session_id != self.core.session_id
            or episode != self.core.episode
        ):
            return ExecutionDecision(action='skip', status=self.core.status, reason='stale_inference')
        return self.core.commit(decision, observation, now)

    def wait_idle(self, timeout: float = 1.0) -> None:
        with self._lock:
            future = self._future
        if future is not None:
            try:
                future.result(timeout=float(timeout))
            except Exception:
                pass

    def invalidate(self) -> None:
        """Discard any current/future result without killing the worker thread."""

        with self._lock:
            self._generation += 1
            future = self._future
        if future is not None and future.cancel():
            with self._lock:
                if self._future is future:
                    self._future = None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
