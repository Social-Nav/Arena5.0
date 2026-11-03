"""Behavior tests for stale-inference isolation and cancel serialization."""
from __future__ import annotations

import threading
import time

from arena_vln_models.core import ModelObservation, ModelPrediction, Pose2D
from arena_vln_models.execution import AsyncInferenceWorker, ExecutionConfig, ExecutionCore
from arena_vln_models.goal_state import ERROR, GoalStateMachine
from arena_vln_models.model_adapter import ModelCapability, ModelRawOutput


def _observation(stamp_ns: int = 10) -> ModelObservation:
    return ModelObservation(
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        instruction='go',
        frame_id='odom',
        stamp_ns=stamp_ns,
        received_at=time.monotonic(),
    )


class _BlockingAdapter:
    name = 'blocking'
    capability = ModelCapability(name='blocking', raw_output_kind='fixture')

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def reset(self, **kwargs):
        return None

    def infer(self, observation):
        self.started.set()
        self.release.wait(5.0)
        return ModelRawOutput(
            adapter=self.name,
            local_waypoints=[(1.0, 0.0, 0.0)],
            stop=False,
            source='native',
        )

    def to_prediction(self, raw, observation):
        return ModelPrediction(
            adapter=self.name,
            poses=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            stop=False,
            status='running',
            source='native',
            raw=raw,
            debug={},
        )


def test_stale_inference_does_not_mutate_new_session_core():
    adapter = _BlockingAdapter()
    core = ExecutionCore(adapter, ExecutionConfig(model='blocking'))
    core.reset('old', 0)
    core.release()
    core.set_instruction('go')
    worker = AsyncInferenceWorker(core)
    try:
        assert worker.submit(_observation(stamp_ns=10), time.monotonic()) is True
        assert adapter.started.wait(2.0)

        worker.invalidate()
        core.reset('new', 1)
        assert core.session_id == 'new'
        assert core.last_decision is None
        assert core.last_observation_stamp == -1
        assert core.status != 'running'

        adapter.release.set()
        deadline = time.monotonic() + 2.0
        decision = None
        while time.monotonic() < deadline:
            decision = worker.harvest()
            if decision is not None:
                break
            time.sleep(0.01)
        assert decision is not None
        assert decision.action == 'skip'
        assert decision.reason == 'stale_inference'

        # The old job must not have committed anything to the new episode.
        assert core.session_id == 'new'
        assert core.episode == 1
        assert core.last_decision is None
        assert core.last_observation_stamp == -1
        assert core.terminal is False
        assert core.status != 'running'
    finally:
        adapter.release.set()
        worker.shutdown()


def _goal(sm: GoalStateMachine, stamp: int):
    return sm.next_goal(poses=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], frame_id='odom', stamp_ns=stamp)


def test_cancel_ack_does_not_send_pending_before_terminal():
    sm = GoalStateMachine(cancel_timeout_sec=1.0)
    first = _goal(sm, 1)
    assert sm.request(first, 0.0) == ['send:1']
    assert sm.goal_response(1, True) == []
    second = _goal(sm, 2)
    assert sm.request(second, 1.0) == ['cancel:1']

    # ACK alone must not send the pending goal.
    assert sm.cancel_response(1, 1.1) == []
    assert sm.state == 'canceling'
    # A non-terminal status must not send either.
    assert sm.goal_result(1, 2) == []
    # Terminal CANCELED is the first point at which the pending goal may be sent.
    assert sm.goal_result(1, 5) == ['send:2']


def test_cancel_timeout_fails_closed_and_never_sends_second_goal():
    sm = GoalStateMachine(cancel_timeout_sec=1.0)
    first = _goal(sm, 1)
    assert sm.request(first, 0.0) == ['send:1']
    assert sm.goal_response(1, True) == []
    second = _goal(sm, 2)
    assert sm.request(second, 1.0) == ['cancel:1']

    assert sm.cancel_timeout(1.5) == []
    assert sm.cancel_timeout(2.1) == ['error:cancel_timeout:1']
    assert sm.state == ERROR
    assert sm.pending is not None
    # Only the real terminal result may release the pending goal.
    assert sm.goal_result(1, 5) == ['send:2']


def test_cancel_ack_then_succeeded_terminal_sends_pending():
    sm = GoalStateMachine(cancel_timeout_sec=1.0)
    first = _goal(sm, 1)
    sm.request(first, 0.0)
    sm.goal_response(1, True)
    second = _goal(sm, 2)
    sm.request(second, 1.0)
    assert sm.cancel_response(1, 1.1) == []
    assert sm.goal_result(1, 4) == ['send:2']


def test_new_session_does_not_reset_or_dispatch_while_old_goal_is_active():
    from arena_vln_models.execution import SessionTransition

    sm = GoalStateMachine(cancel_timeout_sec=1.0)
    first = _goal(sm, 1)
    sm.request(first, 0.0)
    sm.goal_response(1, True)
    assert sm.stop() == ['cancel:1']
    assert sm.state == 'stopping'

    transition = SessionTransition()
    transition.request(('new', 1))
    assert transition.can_complete(worker_idle=True, goals_state='stopping') is False
    # The old goal terminal result returns the state machine to idle.
    assert sm.goal_result(1, 5) == []
    assert sm.state == 'idle'
    assert transition.can_complete(worker_idle=True, goals_state='idle') is True
    assert transition.complete() == ('new', 1)
    assert transition.pending is None


def test_session_transition_does_not_reset_until_blocking_inference_finishes():
    from arena_vln_models.execution import SessionTransition

    adapter = _BlockingAdapter()
    core = ExecutionCore(adapter, ExecutionConfig(model='blocking'))
    core.reset('old', 0)
    core.release()
    core.set_instruction('go')
    worker = AsyncInferenceWorker(core)
    transition = SessionTransition()
    try:
        assert worker.submit(_observation(stamp_ns=10), time.monotonic()) is True
        assert adapter.started.wait(2.0)

        transition.request(('new', 1))
        worker.invalidate()
        core.set_hold('session_transition')
        time.sleep(1.0)
        # Still blocked: the old inference owns the shared adapter/history.
        assert worker.in_flight is True
        assert transition.can_complete(worker_idle=not worker.in_flight, goals_state='idle') is False
        assert core.session_id == 'old'

        adapter.release.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and worker.in_flight:
            time.sleep(0.01)
        assert worker.in_flight is False
        assert transition.can_complete(worker_idle=True, goals_state='idle') is True
        identity = transition.complete()
        core.reset(*identity)
        assert core.session_id == 'new'
        assert core.episode == 1
    finally:
        adapter.release.set()
        worker.shutdown()

