"""Real ``ArenaModelExecutor._tick`` lifecycle tests for LISN grounded goals.

These tests call the production ``_tick``/``_on_reset`` methods on a ROS-free
harness (the node constructor is bypassed) so the release/terminal gate is
exercised where the bug lived, not through a re-implemented helper.
"""
import math
from types import SimpleNamespace

import time

from arena_vln_models.core import Pose2D
from arena_vln_models.execution import ExecutionCore, ExecutionDecision, SessionTransition
from arena_vln_models.executor import ArenaModelExecutor, grounded_goal_decision
from arena_vln_models.goal_state import GoalStateMachine


def _grounded(*, x=1.0, y=2.0, sec=10, nanosec=0):
    return grounded_goal_decision(SimpleNamespace(
        header=SimpleNamespace(frame_id='map', stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)),
    ))


class _Worker:
    in_flight = False

    def invalidate(self):
        pass


class _TickHarness(ArenaModelExecutor):
    """Bypasses the rclpy Node constructor; keeps the production methods."""

    def __init__(self, *, released=True, terminal=False):
        self._lisn_grounding_live = True
        self._lisn_pending_goal = None
        self.adapter = SimpleNamespace(name='stub')
        self.core = ExecutionCore(self.adapter)
        self.core.reset('session', 0)
        self.core.released = bool(released)
        self.core.terminal = bool(terminal)
        self.worker = _Worker()
        self.goals = GoalStateMachine()
        self.goals.session_token = 'session'
        self._session_transition = SessionTransition()
        self.handled = []
        self.actions = []
        self.records = []
        self.start_pose = Pose2D(-0.04, -6.2, 0.0)
        self._fast_guard_veto_topic = ''

    # --- collaborators the production methods call -----------------------
    def _maybe_complete_session_transition(self):
        pass

    def _handle_decision(self, decision):
        self.handled.append(decision)

    def _execute_goal_actions(self, actions):
        self.actions.extend(actions)

    def _cancel_all(self, reason):
        self.actions.extend(self.goals.stop())
        self.records.append(('cancel_all', reason))

    def _publish_status(self, *args, **kwargs):
        pass

    def _publish_pre_episode_gate(self, ready):
        self.records.append(('gate', bool(ready)))

    def _record(self, event, **fields):
        self.records.append((event, fields))

    def _map_pose_at_stamp(self, stamp_ns):
        return self.start_pose


def test_released_episode_consumes_pending_grounded_goal():
    harness = _TickHarness(released=True)
    decision = _grounded()
    harness._lisn_pending_goal = decision
    harness._tick()
    assert harness.handled == [decision]
    assert harness._lisn_pending_goal is None


def test_unreleased_episode_drops_pending_goal_without_sending():
    harness = _TickHarness(released=False)
    harness._lisn_pending_goal = _grounded()
    harness._tick()
    assert harness.handled == []
    assert harness._lisn_pending_goal is None


def test_terminal_episode_drops_pending_goal_without_sending():
    harness = _TickHarness(released=True, terminal=True)
    harness._lisn_pending_goal = _grounded()
    harness._tick()
    assert harness.handled == []
    assert harness._lisn_pending_goal is None


def test_pending_session_transition_blocks_new_goal():
    harness = _TickHarness(released=True)
    harness._session_transition.request(('other-session', 1))
    harness._lisn_pending_goal = _grounded()
    harness._tick()
    assert harness.handled == []
    assert harness._lisn_pending_goal is None


def test_new_goal_replaces_active_path_instead_of_waiting():
    harness = _TickHarness(released=True)
    active = harness.goals.next_goal(poses=[(0.0, 0.0, 0.0)], frame_id='map', stamp_ns=1)
    harness.goals.request(active, time.monotonic())
    assert harness.goals.state != 'idle'
    decision = _grounded()
    harness._lisn_pending_goal = decision
    harness._tick()
    assert harness.handled == [decision]
    assert harness._lisn_pending_goal is None


def test_task_reset_drops_stale_grounded_goal():
    harness = _TickHarness(released=True)
    harness._lisn_pending_goal = _grounded()
    harness._on_reset(SimpleNamespace(data=1))
    assert harness._lisn_pending_goal is None
    assert harness.core.released is False
    assert ('cancel_all', 'task_reset') in harness.records


def test_stop_decision_cancels_active_path_and_drops_goal():
    harness = _TickHarness(released=True)
    active = harness.goals.next_goal(poses=[(0.0, 0.0, 0.0)], frame_id='map', stamp_ns=1)
    harness.goals.request(active, time.monotonic())
    harness._lisn_pending_goal = _grounded()
    stop = ExecutionDecision(action='stop', status='stop', reason='model_stop', terminal=True)
    # Drive the production decision handler directly; the pending goal was
    # already consumed by _tick in the real flow.
    harness._lisn_pending_goal = None
    ArenaModelExecutor._handle_decision(harness, stop)
    assert harness.goals.state == 'stopping'
    assert ('cancel_all', 'model_stop') in harness.records


def test_grounded_single_point_gets_map_start_pose():
    harness = _TickHarness(released=True)
    decision = _grounded(x=-3.0, y=-6.45, sec=31, nanosec=16668284)
    ArenaModelExecutor._request_goal(harness, decision)
    assert harness.goals.active is not None
    assert harness.goals.active.frame_id == 'map'
    assert harness.goals.active.stamp_ns == 31_016_668_284
    heading = math.atan2(-6.45 - (-6.2), -3.0 - (-0.04))
    assert harness.goals.active.poses == [
        (-0.04, -6.2, heading), (-3.0, -6.45, heading),
    ]
    assert harness.actions == ['send:1']


def test_grounded_single_point_fails_closed_without_start_pose():
    harness = _TickHarness(released=True)
    harness.start_pose = None
    decision = _grounded(x=-3.0, y=-6.45, sec=31, nanosec=16668284)
    ArenaModelExecutor._request_goal(harness, decision)
    assert harness.goals.active is None
    assert any(event == 'goal_rejected' for event, _ in harness.records)
