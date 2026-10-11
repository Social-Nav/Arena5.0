"""Cooperative episode boundaries around the unmodified Nav2 FollowPath server.

This helper never accepts paths or publishes velocities. The task generator
owns its asyncio calls; ROS callbacks only update observations.
"""
import asyncio
import json
import math
from pathlib import Path
import time

from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .session import Session


class WaypointLifecycle:
    def __init__(self, node, namespace, *, heartbeat_timeout=2.5, transition_timeout=10.0, record_dir=''):
        self.node = node
        self.namespace = str(namespace).rstrip('/')
        self.session = Session(heartbeat_timeout=heartbeat_timeout)
        self.timeout = transition_timeout
        self.active_goals = set()
        self.status_seen = False
        self.velocity_at = float('-inf')
        self.zero_since = None
        self.stopped = False
        self.controller_active = False
        self.error = ''
        self.trace = Path(record_dir) / 'path_execution.jsonl' if record_dir else None
        self._goal_states = {}
        self._lock = asyncio.Lock()
        self._clients = [
            node.create_client(GetState, self.namespace + '/controller_server/get_state'),
            node.create_client(ChangeState, self.namespace + '/controller_server/change_state'),
            node.create_client(CancelGoal, self.namespace + '/follow_path/_action/cancel_goal'),
            node.create_client(GetState, self.namespace + '/velocity_smoother/get_state'),
        ]
        self.get_state, self.change_state, self.cancel, self.smoother_state = self._clients
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._subscriptions = [
            node.create_subscription(String, self.namespace + '/model/status', self._model_status, 10),
            node.create_subscription(GoalStatusArray, self.namespace + '/follow_path/_action/status',
                                     self._action_status, latched),
            node.create_subscription(Twist, self.namespace + '/cmd_vel', self._velocity, 10),
        ]

    def _model_status(self, msg):
        try:
            self.session.receive(json.loads(msg.data), time.monotonic())
        except (ValueError, TypeError):
            return

    def _action_status(self, msg):
        self.status_seen = True
        for item in msg.status_list:
            goal_id = bytes(item.goal_info.goal_id.uuid).hex()
            if self._goal_states.get(goal_id) != item.status:
                self._goal_states[goal_id] = item.status
                self._record('goal_status', goal_id=goal_id, status=item.status)
        self.active_goals = {
            bytes(item.goal_info.goal_id.uuid) for item in msg.status_list
            if item.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING,
                               GoalStatus.STATUS_CANCELING)
        }

    def _velocity(self, msg):
        now = time.monotonic()
        values = (msg.linear.x, msg.linear.y, msg.linear.z,
                  msg.angular.x, msg.angular.y, msg.angular.z)
        zero = all(math.isfinite(v) and abs(v) < 1e-5 for v in values)
        if zero:
            if self.zero_since is None:
                self.zero_since = now
        else:
            self.zero_since = None
        self.velocity_at = now

    def _record(self, event, **fields):
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with self.trace.open('a') as stream:
                stream.write(json.dumps(dict(event=event, session_id=self.session.session_id,
                    episode=self.session.episode, wall_time=time.time(), **fields)) + '\n')

    async def _call(self, client, request, deadline):
        while not client.service_is_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{client.srv_name} is unavailable')
            await asyncio.sleep(.02)
        future = client.call_async(request)
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                raise TimeoutError(f'{client.srv_name} timed out')
            await asyncio.sleep(.02)
        result = future.result()
        if result is None:
            raise RuntimeError(f'{client.srv_name} returned no result')
        return result

    async def _transition(self, transition, deadline):
        req = ChangeState.Request()
        req.transition.id = transition
        if not (await self._call(self.change_state, req, deadline)).success:
            raise RuntimeError(f'controller lifecycle transition {transition} failed')

    async def stop(self):
        """Close acceptance, drain native goals, then wait for the smoother's zero."""
        async with self._lock:
            if self.stopped:
                return
            boundary = time.monotonic()
            deadline = time.monotonic() + self.timeout
            state = (await self._call(self.get_state, GetState.Request(), deadline)).current_state.id
            # Cancel uses the standard all-zero UUID and timestamp (all goals).
            cancel_error = None
            if state == State.PRIMARY_STATE_ACTIVE:
                try:
                    response = await self._call(self.cancel, CancelGoal.Request(), deadline)
                    if response.return_code != CancelGoal.Response.ERROR_NONE:
                        raise RuntimeError(f'FollowPath cancellation rejected: {response.return_code}')
                except Exception as exc:
                    cancel_error = exc
                    # Still attempt lifecycle deactivation if the cancel service
                    # fails; preserve the failure so the next session cannot open.
                    deadline = time.monotonic() + self.timeout
                await self._transition(Transition.TRANSITION_DEACTIVATE, deadline)
            elif state == State.PRIMARY_STATE_UNCONFIGURED:
                await self._transition(Transition.TRANSITION_CONFIGURE, deadline)
            state = (await self._call(self.get_state, GetState.Request(), deadline)).current_state.id
            if state != State.PRIMARY_STATE_INACTIVE:
                raise RuntimeError(f'controller failed to become inactive: {state}')
            self.controller_active = False
            # A never-activated controller has produced no commands. Otherwise
            # require an observed zero after deactivation, not a stale sample.
            while True:
                now = time.monotonic()
                zero = (self.zero_since is not None and self.velocity_at >= boundary
                        and now - self.zero_since >= .1)
                never_moved = self.velocity_at == float('-inf') and not self.status_seen
                if not self.active_goals and (zero or never_moved):
                    break
                if now >= deadline:
                    raise TimeoutError('FollowPath goals or final cmd_vel did not stop before reset')
                await asyncio.sleep(.02)
            if cancel_error is not None:
                raise RuntimeError(f'FollowPath cancellation failed: {cancel_error}') from cancel_error
            self.stopped = True
            self._record('controller_stopped')

    async def activate(self):
        async with self._lock:
            if self.controller_active:
                return
            deadline = time.monotonic() + self.timeout
            while True:
                smoother = await self._call(self.smoother_state, GetState.Request(), deadline)
                if smoother.current_state.id == State.PRIMARY_STATE_ACTIVE:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('velocity smoother did not become active')
                await asyncio.sleep(.02)
            state = (await self._call(self.get_state, GetState.Request(), deadline)).current_state.id
            if state == State.PRIMARY_STATE_UNCONFIGURED:
                await self._transition(Transition.TRANSITION_CONFIGURE, deadline)
                state = State.PRIMARY_STATE_INACTIVE
            if state == State.PRIMARY_STATE_INACTIVE:
                await self._transition(Transition.TRANSITION_ACTIVATE, deadline)
            state = (await self._call(self.get_state, GetState.Request(), deadline)).current_state.id
            if state != State.PRIMARY_STATE_ACTIVE:
                raise RuntimeError('FollowPath controller did not become active')
            self.controller_active, self.stopped = True, False
            self._record('controller_active')

    async def begin(self, session_id, episode):
        await self.stop()
        self.session.reset(session_id, episode)
        self.error = ''
        self._record('session_started')

    async def poll(self):
        """Called from the task loop, including while waiting at the start barrier."""
        reason = self.session.stop_reason(time.monotonic())
        if reason:
            await self.stop()
        elif not self.controller_active:
            await self.activate()
        return reason

    def destroy(self):
        for subscription in self._subscriptions:
            self.node.destroy_subscription(subscription)
        for client in self._clients:
            self.node.destroy_client(client)
