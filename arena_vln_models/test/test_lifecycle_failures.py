"""Failure boundaries: a cancel acknowledgement alone must never permit reset."""
import asyncio
import time
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('pending_goal, fresh_zero, cancel_error', [
    (True, True, False), (False, False, False), (False, True, True),
])
def test_begin_does_not_change_session_when_stop_cannot_be_confirmed(pending_goal, fresh_zero, cancel_error):
    from lifecycle_msgs.msg import State
    from arena_vln_models.lifecycle import WaypointLifecycle
    from arena_vln_models.session import Session

    async def run():
        helper = WaypointLifecycle.__new__(WaypointLifecycle)
        helper._lock = asyncio.Lock()
        helper.timeout = .01
        helper.stopped = False
        helper.controller_active = True
        helper.session = Session('previous', 1)
        helper.active_goals = {b'pending'} if pending_goal else set()
        helper.status_seen = True
        helper.zero_since = time.monotonic() - 1.
        helper.velocity_at = time.monotonic() - 1.
        helper.get_state, helper.cancel = object(), object()
        helper.trace = None
        state = State.PRIMARY_STATE_ACTIVE

        async def call(client, request, deadline):
            if client is helper.get_state:
                return SimpleNamespace(current_state=SimpleNamespace(id=state))
            if cancel_error:
                raise TimeoutError('cancel service unavailable')
            return SimpleNamespace(return_code=0)  # Cancellation accepted, still executing.

        async def transition(value, deadline):
            nonlocal state
            state = State.PRIMARY_STATE_INACTIVE
            if fresh_zero:
                helper.velocity_at = time.monotonic()

        helper._call, helper._transition = call, transition
        with pytest.raises((TimeoutError, RuntimeError), match='cancellation failed' if cancel_error else 'did not stop'):
            await helper.begin('next', 2)
        assert helper.session.session_id == 'previous'
        assert not helper.stopped
        assert not helper.controller_active

    asyncio.run(run())
