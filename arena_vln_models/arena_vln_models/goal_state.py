"""Pure FollowPath goal replacement state machine.

The executor must not clear an active handle and immediately send a new goal:
the old goal may still be executing.  This state machine keeps at most one
active goal, remembers the latest pending path, waits for cancel/terminal
confirmation (or a bounded timeout), and ignores callbacks from stale serials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

IDLE = 'idle'
SENDING = 'sending'
ACTIVE = 'active'
CANCELING = 'canceling'
STOPPING = 'stopping'
ERROR = 'error'
TERMINAL_STATUSES = frozenset({
    4, 5, 6,
    'SUCCEEDED', 'CANCELED', 'ABORTED',
    'succeeded', 'canceled', 'aborted',
})


@dataclass
class PendingGoal:
    serial: int
    session_token: str
    poses: list[tuple[float, float, float]] = field(default_factory=list)
    frame_id: str = ''
    stamp_ns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class GoalStateMachine:
    def __init__(self, *, cancel_timeout_sec: float = 5.0) -> None:
        self.cancel_timeout_sec = float(cancel_timeout_sec)
        self.state = IDLE
        self.active: Optional[PendingGoal] = None
        self.pending: Optional[PendingGoal] = None
        self.session_token = ''
        self._cancel_deadline = 0.0
        self._next_serial = 0
        self.cancel_acked = False

    def reset(self, session_token: str = '') -> list[str]:
        self.state = IDLE
        self.active = None
        self.pending = None
        self.session_token = str(session_token)
        self._cancel_deadline = 0.0
        self.cancel_acked = False
        return []

    def next_goal(self, *, poses, frame_id: str, stamp_ns: int, metadata=None) -> PendingGoal:
        self._next_serial += 1
        return PendingGoal(
            serial=self._next_serial,
            session_token=self.session_token,
            poses=list(poses),
            frame_id=str(frame_id),
            stamp_ns=int(stamp_ns),
            metadata=dict(metadata or {}),
        )

    def request(self, goal: PendingGoal, now: float) -> list[str]:
        if self.state == IDLE:
            self.active = goal
            self.state = SENDING
            return [f'send:{goal.serial}']
        # Latest path wins.  Never send while a previous goal may still execute.
        self.pending = goal
        if self.state == ACTIVE:
            self.state = CANCELING
            self._cancel_deadline = float(now) + self.cancel_timeout_sec
            return [f'cancel:{self.active.serial}']
        return []

    def goal_response(self, serial: int, accepted: bool) -> list[str]:
        if self.active is None or int(serial) != self.active.serial:
            return []  # stale callback from a previous serial
        if accepted:
            if self.state == CANCELING:
                # Accepted while we were replacing it; cancel and wait.
                return [f'cancel:{serial}']
            self.state = ACTIVE
            return []
        self.state = IDLE
        self.active = None
        return self._take_pending()

    def goal_result(self, serial: int, status: Any) -> list[str]:
        if self.active is None or int(serial) != self.active.serial:
            return []
        if not _is_terminal_status(status):
            return []
        if self.state == STOPPING:
            self.state = IDLE
            self.active = None
            return []
        self.state = IDLE
        self.active = None
        return self._take_pending()

    def cancel_response(self, serial: int, now: float) -> list[str]:
        if self.active is None or int(serial) != self.active.serial:
            return []
        if self.state == STOPPING:
            self.state = IDLE
            self.active = None
            return []
        if self.state != CANCELING:
            return []
        # The cancel response only acknowledges that the request was received.
        # The old goal may still be executing; never send the pending goal until
        # goal_result is terminal for this serial.
        self.cancel_acked = True
        return []

    def cancel_timeout(self, now: float) -> list[str]:
        if self.state != CANCELING:
            return []
        if float(now) < self._cancel_deadline:
            return []
        # Bounded wait expired without a terminal result.  Fail closed: keep the
        # active serial for a later terminal callback, retain the pending goal,
        # and never send a second goal while the old one is unconfirmed.
        self.state = ERROR
        serial = self.active.serial if self.active is not None else 0
        return [f'error:cancel_timeout:{serial}']

    def stop(self) -> list[str]:
        self.pending = None
        if self.active is None:
            self.state = IDLE
            return []
        self.state = STOPPING
        return [f'cancel:{self.active.serial}']

    def _take_pending(self) -> list[str]:
        if self.pending is None:
            self.state = IDLE
            return []
        self.active = self.pending
        self.pending = None
        self.state = SENDING
        return [f'send:{self.active.serial}']


def _is_terminal_status(status: Any) -> bool:
    return status in TERMINAL_STATUSES
