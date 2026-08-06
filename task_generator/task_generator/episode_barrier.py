"""Explicit episode-start barrier and the pedestrian clock it releases.

Why this module exists
----------------------
An Isaac-backed eval spends a substantial amount of wall *and simulated* time
between "the launch graph came up" and "the benchmark episode is actually
observable".  In that window the scene USD is composed, the robot is teleported
to its start pose, HuNav agents are registered, and every video stream has to
converge (the ``sim_top_down`` stream discards ~20 s of unsettled RTX/DLSS
frames on purpose).  Nothing used to mark the end of that window, with two
measured consequences:

* HuNav consumed each pedestrian's entire one-way route in the first 4-18 s
  after ``task_reset``, so the only dynamic part of the episode happened while
  ``sim_top_down.mp4`` was still discarding warm-up frames -- its frame 0 began
  19-130 frames *after* the walk had finished in 8/8 agent-runs.
* The robot timeout, the recorders and the model client each keyed off a
  different edge, so "t = 0" meant something different in every artifact.

This module provides the two primitives needed to fix that:

``await_episode_start_barrier``
    Waits for a set of named, independently observable conditions and returns a
    structured report.  It never blocks forever: on deadline expiry it raises
    :class:`EpisodeStartBarrierTimeout` naming exactly which conditions were
    unsatisfied and what each one last observed.

``PedestrianEpisodeClock``
    The monotone, gate-able clock whose value is handed to HuNav as the request
    header stamp.  HuNav derives its integration step from the *difference*
    between consecutive request stamps (``bt_node.cpp:398``:
    ``time_step_secs = Time(ag->header.stamp) - prev_time_``), so a frozen stamp
    is the only way to hold pedestrians in place that also guarantees the first
    step after release is a normal-sized step rather than one giant catch-up
    step that would consume the whole route instantly.

Both are deliberately free of ROS and asyncio-timing dependencies (the clock and
sleep functions are injectable) so their behaviour can be tested without a
simulator.
"""

from __future__ import annotations

import time
import typing

import attrs


class EpisodeStartBarrierTimeout(RuntimeError):
    """The episode-start barrier did not pass before its deadline.

    Raised rather than logged: a barrier that silently proceeds is worse than no
    barrier at all, because every downstream artifact would then claim an
    episode origin that was never actually reached.
    """

    def __init__(self, report: "BarrierReport"):
        self.report = report
        super().__init__(report.failure_message())


@attrs.define
class BarrierCondition:
    """One independently observable precondition of episode start.

    Args:
        name: Stable identifier used in logs, the report and the published
            ``eval_ready`` payload.
        check: Zero-argument predicate reading *observable state*.  It must not
            be a call whose failure mode is returning ``None``; this project has
            repeatedly shipped readiness probes that never armed because the
            thing they probed silently returned nothing.
        required: ``False`` marks a condition that does not apply to this run
            (for example video-stream readiness when no recorder is attached).
            Skipped conditions are still reported, with the reason.
        skip_reason: Why the condition is not required, when ``required`` is
            ``False``.
        detail: Optional zero-argument callable returning a short human-readable
            description of the last observed state, recorded in the report.
    """

    name: str
    check: typing.Callable[[], bool]
    required: bool = True
    skip_reason: str = ''
    detail: typing.Optional[typing.Callable[[], str]] = None

    def describe(self) -> str:
        if self.detail is None:
            return ''
        try:
            return str(self.detail())
        except Exception as exc:  # pragma: no cover - defensive only
            return f'<detail failed: {exc}>'

    def evaluate(self) -> bool:
        """Evaluate ``check`` treating an exception as "not satisfied yet".

        An exception here is normal early in startup (a topic that has no
        publisher, a service list that is not populated).  It must not abort the
        barrier, but it must not count as satisfied either.
        """
        try:
            return bool(self.check())
        except Exception:
            return False


@attrs.define
class BarrierReport:
    """The outcome of one barrier wait, suitable for logs and artifacts."""

    passed: bool
    timeout_sec: float
    waited_sec: float
    required: typing.List[str] = attrs.field(factory=list)
    satisfied: typing.List[str] = attrs.field(factory=list)
    unsatisfied: typing.List[str] = attrs.field(factory=list)
    skipped: typing.Dict[str, str] = attrs.field(factory=dict)
    details: typing.Dict[str, str] = attrs.field(factory=dict)

    def to_dict(self) -> dict:
        return {
            'passed': bool(self.passed),
            'timeout_sec': float(self.timeout_sec),
            'waited_sec': round(float(self.waited_sec), 3),
            'required': list(self.required),
            'satisfied': list(self.satisfied),
            'unsatisfied': list(self.unsatisfied),
            'skipped': dict(self.skipped),
            'details': dict(self.details),
        }

    def failure_message(self) -> str:
        unsatisfied = ', '.join(
            f'{name}({self.details.get(name, "no detail")})' for name in self.unsatisfied
        ) or '<none recorded>'
        skipped = ', '.join(f'{name}:{reason}' for name, reason in sorted(self.skipped.items())) or '<none>'
        return (
            f'Episode-start barrier timed out after {self.waited_sec:.1f}s '
            f'(limit {self.timeout_sec:.1f}s). Unsatisfied required conditions: {unsatisfied}. '
            f'Satisfied: {", ".join(self.satisfied) or "<none>"}. Not required: {skipped}. '
            'Refusing to declare an episode origin that was never reached.'
        )


async def await_episode_start_barrier(
    conditions: typing.Sequence[BarrierCondition],
    *,
    timeout_sec: float,
    poll_interval_sec: float = 0.1,
    monotonic: typing.Callable[[], float] = time.monotonic,
    sleep: typing.Optional[typing.Callable[[float], typing.Awaitable[None]]] = None,
    on_progress: typing.Optional[typing.Callable[[BarrierReport], None]] = None,
) -> BarrierReport:
    """Wait until every required condition holds, or raise.

    Args:
        conditions: The conditions to evaluate.  Conditions with
            ``required=False`` are reported but never waited on.
        timeout_sec: Hard deadline.  A non-positive value means "evaluate once".
        poll_interval_sec: Delay between evaluation rounds.
        monotonic: Injectable monotonic clock (seconds).
        sleep: Injectable awaitable sleep.  Defaults to ``asyncio.sleep``.
        on_progress: Optional callback invoked with an interim report on every
            round in which the satisfied set changed, so callers can log
            progress without duplicating the evaluation logic.

    Returns:
        The passing :class:`BarrierReport`.

    Raises:
        EpisodeStartBarrierTimeout: If the deadline expires first.  The report is
            attached, listing the unsatisfied conditions and their last observed
            detail.
    """
    if sleep is None:
        import asyncio

        sleep = asyncio.sleep

    started_at = monotonic()
    deadline = started_at + max(float(timeout_sec), 0.0)
    required = [condition for condition in conditions if condition.required]
    skipped = {
        condition.name: condition.skip_reason or 'not_required'
        for condition in conditions
        if not condition.required
    }
    last_satisfied: typing.Optional[typing.Tuple[str, ...]] = None

    while True:
        satisfied: typing.List[str] = []
        unsatisfied: typing.List[str] = []
        details: typing.Dict[str, str] = {}
        for condition in conditions:
            details[condition.name] = condition.describe()
        for condition in required:
            if condition.evaluate():
                satisfied.append(condition.name)
            else:
                unsatisfied.append(condition.name)

        waited = monotonic() - started_at
        report = BarrierReport(
            passed=not unsatisfied,
            timeout_sec=float(timeout_sec),
            waited_sec=waited,
            required=[condition.name for condition in required],
            satisfied=satisfied,
            unsatisfied=unsatisfied,
            skipped=skipped,
            details=details,
        )
        if report.passed:
            return report

        if on_progress is not None and tuple(satisfied) != last_satisfied:
            last_satisfied = tuple(satisfied)
            try:
                on_progress(report)
            except Exception:
                pass

        if monotonic() >= deadline:
            raise EpisodeStartBarrierTimeout(report)

        await sleep(max(float(poll_interval_sec), 0.0))


_NANOSECONDS_PER_SECOND = 1_000_000_000


@attrs.define(init=False)
class PedestrianEpisodeClock:
    """A monotone nanosecond clock that only advances after episode release.

    HuNav integrates pedestrian motion using the difference between consecutive
    ``compute_agents`` request stamps.  Three properties are therefore required
    and each is asserted by a test:

    1. **Held**: while the barrier is closed the clock does not advance, so
       ``dt == 0`` and the behaviour tree cannot make route progress.
    2. **Resumes small**: the first step after release is the size of one
       simulator tick, *not* the whole gated interval.  Simply skipping the
       calls while gated would fail this: the next real stamp would be tens of
       seconds ahead of ``prev_time_`` and HuNav would advance the agent by that
       entire interval in one step, consuming the route instantly.
    3. **Monotone**: ``/clock`` in this project is known to step backwards during
       scene load (the recorder's time-segment defect).  A backwards sample must
       never push the pedestrian clock backwards, because HuNav clamps negative
       ``dt`` to zero and then adopts the lower stamp as its new baseline, which
       would silently re-introduce a large catch-up step later.
    """

    _value_ns: typing.Optional[int]
    _last_sim_ns: typing.Optional[int]
    _origin_ns: typing.Optional[int]
    _released: bool

    def __init__(self) -> None:
        self._value_ns = None
        self._last_sim_ns = None
        self._origin_ns = None
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    @property
    def started(self) -> bool:
        return self._value_ns is not None

    def release(self) -> None:
        """Open the gate.  Idempotent; safe to call from a ROS callback."""
        self._released = True

    def hold(self) -> None:
        """Close the gate again (used on episode rollover and by tests)."""
        self._released = False

    def reset(self) -> None:
        """Forget the accumulated episode clock and close the gate."""
        self._value_ns = None
        self._last_sim_ns = None
        self._origin_ns = None
        self._released = False

    def tick(self, sim_seconds: float) -> float:
        """Advance the clock from the latest ``/clock`` sample and return it.

        Args:
            sim_seconds: The current simulated time in seconds.

        Returns:
            The pedestrian clock value in seconds.  Equal to the first observed
            simulated time until the gate opens, then advancing one-for-one with
            simulated time.
        """
        return self.tick_ns(int(round(float(sim_seconds) * _NANOSECONDS_PER_SECOND))) / _NANOSECONDS_PER_SECOND

    def tick_ns(self, sim_ns: int) -> int:
        """Integer-nanosecond form of :meth:`tick`, used by the ROS caller."""
        sim_ns = int(sim_ns)
        if self._value_ns is None:
            self._value_ns = sim_ns
            self._origin_ns = sim_ns
            self._last_sim_ns = sim_ns
            return self._value_ns
        if self._released and self._last_sim_ns is not None:
            # max(0, ...) keeps the clock monotone across a backwards /clock step.
            self._value_ns += max(0, sim_ns - self._last_sim_ns)
        self._last_sim_ns = sim_ns
        return self._value_ns

    @property
    def value_ns(self) -> int:
        """The current clock value in nanoseconds (0 before the first tick)."""
        return int(self._value_ns or 0)

    @property
    def value(self) -> float:
        """The current clock value in seconds (0.0 before the first tick)."""
        return self.value_ns / _NANOSECONDS_PER_SECOND

    @property
    def episode_elapsed(self) -> float:
        """Seconds of pedestrian motion released so far, i.e. episode time."""
        if self._value_ns is None or self._origin_ns is None:
            return 0.0
        return (self._value_ns - self._origin_ns) / _NANOSECONDS_PER_SECOND

    def stamp_sec_nanosec(self) -> typing.Tuple[int, int]:
        """The clock as a ``(sec, nanosec)`` pair for a ROS message stamp."""
        value_ns = self.value_ns
        return value_ns // _NANOSECONDS_PER_SECOND, value_ns % _NANOSECONDS_PER_SECOND
