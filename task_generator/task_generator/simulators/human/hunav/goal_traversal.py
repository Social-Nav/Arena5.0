"""Pedestrian goal-list traversal modes.

Background
----------
HuNav advances a pedestrian through its waypoints with a strict FIFO deque
rotation.  Measured in ``lightsfm/sfm.hpp`` (``SocialForceModel::updatePosition``,
both overloads) and duplicated in ``hunav_agent_manager``'s
``AgentManager::updateGoal``::

    Goal g = agent.goals.front();
    agent.goals.pop_front();
    if (agent.cyclicGoals) {
      agent.goals.push_back(g);
    }

There is no de-duplication, no reordering, no nearest-goal selection and no
length cap anywhere in that path, and the goal list is written exactly once (at
``AgentManager::initializeAgents``; the goal-refresh block inside
``updateAgents`` is commented out).  Two consequences follow, and both were
verified against the real ``compute_agents`` service rather than assumed:

* ``cyclic_goals: false`` drains the deque and the pedestrian then stops.  This
  is why benchmark pedestrians walk for 4-18 s of a 300 s episode.
* ``cyclic_goals: true`` rotates the deque forever, so the pedestrian returns to
  waypoint 0 and repeats *in the same direction*.

Because the rotation is a plain FIFO with no filtering, reciprocating (ping-pong)
traversal over waypoints ``0..N`` is *exactly* cyclic traversal over the mirrored
sequence ``0,1,..,N,N-1,..,1``.  That equivalence is what this module implements:
the new mode is a property of how the goal list is built, not a new branch inside
a vendored behaviour tree.  Nothing in ``hunav_msgs`` or ``hunav_agent_manager``
has to change, and the authored scenario waypoints are never edited -- the
expansion happens in memory when the agent message is built.

Why an enum rather than a second boolean
----------------------------------------
``cyclic_goals`` is a ``bool`` on the wire (``hunav_msgs/Agent.msg``) and a
``bool`` in config.  A boolean that has grown a third state is a design smell, so
the semantic surface here is a named mode.  The legacy boolean spelling keeps
working and keeps its exact meaning; it is simply mapped onto the enum.
"""

import enum
import math
import typing

__all__ = [
    'GoalTraversal',
    'InvalidGoalTraversal',
    'parse_goal_traversal',
    'resolve_goal_traversal',
    'reciprocating_sequence',
    'expand_goal_sequence',
    'effective_traversal',
]

T = typing.TypeVar('T')

#: Waypoints closer than this are treated as the same point when deciding
#: whether the mirror step would introduce a zero-length segment.  Deliberately
#: tight: it identifies *identical* authored waypoints, and never merges two
#: waypoints a scenario author meant to be distinct.
COINCIDENT_TOLERANCE_M = 1e-9


class InvalidGoalTraversal(ValueError):
    """Raised when a configured traversal mode is not recognised.

    This is intentionally loud.  An unrecognised mode must never degrade to
    "the default" -- a silently ignored opt-in is the failure mode this project
    has hit repeatedly (a capability probe that never arms, a strictness
    request that is accepted and dropped).  A typo in a config must stop the
    run, not produce a run that quietly measures the old behaviour.
    """


class GoalTraversal(str, enum.Enum):
    """How a pedestrian consumes its waypoint list.

    ``ONCE``
        Walk waypoint 0 -> N and then stop.  Today's default, and the exact
        behaviour of ``cyclic_goals: false``.

    ``CYCLIC``
        Walk 0 -> N, return to 0, repeat in the same direction.  The exact
        behaviour of ``cyclic_goals: true``.

    ``RECIPROCATE``
        Walk 0 -> N, then N -> 0, then 0 -> N, indefinitely (ping-pong).  New.
    """

    ONCE = 'once'
    CYCLIC = 'cyclic'
    RECIPROCATE = 'reciprocate'

    @property
    def wire_cyclic_goals(self) -> bool:
        """The value to put in ``hunav_msgs/Agent.cyclic_goals``.

        ``RECIPROCATE`` rides on the existing cyclic rotation, so it sets the
        same wire bit as ``CYCLIC``; the two differ only in the goal list.
        """
        return self is not GoalTraversal.ONCE

    @property
    def repeats(self) -> bool:
        """Whether the pedestrian keeps moving after reaching waypoint N."""
        return self is not GoalTraversal.ONCE


#: Accepted spellings.  Kept small and explicit on purpose -- anything not
#: listed here raises.
_ALIASES: dict[str, GoalTraversal] = {
    'once': GoalTraversal.ONCE,
    'cyclic': GoalTraversal.CYCLIC,
    'reciprocate': GoalTraversal.RECIPROCATE,
}


def parse_goal_traversal(value: typing.Any) -> GoalTraversal:
    """Coerce a config value to a :class:`GoalTraversal`.

    Raises :class:`InvalidGoalTraversal` for anything unrecognised, listing the
    valid spellings so the failure is self-explanatory.
    """
    if isinstance(value, GoalTraversal):
        return value
    if not isinstance(value, str):
        raise InvalidGoalTraversal(
            f'goal_traversal must be a string, got {type(value).__name__} ({value!r}). '
            f'Valid values: {sorted(_ALIASES)}'
        )
    key = value.strip().lower()
    try:
        return _ALIASES[key]
    except KeyError:
        raise InvalidGoalTraversal(
            f'unknown goal_traversal {value!r}. Valid values: {sorted(_ALIASES)}'
        ) from None


def resolve_goal_traversal(
    explicit: typing.Any = None,
    legacy_cyclic_goals: typing.Any = None,
    fallback: typing.Any = GoalTraversal.ONCE,
) -> GoalTraversal:
    """Resolve the effective traversal mode from the layers that can set it.

    Precedence, most specific first:

    1. ``explicit`` -- a ``goal_traversal`` key (per-pedestrian, then run-level).
    2. ``legacy_cyclic_goals`` -- the pre-existing ``cyclic_goals`` boolean.
       ``True`` maps to ``CYCLIC`` and ``False`` to ``ONCE``, preserving the
       meaning that spelling has always had.
    3. ``fallback`` -- the global default.

    ``None`` means "this layer did not express an opinion" and is skipped; note
    that ``legacy_cyclic_goals=False`` is an opinion (it means ``ONCE``) whereas
    ``None`` is not, so a scenario that says ``cyclic_goals: false`` still pins
    ``ONCE`` even when a run-level default asks for something else.
    """
    if explicit is not None:
        return parse_goal_traversal(explicit)
    if legacy_cyclic_goals is not None:
        return GoalTraversal.CYCLIC if bool(legacy_cyclic_goals) else GoalTraversal.ONCE
    return parse_goal_traversal(fallback)


def _coincident(a: typing.Any, b: typing.Any) -> bool:
    """Whether two waypoints occupy the same point (within tolerance)."""
    try:
        return (
            math.isclose(float(a.x), float(b.x), abs_tol=COINCIDENT_TOLERANCE_M, rel_tol=0.0)
            and math.isclose(float(a.y), float(b.y), abs_tol=COINCIDENT_TOLERANCE_M, rel_tol=0.0)
        )
    except AttributeError:
        return False


def reciprocating_sequence(waypoints: typing.Sequence[T]) -> list[T]:
    """Mirror ``waypoints`` so that cycling it reciprocates.

    Returns ``[w0, w1, .., wN, w(N-1), .., w1]`` -- the list followed by its
    reversed *interior*.  The endpoints are deliberately not duplicated: the
    naive ``waypoints + reversed(waypoints)`` would put ``wN`` and ``w0`` next to
    themselves, creating two zero-length segments per lap.

    Edge cases, each with a defined outcome:

    * 0 waypoints -> ``[]``.  Nothing to traverse; the caller must not turn on
      the cyclic wire bit for an empty list.
    * 1 waypoint -> unchanged.  There is no second point to reciprocate toward.
    * 2 waypoints -> unchanged.  Cycling ``[w0, w1]`` already yields
      ``w0 -> w1 -> w0 -> w1``, which *is* reciprocation; mirroring would only
      append a redundant copy.
    * ``w0`` coincident with ``w1`` -> the mirrored tail would end on a copy of
      ``w0`` and the cycle would wrap ``w0 -> w0``, a zero-length segment
      introduced by this function rather than by the author.  That trailing
      element is dropped.  The same applies at the fold when ``w(N-1)`` is
      coincident with ``wN``.

    Authored waypoints are never reordered or merged; only elements this
    function itself would have added are withheld.
    """
    items = list(waypoints)
    if len(items) <= 2:
        return items

    tail = items[-2:0:-1]  # w(N-1) .. w1

    # Do not let the mirror manufacture a zero-length segment at the fold ...
    if tail and _coincident(tail[0], items[-1]):
        tail = tail[1:]
    # ... nor at the wrap back to w0.
    if tail and _coincident(tail[-1], items[0]):
        tail = tail[:-1]

    return items + tail


def expand_goal_sequence(
    waypoints: typing.Sequence[T],
    mode: GoalTraversal,
) -> tuple[list[T], bool]:
    """Build the goal list to transmit, plus the ``cyclic_goals`` wire value.

    ``ONCE`` and ``CYCLIC`` return the waypoints untouched, so those two
    behaviours are byte-for-byte what they were before this module existed.
    Only ``RECIPROCATE`` expands the list.

    An empty waypoint list never sets the cyclic wire bit.  ``AgentManager::
    updateGoal`` calls ``goals.front()`` without an emptiness check, so an empty
    deque plus a rotation request is the one combination worth refusing on
    principle -- even though the shipped behaviour tree happens to gate it
    (``IsGoalReached`` returns false on an empty list, so ``UpdateGoal`` is never
    ticked, which was confirmed by running 120 service calls against the real
    node with zero goals and observing no crash).
    """
    items = list(waypoints)
    if not items:
        return items, False
    if mode is GoalTraversal.RECIPROCATE:
        return reciprocating_sequence(items), mode.wire_cyclic_goals
    return items, mode.wire_cyclic_goals


def effective_traversal(
    mode: GoalTraversal,
    authored_waypoint_count: int,
) -> tuple[GoalTraversal, str]:
    """What the agent will ACTUALLY do, and why, given how many waypoints it has.

    A requested mode is not always achievable.  Returns
    ``(effective_mode, explanation)`` where ``effective_mode`` differing from
    ``mode`` means the request could not be honoured.

    This exists so the degradation is *stated* rather than left to be deduced
    from a waypoint count at three in the morning.  A mode that reports itself
    engaged while behaving as something else is this project's most repeated
    defect, and a single-waypoint agent is exactly that case: it accepts
    ``reciprocate``, reports ``reciprocate``, and then walks to its one goal and
    stands still for the rest of the episode, because the deque rotation is a
    no-op once the agent is inside the goal radius of the only goal it has.
    Measured against the real ``compute_agents`` service: 6.10 m walked, last
    motion at t=7.6 s, then stationary for the remaining 132.3 s of a 140 s
    probe -- and byte-identical messages for ``cyclic`` and ``reciprocate``, so
    the three modes are indistinguishable there by construction.

    Be precise about *why* a single waypoint does not reciprocate: it is a
    consequence of this implementation's scope, NOT a geometric impossibility.
    The agent has an authored spawn pose as well as its one authored waypoint, so
    reciprocating between those two points is perfectly well defined, and it
    needs no new machinery -- prepending the spawn pose as a synthetic goal gives
    a two-element list, which the existing cyclic rotation already reciprocates
    (measured: 139.80 m and motion on 99.9% of ticks over the same 140 s probe,
    versus 6.10 m and 5.5% with the authored list alone).  That was deliberately
    not done here, because it invents a waypoint the scenario author did not
    write, and inventing route geometry is a decision for whoever owns the
    scenarios rather than something this layer should do silently.  If that
    trade is ever wanted, this is the place it belongs.

    Measured counts at the time of writing: of 417 pedestrians across the 170
    scenario files, 0 have no waypoints and 94 have exactly one.  Those 94 are
    maximally concentrated -- they are 100% of the pedestrians in 7 scenarios, all
    of them under ``hospital_1``/``hospital_2``, and 163 of 170 scenarios contain
    none.  All 321 pedestrians in the grscenes benchmark worlds have two or more,
    so the benchmark is unaffected while those 7 hospital scenarios gain nothing
    from a repeating mode.
    """
    n = int(authored_waypoint_count)

    if n == 0:
        if mode is GoalTraversal.ONCE:
            return mode, (
                'no authored waypoints; walking the built-in fallback route once'
            )
        return GoalTraversal.CYCLIC, (
            f'DEGRADED: asked for {mode.value} but there are no authored waypoints, '
            'so the built-in fallback route is used and is NOT mirrored '
            '(there is no authored route to reciprocate over). '
            'Effective behaviour is cyclic over the fallback route.'
        )

    if n == 1:
        if mode is GoalTraversal.ONCE:
            return mode, 'single waypoint; walking to it once'
        return GoalTraversal.ONCE, (
            f'DEGRADED: asked for {mode.value} but this agent has only ONE authored '
            'waypoint, so its authored route has nothing to travel back and forth '
            'between. It will walk to that waypoint and then stand still for the '
            f'rest of the episode -- effective behaviour is once, NOT {mode.value}. '
            f'Give the agent a second waypoint to make {mode.value} meaningful. '
            '(Not a geometric limit: its spawn pose could serve as the second '
            'endpoint, but this layer does not invent route geometry the scenario '
            'author did not write.)'
        )

    if n == 2 and mode is GoalTraversal.RECIPROCATE:
        # Not a degradation: cycling two waypoints already reciprocates, so the
        # requested behaviour is delivered exactly. Named anyway so a reader
        # who notices the goal list was not expanded does not suspect a bug.
        return mode, (
            'two waypoints; cycling them already reciprocates, so the goal list '
            'is intentionally not expanded. Requested behaviour is delivered.'
        )

    return mode, f'{n} authored waypoints; {mode.value} as requested'
