"""Tests for pedestrian goal-list traversal modes.

Three behaviours must coexist:

* ``once``        walk waypoint 0 -> N, then stop.   Today's default.
* ``cyclic``      walk 0 -> N, return to 0, repeat.  Pre-existing, opt-in.
* ``reciprocate`` walk 0 -> N -> 0 -> N, forever.    New.

The first two are pre-existing and must be provably unchanged, so several tests
here exist purely to pin them.

Two of these tests guard on *source text* rather than behaviour.  That is
deliberate and the reason is worth stating: the reciprocating mode is built on
HuNav's cyclic deque rotation, and that rotation only survives across service
calls because a goal-refresh block inside ``AgentManager::updateAgents`` is
commented out.  Both ``cyclic`` and ``reciprocate`` depend on that code staying
commented out.  Someone reinstating it in good faith would break both modes at
once, in a vendored C++ dependency, with nothing else in the tree objecting.  A
test that depends on an *absence* has to look at the source to see it.
"""

import importlib.util
import pathlib
import re
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
_TASK_GENERATOR = _HERE.parents[1]
_SRC_ARENA = _TASK_GENERATOR.parent
_WORKSPACE_SRC = _SRC_ARENA.parent

_GOAL_TRAVERSAL_PY = (
    _TASK_GENERATOR
    / 'task_generator' / 'simulators' / 'human' / 'hunav' / 'goal_traversal.py'
)
_AGENT_MANAGER_CPP = (
    _WORKSPACE_SRC / 'deps' / 'hunav' / 'hunav_sim' / 'hunav_agent_manager'
    / 'src' / 'agent_manager.cpp'
)
#: lightsfm is a header-only third-party library installed into the container at
#: /usr/local/include.  It is NOT vendored in this repository, so no repository
#: search finds it and no commit can pin it -- see the module docstring of
#: goal_traversal.py.  Tests touching it must skip when it is absent.
_LIGHTSFM_SFM_HPP = pathlib.Path('/usr/local/include/lightsfm/sfm.hpp')


def _load_goal_traversal():
    """Import goal_traversal.py directly, with no package and hence no ROS.

    Loading it standalone is itself part of the contract: the traversal logic
    must stay free of ROS imports so it can be reasoned about and tested without
    a sourced workspace.
    """
    spec = importlib.util.spec_from_file_location(
        '_lane_cyc_goal_traversal', _GOAL_TRAVERSAL_PY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gt = _load_goal_traversal()
GoalTraversal = gt.GoalTraversal
InvalidGoalTraversal = gt.InvalidGoalTraversal


class _P:
    """Minimal waypoint stand-in exposing the .x/.y the mirror logic reads."""

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __repr__(self):
        return f'_P({self.x},{self.y})'


# --------------------------------------------------------------------------
# The traversal logic is ROS-free
# --------------------------------------------------------------------------

def test_goal_traversal_module_imports_without_ros():
    """The mode logic must not drag in rclpy or any ROS message package."""
    before = set(sys.modules)
    _load_goal_traversal()
    newly_imported = set(sys.modules) - before
    ros_ish = {m for m in newly_imported if m.split('.')[0] in {
        'rclpy', 'hunav_msgs', 'geometry_msgs', 'builtin_interfaces', 'ament_index_python',
    }}
    assert not ros_ish, f'goal_traversal pulled in ROS modules: {sorted(ros_ish)}'


# --------------------------------------------------------------------------
# Mode surface: all three states expressible, unknown values fail loudly
# --------------------------------------------------------------------------

def test_all_three_modes_are_expressible():
    assert {m.value for m in GoalTraversal} == {'once', 'cyclic', 'reciprocate'}


@pytest.mark.parametrize('spelling,expected', [
    ('once', GoalTraversal.ONCE),
    ('cyclic', GoalTraversal.CYCLIC),
    ('reciprocate', GoalTraversal.RECIPROCATE),
    ('  Reciprocate  ', GoalTraversal.RECIPROCATE),
    ('CYCLIC', GoalTraversal.CYCLIC),
])
def test_parse_accepts_documented_spellings(spelling, expected):
    assert gt.parse_goal_traversal(spelling) is expected


@pytest.mark.parametrize('bad', [
    'recipricate',      # plausible typo
    'reciprocating',
    'pingpong',
    'true',
    '',
    'None',
])
def test_unknown_mode_raises_instead_of_defaulting(bad):
    """A mis-spelled opt-in must stop the run, not silently mean ``once``.

    A silent fallback would make "mode mis-spelled" indistinguishable from
    "mode never wired up", which is the never-arming-opt-in failure this
    project has shipped repeatedly.
    """
    with pytest.raises(InvalidGoalTraversal):
        gt.parse_goal_traversal(bad)


@pytest.mark.parametrize('bad', [True, False, 1, 0, None, 2.5, ['cyclic']])
def test_non_string_mode_raises(bad):
    with pytest.raises(InvalidGoalTraversal):
        gt.parse_goal_traversal(bad)


def test_error_message_lists_the_valid_values():
    with pytest.raises(InvalidGoalTraversal) as excinfo:
        gt.parse_goal_traversal('recipricate')
    text = str(excinfo.value)
    for mode in ('once', 'cyclic', 'reciprocate'):
        assert mode in text, f'error should name {mode!r} so the fix is obvious'


# --------------------------------------------------------------------------
# Wire mapping: reciprocate rides the existing cyclic bit
# --------------------------------------------------------------------------

def test_wire_cyclic_goals_mapping():
    assert GoalTraversal.ONCE.wire_cyclic_goals is False
    assert GoalTraversal.CYCLIC.wire_cyclic_goals is True
    # RECIPROCATE deliberately sets the SAME wire bit as CYCLIC: the two differ
    # only in the goal list, which is what lets the new mode avoid touching the
    # vendored hunav_msgs schema or the behaviour tree.
    assert GoalTraversal.RECIPROCATE.wire_cyclic_goals is True


def test_only_once_stops():
    assert GoalTraversal.ONCE.repeats is False
    assert GoalTraversal.CYCLIC.repeats is True
    assert GoalTraversal.RECIPROCATE.repeats is True


# --------------------------------------------------------------------------
# Resolution precedence, including the legacy boolean
# --------------------------------------------------------------------------

def test_legacy_cyclic_goals_true_still_means_cyclic():
    assert gt.resolve_goal_traversal(legacy_cyclic_goals=True) is GoalTraversal.CYCLIC


def test_legacy_cyclic_goals_false_still_means_once():
    assert gt.resolve_goal_traversal(legacy_cyclic_goals=False) is GoalTraversal.ONCE


def test_explicit_mode_beats_legacy_boolean():
    resolved = gt.resolve_goal_traversal(
        explicit='reciprocate', legacy_cyclic_goals=False
    )
    assert resolved is GoalTraversal.RECIPROCATE


def test_legacy_false_is_an_opinion_and_beats_the_fallback():
    """``cyclic_goals: false`` on a specific pedestrian must pin ``once``.

    Distinguishing "said false" from "said nothing" is what stops a run-level
    switch from silently overriding a scenario that deliberately opted out.
    """
    resolved = gt.resolve_goal_traversal(
        legacy_cyclic_goals=False, fallback=GoalTraversal.RECIPROCATE
    )
    assert resolved is GoalTraversal.ONCE


def test_silence_falls_through_to_the_fallback():
    resolved = gt.resolve_goal_traversal(
        explicit=None, legacy_cyclic_goals=None, fallback='reciprocate'
    )
    assert resolved is GoalTraversal.RECIPROCATE


def test_default_fallback_is_todays_behaviour():
    assert gt.resolve_goal_traversal() is GoalTraversal.ONCE


def test_invalid_fallback_still_raises():
    with pytest.raises(InvalidGoalTraversal):
        gt.resolve_goal_traversal(fallback='recipricate')


# --------------------------------------------------------------------------
# The mirror: shape and the ping-pong property
# --------------------------------------------------------------------------

@pytest.mark.parametrize('n,expected', [
    (3, [0, 1, 2, 1]),
    (4, [0, 1, 2, 3, 2, 1]),
    (5, [0, 1, 2, 3, 4, 3, 2, 1]),
    (6, [0, 1, 2, 3, 4, 5, 4, 3, 2, 1]),
])
def test_mirror_appends_the_reversed_interior(n, expected):
    assert gt.reciprocating_sequence(list(range(n))) == expected


def test_mirror_does_not_duplicate_the_endpoints():
    """The naive ``items + reversed(items)`` is wrong and must not be used.

    It would place wN next to itself at the fold and w0 next to itself at the
    wrap, giving two zero-length segments per lap.
    """
    seq = gt.reciprocating_sequence([0, 1, 2, 3])
    assert seq != [0, 1, 2, 3, 3, 2, 1, 0]
    assert seq.count(0) == 1, 'w0 must appear once; the cycle supplies the repeat'
    assert seq.count(3) == 1, 'wN must appear once'


def test_cycling_the_mirror_yields_ping_pong():
    """Walking the mirrored list cyclically must visit 0..N then N..0, forever.

    This is the property the whole design rests on, checked here by replaying
    HuNav's rotation arithmetic (pop front, push back) rather than by trusting
    the shape of the list.
    """
    waypoints = [0, 1, 2, 3]
    deque = gt.reciprocating_sequence(waypoints)
    visited = []
    for _ in range(len(deque) * 3):
        head = deque.pop(0)
        deque.append(head)          # exactly what cyclicGoals=true does
        visited.append(head)

    # Three laps of 0,1,2,3,2,1
    assert visited == [0, 1, 2, 3, 2, 1] * 3
    # And the direction genuinely reverses: differences alternate in sign runs,
    # never jumping straight from N back to 0 the way plain cyclic does.
    assert 3 not in [
        abs(b - a) for a, b in zip(visited, visited[1:])
    ], 'a 3-step jump would mean it teleported N->0 instead of walking back'


def test_plain_cyclic_does_jump_from_N_back_to_zero():
    """Contrast case pinning that ``cyclic`` is a different behaviour.

    If this ever stopped being true, ``cyclic`` and ``reciprocate`` would have
    silently become the same mode.
    """
    deque = [0, 1, 2, 3]
    visited = []
    for _ in range(8):
        head = deque.pop(0)
        deque.append(head)
        visited.append(head)
    assert visited == [0, 1, 2, 3] * 2
    assert 3 in [abs(b - a) for a, b in zip(visited, visited[1:])]


# --------------------------------------------------------------------------
# Edge cases -- every one has a defined, tested outcome
# --------------------------------------------------------------------------

def test_edge_case_empty_waypoint_list():
    assert gt.reciprocating_sequence([]) == []
    goals, wire = gt.expand_goal_sequence([], GoalTraversal.RECIPROCATE)
    assert goals == []
    # An empty deque must never be paired with a rotation request.
    # AgentManager::updateGoal calls goals.front() with no emptiness check, so
    # this is the one combination worth refusing on principle even though the
    # shipped behaviour tree happens to gate it.
    assert wire is False


def test_edge_case_empty_list_never_sets_the_cyclic_bit_in_any_mode():
    for mode in GoalTraversal:
        goals, wire = gt.expand_goal_sequence([], mode)
        assert goals == []
        assert wire is False, f'{mode.value} set the cyclic bit on an empty list'


def test_edge_case_single_waypoint():
    """One waypoint: nothing to reciprocate toward, so the list is unchanged."""
    assert gt.reciprocating_sequence([_P(2, 0)]) == [_P(2, 0)]
    goals, wire = gt.expand_goal_sequence([_P(2, 0)], GoalTraversal.RECIPROCATE)
    assert goals == [_P(2, 0)]
    assert wire is True


def test_edge_case_two_waypoints_are_not_mirrored():
    """Two waypoints: cycling ALREADY reciprocates, so the mirror adds nothing.

    Cycling ``[w0, w1]`` gives w0 -> w1 -> w0 -> w1, which is reciprocation.
    Appending a mirrored interior would only insert a redundant copy.
    """
    a, b = _P(0, 0), _P(6, 0)
    assert gt.reciprocating_sequence([a, b]) == [a, b]
    goals, wire = gt.expand_goal_sequence([a, b], GoalTraversal.RECIPROCATE)
    assert goals == [a, b]
    assert wire is True


def test_edge_case_two_waypoints_reciprocate_equals_cyclic():
    a, b = _P(0, 0), _P(6, 0)
    assert (
        gt.expand_goal_sequence([a, b], GoalTraversal.RECIPROCATE)
        == gt.expand_goal_sequence([a, b], GoalTraversal.CYCLIC)
    )


def test_edge_case_first_and_last_waypoint_coincide():
    """A closed route must not gain a zero-length segment from the mirror."""
    a, b, c = _P(0, 0), _P(6, 0), _P(6, 6)
    closed = [a, b, c, _P(0, 0)]          # w0 == wN
    seq = gt.reciprocating_sequence(closed)
    assert _no_zero_length_segments_in_cycle(seq), seq


def test_edge_case_first_two_waypoints_coincide_would_wrap_onto_itself():
    """``w0 == w1`` is the case where the mirror itself creates the defect.

    The mirrored tail ends on a copy of w1, and the cycle then wraps w1 -> w0.
    If those are the same point that wrap is a zero-length segment that the
    author did not write -- so the trailing element is dropped.
    """
    dup = [_P(0, 0), _P(0, 0), _P(6, 0), _P(6, 6)]
    seq = gt.reciprocating_sequence(dup)
    # The authored duplicate at the head is preserved (we do not rewrite the
    # author's route) but the mirror must not add a second one at the wrap.
    assert seq[-1] != seq[0], f'mirror created a zero-length wrap segment: {seq}'


def test_edge_case_last_two_waypoints_coincide_at_the_fold():
    dup = [_P(0, 0), _P(6, 0), _P(6, 6), _P(6, 6)]
    seq = gt.reciprocating_sequence(dup)
    # tail would start on a copy of wN; that element is withheld
    assert seq[len(dup)] != seq[len(dup) - 1] if len(seq) > len(dup) else True


def test_edge_case_all_waypoints_coincide_is_degenerate_but_safe():
    same = [_P(1, 1), _P(1, 1), _P(1, 1)]
    seq = gt.reciprocating_sequence(same)
    assert len(seq) == 3, 'must not grow, and must not crash'


def _no_zero_length_segments_in_cycle(seq):
    """Whether cycling ``seq`` ever asks an agent to travel zero distance."""
    if len(seq) < 2:
        return True
    pairs = list(zip(seq, seq[1:])) + [(seq[-1], seq[0])]
    return not any(gt._coincident(a, b) for a, b in pairs)


# --------------------------------------------------------------------------
# Existing behaviours are untouched by the expansion
# --------------------------------------------------------------------------

@pytest.mark.parametrize('mode', [GoalTraversal.ONCE, GoalTraversal.CYCLIC])
@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 7])
def test_once_and_cyclic_never_alter_the_goal_list(mode, n):
    """The two pre-existing modes must transmit exactly the authored waypoints.

    This is the regression guard for "both existing behaviours provably
    unchanged": the only mode allowed to rewrite the list is ``reciprocate``.
    """
    waypoints = [_P(i, i * 2) for i in range(n)]
    goals, wire = gt.expand_goal_sequence(waypoints, mode)
    assert goals == waypoints
    assert goals is not waypoints, 'must return a copy, not alias caller state'
    assert wire is (mode.wire_cyclic_goals and bool(waypoints))


def test_reciprocate_is_the_only_mode_that_expands():
    waypoints = [_P(i, 0) for i in range(4)]
    lengths = {
        mode.value: len(gt.expand_goal_sequence(waypoints, mode)[0])
        for mode in GoalTraversal
    }
    assert lengths == {'once': 4, 'cyclic': 4, 'reciprocate': 6}


def test_expand_does_not_mutate_the_input():
    waypoints = [_P(i, 0) for i in range(4)]
    snapshot = list(waypoints)
    gt.expand_goal_sequence(waypoints, GoalTraversal.RECIPROCATE)
    assert waypoints == snapshot


# --------------------------------------------------------------------------
# Source guards on the vendored C++ this design depends on
# --------------------------------------------------------------------------

def test_agent_manager_does_not_refresh_goals_every_tick():
    """Guard the absence that both repeating modes depend on.

    ``AgentManager::updateAgents`` runs on every ``compute_agents`` call.  Its
    goal-refresh block is commented out, which is the only reason the rotating
    deque survives between calls.  Reinstating it would reset every agent's goal
    list to the authored waypoints each tick, permanently pinning the head at
    waypoint 0 and silently killing BOTH ``cyclic`` and ``reciprocate`` -- and
    the reciprocating expansion would become dead weight, because the expanded
    list would be overwritten before it was ever consumed.

    If this test fails, do not "fix" the test.  Either keep the refresh
    commented out, or move goal-list ownership somewhere the rotation can
    survive.
    """
    if not _AGENT_MANAGER_CPP.is_file():
        pytest.skip(f'vendored HuNav not present at {_AGENT_MANAGER_CPP}')
    source = _AGENT_MANAGER_CPP.read_text()

    body = re.search(
        r'bool\s+AgentManager::updateAgents\s*\([^)]*\)\s*\{(.*?)\n\}',
        source,
        re.DOTALL,
    )
    assert body, 'could not locate AgentManager::updateAgents'

    live = [
        line for line in body.group(1).splitlines()
        if 'goals' in line and not line.lstrip().startswith('//')
    ]
    assert not live, (
        'AgentManager::updateAgents now writes the goal list every tick, which '
        'breaks cyclic and reciprocate traversal. Offending lines: '
        f'{[l.strip() for l in live]}'
    )


def test_agent_manager_rotation_is_still_pop_front_push_back():
    """Pin the rotation semantics the mirror equivalence relies on."""
    if not _AGENT_MANAGER_CPP.is_file():
        pytest.skip(f'vendored HuNav not present at {_AGENT_MANAGER_CPP}')
    source = _AGENT_MANAGER_CPP.read_text()

    body = re.search(
        r'bool\s+AgentManager::updateGoal\s*\([^)]*\)\s*\{(.*?)\n\}',
        source,
        re.DOTALL,
    )
    assert body, 'could not locate AgentManager::updateGoal'
    text = body.group(1)
    assert 'goals.pop_front()' in text.replace(' ', ''), (
        'goal advancement is no longer a FIFO pop; reciprocation by list '
        'mirroring is only valid for strict in-order consumption'
    )
    assert 'push_back' in text and 'cyclicGoals' in text, (
        'the cyclic re-append is gone; reciprocate rides on it'
    )


def test_lightsfm_rotation_matches_the_in_repo_duplicate():
    """The primary rotation lives OUTSIDE this repository.

    ``lightsfm/sfm.hpp`` is a header-only third-party library installed at
    /usr/local/include in the container.  It is not vendored, so no repository
    search finds it and no commit pins it, yet it contains the same
    pop_front/push_back rotation as the in-repo copy -- two edit sites for one
    behaviour, one of them invisible to this repository.  That asymmetry is a
    large part of why this feature was implemented as goal-list construction
    instead of a C++ change.
    """
    if not _LIGHTSFM_SFM_HPP.is_file():
        pytest.skip(
            f'{_LIGHTSFM_SFM_HPP} absent (expected outside the container); '
            'the rotation it contains cannot be verified here'
        )
    text = _LIGHTSFM_SFM_HPP.read_text().replace(' ', '').replace('\n', '')
    assert 'goals.pop_front();' in text
    assert 'cyclicGoals' in text
    assert 'goals.push_back(g);' in text


# --------------------------------------------------------------------------
# Config surface
# --------------------------------------------------------------------------
# Effective behaviour: a requested mode is not always achievable
# --------------------------------------------------------------------------
#
# Measured across the 170 scenario files: 417 pedestrians, 0 with no waypoints,
# 94 with exactly one -- and all 94 of those live in hospital_1/hospital_2. All
# 321 pedestrians in the grscenes benchmark worlds have two or more. So the
# single-waypoint degradation is real but does not touch the benchmark.


@pytest.mark.parametrize('n', [3, 4, 10])
def test_effective_matches_requested_when_the_route_allows_it(n):
    for mode in GoalTraversal:
        effective, why = gt.effective_traversal(mode, n)
        assert effective is mode
        assert 'DEGRADED' not in why


def test_single_waypoint_degrades_repeating_modes_to_once():
    """The case that would otherwise report engaged while behaving as `once`.

    A single-waypoint agent has nothing to travel back and forth between; the
    deque rotation is a no-op once it is inside the goal radius of its only goal,
    so it walks there and stands still. Measured against the real service: 6.10 m
    walked, stationary from tick 77 of 400.
    """
    for mode in (GoalTraversal.CYCLIC, GoalTraversal.RECIPROCATE):
        effective, why = gt.effective_traversal(mode, 1)
        assert effective is GoalTraversal.ONCE
        assert 'DEGRADED' in why
        assert 'ONE authored' in why
        assert mode.value in why, 'the explanation must name what was asked for'


def test_single_waypoint_does_not_flag_once_as_degraded():
    effective, why = gt.effective_traversal(GoalTraversal.ONCE, 1)
    assert effective is GoalTraversal.ONCE
    assert 'DEGRADED' not in why


def test_zero_waypoints_degrades_repeating_modes_to_cyclic_over_the_fallback():
    for mode in (GoalTraversal.CYCLIC, GoalTraversal.RECIPROCATE):
        effective, why = gt.effective_traversal(mode, 0)
        assert effective is GoalTraversal.CYCLIC
        if mode is GoalTraversal.RECIPROCATE:
            assert 'DEGRADED' in why
            assert 'fallback' in why


def test_two_waypoints_is_delivered_not_degraded():
    """Cycling two waypoints already reciprocates, so this is not a degradation."""
    effective, why = gt.effective_traversal(GoalTraversal.RECIPROCATE, 2)
    assert effective is GoalTraversal.RECIPROCATE
    assert 'DEGRADED' not in why
    assert 'not expanded' in why, 'a reader must not mistake this for a bug'


def test_every_degradation_explanation_names_the_effective_behaviour():
    """An explanation that does not say what will happen is not an explanation."""
    for mode in GoalTraversal:
        for n in (0, 1, 2, 3):
            effective, why = gt.effective_traversal(mode, n)
            assert why and len(why) > 20
            if effective is not mode:
                assert 'DEGRADED' in why, (
                    f'{mode.value} with {n} waypoints silently became '
                    f'{effective.value} without saying so'
                )


def test_hunav_logs_the_effective_behaviour_not_just_the_requested_mode():
    """Guard that the degradation is stated in the log, not left to deduction.

    Deducing "authored=1 so it will stop" from a count is exactly what fails at
    three in the morning, so the log line must carry both the requested and the
    effective mode.
    """
    src = _read_or_skip(_HUNAV_PY)
    assert 'effective_traversal(' in src, 'effective behaviour is never computed'
    assert 'requested=' in src and 'effective=' in src, (
        'the log line must report BOTH the requested and the effective mode'
    )


def test_shipped_default_yaml_keeps_todays_behaviour():
    """The new mode must be opt-in: the shipped default stays ``once``."""
    default_yaml = (
        _SRC_ARENA / 'arena_bringup' / 'configs' / 'hunav' / 'default.yaml'
    )
    if not default_yaml.is_file():
        pytest.skip(f'{default_yaml} not present')
    import yaml
    cfg = yaml.safe_load(default_yaml.read_text())
    assert cfg.get('goal_traversal') == 'once', (
        'shipping anything other than "once" by default would silently change '
        'the benchmark for every existing run configuration'
    )
    assert cfg.get('cyclic_goals') is False, 'legacy key must stay consistent'


# --------------------------------------------------------------------------
# Wiring guards: an opt-in that is declared but not forwarded is inert
# --------------------------------------------------------------------------
#
# A launch argument can be written without being declared (so `arg:=value` is
# rejected), declared without being forwarded (so the node never sees it), or
# forwarded under a different name than the consumer reads. Each of those leaves
# the mode looking wired up while doing nothing, which is this project's most
# repeated defect shape. These guards pin each hop by name.

_PARAM_NAME = 'pedestrian_goal_traversal'
_ARENA_LAUNCH = _SRC_ARENA / 'arena_bringup' / 'launch' / 'arena.launch.py'
_INTERNNAV_EVAL = (
    _SRC_ARENA / 'arena_bringup' / 'arena_bringup' / 'internnav_eval.py'
)
_HUNAV_PY = (
    _TASK_GENERATOR / 'task_generator' / 'simulators' / 'human' / 'hunav' / 'hunav.py'
)


def _read_or_skip(path):
    if not path.is_file():
        pytest.skip(f'{path} not present')
    return path.read_text()


def test_launch_argument_is_declared_and_defaults_to_silent():
    src = _read_or_skip(_ARENA_LAUNCH)
    assert f"name='{_PARAM_NAME}'" in src, 'launch argument is not declared'
    decl = src.split(f"name='{_PARAM_NAME}'", 1)[1][:400]
    assert "default_value=''" in decl, (
        'the run-level layer must default to silent so configs/hunav/default.yaml '
        'stays in charge and existing run configurations are unaffected'
    )


def test_launch_argument_is_declared_after_auto_append_is_armed():
    """Declared before ``auto_append`` is armed means it never reaches the LD."""
    src = _read_or_skip(_ARENA_LAUNCH)
    lines = src.splitlines()
    arm = next((i for i, line in enumerate(lines)
                if 'LaunchArgument.auto_append(' in line), None)
    decl = next((i for i, line in enumerate(lines)
                 if f"name='{_PARAM_NAME}'" in line), None)
    assert arm is not None and decl is not None
    assert arm < decl, (
        'the argument is created before auto_append is armed, so it would not be '
        'added to the LaunchDescription and `pedestrian_goal_traversal:=...` '
        'would be rejected at launch time'
    )


def test_launch_argument_is_forwarded_to_the_task_generator():
    src = _read_or_skip(_ARENA_LAUNCH)
    assert f'**{_PARAM_NAME}.dict' in src.replace(' ', ''), (
        'declared but not forwarded: the task generator node would never receive '
        'the parameter and the mode would be silently inert'
    )


def test_hunav_reads_the_same_parameter_name_that_launch_forwards():
    src = _read_or_skip(_HUNAV_PY)
    assert f"'{_PARAM_NAME}'" in src, (
        'consumer reads a different name than launch forwards'
    )


def test_hunav_applies_the_run_level_value_as_a_fallback_not_an_override():
    """A run-wide switch must not silently overrule a per-pedestrian setting."""
    src = _read_or_skip(_HUNAV_PY)
    assert 'default_goal_traversal=self._goal_traversal_default' in src


def test_evaluator_records_the_mode_in_the_run_manifest():
    """The regime must be recoverable from artifacts, not from archaeology."""
    src = _read_or_skip(_INTERNNAV_EVAL)
    assert "'--pedestrian-goal-traversal'" in src, 'no CLI flag'
    assert f'{_PARAM_NAME}:=' in src, 'not passed on the launch command line'
    assert f"'{_PARAM_NAME}': args.{_PARAM_NAME}" in src, (
        'not recorded under run_manifest.yaml parameters, so a future reader '
        'could not tell which pedestrian regime produced a number'
    )
    assert f"'{_PARAM_NAME}_source'" in src, (
        'the provenance of the value is not recorded; "" is ambiguous between '
        '"not set" and "set to the default"'
    )


# --------------------------------------------------------------------------
# Delivery: the launch CHAIN, not just its first link
# --------------------------------------------------------------------------
#
# These exist because a case03 evaluation slot was spent on a void run. The
# argument was declared in arena.launch.py and forwarded into the included
# task_generator description, an in-process capability probe confirmed the build
# supported the behaviour, and the value still never reached the node -- because
# forwarding into an included launch description does nothing unless that
# description ALSO declares the argument and lists it in the node's `parameters`
# allowlist. Every check that ran was on the producer's side of a boundary the
# value never crossed.

_TG_LAUNCH = _TASK_GENERATOR / 'launch' / 'task_generator.launch.py'


def test_included_launch_description_declares_the_argument():
    src = _read_or_skip(_TG_LAUNCH)
    assert 'name="pedestrian_goal_traversal"' in src.replace(' ', ''), (
        'task_generator.launch.py must DECLARE the argument; arena.launch.py '
        'forwarding it is not enough and fails silently'
    )


def test_included_launch_description_passes_it_to_the_node():
    src = _read_or_skip(_TG_LAUNCH).replace(' ', '')
    assert '**pedestrian_goal_traversal.str_param' in src, (
        "the task generator node's `parameters` block is an explicit allowlist; "
        'an argument absent from it never becomes a ROS parameter'
    )


def test_the_argument_is_declared_at_every_hop_of_the_chain():
    """All three sites, by name, so no single hop can be dropped silently."""
    arena = _read_or_skip(_ARENA_LAUNCH).replace(' ', '')
    tg = _read_or_skip(_TG_LAUNCH).replace(' ', '')
    hops = {
        'arena declares': "name='pedestrian_goal_traversal'" in arena,
        'arena forwards': '**pedestrian_goal_traversal.dict' in arena,
        'included declares': 'name="pedestrian_goal_traversal"' in tg,
        'included parameterises': '**pedestrian_goal_traversal.str_param' in tg,
    }
    missing = [k for k, v in hops.items() if not v]
    assert not missing, f'launch chain broken at: {missing}'


def test_the_argument_travels_the_same_hops_as_a_known_working_parameter():
    """Structural parity with a parameter that is known to arrive.

    `episode_start_delay_sec` demonstrably reaches the node. Any hop it has that
    the new argument lacks is a hop the new argument will silently lose.
    """
    tg = _read_or_skip(_TG_LAUNCH).replace(' ', '')
    for site in ('name="episode_start_delay_sec"', '**episode_start_delay_sec.param('):
        assert site in tg, 'reference parameter changed shape; update this test'
    assert 'name="pedestrian_goal_traversal"' in tg
    assert '**pedestrian_goal_traversal.str_param' in tg


def test_hunav_refuses_to_run_when_the_request_did_not_arrive():
    """The consumer-side half: an undelivered request must not degrade quietly.

    The void run's per-agent line read `requested=once effective=once` -- those
    AGREED, so an `effective != requested` check could not have caught it. The
    mismatch was between the run-level request and what the node resolved, which
    is only detectable with a second, independent channel.
    """
    src = _read_or_skip(_HUNAV_PY)
    assert 'ARENA_EVAL_PEDESTRIAN_GOAL_TRAVERSAL' in src, (
        'no independent delivery channel: "not requested" and "requested but '
        'undelivered" would remain indistinguishable at the consumer'
    )
    assert 'NOT DELIVERED' in src, 'undelivered request must raise, not warn'
    assert 'ALTERED in transit' in src, 'a changed value must also raise'


def test_evaluator_exports_the_independent_delivery_channel():
    src = _read_or_skip(_INTERNNAV_EVAL)
    assert "env['ARENA_EVAL_PEDESTRIAN_GOAL_TRAVERSAL']" in src
    assert 'included_declares_arg' in src and 'included_parameterises_arg' in src, (
        'the pre-Isaac gate must check the included launch description too; '
        'checking only arena.launch.py is what let the void run start'
    )
