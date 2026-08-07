"""Message-level tests for pedestrian goal traversal.

These exercise `HunavDynamicObstacle.to_msg()`, so they need the ROS message
packages and are skipped when those are unavailable.  The pure traversal logic is
covered ROS-free in `test_pedestrian_goal_traversal.py`.

What these add over the pure tests: they check the *transmitted* `hunav_msgs/Agent`
-- the goal list and the `cyclic_goals` wire bit that HuNav will actually consume.
They still do not prove the consumer honours it; that requires the real service and
is done by `tmp/lane_cyc/evidence/e2e_positive_control.py`.
"""

import pytest

pytest.importorskip('hunav_msgs.msg', reason='ROS message packages not available')
pytest.importorskip('rclpy', reason='ROS not sourced')

from task_generator.shared import Position                                # noqa: E402
from task_generator.simulators.human.hunav import (                       # noqa: E402
    Goals,
    HunavDynamicObstacle,
    PositionH,
)
from task_generator.simulators.human.hunav.goal_traversal import (        # noqa: E402
    GoalTraversal,
    InvalidGoalTraversal,
)

L_ROUTE = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0)]


def make(mode, waypoints=L_ROUTE, **overrides):
    kwargs = dict(
        name='ped',
        model=HunavDynamicObstacle._default.model,
        extra={},
        id=1,
        type=1,
        skin=0,
        group_id=-1,
        yaw=0.0,
        velocity=None,
        desired_velocity=1.0,
        radius=0.3,
        linear_vel=0.0,
        angular_vel=0.0,
        behavior=HunavDynamicObstacle.Behavior._default,
        behavior_tree='BTRegularNav.xml',
        goal_radius=0.3,
        closest_obs=[],
        init_pose=PositionH(x=0.0, y=0.0, z=1.25, h=0.0),
        goals=Goals([Position(x=x, y=y) for x, y in waypoints]),
        goal_traversal=mode,
        cyclic_goals=GoalTraversal(mode).wire_cyclic_goals
        if not isinstance(mode, GoalTraversal) else mode.wire_cyclic_goals,
    )
    kwargs.update(overrides)
    return HunavDynamicObstacle(**kwargs)


def goal_xy(msg):
    return [(round(g.position.x, 6), round(g.position.y, 6)) for g in msg.goals]


# --------------------------------------------------------------------------
# Transmitted message per mode
# --------------------------------------------------------------------------

def test_once_transmits_authored_waypoints_and_no_cyclic_bit():
    msg = make(GoalTraversal.ONCE).to_msg()
    assert goal_xy(msg) == L_ROUTE
    assert msg.cyclic_goals is False


def test_cyclic_transmits_authored_waypoints_with_the_cyclic_bit():
    msg = make(GoalTraversal.CYCLIC).to_msg()
    assert goal_xy(msg) == L_ROUTE
    assert msg.cyclic_goals is True


def test_reciprocate_transmits_the_mirrored_list_with_the_cyclic_bit():
    msg = make(GoalTraversal.RECIPROCATE).to_msg()
    assert goal_xy(msg) == [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (6.0, 0.0)]
    assert msg.cyclic_goals is True


def test_reciprocate_does_not_touch_the_authored_waypoints_on_the_object():
    """The scenario's waypoints must be untouched; the expansion is transmit-only."""
    obstacle = make(GoalTraversal.RECIPROCATE)
    before = [(p.x, p.y) for p in obstacle.goals]
    obstacle.to_msg()
    after = [(p.x, p.y) for p in obstacle.goals]
    assert before == after == L_ROUTE


def test_to_msg_is_idempotent_for_reciprocate():
    """Calling to_msg twice must not mirror twice."""
    obstacle = make(GoalTraversal.RECIPROCATE)
    first = goal_xy(obstacle.to_msg())
    second = goal_xy(obstacle.to_msg())
    assert first == second
    assert len(first) == 4


# --------------------------------------------------------------------------
# The two fields cannot silently disagree
# --------------------------------------------------------------------------

@pytest.mark.parametrize('mode,bad_bool', [
    (GoalTraversal.ONCE, True),
    (GoalTraversal.CYCLIC, False),
    (GoalTraversal.RECIPROCATE, False),
])
def test_contradicting_cyclic_goals_and_mode_raises(mode, bad_bool):
    """`attrs.evolve(obs, cyclic_goals=...)` alone must not be silently dropped.

    `cyclic_goals` is the wire name and `goal_traversal` the semantic one. If a
    caller changes only the boolean, its request would have no effect, which is
    exactly the accepted-and-ignored shape this project keeps re-introducing.
    """
    obstacle = make(mode, cyclic_goals=bad_bool)
    with pytest.raises(ValueError, match='contradicts'):
        obstacle.to_msg()


# --------------------------------------------------------------------------
# Scenario / config resolution through the real construction paths
# --------------------------------------------------------------------------

def test_scenario_key_reaches_the_message_through_from_dynamic_obstacle():
    """The opt-in must survive the real scenario parsing path.

    Unknown scenario keys reach `extra` via `Named.parse`; this checks the mode
    then survives `from_dynamic_obstacle` and `to_msg` without being dropped.
    """
    cattrs = pytest.importorskip('arena_simulation_setup.utils.cattrs')
    entities = pytest.importorskip('arena_simulation_setup.shared.entities')

    entity = {
        'name': 'ped',
        'model': 'female_adult_business_02',
        'pose': [0.0, 0.0, 0.0],
        'waypoints': [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0]],
        'goal_traversal': 'reciprocate',
    }
    obstacle = cattrs.converter.structure(entity, entities.DynamicObstacle)
    hunav_obstacle = HunavDynamicObstacle.from_dynamic_obstacle(obstacle)
    assert hunav_obstacle.goal_traversal is GoalTraversal.RECIPROCATE

    msg = hunav_obstacle.to_msg()
    assert goal_xy(msg) == [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (6.0, 0.0)]
    assert msg.cyclic_goals is True


def test_run_level_default_applies_when_scenario_is_silent():
    cattrs = pytest.importorskip('arena_simulation_setup.utils.cattrs')
    entities = pytest.importorskip('arena_simulation_setup.shared.entities')
    entity = {
        'name': 'ped',
        'model': 'female_adult_business_02',
        'pose': [0.0, 0.0, 0.0],
        'waypoints': [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0]],
    }
    obstacle = cattrs.converter.structure(entity, entities.DynamicObstacle)
    hunav_obstacle = HunavDynamicObstacle.from_dynamic_obstacle(
        obstacle, default_goal_traversal=GoalTraversal.RECIPROCATE
    )
    assert hunav_obstacle.goal_traversal is GoalTraversal.RECIPROCATE


def test_scenario_beats_the_run_level_default():
    """A run-wide switch must not override an explicit per-pedestrian opt-out."""
    cattrs = pytest.importorskip('arena_simulation_setup.utils.cattrs')
    entities = pytest.importorskip('arena_simulation_setup.shared.entities')
    entity = {
        'name': 'ped',
        'model': 'female_adult_business_02',
        'pose': [0.0, 0.0, 0.0],
        'waypoints': [[0.0, 0.0], [6.0, 0.0]],
        'cyclic_goals': False,
    }
    obstacle = cattrs.converter.structure(entity, entities.DynamicObstacle)
    hunav_obstacle = HunavDynamicObstacle.from_dynamic_obstacle(
        obstacle, default_goal_traversal=GoalTraversal.RECIPROCATE
    )
    assert hunav_obstacle.goal_traversal is GoalTraversal.ONCE


def test_config_parse_accepts_legacy_only_config():
    """A default.yaml with no `goal_traversal` key must still work.

    Otherwise the change would silently require a rebuild before today's
    behaviour was restored.
    """
    legacy = {
        'id': 1, 'group_id': -1, 'skin': 0, 'max_vel': 0.8, 'radius': 0.3,
        'goal_radius': 0.2, 'cyclic_goals': False,
        'init_pose': {'x': 0.0, 'y': 0.0, 'z': 1.25, 'h': 0.0},
        'behavior': {'type': 1},
    }
    parsed = HunavDynamicObstacle.parse(legacy)
    assert parsed.goal_traversal is GoalTraversal.ONCE
    assert parsed.cyclic_goals is False

    parsed_cyclic = HunavDynamicObstacle.parse({**legacy, 'cyclic_goals': True})
    assert parsed_cyclic.goal_traversal is GoalTraversal.CYCLIC
    assert parsed_cyclic.cyclic_goals is True


def test_config_goal_traversal_beats_legacy_key():
    cfg = {
        'id': 1, 'group_id': -1, 'skin': 0, 'max_vel': 0.8, 'radius': 0.3,
        'goal_radius': 0.2, 'cyclic_goals': False, 'goal_traversal': 'reciprocate',
        'init_pose': {'x': 0.0, 'y': 0.0, 'z': 1.25, 'h': 0.0},
        'behavior': {'type': 1},
    }
    parsed = HunavDynamicObstacle.parse(cfg)
    assert parsed.goal_traversal is GoalTraversal.RECIPROCATE
    assert parsed.cyclic_goals is True


def test_invalid_config_value_raises_rather_than_defaulting():
    cfg = {
        'id': 1, 'group_id': -1, 'skin': 0, 'max_vel': 0.8, 'radius': 0.3,
        'goal_radius': 0.2, 'goal_traversal': 'recipricate',
        'init_pose': {'x': 0.0, 'y': 0.0, 'z': 1.25, 'h': 0.0},
        'behavior': {'type': 1},
    }
    with pytest.raises(InvalidGoalTraversal):
        HunavDynamicObstacle.parse(cfg)


# --------------------------------------------------------------------------
# The zero-waypoint fallback path
# --------------------------------------------------------------------------

def test_no_waypoints_uses_the_fallback_and_is_never_mirrored():
    """With no authored waypoints the mirror must not be applied.

    An agent with no waypoints falls back to three hardcoded coordinates that no
    scenario asked for. Reciprocating a route nobody wrote would be a surprising
    thing to infer from `reciprocate`, so on this path it behaves as `cyclic`:
    the fallback list is transmitted verbatim with the mode's wire bit.

    This was found by this test disagreeing with a first implementation that
    mirrored the fallback into four goals, and was then decided deliberately
    rather than by adjusting the assertion to whatever the code did.
    """
    for mode in GoalTraversal:
        msg = make(mode, waypoints=[]).to_msg()
        assert len(msg.goals) == 3, (
            f'{mode.value}: fallback list must be transmitted verbatim, not mirrored'
        )
        assert msg.cyclic_goals is mode.wire_cyclic_goals


def test_no_waypoints_keeps_once_and_cyclic_exactly_as_before():
    """The fallback coordinates and wire bit are the pre-change values."""
    expected = [
        (-3.133759, -4.166653),
        (0.997901, -4.131655),
        (-0.227549, -20.187146),
    ]
    once = make(GoalTraversal.ONCE, waypoints=[]).to_msg()
    assert goal_xy(once) == expected and once.cyclic_goals is False
    cyclic = make(GoalTraversal.CYCLIC, waypoints=[]).to_msg()
    assert goal_xy(cyclic) == expected and cyclic.cyclic_goals is True


def test_default_config_ships_once():
    assert HunavDynamicObstacle._default.goal_traversal is GoalTraversal.ONCE, (
        'the installed configs/hunav/default.yaml must keep todays behaviour; '
        'note _load_config reads the INSTALLED share copy, so this also detects '
        'an installed config that has drifted from the source tree'
    )
