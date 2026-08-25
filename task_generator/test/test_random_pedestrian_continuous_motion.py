"""Random pedestrians use regular navigation without invented BT goals."""

import asyncio
from types import SimpleNamespace

import numpy as np

from task_generator.simulators.human.hunav import HunavDynamicObstacle
from task_generator.tasks.obstacles.random import TM_Random, _Config


class _Logger:
    def get_child(self, name):
        del name
        return self

    def warn(self, message):
        del message


class _Node:
    def __init__(self):
        self.conf = SimpleNamespace(
            General=SimpleNamespace(RNG=SimpleNamespace(value=np.random.default_rng(17)))
        )

    def get_logger(self):
        return _Logger()


class _WorldManager:
    def get_positions_on_map(self, n, safe_dist):
        assert n == 3
        assert safe_dist == 1
        from arena_simulation_setup.shared import Position
        return [Position(x=float(i), y=float(i + 10)) for i in range(n)]


def _random_mode():
    mode = TM_Random.__new__(TM_Random)
    mode._NodeInterface__node = _Node()
    mode._PROPS = SimpleNamespace(world_manager=_WorldManager())
    mode._config = _Config(
        N_STATIC_OBSTACLES=SimpleNamespace(value=(0, 0)),
        N_INTERACTIVE_OBSTACLES=SimpleNamespace(value=(0, 0)),
        N_DYNAMIC_OBSTACLES=SimpleNamespace(value=(1, 1)),
        MODELS_STATIC_OBSTACLES=SimpleNamespace(value=[]),
        MODELS_INTERACTIVE_OBSTACLES=SimpleNamespace(value=[]),
        MODELS_DYNAMIC_OBSTACLES=SimpleNamespace(value=['female_adult_business_02']),
    )
    return mode


def test_random_pedestrian_uses_regular_navigation_tree_and_preserves_sampled_goals():
    obstacles, dynamic = asyncio.run(_random_mode().reset())

    assert obstacles == []
    assert len(dynamic) == 1
    obstacle = dynamic[0]
    assert obstacle.extra['behavior_tree'] == 'BTRegularNav.xml'
    assert [(p.x, p.y) for p in obstacle.waypoints] == [(1.0, 11.0), (2.0, 12.0)]

    hunav = HunavDynamicObstacle.from_dynamic_obstacle(obstacle)
    message = hunav.to_msg()

    assert hunav.behavior_tree == 'BTRegularNav.xml'
    assert hunav.behavior_tree != 'default.xml'
    assert [(p.position.x, p.position.y) for p in message.goals] == [
        (1.0, 11.0),
        (2.0, 12.0),
    ]
    assert (0.0, 0.0) not in [(p.position.x, p.position.y) for p in message.goals]
    assert (5.0, 5.0) not in [(p.position.x, p.position.y) for p in message.goals]


def test_scenario_authored_behavior_tree_override_still_wins():
    from arena_simulation_setup.shared import DynamicObstacle, Pose, Position

    scenario = DynamicObstacle(
        name='scenario_agent',
        model='female_adult_business_02',
        pose=Pose(Position(x=0.0, y=0.0)),
        waypoints=[Position(x=1.0, y=1.0), Position(x=2.0, y=2.0)],
        extra={'behavior_tree': './authored_tree.xml'},
    )
    scenario.included_from = __import__('pathlib').Path('/tmp/scenario.yaml')

    hunav = HunavDynamicObstacle.from_dynamic_obstacle(scenario)

    assert hunav.behavior_tree.endswith('/authored_tree.xml')
    assert hunav.behavior_tree != 'BTRegularNav.xml'
