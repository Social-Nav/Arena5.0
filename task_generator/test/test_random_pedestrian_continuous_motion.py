"""Random pedestrians use regular navigation without invented BT goals."""

import asyncio
from types import SimpleNamespace

import nav_msgs.msg
import numpy as np

from task_generator.manager.world_manager.utils import WorldMap
from task_generator.manager.world_manager.world_manager import WorldManager
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
    def __init__(self):
        msg = nav_msgs.msg.OccupancyGrid()
        msg.info.height = 40
        msg.info.width = 50
        msg.info.resolution = 1.0
        msg.info.origin.orientation.w = 1.0
        msg.data = [0] * (msg.info.height * msg.info.width)
        self.map = WorldMap.from_costmap(msg)

    def _occupancy_to_available(self, occupancy, safe_dist):
        manager = WorldManager.__new__(WorldManager)
        return manager._occupancy_to_available(occupancy, safe_dist)

    def get_positions_on_map(self, n, safe_dist):
        raise AssertionError('TM_Random pedestrian routes must use DensityAwarePositionSampler')


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
    sampled_goals = [(p.x, p.y) for p in obstacle.waypoints]
    assert len(sampled_goals) == 2

    hunav = HunavDynamicObstacle.from_dynamic_obstacle(obstacle)
    message = hunav.to_msg()

    assert hunav.behavior_tree == 'BTRegularNav.xml'
    assert hunav.behavior_tree != 'default.xml'
    assert [(p.position.x, p.position.y) for p in message.goals] == sampled_goals
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
