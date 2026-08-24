"""Functional tests for no-pedestrian dummy-human semantics."""

import asyncio
from types import SimpleNamespace

from task_generator.simulators.human.dummy import DummyHumanSimulator
from task_generator.simulators.human.utils import ObstacleLayer


class _Logger:
    def get_child(self, _name):
        return self

    def debug(self, _message):
        pass

    def info(self, _message):
        pass


class _Node:
    def __init__(self):
        self.logger = _Logger()
        self.publisher_calls = []
        self.client_calls = []
        self.service_calls = []

    def get_logger(self):
        return self.logger

    def create_publisher(self, message_type, topic, qos):
        self.publisher_calls.append((message_type, topic, qos))
        return object()

    def create_client(self, *args, **kwargs):
        self.client_calls.append((args, kwargs))
        return object()

    def create_service(self, *args, **kwargs):
        self.service_calls.append((args, kwargs))
        return object()


class _Simulator:
    def __init__(self):
        self.pedestrian_spawn_calls = []
        self.pedestrian_move_calls = []

    async def pedestrian_spawn(self, obstacles):
        self.pedestrian_spawn_calls.append(tuple(obstacles))
        return (True,) * len(obstacles)

    async def pedestrian_move(self, obstacles):
        self.pedestrian_move_calls.append(tuple(obstacles))
        return (True,) * len(obstacles)


class _MaterializingHumanSimulator(DummyHumanSimulator):
    """Control implementing the old dummy hook behavior."""

    async def _spawn_dynamic_obstacles_impl(self, obstacles):
        return obstacles


def _make_human(simulator_type=DummyHumanSimulator):
    node = _Node()
    simulator = _Simulator()
    human = simulator_type(
        node=node,
        namespace=lambda suffix: f'/test{suffix}',
        simulator=simulator,
    )
    return human, node, simulator


def test_dummy_first_spawn_registers_but_does_not_materialize_pedestrians():
    human, node, simulator = _make_human()
    obstacles = (SimpleNamespace(name='pedestrian_1'),)

    assert asyncio.run(human._spawn_dynamic_obstacles_impl(obstacles)) == ()
    result = asyncio.run(human.spawn_dynamic_obstacles(obstacles))

    known = human._known_obstacles.get('pedestrian_1')
    assert result is None
    assert known is not None
    assert known.obstacle is obstacles[0]
    assert known.spawned is False
    assert known.layer is ObstacleLayer.UNUSED
    assert simulator.pedestrian_spawn_calls == []
    assert simulator.pedestrian_move_calls == []

    # Construction creates only BaseHumanSimulator's goal publisher. Dummy adds
    # no pedestrian publisher, client, or service of its own.
    assert [topic for _, topic, _ in node.publisher_calls] == ['/test/goal']
    assert node.client_calls == []
    assert node.service_calls == []


def test_materializing_hook_control_would_spawn_the_same_input():
    human, _node, simulator = _make_human(_MaterializingHumanSimulator)
    obstacles = (SimpleNamespace(name='pedestrian_1'),)

    asyncio.run(human.spawn_dynamic_obstacles(obstacles))

    known = human._known_obstacles.get('pedestrian_1')
    assert simulator.pedestrian_spawn_calls == [obstacles]
    assert simulator.pedestrian_move_calls == []
    assert known is not None
    assert known.spawned is True
    assert known.layer is ObstacleLayer.INUSE
