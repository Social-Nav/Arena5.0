"""Geometry and integration tests for density-aware pedestrian sampling."""

from types import SimpleNamespace

import nav_msgs.msg
import numpy as np
import pytest

from task_generator.manager.world_manager.density_aware_sampler import (
    DensityAwarePositionSampler,
    DensityAwareSamplingConfig,
    DensityAwareSamplingError,
)
from task_generator.manager.world_manager.utils import WorldMap
from task_generator.manager.world_manager.world_manager import WorldManager


def _world(grid: np.ndarray, resolution: float = 1.0):
    msg = nav_msgs.msg.OccupancyGrid()
    msg.info.height, msg.info.width = grid.shape
    msg.info.resolution = resolution
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.reshape(-1).tolist()
    return WorldMap.from_costmap(msg)


def _provider(occupancy, safe_dist):
    manager = WorldManager.__new__(WorldManager)
    return manager._occupancy_to_available(occupancy, safe_dist)


def _sample(seed=7, count=10, grid=None, config=None):
    if grid is None:
        grid = np.zeros((80, 100), dtype=np.int8)
    return DensityAwarePositionSampler(
        world_map=_world(grid),
        rng=np.random.default_rng(seed),
        candidate_provider=_provider,
        config=config or DensityAwareSamplingConfig(),
    ).sample(count)


@pytest.mark.parametrize('count', [5, 10, 20])
def test_density_sizes_are_complete_and_start_separated(count):
    sample = _sample(count=count)
    assert len(sample.routes) == count
    starts = np.array([[route.start.x, route.start.y] for route in sample.routes])
    distances = np.linalg.norm(starts[:, None, :] - starts[None, :, :], axis=2)
    distances[distances == 0] = np.inf
    assert distances.min() >= 1.0
    assert all(len(route.goals) == 2 for route in sample.routes)


def test_sampling_is_deterministic_and_route_goals_are_not_globally_packed():
    first = _sample(seed=22)
    second = _sample(seed=22)
    encode = lambda sample: [
        [(route.start.x, route.start.y), *[(goal.x, goal.y) for goal in route.goals]]
        for route in sample.routes
    ]
    assert encode(first) == encode(second)
    goals = [(goal.x, goal.y) for route in first.routes for goal in route.goals]
    assert len(goals) == len(set(goals))
    # Route goals obey only per-leg length, not the starts' global packing rule.
    assert all(
        np.linalg.norm(np.array([b.x - a.x, b.y - a.y])) >= 1.0
        for route in first.routes
        for a, b in zip([route.start, *route.goals[:-1]], route.goals)
    )


def test_connected_components_keep_each_route_on_one_footprint_safe_island():
    grid = np.zeros((60, 100), dtype=np.int8)
    grid[:, 49:52] = 100
    sample = _sample(count=10, grid=grid)
    for route in sample.routes:
        xs = [route.start.x, *[goal.x for goal in route.goals]]
        assert all(x < 49 for x in xs) or all(x > 52 for x in xs)


def test_exhaustion_is_fail_closed():
    grid = np.full((10, 10), 100, dtype=np.int8)
    with pytest.raises(DensityAwareSamplingError, match='not enough footprint-safe'):
        _sample(count=1, grid=grid)


def test_legacy_world_manager_sampler_remains_callable():
    grid = np.zeros((20, 20), dtype=np.int8)
    world = _world(grid)
    manager = WorldManager.__new__(WorldManager)
    manager._map = world
    manager._NodeInterface__node = SimpleNamespace(
        conf=SimpleNamespace(General=SimpleNamespace(RNG=SimpleNamespace(value=np.random.default_rng(3))))
    )
    points = manager.get_positions_on_map(n=2, safe_dist=1.0, forbid=False)
    assert len(points) == 2
