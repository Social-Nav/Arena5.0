"""Focused regressions for random-map origin and convolution alignment."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import nav_msgs.msg
import numpy as np
import pytest
import yaml
from PIL import Image

from arena_simulation_setup.shared import Position
from task_generator.manager.world_manager import world_manager_ros
from task_generator.manager.world_manager.utils import WorldMap, WorldOccupancy
from task_generator.manager.world_manager.world_manager import WorldManager
from task_generator.manager.world_manager.world_manager_ros import WorldManagerROS


class _AlwaysEarlier:
    """Minimal previous map timestamp accepted by ``_map_callback``."""

    def __le__(self, other):
        del other
        return True


class _ResolvedWorld:
    path = Path('/tmp/random-map-alignment-world')

    def load(self):
        return SimpleNamespace(name='fixture-world-description')


class _WorldIdentifier:
    def __init__(self, name):
        self.name = name

    async def resolve(self):
        assert self.name == 'asymmetric_origin_world'
        return _ResolvedWorld()


def _costmap(*, origin_x: float, origin_y: float) -> nav_msgs.msg.OccupancyGrid:
    costmap = nav_msgs.msg.OccupancyGrid()
    costmap.info.height = 3
    costmap.info.width = 5
    costmap.info.resolution = 0.25
    costmap.info.origin.position.x = origin_x
    costmap.info.origin.position.y = origin_y
    costmap.info.origin.orientation.w = 1.0
    costmap.info.map_load_time.sec = 1
    costmap.data = [0] * (costmap.info.height * costmap.info.width)
    return costmap


def test_map_callback_preserves_asymmetric_origin_in_world_map_coordinates(monkeypatch):
    """The deferred source-map origin must use ``WorldMap``'s swapped axes."""
    captured = {}
    manager = WorldManagerROS.__new__(WorldManagerROS)
    manager._map = SimpleNamespace(time=_AlwaysEarlier())
    manager._origin = Position(x=12.5, y=-3.75)
    manager._world_name = 'asymmetric_origin_world'
    manager._map_name = None
    manager._callbacks = []
    manager.update_world = lambda **kwargs: captured.update(kwargs)

    monkeypatch.setattr(world_manager_ros.World, 'WorldIdentifier', _WorldIdentifier)
    monkeypatch.setattr(
        world_manager_ros,
        'DynamicPaths',
        SimpleNamespace(WORLD=SimpleNamespace(path=None)),
    )

    asyncio.run(manager._map_callback(_costmap(origin_x=101.0, origin_y=202.0)))

    origin = captured['world_map'].origin
    assert (origin.x, origin.y) == pytest.approx((-3.75, 12.5))
    assert manager._origin is None
    assert manager._map_name == 'asymmetric_origin_world'


def test_direct_yaml_loader_matches_ros_costmap_y_orientation_and_origin(tmp_path):
    """Direct PIL loading is identical to ROS bottom-origin OccupancyGrid data."""
    image = np.array([
        [0, 255, 127, 255],
        [255, 255, 255, 0],
        [255, 0, 255, 255],
    ], dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / 'map.png')
    map_yaml = tmp_path / 'map.yaml'
    metadata = {
        'image': 'map.png',
        'resolution': 0.5,
        'origin': [-3.25, 7.75, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }
    map_yaml.write_text(yaml.safe_dump(metadata))

    manager = WorldManagerROS.__new__(WorldManagerROS)
    manager._origin = Position(x=-3.25, y=7.75)
    manager._NodeInterface__node = SimpleNamespace(
        sim_time=SimpleNamespace(to_msg=lambda: nav_msgs.msg.OccupancyGrid().info.map_load_time)
    )
    direct = manager._load_world_map_from_yaml(str(map_yaml))

    ros_grid = nav_msgs.msg.OccupancyGrid()
    ros_grid.info.height, ros_grid.info.width = image.shape
    ros_grid.info.resolution = metadata['resolution']
    ros_grid.info.origin.position.x, ros_grid.info.origin.position.y = metadata['origin'][:2]
    ros_grid.info.origin.orientation.w = 1.0
    ros_grid.data = [0, 100, 0, 0, 0, 0, 0, 100, 100, 0, -1, 0]
    expected = WorldMap.from_costmap(ros_grid)

    assert (direct.origin.x, direct.origin.y) == pytest.approx((7.75, -3.25))
    assert manager._origin is None
    assert np.array_equal(direct.occupancy.grid, expected.occupancy.grid)
    assert WorldOccupancy.full(direct.occupancy.grid[0, 1])
    assert WorldOccupancy.full(direct.occupancy.grid[1, 3])
    assert WorldOccupancy.full(direct.occupancy.grid[2, 0])
    assert WorldOccupancy.empty(direct.occupancy.grid[0, 0])
    assert WorldOccupancy.empty(direct.occupancy.grid[2, 2])

    position = direct.tf_grid2pos((0, 1))
    assert (position.x, position.y) == pytest.approx((-2.75, 7.75))


class _DeterministicRng:
    def choice(self, population, size, replace=False):
        assert not replace
        return np.arange(size) % population


class _SamplerNode:
    conf = SimpleNamespace(General=SimpleNamespace(RNG=SimpleNamespace(value=_DeterministicRng())))

    def get_logger(self):
        return SimpleNamespace(get_child=lambda _name: SimpleNamespace())


def _sampler_manager(grid):
    manager = WorldManager.__new__(WorldManager)
    manager._NodeInterface__node = _SamplerNode()
    costmap = nav_msgs.msg.OccupancyGrid()
    costmap.info.height, costmap.info.width = grid.shape
    costmap.info.resolution = 1.0
    costmap.info.origin.orientation.w = 1.0
    costmap.data = grid.reshape(-1).tolist()
    manager._map = WorldMap.from_costmap(costmap)
    return manager


def test_sampling_exhaustion_raises_without_unchecked_partial_positions():
    manager = _sampler_manager(np.full((7, 7), 100, dtype=np.int8))

    with pytest.raises(RuntimeError, match=r'requested=2, validated=0, missing=2'):
        manager.get_positions_on_map(n=2, safe_dist=1.0)


def test_sufficient_sampling_is_deterministic_and_returns_only_valid_cells():
    grid = np.zeros((9, 9), dtype=np.int8)
    grid[4, 4] = 100

    first = _sampler_manager(grid).get_positions_on_map(n=2, safe_dist=1.0, forbid=False)
    second = _sampler_manager(grid).get_positions_on_map(n=2, safe_dist=1.0, forbid=False)

    assert [(p.x, p.y) for p in first] == [(p.x, p.y) for p in second]
    for point in first:
        row = int(round(point.y))
        column = int(round(9 - point.x))
        assert 0 <= row < 9
        assert 0 <= column < 9
        lo_r, hi_r = max(0, row - 1), min(9, row + 2)
        lo_c, hi_c = max(0, column - 1), min(9, column + 2)
        assert not np.any(grid[lo_r:hi_r, lo_c:hi_c] == 100)


def test_available_candidates_keep_input_shape_and_kernel_center_alignment():
    """Inflation candidates remain input-grid indices, centered on obstacles."""
    occupancy = np.full((7, 9), WorldOccupancy.EMPTY, dtype=np.uint8)
    occupancy[2, 6] = WorldOccupancy.FULL

    manager = WorldManager.__new__(WorldManager)
    actual = {
        tuple(candidate)
        for candidate in manager._occupancy_to_available(occupancy, safe_dist=1)
    }

    expected = {
        (row, column)
        for row in range(1, 6)
        for column in range(1, 8)
        if not (1 <= row <= 3 and 5 <= column <= 7)
    }

    assert actual == expected
    assert len(actual) == 26
    assert all(0 <= row < occupancy.shape[0] for row, _ in actual)
    assert all(0 <= column < occupancy.shape[1] for _, column in actual)
