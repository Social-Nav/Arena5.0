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
from task_generator.manager.world_manager.utils import WorldOccupancy
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


def test_direct_yaml_loader_preserves_asymmetric_origin_and_in_bounds_round_trip(tmp_path):
    """The production direct loader uses the same internal origin convention."""
    Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(tmp_path / 'map.png')
    map_yaml = tmp_path / 'map.yaml'
    map_yaml.write_text(yaml.safe_dump({
        'image': 'map.png',
        'resolution': 0.5,
        'origin': [-3.25, 7.75, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }))

    manager = WorldManagerROS.__new__(WorldManagerROS)
    manager._origin = Position(x=-3.25, y=7.75)
    manager._NodeInterface__node = SimpleNamespace(
        sim_time=SimpleNamespace(to_msg=lambda: nav_msgs.msg.OccupancyGrid().info.map_load_time)
    )

    world_map = manager._load_world_map_from_yaml(str(map_yaml))

    assert (world_map.origin.x, world_map.origin.y) == pytest.approx((7.75, -3.25))
    assert manager._origin is None
    position = world_map.tf_grid2pos((2, 3))
    assert (position.x, position.y) == pytest.approx((-1.75, 8.75))
    runtime_row = world_map.shape[0] - 1 - round((position.y - 7.75) / 0.5)
    runtime_column = round((position.x - -3.25) / 0.5)
    assert (runtime_row, runtime_column) == (3, 3)
    assert 0 <= runtime_row < world_map.shape[0]
    assert 0 <= runtime_column < world_map.shape[1]


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
