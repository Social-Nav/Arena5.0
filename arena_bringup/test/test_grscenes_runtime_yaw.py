"""Import-to-runtime contract checks; run in the ROS-enabled Arena environment."""
import math
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.tree.World.Scenario import ScenarioView
from test_grscenes_worlds_import import _make_input, _make_bt, _run


ANGLES = (1., -1., 90., -90., 180., -180., 359.)


@pytest.fixture(scope='module')
def imported_worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp('yaw-roundtrip')

    def mutate(world_index, _scenario_name, scenario):
        angle = ANGLES[(world_index - 1) % len(ANGLES)]
        scenario['robot']['pose'][2] = angle
        scenario['robot']['waypoints'][-1][2] = -angle
        scenario['dynamic'][0]['pose'][2] = angle

    inputs = _make_input(root, mutate)
    output = root / 'generated'
    _run(inputs, output, _make_bt(root))
    return output / 'worlds'


def assert_yaw(pose, expected_radians):
    error = pose.orientation.to_yaw() - expected_radians
    assert math.atan2(math.sin(error), math.cos(error)) == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize('index,angle', enumerate(ANGLES, 1))
def test_generated_scenario_loads_robot_and_pedestrian_heading(imported_worlds, index, angle):
    scene = ScenarioView(imported_worlds / f'grscenes_{index}_v1/scenarios/default').load()
    assert_yaw(scene.robots[0].start, math.radians(angle))
    assert_yaw(scene.robots[0].goal, math.radians(-angle))
    assert_yaw(scene.dynamic[0].pose, math.radians(angle))
    assert scene.dynamic[0].waypoints[0].z == 0.


@pytest.mark.parametrize('world,case,expected', [
    ('grscenes_3_v1', 'default_1', -1.419190919314),
    ('grscenes_20_v1', 'default_2', -1.221189249795),
    ('grscenes_10_v1', 'default', 2.854654066844),
    ('grscenes_20_v1', 'default_4', 1.589626509562),
    ('grscenes_8_v1', 'default_4', -.096687575369),
])
def test_existing_benchmark_initial_heading(world, case, expected):
    worlds = Path(__file__).parents[2] / 'arena_simulation_setup/worlds'
    assert_yaw(ScenarioView(worlds / world / 'scenarios' / case).load().robots[0].start, expected)


def test_existing_degree_scenario_semantics_are_preserved(tmp_path):
    (tmp_path / 'scenario.yaml').write_text(yaml.safe_dump({
        'robots': [{'start': [0., 0., 90.], 'goal': [1., 2., -90.]}],
    }))
    scene = ScenarioView(tmp_path).load()
    assert_yaw(scene.robots[0].start, math.pi/2)
    assert_yaw(scene.robots[0].goal, -math.pi/2)
