import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src/arena_simulation_setup/shared/dynamic_waypoints.py"
)
SPEC = importlib.util.spec_from_file_location("dynamic_waypoints", MODULE_PATH)
dynamic_waypoints = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dynamic_waypoints)


@pytest.mark.parametrize(
    ("raw", "expected_waypoints", "expected_velocities"),
    [
        ([[1, 0, 0], [2, 0, 0]], [[1, 0, 0], [2, 0, 0]], [None, None]),
        (
            [[1, 0, 0, 0.4], [2, 0, 0, 0.6]],
            [[1, 0, 0], [2, 0, 0]],
            [0.4, 0.6],
        ),
        (
            [
                {"pose": [1, 0, 0], "desired_velocity": 0.5},
                {"x": 2, "y": 0},
            ],
            [[1, 0, 0], [2, 0, 0]],
            [0.5, None],
        ),
    ],
)
def test_normalize_dynamic_waypoints(
    raw, expected_waypoints, expected_velocities
):
    waypoints, velocities = dynamic_waypoints.normalize_dynamic_waypoints(raw)
    assert waypoints == expected_waypoints
    assert velocities == expected_velocities


@pytest.mark.parametrize("velocity", [0.0, -0.2])
def test_waypoint_velocity_must_be_positive(velocity):
    with pytest.raises(ValueError, match="must be positive"):
        dynamic_waypoints.normalize_dynamic_waypoints([[1, 2, 3, velocity]])
