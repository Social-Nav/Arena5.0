import functools
import itertools
import os
import traceback
import typing
from collections.abc import Iterable

import attrs
import yaml

from arena_simulation_setup.tree import PathView
from arena_simulation_setup.shared import DynamicObstacle, Obstacle, Pose
from arena_simulation_setup.utils.cattrs import converter


def _pose_from_deg(value) -> Pose:
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
        import math
        return Pose.parse([value[0], value[1], math.radians(value[2])])
    return Pose.converter(value)


@attrs.define
class RobotGoal:
    start: Pose = attrs.field(converter=_pose_from_deg)
    goal: Pose = attrs.field(converter=_pose_from_deg)

    @classmethod
    def parse(cls, obj: dict) -> "RobotGoal":
        return cls(
            start=_pose_from_deg(obj.get("start", [])),
            goal=_pose_from_deg(obj.get("goal", [])),
        )

    @classmethod
    def from_scenario_robot(cls, robot: dict) -> "RobotGoal":
        """Parse a single robot entry from a scenario file.

        New format (current):
            robot:
              pose: [x, y, yaw_deg]              -> start
              waypoints: [[x, y, yaw_deg], ...]  -> goal = last waypoint
        Legacy format (still accepted):
            start: [x, y, yaw_deg]
            goal:  [x, y, yaw_deg]
        """
        if "start" in robot or "goal" in robot:
            start = robot.get("start", robot.get("pose", []))
            goal = robot.get("goal", [])
        else:
            start = robot.get("pose", [])
            waypoints = robot.get("waypoints") or []
            goal = waypoints[-1] if waypoints else start
        return cls(start=start, goal=goal)


@attrs.define
class Scenario:
    static: list[Obstacle] = attrs.field(factory=list)
    dynamic: list[DynamicObstacle] = attrs.field(factory=list)
    robots: list[RobotGoal] = attrs.field(factory=list)


class ScenarioView(PathView):

    _names: typing.ClassVar[Iterable[str]] = [
        "scenario.yaml",
        "scenario.json",
    ]

    @functools.cached_property
    def scenario_path(self) -> str:
        """
        Get the path to the scenario file.
        """
        prefix = functools.partial(os.path.join, self.path)
        scenario = next(
            (
                p
                for p
                in map(
                    prefix,
                    self._names
                )
                if os.path.isfile(p)
            ),
            prefix(next(iter(self._names)))
        )
        return scenario

    def load_legacy(self) -> Scenario:
        with open(self.scenario_path, 'r') as f:
            scenario = yaml.safe_load(f)

        assert isinstance(scenario, dict), "Scenario file must contain a dictionary at the top level."

        return Scenario(
            static=[
                converter.structure({**obs, **dict(included_from=self.path)}, Obstacle)
                for obs
                in itertools.chain(
                    scenario.get("obstacles", {}).get("static", []),
                    scenario.get("obstacles", {}).get("interactive", [])
                )
            ],
            dynamic=[
                converter.structure({**obs, **dict(included_from=self.path)}, DynamicObstacle)
                for obs
                in scenario.get("obstacles", {}).get("dynamic", [])
            ],
            robots=self._parse_robots(scenario)
        )

    @staticmethod
    def _parse_robots(raw: dict) -> list[RobotGoal]:
        """Extract robot start/goal(s) from a raw scenario dict.

        Accepts the new singular ``robot:`` mapping, the plural ``robots:``
        list, and the legacy ``start``/``goal`` entries interchangeably.
        """
        if not isinstance(raw, dict):
            return []
        robots = raw.get("robots")
        if robots is None:
            robot = raw.get("robot")
            robots = [robot] if isinstance(robot, dict) else []
        return [
            RobotGoal.from_scenario_robot(r)
            for r in robots
            if isinstance(r, dict)
        ]

    def load(self) -> Scenario:
        load_exc: Exception
        try:
            with open(self.scenario_path, 'r') as f:
                raw = yaml.safe_load(f)
                scenario = converter.structure(raw, Scenario)
                scenario.robots = self._parse_robots(raw)
                for obj in itertools.chain(scenario.static, scenario.dynamic):
                    obj.included_from = self.path
            return scenario
        except Exception as e:
            load_exc = e

        legacy_exc: Exception
        try:
            scenario = self.load_legacy()
            import warnings
            warnings.warn(
                "Loading Scenario in legacy format.",
                DeprecationWarning,
                stacklevel=2
            )
            return scenario
        except Exception as e:
            legacy_exc = e

        raise RuntimeError(
            f"Failed to load scenario from {self.scenario_path}:\n"
            f" - New format error: {load_exc}\n{''.join(traceback.format_exception(type(load_exc), load_exc, load_exc.__traceback__))}\n"
            f" - Legacy format error: {legacy_exc}\n{''.join(traceback.format_exception(type(legacy_exc), legacy_exc, legacy_exc.__traceback__))}"
        )
