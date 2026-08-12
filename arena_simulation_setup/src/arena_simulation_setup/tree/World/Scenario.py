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


# scenario.yaml writes headings in degrees; Pose.converter_deg does the conversion.
# Kept as a module-level alias because it is used as an attrs converter below.
_pose_from_deg = Pose.converter_deg


@attrs.define
class RobotGoal:
    start: Pose = attrs.field(converter=_pose_from_deg)
    goal: Pose = attrs.field(converter=_pose_from_deg)
    # Optional robot social personality preset (passive / neutral / aggressive). Selects a bundle
    # under configs/nav2/profiles/<name>/profile.yaml that is pushed to the MPC + social costmap
    # layers at reset. None -> caller falls back to "neutral".
    social_attributes: typing.Optional[str] = None
    # Toggle for the proactive social-yielding pipeline -- the ONLY place a scenario file can set
    # it. TRI-STATE: True/False = explicit, None = field absent (defer to the launch arg, then
    # False). Lives next to social_attributes because both describe this robot's social behaviour,
    # and this is where you look when tuning one robot.
    #
    # SCOPE CAVEAT: the pipeline is gated by ONE latched topic, /social_yielding/enabled, shared by
    # the trigger and orchestrator nodes. So with several robots in a scenario this is not actually
    # per-robot -- the last robot carrying the field wins, and task_generator warns if two robots
    # disagree. Today's scenario format has a single `robot:` key, so the distinction is moot.
    #
    # An explicit `social_yielding:=true|false` launch arg outranks this.
    social_yielding: typing.Optional[bool] = None
    # Per-robot cruise-speed override, in m/s. None = field absent -> the profile's
    # trajectorizer.desired_linear_vel wins (passive 0.35, aggressive 0.8).
    #
    # Set it when a scenario needs one specific speed without inventing a whole profile --
    # e.g. an overtaking scenario where the robot/pedestrian speed differential IS the
    # point. Pushed AFTER the profile at reset, so it overrides just this one key and
    # leaves every other profile knob (proxemics, social weights, the ceiling) intact.
    #
    # NOT A CEILING. Two hard clamps sit above it and neither moves with it:
    # optimizer.max_linear_velocity (the Ceres upper bound on vx) and
    # velocity_smoother_max_velocity[0] in model_params.yaml -- both 0.8 today. Asking for
    # more is silently clipped, so task_generator warns rather than letting it look applied.
    desired_linear_vel: typing.Optional[float] = None

    @staticmethod
    def _opt_bool(v) -> typing.Optional[bool]:
        """None stays None (absent != False); anything else is coerced."""
        return None if v is None else bool(v)

    @staticmethod
    def _opt_float(v) -> typing.Optional[float]:
        """None stays None (absent != 0.0); anything else is coerced to float."""
        return None if v is None else float(v)

    @classmethod
    def parse(cls, obj: dict) -> "RobotGoal":
        return cls(
            start=_pose_from_deg(obj.get("start", [])),
            goal=_pose_from_deg(obj.get("goal", [])),
            social_attributes=obj.get("social_attributes"),
            social_yielding=cls._opt_bool(obj.get("social_yielding")),
            desired_linear_vel=cls._opt_float(obj.get("desired_linear_vel")),
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
        return cls(start=start, goal=goal, social_attributes=robot.get("social_attributes"),
                   social_yielding=cls._opt_bool(robot.get("social_yielding")),
                   desired_linear_vel=cls._opt_float(robot.get("desired_linear_vel")))


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
            robots=self._parse_robots(scenario),
            social_yielding=(None if scenario.get("social_yielding") is None
                             else bool(scenario.get("social_yielding"))),
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
