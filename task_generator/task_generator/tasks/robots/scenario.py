import math
import os

from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import RobotGoal, Scenario

from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from task_generator.shared import Orientation, Pose, PositionRadius
from task_generator.tasks.robots import TM_Robots

# Gate for the always-running social_yielding nodes. Absolute name (they run in the root
# namespace) + TRANSIENT_LOCAL so a late-starting node still gets the last value.
_SOCIAL_YIELDING_ENABLED_TOPIC = "/social_yielding/enabled"


class TM_Scenario(TM_Robots):
    """
    This class represents a scenario for robots in the task generator.
    It inherits from TM_Robots class and Node class.

    Attributes:
        _config (Config): The configuration object for the scenario.
    """

    _config: ROSParamT[list[RobotGoal]]

    def _world_name(self) -> str:
        configured = str(getattr(self.node.conf.Arena.WORLD, 'value', '') or '').strip()
        return configured or self.node._world_manager.world_name

    def _scenario_param(self) -> str:
        env_value = str(os.environ.get('ARENA_SCENARIO_FILE', '') or '').strip()
        if env_value:
            return env_value
        try:
            value = self.node.get_parameter(self.namespace('file')).value
        except Exception:
            value = self._config.param
        return str(value or '').strip()

    def _resolve_scenario(self, scenario: str) -> Scenario:
        return WorldIdentifier(self._world_name()).resolve_sync().scenario(str(scenario)).resolve_sync().load()

    def _parse_scenario(self, scenario: str) -> list[RobotGoal]:
        return self._resolve_scenario(scenario).robots

    async def reset(self, **kwargs):
        await super().reset(**kwargs)

        # Re-resolve against the current world_name at reset time to avoid
        # stale cache from startup (file param fires before world param is set).
        scenario = self._resolve_scenario(self._scenario_param())
        SCENARIO_ROBOTS = scenario.robots

        # check robot manager length
        managed_robots = list(self._PROPS.robots.values())

        scenario_robots_length = len(SCENARIO_ROBOTS)
        setup_robot_length = len(managed_robots)

        if scenario_robots_length == 0 and setup_robot_length > 0:
            self._logger.warn(
                "Scenario file contains no robot start/goal entries; generating random robot tasks on the map.",
                once=True,
            )
            biggest_robot = max((robot.safe_distance for robot in managed_robots), default=0)
            orientations = 2 * math.pi * self.node.conf.General.RNG.value.random(2 * setup_robot_length)
            positions = self._PROPS.world_manager.get_positions_on_map(
                n=2 * setup_robot_length,
                safe_dist=biggest_robot,
            )
            generated_positions = [
                Pose(position, Orientation.from_yaw(orientation))
                for orientation, position in zip(orientations, positions)
            ]
            SCENARIO_ROBOTS = [
                RobotGoal(start=start, goal=goal)
                for start, goal in zip(generated_positions[::2], generated_positions[1::2])
            ]
            scenario_robots_length = len(SCENARIO_ROBOTS)

        if setup_robot_length > scenario_robots_length:
            managed_robots = managed_robots[:scenario_robots_length]
            self._logger.warn(
                "Robot setup contains more robots than the scenario file.", once=True)

        if scenario_robots_length > setup_robot_length:
            SCENARIO_ROBOTS = SCENARIO_ROBOTS[:setup_robot_length]
            self._logger.warn(
                "Scenario file contains more robots than setup.", once=True)

        # Latched, so the always-running trigger/orchestrator pick it up whenever they reconnect.
        # PRECEDENCE (highest first): launch arg > robots[].social_yielding > False.
        # The launch arg is a tri-state STRING ("auto" = not specified), so an explicit command-line
        # value can override the file without "not passed" reading as "passed false".
        # Resolved AFTER the truncation above, so a scenario robot with no managed robot cannot set
        # a flag for a robot that never spawns.
        launch_arg = str(self.node.rosparam[str].get('social_yielding', 'auto')).strip().lower()

        robot_flags = [(i, r.social_yielding)
                       for i, r in enumerate(SCENARIO_ROBOTS)
                       if r.social_yielding is not None]

        if launch_arg in ('true', 'false'):
            enabled = launch_arg == 'true'
            source = f"launch override (social_yielding:={launch_arg})"
        elif robot_flags:
            # One latched topic gates the whole pipeline, so it cannot be per-robot. Take the
            # last (matching yaml duplicate-key resolution) but warn instead of silently picking.
            if len({v for _, v in robot_flags}) > 1:
                self._logger.warn(
                    f"robots disagree on social_yielding {robot_flags}; the pipeline is gated by "
                    f"ONE latched topic so it cannot be per-robot. Using the last: "
                    f"{robot_flags[-1][1]}.")
            enabled = bool(robot_flags[-1][1])
            source = f"robots[{robot_flags[-1][0]}]"
        else:
            enabled = False
            source = "default"
            if launch_arg != 'auto':
                # Unrecognised value: say so instead of silently treating it as off.
                self._logger.warn(
                    f"social_yielding launch arg {launch_arg!r} is not auto/true/false; "
                    "treating as 'auto' -> default False.")
        self._enabled_pub.publish(Bool(data=enabled))
        self._logger.info(f"social_yielding = {enabled} ({source})")

        for robot, config in zip(managed_robots, SCENARIO_ROBOTS):
            await robot.reset(start_pos=config.start, goal_pos=config.goal)
            # Absent field -> "neutral". Chosen as the default because passive's early, frequent
            # braking is too weak a baseline to produce real interaction negotiation, while
            # aggressive largely drives through. Pushed to controller_server + both costmaps;
            # both hot-reload, no relaunch needed.
            await robot.set_social_attributes(config.social_attributes or "neutral")
            # AFTER the profile: both write trajectorizer.desired_linear_vel and last write wins,
            # which is what makes the scenario's speed outrank it. Absent field -> no-op.
            await robot.set_desired_linear_vel(config.desired_linear_vel)
            self._PROPS.world_manager.forbid(
                [
                    PositionRadius(
                        x=config.start.position.x, y=config.start.position.y, radius=robot.safe_distance
                    ),
                    PositionRadius(
                        x=config.goal.position.x, y=config.goal.position.y, radius=robot.safe_distance
                    ),
                ]
            )

    def __init__(self, **kwargs):
        TM_Robots.__init__(self, **kwargs)

        self._config = self.node.ROSParam[list[RobotGoal]](
            self.namespace('file'),
            'default.json',
            parse=self._parse_scenario,
        )

        self._enabled_pub = self.node.create_publisher(
            Bool,
            _SOCIAL_YIELDING_ENABLED_TOPIC,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
