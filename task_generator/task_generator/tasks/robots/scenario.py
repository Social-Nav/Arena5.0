import math
import os

from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import RobotGoal

from task_generator.shared import Orientation, Pose, PositionRadius
from task_generator.tasks.robots import TM_Robots


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

    def _parse_scenario(self, scenario: str) -> list[RobotGoal]:
        return WorldIdentifier(self._world_name()).resolve_sync().scenario(str(scenario)).resolve_sync().load().robots

    async def reset(self, **kwargs):
        await super().reset(**kwargs)

        # Re-resolve against the current world_name at reset time to avoid
        # stale cache from startup (file param fires before world param is set).
        SCENARIO_ROBOTS = self._parse_scenario(self._scenario_param())

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

        for robot, config in zip(managed_robots, SCENARIO_ROBOTS):
            await robot.reset(start_pos=config.start, goal_pos=config.goal)
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
