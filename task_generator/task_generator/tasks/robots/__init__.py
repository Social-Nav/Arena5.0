import asyncio
import time

from task_generator.shared import Pose
from task_generator.tasks import TaskMode


class TM_Robots(TaskMode):
    """
    Task mode for controlling one or multiple robots.

    Args:
        **kwargs: Additional keyword arguments.

    Attributes:
        _PROPS (TaskProperties): Task properties object.

    """

    _last_reset: int

    async def reset(self, **kwargs):
        self._last_reset = self._PROPS.clock.clock.sec
        self._last_reset_wall = time.monotonic()

    async def set_position(self, pose: Pose):
        """
        Set the position of all robots.

        Args:
            position (Pose): The desired position and orientation.

        """
        for robot_manager in self._PROPS.robots.values():
            await robot_manager.reset(pose, None)

    async def set_goal(self, pose: Pose):
        """
        Set the goal position for all robots.

        Args:
            position (Pose): The desired goal position and orientation.

        """
        for robot_manager in self._PROPS.robots.values():
            await robot_manager.reset(None, pose)

    @property
    async def done(self) -> bool:
        """
        Check if all robots have completed their tasks.

        Returns:
            bool: True if all robots are done, False otherwise.

        """
        timeout_sec = self.node.conf.Robot.TIMEOUT.value
        sim_elapsed = self._PROPS.clock.clock.sec - self._last_reset
        wall_elapsed = time.monotonic() - getattr(self, '_last_reset_wall', time.monotonic())

        if sim_elapsed > timeout_sec or wall_elapsed > timeout_sec:
            return True

        if not self._PROPS.robots:
            return False
        if not all(await asyncio.gather(*(robot_manager.is_done for robot_manager in self._PROPS.robots.values()))):
            return False
        return True
