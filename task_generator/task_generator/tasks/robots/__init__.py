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

    def mark_episode_started(self) -> None:
        """Start timeout accounting at the released episode boundary.

        In Isaac-backed evals the task reset can spend substantial wall-clock
        time waiting for world geometry, robot/camera readiness, HuNav agents,
        and InternNav readiness before ``task_reset`` is published and the
        navigation goal is released.  Counting that readiness window against the
        robot timeout makes the first episode finish almost immediately after it
        starts on slow simulator/model runs.  Refresh the timeout origin at the
        public episode-start edge instead.
        """
        self._last_reset = self._PROPS.clock.clock.sec
        self._last_reset_wall = time.monotonic()


    async def wait_navigation_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        for robot_manager in self._PROPS.robots.values():
            remaining = max(deadline - time.monotonic(), 0.0)
            await robot_manager.wait_for_pending_goal(remaining)

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
        wall_timeout_sec = self.node.rosparam[float].get('timeout_wall_sec', 0.0)
        if wall_timeout_sec <= 0.0:
            wall_timeout_factor = self.node.rosparam[float].get('timeout_wall_factor', 5.0)
            wall_timeout_sec = max(timeout_sec * max(wall_timeout_factor, 1.0), timeout_sec + 120.0)

        if sim_elapsed > timeout_sec or wall_elapsed > wall_timeout_sec:
            return True

        if not self._PROPS.robots:
            return False
        if not all(await asyncio.gather(*(robot_manager.is_done for robot_manager in self._PROPS.robots.values()))):
            return False
        return True
