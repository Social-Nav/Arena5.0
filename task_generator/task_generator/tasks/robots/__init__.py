import asyncio

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
    _last_clock_seen: int | None = None  # for clock-jump/rewind detection (B2)
    # A single is_done poll is 0.5 s apart; any sim-clock step larger than this (s)
    # between polls is treated as a pause/unpause jump, not real elapsed episode time.
    _TIMEOUT_REBASELINE_STEP: int = 5

    async def reset(self, **kwargs):
        self._last_reset = self._PROPS.clock.clock.sec
        self._last_clock_seen = self._last_reset

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
        now = self._PROPS.clock.clock.sec

        # B2: the sim clock is frozen during the pause/unpause of a reset, and Isaac
        # can jump it forward (or backward) on unpause. A raw (now - _last_reset)
        # would then blow past TIMEOUT in a single step and fire a spurious reset —
        # the "reset immediately after a reset" loop. Detect a non-monotonic /
        # implausibly large step and re-baseline instead of counting it as elapsed.
        if self._last_clock_seen is not None:
            step = now - self._last_clock_seen
            if step < 0 or step > self._TIMEOUT_REBASELINE_STEP:
                self._last_reset = now  # clock jumped/rewound: restart the window
        self._last_clock_seen = now

        if (now - self._last_reset) > self.node.conf.Robot.TIMEOUT.value:
            return True

        if not self._PROPS.robots:
            return False
        if not all(await asyncio.gather(*(robot_manager.is_done for robot_manager in self._PROPS.robots.values()))):
            return False
        return True
