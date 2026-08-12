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

        elapsed = now - self._last_reset
        timeout = self.node.conf.Robot.TIMEOUT.value
        if elapsed > timeout:
            # WHY log here: `done` returning True is the ONLY input to the auto-reset loop
            # (node._check_task_status), so this is the single place that knows the real
            # cause. Without it a reset just happens, and a timeout reset is
            # indistinguishable from a goal-reached reset in the log.
            self.node.get_logger().debug(
                f"[Reset-reason] TIMEOUT: {elapsed}s elapsed since last reset > "
                f"robot timeout {timeout}s. The robot did not reach its goal in time "
                f"(nav aborted, blocked, or the goal is unreachable)."
            )
            return True

        if not self._PROPS.robots:
            return False

        states = await asyncio.gather(*(
            robot_manager.is_done for robot_manager in self._PROPS.robots.values()))
        if not all(states):
            return False

        # Every robot reports goal-reached. Name them, because `is_done` is driven by
        # navigate_to_pose reporting STATUS_SUCCEEDED -- which a stale status array from
        # the PREVIOUS episode can also produce, and that misfires as an instant reset.
        reached = ', '.join(
            name for name, done in zip(self._PROPS.robots.keys(), states) if done)
        self.node.get_logger().debug(
            f"[Reset-reason] GOAL REACHED by [{reached}] after {elapsed}s "
            f"(nav2 reported STATUS_SUCCEEDED)."
        )
        return True
