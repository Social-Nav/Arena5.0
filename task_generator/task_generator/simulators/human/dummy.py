from collections.abc import Sequence
from task_generator.simulators.human import BaseHumanSimulator
from task_generator.shared import DynamicObstacle, Obstacle


class DummyHumanSimulator(BaseHumanSimulator):

    async def _spawn_obstacles_impl(
        self,
        obstacles,
    ) -> Sequence[Obstacle | None]:
        return obstacles

    async def _spawn_dynamic_obstacles_impl(
        self,
        obstacles,
    ) -> Sequence[DynamicObstacle | None]:
        """Leave dynamic obstacles unmaterialized in no-human dummy mode.

        The empty result keeps the inputs registered but unspawned and makes the
        base implementation skip the simulator's pedestrian spawn call.
        """
        return ()

    async def _remove_obstacles_impl(
        self,
    ) -> bool:
        return True

    async def _spawn_walls_impl(
        self,
        walls,
    ) -> bool:
        return True

    async def _spawn_doors_impl(
        self,
        doors,
    ) -> bool:
        return True

    async def _spawn_robot_impl(
        self,
        robots,
    ) -> Sequence[bool]:
        return (True,) * len(robots)

    async def _remove_robot_impl(
        self,
        robots,
    ) -> Sequence[bool]:
        return (True,) * len(robots)

    async def _move_robot_impl(
        self,
        robots,
    ) -> Sequence[bool]:
        return (True,) * len(robots)
