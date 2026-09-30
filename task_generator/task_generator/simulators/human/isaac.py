import asyncio
import typing
from collections.abc import Sequence

from task_generator.shared import DynamicObstacle
from task_generator.simulators.human.dummy import DummyHumanSimulator
from task_generator.simulators.human.utils import ObstacleLayer
from task_generator.simulators.sim import BaseSim
from task_generator.simulators.sim.isaac_simulator import IsaacSimulator


class IsaacHumanSimulator(DummyHumanSimulator):

    def __init__(self, *args, simulator: BaseSim, **kwargs):
        if not isinstance(simulator, IsaacSimulator):
            raise ValueError("IsaacEntityManager only works with IsaacSimulator")
        super().__init__(*args, simulator=simulator, **kwargs)
        self._simulator = typing.cast(IsaacSimulator, self._simulator)

        self._walls: list[str] = []

    async def spawn_dynamic_obstacles(
        self,
        obstacles: Sequence[DynamicObstacle],
    ):
        self._logger.debug(f'spawning {len(obstacles)} dynamic obstacles')
        futures: list[typing.Awaitable] = []
        for obstacle in obstacles:
            self._logger.info(f"Attempting to spawn model: {obstacle.name}")
            self._logger.info(f"waypoints:{obstacle.waypoints}")
            known = self._known_obstacles.get(obstacle.name)
            if known is None:
                known = self._known_obstacles.create_or_get(
                    name=obstacle.name,
                    obstacle=obstacle
                )
                futures.append(self._simulator.pedestrian_spawn((obstacle,)))
            else:
                futures.append(self._simulator.pedestrian_move((obstacle,)))
            known.layer = ObstacleLayer.INUSE

        await asyncio.gather(*futures)
