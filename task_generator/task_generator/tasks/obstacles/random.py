from __future__ import annotations

import functools
import itertools
import typing
from typing import Callable, Iterator

import attrs
import numpy as np
import rclpy
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree import Identifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.assets.Pedestrian import PedestrianIdentifier

try:
    from typing import Self
except ImportError:
    Self = typing.TypeVar('Self')

from task_generator.manager.world_manager.density_aware_sampler import (
    DensityAwarePositionSampler,
    DensityAwareSamplingConfig,
)
from task_generator.shared import DynamicObstacle, Obstacle, Orientation, Pose
from task_generator.tasks import identifier_to_available
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles


@attrs.define()
class _Config:
    N_STATIC_OBSTACLES: ROSParamT[tuple[int, int]]
    N_INTERACTIVE_OBSTACLES: ROSParamT[tuple[int, int]]
    N_DYNAMIC_OBSTACLES: ROSParamT[tuple[int, int]]

    MODELS_STATIC_OBSTACLES: ROSParamT[list[str]]
    MODELS_INTERACTIVE_OBSTACLES: ROSParamT[list[str]]
    MODELS_DYNAMIC_OBSTACLES: ROSParamT[list[str]]
    POOL_SIZE: ROSParamT[int] | None = None


class TM_Random(TM_Obstacles):
    """
    Random task generator for obstacles.

    This class generates random obstacles for a task scenario.

    Attributes:
        _config (Config): Configuration object for obstacle generation.

    Methods:
        prefix(*args): Prefixes the given arguments with "scenario".
        __init__(**kwargs): Initializes the TM_Random object.
        reconfigure(config): Reconfigures the obstacle generation based on the given configuration.
        reset(**kwargs): Resets the obstacle generation with the specified parameters.

    """

    _config: _Config

    async def reset(self, **kwargs) -> Obstacles:
        """
        Resets the obstacle generation with the specified parameters.

        Args:
            **kwargs: Additional keyword arguments for customizing the obstacle generation.
                N_STATIC_OBSTACLES (int): Number of static obstacles.
                N_INTERACTIVE_OBSTACLES (int): Number of interactive obstacles.
                N_DYNAMIC_OBSTACLES (int): Number of dynamic obstacles.
                MODELS_STATIC_OBSTACLES (dict[str, float]): dictionary of static obstacle models and their weights.
                MODELS_INTERACTIVE_OBSTACLES (dict[str, float]): dictionary of interactive obstacle models and their weights.
                MODELS_DYNAMIC_OBSTACLES (dict[str, float]): dictionary of dynamic obstacle models and their weights.

        Returns:
            tuple[list[Obstacle], list[DynamicObstacle]]: A tuple containing the generated obstacles and dynamic obstacles.

        """

        N_STATIC_OBSTACLES: int = kwargs.get(
            "N_STATIC_OBSTACLES",
            self.node.conf.General.RNG.value.integers(
                *self._config.N_STATIC_OBSTACLES.value,
                endpoint=True
            ),
        )
        N_INTERACTIVE_OBSTACLES: int = kwargs.get(
            "N_INTERACTIVE_OBSTACLES",
            self.node.conf.General.RNG.value.integers(
                *self._config.N_INTERACTIVE_OBSTACLES.value,
                endpoint=True
            ),
        )

        N_DYNAMIC_OBSTACLES: int = kwargs.get(
            "N_DYNAMIC_OBSTACLES",
            self.node.conf.General.RNG.value.integers(
                *self._config.N_DYNAMIC_OBSTACLES.value,
                endpoint=True
            ),
        )

        class ModelList(dict[str, float]):

            @classmethod
            def fromkeys(cls, *args, **kwargs) -> Self:
                result = cls(super().fromkeys(*args, **kwargs))
                if not len(result):
                    self._logger.warn('Empty model list passed. Defaulting to empty string.')
                    result[""] = 1.0
                return result

            @property
            def a(self) -> list[str]:
                return list(self.keys())

            @property
            def p(self) -> list[float]:
                _p = np.array(list(self.values()), dtype=float)
                _p /= _p.sum()
                return _p.tolist()

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        MODELS_STATIC_OBSTACLES = ModelList.fromkeys(
            kwargs.get(
                "MODELS_STATIC_OBSTACLES",
                self._config.MODELS_STATIC_OBSTACLES.value
            ),
            1,
        )
        MODELS_INTERACTIVE_OBSTACLES = ModelList.fromkeys(
            kwargs.get(
                "MODELS_INTERACTIVE_OBSTACLES",
                self._config.MODELS_INTERACTIVE_OBSTACLES.value
            ),
            1,
        )

        MODELS_DYNAMIC_OBSTACLES = ModelList.fromkeys(
            kwargs.get(
                "MODELS_DYNAMIC_OBSTACLES",
                self._config.MODELS_DYNAMIC_OBSTACLES.value
            ),
            1,
        )

        # rospy.logwarn(f"{MODELS_DYNAMIC_OBSTACLES}")

        def indexer() -> Callable[..., int]:
            indices: dict[str, Iterator[int]] = dict()

            def index(model: str):
                if model not in indices:
                    indices[model] = itertools.count(1)
                return next(indices[model])

            return index

        waypoints_per_ped = 2
        static_and_interactive = self._PROPS.world_manager.get_positions_on_map(
            n=N_STATIC_OBSTACLES + N_INTERACTIVE_OBSTACLES,
            safe_dist=1,
        ) if N_STATIC_OBSTACLES + N_INTERACTIVE_OBSTACLES else []
        configured_pool_size = (
            int(self._config.POOL_SIZE.value)
            if self._config.POOL_SIZE is not None
            else -1
        )
        if configured_pool_size >= 0 and N_STATIC_OBSTACLES + N_INTERACTIVE_OBSTACLES:
            raise ValueError(
                'task.random.dynamic.pool_size requires static and interactive counts to be zero'
            )
        pool_size = max(N_DYNAMIC_OBSTACLES, configured_pool_size)
        pedestrian_sample = DensityAwarePositionSampler(
            world_map=self._PROPS.world_manager.map,
            rng=self.node.conf.General.RNG.value,
            candidate_provider=self._PROPS.world_manager._occupancy_to_available,
            config=DensityAwareSamplingConfig(goals_per_agent=waypoints_per_ped),
        ).sample(pool_size)
        selected_pedestrian_routes = pedestrian_sample.routes[:N_DYNAMIC_OBSTACLES]
        self._logger.warn(
            f'[PedestrianPopulation] requested={N_DYNAMIC_OBSTACLES} '
            f'pool_size={pool_size} selected={len(selected_pedestrian_routes)}'
        )
        if configured_pool_size < 0:
            # Preserve the original RNG consumption outside a nested-prefix
            # experiment.
            positions = map(
                lambda pos: Pose(
                    pos,
                    orientation=Orientation.from_yaw(
                        2 * np.pi * self.node.conf.General.RNG.value.random()
                    ),
                ),
                [*static_and_interactive, *(route.start for route in selected_pedestrian_routes)],
            )
            pedestrian_models = self.node.conf.General.RNG.value.choice(
                a=MODELS_DYNAMIC_OBSTACLES.a,
                p=MODELS_DYNAMIC_OBSTACLES.p,
                size=N_DYNAMIC_OBSTACLES,
            )
        else:
            # Draw attributes for the complete cap-sized pool before slicing.
            # N=5 is therefore byte-for-byte the first five records of N=10.
            pedestrian_orientations = (
                2 * np.pi * self.node.conf.General.RNG.value.random(pool_size)
            )
            pedestrian_models = self.node.conf.General.RNG.value.choice(
                a=MODELS_DYNAMIC_OBSTACLES.a,
                p=MODELS_DYNAMIC_OBSTACLES.p,
                size=pool_size,
            )
            positions = iter(
                Pose(route.start, Orientation.from_yaw(yaw))
                for route, yaw in zip(selected_pedestrian_routes, pedestrian_orientations)
            )
        pedestrian_routes = iter(selected_pedestrian_routes)

        obstacles: list[Obstacle] = []

        # Create static obstacles
        if N_STATIC_OBSTACLES:
            index = indexer()
            obstacles += [
                Obstacle(
                    name=f"S_{model}_{index(model)}",
                    model=model,
                    pose=next(positions),
                )
                for model in self.node.conf.General.RNG.value.choice(
                    a=MODELS_STATIC_OBSTACLES.a,
                    p=MODELS_STATIC_OBSTACLES.p,
                    size=N_STATIC_OBSTACLES,
                )
            ]

        # Create interactive obstacles
        if N_INTERACTIVE_OBSTACLES:
            index = indexer()

            obstacles += [
                Obstacle(
                    name=f"I_{model}_{index(model)}",
                    model=model,
                    pose=next(positions),
                )
                for model in self.node.conf.General.RNG.value.choice(
                    a=MODELS_INTERACTIVE_OBSTACLES.a,
                    p=MODELS_INTERACTIVE_OBSTACLES.p,
                    size=N_INTERACTIVE_OBSTACLES,
                )
            ]

        # Create dynamic obstacles

        dynamic_obstacles: list[DynamicObstacle] = []

        if N_DYNAMIC_OBSTACLES:
            index = indexer()

            dynamic_obstacles += [
                DynamicObstacle(
                    name=f"Pedestrian_{i}",
                    model=model,
                    waypoints=list(next(pedestrian_routes).goals),
                    pose=next(positions),
                    extra={"behavior_tree": "BTRegularNav.xml"},
                )
                for i, model in enumerate(pedestrian_models[:N_DYNAMIC_OBSTACLES])
            ]

        return obstacles, dynamic_obstacles

    def __init__(self, **kwargs):
        TM_Obstacles.__init__(self, **kwargs)

        def param_to_tuple(v: typing.Any) -> tuple[int, int]:
            lo = int(v[0])
            hi = int(v[1] if len(v) >= 2 else v[0])
            lo, hi = min(lo, hi), max(lo, hi)
            return lo, hi

        def param_to_modellist(identifier: typing.Type[Identifier], v: typing.Any) -> list[str]:
            if len(v):
                return v
            return list(identifier_to_available(identifier))

        STATIC = 'static'
        INTERACTIVE = 'interactive'
        DYNAMIC = 'dynamic'

        self._config = _Config(
            N_STATIC_OBSTACLES=self.node.ROSParam[tuple[int, int]](
                self.namespace(STATIC, 'n'),
                [5, 15],
                parse=param_to_tuple
            ),
            N_INTERACTIVE_OBSTACLES=self.node.ROSParam[tuple[int, int]](
                self.namespace(INTERACTIVE, 'n'),
                [0, 0],
                parse=param_to_tuple
            ),
            N_DYNAMIC_OBSTACLES=self.node.ROSParam[tuple[int, int]](
                self.namespace(DYNAMIC, 'n'),
                [1, 5],
                parse=param_to_tuple
            ),

            MODELS_STATIC_OBSTACLES=self.node.ROSParam[list[str]](
                self.namespace(STATIC, 'models'),
                [],
                type_=rclpy.Parameter.Type.STRING_ARRAY,
                parse=functools.partial(param_to_modellist, ObjectIdentifier)
            ),
            MODELS_INTERACTIVE_OBSTACLES=self.node.ROSParam[list[str]](
                self.namespace(INTERACTIVE, 'models'),
                [],
                type_=rclpy.Parameter.Type.STRING_ARRAY,
                parse=functools.partial(param_to_modellist, ObjectIdentifier)
            ),
            MODELS_DYNAMIC_OBSTACLES=self.node.ROSParam[list[str]](
                self.namespace(DYNAMIC, 'models'),
                [],
                type_=rclpy.Parameter.Type.STRING_ARRAY,
                parse=functools.partial(param_to_modellist, PedestrianIdentifier)
            ),
            POOL_SIZE=self.node.ROSParam[int](
                self.namespace(DYNAMIC, 'pool_size'),
                -1,
            ),
        )
