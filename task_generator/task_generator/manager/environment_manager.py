import asyncio
import itertools
import os
import typing
from collections.abc import Callable, Collection, Iterator, Sequence
from typing import Any, Union

import attrs
from arena_simulation_setup.shared import Elevator
from arena_simulation_setup.tree.World import WorldDescription, USDWorldDescription

from task_generator import NodeInterface
from task_generator.shared import (
    Door,
    DynamicObstacle,
    Entity,
    Floor,
    FrameNamespace,
    Obstacle,
    Orientation,
    Pose,
    Position,
    Robot,
    Wall,
)
from task_generator.simulators.human import BaseHumanSimulator
from task_generator.simulators.human.utils import ObstacleLayer
from task_generator.simulators.sim import BaseSim
from arena_simulation_setup.utils.geometry import Position

EntityPropsT = typing.TypeVar('EntityPropsT', bound=Entity)


class _Realizer:
    @attrs.frozen()
    class _Configuration:
        x: float = 0.0
        y: float = 0.0
        prefix: str = ''

    _config: _Configuration

    @typing.overload
    def realize(self, target: str) -> str: ...

    def _prefix(self, *s: str) -> str:
        return str(FrameNamespace(self._config.prefix)(*s))

    @typing.overload
    def realize(self, target: Position) -> Position: ...

    def _realize_position(self, position: Position) -> Position:
        return Position(
            x=position.x + self._config.x,
            y=position.y + self._config.y,
            z=position.z,
        )

    def _realize_position_inv(self, position: Position) -> Position:
        return Position(
            x=position.x - self._config.x,
            y=position.y - self._config.y,
            z=position.z,
        )

    def _realize_orientation(self, orientation: Orientation) -> Orientation:
        return Orientation(*orientation)

    def _realize_pose(self, pose: Pose) -> Pose:
        return Pose(
            self._realize_position(pose.position),
            self._realize_orientation(pose.orientation)
        )

    @typing.overload
    def realize(self, target: EntityPropsT) -> EntityPropsT: ...

    @typing.overload
    def realize(self, target: Pose) -> Pose: ...

    def _realize_entity(self, entity: EntityPropsT) -> EntityPropsT:
        entity = attrs.evolve(
            entity,
            pose=self._realize_pose(entity.pose),
        )
        return entity

    @typing.overload
    def realize(self, target: Wall) -> Wall: ...

    def _realize_wall(self, wall: Wall) -> Wall:
        return attrs.evolve(
            wall,
            start=self._realize_position(wall.start),
            end=self._realize_position(wall.end),
        )

    @typing.overload
    def realize(self, target: Floor) -> Floor: ...

    def _realize_floor(self, floor: Floor) -> Floor:
        return attrs.evolve(
            floor,
            name=self._prefix(floor.name),
            pos=self._realize_position(floor.pos),
        )

    @typing.overload
    def realize(self, target: Door) -> Door: ...

    def _realize_door(self, door: Door) -> Door:
        return attrs.evolve(
            door,
            name=self._prefix(door.name),
            start=self._realize_position(door.start),
            end=self._realize_position(door.end),
        )

    @typing.overload
    def realize(self, target: Elevator) -> Elevator: ...

    def _realize_elevator(self, elevator: Elevator) -> Elevator:
        pos = list(elevator.position)
        if len(pos) >= 2:
            pos[0] += self._config.x
            pos[1] += self._config.y
        # Create Position object from modified list
        new_position = Position(x=pos[0], y=pos[1], z=pos[2] if len(pos) > 2 else 0.0)
        name = self._prefix(elevator.name)
        destination = self._prefix(elevator.destination) if getattr(elevator, 'destination', None) else elevator.destination
        return attrs.evolve(
            elevator,
            name=name,
            position=new_position,
            destination=destination,
        )

    def realize(
        self,
        target
    ):
        if isinstance(target, str):
            return self._prefix(target)

        if isinstance(target, Position):
            return self._realize_position(target)

        if isinstance(target, Pose):
            return self._realize_pose(target)

        if isinstance(target, Wall):
            return self._realize_wall(target)

        res = None

        if isinstance(target, Entity):
            res = self._realize_entity(target)

        elif isinstance(target, Door):
            res = self._realize_door(target)

        elif isinstance(target, Floor):
            res = self._realize_floor(target)

        elif isinstance(target, Elevator):
            res = self._realize_elevator(target)

        if res is None:
            raise TypeError(f'realization not implemented for type {type(target)}')

        res.sim_path = self._prefix(res.name)
        return res


class EnvironmentManager(NodeInterface, _Realizer):

    _namespace: str
    _human_simulator: BaseHumanSimulator
    _simulator: BaseSim

    id_generator: Iterator[int]

    def __init__(
        self,
        *args,
        namespace,
        simulator: BaseSim,
        entity_manager: BaseHumanSimulator,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._namespace = namespace
        self._simulator = simulator
        self._human_simulator = entity_manager

        ref_x, ref_y = self.node.rosparam[tuple[float, float]].get('reference', (0.0, 0.0))
        prefix = self.node.rosparam[str].get('prefix', '')
        self._config = self._Configuration(
            x=ref_x,
            y=ref_y,
            prefix=prefix,
        )

        self.id_generator = itertools.count(434)
        self._usd_scene_load_lock = asyncio.Lock()
        self._loaded_usd_scene_paths: set[str] = set()
        self._inflight_usd_scene_paths: set[str] = set()

    async def spawn_world_obstacles(self, world: Union[WorldDescription, USDWorldDescription]):
        """
        Loads given obstacles into the simulator,
        the map file is retrieved from launch parameter "world"

        For USD worlds (e.g., GRScenes), the scene is loaded directly via Isaac Sim
        and we skip programmatic wall/floor/door spawning since they're built into the USD.
        """
        
        # Check if this is a USD world - load the complete USD scene
        # Fallback: if world is not USDWorldDescription yet, try resolving the
        # world parameter to see if the map callback hasn't updated it yet.
        if not isinstance(world, USDWorldDescription):
            if hasattr(self, '_node') and hasattr(self._node, 'get_parameter'):
                try:
                    from arena_simulation_setup.tree.World import World as WorldModule
                    world_name_param = self._node.get_parameter('world')
                    world_name = world_name_param.value if world_name_param.type_ != '' else ''
                    if world_name:
                        resolved_world = await WorldModule.WorldIdentifier(world_name).resolve()
                        loaded_desc = resolved_world.load()
                        if isinstance(loaded_desc, USDWorldDescription):
                            self._logger.info(
                                f"Fallback: world param '{world_name}' resolved to USD; "
                                f"upgrading from {type(world).__name__} to {type(loaded_desc).__name__}"
                            )
                            world = loaded_desc
                except Exception as exc:
                    self._logger.debug(f'USD fallback resolution failed: {exc}')

        if isinstance(world, USDWorldDescription):
            self._logger.info(f"USD world detected - loading USD scene into Isaac Sim (type={type(world).__name__})")

            usd_path = world.get_usd_path()
            self._logger.info(f"USD world get_usd_path() returned: {usd_path}")
            if usd_path:
                raw_usd_path = str(usd_path)
                resolved_usd_path = raw_usd_path
                self._logger.info(f"USD raw_usd_path={raw_usd_path}, isabs={os.path.isabs(resolved_usd_path)}, exists={os.path.exists(resolved_usd_path)}")
                if not os.path.isabs(resolved_usd_path):
                    resolved_usd_path = os.path.join(str(getattr(world, 'path', '') or ''), resolved_usd_path)

                if not os.path.exists(resolved_usd_path):
                    fallback_dir = str(getattr(world, 'path', '') or '')
                    fallback_name = os.path.basename(raw_usd_path)
                    fallback_path = os.path.join(fallback_dir, fallback_name) if fallback_dir and fallback_name else ''
                    if fallback_path and os.path.exists(fallback_path):
                        self._logger.warning(
                            'USD world scene path from world.yaml does not exist; '
                            f'using packaged fallback asset instead: {raw_usd_path} -> {fallback_path}'
                        )
                        resolved_usd_path = fallback_path
                    else:
                        self._logger.error(
                            'USD world scene path does not exist and no packaged fallback was found: '
                            f'{raw_usd_path}'
                        )

                async with self._usd_scene_load_lock:
                    if resolved_usd_path in self._loaded_usd_scene_paths:
                        self._logger.info(f"USD scene already loaded; skipping duplicate request: {resolved_usd_path}")
                    elif resolved_usd_path in self._inflight_usd_scene_paths:
                        self._logger.info(f"USD scene load already in flight; skipping duplicate request: {resolved_usd_path}")
                    elif hasattr(self._simulator, 'load_usd_scene'):
                        self._inflight_usd_scene_paths.add(resolved_usd_path)
                        usd_scene = world.usd_scene
                        scale = usd_scene.scale if hasattr(usd_scene, 'scale') else 1.0
                        position = usd_scene.position if hasattr(usd_scene, 'position') else None
                        orientation = usd_scene.orientation if hasattr(usd_scene, 'orientation') else None

                        try:
                            success = await self._simulator.load_usd_scene(
                                usd_path=resolved_usd_path,
                                scene_prim_path="/World/Scene",
                                scale=scale,
                                position=position,
                                orientation=orientation,
                                add_colliders=False,  # GRScenes navigation USD already has collisions
                                disable_collision_cooking=True,
                            )
                        finally:
                            self._inflight_usd_scene_paths.discard(resolved_usd_path)

                        if success:
                            self._loaded_usd_scene_paths.add(resolved_usd_path)
                            self._logger.info(f"USD scene loaded successfully: {resolved_usd_path}")
                        else:
                            self._logger.error(f"Failed to load USD scene: {resolved_usd_path}")
                    else:
                        self._logger.warning("Simulator does not support USD scene loading")

            # For USD worlds, still initialize HuNav but without walls/doors
            await self._human_simulator.spawn_world(
                walls=tuple(),
                doors=tuple(),
            )
            return
        
        # Arena based Yaml world spawning
        self._logger.info(f"Non-USD world spawning (type={type(world).__name__}), walls={len(tuple(world.all_walls))}, doors={len(tuple(world.all_doors))}, floors={len(tuple(world.all_floors))}")

        async def guarded_world_spawn(label: str, coro: typing.Awaitable, timeout_s: float = 45.0):
            self._logger.info(f"[world-geometry] starting {label}")
            try:
                result = await asyncio.wait_for(coro, timeout=max(float(timeout_s), 0.0))
                self._logger.info(f"[world-geometry] finished {label}")
                return result
            except asyncio.TimeoutError:
                self._logger.warning(
                    f"[world-geometry] timed out while spawning {label} after {timeout_s:.1f}s; "
                    "continuing so robot/navigation bringup is not blocked"
                )
                return False
            except Exception as exc:
                self._logger.warning(
                    f"[world-geometry] failed while spawning {label}: {exc!r}; "
                    "continuing so robot/navigation bringup is not blocked"
                )
                return False

        futures: list[typing.Awaitable] = []

        walls = tuple(world.all_walls)
        doors = tuple(world.all_doors)
        floors = tuple(world.all_floors)
        elevators = tuple(world.all_elevators)
        if floors:
            futures.append(guarded_world_spawn('floors', self._simulator.spawn_floors(tuple(map(self.realize, floors)))))

        if walls or doors:
            futures.append(
                guarded_world_spawn(
                    'walls/doors',
                    self._human_simulator.spawn_world(
                        tuple(map(self.realize, walls)),
                        tuple(map(self.realize, doors)),
                    ),
                    timeout_s=75.0,
                )
            )

        futures.append(
            guarded_world_spawn(
                'static obstacles',
                self._human_simulator.spawn_obstacles(
                    tuple(map(self.realize, world.all_static_entities)),
                    layer=ObstacleLayer.WORLD,
                ),
                timeout_s=30.0,
            )
        )
        if elevators:
            self._logger.debug(f"Realized elevators for world: {[e.name for e in elevators]}")
            futures.append(
                guarded_world_spawn(
                    'elevators',
                    self._simulator.spawn_elevators(
                        tuple(map(self.realize, elevators))
                    ),
                )
            )

        await asyncio.gather(*futures)

    async def spawn_dynamic_obstacles(self, setups: Collection[DynamicObstacle]):
        """
        Loads given dynamic obstacles into the simulator.
        """

        await self._human_simulator.spawn_dynamic_obstacles(
            tuple(map(self.realize, setups))
        )

    async def spawn_obstacles(self, setups: Collection[Obstacle]):
        """
        Loads given obstacles into the simulator.
        """

        await self._human_simulator.spawn_obstacles(tuple(map(self.realize, setups)))

    async def spawn_robot(self, robots: Sequence[Robot]) -> Sequence[Robot]:
        """
        Loads given robot into the simulator
        """
        await self._human_simulator.spawn_robot(robots=tuple(map(self.realize, robots)))
        return robots

    async def move_robot(self, robots: Sequence[Robot]) -> Sequence[bool]:
        """
        Moves given robot
        """
        return await self._human_simulator.move_robot(tuple(map(self.realize, robots)))

    async def remove_robot(self, robots: Sequence[Robot]) -> Sequence[bool]:
        """
        Deletes given robot
        """
        return await self._human_simulator.remove_robot(tuple(map(self.realize, robots)))

    async def respawn(self, callback: Callable[[], typing.Awaitable[Any]]):
        """
        Unuse obstacles, (re-)use them in callback, finally remove unused obstacles
        @callback: Function to call between unuse and remove
        """
        await self._human_simulator.unuse_obstacles()
        await callback()
        await self._human_simulator.remove_obstacles(purge=ObstacleLayer.UNUSED)

    async def reset(self, purge: ObstacleLayer = ObstacleLayer.INUSE):
        """
        Unuse and remove all obstacles
        """
        await self._human_simulator.remove_obstacles(purge=purge)
