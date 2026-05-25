import asyncio
import itertools
import os
import random
import time
import traceback
import typing
from collections.abc import Sequence

import arena_people_msgs.msg
import arena_robots.Robot
import geometry_msgs.msg
import isaacsim_msgs.msg
import launch
import launch_ros.actions
import numpy as np
import std_msgs.msg
import std_srvs.srv
from arena_rclpy_mixins.Async import ClientWrapper
from arena_simulation_setup.shared import Obstacle as ObstacleDefinition
from arena_simulation_setup.tree.Wall import WallSegment
from isaacsim_msgs.msg import (
    Door,
    Elevator,
    Floor,
    Material,
    Pedestrian,
    PedestrianGoal,
    Prim,
    Scale,
    Wall,
)
from isaacsim_msgs.srv import (
    DeletePrims,
    EditPrims,
    LoadUsdScene,
    NavigatePedestrians,
    SpawnDoors,
    SpawnElevators,
    SpawnFloors,
    SpawnPedestrians,
    SpawnPrims,
    SpawnUrdf,
    SpawnUsdRobot,
    SpawnWalls,
)
from std_msgs.msg import String as StdString

from task_generator.shared import Door as DoorDefinition
from task_generator.shared import (
    DynamicObstacle,
    ModelType,
    Namespace,
    Obstacle,
    Pose,
    Robot,
)
from task_generator.shared import Elevator as ElevatorDefinition
from task_generator.shared import Floor as FloorDefinition
from task_generator.shared import Wall as WallDefinition

from .isaac_simulator import IsaacSimulator, material_to_msg


async def resolve_material_to_msg(material_ref, logger, label: str) -> isaacsim_msgs.msg.Material:
    try:
        return material_to_msg(await material_ref.resolve())
    except Exception as exc:
        logger.warning(f"Falling back to default Isaac material for {label}: {exc}")
        return Material()


class IsaacEvalSimulator(IsaacSimulator):
    """Evaluation-oriented Isaac simulator overlay.

    Keeps `IsaacSimulator` as the origin/jazzy baseline and only layers the
    feature-branch evaluation deltas that are still needed for GRScenes +
    InternNav smoke/eval runs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_robot_state_publishers: set[str] = set()
        self._spawned_usd_robots: set[str] = set()
        self._legacy_spawn_usd_robot = self.node.create_client_wrapper(
            SpawnUsdRobot,
            "/isaac/SpawnUsdRobot_srv",
        )
        self._spawn_usd_robot_client = self._clients.SpawnUsdRobot

    async def _ensure_static_robot_state(
        self,
        robot_name: str,
        base_frame: str,
        odom_frame: str,
        publish_fallback_odom_tf: bool = False,
    ) -> None:
        del publish_fallback_odom_tf
        if robot_name in self._static_robot_state_publishers:
            return

        self._logger.info(
            'IsaacEvalSimulator strict readiness mode enabled for '
            f'{robot_name}: not spawning fallback odom/TF/camera publishers; '
            f'eval will wait for real topics using base_frame={base_frame or "base_link"} '
            f'and odom_frame={odom_frame or "odom"}, then fail fast if they never appear.'
        )
        self._static_robot_state_publishers.add(robot_name)

    async def robot_spawn(self, robots):
        async def impl(robot: Robot) -> bool:
            try:
                resolved_robot_model = await robot.model.resolve()
                try:
                    model = await resolved_robot_model.model.get(ModelType.USD, loader_args=robot.asdict())
                except FileNotFoundError:
                    self._logger.debug(
                        f"USD model for {robot.model.name} not found; falling back to URDF"
                    )
                    model = await resolved_robot_model.model.get(ModelType.URDF, loader_args=robot.asdict())

                robot_params = (await arena_robots.Robot.RobotIdentifier(robot.model.name).resolve()).model_params
                fq_name = self._NS_ROBOT(robot.sim_path)

                if model.type == ModelType.USD:
                    assert model.path is not None, f"USD model {model.name} must have a valid file path"
                    await self._ensure_static_robot_state(
                        robot.name,
                        robot_params.base_frame,
                        robot_params.odom_frame,
                        publish_fallback_odom_tf=False,
                    )
                    await self._spawn_usd_robot_client.call_fire_and_forget(
                        SpawnUsdRobot.Request(
                            name=fq_name,
                            usd_path=str(model.path),
                            robot_namespace=str(self.node.service_namespace(robot.name)).lstrip('/'),
                            base_frame=robot_params.base_frame or '',
                            pose=robot.pose.to_msg(),
                        )
                    )
                    self._logger.info(
                        f"Requested USD robot spawn for '{robot.name}' via {self._spawn_usd_robot_client.client.srv_name}; "
                        "strict eval mode will wait for real TF/odom/camera readiness before releasing the episode"
                    )
                    await asyncio.sleep(1.0)
                    self._spawned_usd_robots.add(robot.name)
                    await self._ensure_map_to_world_tf(robot.name)
                    await self._publish_sensor_frame_tfs(
                        robot.name,
                        robot_params.sensor_frame_transforms,
                    )
                    return True

                if model.type == ModelType.URDF:
                    assert model.path is not None, f"URDF model {model.name} must have a valid file path"
                    await self._ensure_static_robot_state(
                        robot.name,
                        robot_params.base_frame,
                        robot_params.odom_frame,
                        publish_fallback_odom_tf=False,
                    )
                    await self._clients.SpawnUrdf.call_timeout(
                        SpawnUrdf.Request(
                            name=fq_name,
                            urdf_path=str(model.path),
                            robot_model=robot.model.name,
                            localization=True,
                            tf_prefix=robot.name,
                            base_frame=robot_params.base_frame,
                            odom_frame=robot_params.odom_frame,
                            pose=robot.pose.to_msg(),
                            cmd_vel_topic=self.node.service_namespace(robot.name, 'cmd_vel'),
                            joint_states_topic=self.node.service_namespace(robot.name, 'joint_states'),
                            odom_topic=self.node.service_namespace(robot.name, 'odom'),
                        ),
                        timeout_sec=5.0,
                    )

                    base_frame = robot_params.base_frame
                    robot_prim_path = os.path.join('/World', fq_name, base_frame)
                    try:
                        if self._reg_pub:
                            self._reg_pub.publish(StdString(data=f'robot|{robot_prim_path}'))
                            self._logger.debug(f'Published registration for robot: {robot_prim_path}')
                        else:
                            self._logger.warning(
                                'Registration publisher not available; robot not registered with IsaacSim DoorManager'
                            )
                    except Exception as exc:
                        self._logger.warning(
                            f'Failed to publish robot registration: {exc}\n{traceback.format_exc()}'
                        )
                    return True

                raise NotImplementedError(
                    f"robot model of type {model.type} can't be spawned by {self.__class__.__name__}"
                )
            except Exception as exc:
                self._logger.error(f'{repr(exc)}\n{traceback.format_exc()}')
                return False

        return await asyncio.gather(*map(impl, robots))

    async def obstacle_spawn(self, obstacles):
        async def impl(obstacle: Obstacle) -> Prim | None:
            try:
                resolved_model = await asyncio.wait_for(obstacle.model.resolve(), timeout=5.0)
                model = await asyncio.wait_for(resolved_model.get([ModelType.USD]), timeout=5.0)
                if model.type is ModelType.UNKNOWN:
                    raise ValueError(f'obstacle model {obstacle.model.name} has no USD representation')
            except asyncio.TimeoutError:
                self._logger.warning(
                    f'Skipping obstacle model for {obstacle.name}: resolving {obstacle.model.name} timed out'
                )
                return None
            except Exception as exc:
                self._logger.warning(f'Skipping unresolved obstacle model for {obstacle.name}: {exc}')
                return None

            assert model.path is not None, f"USD model {model.name} must have a valid file path"
            prim = Prim()
            prim.usd_path = str(model.path)
            prim.name = self._NS_PRIM(obstacle.sim_path)
            prim.pose = obstacle.pose.to_msg()
            if obstacle.scale is not None:
                prim.scale.x = obstacle.scale.x
                prim.scale.y = obstacle.scale.y
                prim.scale.z = obstacle.scale.z
            return prim

        prims = await asyncio.gather(*map(impl, obstacles))
        if not any(prim is not None for prim in prims):
            self._logger.warning(
                f'No Isaac USD obstacle assets resolved out of {len(obstacles)} obstacle(s); '
                'continuing with procedural geometry only.'
            )
            return tuple(False for _ in obstacles)

        response = await self._clients.SpawnPrims.call_timeout(
            SpawnPrims.Request(prims=list(filter(None, prims))),
            timeout_sec=20.0,
        )
        if response is None:
            self._logger.warning(
                'SpawnPrims returned no ROS response; assuming Isaac processed the request.'
            )
            return tuple(prim is not None for prim in prims)

        response_iter = iter(response.ret)
        return tuple((prim is not None) and next(response_iter) for prim in prims)

    async def robot_move(self, robots):
        async def move_robot(robot: Robot) -> bool:
            try:
                if robot.name in self._spawned_usd_robots:
                    self._logger.info(
                        f"Moving spawned USD robot '{robot.name}' to reset pose via EditPrims"
                    )
                    return await self._move_entity(self._NS_ROBOT(robot.sim_path), robot.pose)

                return await self._move_entity(self._NS_ROBOT(robot.sim_path), robot.pose)
            except Exception as exc:
                self._logger.error(
                    f'Failed to move robot {robot.name}: {exc}\n{traceback.format_exc()}'
                )
                return False

        return await asyncio.gather(*map(move_robot, robots))

    async def spawn_walls(self, walls):
        self._logger.info(f'Attempting to spawn {len(walls)} wall definition(s) into Isaac Sim')

        async def create_segment(segment: WallSegment) -> Wall | None:
            end = segment.end.to_msg()
            end.z += segment.height
            try:
                wall_name = self.node._environment_manager.realize(f'wall_{next(self.wall_counter)}')
                material = await resolve_material_to_msg(segment.material, self._logger, f'wall {wall_name}')
                return Wall(
                    name=self._NS_WALL(wall_name),
                    start=segment.start.to_msg(),
                    end=end,
                    material=material,
                    thickness=segment.width,
                )
            except Exception as exc:
                self._logger.error(f'Failed to spawn wall: {exc}\n{traceback.format_exc()}')
                return None

        async def create_obstacle(obstacle: ObstacleDefinition) -> Prim | None:
            try:
                prim_name = self.node._environment_manager.realize(f'obstacle_{next(self.wall_counter)}')
                model = await (await obstacle.model.resolve()).get(ModelType.USD)
                if model.type is ModelType.UNKNOWN:
                    return None
                assert model.path is not None, f"USD model {model.name} must have a valid file path"
                prim = Prim()
                prim.usd_path = str(model.path)
                prim.name = self._NS_WALL(prim_name)
                prim.pose = obstacle.pose.to_msg()
                return prim
            except Exception as exc:
                self._logger.warning(f'Skipping unresolved wall obstacle asset: {exc}')
                return None

        async def create_wall(wall: WallDefinition):
            segments, obstacles = await wall.assets()
            return map(create_segment, segments), map(create_obstacle, obstacles)

        wall_futures = await asyncio.gather(*map(create_wall, walls))
        if not wall_futures:
            self._logger.info('No wall definitions provided; skipping wall and obstacle spawning.')
            return True

        segment_futures, obstacle_futures = zip(*wall_futures)
        walls_req = SpawnWalls.Request(
            walls=list(
                filter(
                    None,
                    await asyncio.gather(*itertools.chain.from_iterable(segment_futures)),
                )
            )
        )
        prims_req = SpawnPrims.Request(
            prims=list(
                filter(
                    None,
                    await asyncio.gather(*itertools.chain.from_iterable(obstacle_futures)),
                )
            )
        )

        if walls_req.walls:
            await self._clients.SpawnWalls.call_fire_and_forget(walls_req)
        if prims_req.prims:
            await self._clients.SpawnPrims.call_fire_and_forget(prims_req)
        self._logger.info('All walls spawned.')
        return True

    async def spawn_floors(self, floors) -> bool:
        self._logger.info(f'Attempting to spawn {len(floors)} floor definition(s) into Isaac Sim')

        async def impl(floor: FloorDefinition) -> Floor | None:
            try:
                material = await resolve_material_to_msg(floor.material, self._logger, f'floor {floor.name}')
                return Floor(
                    name=self._NS_FLOOR(floor.sim_path),
                    x_length=floor.x_length,
                    y_length=floor.y_length,
                    pos=floor.pos.to_msg(),
                    material=material,
                )
            except Exception:
                self._logger.error(f'Failed to spawn floor: {floor.name}\n{traceback.format_exc()}')
                return None

        floors_req = SpawnFloors.Request(floors=list(filter(None, await asyncio.gather(*map(impl, floors)))))
        if floors_req.floors:
            await self._clients.SpawnFloors.call_fire_and_forget(floors_req)
        self._logger.info('All floors spawned successfully.')
        return True

    async def spawn_doors(self, doors) -> bool:
        async def impl(door: DoorDefinition) -> Door | None:
            try:
                end = door.end.to_msg()
                end.z += door.height
                material = await resolve_material_to_msg(door.material, self._logger, f'door {door.name}')
                return Door(
                    name=self._NS_DOOR(door.name),
                    start=door.start.to_msg(),
                    end=end,
                    material=material,
                    thickness=0.1,
                    kind=door.kind,
                )
            except Exception as exc:
                self._logger.error(f'Failed to spawn door: {exc}\n{traceback.format_exc()}')
                return None

        doors_req = SpawnDoors.Request(doors=list(filter(None, await asyncio.gather(*map(impl, doors)))))
        if doors_req.doors:
            await self._clients.SpawnDoors.call_fire_and_forget(doors_req)
        self._logger.info('All doors spawned successfully.')
        return True

    async def spawn_elevators(self, elevators) -> bool:
        self._logger.debug(f'IsaacEvalSimulator.spawn_elevators called with: {[e.name for e in elevators]}')

        async def impl(elevator: ElevatorDefinition) -> Elevator | None:
            try:
                size = Scale(x=elevator.size[0], y=elevator.size[1], z=elevator.size[2])
                material = await resolve_material_to_msg(
                    elevator.material,
                    self._logger,
                    f'elevator {elevator.name}',
                )
                return Elevator(
                    name=elevator.sim_path,
                    position=elevator.position.to_msg(),
                    size=size,
                    height_min=elevator.height_min,
                    height_max=elevator.height_max,
                    material=material,
                    destination=elevator.destination if hasattr(elevator, 'destination') else '',
                )
            except Exception as exc:
                self._logger.error(
                    f'Failed to append elevator: {elevator.name}: {exc}\n{traceback.format_exc()}'
                )
                return None

        req = SpawnElevators.Request(elevators=list(filter(None, await asyncio.gather(*map(impl, elevators)))))
        if req.elevators:
            await self._clients.SpawnElevators.call_fire_and_forget(req)
        self._logger.debug('All elevators spawned successfully.')
        return True

    async def after_reset_task(self):
        await self._unpause()
        await asyncio.sleep(0.35)
        return True

    async def _pause(self):
        return True

    async def _unpause(self):
        return True

    async def pedestrian_spawn(self, pedestrians):
        async def impl(pedestrian: DynamicObstacle) -> Pedestrian | None:
            available_models: dict[str, str] = {
                'F_Business_02': 'F_Business_02',
                'F_Medical_01': 'F_Medical_01',
                'M_Medical_01': 'M_Medical_01',
                'biped_demo_meters': 'biped_demo',
                'female_adult_business_02': 'original_female_adult_business_02',
                'female_adult_medical_01': 'original_female_adult_medical_01',
                'female_adult_police_01': 'original_female_adult_police_01',
                'female_adult_police_01_new': 'female_adult_police_01_new',
                'female_adult_police_02': 'original_female_adult_police_02',
                'female_adult_police_03': 'original_female_adult_police_03',
                'female_adult_police_03_new': 'female_adult_police_03_new',
                'male_adult_construction_01': 'original_male_adult_construction_01',
                'male_adult_construction_01_new': 'male_adult_construction_01_new',
                'male_adult_construction_02': 'original_male_adult_construction_02',
                'male_adult_construction_03': 'original_male_adult_construction_03',
                'male_adult_construction_05': 'original_male_adult_construction_05',
                'male_adult_construction_05_new': 'male_adult_construction_05_new',
                'male_adult_medical_01': 'original_male_adult_medical_01',
                'male_adult_police_04': 'original_male_adult_police_04',
            }
            model_name = pedestrian.model.name if pedestrian.model.name in available_models else random.choice(tuple(available_models.keys()))
            ped = Pedestrian()
            ped.name = self._NS_PEDESTRIAN(pedestrian.sim_path)
            ped.character_name = available_models[model_name]
            ped.pose = pedestrian.pose.to_msg()
            ped.controller_stats = False
            return ped

        req = SpawnPedestrians.Request(pedestrians=list(filter(None, await asyncio.gather(*map(impl, pedestrians)))))
        if not req.pedestrians:
            return tuple(False for _ in pedestrians)

        await self._clients.SpawnPedestrians.call_fire_and_forget(req)
        await self.pedestrian_update(
            arena_people_msgs.msg.Pedestrians(
                pedestrians=[
                    arena_people_msgs.msg.Pedestrian(name=ped.sim_path, pose=ped.pose.to_msg())
                    for ped in pedestrians
                ]
            )
        )
        return tuple(True for _ in pedestrians)

    async def pedestrian_update(self, pedestrians):
        async def impl(ped: arena_people_msgs.msg.Pedestrian) -> PedestrianGoal | None:
            goal = PedestrianGoal()
            goal.name = self._NS_PEDESTRIAN(ped.name)
            goal.position = ped.pose.position
            goal.velocity = np.linalg.norm([ped.twist.linear.x, ped.twist.linear.y])
            return goal

        goals = list(filter(None, await asyncio.gather(*map(impl, pedestrians.pedestrians))))
        if not goals:
            return tuple()

        await self._clients.NavigatePedestrians.call_fire_and_forget(
            NavigatePedestrians.Request(goals=goals)
        )
        return tuple(True for _ in goals)

    async def _delete_entity(self, name: str) -> bool:
        self._logger.debug(f'Skipping DeletePrims for fresh Isaac eval stage: {name}')
        return True

    async def _delete_pedestrians(self, prim_path):
        self._logger.info(f'Skipping DeletePedestrians for fresh Isaac eval stage: {prim_path}')
        return True

    async def _move_entities(self, actions: Sequence[tuple[str, Pose]]) -> Sequence[bool]:
        await self._clients.EditPrims.call_fire_and_forget(
            EditPrims.Request(
                prims=[
                    Prim(name=name, pose=pose.to_msg() if hasattr(pose, 'to_msg') else pose)
                    for name, pose in actions
                ],
                pose=True,
            )
        )
        return [True] * len(actions)

    async def setup(self):
        self._logger.info('Setting up IsaacEvalSimulator service clients...')
        clients = [
            typing.cast(ClientWrapper, client)
            for client in self._clients.__dict__.values()
        ]

        # SpawnUsdRobot is the only Isaac service that exists under two names in
        # the Humble eval setup.  The Isaac-side service wrapper registers the
        # generated service as `/isaac/SpawnUsdRobot_srv`, while some newer
        # Arena code paths use `/isaac/SpawnUsdRobot`.  Do not require the
        # canonical name up front; otherwise the eval launcher can pass service
        # discovery with a stale graph, fire-and-forget the request to the wrong
        # endpoint, and then fail later because no real odom/TF/camera topics
        # were ever produced.
        canonical_spawn_service = self._clients.SpawnUsdRobot.client.srv_name
        legacy_spawn_service = self._legacy_spawn_usd_robot.client.srv_name
        required_services = {
            client.client.srv_name
            for client in clients
            if client.client.srv_name != canonical_spawn_service
        }
        for service_name in sorted(required_services):
            self._logger.info(f'Initializing service client: {service_name}')

        # In the mixed Humble task-generator + Isaac Sim ROS bridge graph,
        # rclpy Client.wait_for_service can remain blocked even after direct
        # graph discovery reports the service.  Poll graph discovery here and
        # let the concrete service calls below fail fast with their own bounded
        # call_timeout if a server disappears.  This keeps eval startup bounded
        # without introducing any dummy simulator state.
        deadline = time.monotonic() + 120.0
        available_services: set[str] = set()
        while time.monotonic() < deadline:
            available_services = {
                name for name, _types in self.node.get_service_names_and_types()
            }
            unavailable = sorted(required_services - available_services)
            spawn_available = (
                canonical_spawn_service in available_services
                or legacy_spawn_service in available_services
            )
            if not unavailable and spawn_available:
                break
            await asyncio.sleep(0.25)
        else:
            unavailable = sorted(required_services - available_services)

        missing_services = list(unavailable)
        if (
            canonical_spawn_service not in available_services
            and legacy_spawn_service not in available_services
        ):
            missing_services.append(
                f"one of {canonical_spawn_service} or {legacy_spawn_service}"
            )

        if missing_services:
            raise RuntimeError(f"Isaac service(s) unavailable after 120s: {', '.join(sorted(missing_services))}")

        spawn_service_override = os.environ.get('ARENA_ISAAC_USD_SPAWN_SERVICE', '').strip()
        if legacy_spawn_service in available_services:
            if legacy_spawn_service in available_services:
                self._spawn_usd_robot_client = self._legacy_spawn_usd_robot
                self._logger.info(
                    f'Using legacy SpawnUsdRobot endpoint {legacy_spawn_service} for eval compatibility.'
                )
        elif spawn_service_override.endswith('_srv'):
            raise RuntimeError(
                f'ARENA_ISAAC_USD_SPAWN_SERVICE requested {legacy_spawn_service}, but it was unavailable.'
            )
        else:
            self._spawn_usd_robot_client = self._clients.SpawnUsdRobot

        self.node.create_publisher(std_msgs.msg.String, '/isaac/add_pedestrians_topic', 10).publish(
            std_msgs.msg.String(data=self.node.service_namespace('arena_peds'))
        )

        self._logger.info('Skipping leftover robot prim cleanup for fresh Isaac eval process.')

    async def load_usd_scene(
        self,
        usd_path: str,
        scene_prim_path: str = '/World/Scene',
        scale: float = 1.0,
        position: list | None = None,
        orientation: list | None = None,
        add_colliders: bool = True,
        disable_collision_cooking: bool = True,
    ) -> bool:
        self._logger.info(f'Loading USD scene: {usd_path}')
        req = LoadUsdScene.Request()
        req.usd_path = usd_path
        req.scene_prim_path = scene_prim_path
        req.scale = float(scale)
        req.position = position or [0.0, 0.0, 0.0]
        req.orientation = orientation or [0.0, 0.0, 0.0, 1.0]
        req.add_colliders = add_colliders
        req.disable_collision_cooking = disable_collision_cooking

        try:
            response = await self._clients.LoadUsdScene.call_timeout(req, timeout_sec=600.0)
            if response is None:
                self._logger.error('LoadUsdScene service timed out')
                return False
            if response.success:
                self._logger.info(f'USD scene loaded: {response.scene_prim_path}')
                return True
            self._logger.error(f'Failed to load USD scene: {response.message}')
            return False
        except Exception as exc:
            self._logger.error(f'Exception loading USD scene: {exc}\n{traceback.format_exc()}')
            return False
