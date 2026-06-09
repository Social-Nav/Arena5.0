import asyncio
import itertools
import os
import sys
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
        # Legacy _srv client for LoadUsdScene (same FastDDS workaround as SpawnUsdRobot)
        self._legacy_load_usd_scene = self.node.create_client_wrapper(
            LoadUsdScene,
            "/isaac/LoadUsdScene_srv",
        )
        self._load_usd_scene_client = self._clients.LoadUsdScene

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
                    # Use the same fresh-DDS-participant workaround as LoadUsdScene.
                    # The long-lived task_generator rclpy participant can discover
                    # the Isaac service but still miss requests or responses across
                    # Docker/FastDDS boundaries; a short-lived subprocess behaves
                    # like `ros2 service call` and has proven reliable for these
                    # heavyweight Isaac services.
                    response = await self._spawn_usd_robot_subprocess(
                        name=fq_name,
                        usd_path=str(model.path),
                        robot_namespace=str(self.node.service_namespace(robot.name)).lstrip('/'),
                        base_frame=robot_params.base_frame or '',
                        pose=robot.pose,
                        timeout_sec=float(os.environ.get('ARENA_ISAAC_SPAWN_USD_ROBOT_TIMEOUT_SEC', '300.0')),
                    )
                    if response is None:
                        self._logger.error(
                            f"SpawnUsdRobot failed for '{robot.name}' via {self._spawn_usd_robot_client.client.srv_name}; "
                            'strict eval mode will not continue without a real robot spawn.'
                        )
                        return False
                    spawn_path = str(response.get('path', '') or '').strip()
                    if spawn_path:
                        self._logger.info(
                            f"Spawned USD robot '{robot.name}' via {self._spawn_usd_robot_client.client.srv_name} at "
                            f"{spawn_path}; strict eval mode will wait for real TF/odom/camera readiness before releasing the episode"
                        )
                    else:
                        self._logger.error(
                            f"SpawnUsdRobot returned an empty path for '{robot.name}' via "
                            f"{self._spawn_usd_robot_client.client.srv_name}; strict eval mode will not continue "
                            'without a confirmed live robot prim.'
                        )
                        return False
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
                    response = await self._clients.SpawnUrdf.call_timeout(
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
                    if response is None or not str(getattr(response, 'path', '') or '').strip():
                        self._logger.error(
                            f"SpawnUrdf failed for '{robot.name}'; strict eval mode will not continue without a real robot spawn."
                        )
                        return False

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

        req = SpawnPrims.Request(prims=list(filter(None, prims)))
        response = await self._clients.SpawnPrims.call_timeout(req)
        if response is None:
            return tuple(False for _ in obstacles)

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

        walls_res = await self._clients.SpawnWalls.call_timeout(walls_req) if walls_req.walls else None
        prims_res = await self._clients.SpawnPrims.call_timeout(prims_req) if prims_req.prims else None
        if walls_req.walls and (walls_res is None or not all(walls_res.ret)):
            return False
        if prims_req.prims and (prims_res is None or not all(prims_res.ret)):
            return False
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
            floors_res = await self._clients.SpawnFloors.call_timeout(floors_req)
            if floors_res is None or not all(floors_res.ret):
                return False
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
            doors_res = await self._clients.SpawnDoors.call_timeout(doors_req)
            if doors_res is None or not all(doors_res.ret):
                return False
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
            elevators_res = await self._clients.SpawnElevators.call_timeout(req)
            if elevators_res is None or not all(elevators_res.ret):
                return False
        self._logger.debug('All elevators spawned successfully.')
        return True

    async def after_reset_task(self):
        await self._unpause()
        await asyncio.sleep(0.35)
        return True

    async def _pause(self):
        await super()._pause()
        return True

    async def _unpause(self):
        await super()._unpause()
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

        res = await self._clients.SpawnPedestrians.call_timeout(req)
        if res is None:
            return tuple(False for _ in pedestrians)
        await self.pedestrian_update(
            arena_people_msgs.msg.Pedestrians(
                pedestrians=[
                    arena_people_msgs.msg.Pedestrian(name=ped.sim_path, pose=ped.pose.to_msg())
                    for status, ped in zip(res.ret, pedestrians)
                    if status
                ]
            )
        )
        return res.ret

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

        res = await self._clients.NavigatePedestrians.call_timeout(NavigatePedestrians.Request(goals=goals))
        return tuple(a and b for a, b in zip(goals, res and res.ret or ()))

    async def _delete_entity(self, name: str) -> bool:
        self._logger.debug(f'Skipping DeletePrims for fresh Isaac eval stage: {name}')
        return True

    async def _delete_pedestrians(self, prim_path):
        self._logger.info(f'Skipping DeletePedestrians for fresh Isaac eval stage: {prim_path}')
        return True

    async def _move_entities(self, actions: Sequence[tuple[str, Pose]]) -> Sequence[bool]:
        response = await self._clients.EditPrims.call_timeout(
            EditPrims.Request(
                prims=[
                    Prim(name=name, pose=pose.to_msg() if hasattr(pose, 'to_msg') else pose)
                    for name, pose in actions
                ],
                pose=True,
            )
        )
        if response is None:
            return [False] * len(actions)

        return response.ret

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
        canonical_load_service = self._clients.LoadUsdScene.client.srv_name
        required_services = {
            client.client.srv_name
            for client in clients
            if client.client.srv_name not in (canonical_spawn_service, canonical_load_service)
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
            load_available = (
                canonical_load_service in available_services
                or self._legacy_load_usd_scene.client.srv_name in available_services
            )
            if not unavailable and spawn_available and load_available:
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
        legacy_load_service = self._legacy_load_usd_scene.client.srv_name
        if (
            canonical_load_service not in available_services
            and legacy_load_service not in available_services
        ):
            missing_services.append(
                f"one of {canonical_load_service} or {legacy_load_service}"
            )

        if missing_services:
            raise RuntimeError(f"Isaac service(s) unavailable after 120s: {', '.join(sorted(missing_services))}")

        spawn_service_override = os.environ.get('ARENA_ISAAC_USD_SPAWN_SERVICE', '').strip()
        # Prefer legacy _srv endpoint over canonical SpawnUsdRobot.
        # The canonical /isaac/SpawnUsdRobot appears in graph queries but
        # rclpy clients cannot connect to it (wait_for_service always fails).
        # The legacy /isaac/SpawnUsdRobot_srv is the one that actually works.
        if legacy_spawn_service in available_services:
            self._spawn_usd_robot_client = self._legacy_spawn_usd_robot
            self._logger.info(
                f'Using legacy SpawnUsdRobot endpoint {legacy_spawn_service} for eval compatibility.'
            )
        elif canonical_spawn_service in available_services:
            self._spawn_usd_robot_client = self._clients.SpawnUsdRobot
            self._logger.info(
                f'Using canonical SpawnUsdRobot endpoint {canonical_spawn_service}; legacy endpoint was unavailable.'
            )
        elif spawn_service_override.endswith('_srv'):
            raise RuntimeError(
                f'ARENA_ISAAC_USD_SPAWN_SERVICE requested {legacy_spawn_service}, but it was unavailable.'
            )
        else:
            self._spawn_usd_robot_client = self._clients.SpawnUsdRobot

        # Prefer legacy _srv endpoint for LoadUsdScene (same FastDDS workaround).
        legacy_load_service = self._legacy_load_usd_scene.client.srv_name
        if legacy_load_service in available_services:
            self._load_usd_scene_client = self._legacy_load_usd_scene
            self._logger.info(
                f'Using legacy LoadUsdScene endpoint {legacy_load_service} for eval compatibility.'
            )

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
        timeout_sec = max(float(os.environ.get('ARENA_ISAAC_LOAD_USD_TIMEOUT_SEC', '1800.0')), 1.0)

        # Workaround: the task_generator's rclpy context cannot reliably send
        # service requests to Isaac Sim across Docker containers due to a
        # FastDDS participant-level routing issue.  The `ros2 service call`
        # CLI works because it creates a fresh DDS participant.  Use a
        # subprocess to call the service with a clean rclpy context.
        try:
            success = await self._load_usd_scene_subprocess(
                usd_path=usd_path,
                scene_prim_path=scene_prim_path,
                scale=scale,
                position=position,
                orientation=orientation,
                add_colliders=add_colliders,
                disable_collision_cooking=disable_collision_cooking,
                timeout_sec=timeout_sec,
            )
            if success:
                self._logger.info(f'USD scene loaded via subprocess: {usd_path}')
            else:
                self._logger.error(f'Failed to load USD scene via subprocess: {usd_path}')
            return success
        except Exception as exc:
            self._logger.error(f'Exception loading USD scene: {exc}\n{traceback.format_exc()}')
            return False

    async def _spawn_usd_robot_subprocess(
        self,
        *,
        name: str,
        usd_path: str,
        robot_namespace: str,
        base_frame: str,
        pose: Pose,
        timeout_sec: float = 300.0,
    ) -> dict | None:
        """Call SpawnUsdRobot via subprocess to get a fresh DDS participant."""
        srv_name = self._spawn_usd_robot_client.client.srv_name
        msg_pose = pose.to_msg() if hasattr(pose, 'to_msg') else pose
        timeout_sec = max(float(timeout_sec), 1.0)

        script = (
            "import json, os, sys, time\n"
            "import rclpy\n"
            "os.environ.setdefault('ROS_DOMAIN_ID', '0')\n"
            "os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')\n"
            "os.environ.setdefault('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET')\n"
            "os.environ.setdefault('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4')\n"
            "from isaacsim_msgs.srv import SpawnUsdRobot\n"
            "rclpy.init()\n"
            "node = rclpy.create_node('spawn_usd_robot_subprocess')\n"
            f"client = node.create_client(SpawnUsdRobot, {srv_name!r})\n"
            f"service_wait_timeout = min(max({timeout_sec!r}, 30.0), 180.0)\n"
            "if not client.wait_for_service(timeout_sec=service_wait_timeout):\n"
            "    print(json.dumps({'success': False, 'message': 'Service unavailable', 'path': ''}), flush=True)\n"
            "    node.destroy_node()\n"
            "    rclpy.shutdown()\n"
            "    sys.exit(1)\n"
            "req = SpawnUsdRobot.Request()\n"
            f"req.name = {name!r}\n"
            f"req.usd_path = {usd_path!r}\n"
            f"req.robot_namespace = {robot_namespace!r}\n"
            f"req.base_frame = {base_frame!r}\n"
            f"req.pose.position.x = {float(msg_pose.position.x)!r}\n"
            f"req.pose.position.y = {float(msg_pose.position.y)!r}\n"
            f"req.pose.position.z = {float(msg_pose.position.z)!r}\n"
            f"req.pose.orientation.x = {float(msg_pose.orientation.x)!r}\n"
            f"req.pose.orientation.y = {float(msg_pose.orientation.y)!r}\n"
            f"req.pose.orientation.z = {float(msg_pose.orientation.z)!r}\n"
            f"req.pose.orientation.w = {float(msg_pose.orientation.w)!r}\n"
            "future = client.call_async(req)\n"
            f"deadline = time.monotonic() + {timeout_sec!r}\n"
            "while time.monotonic() < deadline and not future.done():\n"
            "    rclpy.spin_once(node, timeout_sec=0.5)\n"
            "if future.done():\n"
            "    try:\n"
            "        r = future.result()\n"
            "        print(json.dumps({'success': True, 'message': '', 'path': r.path}), flush=True)\n"
            "    except Exception as exc:\n"
            "        print(json.dumps({'success': False, 'message': str(exc), 'path': ''}), flush=True)\n"
            "else:\n"
            "    print(json.dumps({'success': False, 'message': 'Timeout', 'path': ''}), flush=True)\n"
            "node.destroy_node()\n"
            "rclpy.shutdown()\n"
        )

        self._logger.info(
            f'Calling SpawnUsdRobot via subprocess '
            f'(srv={srv_name}, name={name}, timeout={timeout_sec:.0f}s)'
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                '-c',
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec + 30.0)
        except asyncio.TimeoutError:
            proc.kill()
            self._logger.error(f'SpawnUsdRobot subprocess timed out after {timeout_sec:.0f}s')
            return None

        stdout_text = stdout.decode('utf-8', errors='replace').strip()
        stderr_text = stderr.decode('utf-8', errors='replace').strip()

        if stderr_text:
            self._logger.debug(f'SpawnUsdRobot subprocess stderr: {stderr_text[-500:]}')

        if not stdout_text:
            self._logger.error(f'SpawnUsdRobot subprocess returned no output (rc={proc.returncode})')
            return None

        for line in reversed(stdout_text.split('\n')):
            line = line.strip()
            if not line:
                continue
            try:
                import json
                result = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            if result.get('success'):
                self._logger.info(f"SpawnUsdRobot succeeded: path={result.get('path', '')}")
                return result

            self._logger.error(
                f"SpawnUsdRobot failed via subprocess: {result.get('message', '') or 'unknown error'}"
            )
            return None

        self._logger.error(f'SpawnUsdRobot subprocess: could not parse output: {stdout_text[:200]}')
        return None

    async def _load_usd_scene_subprocess(
        self,
        usd_path: str,
        scene_prim_path: str = '/World/Scene',
        scale: float = 1.0,
        position: list | None = None,
        orientation: list | None = None,
        add_colliders: bool = True,
        disable_collision_cooking: bool = True,
        timeout_sec: float = 1800.0,
    ) -> bool:
        """Call LoadUsdScene via subprocess to get a fresh DDS participant."""
        import asyncio
        pos = position or [0.0, 0.0, 0.0]
        ori = orientation or [0.0, 0.0, 0.0, 1.0]
        srv_name = self._load_usd_scene_client.client.srv_name

        script = (
            "import rclpy, time, os, sys, json\n"
            "os.environ.setdefault('ROS_DOMAIN_ID', '0')\n"
            "os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')\n"
            "os.environ.setdefault('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET')\n"
            "os.environ.setdefault('FASTDDS_BUILTIN_TRANSPORTS', 'UDPv4')\n"
            "from isaacsim_msgs.srv import LoadUsdScene\n"
            "rclpy.init()\n"
            f"node = rclpy.create_node('load_usd_scene_subprocess')\n"
            f"client = node.create_client(LoadUsdScene, '{srv_name}')\n"
            "if not client.wait_for_service(timeout_sec=30.0):\n"
            "    print(json.dumps({'success': False, 'message': 'Service unavailable'}), flush=True)\n"
            "    node.destroy_node()\n"
            "    rclpy.shutdown()\n"
            "    sys.exit(1)\n"
            "req = LoadUsdScene.Request()\n"
            f"req.usd_path = {usd_path!r}\n"
            f"req.scene_prim_path = {scene_prim_path!r}\n"
            f"req.scale = {float(scale)}\n"
            f"req.position = {list(pos)!r}\n"
            f"req.orientation = {list(ori)!r}\n"
            f"req.add_colliders = {bool(add_colliders)}\n"
            f"req.disable_collision_cooking = {bool(disable_collision_cooking)}\n"
            "future = client.call_async(req)\n"
            f"deadline = time.monotonic() + {float(timeout_sec)}\n"
            "while time.monotonic() < deadline and not future.done():\n"
            "    rclpy.spin_once(node, timeout_sec=0.5)\n"
            "if future.done():\n"
            "    r = future.result()\n"
            "    print(json.dumps({'success': r.success, 'message': r.message, 'scene_prim_path': r.scene_prim_path}), flush=True)\n"
            "else:\n"
            "    print(json.dumps({'success': False, 'message': 'Timeout'}), flush=True)\n"
            "node.destroy_node()\n"
            "rclpy.shutdown()\n"
        )

        self._logger.info(f'Calling LoadUsdScene via subprocess (srv={srv_name}, timeout={timeout_sec:.0f}s)')
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, '-c', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec + 30.0)
        except asyncio.TimeoutError:
            proc.kill()
            self._logger.error(f'LoadUsdScene subprocess timed out after {timeout_sec:.0f}s')
            return False

        stdout_text = stdout.decode('utf-8', errors='replace').strip()
        stderr_text = stderr.decode('utf-8', errors='replace').strip()

        if stderr_text:
            self._logger.debug(f'LoadUsdScene subprocess stderr: {stderr_text[-500:]}')

        if not stdout_text:
            self._logger.error(f'LoadUsdScene subprocess returned no output (rc={proc.returncode})')
            return False

        # Parse the last line of stdout as JSON
        for line in reversed(stdout_text.split('\n')):
            line = line.strip()
            if not line:
                continue
            try:
                import json
                result = json.loads(line)
                success = result.get('success', False)
                msg = result.get('message', '')
                prim = result.get('scene_prim_path', '')
                if success:
                    self._logger.info(f'LoadUsdScene succeeded: prim={prim}')
                else:
                    # Scene already loaded is not a fatal error
                    if 'already exists' in msg:
                        self._logger.info(f'LoadUsdScene: scene prim already exists ({prim}), treating as success')
                        return True
                    self._logger.error(f'LoadUsdScene failed: {msg}')
                return bool(success)
            except (json.JSONDecodeError, ValueError):
                continue

        self._logger.error(f'LoadUsdScene subprocess: could not parse output: {stdout_text[:200]}')
        return False
