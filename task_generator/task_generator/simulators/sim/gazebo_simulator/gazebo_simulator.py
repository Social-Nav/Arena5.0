import asyncio
import itertools
import math
import tempfile
import time
import traceback
from pathlib import Path
import typing

import arena_robots.Robot
import launch_ros
import rclpy.time
import rclpy.duration
import tf2_ros
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from ros_gz_interfaces.msg import Entity as EntityMsg
from ros_gz_interfaces.msg import EntityFactory, WorldControl
from ros_gz_interfaces.srv import ControlWorld, DeleteEntity, SetEntityPose, SpawnEntity

import launch
from task_generator.shared import (
    Entity,
    Model,
    ModelType,
    ModelWrapper,
    Pose,
    Robot,
    Wall,
    FrameNamespace,
)
from task_generator.simulators.sim import BaseSim

from .robot_bridge import BridgeConfiguration

# sanitize frames, gazebo does not support slashes
FrameNamespace.auto_sanitize()

_GZ_BRIDGED_SERVICE_TIMEOUT_SEC = 3.0


class GazeboSimulator(BaseSim):

    def __init__(self, *args, namespace, **kwargs):
        super().__init__(*args, namespace=namespace, **kwargs)

        self._semaphore = asyncio.Semaphore(5)
        self._control_world_available = False

        self._logger.info(f"Initializing GazeboSimulator with namespace: {namespace}")

        self._goal_pub = self.node.create_publisher(
            PoseStamped,
            self._namespace("goal"),
            10,
        )
        self._spawn_model_files: dict[str, Path] = {}
        self.entities: dict[str, Entity] = {}
        self._walls_entities: list[str] = []
        self._wall_counter = itertools.count()

        # TF buffer for looking up current odom→base_link transforms
        # Used to compute correct map→odom TF after robot teleportation
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

    async def before_reset_task(self):
        self._logger.warn("Pausing simulation before reset")
        return bool(await self.pause_simulation())

    async def after_reset_task(self):
        unpause_result = await self.unpause_simulation()

        if not unpause_result:
            self._logger.warning("Failed to unpause simulation after reset")

        # Small delay for sensors and physics to stabilize after reset
        await asyncio.sleep(0.3)

        return unpause_result

    async def obstacle_spawn(self, obstacles):
        return await asyncio.gather(*map(self._spawn_entity, obstacles))

    async def pedestrian_spawn(self, pedestrians):
        return await asyncio.gather(*map(self._spawn_entity, pedestrians))

    async def robot_spawn(self, robots):

        async def impl(robot: Robot) -> bool:
            if not await self._spawn_entity(robot):
                return False
            _loader_args = {**robot.asdict(), 'sim_path': getattr(robot, 'sim_path', robot.name)}
            model = await (await robot.model.resolve()).model.get(
                ModelType.URDF, loader_args=_loader_args
            )
            if model.type is ModelType.UNKNOWN:
                return False
            model_description = model.description
            self._robot_initialpose(robot)
            await self._robot_bridge(robot, model_description)
            return True

        success = await asyncio.gather(*map(impl, robots))
        return success

    async def obstacle_move(self, obstacles):
        return await asyncio.gather(*map(self._move_entity, obstacles))

    async def pedestrian_move(self, pedestrians):
        # Gazebo does not support modifying actors after spawning
        return (True,) * len(pedestrians)

    async def robot_move(self, robots):
        async def impl(robot: Robot) -> bool:
            return (await self._move_entity(robot)) and (await self._robot_move(robot))

        return await asyncio.gather(*map(impl, robots))

    async def obstacle_delete(self, obstacles):
        return await asyncio.gather(*(self._delete_entity(o.name) for o in obstacles))

    async def pedestrian_delete(self, pedestrians):
        # Gazebo does not support deleting actors after spawning
        return (True,) * len(pedestrians)

    async def robot_delete(self, robots):
        return await asyncio.gather(
            *(self._delete_entity(robot.name) for robot in robots)
        )

    async def pedestrian_update(self, pedestrians):
        # Gazebo does not support modifying actors after spawning
        return (True,) * len(pedestrians.pedestrians)

    async def spawn_floors(self, floors):
        # Gazebo does not support spawning floors
        del floors
        return True

    async def spawn_doors(self, doors):
        # Gazebo does not support spawning doors
        del doors
        return True

    async def spawn_elevators(self, elevators):
        # Gazebo does not support spawning elevators
        del elevators
        return True

    # IMPL

    async def _move_entity(self, entity: Entity):
        async with self._semaphore:
            name = entity.sim_path
            pose = entity.pose
            self._logger.debug(f"Attempting to move entity: {name}")
            self._logger.debug(f"Moving entity {name} to position: {pose}")

            request = SetEntityPose.Request()
            request.entity = EntityMsg(
                name=name,
                type=EntityMsg.MODEL,
            )
            request.pose = pose.to_msg()

            try:
                await self._service_set_entity_pose.ensure()
                result = await self._service_set_entity_pose.call_timeout(
                    request,
                    timeout_sec=_GZ_BRIDGED_SERVICE_TIMEOUT_SEC,
                )

                if result is None:
                    self._logger.warning(
                        f"Move service call timed out for {name}; retrying with `gz service`"
                    )
                    return await self._move_entity_via_gz(name, pose)

                self._logger.info(f"Move result for {name}: {result.success}")

                return result.success

            except Exception as e:
                self._logger.error(f"Error moving entity {name}: {str(e)}")
                traceback.print_exc()
                return False

    async def _spawn_entity(self, entity: Entity) -> bool:
        async with self._semaphore:
            try:

                # Get model description
                try:
                    if isinstance(entity, Robot):
                        _loader_args = {**entity.asdict(), 'sim_path': getattr(entity, 'sim_path', entity.name)}
                        model = await (await entity.model.resolve()).model.get(
                            ModelType.URDF, loader_args=_loader_args
                        )
                    else:
                        model = await (await entity.model.resolve()).get(ModelType.SDF)
                except Exception as e:
                    self._logger.error(
                        f"Error resolving model for entity {entity.name}: {e}\n{traceback.format_exc()}"
                    )
                    return False

                if model.type is ModelType.UNKNOWN:
                    self._logger.error(
                        f"Error resolving model for entity {entity.name}: unknown model type {model}"
                    )
                    return False

                if model.path and model.type not in (ModelType.URDF,):
                    # direct path available, use gz cli call
                    return await self._spawn_entity_via_gz(
                        entity=entity,
                        model_file=str(model.path),
                    )

                else:
                    # no direct path available, use ros_gz_bridge

                    # Create spawn request
                    request = SpawnEntity.Request()
                    request.entity_factory = EntityFactory()
                    request.entity_factory.name = entity.sim_path
                    model_description = model.description
                    request.entity_factory.sdf = model_description

                    # Set pose
                    request.entity_factory.pose = entity.pose.to_msg()

                    self._logger.info(
                        f"Spawn position for {entity.name}: x={entity.pose.position.x}, y={entity.pose.position.y}"
                    )

                    self._logger.debug(f"Sending spawn request for {entity.name}")
                    result = await self._service_spawn_entity.call_timeout(
                        request,
                        timeout_sec=_GZ_BRIDGED_SERVICE_TIMEOUT_SEC,
                    )

                    if result is None:
                        self._logger.warning(
                            f"Spawn service call timed out for {entity.name}; retrying with `gz service`"
                        )
                        return await self._spawn_entity_via_gz(
                            entity=entity,
                            model_text=model.description,
                            model_type=model.type,
                        )

                    self._logger.info(f"Spawn result for {entity.name}: {result.success}")

                    self.entities[entity.name] = entity

                    return result.success

            except Exception as e:
                self._logger.error(f"Error spawning entity {entity.name}: {str(e)}")
                traceback.print_exc()
                return False

    async def _delete_entity(self, name: str):
        return True
        async with self._semaphore:
            name = name

            self._logger.debug(f"Attempting to delete entity: {name}")

            if name not in self.entities:
                return False

            self._logger.debug(f"Attempting to delete entity: {name}")
            request = DeleteEntity.Request()
            request.entity = EntityMsg(
                name=name,
                type=EntityMsg.MODEL,
            )

            try:
                result = await self._service_delete_entity.call_timeout(request)

                if result is None:
                    self._logger.error(f"Delete service call failed for {name}")
                    return False

                self._logger.debug(f"Delete result for {name}: {result.success}")

                if result.success:
                    del self.entities[name]

                return result.success

            except Exception as e:
                self._logger.error(f"Error deleting entity {name}: {str(e)}")
                traceback.print_exc()
                return False

    async def pause_simulation(self):
        async with self._semaphore:
            if not self._control_world_available:
                self._logger.warning("Control world service unavailable; skipping pause request")
                return True
            self._logger.debug("Attempting to pause simulation")
            request = ControlWorld.Request()
            request.world_control = WorldControl()
            request.world_control.pause = True

            try:
                result = await self._service_control_world.call_timeout(request)

                if result is None:
                    self._logger.error("Pause service call failed")
                    return False

                self._logger.debug(f"Pause result: {result.success}")
                return result.success

            except Exception as e:
                self._logger.error(f"Error pausing simulation: {str(e)}")
                traceback.print_exc()
                return False

    async def unpause_simulation(self):
        async with self._semaphore:
            if not self._control_world_available:
                self._logger.warning("Control world service unavailable; skipping unpause request")
                return True
            self._logger.debug("Attempting to unpause simulation")
            request = ControlWorld.Request()
            request.world_control = WorldControl()
            request.world_control.pause = False

            try:
                result = await self._service_control_world.call_timeout(request)

                if result is None:
                    self._logger.error("Unpause service call failed")
                    return False

                self._logger.debug(f"Unpause result: {result.success}")
                return result.success

            except Exception as e:
                self._logger.error(f"Error unpausing simulation: {str(e)}")
                traceback.print_exc()
                return False

    async def step_simulation(self, steps):
        async with self._semaphore:
            if not self._control_world_available:
                self._logger.warning("Control world service unavailable; skipping step request")
                return True
            self._logger.debug(f"Stepping simulation by {steps} steps")
            request = ControlWorld.Request()
            request.world_control = WorldControl()
            request.world_control.multi_step = steps

            try:
                result = await self._service_control_world.call_timeout(request)

                if result is None:
                    self._logger.error("Step service call failed")
                    return False

                self._logger.debug(f"Step result: {result.success}")
                return result.success

            except Exception as e:
                self._logger.error(f"Error stepping simulation: {str(e)}")
                traceback.print_exc()
                return False

    def _publish_goal(self, goal: Pose):
        self._logger.info(
            f"Publishing goal: x={goal.position.x}, y={goal.position.y}, orientation={goal.orientation}"
        )
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.node.sim_time.to_msg()
        goal_msg.header.frame_id = "map"
        goal_msg.pose = goal.to_msg()
        self._goal_pub.publish(goal_msg)
        self._logger.info("Goal published")

    async def spawn_walls(self, walls) -> bool:
        await self.remove_world()  # Clear existing walls
        for wall in walls:  # only walls, ignore obstacles
            wall_name = self.node._environment_manager.realize(
                f"wall_{next(self._wall_counter)}"
            )
            wall_height = 2.0  # Wall height in meters
            wall_thickness = 0.05  # Wall thickness in meters
            base_position = (0, 0, 0)  # Offset the wall to (10, 10, 0)

            self._logger.debug(
                f"Attempting to spawn wall: {wall_name} from {wall.start} to {wall.end}"
            )

            # Generate the SDF string for walls
            wall_sdf = _generate_wall_sdf(
                name=wall_name,
                walls=[wall],
                height=wall_height,
                thickness=wall_thickness,
                base_position=base_position,
            )

            if not wall_sdf:
                self._logger.error(f"Failed to generate SDF for wall: {wall_name}")
                continue

            entity = Entity(
                pose=Pose(),
                model=ObjectIdentifier.inline(
                    ModelWrapper.from_model(
                        Model(
                            type=ModelType.SDF,
                            name=wall_name,
                            description=wall_sdf,
                            path=None,
                        )
                    )
                ),
                name=wall_name,
                extra={},
            )

            await self._spawn_entity(entity)
            self._walls_entities.append(wall_name)

        return True

    async def remove_world(self) -> bool:
        for entity in self._walls_entities:
            await self._delete_entity(entity)
        self._walls_entities = []
        self._wall_counter = itertools.count()
        return True

    async def _robot_bridge(self, robot: Robot, description: str):
        launch_description = launch.LaunchDescription()

        launch_description.add_action(
            launch_ros.actions.PushRosNamespace(
                namespace=self.node.service_namespace(robot.name)
            )
        )

        robot_config = arena_robots.Robot.RobotIdentifier(
            robot.model.name
        ).resolve_sync()

        mappings = BridgeConfiguration.from_file(robot_config.mappings).substitute(
            {
                "robot_name": robot.sim_path,
                "world": "/world/default",
            }
        )

        bridge_arguments = mappings.as_args()
        remappings = mappings.as_remappings()

        # Add parameter_bridge node
        launch_description.add_action(
            launch_ros.actions.Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                output="screen",
                arguments=bridge_arguments,
                remappings=remappings,
                parameters=[{"use_sim_time": True}],
            )
        )
        launch_description.add_action(
            launch_ros.actions.Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                remappings=[
                    ("/tf_static", "/tf_static"),
                ],
                parameters=[
                    {"use_sim_time": True},
                    {"robot_description": description},
                    {"frame_prefix": robot.frame + "/"},  # add trailing slash
                ],
            )
        )

        # launch_description.add_action(
        #     launch_ros.actions.Node(
        #         package='joint_state_publisher',
        #         executable='joint_state_publisher',
        #         output='screen',
        #         parameters=[
        #             {'use_sim_time': True},
        #             {'robot_description': description},  # Ensure URDF is passed here too
        #         ],
        #         remappings=[('/joint_states', '/joint_states')]
        #     )
        # )
        await self.node.do_launch(launch_description)

    def _robot_initialpose(self, robot: Robot):
        pose = PoseWithCovarianceStamped()
        pose.pose.pose = robot.pose.to_msg()
        pose.header.frame_id = "map"

        self.node.create_publisher(
            PoseWithCovarianceStamped,
            self.node.service_namespace(robot.name, "initialpose"),
            qos_profile=1,
        ).publish(pose)

    @staticmethod
    def _quaternion_to_yaw(qx, qy, qz, qw):
        """Extract yaw angle from quaternion."""
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_to_quaternion(yaw):
        """Convert yaw angle to quaternion (x, y, z, w)."""
        return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)

    def _compute_map_to_odom_tf(
        self,
        desired_x, desired_y, desired_z,
        desired_qx, desired_qy, desired_qz, desired_qw,
        odom_frame_name: str,
        base_frame_name: str,
    ):
        """Compute the correct map→odom TF accounting for current DiffDrive odom.

        The DiffDrive odometry plugin integrates wheel rotations and does NOT
        reset when the robot is teleported. So after teleportation, the odom
        frame still has the accumulated wheel motion offset. We must compute:
            map_to_odom = desired_map_pose * inv(current_odom_to_base)
        so that the robot appears at the desired position in the map frame.

        Returns:
            tuple: (tf_x, tf_y, tf_z, tf_qx, tf_qy, tf_qz, tf_qw)
        """
        try:
            # Look up odom → base_link TF (the current DiffDrive odom value)
            odom_tf = self._tf_buffer.lookup_transform(
                odom_frame_name,
                base_frame_name,
                rclpy.time.Time(),  # latest available
                timeout=rclpy.duration.Duration(seconds=2.0),
            )

            odom_x = odom_tf.transform.translation.x
            odom_y = odom_tf.transform.translation.y
            odom_yaw = self._quaternion_to_yaw(
                odom_tf.transform.rotation.x,
                odom_tf.transform.rotation.y,
                odom_tf.transform.rotation.z,
                odom_tf.transform.rotation.w,
            )

            desired_yaw = self._quaternion_to_yaw(
                desired_qx, desired_qy, desired_qz, desired_qw,
            )

            # Compute map→odom TF: desired_map_pos = TF * odom_pos
            # Orientation: tf_yaw = desired_yaw - odom_yaw
            tf_yaw = desired_yaw - odom_yaw

            # Position: desired = tf_translation + R(tf_yaw) * odom_translation
            # => tf_translation = desired - R(tf_yaw) * odom_translation
            cos_tf = math.cos(tf_yaw)
            sin_tf = math.sin(tf_yaw)
            tf_x = desired_x - (odom_x * cos_tf - odom_y * sin_tf)
            tf_y = desired_y - (odom_x * sin_tf + odom_y * cos_tf)
            tf_z = desired_z

            tf_qx, tf_qy, tf_qz, tf_qw = self._yaw_to_quaternion(tf_yaw)

            self._logger.info(
                f"Corrected map→odom TF: "
                f"odom=({odom_x:.2f}, {odom_y:.2f}, {math.degrees(odom_yaw):.1f}°) "
                f"desired=({desired_x:.2f}, {desired_y:.2f}) "
                f"TF=({tf_x:.2f}, {tf_y:.2f}, {math.degrees(tf_yaw):.1f}°)"
            )

            return tf_x, tf_y, tf_z, tf_qx, tf_qy, tf_qz, tf_qw

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self._logger.warn(
                f"Could not look up odom TF ({odom_frame_name} → {base_frame_name}), "
                f"using desired position directly (OK for first spawn): {e}"
            )
            # Fallback: use desired position directly (correct when odom = identity)
            return (
                desired_x, desired_y, desired_z,
                desired_qx, desired_qy, desired_qz, desired_qw,
            )

    async def _robot_move(self, robot: Robot) -> bool:
        name = robot.name
        try:

            self._robot_initialpose(robot)

            max_attempts = 3
            attempt = 1
            initial_pose_triggered = False

            while attempt <= max_attempts and not initial_pose_triggered:
                self._logger.info(
                    f"Attempt {attempt}/{max_attempts}: Triggering initial pose update for robot {name}"
                )
                try:
                    self._robot_initialpose(robot)
                    initial_pose_triggered = True
                    self._logger.info(
                        f"Initial pose update for {name} succeeded on attempt {attempt}"
                    )
                except Exception as e:
                    self._logger.error(
                        f"Attempt {attempt}/{max_attempts} failed for {name}: {str(e)}"
                    )
                    traceback.print_exc()
                    if attempt < max_attempts:
                        self._logger.info("Waiting 1 second before retrying...")
                        time.sleep(1)
                    attempt += 1

            if not initial_pose_triggered:
                self._logger.error(
                    f"Failed to set initial pose for {name} after {max_attempts} attempts"
                )

            robot_config = arena_robots.Robot.RobotIdentifier(robot.model.name).resolve_sync()
            odom_frame = robot_config.model_params.odom_frame
            base_frame = robot_config.model_params.base_frame

            # Get frame names as raw strings (FrameNamespace.__str__ is sanitized
            # by auto_sanitize, but TF frames in the tree use unsanitized '/' names)
            odom_frame_name = str.__str__(robot.frame(odom_frame))
            base_frame_name = str.__str__(robot.frame(base_frame))

            # Compute the correct map→odom TF accounting for DiffDrive odom offset
            tf_x, tf_y, tf_z, tf_qx, tf_qy, tf_qz, tf_qw = self._compute_map_to_odom_tf(
                desired_x=robot.pose.position.x,
                desired_y=robot.pose.position.y,
                desired_z=robot.pose.position.z,
                desired_qx=robot.pose.orientation.x,
                desired_qy=robot.pose.orientation.y,
                desired_qz=robot.pose.orientation.z,
                desired_qw=robot.pose.orientation.w,
                odom_frame_name=odom_frame_name,
                base_frame_name=base_frame_name,
            )

            transform_pub_node = launch_ros.actions.Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_odomframe_publisher",
                arguments=[
                    str(tf_x),
                    str(tf_y),
                    str(tf_z),
                    str(tf_qx),
                    str(tf_qy),
                    str(tf_qz),
                    str(tf_qw),
                    "map",
                    f"{robot.frame}/{odom_frame}",
                ],
                parameters=[{"use_sim_time": True}],
            )
            await self.node.do_launch(launch.LaunchDescription([transform_pub_node]))

            return True

        except Exception as e:
            self._logger.error(f"Error moving robot {name}: {str(e)}")
            return False

    async def _call_gz_service(
        self,
        *,
        service_name: str,
        reqtype: str,
        reptype: str,
        request_payload: str,
        timeout_ms: int = 5000,
    ) -> bool:
        process = await asyncio.create_subprocess_exec(
            "gz",
            "service",
            "-s",
            service_name,
            "--reqtype",
            reqtype,
            "--reptype",
            reptype,
            "--timeout",
            str(timeout_ms),
            "--req",
            request_payload,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode(errors="replace").strip()
        stderr_text = stderr.decode(errors="replace").strip()

        if process.returncode != 0:
            self._logger.error(
                f"`gz service` call failed for {service_name}: {stderr_text or stdout_text}"
            )
            return False

        if stdout_text:
            self._logger.info(f"`gz service` response for {service_name}: {stdout_text}")

        return True

    def _persist_spawn_model(
        self,
        *,
        entity_name: str,
        model_text: str,
        model_type: ModelType | None,
    ) -> str:
        suffix = ".sdf"
        if model_type is ModelType.URDF:
            suffix = ".urdf"

        cache_dir = Path(tempfile.gettempdir()) / "arena_gz_spawn_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / f"{entity_name}{suffix}"
        model_path.write_text(model_text, encoding="utf-8")
        self._spawn_model_files[entity_name] = model_path
        return str(model_path)

    async def _spawn_entity_via_gz(
        self,
        *,
        entity: Entity,
        model_file: str | None = None,
        model_text: str | None = None,
        model_type: ModelType | None = None,
    ) -> bool:
        if model_file is None:
            model_file = self._persist_spawn_model(
                entity_name=entity.sim_path,
                model_text=model_text or "",
                model_type=model_type,
            )

        request_payload = (
            f'name: "{entity.sim_path}" '
            f'sdf_filename: "{model_file}" '
            f'pose: {{ '
            f'position: {{ x: {entity.pose.position.x}, y: {entity.pose.position.y}, z: {entity.pose.position.z} }} '
            f'orientation: {{ x: {entity.pose.orientation.x}, y: {entity.pose.orientation.y}, z: {entity.pose.orientation.z}, w: {entity.pose.orientation.w} }} '
            f'}}'
        )

        success = await self._call_gz_service(
            service_name="/world/default/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            request_payload=request_payload,
        )
        if success:
            self.entities[entity.name] = entity
        return success

    async def _move_entity_via_gz(self, name: str, pose: Pose) -> bool:
        request_payload = (
            f'name: "{name}" '
            f'position: {{ x: {pose.position.x}, y: {pose.position.y}, z: {pose.position.z} }} '
            f'orientation: {{ x: {pose.orientation.x}, y: {pose.orientation.y}, z: {pose.orientation.z}, w: {pose.orientation.w} }}'
        )

        return await self._call_gz_service(
            service_name="/world/default/set_pose",
            reqtype="gz.msgs.Pose",
            reptype="gz.msgs.Boolean",
            request_payload=request_payload,
        )

    async def _gz_service_has_provider(self, service_name: str) -> bool:
        """Check whether Gazebo transport currently advertises a service provider."""
        try:
            process = await asyncio.create_subprocess_exec(
                "gz",
                "service",
                "-s",
                service_name,
                "--info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        except FileNotFoundError:
            self._logger.warning("`gz` CLI not found while probing Gazebo transport services")
            return False

        output = b"\n".join(part for part in (stdout, stderr) if part).decode(
            errors="replace"
        )

        return process.returncode == 0 and "No service providers" not in output

    async def _wait_for_gz_service_provider(
        self,
        service_name: str,
        timeout_sec: float = 30.0,
        interval_sec: float = 0.5,
    ) -> bool:
        """Wait until Gazebo transport exposes a provider for a service."""
        start = time.monotonic()

        while (time.monotonic() - start) < timeout_sec:
            if await self._gz_service_has_provider(service_name):
                self._logger.info(
                    f"Gazebo transport service ready: {service_name}"
                )
                return True

            await asyncio.sleep(interval_sec)

        self._logger.warning(
            f"Gazebo transport service not ready after {timeout_sec:.1f}s: {service_name}"
        )
        return False

    async def _set_up_services(self):
        futures: list[typing.Awaitable] = []
        futures.append(
            self.node.do_launch(
                launch.LaunchDescription(
                    [
                        launch_ros.actions.Node(
                            package="ros_gz_bridge",
                            executable="parameter_bridge",
                            name="gz_services_bridge",
                            output="screen",
                            arguments=[
                                "/world/default/create@ros_gz_interfaces/srv/SpawnEntity@gz.msgs.EntityFactory@gz.msgs.Boolean",
                                "/world/default/remove@ros_gz_interfaces/srv/DeleteEntity@gz.msgs.Entity@gz.msgs.Boolean",
                                "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose@gz.msgs.Pose@gz.msgs.Boolean",
                                "/world/default/control@ros_gz_interfaces/srv/ControlWorld@gz.msgs.WorldControl@gz.msgs.Boolean",
                            ],
                            parameters=[{"use_sim_time": True}],
                        )
                    ]
                )
            )
        )

        # Initialize service clients
        # https://gazebosim.org/api/sim/8/entity_creation.html
        self._service_spawn_entity = self.node.create_client_wrapper(
            SpawnEntity,
            "/world/default/create",
        )
        self._service_delete_entity = self.node.create_client_wrapper(
            DeleteEntity,
            "/world/default/remove",
        )
        self._service_set_entity_pose = self.node.create_client_wrapper(
            SetEntityPose,
            "/world/default/set_pose",
        )
        self._service_control_world = self.node.create_client_wrapper(
            ControlWorld,
            "/world/default/control",
        )

        self._logger.info("Waiting for Gazebo transport services...")
        for service_name in (
            "/world/default/create",
            "/world/default/remove",
            "/world/default/set_pose",
            "/world/default/control",
        ):
            await self._wait_for_gz_service_provider(service_name)

        self._logger.info("Waiting for gazebo services...")
        required_services = (
            (self._service_spawn_entity, "spawn entity"),
            (self._service_delete_entity, "delete entity"),
            (self._service_set_entity_pose, "set entity pose"),
        )

        for service, name in required_services:
            self._logger.info(f"Waiting for {name} service...")
            futures.append(service.ensure())

        self._logger.info("Probing optional control world service...")

        await asyncio.gather(*futures)

        self._control_world_available = await self._service_control_world.ensure(timeout_sec=2.0)
        if self._control_world_available:
            self._logger.info("Control world service is available.")
        else:
            self._logger.warning("Control world service is unavailable; continuing without pause/unpause support.")
        self._logger.info("All Gazebo services are available now.")

    @classmethod
    async def create(cls, *args, namespace, **kwargs) -> "GazeboSimulator":
        simulator = cls(*args, namespace=namespace, **kwargs)
        await simulator._set_up_services()
        return simulator


def _generate_wall_sdf(
    name: str,
    walls: list[Wall],
    height: float,
    thickness: float,
    base_position: tuple[float, float, float] = (0, 0, 0),
) -> str:
    """
    Generate an SDF string for a wall structure based on given parameters and base position.
    """
    sdf_template = """
        <sdf version="1.6">
            <model name="{name}">
                <pose>{base_x} {base_y} {base_z} 0 0 0</pose>
                {links}
                <static>true</static>
            </model>
        </sdf>
        """
    link_template = """
        <link name="wall_segment_{index}">
            <visual name="visual">
                <geometry>
                    <box>
                        <size>{length} {thickness} {height}</size>
                    </box>
                </geometry>
                <material>
                    <ambient>0.7 0.7 0.7 1</ambient>
                </material>
            </visual>
            <collision name="collision">
                <geometry>
                    <box>
                        <size>{length} {thickness} {height}</size>
                    </box>
                </geometry>
            </collision>
            <pose>{x} {y} {z} 0 0 {orientation}</pose>
        </link>
        """
    links = []
    base_x, base_y, base_z = base_position
    z = height / 2.0  # Center the wall height relative to the base

    for i, w in enumerate(walls):
        x1, y1, x2, y2 = w.start.x, w.start.y, w.end.x, w.end.y
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        orientation = math.atan2(y2 - y1, x2 - x1)
        x = (x1 + x2) / 2 + base_x
        y = (y1 + y2) / 2 + base_y

        links.append(
            link_template.format(
                index=i,
                length=length,
                thickness=thickness,
                height=height,
                x=x,
                y=y,
                z=z + base_z,
                orientation=orientation,
            )
        )

    return sdf_template.format(
        name=name, base_x=base_x, base_y=base_y, base_z=base_z, links="\n".join(links)
    )
