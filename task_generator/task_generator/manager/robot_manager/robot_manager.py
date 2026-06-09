import asyncio
import json
import math
import os
import time
import typing

import action_msgs.msg
import ament_index_python
import geometry_msgs.msg
import launch.launch_description_sources
import lifecycle_msgs.msg
import nav_msgs.msg as nav_msgs
import rclpy
import rclpy.client
import rclpy.action
import rclpy.logging
import rclpy.publisher
import rclpy.timer
import sensor_msgs.msg as sensor_msgs
from std_msgs.msg import String
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Robot import RobotView
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearCostmapAroundRobot, ClearEntireCostmap
from rosnav_rl_msgs.srv import GetCommand

import launch
import task_generator.utils.arena as Utils
from arena_bringup.extensions import SetGlobalLogLevelAction
from task_generator import NodeInterface
from task_generator.constants import Constants
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.shared import Orientation, Pose, Position, Robot

import rclpy.node

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - PyYAML is present in ROS images
    yaml = None

_TERMINAL_NAV_STATUSES = frozenset({
    action_msgs.msg.GoalStatus.STATUS_SUCCEEDED,
    action_msgs.msg.GoalStatus.STATUS_ABORTED,
    action_msgs.msg.GoalStatus.STATUS_CANCELED,
})


class RobotManager(NodeInterface):
    """
    The robot manager manages the goal and start
    position of a robot for all task modes.

    Args:
        namespace (Namespace): The namespace for the robot.
        environment_manager (EnvironmentManager): The environment manager.
        robot (Robot): The robot instance.
    """

    _namespace: Namespace
    _environment_manager: EnvironmentManager
    _start_pos: Pose
    _goal_pos: Pose
    _pose: Pose
    _robot_radius: float
    _goal_tolerance_distance: float
    _goal_tolerance_angle: float
    _robot: Robot
    _move_base_pub: rclpy.publisher.Publisher
    _goal_pub: rclpy.publisher.Publisher
    _cmd_vel_pub: rclpy.publisher.Publisher
    _pub_goal_timer: rclpy.timer.Timer
    _clear_costmap_around_robot_srv: rclpy.client.Client
    _is_goal_reached: bool
    _nav_stop_ticks: int
    _stop_vel_timer: typing.Optional[rclpy.timer.Timer]
    _rate_setup: rclpy.timer.Rate
    _config: RobotView

    @property
    def robot(self) -> Robot:
        """Get the robot instance.

        Returns:
            Robot: The robot instance.
        """
        return self._robot

    @property
    def start_pos(self) -> Pose:
        """Get the start position.

        Returns:
            Pose: The start position.
        """
        return self._start_pos

    @property
    def goal_pos(self) -> Pose:
        """Get the goal position.

        Returns:
            Pose: The goal position.
        """
        return self._goal_pos

    def __init__(
        self,
        *args,
        namespace: Namespace,
        environment_manager: EnvironmentManager,
        robot: Robot,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rate_setup = self.node.create_rate(.1)

        self._config = robot.model.resolve_sync()

        self._namespace = namespace
        self._environment_manager = environment_manager

        self._start_pos = Pose()
        self._goal_pos = Pose()
        self._is_goal_reached = False
        self._robot_radius = 0.25
        self._cmd_vel_pub = None
        self._nav_stop_ticks = 0
        self._stop_vel_timer = None

        self._goal_tolerance_distance = self.node.conf.Robot.GOAL_TOLERANCE_RADIUS.value
        self._goal_tolerance_angle = self.node.conf.Robot.GOAL_TOLERANCE_ANGLE.value
        self._safety_distance = self.node.conf.Robot.SPAWN_ROBOT_SAFE_DIST.value
        self._navigate_to_pose_client: typing.Optional[rclpy.action.ActionClient] = None
        self._navigate_goal_handle = None
        self._navigate_result_future = None
        self._active_navigation_goal_uuid: tuple[int, ...] | None = None

        self._robot = self.node._environment_manager.realize(robot)
        self._robot.extra.setdefault('namespace', self.namespace)
        self._pose = self._start_pos
        self._goal_timer = None
        self._last_goal_msg: typing.Optional[geometry_msgs.msg.PoseStamped] = None
        self._goal_republish_ticks = 0
        self._camera_ready_topics: dict[str, str] = {}
        self._camera_ready_seen: dict[str, bool] = {}
        self._camera_ready_seen_wall_time: dict[str, float] = {}
        self._dual_vln_status_topic: str | None = None
        self._dual_vln_command_service: str | None = None
        self._dual_vln_status_subscription = None
        self._dual_vln_status: str = 'startup'
        self._dual_vln_status_wall_time: float = 0.0
        self._dual_vln_status_payload: dict = {}
        self._direct_dual_vln_client: rclpy.client.Client | None = None
        self._direct_dual_vln_timer: rclpy.timer.Timer | None = None
        self._direct_dual_vln_future = None
        self._direct_dual_vln_status_bridge_active = False
        self._direct_dual_vln_last_status_twist: geometry_msgs.msg.Twist | None = None
        self._direct_dual_vln_map_bounds: tuple[float, float, float, float] | None = None
        self._direct_dual_vln_bounds_warning_emitted = False

        self._publish_goal_task: typing.Optional[asyncio.Task] = None

    async def set_up_robot(self, node_names: set[str]):
        """Set up the robot by configuring its model and spawning it in the environment.
        """

        self._robot.pose.position.z += self._config.model_params.z_offset
        self._robot = (await self._environment_manager.spawn_robot((self._robot,)))[0]

        _gen_goal_topic = self.namespace("episode_goal_pose")

        goal_qos = QoSProfile(depth=1)
        goal_qos.reliability = ReliabilityPolicy.RELIABLE
        goal_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._start_pub = self.node.create_publisher(
            geometry_msgs.msg.PoseStamped,
            self.namespace("episode_start_pose"),
            goal_qos,
        )
        self._goal_metadata_pub = self.node.create_publisher(
            geometry_msgs.msg.PoseStamped,
            self.namespace("episode_goal_pose_metadata"),
            goal_qos,
        )
        self._goal_pub = self.node.create_publisher(
            geometry_msgs.msg.PoseStamped,
            _gen_goal_topic,
            goal_qos,
        )

        self._cmd_vel_pub = self.node.create_publisher(
            geometry_msgs.msg.Twist,
            self.namespace('cmd_vel'),
            1,
        )

        self._stop_vel_timer = self.node.create_timer(
            0.1,
            self._stop_vel_timer_cb,
        )

        self.node.create_subscription(
            nav_msgs.Odometry,
            self.namespace("odom"),
            self._robot_pos_callback,
            10
        )

        self.node.create_subscription(
            action_msgs.msg.GoalStatusArray,
            self.namespace('navigate_to_pose', '_action', 'status'),
            self._goal_status_callback,
            1
        )

        self._setup_camera_readiness_subscriptions()
        self._ensure_dual_vln_status_subscription()

        await self._launch_robot(node_names)

        self._navigate_to_pose_client = rclpy.action.ActionClient(
            self.node,
            NavigateToPose,
            str(self.namespace('navigate_to_pose')),
        )

        self._robot_radius = self.node.rosparam[float].get(
            'robot_radius',
            self._robot_radius,
        )

    @property
    def safe_distance(self) -> float:
        """Get the safe distance for the robot.

        Returns:
            float: The safe distance for the robot.
        """
        return self._robot_radius + self._safety_distance

    @property
    def model_name(self) -> str:
        """Get the model name of the robot.

        Returns:
            str: The model name of the robot.
        """
        return self._robot.model.name

    @property
    def name(self) -> str:
        """Get the name of the robot.

        Returns:
            str: The name of the robot.
        """
        return self._robot.name

    @property
    def frame(self) -> Namespace:
        """Get the tf2 frame of the robot.

        Returns:
            Namespace: The tf2 frame of the robot.
        """
        return self._robot.frame

    @property
    def namespace(self) -> Namespace:
        """Get the ROS2 namespace of the robot.

        Returns:
            Namespace: The ROS2 namespace of the robot.
        """
        if Utils.get_arena_type() == Constants.ArenaType.TRAINING:
            return Namespace(
                f"{self._namespace}{self._namespace}_{self.model_name}"
            )

        return self._namespace(self.name)

    @property
    async def is_done(self) -> bool:
        """Check if the robot has reached its goal.

        Returns:
            bool: True if the goal is reached, False otherwise.
        """
        return self._is_goal_reached

    async def move_robot_to_pos(self, pose: Pose):
        """Move the robot to the specified pose.

        Args:
            pose(Pose): The target pose for the robot.
        """
        pose.position.z += self._config.model_params.z_offset
        self.robot.pose = pose
        await self._environment_manager.move_robot((self.robot,))
        # Yield once so downstream consumers can process the teleport request.
        await asyncio.sleep(0.05)
        await self._clear_local_costmap(-1)

    async def _clear_local_costmap(self, reset_distance: float = -1) -> bool:
        """Clear the local costmap around the robot.

        Args:
            reset_distance(float, optional): The distance to reset the costmap. Defaults to - 1. If reset_distance is -1, the entire costmap will be cleared. If reset_distance is >= 0, only the costmap around the robot will be cleared.

        Returns:
            bool: True if the costmap was cleared successfully, False otherwise.
        """
        node_name = self.node.service_namespace(self.name, 'local_costmap/local_costmap')

        if reset_distance < 0:
            srv_name = os.path.abspath(node_name('../clear_entirely_local_costmap'))
            srv_type = ClearEntireCostmap
            req = ClearEntireCostmap.Request()
        else:
            srv_name = os.path.abspath(node_name('../clear_around_local_costmap'))
            srv_type = ClearCostmapAroundRobot
            req = ClearCostmapAroundRobot.Request()
            req.reset_distance = reset_distance

        state = await self.node.get_lifecycle_state_async(node_name)
        if state.id != lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
            return False

        self._logger.info(f"Service name: {srv_name}")
        cli = self.node.create_client_wrapper(
            srv_type,
            srv_name,
        )
        await cli.ensure()

        result = await cli.call_timeout(req)
        if result is None:
            self._logger.error(
                f"service call failed for {srv_name}")
            return False
        self._logger.info(
            f"successfull service call for {srv_name}"
        )
        return True

    async def _wait_for_sim_tick(self, timeout_s: float = 1.5) -> bool:
        """Wait until simulation time advances at least once.

        In Isaac reset flow the simulator is paused while positions are reassigned.
        Waiting for a sim tick avoids sending a new goal while nav2 still sees the
        previous odometry state.
        """
        start = self.node.sim_time.to_msg()
        start_stamp = (start.sec, start.nanosec)
        observed_clock_stamp = None

        # `TimeNode.sim_time` is maintained by a shared /clock subscription on
        # the task_generator node.  During Isaac eval startup the node can be
        # busy launching robot subprocesses and serving reset coroutines; add a
        # short-lived local /clock observer so this readiness gate is tied to the
        # real simulator clock topic instead of a potentially stale cached value.
        try:
            import rosgraph_msgs.msg

            def _on_clock(msg: rosgraph_msgs.msg.Clock) -> None:
                nonlocal observed_clock_stamp
                observed_clock_stamp = (msg.clock.sec, msg.clock.nanosec)

            clock_sub = self.node.create_subscription(
                rosgraph_msgs.msg.Clock,
                '/clock',
                _on_clock,
                10,
            )
        except Exception as exc:
            clock_sub = None
            self._logger.warn(f'Unable to create temporary /clock readiness subscription: {exc}')

        period_s = 0.05
        waited_s = 0.0

        try:
            while waited_s < timeout_s:
                await asyncio.sleep(period_s)
                now = self.node.sim_time.to_msg()
                if (now.sec, now.nanosec) != start_stamp:
                    return True
                if observed_clock_stamp is not None and observed_clock_stamp != start_stamp:
                    return True
                waited_s += period_s
        finally:
            if clock_sub is not None:
                try:
                    self.node.destroy_subscription(clock_sub)
                except Exception:
                    pass

        return False

    async def _wait_for_pose_sync(
        self,
        target: Pose,
        timeout_s: float = 2.0,
        xy_tolerance: float = 0.20,
    ) -> bool:
        """Wait until odometry pose is near the teleported start pose."""
        period_s = 0.05
        waited_s = 0.0

        while waited_s < timeout_s:
            dx = self._pose.position.x - target.position.x
            dy = self._pose.position.y - target.position.y
            if (dx * dx + dy * dy) ** 0.5 <= xy_tolerance:
                return True

            await asyncio.sleep(period_s)
            waited_s += period_s

        return False

    async def reset(
        self,
        start_pos: typing.Optional[Pose],
        goal_pos: typing.Optional[Pose],
    ) -> tuple[Pose, Pose]:
        """Reset the robot's position and / or goal.

        Args:
            start_pos(typing.Optional[Pose]): The new starting position of the robot.
            goal_pos(typing.Optional[Pose]): The new goal position of the robot.

        Returns:
            tuple[Pose, Pose]: The new starting and goal positions of the robot.
        """
        if start_pos is not None:
            self._start_pos = self._environment_manager.realize(start_pos)
            await self.move_robot_to_pos(start_pos)
            self._start_pub.publish(self._pose_stamped(self._start_pos))

            if self._robot.record_data_dir:
                self.node.rosparam[list[float]].set(
                    self.namespace.robot_ns.ParamNamespace()("start"),
                    [self.start_pos.position.x, self.start_pos.position.y, self.start_pos.orientation.to_yaw()]
                )
        if goal_pos is not None:
            self._goal_pos = self._environment_manager.realize(goal_pos)
            self._is_goal_reached = False
            self._nav_stop_ticks = 0  # new goal incoming, stop publishing stop-zeros
            self._active_navigation_goal_uuid = None
            self._goal_metadata_pub.publish(self._pose_stamped(self._goal_pos))

            await self._cancel_navigation_goal()

            if self._publish_goal_task is not None:
                self._publish_goal_task.cancel()

            start_target = self._start_pos if start_pos is not None else None
            self._publish_goal_task = asyncio.create_task(
                self._publish_goal_loop(goal=self._goal_pos, start_target=start_target)
            )

            if self._robot.record_data_dir:
                self.node.rosparam[list[float]].set(
                    self.namespace.robot_ns.ParamNamespace()("goal"),
                    [self.goal_pos.position.x, self.goal_pos.position.y, self.goal_pos.orientation.to_yaw()]
                )
        return self._pose, self._goal_pos

    def _pose_stamped(self, pose: Pose) -> geometry_msgs.msg.PoseStamped:
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.node.sim_time.to_msg()
        msg.pose = pose.to_msg()
        return msg

    async def _publish_goal_loop(
        self,
        *,
        goal: Pose,
        start_target: typing.Optional[Pose] = None,
    ):
        """Publish the goal to the robot once reset state is synchronized."""
        wait_for_world_geometry_ready = getattr(self.node, 'wait_for_world_geometry_ready', None)
        if callable(wait_for_world_geometry_ready):
            # Large USD scenes can spend several minutes inside the Isaac
            # LoadUsdScene service before the task generator marks world
            # geometry ready.  Keep the goal barrier longer than the per-sensor
            # readiness checks so slow scene composition does not race goal
            # publication.
            geometry_timeout_s = float(
                self.node.rosparam[float].get('world_geometry_ready_timeout_sec', 600.0)
            )
            if not await wait_for_world_geometry_ready(timeout_s=geometry_timeout_s):
                raise RuntimeError('World geometry did not report ready before goal publish.')

        if not await self._wait_for_sim_tick(timeout_s=120.0):
            raise RuntimeError('Simulation time did not advance before goal publish.')

        if start_target is not None and not await self._wait_for_pose_sync(start_target, timeout_s=120.0):
            raise RuntimeError('Odometry did not reach reset start pose before goal publish.')

        self._reset_navigation_readiness_state()
        self._ensure_dual_vln_status_subscription()
        if not await self._wait_for_camera_ready_before_navigation(timeout_s=90.0):
            raise RuntimeError('Timed out waiting for real camera topics before publishing VLN navigation goal.')
        if not await self._wait_for_dual_vln_status_before_navigation(timeout_s=120.0):
            raise RuntimeError('Timed out waiting for InternNav backend_ready status with fresh real inputs.')
        if not await self._wait_for_dual_vln_command_service_before_navigation(timeout_s=180.0):
            raise RuntimeError('Timed out waiting for dual_vln get_command service before publishing navigation goal.')

        self._logger.info(
            f"Publishing goal once: x={goal.position.x}, y={goal.position.y}, orientation={goal.orientation.to_yaw()}"
        )

        if self._goal_timer is not None:
            self._goal_timer.cancel()
            self._goal_timer.destroy()

        goal_msg = self._pose_stamped(goal)
        self._goal_pub.publish(goal_msg)
        self._last_goal_msg = goal_msg
        self._goal_republish_ticks = 10
        self._start_direct_dual_vln_command_bridge()

        if self._goal_timer is None:
            self._goal_timer = self.node.create_timer(
                0.2,
                self._republish_goal_timer_cb,
            )

        await self._send_navigation_goal(goal_msg)
        self._goal_start_time = self.node.sim_time

    def _default_camera_topics_for_readiness(self) -> dict[str, str] | None:
        if not self._is_dual_vln_robot():
            return None

        model_name = str(self.model_name or '').strip().lower()
        if model_name == 'turtlebot':
            return {
                'rgb': str(self.namespace('rgbd_camera', 'image')),
                'depth': str(self.namespace('rgbd_camera', 'depth_image')),
                'camera_info': str(self.namespace('rgbd_camera', 'camera_info')),
            }
        if model_name in {'ai2_bot2', 'linkhou_s2'}:
            return {
                'rgb': str(self.namespace('head_camera', 'image')),
                'depth': str(self.namespace('head_camera', 'depth')),
                'camera_info': str(self.namespace('head_camera', 'camera_info')),
            }
        return {
            'rgb': str(self.namespace('head_camera', 'image')),
            'depth': str(self.namespace('head_camera', 'depth')),
            'camera_info': str(self.namespace('head_camera', 'camera_info')),
        }

    def _configured_camera_topics_for_readiness(self) -> dict[str, str] | None:
        if not self._is_dual_vln_robot():
            return None

        defaults = self._default_camera_topics_for_readiness() or {}
        topics = {}
        for key, primary_name, legacy_name in (
            ('rgb', 'internnav_rgb_topic', 'dual_vln_rgb_topic'),
            ('depth', 'internnav_depth_topic', 'dual_vln_depth_topic'),
            ('camera_info', 'internnav_camera_info_topic', 'dual_vln_camera_info_topic'),
        ):
            configured = self._get_compat_rosparam(str, primary_name, legacy_name, '', empty_is_missing=True)
            if configured:
                configured = str(configured)
                topics[key] = configured if configured.startswith('/') else str(self.namespace(configured))
            else:
                topics[key] = defaults.get(key, '')
        return topics

    def _reset_navigation_readiness_state(self) -> None:
        if self._camera_ready_seen:
            self._camera_ready_seen = {key: False for key in self._camera_ready_seen}
            self._camera_ready_seen_wall_time = {key: 0.0 for key in self._camera_ready_seen}
        self._dual_vln_status = 'startup'
        self._dual_vln_status_wall_time = 0.0
        self._dual_vln_status_payload = {}
        self._direct_dual_vln_last_status_twist = None

    def _is_dual_vln_robot(self) -> bool:
        """Return true when this robot is configured for the dual_vln planner."""
        candidates = [getattr(self._robot, 'local_planner', '')]
        try:
            candidates.append(self._get_compat_rosparam(str, 'local_planner', 'local_planner', ''))
        except Exception:
            pass
        return any(str(value).strip().lower() == 'dual_vln' for value in candidates)

    def _setup_camera_readiness_subscriptions(self) -> None:
        camera_topics = self._configured_camera_topics_for_readiness()
        if not camera_topics:
            return

        self._camera_ready_topics = camera_topics
        self._camera_ready_seen = {key: False for key in self._camera_ready_topics}
        self._camera_ready_seen_wall_time = {key: 0.0 for key in self._camera_ready_topics}

        # Isaac Sim camera topics are sensor-data streams and are commonly offered
        # as BEST_EFFORT.  Use matching QoS here so the navigation-start readiness
        # barrier observes the same frames as the eval video recorder.
        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        self.node.create_subscription(
            sensor_msgs.Image,
            self._camera_ready_topics['rgb'],
            lambda _msg: self._mark_camera_ready('rgb'),
            sensor_qos,
        )
        self.node.create_subscription(
            sensor_msgs.Image,
            self._camera_ready_topics['depth'],
            lambda _msg: self._mark_camera_ready('depth'),
            sensor_qos,
        )
        self.node.create_subscription(
            sensor_msgs.CameraInfo,
            self._camera_ready_topics['camera_info'],
            lambda _msg: self._mark_camera_ready('camera_info'),
            sensor_qos,
        )

        self._ensure_dual_vln_status_subscription()

    def _ensure_dual_vln_status_subscription(self) -> None:
        if self._dual_vln_status_subscription is not None:
            return
        if not self._is_dual_vln_robot():
            return

        configured_status_topic = self._get_compat_rosparam(
            str,
            'internnav_status_topic',
            'dual_vln_status_topic',
            '',
            empty_is_missing=True,
        )
        self._dual_vln_status_topic = str(configured_status_topic or self.namespace('internnav', 'status'))
        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._dual_vln_status_subscription = self.node.create_subscription(
            String,
            self._dual_vln_status_topic,
            self._on_dual_vln_status,
            status_qos,
        )

    def _recreate_dual_vln_status_subscription(self) -> None:
        if self._dual_vln_status_subscription is not None:
            try:
                self.node.destroy_subscription(self._dual_vln_status_subscription)
            except Exception:
                pass
            self._dual_vln_status_subscription = None
        self._ensure_dual_vln_status_subscription()

    def _dual_vln_command_service_name(self) -> str:
        if self._dual_vln_command_service:
            return self._dual_vln_command_service
        configured = self._get_compat_rosparam(
            str,
            'internnav_command_service',
            'dual_vln_command_service',
            '',
            empty_is_missing=True,
        )
        self._dual_vln_command_service = str(configured or self.namespace('get_command'))
        return self._dual_vln_command_service

    def _requires_real_internnav_backend(self) -> bool:
        return bool(self._get_compat_rosparam(
            bool,
            'internnav_require_real_backend',
            'dual_vln_require_real_backend',
            False,
        ))

    def _mark_camera_ready(self, key: str) -> None:
        if key in self._camera_ready_seen:
            self._camera_ready_seen[key] = True
            self._camera_ready_seen_wall_time[key] = time.monotonic()

    def _on_dual_vln_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        self._dual_vln_status = str(payload.get('status', '') or 'unknown')
        self._dual_vln_status_wall_time = time.monotonic()
        self._dual_vln_status_payload = payload
        self._update_direct_dual_vln_status_twist(payload)
        self._publish_direct_dual_vln_status_command(payload)

    def _dual_vln_status_payload_is_command(self, payload: dict) -> bool:
        """Return true when an InternNav status payload carries a Twist command.

        Heuristic InternNav backends use semantic control statuses such as
        ``drive_to_goal`` / ``arc_to_goal`` rather than the legacy generic
        ``internnav_command`` status.  Treat the presence of finite velocity
        fields as the command contract, while excluding readiness / waiting /
        error states that may also include diagnostic zero values.
        """
        status = str(payload.get('status', '') or '')
        non_command_statuses = {
            'startup',
            'backend_ready',
            'waiting_for_camera',
            'waiting_for_real_backend',
            'camera_waiting',
            'inference_in_progress',
            'missing_pose_or_goal',
            'goal_reached',
            'error',
        }
        if status in non_command_statuses:
            return False
        if 'linear_x' not in payload or 'angular_z' not in payload:
            return False
        try:
            linear_x = float(payload.get('linear_x', 0.0) or 0.0)
            angular_z = float(payload.get('angular_z', 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        return math.isfinite(linear_x) and math.isfinite(angular_z)

    def _dual_vln_status_has_fresh_sensors(self, payload: dict) -> bool:
        debug = payload.get('debug') if isinstance(payload.get('debug'), dict) else {}
        ages = debug.get('sensor_ages_sec') if isinstance(debug.get('sensor_ages_sec'), dict) else {}
        topics = debug.get('topics') if isinstance(debug.get('topics'), dict) else {}
        missing_inputs = set(debug.get('missing_inputs') or []) if isinstance(debug.get('missing_inputs'), list) else set()
        if debug.get('tf_ready') is False:
            return False
        stale_after = debug.get('stale_after_sec', 2.0)
        try:
            stale_after = float(stale_after)
        except (TypeError, ValueError):
            stale_after = 2.0
        required_keys = [
            key for key in ('rgb', 'depth', 'camera_info', 'odom')
            if str(topics.get(key, '') or '').strip()
        ]
        if not required_keys:
            required_keys = ['rgb', 'depth', 'odom']
        for key in required_keys:
            if key in missing_inputs:
                return False
            value = ages.get(key)
            if not isinstance(value, (float, int)):
                return False
            if stale_after > 0.0 and float(value) > stale_after:
                return False
        return True

    def _update_direct_dual_vln_status_twist(self, payload: dict) -> None:
        if not self._dual_vln_status_payload_is_command(payload) and str(payload.get('status', '') or '') != 'internnav_command':
            return
        try:
            linear_x = float(payload.get('linear_x', 0.0) or 0.0)
            angular_z = float(payload.get('angular_z', 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(linear_x) and math.isfinite(angular_z)):
            return

        twist = geometry_msgs.msg.Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self._direct_dual_vln_last_status_twist = twist

    def _publish_direct_dual_vln_status_command(self, payload: dict) -> None:
        """Publish InternNav status commands directly to cmd_vel when active.

        Nav2's dual_vln controller may successfully call get_command and update
        the InternNav status stream while still not emitting a Twist on cmd_vel.
        The status payload already contains the converted linear/angular command,
        so mirror it to cmd_vel during an active dual_vln episode.
        """
        if not self._direct_dual_vln_status_bridge_active:
            return
        if self._cmd_vel_pub is None or self._is_goal_reached or self._nav_stop_ticks > 0:
            return
        if not self._dual_vln_status_payload_is_command(payload) and str(payload.get('status', '') or '') != 'internnav_command':
            return
        twist = self._direct_dual_vln_last_status_twist
        if twist is None:
            return
        if not self._direct_dual_vln_pose_inside_map_bounds():
            self._publish_direct_dual_vln_stop()
            return
        self._cmd_vel_pub.publish(twist)

    def _publish_direct_dual_vln_stop(self) -> None:
        if self._cmd_vel_pub is None:
            return
        self._direct_dual_vln_last_status_twist = None
        self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())

    def _load_direct_dual_vln_map_bounds(self) -> tuple[float, float, float, float] | None:
        if self._direct_dual_vln_map_bounds is not None:
            return self._direct_dual_vln_map_bounds
        if yaml is None:
            return None
        try:
            world = str(self.node.conf.Arena.WORLD.value)
            share = ament_index_python.packages.get_package_share_directory('arena_simulation_setup')
            map_yaml = os.path.join(share, 'worlds', world, 'map', 'map.yaml')
            with open(map_yaml, 'r', encoding='utf-8') as map_file:
                metadata = yaml.safe_load(map_file) or {}
            resolution = float(metadata.get('resolution', 0.0) or 0.0)
            origin = metadata.get('origin') or [0.0, 0.0, 0.0]
            image_path = str(metadata.get('image', '') or '')
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(map_yaml), image_path)
            if resolution <= 0.0 or len(origin) < 2 or not image_path:
                return None
            width, height = self._read_map_image_dimensions(image_path)
            min_x = float(origin[0])
            min_y = float(origin[1])
            max_x = min_x + width * resolution
            max_y = min_y + height * resolution
            self._direct_dual_vln_map_bounds = (min_x, max_x, min_y, max_y)
        except Exception as exc:
            if not self._direct_dual_vln_bounds_warning_emitted:
                self._logger.warn(f'Failed to load map bounds for direct dual_vln bridge safety guard: {exc}')
                self._direct_dual_vln_bounds_warning_emitted = True
            return None
        return self._direct_dual_vln_map_bounds

    def _read_map_image_dimensions(self, image_path: str) -> tuple[int, int]:
        with open(image_path, 'rb') as image_file:
            signature = image_file.read(24)
            if signature.startswith(b'\x89PNG\r\n\x1a\n'):
                return int.from_bytes(signature[16:20], 'big'), int.from_bytes(signature[20:24], 'big')

            image_file.seek(0)
            magic = image_file.readline().strip()
            if magic not in {b'P2', b'P5'}:
                raise ValueError(f'unsupported map image format {magic!r}')
            tokens: list[bytes] = []
            while len(tokens) < 2:
                line = image_file.readline()
                if not line:
                    break
                line = line.split(b'#', 1)[0]
                tokens.extend(line.split())
            if len(tokens) < 2:
                raise ValueError('PGM header does not contain width/height')
            return int(tokens[0]), int(tokens[1])

    def _direct_dual_vln_pose_inside_map_bounds(self) -> bool:
        if not self._is_dual_vln_robot():
            return True
        # Map bounds guard is disabled by default.  The simulation's
        # wall collision and spawn logic already guarantee the robot
        # stays inside the navigable area.  Keeping the code path
        # available so it can be re-enabled via rosparam if needed.
        if not self.node.rosparam[bool].get('direct_dual_vln_map_bounds_enabled', False):
            return True
        bounds = self._load_direct_dual_vln_map_bounds()
        if bounds is None:
            return True
        margin = self.node.rosparam[float].get('direct_dual_vln_map_bounds_margin_m', 0.5)
        min_x, max_x, min_y, max_y = bounds
        x = float(self._pose.position.x)
        y = float(self._pose.position.y)
        inside = (min_x + margin) <= x <= (max_x - margin) and (min_y + margin) <= y <= (max_y - margin)
        if not inside and not self._direct_dual_vln_bounds_warning_emitted:
            self._logger.warn(
                'Direct dual_vln bridge stopped because robot pose is outside the safe map bounds: '
                f'pose=({x:.2f}, {y:.2f}), bounds=({min_x:.2f}, {max_x:.2f}, {min_y:.2f}, {max_y:.2f}), '
                f'margin={margin:.2f}'
            )
            self._direct_dual_vln_bounds_warning_emitted = True
        return inside

    async def _wait_for_camera_ready_before_navigation(self, *, timeout_s: float) -> bool:
        if not self._camera_ready_seen:
            return True
        if all(self._camera_ready_seen.values()):
            return True

        self._logger.info(
            'Waiting for real camera frames before publishing VLN navigation goal: '
            + ', '.join(f'{name}={topic}' for name, topic in self._camera_ready_topics.items())
        )
        deadline = asyncio.get_running_loop().time() + max(float(timeout_s), 0.0)
        last_missing: tuple[str, ...] = tuple()
        while asyncio.get_running_loop().time() < deadline:
            missing = tuple(name for name, seen in self._camera_ready_seen.items() if not seen)
            if not missing:
                self._logger.info('Camera readiness barrier passed; publishing VLN navigation goal.')
                return True
            last_missing = missing
            await asyncio.sleep(0.1)

        self._logger.warn(
            'Timed out waiting for camera readiness before VLN navigation goal; missing '
            + ', '.join(last_missing)
            + '. Failing fast because real camera inputs are required.'
        )
        return False

    async def _wait_for_dual_vln_status_before_navigation(self, *, timeout_s: float) -> bool:
        if not self._is_dual_vln_robot():
            return True

        status_topic = self._dual_vln_status_topic or str(self.namespace('internnav', 'status'))
        external_server_param = self._get_compat_rosparam(
            bool,
            'internnav_external_server',
            'dual_vln_external_server',
            False,
        )
        configured_command_service = self._get_compat_rosparam(
            str,
            'internnav_command_service',
            'dual_vln_command_service',
            '',
            empty_is_missing=True,
        )
        # Some launch paths can lose the compatibility boolean even though the
        # external command endpoint is configured explicitly.  The next barrier
        # still waits for the real service, so this only avoids a stale status
        # circular wait; it does not synthesize backend/sensor readiness.
        external_server = bool(external_server_param) or bool(
            str(configured_command_service or '').strip().startswith('/')
        )
        command_client = None
        if external_server:
            command_client = self.node.create_client(GetCommand, self._dual_vln_command_service_name())
        require_real_backend = self._requires_real_internnav_backend()
        accepted_statuses = {'backend_ready'}
        self._logger.info(
            'Waiting for InternNav status before publishing VLN navigation goal: '
            f'{status_topic} (accepted={sorted(accepted_statuses)})'
        )
        try:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            deadline = start_time + max(float(timeout_s), 0.0)
            last_status = self._dual_vln_status
            status_subscription_recreated = False
            while loop.time() < deadline:
                last_status = self._dual_vln_status
                status_age = time.monotonic() - self._dual_vln_status_wall_time if self._dual_vln_status_wall_time > 0.0 else float('inf')
                if (
                    self._dual_vln_status in accepted_statuses
                    and self._dual_vln_status_wall_time > 0.0
                    and status_age <= 2.5
                    and self._dual_vln_status_has_fresh_sensors(self._dual_vln_status_payload)
                ):
                    self._logger.info(
                        f'InternNav status barrier passed with status={self._dual_vln_status}; publishing VLN navigation goal.'
                    )
                    return True

                if (
                    not status_subscription_recreated
                    and self._dual_vln_status_wall_time <= 0.0
                    and (loop.time() - start_time) >= 2.0
                ):
                    # Long-running external InternNav servers can already be
                    # publishing before task_generator reaches this barrier.
                    # Rejoin the TRANSIENT_LOCAL status topic once after reset so
                    # DDS redelivers the current latched backend_ready heartbeat
                    # instead of waiting out the entire goal-readiness timeout.
                    self._logger.warn(
                        'No InternNav status sample received within 2s; recreating status subscription to rejoin '
                        f'{status_topic} before continuing the backend_ready barrier.'
                    )
                    self._recreate_dual_vln_status_subscription()
                    status_subscription_recreated = True

                # The external InternNav server exposes get_command as the true
                # actuation interface.  In long-running Docker eval sessions DDS
                # can occasionally discover the service while the latched status
                # sample is missed by late-joining task_generator subscriptions;
                # blocking here consumes the whole episode timeout before the
                # direct get_command->cmd_vel bridge can even start.  Once the
                # external service is visible for a few seconds, let the service
                # readiness barrier below be authoritative and proceed.
                if external_server and (loop.time() - start_time) >= 5.0:
                    # The following get_command service barrier is the
                    # authoritative readiness check.  Do not let a missed/stale
                    # transient status sample consume the whole episode before
                    # the direct command bridge can start; DDS can expose the
                    # service to the eval preflight while a just-created client
                    # here still needs discovery time.  This timeout is short
                    # enough to preserve startup diagnostics while avoiding the
                    # old circular wait.
                    service_visible = False
                    if command_client is not None:
                        try:
                            service_visible = bool(command_client.wait_for_service(timeout_sec=0.0))
                        except Exception:
                            service_visible = False
                    if service_visible or not require_real_backend:
                        self._logger.warn(
                            'InternNav status did not publish a fresh backend_ready sample within 5s, but an external '
                            f'server is configured; proceeding with service-readiness gating. '
                            f'service_visible_now={service_visible} require_real_backend={require_real_backend} '
                            f'last_status={last_status!r}'
                        )
                        return True
                await asyncio.sleep(0.1)

            self._logger.warn(
                'Timed out waiting for fresh InternNav backend_ready status before VLN navigation goal; '
                f'last_status={last_status!r}. Failing fast because real backend status is required.'
            )
            return False
        finally:
            if command_client is not None:
                try:
                    self.node.destroy_client(command_client)
                except Exception:
                    pass

    async def _wait_for_dual_vln_command_service_before_navigation(self, *, timeout_s: float) -> bool:
        if not self._is_dual_vln_robot():
            return True

        service_name = self._dual_vln_command_service_name()
        client = self.node.create_client(GetCommand, service_name)
        try:
            self._logger.info(f'Waiting for dual_vln get_command service before publishing navigation goal: {service_name}')
            deadline = asyncio.get_running_loop().time() + max(float(timeout_s), 0.0)
            while asyncio.get_running_loop().time() < deadline:
                if client.wait_for_service(timeout_sec=0.1):
                    self._logger.info('dual_vln get_command service is ready; publishing VLN navigation goal.')
                    return True
                await asyncio.sleep(0.1)
            self._logger.warn(
                f'Timed out waiting for dual_vln get_command service {service_name}; failing fast.'
            )
            return False
        finally:
            try:
                self.node.destroy_client(client)
            except Exception:
                pass


    async def wait_for_pending_goal(self, timeout_s: float) -> bool:
        task = self._publish_goal_task
        if task is None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=max(float(timeout_s), 0.0))
            return True
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f'Timed out waiting for navigation goal readiness for robot {self.name}. '
                'Required TF/odom/camera inputs did not become ready in time.'
            ) from exc
        finally:
            if self._publish_goal_task is task and task.done():
                self._publish_goal_task = None

    def _start_direct_dual_vln_command_bridge(self) -> None:
        """Poll dual_vln get_command and publish its Twist directly.

        In external InternNav Isaac evals the model server can produce valid
        command responses while Nav2's controller path emits no cmd_vel.  The
        namespaced cmd_vel topic is the command source consumed by both the
        Isaac robot graph and the Arena fallback odom bridge, so publish the
        service response directly for the active episode.
        """
        if self._cmd_vel_pub is None:
            return
        self._ensure_dual_vln_status_subscription()
        if self._direct_dual_vln_client is None:
            self._direct_dual_vln_client = self.node.create_client(GetCommand, self._dual_vln_command_service_name())
        self._direct_dual_vln_status_bridge_active = True
        if self._direct_dual_vln_timer is None:
            self._logger.info('Starting direct dual_vln get_command -> cmd_vel bridge')
            self._direct_dual_vln_timer = self.node.create_timer(0.1, self._direct_dual_vln_timer_cb)

    def _stop_direct_dual_vln_command_bridge(self) -> None:
        if self._direct_dual_vln_timer is not None:
            try:
                self._direct_dual_vln_timer.cancel()
                self._direct_dual_vln_timer.destroy()
            except Exception:
                pass
            self._direct_dual_vln_timer = None
        self._direct_dual_vln_future = None
        self._direct_dual_vln_status_bridge_active = False

    def _direct_dual_vln_timer_cb(self) -> None:
        if self._cmd_vel_pub is None or self._is_goal_reached or self._nav_stop_ticks > 0:
            return
        if self._direct_dual_vln_last_status_twist is not None:
            if self._direct_dual_vln_pose_inside_map_bounds():
                self._cmd_vel_pub.publish(self._direct_dual_vln_last_status_twist)
            else:
                self._publish_direct_dual_vln_stop()
                return
        client = self._direct_dual_vln_client
        if client is None:
            return
        if self._direct_dual_vln_future is not None and not self._direct_dual_vln_future.done():
            return
        if not client.service_is_ready():
            return

        future = client.call_async(GetCommand.Request())
        self._direct_dual_vln_future = future

        def _publish_response(done_future) -> None:
            if self._cmd_vel_pub is None or self._is_goal_reached or self._nav_stop_ticks > 0:
                return
            try:
                response = done_future.result()
                twist = getattr(response, 'twist', None)
                if twist is None:
                    return
                self._direct_dual_vln_last_status_twist = twist
                if not self._direct_dual_vln_pose_inside_map_bounds():
                    self._publish_direct_dual_vln_stop()
                    return
                self._cmd_vel_pub.publish(twist)
            except Exception as exc:
                self._logger.warn(f'direct dual_vln get_command bridge failed: {exc}')

        future.add_done_callback(_publish_response)

    def _republish_goal_timer_cb(self) -> None:
        if self._goal_republish_ticks <= 0 or self._last_goal_msg is None:
            return

        goal_msg = geometry_msgs.msg.PoseStamped()
        goal_msg.header = self._last_goal_msg.header
        goal_msg.header.stamp = self.node.sim_time.to_msg()
        goal_msg.pose = self._last_goal_msg.pose
        self._goal_pub.publish(goal_msg)
        self._goal_republish_ticks -= 1

    async def _cancel_navigation_goal(self) -> None:
        if self._navigate_goal_handle is None:
            return

        try:
            cancel_future = self._navigate_goal_handle.cancel_goal_async()
            await self._wait_for_rclpy_future(cancel_future, timeout_s=5.0)
        except Exception as exc:
            self._logger.warn(f'Failed to cancel previous navigate_to_pose goal: {exc}')
        finally:
            self._navigate_goal_handle = None
            self._navigate_result_future = None
            self._active_navigation_goal_uuid = None

    async def _wait_for_rclpy_future(self, future, *, timeout_s: float = 5.0):
        waited_s = 0.0
        period_s = 0.05
        while not future.done() and waited_s < timeout_s:
            await asyncio.sleep(period_s)
            waited_s += period_s

        if not future.done():
            raise TimeoutError('rclpy future timed out')

        return future.result()

    async def _send_navigation_goal(self, goal_msg: geometry_msgs.msg.PoseStamped) -> None:
        if self._navigate_to_pose_client is None:
            self._logger.warn('navigate_to_pose action client is not initialized; goal will only be published for observers')
            return

        self._active_navigation_goal_uuid = None

        bt_node_path = str(self.namespace('bt_navigator'))
        bt_wait_timeout_s = 60.0
        if not await self.node.wait_for_lifecycle_state_async(
            bt_node_path,
            lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE,
            timeout=bt_wait_timeout_s,
        ):
            self._logger.warn('bt_navigator did not become active before sending navigate_to_pose goal')
            return

        action_server_ready = False
        for _ in range(15):
            if self._navigate_to_pose_client.wait_for_server(timeout_sec=2.0):
                action_server_ready = True
                break
            await asyncio.sleep(0.2)

        if not action_server_ready:
            self._logger.warn('navigate_to_pose action server is unavailable; goal will only be published for observers')
            return

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg

        try:
            send_goal_future = self._navigate_to_pose_client.send_goal_async(nav_goal)
            goal_handle = await self._wait_for_rclpy_future(send_goal_future, timeout_s=5.0)
        except Exception as exc:
            self._logger.error(f'Failed to send navigate_to_pose action goal: {exc}')
            return

        if not goal_handle.accepted:
            self._logger.warn('navigate_to_pose action goal was rejected')
            self._navigate_goal_handle = None
            self._navigate_result_future = None
            self._active_navigation_goal_uuid = None
            return

        self._navigate_goal_handle = goal_handle
        self._navigate_result_future = goal_handle.get_result_async()
        self._active_navigation_goal_uuid = self._goal_uuid_tuple(getattr(goal_handle, 'goal_id', None))
        if self._active_navigation_goal_uuid is None:
            self._logger.warn('navigate_to_pose action goal accepted without a readable goal UUID; status filtering is disabled for this goal')
        self._logger.info('navigate_to_pose action goal accepted')

        def _done_callback(future):
            try:
                result = future.result()
                status = result.status if result is not None else 'unknown'
                self._logger.info(f'navigate_to_pose action finished with status={status}')
            except Exception as exc:
                self._logger.warn(f'navigate_to_pose action result retrieval failed: {exc}')

        self._navigate_result_future.add_done_callback(_done_callback)

    async def _launch_robot(self, node_paths: set[str]):
        """Launch the robot external nodes.
        """
        self._logger.info(f"LAUNCH ROBOT {self.name}")

        if Utils.get_arena_type() != Constants.ArenaType.TRAINING:
            launch_description = launch.LaunchDescription()
            current_log_level = rclpy.logging.get_logger_effective_level(self.node.get_logger().name).name.lower()
            launch_description.add_action(SetGlobalLogLevelAction(current_log_level))

            internnav_mode = self._get_compat_rosparam(str, 'internnav_mode', 'dual_vln_mode', 'heuristic')
            internnav_model_path = self._get_compat_rosparam(
                str, 'internnav_model_path', 'dual_vln_model_path', '', empty_is_missing=True
            )
            internnav_device = self._get_compat_rosparam(str, 'internnav_device', 'dual_vln_device', 'cpu')
            internnav_inference_rate_hz = self._get_compat_rosparam(
                float, 'internnav_inference_rate_hz', 'dual_vln_inference_rate_hz', 10.0
            )
            internnav_inference_timeout_sec = self._get_compat_rosparam(
                float, 'internnav_inference_timeout_sec', 'dual_vln_inference_timeout_sec', 0.2
            )
            internnav_rgb_topic = self._get_compat_rosparam(
                str, 'internnav_rgb_topic', 'dual_vln_rgb_topic', '', empty_is_missing=True
            )
            internnav_depth_topic = self._get_compat_rosparam(
                str, 'internnav_depth_topic', 'dual_vln_depth_topic', '', empty_is_missing=True
            )
            internnav_camera_info_topic = self._get_compat_rosparam(
                str, 'internnav_camera_info_topic', 'dual_vln_camera_info_topic', '', empty_is_missing=True
            )
            internnav_python_executable = self._get_compat_rosparam(
                str, 'internnav_python_executable', 'dual_vln_python_executable', '', empty_is_missing=True
            )
            internnav_adapter_target = self._get_compat_rosparam(
                str, 'internnav_adapter_target', 'dual_vln_adapter_target', '', empty_is_missing=True
            )
            internnav_require_real_backend = self._get_compat_rosparam(
                bool, 'internnav_require_real_backend', 'dual_vln_require_real_backend', False
            )
            internnav_strict_device = self._get_compat_rosparam(
                bool, 'internnav_strict_device', 'dual_vln_strict_device', False
            )
            internnav_look_down = self._get_compat_rosparam(
                bool, 'internnav_look_down', 'dual_vln_look_down', False
            )
            internnav_enable_visualization = self._get_compat_rosparam(
                bool, 'internnav_enable_visualization', 'dual_vln_enable_visualization', False
            )
            internnav_visualization_topic = self._get_compat_rosparam(
                str,
                'internnav_visualization_topic',
                'dual_vln_visualization_topic',
                'internnav/debug_image',
                empty_is_missing=True,
            )
            internnav_visualization_rate_hz = self._get_compat_rosparam(
                float, 'internnav_visualization_rate_hz', 'dual_vln_visualization_rate_hz', 5.0
            )
            internnav_external_server = self._get_compat_rosparam(
                bool, 'internnav_external_server', 'dual_vln_external_server', False
            )
            if (
                internnav_mode.strip().lower() == 'internnav'
                or os.environ.get('ARENA_INTERNNAV_EXTERNAL_SERVER', '').strip().lower() in {'1', 'true', 'yes', 'on'}
            ):
                internnav_external_server = True

            launch_arguments = {
                'robot': self.model_name,
                # 'simulator': self.node.conf.Arena.SIM.value.value,
                # 'name': self.name,
                'task_generator_node': os.path.join(self.node.get_namespace(), self.node.get_name()),
                'namespace': self.namespace,
                # 'use_namespace': 'True',
                'frame': f'{self._robot.frame}/',
                'inter_planner': self._robot.inter_planner,
                'global_planner': self._robot.global_planner,
                'local_planner': self._robot.local_planner,
                # 'complexity': self.node.declare_parameter('complexity', 1).value,
                'train_mode': str(self.node._train_mode).lower(),
                'agent_name': self._robot.agent,
                'use_sim_time': 'True',
                'amcl': 'true' if self.node.conf.Arena.SIM.value in (Constants.SimSimulator.GAZEBO,) else 'false',
                'internnav_mode': internnav_mode,
                'dual_vln_mode': internnav_mode,
                'internnav_model_path': internnav_model_path,
                'dual_vln_model_path': internnav_model_path,
                'internnav_device': internnav_device,
                'dual_vln_device': internnav_device,
                'internnav_inference_rate_hz': str(internnav_inference_rate_hz),
                'dual_vln_inference_rate_hz': str(internnav_inference_rate_hz),
                'internnav_inference_timeout_sec': str(internnav_inference_timeout_sec),
                'dual_vln_inference_timeout_sec': str(internnav_inference_timeout_sec),
                'internnav_rgb_topic': internnav_rgb_topic,
                'dual_vln_rgb_topic': internnav_rgb_topic,
                'internnav_depth_topic': internnav_depth_topic,
                'dual_vln_depth_topic': internnav_depth_topic,
                'internnav_camera_info_topic': internnav_camera_info_topic,
                'dual_vln_camera_info_topic': internnav_camera_info_topic,
                'internnav_python_executable': internnav_python_executable,
                'dual_vln_python_executable': internnav_python_executable,
                'internnav_adapter_target': internnav_adapter_target,
                'dual_vln_adapter_target': internnav_adapter_target,
                'internnav_require_real_backend': str(internnav_require_real_backend).lower(),
                'dual_vln_require_real_backend': str(internnav_require_real_backend).lower(),
                'internnav_strict_device': str(internnav_strict_device).lower(),
                'dual_vln_strict_device': str(internnav_strict_device).lower(),
                'internnav_look_down': str(internnav_look_down).lower(),
                'dual_vln_look_down': str(internnav_look_down).lower(),
                'internnav_enable_visualization': str(internnav_enable_visualization).lower(),
                'dual_vln_enable_visualization': str(internnav_enable_visualization).lower(),
                'internnav_visualization_topic': internnav_visualization_topic,
                'dual_vln_visualization_topic': internnav_visualization_topic,
                'internnav_visualization_rate_hz': str(internnav_visualization_rate_hz),
                'dual_vln_visualization_rate_hz': str(internnav_visualization_rate_hz),
                'internnav_external_server': str(internnav_external_server).lower(),
                'dual_vln_external_server': str(internnav_external_server).lower(),
                # Nav2 Jazzy collision_monitor currently rejects the model-wrapper
                # polygon parameters during lifecycle configure on the dual_vln /
                # InternNav path. Disable it for that local planner so eval bringup
                # is gated by the actual controller/model readiness instead of an
                # unrelated parameter typing issue inside collision_monitor.
                'enable_collision_monitor': str(
                    self.node.rosparam[bool].get(
                        'enable_collision_monitor',
                        not self._is_dual_vln_robot(),
                    )
                ).lower(),
            }

            if self._robot.record_data_dir:
                launch_arguments.update({
                    'record_data_dir': self._robot.record_data_dir,
                })

            launch_description.add_action(
                launch.actions.IncludeLaunchDescription(
                    launch.launch_description_sources.PythonLaunchDescriptionSource(
                        os.path.join(
                            ament_index_python.packages.get_package_share_directory('arena_simulation_setup'),
                            'launch/robot.launch.py'
                        )
                    ),
                    launch_arguments=launch_arguments.items(),
                )
            )
            await self.node.do_launch(launch_description)

            bt_node_path = str(self.namespace('bt_navigator'))
            self._logger.info(f'waiting for {bt_node_path}')
            while bt_node_path not in node_paths:
                await asyncio.sleep(0.01)

    def _robot_pos_callback(self, data: nav_msgs.Odometry):
        """Callback for robot position updates.

        Args:
            data(nav_msgs.Odometry): The odometry data containing the robot's position.
        """
        current_position = data.pose.pose
        quat = current_position.orientation

        if not all(math.isfinite(float(value)) for value in (
            current_position.position.x,
            current_position.position.y,
            quat.x,
            quat.y,
            quat.z,
            quat.w,
        )):
            self._logger.warn('Ignoring non-finite odometry pose update')
            return

        self._pose = Pose(
            Position(
                current_position.position.x,
                current_position.position.y,
            ),
            Orientation.from_msg(quat)
        )

        # Treat an episode as complete only when the robot is physically close
        # to the active task goal.  Nav2 action status arrays can transiently
        # contain terminal states from earlier/canceled goals; using pose here
        # prevents the task generator from ending an eval while the robot is
        # still several meters away from the actual goal.
        if self._distance_to_goal() <= self._goal_tolerance_distance:
            self._is_goal_reached = True

    def _distance_to_goal(self) -> float:
        try:
            dx = self._pose.position.x - self._goal_pos.position.x
            dy = self._pose.position.y - self._goal_pos.position.y
            return math.hypot(dx, dy)
        except Exception:
            return float('inf')

    def _get_compat_rosparam(
        self,
        type_: type,
        primary_name: str,
        legacy_name: str,
        default,
        *,
        empty_is_missing: bool = False,
    ):
        value = self.node.rosparam[type_].get(primary_name, None)
        if value is not None and (not empty_is_missing or value != ''):
            return value

        legacy_value = self.node.rosparam[type_].get(legacy_name, None)
        if legacy_value is not None and (not empty_is_missing or legacy_value != ''):
            return legacy_value

        return default

    def _stop_vel_timer_cb(self):
        """Publish zero velocity when the robot should be stopped."""
        if self._nav_stop_ticks > 0 and self._cmd_vel_pub is not None:
            self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())
            self._nav_stop_ticks -= 1

    @staticmethod
    def _goal_uuid_tuple(goal_identifier) -> tuple[int, ...] | None:
        if goal_identifier is None:
            return None

        goal_info = getattr(goal_identifier, 'goal_info', None)
        if goal_info is not None:
            return RobotManager._goal_uuid_tuple(getattr(goal_info, 'goal_id', None))

        uuid_value = getattr(goal_identifier, 'uuid', None)
        if uuid_value is None:
            return None

        if isinstance(uuid_value, (bytes, bytearray)):
            return tuple(int(value) for value in uuid_value)

        if isinstance(uuid_value, (list, tuple)):
            try:
                return tuple(int(value) for value in uuid_value)
            except (TypeError, ValueError):
                return None

        return None

    def _goal_status_callback(self, data: action_msgs.msg.GoalStatusArray):
        """Callback for goal status updates.

        Args:
            data(action_msgs.msg.GoalStatusArray): The goal status data.
        """
        active_goal_uuid = self._active_navigation_goal_uuid
        if active_goal_uuid is None:
            return

        last_goal = next(
            (
                goal_status
                for goal_status in reversed(list(data.status_list))
                if self._goal_uuid_tuple(goal_status) == active_goal_uuid
            ),
            None,
        )
        if last_goal is None:
            return

        status = last_goal.status if last_goal is not None else None
        reached = (
            status == action_msgs.msg.GoalStatus.STATUS_SUCCEEDED
            and self._distance_to_goal() <= max(self._goal_tolerance_distance, 0.05)
        )
        if status == action_msgs.msg.GoalStatus.STATUS_SUCCEEDED and not reached:
            self._logger.warn(
                'Ignoring navigate_to_pose success status because robot is still '
                f'{self._distance_to_goal():.2f} m from goal; waiting for physical goal reach.'
            )
        if status in _TERMINAL_NAV_STATUSES:
            self._active_navigation_goal_uuid = None
            self._goal_republish_ticks = 0
            # In dual_vln / InternNav mode Nav2 may report SUCCEEDED
            # while the physical robot is still far from the goal.
            # Only tear down the command bridge when the robot has
            # actually reached the goal or the navigation was aborted
            # / cancelled.
            if status != action_msgs.msg.GoalStatus.STATUS_SUCCEEDED or reached:
                self._stop_direct_dual_vln_command_bridge()
                if self._cmd_vel_pub is not None:
                    self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())
                self._nav_stop_ticks = 15  # 1.5 s at 10 Hz
        if reached:
            self._is_goal_reached = True

    async def update(self):
        """Live - update some kwargs of robot
        """
        # TODO implement record data dir

    async def destroy(self):
        """Destroy robot and remove from simulation and navigation stack.
        """
        if self._goal_timer is not None:
            self._goal_timer.cancel()
            self._goal_timer.destroy()
            self._goal_timer = None
        self._stop_direct_dual_vln_command_bridge()
        if self._direct_dual_vln_client is not None:
            try:
                self.node.destroy_client(self._direct_dual_vln_client)
            except Exception:
                pass
            self._direct_dual_vln_client = None
        if self._stop_vel_timer is not None:
            self._stop_vel_timer.cancel()
            self._stop_vel_timer.destroy()
            self._stop_vel_timer = None
        await self._environment_manager.remove_robot((self.robot,))
        # TODO kill node in navigation stack
