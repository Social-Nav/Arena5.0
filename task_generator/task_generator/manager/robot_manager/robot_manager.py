import asyncio
import os
import typing

import yaml

import action_msgs.msg
from action_msgs.srv import CancelGoal
import ament_index_python
import arena_bringup.extensions.NodeLogLevelExtension as NodeLogLevelExtension
import geometry_msgs.msg
import launch.launch_description_sources
import launch_ros
import lifecycle_msgs.msg
import nav_msgs.msg as nav_msgs
import rclpy
import rclpy.client
import rclpy.logging
import rclpy.publisher
import rclpy.timer
import std_msgs.msg
from std_srvs.srv import Trigger
import tf2_ros
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Robot import RobotView
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import ClearCostmapAroundRobot, ClearEntireCostmap
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType

import launch
import task_generator.utils.arena as Utils
from task_generator import NodeInterface
from task_generator.constants import Constants
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.manager.robot_manager.scene_readiness import LocalCostmapReadiness
from task_generator.shared import Orientation, Pose, Position, Robot

import rclpy.node

_TERMINAL_NAV_STATUSES = frozenset({
    action_msgs.msg.GoalStatus.STATUS_SUCCEEDED,
    action_msgs.msg.GoalStatus.STATUS_ABORTED,
    action_msgs.msg.GoalStatus.STATUS_CANCELED,
})
_ACTIVE_NAV_STATUSES = frozenset({
    action_msgs.msg.GoalStatus.STATUS_ACCEPTED,
    action_msgs.msg.GoalStatus.STATUS_EXECUTING,
    action_msgs.msg.GoalStatus.STATUS_CANCELING,
})
_DEFAULT_GOAL_TOLERANCE = 0.25
_GOAL_MATCH_TOLERANCE = 1.0e-3


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
    _route_status_pub: rclpy.publisher.Publisher
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
        # Name of the profile applied by the last set_social_attributes call. Only used to
        # look up that profile's vx ceiling when warning about a too-large scenario override.
        self._active_social_attributes: typing.Optional[str] = None

        self._goal_tolerance_distance = self.node.conf.Robot.GOAL_TOLERANCE_RADIUS.value
        self._goal_tolerance_angle = self.node.conf.Robot.GOAL_TOLERANCE_ANGLE.value
        self._safety_distance = self.node.conf.Robot.SPAWN_ROBOT_SAFE_DIST.value

        self._robot = self.node._environment_manager.realize(robot)
        self._robot.extra.setdefault('namespace', self.namespace)
        self._pose = self._start_pos
        self._goal_timer = None

        self._publish_goal_task: typing.Optional[asyncio.Task] = None
        self._goal_waypoints: list[Pose] = []
        self._goal_waypoint_index = 0
        self._active_nav_goal_id: bytes | None = None
        self._ignored_nav_goal_ids: set[bytes] = set()
        self._known_nav_goal_ids: set[bytes] = set()
        self._goal_id_baseline: set[bytes] = set()
        self._latest_nav_statuses: list = []
        self._goal_publish_stamp_ns: int | None = None
        self._waypoint_tolerance = _DEFAULT_GOAL_TOLERANCE
        self._final_goal_tolerance = _DEFAULT_GOAL_TOLERANCE
        self._social_yielding_active = False
        self._awaiting_original_goal = False
        self._resume_prepared_baseline: set[bytes] | None = None

        # Isaac publishes the episode id only after spawned characters have live animation
        # graphs and the scene has accumulated a short real-simulation settling window.
        # The original Nav2 goal is withheld until the ids match, so the robot stays at A
        # while pedestrians and rolling costmaps initialize.
        self._scene_episode: typing.Optional[int] = None
        self._isaac_scene_ready_episode: typing.Optional[int] = None
        self._scene_motion_released = False
        scene_qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._scene_ready_pub = self.node.create_publisher(
            std_msgs.msg.Int16, '/scene_ready', scene_qos)
        self._scene_ready_pub.publish(std_msgs.msg.Int16(data=-1))
        self.node.create_subscription(
            std_msgs.msg.Int16,
            '/task_generator_node/task_reset',
            self._scene_episode_callback,
            scene_qos,
        )
        self.node.create_subscription(
            std_msgs.msg.Int16,
            '/isaac/scene_ready',
            self._isaac_scene_ready_callback,
            scene_qos,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
        self._local_costmap_readiness = LocalCostmapReadiness(
            sensor_frames=self._local_costmap_sensor_frames(),
            lookup_xy=self._lookup_frame_xy,
        )
        self.node.create_subscription(
            Costmap,
            self.namespace('local_costmap', 'costmap_raw'),
            self._local_costmap_readiness.receive,
            10,
        )

    async def _odom_base_transform(self):
        """Launch a static transform publisher for odometry to base frame.
        """
        await self.node.do_launch(
            launch.LaunchDescription([
                launch_ros.actions.Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name="odom_to_baseframe_publisher",
                    arguments=[
                        "0", "0", "0",
                        "0", "0", "0", "1",
                        self.frame(self._config.model_params.odom_frame),
                        self.frame(self._config.model_params.base_frame),
                    ],
                    parameters=[{'use_sim_time': True}],
                )
            ])
        )

    async def set_up_robot(self, node_names: set[str]):
        """Set up the robot by configuring its model and spawning it in the environment.
        """

        self._robot.pose.position.z += self._config.model_params.z_offset
        self._robot = (await self._environment_manager.spawn_robot((self._robot,)))[0]

        _gen_goal_topic = self.namespace("goal_pose")

        self._goal_pub = self.node.create_publisher(
            geometry_msgs.msg.PoseStamped,
            _gen_goal_topic,
            10,
        )
        self._route_status_pub = self.node.create_publisher(
            std_msgs.msg.Int8,
            self.namespace("route_status"),
            10,
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
        self.node.create_subscription(
            std_msgs.msg.Bool,
            '/social_yielding/active',
            self._social_yielding_active_callback,
            rclpy.qos.QoSProfile(
                depth=1,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        # Observe every goal sent to Nav2, including temporary yielding goals published by
        # the orchestrator. A goal matching the scenario's final goal marks the hand-back to
        # normal navigation after yielding.
        self.node.create_subscription(
            geometry_msgs.msg.PoseStamped,
            _gen_goal_topic,
            self._goal_command_callback,
            10,
        )
        self.node.create_service(
            Trigger,
            self.namespace("prepare_original_goal_resume"),
            self._prepare_original_goal_resume_callback,
        )

        await self._launch_robot(node_names)
        if self.node.conf.Arena.SIM.value != Constants.SimSimulator.ISAAC:
            await self._odom_base_transform()

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

    async def _lifecycle_state_retry(
        self,
        node_name: str,
        *,
        attempts: int = 3,
        timeout: float = 3.0,
    ) -> typing.Optional[lifecycle_msgs.msg.State]:
        """get_lifecycle_state_async, retried, for use on the reset path.

        WHY THIS EXISTS. The whole reset runs between Isaac's PauseSimulation and
        UnpauseSimulation (isaac_simulator.before_reset_task / after_reset_task). While Isaac is
        paused its main loop never calls world.step(), so the ROS2 bridge stops publishing /clock
        and SIM TIME IS FROZEN. Every nav2 node here runs with use_sim_time:=True, so a frozen
        clock means their wait sets never time out and their service executors can stall — a
        get_state request issued in that window can simply go unanswered until the unpause.

        A single 3 s attempt therefore reports "node not up yet?" for a node that is up and
        ACTIVE, and the caller silently skips real work: a skipped social_attributes push leaves
        the episode running the PREVIOUS scenario's profile, which is invisible in the logs.

        Retrying is the cheap fix (the frozen window is short and bounded by the unpause) and it
        keeps the honest failure mode: still None after all attempts -> genuinely absent.
        """
        for attempt in range(1, attempts + 1):
            state = await self.node.get_lifecycle_state_async(node_name, timeout=timeout)
            if state is not None:
                if attempt > 1:
                    self._logger.info(
                        f"{node_name} get_state answered on attempt {attempt}/{attempts}")
                return state
            if attempt < attempts:
                self._logger.warn(
                    f"{node_name} get_state timed out (attempt {attempt}/{attempts}); "
                    "sim clock is frozen while Isaac is paused — retrying")
                await asyncio.sleep(0.5)
        return None

    async def _clear_costmap(self, node_name: str, srv_name: str) -> bool:
        """Call ClearEntireCostmap on the given service, guarded by lifecycle state."""
        state = await self._lifecycle_state_retry(node_name)
        if state is None:
            self._logger.warn(
                f"{node_name} get_state unavailable (node not up yet?); skipping costmap clear")
            return False
        if state.id != lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
            return False

        self._logger.info(f"Service name: {srv_name}")
        cli = self.node.create_client_wrapper(ClearEntireCostmap, srv_name)
        try:
            await cli.ensure()

            result = await cli.call_timeout(ClearEntireCostmap.Request())
            if result is None:
                self._logger.error(f"service call failed for {srv_name}")
                return False
            self._logger.info(f"successfull service call for {srv_name}")
            return True
        finally:
            # Destroy: one client per reset would otherwise accumulate for the whole session.
            self.node.destroy_client(cli.client)

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

        state = await self._lifecycle_state_retry(node_name)
        if state is None:
            self._logger.warn(
                f"{node_name} get_state unavailable (node not up yet?); skipping costmap clear")
            return False
        if state.id != lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
            return False

        self._logger.info(f"Service name: {srv_name}")
        cli = self.node.create_client_wrapper(
            srv_type,
            srv_name,
        )
        try:
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
        finally:
            # Destroy: one client per reset would otherwise accumulate for the whole session.
            self.node.destroy_client(cli.client)

    async def _clear_global_costmap(self) -> bool:
        """Clear the entire global costmap (removes dynamic obstacle marks from previous episode)."""
        node_name = self.node.service_namespace(self.name, 'global_costmap/global_costmap')
        srv_name = os.path.abspath(node_name('../clear_entirely_global_costmap'))
        return await self._clear_costmap(node_name, srv_name)

    # Map a profile.yaml section -> (target node, param-name prefix).
    #
    # The knobs live on the controller server, so a profile push is grouped by its
    # parameter prefix:
    #   - controller_server: the MPC. It is wrapped by RotationShim (plugin "FollowPath"), which
    #     configures the primary controller with the SAME name, so social_mpc params sit directly
    #     under FollowPath.* (verified). Hot-reloaded via the MPC's dynamic-parameter callback.
    # Global-costmap SocialLayer parameters are shared across profiles and stay in
    # nav2.yaml; local avoidance comes from obstacle/inflation layers plus the MPC's
    # own /people critics.
    _PROFILE_TARGETS: dict[str, tuple[str, str]] = {
        'weights': ('controller_server', 'FollowPath.optimizer.weights.'),
        # Optimizer knobs that live directly under `optimizer.` rather than `optimizer.weights.`
        # (currently just max_linear_velocity, the hard vx ceiling + VelocityCost target). Kept as
        # its own section because the MPC's dynamic-param callback matches these two prefixes
        # separately, and the weights.* branch would not recognize this key.
        'weights_optimizer': ('controller_server', 'FollowPath.optimizer.'),
        'trajectorizer': ('controller_server', 'FollowPath.trajectorizer.'),
    }

    def _load_social_attributes(self, name: str) -> dict[str, dict[str, float]]:
        """Read configs/nav2/profiles/<name>/profile.yaml -> {node: {param: value}}."""
        share = ament_index_python.packages.get_package_share_directory('arena_simulation_setup')
        path = os.path.join(share, 'configs', 'nav2', 'profiles', name, 'profile.yaml')
        if not os.path.isfile(path):
            self._logger.error(f"social_attributes '{name}' not found at {path}; skipping")
            return {}
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        unknown = set(data.keys()) - set(self._PROFILE_TARGETS.keys())
        if unknown:
            self._logger.warn(
                f"social_attributes '{name}': ignoring unknown section(s) {sorted(unknown)}")

        by_node: dict[str, dict[str, float]] = {}
        for section, (node, prefix) in self._PROFILE_TARGETS.items():
            for key, value in (data.get(section) or {}).items():
                by_node.setdefault(node, {})[prefix + key] = float(value)
        return by_node

    async def _push_params(self, node: str, params: dict[str, float], label: str) -> bool:
        """SetParameters on one lifecycle node, guarded by its lifecycle state."""
        node_name = self.node.service_namespace(self.name, node)
        srv_name = os.path.abspath(node_name('set_parameters'))

        # Retried: a single attempt inside the Isaac pause window reports None for a node that is
        # ACTIVE, and silently skipping the push leaves the PREVIOUS profile in force for the
        # whole episode. Escalated to error() because that is a real, silent behavior regression.
        state = await self._lifecycle_state_retry(str(node_name))
        if state is None:
            self._logger.error(
                f"{node_name} get_state unavailable; SKIPPED {label} push — this robot is still "
                "running the previously-applied profile for this episode")
            return False
        if state.id != lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
            self._logger.warn(f"{node_name} not ACTIVE; skipping {label} push")
            return False

        req = SetParameters.Request()
        req.parameters = [
            ParameterMsg(
                name=name,
                value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=val),
            )
            for name, val in params.items()
        ]
        cli = self.node.create_client_wrapper(SetParameters, srv_name)
        try:
            await cli.ensure()
            result = await cli.call_timeout(req)
            if result is None or not all(r.successful for r in result.results):
                self._logger.error(f"{label} push failed on {srv_name}")
                return False
            self._logger.info(f"applied {label} ({len(params)} params) to {node_name}")
            return True
        finally:
            # Destroy: one client per reset would otherwise accumulate for the whole session.
            self.node.destroy_client(cli.client)

    async def set_social_attributes(self, attributes: typing.Optional[str]) -> bool:
        """Push a social-attributes preset to this robot's MPC at reset.

        The MPC hot-reloads the profile via its dynamic-parameter callback, so the
        preset takes effect immediately without a relaunch.
        No-op when attributes is falsy (caller resolves the default profile).
        """
        if not attributes:
            return True
        by_node = self._load_social_attributes(attributes)
        if not by_node:
            return False
        # Remembered so set_desired_linear_vel can read this profile's vx ceiling for its
        # clipping warning without racing the push it just made.
        self._active_social_attributes = attributes

        label = f"social_attributes '{attributes}'"
        results = [
            await self._push_params(node, params, label)
            for node, params in by_node.items()
        ]
        return all(results)

    async def set_desired_linear_vel(self, vel: typing.Optional[float]) -> bool:
        """Push a per-robot cruise-speed override, on top of whatever profile just went out.

        MUST be called AFTER set_social_attributes: both write the same key
        (FollowPath.trajectorizer.desired_linear_vel), and last write wins. That ordering is
        what makes the scenario field outrank the profile while leaving every other profile
        knob untouched.

        No-op when vel is None (field absent -> the profile's value stands).
        """
        if vel is None:
            return True

        if vel <= 0.0:
            self._logger.error(
                f"desired_linear_vel={vel} is not positive; ignoring it and keeping the "
                f"profile's cruise speed")
            return False

        # Warn on a value the stack cannot deliver. Two independent clamps sit above this
        # knob and NEITHER moves with it, so a larger value is silently clipped and the robot
        # just runs at the ceiling -- which looks like "the field did nothing".
        #   optimizer.max_linear_velocity        -> Ceres upper bound on vx (profile key)
        #   velocity_smoother_max_velocity[0]    -> model_params.yaml, clips the output
        ceiling = self._profile_max_linear_velocity()
        if ceiling is not None and vel > ceiling:
            self._logger.warn(
                f"desired_linear_vel={vel} exceeds the vx ceiling {ceiling} "
                f"(optimizer.max_linear_velocity); it will be clipped, so the robot will "
                f"cruise at {ceiling} instead. Raise the ceiling in the profile AND "
                f"velocity_smoother_max_velocity[0] if you really want more.")

        node, prefix = self._PROFILE_TARGETS['trajectorizer']
        return await self._push_params(
            node, {prefix + 'desired_linear_vel': float(vel)},
            f"desired_linear_vel {vel}")

    def _profile_max_linear_velocity(self) -> typing.Optional[float]:
        """The vx ceiling the ACTIVE profile just pushed, for the warning above.

        Read from the profile file rather than the live node: the profile push and this call
        happen in the same reset, and reading it back over a service would race that push.
        Returns None if unavailable -- a missing ceiling must not block the override.
        """
        try:
            by_node = self._load_social_attributes(self._active_social_attributes or 'neutral')
            node, prefix = self._PROFILE_TARGETS['weights_optimizer']
            return by_node.get(node, {}).get(prefix + 'max_linear_velocity')
        except Exception:
            return None

    async def _wait_for_odom_tf(self, timeout_s: float = 10.0) -> bool:
        """Wait until a *fresh* odom TF is available after the current reset.

        rclpy.time.Time() accepts any cached TF including stale ones from the
        previous episode. Instead we require the TF stamp to be newer than the
        sim time captured at the start of the wait, ensuring pose_to_tf has
        already published at least one transform reflecting the new robot pose.
        """
        odom_frame = self.frame(self._config.model_params.odom_frame)
        period_s = 0.1
        waited_s = 0.0
        # Capture sim time now — we want a TF stamped *after* this point.
        start_stamp = self.node.sim_time
        while waited_s < timeout_s:
            try:
                tf = self._tf_buffer.lookup_transform("map", odom_frame, rclpy.time.Time())
                tf_stamp = rclpy.time.Time.from_msg(tf.header.stamp)
                if tf_stamp >= start_stamp:
                    return True
            except Exception:
                pass
            await asyncio.sleep(period_s)
            waited_s += period_s
        return False

    async def _wait_for_sim_tick(self, timeout_s: float = 1.5) -> bool:
        """Wait until simulation time advances at least once.

        In Isaac reset flow the simulator is paused while positions are reassigned.
        Waiting for a sim tick avoids sending a new goal while nav2 still sees the
        previous odometry state.
        """
        start = self.node.sim_time.to_msg()
        start_stamp = (start.sec, start.nanosec)
        period_s = 0.05
        waited_s = 0.0

        while waited_s < timeout_s:
            await asyncio.sleep(period_s)
            now = self.node.sim_time.to_msg()
            if (now.sec, now.nanosec) != start_stamp:
                return True
            waited_s += period_s

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
        goal_waypoints: typing.Optional[typing.Sequence[Pose]] = None,
        waypoint_tolerance: typing.Optional[float] = None,
        final_goal_tolerance: typing.Optional[float] = None,
    ) -> tuple[Pose, Pose]:
        """Reset the robot's position and / or goal.

        Args:
            start_pos(typing.Optional[Pose]): The new starting position of the robot.
            goal_pos(typing.Optional[Pose]): The new goal position of the robot.
            goal_waypoints: Ordered goals. When supplied, each goal is published only after the
                previous one succeeds. The episode completes at the final waypoint.
            waypoint_tolerance: XY tolerance for every intermediate waypoint.
            final_goal_tolerance: XY tolerance for the final waypoint.

        Returns:
            tuple[Pose, Pose]: The new starting and goal positions of the robot.
        """
        if self._scene_gate_enabled():
            # Close synchronously in the episode reset path. Do not depend on a separate
            # unnumbered reset event: it could arrive after the numbered task_reset and close
            # this new episode's ready state again.
            self._scene_episode = None
            self._isaac_scene_ready_episode = None
            self._scene_motion_released = False
            if self._publish_goal_task is not None:
                self._publish_goal_task.cancel()
            self._scene_ready_pub.publish(std_msgs.msg.Int16(data=-1))
            self._local_costmap_readiness.reset()
            await self._cancel_navigation_goals()

        if start_pos is not None:
            self._start_pos = self._environment_manager.realize(start_pos)
            await self.move_robot_to_pos(start_pos)

            if self._robot.record_data_dir:
                self.node.rosparam[list[float]].set(
                    self.namespace.robot_ns.ParamNamespace()("start"),
                    [self.start_pos.position.x, self.start_pos.position.y, self.start_pos.orientation.to_yaw()]
                )

        await self._clear_local_costmap()
        await self._clear_global_costmap()
        route = list(goal_waypoints or ())
        if not route and goal_pos is not None:
            route = [goal_pos]

        if route:
            self._goal_waypoints = [
                self._environment_manager.realize(waypoint)
                for waypoint in route
            ]
            self._goal_waypoint_index = 0
            self._goal_pos = self._goal_waypoints[-1]
            self._waypoint_tolerance = (
                _DEFAULT_GOAL_TOLERANCE
                if waypoint_tolerance is None else float(waypoint_tolerance)
            )
            self._final_goal_tolerance = (
                _DEFAULT_GOAL_TOLERANCE
                if final_goal_tolerance is None else float(final_goal_tolerance)
            )
            self._is_goal_reached = False
            self._nav_stop_ticks = 0  # new goal incoming, stop publishing stop-zeros
            self._active_nav_goal_id = None
            self._ignored_nav_goal_ids = set(self._known_nav_goal_ids)
            self._goal_id_baseline = set(self._known_nav_goal_ids)
            self._goal_publish_stamp_ns = None
            self._social_yielding_active = False
            self._awaiting_original_goal = False
            self._resume_prepared_baseline = None

            if self._publish_goal_task is not None:
                self._publish_goal_task.cancel()

            start_target = self._start_pos if start_pos is not None else None
            self._publish_goal_task = asyncio.create_task(
                self._publish_goal_loop(
                    goal=self._goal_waypoints[0],
                    start_target=start_target,
                )
            )

            if self._robot.record_data_dir:
                self.node.rosparam[list[float]].set(
                    self.namespace.robot_ns.ParamNamespace()("goal"),
                    [self.goal_pos.position.x, self.goal_pos.position.y, self.goal_pos.orientation.to_yaw()]
                )
        return self._pose, self._goal_pos

    async def _cancel_navigation_goals(self) -> bool:
        """Cancel any NavigateToPose action left over from the previous episode."""
        active_before = {
            bytes(status.goal_info.goal_id.uuid)
            for status in self._latest_nav_statuses
            if status.status in _ACTIVE_NAV_STATUSES
        }
        if not active_before:
            return True

        srv_name = os.path.abspath(
            self.namespace('navigate_to_pose', '_action', 'cancel_goal'))
        client = self.node.create_client_wrapper(CancelGoal, srv_name)
        try:
            if not await client.ensure(timeout_sec=2.0):
                self._logger.warn(
                    f"{srv_name} unavailable; zero-velocity scene hold remains active")
                return False
            response = await client.call_timeout(CancelGoal.Request(), timeout_sec=2.0)
            if response is None:
                self._logger.warn(
                    f"{srv_name} did not answer; zero-velocity scene hold remains active")
                return False
            if response.return_code != CancelGoal.Response.ERROR_NONE:
                active_now = {
                    bytes(status.goal_info.goal_id.uuid)
                    for status in self._latest_nav_statuses
                    if status.status in _ACTIVE_NAV_STATUSES
                }
                if not active_now:
                    return True
                self._logger.warn(
                    f"{srv_name} rejected cancellation (code={response.return_code})")
                return False
            canceling = {
                bytes(info.goal_id.uuid) for info in response.goals_canceling
            }
            # Cancellation acknowledgement precedes terminal action status. Do not release
            # motion while a previous controller/recovery may still be winding down.
            for _ in range(100):
                active = {
                    bytes(status.goal_info.goal_id.uuid)
                    for status in self._latest_nav_statuses
                    if status.status in _ACTIVE_NAV_STATUSES
                }
                if not (active & canceling):
                    return True
                await asyncio.sleep(0.05)
            self._logger.warn("Previous navigation action is still canceling; keeping scene hold")
            return False
        finally:
            self.node.destroy_client(client.client)

    async def _publish_goal_loop(
        self,
        *,
        goal: Pose,
        start_target: typing.Optional[Pose] = None,
    ):
        """Publish the goal to the robot once reset state is synchronized."""
        if self._scene_gate_enabled():
            last_reason = None
            while (
                self._scene_episode is None
                or self._isaac_scene_ready_episode != self._scene_episode
                or not self._local_costmap_readiness.ready()
            ):
                reason = (
                    "waiting-for-task-reset"
                    if self._scene_episode is None
                    else "waiting-for-Isaac-scene"
                    if self._isaac_scene_ready_episode != self._scene_episode
                    else self._local_costmap_readiness.reason
                )
                if reason != last_reason:
                    self._logger.warn(
                        f"Holding robot and pedestrian goals: scene gate {reason}")
                    last_reason = reason
                await asyncio.sleep(0.05)
            self._logger.info(
                f"Scene ready for episode #{self._scene_episode}: Isaac + local costmap + "
                "sensor TFs valid; preparing synchronized goal release")

        if not await self._wait_for_sim_tick(timeout_s=5.0):
            self._logger.warn(
                "Simulation time did not advance before goal publish; nav may use stale pose."
            )

        # Wait for a fresh odom TF first — pose_sync depends on self._pose
        # which is driven by odom, so odom must be live before we check it.
        if not await self._wait_for_odom_tf(timeout_s=10.0):
            self._logger.warn(
                "odom TF frame not available before goal publish; nav2 may fail."
            )

        if start_target is not None and not await self._wait_for_pose_sync(start_target, timeout_s=10.0):
            self._logger.warn(
                "Odometry did not reach reset start pose before goal publish; proceeding anyway."
            )

        await self._publish_current_goal(goal)

    def _scene_episode_callback(self, msg: std_msgs.msg.Int16):
        self._scene_episode = int(msg.data)

    def _isaac_scene_ready_callback(self, msg: std_msgs.msg.Int16):
        episode = int(msg.data)
        self._isaac_scene_ready_episode = episode if episode >= 0 else None

    def _local_costmap_sensor_frames(self) -> tuple[str, ...]:
        params = self._config.model_params
        frames = [str(self.frame(params.base_frame))]
        source_names = params.get('observation_sources', ())
        source_configs = params.get('observation_sources_dict', {})
        for source_name in source_names:
            sensor_frame = source_configs.get(source_name, {}).get('sensor_frame')
            if not sensor_frame:
                continue
            child = str(sensor_frame).replace('${frame}', '').strip('/')
            frames.append(str(self.frame(child)))
        return tuple(dict.fromkeys(frames))

    def _lookup_frame_xy(
        self, target_frame: str, source_frame: str
    ) -> typing.Optional[tuple[float, float]]:
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            translation = transform.transform.translation
            return float(translation.x), float(translation.y)
        except Exception:
            return None

    def _scene_gate_enabled(self) -> bool:
        try:
            return self.node.conf.Arena.SIM.value == Constants.SimSimulator.ISAAC
        except AttributeError:
            # Lightweight unit-test managers and non-Arena embeddings do not have the
            # runtime configuration tree; scene gating is an Isaac integration feature.
            return False

    def _current_goal_tolerance(self) -> float:
        if self._goal_waypoint_index == len(self._goal_waypoints) - 1:
            return self._final_goal_tolerance
        return self._waypoint_tolerance

    async def _set_goal_tolerance(self, tolerance: float) -> bool:
        node_name = self.node.service_namespace(self.name, 'controller_server')
        srv_name = os.path.abspath(node_name('set_parameters'))
        cli = self.node.create_client_wrapper(SetParameters, srv_name)
        try:
            if not await cli.ensure(timeout_sec=3.0):
                self._logger.error(
                    f"{srv_name} unavailable; cannot apply goal tolerance {tolerance}")
                return False
            request = SetParameters.Request(parameters=[
                ParameterMsg(
                    name='goal_checker.xy_goal_tolerance',
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_DOUBLE,
                        double_value=tolerance,
                    ),
                )
            ])
            for attempt in range(1, 4):
                result = await cli.call_timeout(request, timeout_sec=3.0)
                if result is not None and result.results and result.results[0].successful:
                    return True
                reason = (
                    result.results[0].reason
                    if result is not None and result.results else 'no response'
                )
                self._logger.warn(
                    f"failed to apply goal tolerance {tolerance} on {srv_name} "
                    f"(attempt {attempt}/3): {reason}")
                if attempt < 3:
                    await asyncio.sleep(0.25)
            return False
        finally:
            self.node.destroy_client(cli.client)

    async def _publish_current_goal(self, goal: Pose) -> bool:
        waypoint_number = self._goal_waypoint_index + 1
        waypoint_count = len(self._goal_waypoints)
        tolerance = self._current_goal_tolerance()
        release_scene = self._scene_gate_enabled() and not self._scene_motion_released
        if not await self._set_goal_tolerance(tolerance):
            self._logger.error(
                f"Not publishing waypoint {waypoint_number}/{waypoint_count}: "
                f"could not apply xy tolerance {tolerance} m")
            self._is_goal_reached = False
            self._route_status_pub.publish(std_msgs.msg.Int8(
                data=action_msgs.msg.GoalStatus.STATUS_ABORTED))
            if self._cmd_vel_pub is not None:
                self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())
            self._nav_stop_ticks = 15
            return False

        if release_scene:
            # Keep the final cmd_vel hold active through TF/pose synchronization and the
            # parameter service call. Release it only at the publication boundary of this
            # episode's first original goal, after one last best-effort cancellation of any
            # previous action.
            while not await self._cancel_navigation_goals():
                await asyncio.sleep(0.25)
            if self._cmd_vel_pub is not None:
                self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())
            # The original goal is now allowed into Nav2 for planning. Physical motion
            # remains blocked by motion_hold_gate until the yielding goal Y is ready.
            self._scene_motion_released = True

        self._logger.info(
            f"Publishing robot waypoint {waypoint_number}/{waypoint_count}: "
            f"x={goal.position.x}, y={goal.position.y}, "
            f"orientation={goal.orientation.to_yaw()}, xy_tolerance={tolerance}"
        )

        if self._goal_timer is not None:
            self._goal_timer.cancel()
            self._goal_timer.destroy()

        goal_msg = geometry_msgs.msg.PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.node.sim_time.to_msg()
        goal_msg.pose = goal.to_msg()
        self._goal_id_baseline = set(self._known_nav_goal_ids)
        self._goal_pub.publish(goal_msg)
        if release_scene and self._scene_episode is not None:
            self._scene_ready_pub.publish(
                std_msgs.msg.Int16(data=self._scene_episode))
            self._logger.info(
                f"Released robot and pedestrian goals for episode #{self._scene_episode}")

        self._goal_start_time = self.node.sim_time
        self._goal_publish_stamp_ns = (
            goal_msg.header.stamp.sec * 1_000_000_000
            + goal_msg.header.stamp.nanosec
        )
        return True

    async def _launch_robot(self, node_paths: set[str]):
        """Launch the robot external nodes.
        """
        self._logger.info(f"LAUNCH ROBOT {self.name}")

        if Utils.get_arena_type() != Constants.ArenaType.TRAINING:
            launch_description = launch.LaunchDescription()
            current_log_level = rclpy.logging.get_logger_effective_level(self.node.get_logger().name).name.lower()
            launch_description.add_action(NodeLogLevelExtension.SetGlobalLogLevelAction(current_log_level))  # type: ignore

            launch_arguments = {
                'robot': self.model_name,
                # 'simulator': self.node.conf.Arena.SIM.value.value,
                # 'name': self.name,
                'task_generator_node': os.path.join(self.node.get_namespace(), self.node.get_name()),
                'namespace': self.namespace,
                # 'use_namespace': 'True',
                'frame': self._robot.frame(''),  # trailing slash
                'inter_planner': self._robot.inter_planner,
                'global_planner': self._robot.global_planner,
                'local_planner': self._robot.local_planner,
                # 'complexity': self.node.declare_parameter('complexity', 1).value,
                'train_mode': str(self.node._train_mode).lower(),
                'agent_name': self._robot.agent,
                'use_sim_time': 'True',
                'amcl': 'true' if self.node.conf.Arena.SIM.value in (Constants.SimSimulator.GAZEBO,) else 'false',
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

        self._pose = Pose(
            Position(
                current_position.position.x,
                current_position.position.y,
            ),
            Orientation.from_msg(quat)
        )

    def _stop_vel_timer_cb(self):
        """Publish zero velocity when the robot should be stopped."""
        scene_hold = (
            self._scene_gate_enabled()
            and not self._scene_motion_released
        )
        if (scene_hold or self._nav_stop_ticks > 0) and self._cmd_vel_pub is not None:
            self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())
            if not scene_hold:
                self._nav_stop_ticks -= 1

    def _social_yielding_active_callback(self, msg: std_msgs.msg.Bool):
        active = bool(msg.data)
        if active and not self._social_yielding_active:
            # The temporary yielding goal will preempt the scenario goal. Ignore both that
            # cancellation and every terminal status produced while the orchestrator owns
            # navigation.
            self._active_nav_goal_id = None
            self._ignored_nav_goal_ids.update(self._known_nav_goal_ids)
            self._awaiting_original_goal = True
            self._goal_publish_stamp_ns = None
            self._goal_id_baseline = set(self._known_nav_goal_ids)
        self._social_yielding_active = active

    def _prepare_original_goal_resume_callback(self, _request, response):
        """Create an explicit UUID boundary before the orchestrator republishes G."""
        self._active_nav_goal_id = None
        self._ignored_nav_goal_ids.update(self._known_nav_goal_ids)
        self._resume_prepared_baseline = set(self._known_nav_goal_ids)
        self._goal_id_baseline = set(self._known_nav_goal_ids)
        self._awaiting_original_goal = True
        response.success = True
        response.message = "ready for restored original goal"
        self._logger.info(
            "Prepared original-goal resume UUID boundary "
            f"({len(self._resume_prepared_baseline)} known actions)")
        return response

    def _goal_command_callback(self, msg: geometry_msgs.msg.PoseStamped):
        if not self._goal_waypoints:
            return
        matching_waypoint = next(
            (
                index
                for index, waypoint in enumerate(self._goal_waypoints)
                if (
                    abs(msg.pose.position.x - waypoint.position.x)
                    <= _GOAL_MATCH_TOLERANCE
                    and abs(msg.pose.position.y - waypoint.position.y)
                    <= _GOAL_MATCH_TOLERANCE
                )
            ),
            None,
        )
        if matching_waypoint is None:
            # Any goal outside the scenario route is owned by an external controller such as
            # social yielding. Its action may abort before the latched /active=True callback is
            # delivered, so classify it directly from the command topic and suppress all of its
            # terminal statuses. The original scenario goal callback will rebind the route later.
            self._active_nav_goal_id = None
            self._ignored_nav_goal_ids.update(self._known_nav_goal_ids)
            self._goal_id_baseline = set(self._known_nav_goal_ids)
            self._goal_publish_stamp_ns = None
            self._awaiting_original_goal = True
            self._logger.info(
                "External temporary goal observed; route completion suspended until a "
                "scenario goal is restored")
            return

        # Intermediate scenario waypoints are managed by _goal_status_callback. Only the final
        # waypoint can be an orchestrator hand-back that needs route/action rebinding here.
        if matching_waypoint != len(self._goal_waypoints) - 1:
            return

        # This is either the normal final waypoint or the orchestrator restoring original G.
        # Rebind to the next NavigateToPose action created from this publication. When yielding
        # interrupted an intermediate waypoint, resuming G intentionally makes G the final route
        # target; only its SUCCESS may complete/save the episode.
        self._goal_waypoint_index = len(self._goal_waypoints) - 1
        self._active_nav_goal_id = None
        # Bind by UUID generation, not timestamps: this goal may be stamped by the orchestrator's
        # wall clock while Nav2 GoalInfo uses simulation time. The prepare service normally gives
        # us an exact pre-publication boundary. It also lets us recover G when its action status
        # beats this goal-topic callback through the ROS graph.
        prepared_baseline = self._resume_prepared_baseline
        self._ignored_nav_goal_ids.update(self._known_nav_goal_ids)
        resumed_action = None
        if prepared_baseline is not None:
            resumed_action = next(
                (
                    status
                    for status in reversed(self._latest_nav_statuses)
                    if bytes(status.goal_info.goal_id.uuid) not in prepared_baseline
                ),
                None,
            )
            self._goal_id_baseline = set(prepared_baseline)
        else:
            # Compatibility fallback if an older orchestrator cannot call the prepare service.
            # Everything already visible is old; wait for the next UUID after this callback.
            self._goal_id_baseline = set(self._known_nav_goal_ids)
        self._resume_prepared_baseline = None
        if resumed_action is not None:
            resumed_id = bytes(resumed_action.goal_info.goal_id.uuid)
            self._active_nav_goal_id = resumed_id
            self._ignored_nav_goal_ids.discard(resumed_id)
        self._goal_publish_stamp_ns = (
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )
        self._awaiting_original_goal = False
        self._is_goal_reached = False
        if self._active_nav_goal_id is None:
            self._logger.info(
                "Original scenario goal G restored; waiting for its new Nav2 action UUID")
        else:
            self._logger.info(
                "Original scenario goal G restored; associated early Nav2 action UUID "
                f"{self._active_nav_goal_id.hex()[:8]}")

    async def _goal_status_callback(self, data: action_msgs.msg.GoalStatusArray):
        statuses = list(data.status_list)
        self._latest_nav_statuses = statuses
        for goal_status in statuses:
            self._known_nav_goal_ids.add(bytes(goal_status.goal_info.goal_id.uuid))

        if self._scene_gate_enabled() and not self._scene_motion_released:
            self._ignored_nav_goal_ids.update(self._known_nav_goal_ids)
            return

        if self._awaiting_original_goal or (
            self._social_yielding_active and self._goal_publish_stamp_ns is None
        ):
            # Record IDs so none can be mistaken for the resumed scenario goal later, but do
            # not advance the route and do not publish route_status while yielding owns Nav2.
            self._ignored_nav_goal_ids.update(self._known_nav_goal_ids)
            return

        current = None
        if self._active_nav_goal_id is not None:
            current = next(
                (
                    goal_status
                    for goal_status in reversed(statuses)
                    if bytes(goal_status.goal_info.goal_id.uuid) == self._active_nav_goal_id
                ),
                None,
            )
        else:
            current = next(
                (
                    goal_status
                    for goal_status in reversed(statuses)
                    if (
                        bytes(goal_status.goal_info.goal_id.uuid)
                        not in self._goal_id_baseline
                    )
                ),
                None,
            )
            if current is not None:
                self._active_nav_goal_id = bytes(current.goal_info.goal_id.uuid)
                self._logger.info(
                    "Associated route goal with Nav2 action UUID "
                    f"{self._active_nav_goal_id.hex()[:8]}")

        if current is None:
            return

        status = current.status
        if status not in _TERMINAL_NAV_STATUSES:
            self._is_goal_reached = False
            return

        goal_id = bytes(current.goal_info.goal_id.uuid)
        if goal_id in self._ignored_nav_goal_ids:
            return
        self._ignored_nav_goal_ids.add(goal_id)
        self._active_nav_goal_id = None

        if status == action_msgs.msg.GoalStatus.STATUS_SUCCEEDED:
            next_index = self._goal_waypoint_index + 1
            if next_index < len(self._goal_waypoints):
                self._goal_waypoint_index = next_index
                self._is_goal_reached = False
                self._nav_stop_ticks = 0
                if self._cmd_vel_pub is not None:
                    self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())
                await self._publish_current_goal(self._goal_waypoints[next_index])
                return

            self._is_goal_reached = True
            self._logger.info(
                "Scenario final goal reached; publishing route_status=SUCCEEDED")
            self._route_status_pub.publish(std_msgs.msg.Int8(data=status))
            if self._cmd_vel_pub is not None:
                self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())
            self._nav_stop_ticks = 15  # 1.5 s at 10 Hz
            return

        self._is_goal_reached = False
        self._logger.warn(
            f"Scenario route ended with Nav2 status {status}; publishing route_status")
        self._route_status_pub.publish(std_msgs.msg.Int8(data=status))
        if self._cmd_vel_pub is not None:
            self._cmd_vel_pub.publish(geometry_msgs.msg.Twist())
        self._nav_stop_ticks = 15  # 1.5 s at 10 Hz

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
        if self._stop_vel_timer is not None:
            self._stop_vel_timer.cancel()
            self._stop_vel_timer.destroy()
            self._stop_vel_timer = None
        await self._environment_manager.remove_robot((self.robot,))
        # TODO kill node in navigation stack
