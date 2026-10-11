import asyncio
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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import tf2_ros
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Robot import RobotView
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearCostmapAroundRobot, ClearEntireCostmap
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType

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
        # Name of the profile applied by the last set_social_attributes call. Only used to
        # look up that profile's vx ceiling when warning about a too-large scenario override.
        self._active_social_attributes: typing.Optional[str] = None

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
        self._model_declared_stop = False
        self._model_declared_stop_payload = {}
        self._waypoint_lifecycle = None

        self._publish_goal_task: typing.Optional[asyncio.Task] = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
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

        # The velocity smoother is the only final publisher in FollowPath mode.
        if not self._uses_waypoints():
            self._cmd_vel_pub = self.node.create_publisher(
                geometry_msgs.msg.Twist,
                self.namespace('cmd_vel'),
                1,
            )

            self._stop_vel_timer = self.node.create_timer(0.1, self._stop_vel_timer_cb)

        self.node.create_subscription(
            nav_msgs.Odometry,
            self.namespace("odom"),
            self._robot_pos_callback,
            10
        )

        if not self._uses_waypoints():
            self.node.create_subscription(
                action_msgs.msg.GoalStatusArray,
                self.namespace('navigate_to_pose', '_action', 'status'),
                self._goal_status_callback,
                1
            )

        self._setup_camera_readiness_subscriptions()
        if self._uses_waypoints():
            from arena_vln_models.lifecycle import WaypointLifecycle
            self._waypoint_lifecycle = WaypointLifecycle(
                self.node, self.namespace,
                heartbeat_timeout=self.node.rosparam[float].get('model_heartbeat_timeout_sec', 2.5),
                record_dir=self._robot.record_data_dir or '',
            )
        await self._launch_robot(node_names)
        if self._waypoint_lifecycle:
            await self._waypoint_lifecycle.begin(self.node._vln_session_id, self.node._number_of_resets)


        if self._uses_waypoints():
            self._navigate_to_pose_client = None
        else:
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
        if self._waypoint_lifecycle and self._waypoint_lifecycle.session.released:
            try:
                if self._waypoint_lifecycle.session.terminal:
                    self.node._publish_eval_ready('episode', False, reason='model_terminal')
                    self._waypoint_lifecycle.session.released = False
                reason = await self._waypoint_lifecycle.poll()
            except Exception as exc:
                self._waypoint_lifecycle.error = str(exc)
                raise
            if reason in {'model_stop', 'model_error'}:
                self._model_declared_stop = True
                self._model_declared_stop_payload = {'reason': reason, 'detail': self._waypoint_lifecycle.session.reason}
        return self._is_goal_reached or self._model_declared_stop


    @property
    def done_reason(self) -> str:
        if self._model_declared_stop:
            return self._model_declared_stop_payload.get('reason', 'model_stop')
        return 'goal_reached' if self._is_goal_reached else 'running'


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
        if self._uses_waypoints():
            # The tracking costmap has no obstacle layers and is reset by lifecycle.
            return True

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
        if self._uses_waypoints():
            return True
        node_name = self.node.service_namespace(self.name, 'global_costmap/global_costmap')
        srv_name = os.path.abspath(node_name('../clear_entirely_global_costmap'))
        return await self._clear_costmap(node_name, srv_name)

    # Map a profile.yaml section -> (target node, param-name prefix).
    #
    # The knobs live on THREE different nodes, so a profile push is fanned out per node:
    #   - controller_server: the MPC. It is wrapped by RotationShim (plugin "FollowPath"), which
    #     configures the primary controller with the SAME name, so social_mpc params sit directly
    #     under FollowPath.* (verified). Hot-reloaded via the MPC's dynamic-parameter callback.
    #   - global_costmap: the SocialLayer pedestrian bubbles. No callback needed — the layer
    #     re-reads every parameter on each updateCosts() (social_layer.cpp get_parameters()), so a
    #     push lands on the next costmap cycle. Only the GLOBAL costmap is targeted: the local one
    #     no longer loads social_layer (see nav2.yaml), because obstacle_layer+inflation already
    #     cover peds there and the MPC's own /people critics do the real local avoidance.
    _PROFILE_TARGETS: dict[str, tuple[str, str]] = {
        'weights': ('controller_server', 'FollowPath.optimizer.weights.'),
        # Optimizer knobs that live directly under `optimizer.` rather than `optimizer.weights.`
        # (currently just max_linear_velocity, the hard vx ceiling + VelocityCost target). Kept as
        # its own section because the MPC's dynamic-param callback matches these two prefixes
        # separately, and the weights.* branch would not recognize this key.
        'weights_optimizer': ('controller_server', 'FollowPath.optimizer.'),
        'trajectorizer': ('controller_server', 'FollowPath.trajectorizer.'),
        'global_social_layer': ('global_costmap/global_costmap', 'social_layer.'),
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
        """Push a social-attributes preset to this robot's MPC and social costmap layers at reset.

        Fans out over the nodes in _PROFILE_TARGETS. The MPC hot-reloads its half via a
        dynamic-parameter callback and SocialLayer re-reads its half every costmap cycle, so the
        preset takes effect immediately without a relaunch.
        No-op when attributes is falsy (caller resolves the default profile).
        """
        if not attributes or self._uses_waypoints():
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
        if vel is None or self._uses_waypoints():
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

        await self._clear_local_costmap()
        await self._clear_global_costmap()
        if goal_pos is not None:
            self._goal_pos = self._environment_manager.realize(goal_pos)
            self._is_goal_reached = False
            self._model_declared_stop = False
            self._model_declared_stop_payload = {}
            self._nav_stop_ticks = 0  # new goal incoming, stop publishing stop-zeros
            self._active_navigation_goal_uuid = None
            self._goal_metadata_pub.publish(self._pose_stamped(self._goal_pos))

            await self._cancel_navigation_goal()

            if self._publish_goal_task is not None:
                self._publish_goal_task.cancel()
                try:
                    await self._publish_goal_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    self._logger.warn(f'Previous navigation goal readiness task ended while cancelling reset: {exc}')

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

        sim_tick_timeout_s = float(os.environ.get('ARENA_ISAAC_SIM_TICK_TIMEOUT_SEC', '600.0'))
        if not await self._wait_for_sim_tick(timeout_s=sim_tick_timeout_s):
            raise RuntimeError('Simulation time did not advance before goal publish.')

        if not await self._wait_for_odom_tf(timeout_s=10.0):
            self._logger.warn("odom TF frame not available before goal publish; nav may fail.")

        if start_target is not None and not await self._wait_for_pose_sync(start_target, timeout_s=120.0):
            if str(os.environ.get('ARENA_ISAAC_ALLOW_LIVE_RESET_POSE_DESYNC', '1')).strip().lower() in {
                '1', 'true', 'yes', 'on'
            }:
                self._logger.warning(
                    'Odometry did not reach reset start pose before goal publish; '
                    'continuing with real current odometry because Isaac GRScenes can drop live reset service callbacks.'
                )
            else:
                raise RuntimeError('Odometry did not reach reset start pose before goal publish.')

        self._reset_navigation_readiness_state()
        camera_ready_timeout_s = float(
            self.node.rosparam[float].get(
                'camera_ready_before_navigation_timeout_sec',
                float(os.environ.get('ARENA_ISAAC_CAMERA_READY_BEFORE_NAVIGATION_TIMEOUT_SEC', '900.0')),
            )
        )
        if not await self._wait_for_camera_ready_before_navigation(timeout_s=camera_ready_timeout_s):
            raise RuntimeError('Timed out waiting for real camera topics before publishing VLN navigation goal.')
        if self._waypoint_lifecycle:
            await self._waypoint_lifecycle.activate()
            self._goal_start_time = self.node.sim_time
            return

        self._logger.info(
            f"Publishing goal once: x={goal.position.x}, y={goal.position.y}, orientation={goal.orientation.to_yaw()}"
        )

        self._destroy_goal_republish_timer()

        goal_msg = self._pose_stamped(goal)
        self._goal_pub.publish(goal_msg)
        self._last_goal_msg = goal_msg
        self._goal_republish_ticks = 10

        if self._goal_timer is None:
            self._goal_timer = self.node.create_timer(
                0.2,
                self._republish_goal_timer_cb,
            )

        await self._send_navigation_goal(goal_msg)
        self._goal_start_time = self.node.sim_time

    def _destroy_goal_republish_timer(self) -> None:
        timer = self._goal_timer
        if timer is None:
            return

        # Clear the reference first so a racing reset/cancel path cannot try to
        # reuse the same rclpy handle after node shutdown has started.  In heavy
        # Isaac/GRScenes evals the launch wrapper can request destruction while
        # the async goal-readiness task is still unwinding; rclpy then raises
        # InvalidHandle from Timer.cancel()/destroy().  Treat that as already
        # destroyed rather than letting episode rollover wedge before publishing
        # the next task_reset/eval_ready sample.
        self._goal_timer = None
        for action_name, action in (('cancel', timer.cancel), ('destroy', timer.destroy)):
            try:
                action()
            except Exception as exc:
                self._logger.warn(f'Ignoring goal republish timer {action_name} failure during reset: {exc}')

    def _default_camera_topics_for_readiness(self) -> dict[str, str] | None:
        if not self._uses_waypoints():
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

    def _uses_waypoints(self) -> bool:
        return self.node.rosparam[str].get('execution_mode', '') == 'nav2_follow_path'


    def _configured_camera_topics_for_readiness(self):
        if not self._uses_waypoints():
            return None
        defaults = self._default_camera_topics_for_readiness() or {}
        # Arena records RGB; model repositories decide whether depth is needed.
        configured = self.node.rosparam[str].get('vln_rgb_topic', '') or defaults.get('rgb', '')
        return {'rgb': configured if configured.startswith('/') else str(self.namespace(configured))}


    def _reset_navigation_readiness_state(self):
        self._camera_ready_seen = {key: False for key in self._camera_ready_seen}
        self._camera_ready_seen_wall_time = {key: 0.0 for key in self._camera_ready_seen}
        self._model_declared_stop = False
        self._model_declared_stop_payload = {}


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
        if 'depth' in self._camera_ready_topics:
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



    def _mark_camera_ready(self, key: str) -> None:
        if key in self._camera_ready_seen:
            self._camera_ready_seen[key] = True
            self._camera_ready_seen_wall_time[key] = time.monotonic()






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


    async def wait_for_pending_goal(self, timeout_s: float) -> bool:
        task = self._publish_goal_task
        if task is None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
            return True
        except asyncio.TimeoutError as exc:
            raise RuntimeError('Robot sensors/controller were not ready before episode deadline') from exc

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
        if self._uses_waypoints():
            return
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
        if Utils.get_arena_type() == Constants.ArenaType.TRAINING:
            return
        ld = launch.LaunchDescription()
        level = rclpy.logging.get_logger_effective_level(self.node.get_logger().name).name.lower()
        ld.add_action(SetGlobalLogLevelAction(level))
        waypoint_mode = self._uses_waypoints()
        package = 'arena_vln_models' if waypoint_mode else 'arena_simulation_setup'
        filename = ('follow_path.launch.py' if waypoint_mode else
                    os.path.basename(self.node.rosparam[str].get('robot_launch_file', 'robot.launch.py')))
        arguments = {
            'robot': self.model_name,
            'task_generator_node': os.path.join(self.node.get_namespace(), self.node.get_name()),
            'namespace': str(self.namespace), 'frame': f'{self._robot.frame}/',
            'inter_planner': str(self._robot.inter_planner),
            'global_planner': str(self._robot.global_planner),
            'local_planner': 'regulated_pure_pursuit' if waypoint_mode else str(self._robot.local_planner),
            'train_mode': str(self.node._train_mode).lower(),
            'agent_name': str(self._robot.agent), 'use_sim_time': 'true',
            'amcl': 'true' if self.node.conf.Arena.SIM.value == Constants.SimSimulator.GAZEBO else 'false',
            'enable_collision_monitor': str(self.node.rosparam[bool].get('enable_collision_monitor', True)).lower(),
            'record_data_dir': str(self._robot.record_data_dir or ''),
        }
        ld.add_action(launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(os.path.join(
                ament_index_python.packages.get_package_share_directory(package), 'launch', filename)),
            launch_arguments=arguments.items()))
        await self.node.do_launch(ld)
        if not waypoint_mode:
            bt_path = str(self.namespace('bt_navigator'))
            while bt_path not in node_paths:
                await asyncio.sleep(.01)


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
            # Nav2 may report SUCCEEDED
            # while the physical robot is still far from the goal.
            # Only tear down the command bridge when the robot has
            # actually reached the goal or the navigation was aborted
            # / cancelled.
            if status != action_msgs.msg.GoalStatus.STATUS_SUCCEEDED or reached:
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
        self._destroy_goal_republish_timer()
        if self._waypoint_lifecycle:
            await self._waypoint_lifecycle.stop()
            self._waypoint_lifecycle.destroy()
        await self._cancel_navigation_goal()
        if self._stop_vel_timer is not None:
            self._stop_vel_timer.cancel()
            self._stop_vel_timer.destroy()
            self._stop_vel_timer = None
        await self._environment_manager.remove_robot((self.robot,))

        # TODO kill node in navigation stack
