import json
import math
import os
import sys
import threading
import time
from typing import Optional


def _maybe_reexec_with_arena_python() -> None:
    target = os.environ.get('ARENA_PYTHON', '').strip()
    if not target:
        return

    current = os.path.realpath(sys.executable)
    desired = os.path.realpath(target)
    if current == desired:
        return

    if os.environ.get('ARENA_PYTHON_REEXEC') == '1':
        raise RuntimeError(
            f'ARENA_PYTHON re-exec requested {target}, but process is still running as {sys.executable}'
        )

    env = os.environ.copy()
    env['ARENA_PYTHON_REEXEC'] = '1'
    os.execve(target, [target, *sys.argv], env)


_maybe_reexec_with_arena_python()

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from rosnav_rl_msgs.srv import GetCommand
from std_msgs.msg import String

from arena_vln_models.backends import ModelSimDecision, ModelSimObservation, Pose2D, create_model_backend
from arena_vln_models.visualization import image_msg_to_numpy, numpy_to_image_msg, render_debug_overlay

def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    # yaw (Z) from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _finite_values(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


class BaseModelSimServer(Node):
    """Base ROS GetCommand wrapper for model-simulator integrations."""

    NODE_NAME = 'model_sim_server'
    SERVER_LABEL = 'model_sim'
    MODEL_INSTANCE = 'generic'
    DEFAULT_VISUALIZATION_TOPIC = 'model_sim/debug_image'
    DEFAULT_STATUS_TOPIC = 'model_sim/status'
    COMPUTE_THREAD_NAME = 'model_sim_compute_worker'

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # Topics (relative to this node namespace)
        self.declare_parameter('pose_topic', 'pose')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('goal_topic', 'goal_pose')
        self.declare_parameter('subgoal_topic', 'subgoal')
        self.declare_parameter('instruction_topic', 'vln_instruction')
        self.declare_parameter('rgb_topic', '')
        self.declare_parameter('depth_topic', '')
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('visualization_topic', self.DEFAULT_VISUALIZATION_TOPIC)
        self.declare_parameter('visualization_rate_hz', 5.0)
        self.declare_parameter('status_topic', self.DEFAULT_STATUS_TOPIC)

        # Control params
        # NOTE: "command_timeout_sec" historically acted as a compute throttle.
        # Keep it for compatibility, but prefer inference_rate_hz for model.
        self.declare_parameter('mode', 'heuristic')  # heuristic | model
        self.declare_parameter('command_timeout_sec', 0.2)
        self.declare_parameter('inference_rate_hz', 10.0)
        self.declare_parameter('inference_timeout_sec', 0.2)
        self.declare_parameter('camera_ready_timeout_sec', 120.0)
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('adapter_target', '')
        self.declare_parameter('require_real_backend', False)
        self.declare_parameter('strict_device', False)
        self.declare_parameter('look_down', False)
        self.declare_parameter('max_linear', 0.6)
        self.declare_parameter('max_angular', 1.5)
        self.declare_parameter('k_lin', 1.2)
        self.declare_parameter('k_ang', 2.0)
        self.declare_parameter('goal_tolerance', 0.45)
        self.declare_parameter('angle_tolerance', 0.25)
        self.declare_parameter('min_lin_when_aligned', 0.05)

        self._pose: Optional[Pose2D] = None
        self._last_odom_pose_ts: float = 0.0
        self._goal: Optional[Pose2D] = None
        self._subgoal: Optional[Pose2D] = None
        self._instruction: str = 'navigate'
        self._last_compute_ts: float = 0.0
        self._last_visualization_ts: float = 0.0
        self._last_cmd: Twist = Twist()
        self._last_decision: ModelSimDecision = ModelSimDecision(status='startup', degraded=True)
        self._state_lock = threading.Lock()
        self._compute_in_progress = False
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_rgb_msg: Optional[Image] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_depth_msg: Optional[Image] = None
        self._camera_intrinsics: Optional[tuple[float, ...]] = None

        self._params = {
            'command_timeout_sec': float(self.get_parameter('command_timeout_sec').value),
            'inference_rate_hz': float(self.get_parameter('inference_rate_hz').value),
            'inference_timeout_sec': float(self.get_parameter('inference_timeout_sec').value),
            'camera_ready_timeout_sec': float(self.get_parameter('camera_ready_timeout_sec').value),
            'model_path': str(self.get_parameter('model_path').value),
            'device': str(self.get_parameter('device').value),
            'adapter_target': str(self.get_parameter('adapter_target').value),
            'require_real_backend': bool(self.get_parameter('require_real_backend').value),
            'strict_device': bool(self.get_parameter('strict_device').value),
            'look_down': bool(self.get_parameter('look_down').value),
            'max_linear': float(self.get_parameter('max_linear').value),
            'max_angular': float(self.get_parameter('max_angular').value),
            'k_lin': float(self.get_parameter('k_lin').value),
            'k_ang': float(self.get_parameter('k_ang').value),
            'goal_tolerance': float(self.get_parameter('goal_tolerance').value),
            'angle_tolerance': float(self.get_parameter('angle_tolerance').value),
            'min_lin_when_aligned': float(self.get_parameter('min_lin_when_aligned').value),
        }

        # Latching instruction subscriber (matches publisher durability)
        instr_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(PoseStamped, self.get_parameter('pose_topic').value, self._on_pose, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self._on_odom, 10)
        self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self._on_goal, 10)
        self.create_subscription(PoseStamped, self.get_parameter('subgoal_topic').value, self._on_subgoal, 10)
        self.create_subscription(String, self.get_parameter('instruction_topic').value, self._on_instruction, instr_qos)

        rgb_topic = str(self.get_parameter('rgb_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        # Isaac Sim camera writers publish sensor streams with BEST_EFFORT QoS.
        # A default RELIABLE subscription is incompatible with those publishers and
        # leaves InternNav stuck in the initial waiting_for_camera barrier even
        # while other BEST_EFFORT consumers (for example the eval video recorder)
        # receive frames on the same topics.
        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE
        if rgb_topic:
            self.create_subscription(Image, rgb_topic, self._on_rgb, sensor_qos)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self._on_depth, sensor_qos)
        if camera_info_topic:
            self.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, sensor_qos)

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            status_qos,
        )

        self._visualization_enabled = bool(self.get_parameter('enable_visualization').value)
        self._visualization_publisher = None
        if self._visualization_enabled:
            self._visualization_publisher = self.create_publisher(
                Image,
                str(self.get_parameter('visualization_topic').value),
                10,
            )
            if not rgb_topic:
                self.get_logger().warn(
                    f'{self.SERVER_LABEL} visualization enabled but rgb_topic is empty; debug image publishing will stay idle'
                )

        mode = str(self.get_parameter('mode').value)
        self._wait_for_initial_camera_if_required(
            mode=mode,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
            timeout_sec=self._params['camera_ready_timeout_sec'],
        )
        try:
            self._backend = create_model_backend(mode=mode, logger=self.get_logger(), params=self._params)
        except Exception as exc:
            if self._params['require_real_backend']:
                raise RuntimeError(
                    f"Failed to create required real backend for mode='{mode}': {exc}"
                ) from exc
            self.get_logger().error(f"Failed to create backend for mode='{mode}': {exc}; falling back to heuristic")
            self._backend = create_model_backend(mode='heuristic', logger=self.get_logger(), params=self._params)
        self._fallback_backend = create_model_backend(mode='heuristic', logger=self.get_logger(), params=self._params)

        self.create_service(GetCommand, 'get_command', self._on_get_command)

        self.get_logger().info(
            (
                f'{self.SERVER_LABEL}_server started '
                f'(mode={mode}, backend={self._backend.backend_type}, '
                f'model_path={self._params["model_path"]}, device={self._params["device"]}, '
                f'adapter_target={self._params["adapter_target"] or "<disabled>"}, '
                f'require_real_backend={self._params["require_real_backend"]}, '
                f'strict_device={self._params["strict_device"]}, '
                f'rate_hz={self._params["inference_rate_hz"]:.3f}, '
                f'timeout={self._params["inference_timeout_sec"]:.3f}s) '
                f'pose={self.get_parameter("pose_topic").value} '
                f'odom={self.get_parameter("odom_topic").value} '
                f'goal={self.get_parameter("goal_topic").value} '
                f'subgoal={self.get_parameter("subgoal_topic").value} '
                f'instruction={self.get_parameter("instruction_topic").value} '
                f'rgb={rgb_topic or "<disabled>"} depth={depth_topic or "<disabled>"} '
                f'camera_info={camera_info_topic or "<disabled>"} look_down={self._params["look_down"]} '
                f'visualization={self._visualization_enabled}'
            )
        )
        self.get_logger().info(f'{self.SERVER_LABEL} backend ready: {self._backend.describe()}')
        adapter_available = getattr(self._backend, '_adapter_callable', True) is not None
        self._last_decision = ModelSimDecision(
            status='backend_ready' if adapter_available else 'backend_unavailable',
            degraded=not adapter_available,
            debug={
                'backend_type': self._backend.backend_type,
                'backend_description': self._backend.describe(),
                'uses_model_inference': bool(self._backend.uses_model_inference),
                'model_path': self._params['model_path'],
                'device': self._params['device'],
                'adapter_target': self._params['adapter_target'],
                'require_real_backend': self._params['require_real_backend'],
                'strict_device': self._params['strict_device'],
            },
        )
        self._publish_status(self._last_decision)
        self.get_logger().info(
            f'{self.SERVER_LABEL} wrapper active; current model instance={self.MODEL_INSTANCE}. '
            'InternNav inference notebook expects checkpoint clone from '
            'https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN via git-lfs and '
            'uses agent.step(rgb, depth, camera_pose, instruction, intrinsic=..., look_down=...) '
            'before adaptation to ROS cmd_vel output'
        )

    def _requires_initial_camera(self, *, mode: str, rgb_topic: str, depth_topic: str, camera_info_topic: str) -> bool:
        return False

    def _camera_wait_label(self) -> str:
        return self.SERVER_LABEL

    def _backend_wait_label(self) -> str:
        return self.MODEL_INSTANCE

    def _status_file_label(self) -> str:
        return self.SERVER_LABEL

    def _wait_for_initial_camera_if_required(
        self,
        *,
        mode: str,
        rgb_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        timeout_sec: float,
    ) -> None:
        del mode, rgb_topic, depth_topic, camera_info_topic, timeout_sec

    def _to_jsonable(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        if hasattr(value, 'tolist'):
            return self._to_jsonable(value.tolist())
        return str(value)

    def _build_observation(self) -> ModelSimObservation:
        return ModelSimObservation(
            pose=self._pose,
            goal=self._goal,
            subgoal=self._subgoal,
            instruction=self._instruction,
            rgb_image=self._latest_rgb,
            depth_image=self._latest_depth,
            camera_intrinsics=self._camera_intrinsics,
            camera_frame_id=(
                self._latest_rgb_msg.header.frame_id
                if self._latest_rgb_msg is not None
                else self._latest_depth_msg.header.frame_id if self._latest_depth_msg is not None else ''
            ),
            look_down=bool(self._params['look_down']),
            metadata={
                'namespace': self.get_namespace(),
                'rgb_available': self._latest_rgb is not None,
                'depth_available': self._latest_depth is not None,
                'camera_info_available': self._camera_intrinsics is not None,
            },
        )

    def _decision_to_twist(self, decision: ModelSimDecision) -> Twist:
        cmd = Twist()
        cmd.linear.x = float(decision.linear_x)
        cmd.angular.z = float(decision.angular_z)
        return cmd

    def _publish_status(self, decision: ModelSimDecision) -> None:
        message = String()
        message.data = json.dumps(
            {
                'status': decision.status,
                'degraded': bool(decision.degraded),
                'linear_x': float(decision.linear_x),
                'angular_z': float(decision.angular_z),
                'debug': self._to_jsonable(decision.debug),
                'wrapper': self.SERVER_LABEL,
                'model_instance': self.MODEL_INSTANCE,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        self._status_publisher.publish(message)

    def _should_compute(self, now: float) -> bool:
        if self._backend.uses_model_inference:
            rate_hz = self._params['inference_rate_hz']
            period = (1.0 / rate_hz) if rate_hz > 0 else self._params['command_timeout_sec']
        else:
            period = self._params['command_timeout_sec']
        return (now - self._last_compute_ts) >= period

    def _set_last_decision(self, decision: ModelSimDecision) -> None:
        with self._state_lock:
            self._last_decision = decision
            self._last_cmd = self._decision_to_twist(decision)

    def _get_last_cmd(self) -> Twist:
        with self._state_lock:
            return self._last_cmd

    def _start_background_compute(self, observation: ModelSimObservation, now: float) -> bool:
        with self._state_lock:
            if self._compute_in_progress:
                return False
            self._compute_in_progress = True
            self._last_compute_ts = now

        thread = threading.Thread(
            target=self._compute_worker,
            args=(observation,),
            name=self.COMPUTE_THREAD_NAME,
            daemon=True,
        )
        thread.start()
        return True

    def _compute_worker(self, observation: ModelSimObservation) -> None:
        try:
            decision = self._backend.compute(observation)
        except Exception as exc:
            self.get_logger().error(f'background get_command compute failed: {exc}')
            decision = ModelSimDecision(
                status='exception',
                degraded=True,
                debug={'failure_reason': str(exc), 'safe_stop': True},
            )

        with self._state_lock:
            self._last_decision = decision
            self._last_cmd = self._decision_to_twist(decision)
            self._compute_in_progress = False

        self._publish_status(decision)
        self._maybe_publish_visualization(observation, decision)

    def _publish_fallback_while_computing(self, observation: ModelSimObservation) -> None:
        try:
            decision = self._fallback_backend.compute(observation)
            decision.status = 'inference_in_progress_cached_' + decision.status
            decision.degraded = True
            decision.debug['background_compute_in_progress'] = True
            self._set_last_decision(decision)
            self._publish_status(decision)
        except Exception as exc:
            self.get_logger().warn(f'fallback command while model inference is running failed: {exc}')

    def _maybe_publish_visualization(self, observation: ModelSimObservation, decision: ModelSimDecision) -> None:
        if not self._visualization_enabled or self._visualization_publisher is None:
            return
        if self._latest_rgb is None or self._latest_rgb_msg is None:
            return

        now = time.monotonic()
        rate_hz = float(self.get_parameter('visualization_rate_hz').value)
        period = (1.0 / rate_hz) if rate_hz > 0 else 0.0
        if period > 0.0 and (now - self._last_visualization_ts) < period:
            return

        try:
            image = render_debug_overlay(
                self._latest_rgb,
                observation,
                decision,
                backend_name=self._backend.backend_type,
            )
            self._visualization_publisher.publish(numpy_to_image_msg(image, self._latest_rgb_msg))
            self._last_visualization_ts = now
        except Exception as exc:
            self.get_logger().warn(f'Failed to publish {self.SERVER_LABEL} visualization: {exc}')

    def _on_instruction(self, msg: String) -> None:
        if msg.data:
            self._instruction = msg.data

    def _on_pose(self, msg: PoseStamped) -> None:
        # In Isaac USD fallback mode the /pose stream can be delayed or sourced
        # differently from the odom/TF frame that Nav2 is actually controlling.
        # Prefer recent odom so the fallback controller tracks the same moving
        # state as Nav2 instead of steering against a stale pose snapshot.
        if (time.monotonic() - self._last_odom_pose_ts) < 1.0:
            return
        q = msg.pose.orientation
        if not _finite_values(msg.pose.position.x, msg.pose.position.y, q.x, q.y, q.z, q.w):
            self.get_logger().warn('Ignoring non-finite pose update')
            return
        self._pose = Pose2D(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            yaw=_yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w)),
        )

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        if not _finite_values(msg.pose.pose.position.x, msg.pose.pose.position.y, q.x, q.y, q.z, q.w):
            self.get_logger().warn('Ignoring non-finite odom pose update')
            return
        self._pose = Pose2D(
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=_yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w)),
        )
        self._last_odom_pose_ts = time.monotonic()

    def _on_goal(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self._goal = Pose2D(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            yaw=_yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w)),
        )

    def _on_subgoal(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self._subgoal = Pose2D(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            yaw=_yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w)),
        )

    def _on_rgb(self, msg: Image) -> None:
        rgb = image_msg_to_numpy(msg)
        if rgb is None:
            return
        self._latest_rgb = rgb
        self._latest_rgb_msg = msg

    def _on_depth(self, msg: Image) -> None:
        depth = image_msg_to_numpy(msg)
        if depth is None:
            return
        self._latest_depth = depth
        self._latest_depth_msg = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_intrinsics = tuple(float(value) for value in msg.k)

    def _on_get_command(self, request: GetCommand.Request, response: GetCommand.Response) -> GetCommand.Response:
        del request
        now = time.monotonic()

        try:
            observation = self._build_observation()
            if self._should_compute(now):
                if self._backend.uses_model_inference:
                    started = self._start_background_compute(observation, now)
                    if started:
                        self.get_logger().info(
                            f'Started background {self.SERVER_LABEL} inference; returning cached/fallback command to keep Nav2 responsive.'
                        )
                    with self._state_lock:
                        current_status = self._last_decision.status
                        current_cmd = self._last_cmd
                        compute_in_progress = self._compute_in_progress
                    if started or compute_in_progress or current_status in {'startup', 'backend_ready', 'backend_unavailable'} or (
                        abs(current_cmd.linear.x) < 1e-6 and abs(current_cmd.angular.z) < 1e-6
                    ):
                        self._publish_fallback_while_computing(observation)
                else:
                    decision = self._backend.compute(observation)
                    self._set_last_decision(decision)
                    self._last_compute_ts = now
                    self._publish_status(decision)
            with self._state_lock:
                last_decision = self._last_decision
            self._maybe_publish_visualization(observation, last_decision)
        except Exception as exc:
            self.get_logger().error(f'get_command failed: {exc}')
            decision = ModelSimDecision(
                status='exception',
                degraded=True,
                debug={'failure_reason': str(exc), 'safe_stop': True},
            )
            self._set_last_decision(decision)
            self._publish_status(decision)

        response.twist = self._get_last_cmd()
        return response


class InternNavServer(BaseModelSimServer):
    """InternNav-specific model-sim wrapper.

    The wrapper is named after the concrete model family. `dual_vln` is treated as
    one possible model instance / planner integration mode rather than the wrapper's
    architectural name.
    """

    NODE_NAME = 'internnav_server'
    SERVER_LABEL = 'internnav'
    MODEL_INSTANCE = 'dual_vln'
    DEFAULT_VISUALIZATION_TOPIC = 'internnav/debug_image'
    DEFAULT_STATUS_TOPIC = 'internnav/status'
    COMPUTE_THREAD_NAME = 'internnav_compute_worker'

    def _requires_initial_camera(self, *, mode: str, rgb_topic: str, depth_topic: str, camera_info_topic: str) -> bool:
        mode_lower = str(mode or '').strip().lower()
        model_path_lower = str(self._params.get('model_path', '')).lower()
        adapter_lower = str(self._params.get('adapter_target', '')).lower()
        camera_topics_configured = bool(rgb_topic and depth_topic and camera_info_topic)
        internnav_like = (
            mode_lower in {'internnav', 'model'}
            or 'internnav' in adapter_lower
            or 'internvla' in adapter_lower
            or 'internnav' in model_path_lower
            or 'internvla' in model_path_lower
        )
        return camera_topics_configured and internnav_like

    def _camera_missing_inputs(self) -> list[str]:
        missing: list[str] = []
        if self._latest_rgb is None:
            missing.append('rgb')
        if self._latest_depth is None:
            missing.append('depth')
        if self._camera_intrinsics is None:
            missing.append('camera_info')
        return missing

    def _wait_for_initial_camera_if_required(
        self,
        *,
        mode: str,
        rgb_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        timeout_sec: float,
    ) -> None:
        if not self._requires_initial_camera(
            mode=mode,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
        ):
            return

        timeout_sec = max(float(timeout_sec), 0.0)
        self.get_logger().info(
            'Waiting for first real camera messages before loading InternNav adapter: '
            f'rgb={rgb_topic}, depth={depth_topic}, camera_info={camera_info_topic}, timeout={timeout_sec:.1f}s'
        )
        waiting = ModelSimDecision(
            status='waiting_for_camera',
            degraded=True,
            debug={
                'rgb_topic': rgb_topic,
                'depth_topic': depth_topic,
                'camera_info_topic': camera_info_topic,
                'timeout_sec': timeout_sec,
            },
        )
        self._publish_status(waiting)

        deadline = time.monotonic() + timeout_sec
        missing = self._camera_missing_inputs()
        while rclpy.ok() and missing and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            missing = self._camera_missing_inputs()

        if not missing:
            self.get_logger().info('Initial camera readiness barrier passed; loading InternNav adapter now.')
            return

        self.get_logger().warn(
            'Timed out waiting for initial camera messages before InternNav adapter load; missing '
            + ', '.join(missing)
            + '. Continuing so the eval can surface the failure explicitly.'
        )

def main() -> None:
    rclpy.init()
    node = InternNavServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
