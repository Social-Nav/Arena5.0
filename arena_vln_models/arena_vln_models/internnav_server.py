import json
import math
import os
import sys
import threading
import time
from copy import deepcopy
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
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from rosnav_rl_msgs.srv import GetCommand
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from arena_vln_models.backends import ModelSimDecision, ModelSimObservation, Pose2D, create_model_backend
from arena_vln_models.visualization import image_msg_to_numpy, numpy_to_image_msg, render_debug_overlay


DEFAULT_INTERNNAV_ADAPTER_TARGET = 'arena_vln_models.internnav:load_internnav_adapter'
REALWORLD_HTTP_ADAPTER_TARGET = 'arena_vln_models.internnav:load_internvla_realworld_http_adapter'
LEGACY_INTERNNAV_ADAPTER_TARGETS = {
    'internnav.agent.internvla_n1_agent_realworld.InternVLAN1AsyncAgent': DEFAULT_INTERNNAV_ADAPTER_TARGET,
}
HTTP_ADAPTER_REPLACED_TARGETS = {
    DEFAULT_INTERNNAV_ADAPTER_TARGET,
    *LEGACY_INTERNNAV_ADAPTER_TARGETS.keys(),
}


def _env_override(*names: str) -> tuple[str, str | None]:
    for name in names:
        value = os.environ.get(name, '').strip()
        if value:
            return value, name
    return '', None


def _resolve_bool(raw_value, *, env_names: tuple[str, ...] = ()) -> tuple[bool, str | None]:
    env_value, env_name = _env_override(*env_names)
    if env_name is not None:
        return env_value.lower() in {'1', 'true', 'yes', 'on'}, f'env:{env_name}'
    return bool(raw_value), None


def _resolve_float(raw_value, *, env_names: tuple[str, ...] = ()) -> tuple[float, str | None]:
    env_value, env_name = _env_override(*env_names)
    if env_name is not None:
        try:
            return float(env_value), f'env:{env_name}'
        except (TypeError, ValueError):
            return float(raw_value), f'invalid-env:{env_name}'
    return float(raw_value), None


def _resolve_string(raw_value, *, env_names: tuple[str, ...] = (), allow_empty: bool = False) -> tuple[str, str | None]:
    env_value, env_name = _env_override(*env_names)
    if env_name is not None:
        return env_value, f'env:{env_name}'
    value = str(raw_value)
    if not allow_empty:
        value = value.strip()
    return value, None


def _normalize_internnav_adapter_target(mode: str, adapter_target: str) -> tuple[str, str | None]:
    normalized_mode = str(mode or '').strip().lower()
    normalized_target = str(adapter_target or '').strip()
    if not normalized_target and normalized_mode == 'internnav':
        return DEFAULT_INTERNNAV_ADAPTER_TARGET, 'default'

    mapped_target = LEGACY_INTERNNAV_ADAPTER_TARGETS.get(normalized_target)
    if mapped_target is not None:
        return mapped_target, f'legacy:{normalized_target}'

    return normalized_target, None


def _resolve_mode_for_http_adapter(mode: str, internnav_http_url: str) -> tuple[str, str | None]:
    normalized_mode = str(mode or '').strip().lower()
    if str(internnav_http_url or '').strip() and normalized_mode != 'internnav':
        return 'internnav', 'internnav_http_url'
    return mode, None


def _resolve_adapter_target_for_http_adapter(adapter_target: str, internnav_http_url: str) -> tuple[str, str | None]:
    normalized_target = str(adapter_target or '').strip()
    if not str(internnav_http_url or '').strip():
        return normalized_target, None
    if not normalized_target or normalized_target in HTTP_ADAPTER_REPLACED_TARGETS:
        return REALWORLD_HTTP_ADAPTER_TARGET, 'internnav_http_url'
    return normalized_target, None


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
    DEFAULT_ACTION_VISUALIZATION_TOPIC = 'model_sim/action_image'
    DEFAULT_STATUS_TOPIC = 'model_sim/status'
    DEFAULT_MODEL_OUTPUT_TOPIC = 'model_sim/model_output'
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
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('visualization_topic', self.DEFAULT_VISUALIZATION_TOPIC)
        self.declare_parameter('action_visualization_topic', self.DEFAULT_ACTION_VISUALIZATION_TOPIC)
        self.declare_parameter('visualization_rate_hz', 5.0)
        self.declare_parameter('status_topic', self.DEFAULT_STATUS_TOPIC)
        self.declare_parameter('model_output_topic', self.DEFAULT_MODEL_OUTPUT_TOPIC)

        # Control params
        # NOTE: "command_timeout_sec" historically acted as a compute throttle.
        # Keep it for compatibility, but prefer inference_rate_hz for model.
        self.declare_parameter('mode', 'heuristic')  # heuristic | model
        self.declare_parameter('command_timeout_sec', 0.2)
        self.declare_parameter('inference_rate_hz', 10.0)
        self.declare_parameter('inference_timeout_sec', 0.2)
        self.declare_parameter('camera_ready_timeout_sec', 120.0)
        self.declare_parameter('camera_stale_after_sec', 2.0)
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('adapter_target', '')
        self.declare_parameter('internnav_http_url', '')
        self.declare_parameter('internnav_http_timeout_sec', 0.0)
        self.declare_parameter('require_real_backend', False)
        self.declare_parameter('strict_device', False)
        self.declare_parameter('look_down', False)
        self.declare_parameter('model_output_policy', 'trajectory')
        self.declare_parameter('internnav_symbolic_fallback_policy', '')
        self.declare_parameter('synthetic_action_sequence', '')
        self.declare_parameter('official_discrete_action_tail_limit', 0)
        self.declare_parameter('official_discrete_forward_speed', 0.0)
        self.declare_parameter('official_discrete_turn_speed', 0.0)
        self.declare_parameter('require_route_instruction', True)
        self.declare_parameter('discrete_arc_turn', False)
        self.declare_parameter('max_linear', 0.6)
        self.declare_parameter('max_angular', 1.5)
        self.declare_parameter('k_lin', 1.2)
        self.declare_parameter('k_ang', 2.0)
        self.declare_parameter('goal_tolerance', 0.45)
        self.declare_parameter('angle_tolerance', 0.25)
        self.declare_parameter('min_lin_when_aligned', 0.05)
        self.declare_parameter('trace_path', '')
        # Discrete action → Twist mapping is implemented in backends._action_to_command().
        # See that function for the ROS coordinate convention (REP 103) and the rationale
        # for not making the turn sign configurable.  Turn actions default to in-place
        # rotation; discrete_arc_turn keeps the old forward-arc behavior available for
        # explicit compatibility testing.

        self._pose: Optional[Pose2D] = None
        self._last_odom_pose_ts: float = 0.0
        self._goal: Optional[Pose2D] = None
        self._subgoal: Optional[Pose2D] = None
        self._instruction: str = 'navigate'
        self._last_compute_ts: float = 0.0
        self._last_visualization_ts: float = 0.0
        self._last_cmd: Twist = Twist()
        self._last_decision: ModelSimDecision = ModelSimDecision(status='startup', degraded=True)
        self._last_model_decision: Optional[ModelSimDecision] = None
        self._state_lock = threading.Lock()
        self._compute_in_progress = False
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_rgb_msg: Optional[Image] = None
        self._latest_rgb_ts: float = 0.0
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_depth_msg: Optional[Image] = None
        self._latest_depth_ts: float = 0.0
        self._camera_intrinsics: Optional[tuple[float, ...]] = None
        self._camera_info_ts: float = 0.0
        self._required_readiness_topics: dict[str, str] = {'rgb': '', 'depth': '', 'camera_info': '', 'odom': ''}
        self._required_tf_frames: dict[str, str] = {'base': '', 'odom': '', 'global': ''}
        self._camera_required: bool = False
        self._required_camera_topics: dict[str, str] = {'rgb': '', 'depth': '', 'camera_info': ''}
        self._initial_camera_timed_out: bool = False
        self._trace_path: str = ''
        self._trace_seq: int = 0
        self._model_output_seq: int = 0
        self._action_history: list[int] = []
        self._previous_pose_for_action_trace: Optional[Pose2D] = None
        self._backend = None
        self._backend_mode: str = ''
        self._backend_init_error: str = ''
        self._last_instruction_quality_warning: str = ''
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        mode, mode_source = _resolve_string(
            self.get_parameter('mode').value,
            env_names=('ARENA_EVAL_INTERNNAV_MODE',),
        )
        model_path, model_path_source = _resolve_string(
            self.get_parameter('model_path').value,
            env_names=('ARENA_EVAL_INTERNNAV_MODEL_PATH',),
            allow_empty=True,
        )
        device, device_source = _resolve_string(
            self.get_parameter('device').value,
            env_names=('ARENA_EVAL_INTERNNAV_DEVICE',),
        )
        rgb_topic, rgb_topic_source = _resolve_string(
            self.get_parameter('rgb_topic').value,
            env_names=('ARENA_EVAL_INTERNNAV_RGB_TOPIC',),
            allow_empty=True,
        )
        depth_topic, depth_topic_source = _resolve_string(
            self.get_parameter('depth_topic').value,
            env_names=('ARENA_EVAL_INTERNNAV_DEPTH_TOPIC',),
            allow_empty=True,
        )
        camera_info_topic, camera_info_topic_source = _resolve_string(
            self.get_parameter('camera_info_topic').value,
            env_names=('ARENA_EVAL_INTERNNAV_CAMERA_INFO_TOPIC',),
            allow_empty=True,
        )
        inference_rate_hz, inference_rate_hz_source = _resolve_float(
            self.get_parameter('inference_rate_hz').value,
            env_names=('ARENA_EVAL_INTERNNAV_INFERENCE_RATE_HZ',),
        )
        inference_timeout_sec, inference_timeout_sec_source = _resolve_float(
            self.get_parameter('inference_timeout_sec').value,
            env_names=('ARENA_EVAL_INTERNNAV_INFERENCE_TIMEOUT_SEC',),
        )
        require_real_backend, require_real_backend_source = _resolve_bool(
            self.get_parameter('require_real_backend').value,
            env_names=('ARENA_EVAL_INTERNNAV_REQUIRE_REAL_BACKEND',),
        )
        strict_device, strict_device_source = _resolve_bool(
            self.get_parameter('strict_device').value,
            env_names=('ARENA_EVAL_INTERNNAV_STRICT_DEVICE',),
        )
        look_down, look_down_source = _resolve_bool(
            self.get_parameter('look_down').value,
            env_names=('ARENA_EVAL_INTERNNAV_LOOK_DOWN',),
        )
        require_route_instruction, require_route_instruction_source = _resolve_bool(
            self.get_parameter('require_route_instruction').value,
            env_names=('ARENA_EVAL_INTERNNAV_REQUIRE_ROUTE_INSTRUCTION', 'ARENA_INTERNNAV_REQUIRE_ROUTE_INSTRUCTION'),
        )
        enable_visualization, enable_visualization_source = _resolve_bool(
            self.get_parameter('enable_visualization').value,
            env_names=('ARENA_EVAL_INTERNNAV_ENABLE_VISUALIZATION',),
        )
        trace_path, trace_path_source = _resolve_string(
            self.get_parameter('trace_path').value,
            env_names=('ARENA_EVAL_INTERNNAV_TRACE_PATH', 'ARENA_INTERNNAV_TRACE_PATH'),
            allow_empty=True,
        )
        model_output_policy, model_output_policy_source = _resolve_string(
            self.get_parameter('model_output_policy').value,
            env_names=('ARENA_EVAL_INTERNNAV_MODEL_OUTPUT_POLICY', 'ARENA_INTERNNAV_MODEL_OUTPUT_POLICY'),
        )
        symbolic_fallback_policy, symbolic_fallback_policy_source = _resolve_string(
            self.get_parameter('internnav_symbolic_fallback_policy').value,
            env_names=(
                'ARENA_EVAL_INTERNNAV_SYMBOLIC_FALLBACK_POLICY',
                'ARENA_INTERNNAV_SYMBOLIC_FALLBACK_POLICY',
            ),
            allow_empty=True,
        )
        synthetic_action_sequence, synthetic_action_sequence_source = _resolve_string(
            self.get_parameter('synthetic_action_sequence').value,
            env_names=(
                'ARENA_EVAL_INTERNNAV_SYNTHETIC_ACTION_SEQUENCE',
                'ARENA_INTERNNAV_SYNTHETIC_ACTION_SEQUENCE',
            ),
            allow_empty=True,
        )
        official_action_tail_limit, official_action_tail_limit_source = _resolve_float(
            self.get_parameter('official_discrete_action_tail_limit').value,
            env_names=(
                'ARENA_EVAL_INTERNNAV_OFFICIAL_ACTION_TAIL_LIMIT',
                'ARENA_INTERNNAV_OFFICIAL_ACTION_TAIL_LIMIT',
            ),
        )
        official_forward_speed, official_forward_speed_source = _resolve_float(
            self.get_parameter('official_discrete_forward_speed').value,
            env_names=(
                'ARENA_EVAL_INTERNNAV_OFFICIAL_DISCRETE_FORWARD_SPEED',
                'ARENA_INTERNNAV_OFFICIAL_DISCRETE_FORWARD_SPEED',
            ),
        )
        official_turn_speed, official_turn_speed_source = _resolve_float(
            self.get_parameter('official_discrete_turn_speed').value,
            env_names=(
                'ARENA_EVAL_INTERNNAV_OFFICIAL_DISCRETE_TURN_SPEED',
                'ARENA_INTERNNAV_OFFICIAL_DISCRETE_TURN_SPEED',
            ),
        )
        discrete_arc_turn, discrete_arc_turn_source = _resolve_bool(
            self.get_parameter('discrete_arc_turn').value,
            env_names=('ARENA_EVAL_INTERNNAV_DISCRETE_ARC_TURN', 'ARENA_INTERNNAV_DISCRETE_ARC_TURN'),
        )
        visualization_topic, visualization_topic_source = _resolve_string(
            self.get_parameter('visualization_topic').value,
            env_names=('ARENA_EVAL_INTERNNAV_VISUALIZATION_TOPIC',),
        )
        action_visualization_topic, action_visualization_topic_source = _resolve_string(
            self.get_parameter('action_visualization_topic').value,
            env_names=('ARENA_EVAL_INTERNNAV_ACTION_VISUALIZATION_TOPIC',),
        )
        visualization_rate_hz, visualization_rate_hz_source = _resolve_float(
            self.get_parameter('visualization_rate_hz').value,
            env_names=('ARENA_EVAL_INTERNNAV_VISUALIZATION_RATE_HZ',),
        )
        model_output_topic, model_output_topic_source = _resolve_string(
            self.get_parameter('model_output_topic').value,
            env_names=('ARENA_EVAL_INTERNNAV_MODEL_OUTPUT_TOPIC',),
        )
        adapter_target_raw, adapter_target_env_source = _resolve_string(
            self.get_parameter('adapter_target').value,
            env_names=('ARENA_EVAL_INTERNNAV_ADAPTER_TARGET',),
            allow_empty=True,
        )
        internnav_http_url, internnav_http_url_source = _resolve_string(
            self.get_parameter('internnav_http_url').value,
            env_names=('ARENA_EVAL_INTERNNAV_HTTP_URL', 'ARENA_INTERNNAV_HTTP_URL'),
            allow_empty=True,
        )
        internnav_http_timeout_sec, internnav_http_timeout_sec_source = _resolve_float(
            self.get_parameter('internnav_http_timeout_sec').value,
            env_names=('ARENA_EVAL_INTERNNAV_HTTP_TIMEOUT_SEC', 'ARENA_INTERNNAV_HTTP_TIMEOUT_SEC'),
        )
        resolved_mode, http_mode_source = _resolve_mode_for_http_adapter(mode, internnav_http_url)
        if http_mode_source is not None:
            mode = resolved_mode
            mode_source = http_mode_source
        adapter_target_raw, http_adapter_target_source = _resolve_adapter_target_for_http_adapter(
            adapter_target_raw,
            internnav_http_url,
        )
        if http_adapter_target_source is not None:
            adapter_target_env_source = http_adapter_target_source
        adapter_target, adapter_target_source = _normalize_internnav_adapter_target(
            mode,
            adapter_target_raw,
        )
        parameter_overrides = []
        for name, value, source in (
            ('mode', mode, mode_source),
            ('model_path', model_path, model_path_source),
            ('device', device, device_source),
            ('rgb_topic', rgb_topic, rgb_topic_source),
            ('depth_topic', depth_topic, depth_topic_source),
            ('camera_info_topic', camera_info_topic, camera_info_topic_source),
            ('inference_rate_hz', inference_rate_hz, inference_rate_hz_source),
            ('inference_timeout_sec', inference_timeout_sec, inference_timeout_sec_source),
            ('require_real_backend', require_real_backend, require_real_backend_source),
            ('strict_device', strict_device, strict_device_source),
            ('look_down', look_down, look_down_source),
            ('require_route_instruction', require_route_instruction, require_route_instruction_source),
            ('enable_visualization', enable_visualization, enable_visualization_source),
            ('trace_path', trace_path, trace_path_source),
            ('model_output_policy', model_output_policy, model_output_policy_source),
            ('internnav_symbolic_fallback_policy', symbolic_fallback_policy, symbolic_fallback_policy_source),
            ('synthetic_action_sequence', synthetic_action_sequence, synthetic_action_sequence_source),
            ('official_discrete_action_tail_limit', int(official_action_tail_limit), official_action_tail_limit_source),
            ('official_discrete_forward_speed', official_forward_speed, official_forward_speed_source),
            ('official_discrete_turn_speed', official_turn_speed, official_turn_speed_source),
            ('discrete_arc_turn', discrete_arc_turn, discrete_arc_turn_source),
            ('visualization_topic', visualization_topic, visualization_topic_source),
            ('action_visualization_topic', action_visualization_topic, action_visualization_topic_source),
            ('visualization_rate_hz', visualization_rate_hz, visualization_rate_hz_source),
            ('model_output_topic', model_output_topic, model_output_topic_source),
            ('internnav_http_url', internnav_http_url, internnav_http_url_source),
            ('internnav_http_timeout_sec', internnav_http_timeout_sec, internnav_http_timeout_sec_source),
        ):
            if source is not None:
                parameter_overrides.append(Parameter(name, value=value))
        if adapter_target != str(self.get_parameter('adapter_target').value) or adapter_target_env_source is not None:
            parameter_overrides.append(Parameter('adapter_target', value=adapter_target))
        if parameter_overrides:
            self.set_parameters(parameter_overrides)
        self._params = {
            'command_timeout_sec': float(self.get_parameter('command_timeout_sec').value),
            'inference_rate_hz': inference_rate_hz,
            'inference_timeout_sec': inference_timeout_sec,
            'camera_ready_timeout_sec': float(self.get_parameter('camera_ready_timeout_sec').value),
            'camera_stale_after_sec': float(self.get_parameter('camera_stale_after_sec').value),
            'model_path': model_path,
            'device': device,
            'adapter_target': adapter_target,
            'internnav_http_url': internnav_http_url,
            'internnav_http_timeout_sec': internnav_http_timeout_sec if internnav_http_timeout_sec > 0.0 else inference_timeout_sec,
            'require_real_backend': require_real_backend,
            'strict_device': strict_device,
            'look_down': look_down,
            'require_route_instruction': require_route_instruction,
            'max_linear': float(self.get_parameter('max_linear').value),
            'max_angular': float(self.get_parameter('max_angular').value),
            'k_lin': float(self.get_parameter('k_lin').value),
            'k_ang': float(self.get_parameter('k_ang').value),
            'goal_tolerance': float(self.get_parameter('goal_tolerance').value),
            'angle_tolerance': float(self.get_parameter('angle_tolerance').value),
            'min_lin_when_aligned': float(self.get_parameter('min_lin_when_aligned').value),
            'model_output_policy': model_output_policy,
            'internnav_symbolic_fallback_policy': symbolic_fallback_policy,
            'synthetic_action_sequence': synthetic_action_sequence,
            'official_discrete_action_tail_limit': int(official_action_tail_limit),
            'official_discrete_forward_speed': float(official_forward_speed),
            'official_discrete_turn_speed': float(official_turn_speed),
            'discrete_arc_turn': discrete_arc_turn,
        }
        self._trace_path = trace_path

        # Latching instruction subscriber (matches publisher durability)
        instr_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        rgb_topic = rgb_topic
        depth_topic = depth_topic
        camera_info_topic = camera_info_topic
        # Isaac eval publishes camera and odom streams from simulator-side bridges
        # with BEST_EFFORT QoS. A default RELIABLE subscription is incompatible
        # with those publishers and leaves InternNav stuck in the strict
        # TF/odom/camera readiness barrier even while other BEST_EFFORT consumers
        # (for example the eval video recorder) receive the same data.
        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        self.create_subscription(PoseStamped, self.get_parameter('pose_topic').value, self._on_pose, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self._on_odom, sensor_qos)
        self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self._on_goal, 10)
        self.create_subscription(PoseStamped, self.get_parameter('subgoal_topic').value, self._on_subgoal, 10)
        self.create_subscription(String, self.get_parameter('instruction_topic').value, self._on_instruction, instr_qos)

        if rgb_topic:
            self.create_subscription(Image, rgb_topic, self._on_rgb, sensor_qos)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self._on_depth, sensor_qos)
        if camera_info_topic:
            self.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, sensor_qos)

        self._required_camera_topics = {
            'rgb': rgb_topic,
            'depth': depth_topic,
            'camera_info': camera_info_topic,
        }
        self._required_readiness_topics = {
            'rgb': rgb_topic,
            'depth': depth_topic,
            'camera_info': camera_info_topic,
            'odom': str(self.get_parameter('odom_topic').value or '').strip(),
        }
        self._required_tf_frames = {
            'base': self._namespaced_frame(str(self.get_parameter('base_frame').value or 'base_link')),
            'odom': self._namespaced_frame(str(self.get_parameter('odom_frame').value or 'odom')),
            'global': str(self.get_parameter('global_frame').value or 'map').strip() or 'map',
        }
        self._camera_required = self._requires_initial_camera(
            mode=mode,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
        )

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
        self._model_output_publisher = self.create_publisher(
            String,
            str(self.get_parameter('model_output_topic').value),
            status_qos,
        )

        self._visualization_enabled = enable_visualization
        self._visualization_publisher = None
        self._action_visualization_publisher = None
        if self._visualization_enabled:
            self._visualization_publisher = self.create_publisher(
                Image,
                str(self.get_parameter('visualization_topic').value),
                10,
            )
            self._action_visualization_publisher = self.create_publisher(
                Image,
                str(self.get_parameter('action_visualization_topic').value),
                10,
            )
            if not rgb_topic:
                self.get_logger().warn(
                    f'{self.SERVER_LABEL} visualization enabled but rgb_topic is empty; debug image publishing will stay idle'
                )

        if adapter_target_source == 'default':
            self.get_logger().info(
                'InternNav mode requested without adapter_target; defaulting to '
                + DEFAULT_INTERNNAV_ADAPTER_TARGET
            )
        elif adapter_target_source is not None:
            self.get_logger().warn(
                f'Normalizing legacy InternNav adapter_target to {adapter_target}: {adapter_target_source}'
            )
        for label, source in (
            ('mode', mode_source),
            ('model_path', model_path_source),
            ('device', device_source),
            ('rgb_topic', rgb_topic_source),
            ('depth_topic', depth_topic_source),
            ('camera_info_topic', camera_info_topic_source),
            ('inference_rate_hz', inference_rate_hz_source),
            ('inference_timeout_sec', inference_timeout_sec_source),
            ('require_real_backend', require_real_backend_source),
            ('strict_device', strict_device_source),
            ('look_down', look_down_source),
            ('enable_visualization', enable_visualization_source),
            ('trace_path', trace_path_source),
            ('model_output_policy', model_output_policy_source),
            ('internnav_symbolic_fallback_policy', symbolic_fallback_policy_source),
            ('synthetic_action_sequence', synthetic_action_sequence_source),
            ('official_discrete_action_tail_limit', official_action_tail_limit_source),
            ('official_discrete_forward_speed', official_forward_speed_source),
            ('official_discrete_turn_speed', official_turn_speed_source),
            ('discrete_arc_turn', discrete_arc_turn_source),
            ('visualization_topic', visualization_topic_source),
            ('action_visualization_topic', action_visualization_topic_source),
            ('visualization_rate_hz', visualization_rate_hz_source),
            ('model_output_topic', model_output_topic_source),
            ('internnav_http_url', internnav_http_url_source),
            ('internnav_http_timeout_sec', internnav_http_timeout_sec_source),
        ):
            if source is not None:
                self.get_logger().info(f'Using InternNav {label} override from {source}')

        self._backend_mode = mode
        self._fallback_backend = create_model_backend(mode='heuristic', logger=self.get_logger(), params=self._params)

        # Advertise the command service before the strict initial sensor wait so
        # externally launched servers are discoverable by eval preflight while
        # they are still waiting for the Arena/Isaac graph to publish real
        # TF/odom/camera inputs.  The task_generator still gates goal release on
        # a fresh backend_ready status, so service discovery alone cannot bypass
        # real-input readiness.
        self.create_service(GetCommand, 'get_command', self._on_get_command)

        self._wait_for_initial_camera_if_required(
            mode=mode,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
            timeout_sec=self._params['camera_ready_timeout_sec'],
        )
        self._try_create_backend_if_possible()

        backend = getattr(self, '_backend', None)

        self.get_logger().info(
            (
                f'{self.SERVER_LABEL}_server started '
                f'(mode={mode}, backend={getattr(backend, "backend_type", "<pending>")}, '
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
                f'camera_info={camera_info_topic or "<disabled>"} '
                f'base_frame={self._required_tf_frames["base"] or "<disabled>"} '
                f'odom_frame={self._required_tf_frames["odom"] or "<disabled>"} '
                f'global_frame={self._required_tf_frames["global"] or "<disabled>"} '
                f'look_down={self._params["look_down"]} '
                f'visualization={self._visualization_enabled} trace={self._trace_path or "<disabled>"} '
                f'action_visualization_topic={self.get_parameter("action_visualization_topic").value} '
                f'model_output_topic={self.get_parameter("model_output_topic").value}'
            )
        )
        adapter_available = backend is not None and getattr(backend, '_adapter_callable', True) is not None
        missing_inputs, stale_inputs = self._required_input_issues(require_fresh=False)
        startup_status = 'backend_ready' if adapter_available else 'backend_unavailable'
        startup_degraded = not adapter_available
        if missing_inputs or stale_inputs:
            startup_status = 'camera_timeout' if self._initial_camera_timed_out else 'waiting_for_camera'
            startup_degraded = True
        self._last_decision = ModelSimDecision(
            status=startup_status,
            degraded=startup_degraded,
            debug={
                'backend_type': getattr(backend, 'backend_type', ''),
                'backend_description': backend.describe() if backend is not None else '',
                'uses_model_inference': bool(getattr(backend, 'uses_model_inference', False)),
                'model_path': self._params['model_path'],
                'device': self._params['device'],
                'adapter_target': self._params['adapter_target'],
                'require_real_backend': self._params['require_real_backend'],
                'strict_device': self._params['strict_device'],
                'backend_pending': backend is None,
                'backend_init_error': self._backend_init_error,
                'missing_inputs': missing_inputs,
                'stale_inputs': stale_inputs,
                'sensor_ages_sec': self._camera_sensor_ages(),
                'stale_after_sec': self._params['camera_stale_after_sec'],
                'topics': self._required_readiness_topics,
                'tf_frames': self._required_tf_frames,
                'tf_ready': self._tf_tree_ready(),
            },
        )
        self._publish_status(self._last_decision)
        self._publish_model_output(None, self._last_decision, event_type='startup')
        self.create_timer(0.5, self._publish_readiness_status_if_ready)
        self.create_timer(1.0, self._republish_last_status)
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

    def _namespaced_frame(self, frame_name: str) -> str:
        normalized = str(frame_name or '').strip().strip('/')
        if not normalized:
            return ''
        # TF frame ids are data-plane identifiers, not ROS topic/service names.
        # They must not be prefixed with the node namespace (for example,
        # `/task_generator_node/Ai2_Bot2`) because the actual Isaac odom/TF tree
        # uses robot frame ids such as `Ai2_Bot2/odom` -> `Ai2_Bot2/base_link`.
        # Namespacing them here causes the readiness gate to wait for impossible
        # frames until a later odom sample teaches the wrapper the real pair.
        return normalized

    def _camera_sensor_ages(self) -> dict[str, float | None]:
        now = time.monotonic()
        last_odom_pose_ts = float(getattr(self, '_last_odom_pose_ts', 0.0) or 0.0)
        return {
            'rgb': (now - self._latest_rgb_ts) if self._latest_rgb_ts > 0.0 else None,
            'depth': (now - self._latest_depth_ts) if self._latest_depth_ts > 0.0 else None,
            'camera_info': (now - self._camera_info_ts) if self._camera_info_ts > 0.0 else None,
            'odom': (now - last_odom_pose_ts) if last_odom_pose_ts > 0.0 else None,
        }

    def _camera_input_issues(self, *, require_fresh: bool) -> tuple[list[str], list[str]]:
        if not self._camera_required:
            return [], []

        stale_after_sec = max(float(self._params.get('camera_stale_after_sec', 0.0)), 0.0)
        now = time.monotonic()
        missing: list[str] = []
        stale: list[str] = []
        for key, topic in self._required_camera_topics.items():
            if not topic:
                continue
            value_present = False
            last_ts = 0.0
            if key == 'rgb':
                value_present = self._latest_rgb is not None
                last_ts = self._latest_rgb_ts
            elif key == 'depth':
                value_present = self._latest_depth is not None
                last_ts = self._latest_depth_ts
            elif key == 'camera_info':
                value_present = self._camera_intrinsics is not None
                last_ts = self._camera_info_ts

            if not value_present:
                missing.append(key)
                continue
            if require_fresh and stale_after_sec > 0.0 and (now - last_ts) > stale_after_sec:
                stale.append(key)
        return missing, stale

    def _odom_input_issues(self, *, require_fresh: bool) -> tuple[list[str], list[str]]:
        odom_topic = str(self._required_readiness_topics.get('odom', '') or '').strip()
        if not odom_topic:
            return ['odom'], []

        missing: list[str] = []
        stale: list[str] = []
        if self._last_odom_pose_ts <= 0.0 or self._pose is None:
            missing.append('odom')
            return missing, stale

        stale_after_sec = max(float(self._params.get('camera_stale_after_sec', 0.0)), 0.0)
        if require_fresh and stale_after_sec > 0.0 and (time.monotonic() - self._last_odom_pose_ts) > stale_after_sec:
            stale.append('odom')
        return missing, stale

    def _tf_readiness_details(self) -> dict[str, object]:
        odom_frame = str(self._required_tf_frames.get('odom', '') or '').strip()
        base_frame = str(self._required_tf_frames.get('base', '') or '').strip()
        global_frame = str(self._required_tf_frames.get('global', '') or '').strip()

        details: dict[str, object] = {
            'odom_frame': odom_frame,
            'base_frame': base_frame,
            'global_frame': global_frame,
            'odom_to_base_ready': False,
            'global_to_odom_ready': False,
            'global_to_base_ready': False,
        }

        if not odom_frame or not base_frame:
            return details

        try:
            odom_to_base_ready = bool(self._tf_buffer.can_transform(odom_frame, base_frame, Time()))
        except Exception:
            odom_to_base_ready = False
        details['odom_to_base_ready'] = odom_to_base_ready

        # The Isaac/Jazzy eval path depends on the complete global/map -> odom ->
        # base chain, not just the local odom -> base edge.  If map/world is
        # disconnected while odom -> base exists, reporting backend_ready here is
        # a false positive: Nav2 behavior/planner nodes still fail with missing
        # global-frame transforms even though the InternNav wrapper believes TF is
        # ready.
        if global_frame and global_frame != odom_frame:
            try:
                global_to_odom_ready = bool(self._tf_buffer.can_transform(global_frame, odom_frame, Time()))
            except Exception:
                global_to_odom_ready = False
            try:
                global_to_base_ready = bool(self._tf_buffer.can_transform(global_frame, base_frame, Time()))
            except Exception:
                global_to_base_ready = False
        else:
            global_to_odom_ready = odom_to_base_ready
            global_to_base_ready = odom_to_base_ready

        details['global_to_odom_ready'] = global_to_odom_ready
        details['global_to_base_ready'] = global_to_base_ready
        return details

    def _tf_tree_ready(self) -> bool:
        details = self._tf_readiness_details()
        return bool(
            details.get('odom_to_base_ready')
            and details.get('global_to_odom_ready')
            and details.get('global_to_base_ready')
        )

    def _learn_tf_frames_from_odom(self, msg: Odometry) -> None:
        """Use real odometry frame ids for TF readiness.

        The InternNav node runs below a ROS namespace, but TF frame ids are data
        fields and are not automatically remapped by ROS namespaces.  Isaac eval
        odometry advertises frames like ``Ai2_Bot2/odom`` ->
        ``Ai2_Bot2/base_footprint``.  If the wrapper synthesizes names from the
        node namespace instead, it remains in safe-stop readiness despite real
        TF/odom being available.  Once real odometry arrives, align the required
        TF pair to the message header/child_frame_id.
        """
        odom_frame = str(getattr(msg.header, 'frame_id', '') or '').strip().strip('/')
        base_frame = str(getattr(msg, 'child_frame_id', '') or '').strip().strip('/')
        updated = False
        if odom_frame and odom_frame != self._required_tf_frames.get('odom'):
            self._required_tf_frames['odom'] = odom_frame
            updated = True
        if base_frame and base_frame != self._required_tf_frames.get('base'):
            self._required_tf_frames['base'] = base_frame
            updated = True
        if updated:
            self.get_logger().info(
                f'{self.SERVER_LABEL} learned TF frames from odom: '
                f'odom={self._required_tf_frames.get("odom") or "<unset>"} '
                f'base={self._required_tf_frames.get("base") or "<unset>"}'
            )

    def _required_input_issues(self, *, require_fresh: bool) -> tuple[list[str], list[str]]:
        missing, stale = self._camera_input_issues(require_fresh=require_fresh)
        odom_missing, odom_stale = self._odom_input_issues(require_fresh=require_fresh)
        missing.extend(odom_missing)
        stale.extend(odom_stale)
        if not self._tf_tree_ready():
            missing.append('tf')
        return missing, stale

    def _camera_gate_decision(self) -> ModelSimDecision | None:
        missing, stale = self._required_input_issues(require_fresh=True)
        if not missing and not stale:
            return None

        status = 'waiting_for_camera' if missing else 'stale_camera'
        if self._initial_camera_timed_out and (missing or stale):
            status = 'camera_timeout' if missing else 'stale_camera'
        return ModelSimDecision(
            status=status,
            degraded=True,
            debug={
                'safe_stop': True,
                'missing_inputs': missing,
                'stale_inputs': stale,
                'sensor_ages_sec': self._camera_sensor_ages(),
                'stale_after_sec': float(self._params.get('camera_stale_after_sec', 0.0)),
                'topics': self._required_readiness_topics,
                'tf_frames': self._required_tf_frames,
                'tf_checks': self._tf_readiness_details(),
                'tf_ready': self._tf_tree_ready(),
            },
        )

    def _backend_unavailable_decision(self) -> ModelSimDecision:
        return ModelSimDecision(
            status='backend_unavailable',
            degraded=True,
            debug={
                'safe_stop': True,
                'backend_pending': getattr(self, '_backend', None) is None,
                'backend_init_error': self._backend_init_error,
                'model_path': self._params['model_path'],
                'device': self._params['device'],
                'adapter_target': self._params['adapter_target'],
                'topics': self._required_readiness_topics,
                'tf_frames': self._required_tf_frames,
                'tf_checks': self._tf_readiness_details(),
                'tf_ready': self._tf_tree_ready(),
                'sensor_ages_sec': self._camera_sensor_ages(),
                'stale_after_sec': float(self._params.get('camera_stale_after_sec', 0.0)),
            },
        )

    def _try_create_backend_if_possible(self) -> bool:
        if getattr(self, '_backend', None) is not None:
            return True

        missing, stale = self._required_input_issues(require_fresh=False)
        if self._params['require_real_backend'] and (missing or stale):
            return False

        mode = self._backend_mode or str(self._params.get('mode', 'heuristic'))
        try:
            backend = create_model_backend(mode=mode, logger=self.get_logger(), params=self._params)
        except Exception as exc:
            message = f"Failed to create required real backend for mode='{mode}': {exc}"
            if self._params['require_real_backend']:
                if message != self._backend_init_error:
                    self.get_logger().error(message)
                self._backend_init_error = message
                return False

            fallback_message = f"Failed to create backend for mode='{mode}': {exc}; falling back to heuristic"
            if fallback_message != self._backend_init_error:
                self.get_logger().error(fallback_message)
            self._backend_init_error = fallback_message
            backend = create_model_backend(mode='heuristic', logger=self.get_logger(), params=self._params)

        self._backend = backend
        self._backend_init_error = ''
        self.get_logger().info(f'{self.SERVER_LABEL} backend ready: {backend.describe()}')
        return True

    def _publish_readiness_status_if_ready(self) -> None:
        """Recover from a startup camera race once Isaac frames arrive.

        Isaac robot/camera graph creation can complete a few seconds after the
        InternNav server starts.  If the short startup camera barrier times out,
        the server used to keep the latched status at ``camera_timeout`` until a
        Nav2 ``get_command`` call arrived.  The robot manager waits for
        ``backend_ready`` before publishing that navigation goal, so this caused
        a circular wait.  Publish ``backend_ready`` as soon as the subscribed
        RGB/depth streams are actually present.
        """
        if getattr(self, '_backend', None) is None and not self._try_create_backend_if_possible():
            missing, stale = self._required_input_issues(require_fresh=False)
            if not missing and not stale:
                decision = self._backend_unavailable_decision()
                self._set_last_decision(decision)
                self._publish_status(decision)
                self._publish_model_output(None, decision, event_type='readiness')
            return
        adapter_available = getattr(self._backend, '_adapter_callable', True) is not None
        if not adapter_available:
            return
        missing, stale = self._required_input_issues(require_fresh=False)
        if missing or stale:
            return
        with self._state_lock:
            current_status = self._last_decision.status
        if current_status == 'backend_ready':
            # Keep the latched readiness sample fresh.  RobotManager validates
            # sensor ages from the status payload before releasing the VLN goal;
            # republishing the original backend_ready decision here can overwrite
            # the transient-local cache with stale startup ages and make late
            # subscribers miss an otherwise healthy InternNav server.
            self._republish_last_status()
            return
        if current_status not in {'startup', 'waiting_for_camera', 'camera_timeout', 'stale_camera', 'waiting_for_instruction'}:
            return
        decision = ModelSimDecision(
            status='backend_ready',
            degraded=False,
            debug={
                'backend_type': self._backend.backend_type,
                'backend_description': self._backend.describe(),
                'uses_model_inference': bool(self._backend.uses_model_inference),
                'model_path': self._params['model_path'],
                'device': self._params['device'],
                'adapter_target': self._params['adapter_target'],
                'backend_init_error': self._backend_init_error,
                'startup_camera_timeout_recovered': bool(self._initial_camera_timed_out),
                'sensor_ages_sec': self._camera_sensor_ages(),
                'topics': self._required_readiness_topics,
                'tf_frames': self._required_tf_frames,
                'tf_checks': self._tf_readiness_details(),
                'tf_ready': self._tf_tree_ready(),
            },
        )
        self._set_last_decision(decision)
        self._publish_status(decision)
        self._publish_model_output(None, decision, event_type='readiness')

    def _republish_last_status(self) -> None:
        with self._state_lock:
            decision = deepcopy(self._last_decision)
        decision.debug['sensor_ages_sec'] = self._camera_sensor_ages()
        decision.debug['stale_after_sec'] = float(self._params.get('camera_stale_after_sec', 0.0))
        self._publish_status(decision)

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
                'sensor_ages_sec': self._camera_sensor_ages(),
                'stale_after_sec': float(self._params.get('camera_stale_after_sec', 0.0)),
            },
        )

    def _goal_geometry(self, observation: ModelSimObservation) -> dict[str, float | list[float] | None]:
        pose = observation.pose
        goal = observation.goal or observation.subgoal
        if pose is None or goal is None:
            return {
                'pose': None,
                'goal': None,
                'goal_distance': None,
                'yaw_error': None,
            }
        dx = float(goal.x) - float(pose.x)
        dy = float(goal.y) - float(pose.y)
        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - float(pose.yaw)), math.cos(target_yaw - float(pose.yaw)))
        return {
            'pose': [float(pose.x), float(pose.y), float(pose.yaw)],
            'goal': [float(goal.x), float(goal.y), float(goal.yaw)],
            'goal_distance': math.hypot(dx, dy),
            'yaw_error': yaw_error,
        }

    def _instruction_diagnostics(self, instruction: str) -> dict[str, object]:
        text = str(instruction or '').strip()
        normalized = ' '.join(text.lower().split())
        generic_defaults = {'', 'navigate', 'go', 'start', 'default', 'none', 'null'}
        diagnostics: dict[str, object] = {
            'instruction_normalized': normalized,
            'instruction_is_empty': not text,
            'instruction_is_default': normalized in generic_defaults,
            'instruction_quality': 'ok',
        }
        if not text:
            diagnostics['instruction_quality'] = 'empty'
            diagnostics['instruction_warning'] = (
                'InternNav/DualVLN expects a natural-language VLN instruction; received an empty instruction.'
            )
        elif normalized in generic_defaults:
            diagnostics['instruction_quality'] = 'generic_default'
            diagnostics['instruction_warning'] = (
                f"InternNav/DualVLN expects a route-specific natural-language instruction; received generic '{text}'."
            )
        return diagnostics

    def _instruction_gate_decision(self, observation: ModelSimObservation) -> ModelSimDecision | None:
        """Return a safe-stop decision until a real per-episode VLN instruction arrives."""
        if not bool(self._params.get('require_route_instruction', True)):
            return None

        diagnostics = self._instruction_diagnostics(observation.instruction)
        if diagnostics.get('instruction_quality') == 'ok':
            return None

        return ModelSimDecision(
            status='waiting_for_instruction',
            degraded=True,
            debug={
                'safe_stop': True,
                'instruction_gate': True,
                'route_instruction_required': True,
                'instruction_topic': str(self.get_parameter('instruction_topic').value or ''),
                'instruction_gate_reason': diagnostics.get('instruction_quality'),
                'instruction_warning': diagnostics.get('instruction_warning'),
            },
        )

    def _write_trace_record(
        self,
        observation: ModelSimObservation,
        decision: ModelSimDecision,
        *,
        event_type: str = 'model_result',
    ) -> None:
        if not self._trace_path:
            return
        selected_action = decision.debug.get('selected_action')
        try:
            selected_action_int = int(selected_action) if selected_action is not None else None
        except (TypeError, ValueError):
            selected_action_int = None
        self._trace_seq += 1
        geometry = self._goal_geometry(observation)
        debug = self._to_jsonable(decision.debug)
        sensor_ages = self._camera_sensor_ages()
        record = {
            'seq': self._trace_seq,
            'stamp_wall_time': time.time(),
            'stamp_monotonic': time.monotonic(),
            'namespace': self.get_namespace(),
            'wrapper': self.SERVER_LABEL,
            'model_instance': self.MODEL_INSTANCE,
            'backend_type': getattr(self._backend, 'backend_type', ''),
            'event_type': event_type,
            'status': decision.status,
            'degraded': bool(decision.degraded),
            'command': {
                'linear_x': float(decision.linear_x),
                'angular_z': float(decision.angular_z),
            },
            'action': {
                'selected': selected_action_int,
                'label': debug.get('action_label'),
                'native_label': debug.get('native_action_label', debug.get('action_label')),
                'effective_label': debug.get('effective_action_label', debug.get('action_label')),
                'converted_status': debug.get('converted_status'),
                'command_generation_stage': debug.get('command_generation_stage'),
                'official_discrete_selected': debug.get('official_discrete_selected'),
                'official_discrete_primitive': debug.get('official_discrete_primitive'),
                'primitive_interface': debug.get('primitive_interface'),
                'primitive_forward_speed': debug.get('primitive_forward_speed'),
                'primitive_turn_speed': debug.get('primitive_turn_speed'),
                'arc_turn': bool(debug.get('arc_turn', False)),
                'history_tail': list(self._action_history[-24:]),
                'remaining_action_queue': debug.get('remaining_action_queue'),
                'queued_action': debug.get('queued_action'),
                'queued_action_sequence_tail': debug.get('queued_action_sequence_tail'),
                'dropped_action_sequence_tail': debug.get('dropped_action_sequence_tail'),
                'official_discrete_action_tail_limit': debug.get('official_discrete_action_tail_limit'),
                'cached_selected_action': debug.get('cached_selected_action'),
            },
            'action_effect': debug.get('action_effect'),
            'goal': geometry,
            'observation': {
                'rgb_available': observation.rgb_image is not None,
                'depth_available': observation.depth_image is not None,
                'camera_info_available': observation.camera_intrinsics is not None,
                'rgb_shape': debug.get('rgb_shape'),
                'depth_shape': debug.get('depth_shape'),
                'camera_frame_id': observation.camera_frame_id,
                'sensor_ages_sec': self._to_jsonable(sensor_ages),
                'stale_after_sec': float(self._params.get('camera_stale_after_sec', 0.0)),
                'look_down': bool(observation.look_down),
            },
            'instruction': {
                'length': len(observation.instruction),
                'preview': observation.instruction[:220],
                'quality': debug.get('instruction_quality'),
                'warning': debug.get('instruction_warning'),
            },
            'timing': {
                'infer_time_sec': debug.get('infer_time_sec'),
                'subprocess_compute_sec': debug.get('subprocess_compute_sec'),
            },
            'llm': {
                'raw_output_text': debug.get('raw_output_text') or debug.get('subprocess_llm_output') or debug.get('adapter_llm_output') or debug.get('llm_output'),
                'llm_digits': debug.get('llm_digits', debug.get('digit_groups', [])),
                'digit_groups': debug.get('digit_groups', debug.get('llm_digits', [])),
                'output_mode': debug.get('model_generation_output_mode'),
                'pixel_goal': debug.get('pixel_goal') or debug.get('target_pixel'),
                'symbolic_action_seq': debug.get('symbolic_action_seq'),
            },
            'debug': debug,
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._trace_path)), exist_ok=True)
            with open(self._trace_path, 'a', encoding='utf-8') as trace_file:
                trace_file.write(json.dumps(self._to_jsonable(record), ensure_ascii=False, sort_keys=True) + '\n')
        except Exception as exc:
            self.get_logger().warn(f'Failed to append {self.SERVER_LABEL} trace record: {exc}')

    def _annotate_decision_for_diagnostics(self, observation: ModelSimObservation, decision: ModelSimDecision) -> None:
        instruction_debug = self._instruction_diagnostics(observation.instruction)
        for key, value in instruction_debug.items():
            decision.debug.setdefault(key, value)
        instruction_warning = str(instruction_debug.get('instruction_warning', '') or '')
        if instruction_warning and instruction_warning != self._last_instruction_quality_warning:
            self.get_logger().warn(instruction_warning)
            self._last_instruction_quality_warning = instruction_warning

        output_policy = str(self._params.get('model_output_policy', '') or '').strip().lower()
        trajectory_requested = output_policy in {'trajectory', 'continuous', 'continuous_trajectory', 'output_trajectory', 'traj', 'auto'}
        decision.debug.setdefault('trajectory_policy_requested', trajectory_requested)
        if trajectory_requested:
            selected_output_mode = str(decision.debug.get('selected_output_mode', '') or '').strip().lower()
            has_trajectory = any(
                key in decision.debug
                for key in ('trajectory_control_step', 'trajectory_first_step', 'trajectory_preview')
            )
            decision.debug.setdefault('trajectory_available', has_trajectory)
            if selected_output_mode and selected_output_mode != 'trajectory':
                decision.debug.setdefault('trajectory_policy_fallback', True)
                decision.debug.setdefault(
                    'trajectory_warning',
                    f"model_output_policy=trajectory but selected_output_mode={selected_output_mode}; "
                    'falling back to the available non-trajectory output.',
                )
            if not has_trajectory:
                decision.debug.setdefault('trajectory_missing_under_policy', True)
                decision.debug.setdefault(
                    'trajectory_missing_warning',
                    'model_output_policy=trajectory but adapter output did not contain output_trajectory/trajectory.',
                )

        selected_action = decision.debug.get('selected_action')
        try:
            selected_action_int = int(selected_action) if selected_action is not None else None
        except (TypeError, ValueError):
            selected_action_int = None
        if selected_action_int is not None:
            self._action_history.append(selected_action_int)
            if len(self._action_history) > 256:
                self._action_history = self._action_history[-256:]
        geometry = self._goal_geometry(observation)
        if geometry.get('goal_distance') is not None:
            decision.debug.setdefault('goal_distance', geometry.get('goal_distance'))
        if geometry.get('yaw_error') is not None:
            decision.debug.setdefault('yaw_error', geometry.get('yaw_error'))
        decision.debug.setdefault('sensor_ages_sec', self._camera_sensor_ages())
        decision.debug.setdefault('stale_after_sec', float(self._params.get('camera_stale_after_sec', 0.0)))
        decision.debug['action_history_tail'] = list(self._action_history[-12:])

        current_pose = observation.pose
        previous_pose = self._previous_pose_for_action_trace
        if current_pose is not None and previous_pose is not None:
            dx = float(current_pose.x) - float(previous_pose.x)
            dy = float(current_pose.y) - float(previous_pose.y)
            dyaw = math.atan2(
                math.sin(float(current_pose.yaw) - float(previous_pose.yaw)),
                math.cos(float(current_pose.yaw) - float(previous_pose.yaw)),
            )
            action_effect = {
                'previous_pose': [float(previous_pose.x), float(previous_pose.y), float(previous_pose.yaw)],
                'current_pose': [float(current_pose.x), float(current_pose.y), float(current_pose.yaw)],
                'delta_xy': [dx, dy],
                'delta_distance': math.hypot(dx, dy),
                'delta_yaw': dyaw,
            }
            if selected_action_int in {2, 3} and abs(dyaw) > 1e-5:
                expected_sign = 1 if selected_action_int == 2 else -1
                actual_sign = 1 if dyaw > 0.0 else -1
                action_effect['expected_yaw_sign'] = expected_sign
                action_effect['actual_yaw_sign'] = actual_sign
                action_effect['yaw_sign_matches_action'] = expected_sign == actual_sign
            decision.debug['action_effect'] = action_effect
        if current_pose is not None:
            self._previous_pose_for_action_trace = current_pose

    def _cached_model_decision_while_computing(self, observation: ModelSimObservation) -> ModelSimDecision | None:
        with self._state_lock:
            last_model_decision = deepcopy(self._last_model_decision)

        if last_model_decision is None:
            return None
        if last_model_decision.degraded:
            return None

        cached_debug = dict(last_model_decision.debug)
        cached_debug['background_compute_in_progress'] = True
        cached_debug['cached_previous_model_command'] = True
        cached_debug['cached_previous_model_status'] = last_model_decision.status
        selected_action = cached_debug.pop('selected_action', None)
        if selected_action is not None:
            cached_debug['cached_selected_action'] = selected_action
        cached_debug['sensor_ages_sec'] = self._camera_sensor_ages()
        cached_debug['stale_after_sec'] = float(self._params.get('camera_stale_after_sec', 0.0))

        geometry = self._goal_geometry(observation)
        if geometry.get('goal_distance') is not None:
            cached_debug['goal_distance'] = geometry.get('goal_distance')
        if geometry.get('yaw_error') is not None:
            cached_debug['yaw_error'] = geometry.get('yaw_error')

        return ModelSimDecision(
            linear_x=last_model_decision.linear_x,
            angular_z=last_model_decision.angular_z,
            status='inference_in_progress_cached_' + last_model_decision.status,
            degraded=True,
            debug=cached_debug,
        )

    def _decision_to_twist(self, decision: ModelSimDecision) -> Twist:
        cmd = Twist()
        cmd.linear.x = float(decision.linear_x)
        cmd.angular.z = float(decision.angular_z)
        return cmd

    def _publish_status(self, decision: ModelSimDecision) -> None:
        message = String()
        debug = self._to_jsonable(decision.debug)
        message.data = json.dumps(
            {
                'status': decision.status,
                'degraded': bool(decision.degraded),
                'linear_x': float(decision.linear_x),
                'angular_z': float(decision.angular_z),
                'llm': {
                    'raw_output_text': debug.get('raw_output_text') or debug.get('subprocess_llm_output') or debug.get('adapter_llm_output') or debug.get('llm_output'),
                    'llm_digits': debug.get('llm_digits', debug.get('digit_groups', [])),
                    'digit_groups': debug.get('digit_groups', debug.get('llm_digits', [])),
                    'output_mode': debug.get('model_generation_output_mode'),
                    'pixel_goal': debug.get('pixel_goal') or debug.get('target_pixel'),
                    'symbolic_action_seq': debug.get('symbolic_action_seq'),
                },
                'debug': debug,
                'wrapper': self.SERVER_LABEL,
                'model_instance': self.MODEL_INSTANCE,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        self._status_publisher.publish(message)

    def _publish_model_output(
        self,
        observation: Optional[ModelSimObservation],
        decision: ModelSimDecision,
        *,
        event_type: str = 'model_result',
    ) -> None:
        self._model_output_seq += 1
        debug = self._to_jsonable(decision.debug)
        selected_action = debug.get('selected_action')
        try:
            selected_action_int = int(selected_action) if selected_action is not None else None
        except (TypeError, ValueError):
            selected_action_int = None

        geometry = self._goal_geometry(observation) if observation is not None else {
            'pose': None,
            'goal': None,
            'goal_distance': debug.get('goal_distance'),
            'yaw_error': debug.get('yaw_error'),
        }
        raw_model_output = debug.get('raw_model_output')
        if raw_model_output is None:
            raw_model_output = {
                'linear_x': float(decision.linear_x),
                'angular_z': float(decision.angular_z),
                'status': decision.status,
                'degraded': bool(decision.degraded),
            }
            if selected_action_int is not None:
                raw_model_output['discrete_action'] = selected_action_int

        backend = getattr(self, '_backend', None)

        record = {
            'seq': self._model_output_seq,
            'stamp_wall_time': time.time(),
            'stamp_monotonic': time.monotonic(),
            'namespace': self.get_namespace(),
            'wrapper': self.SERVER_LABEL,
            'model_instance': self.MODEL_INSTANCE,
            'backend_type': getattr(backend, 'backend_type', ''),
            'event_type': event_type,
            'status': decision.status,
            'degraded': bool(decision.degraded),
            'raw_model_output_type': debug.get('raw_model_output_type'),
            'raw_model_output': raw_model_output,
            'converted_command': {
                'linear_x': float(decision.linear_x),
                'angular_z': float(decision.angular_z),
            },
            'action': {
                'selected': selected_action_int,
                'label': debug.get('action_label'),
                'native_label': debug.get('native_action_label', debug.get('action_label')),
                'effective_label': debug.get('effective_action_label', debug.get('action_label')),
                'converted_status': debug.get('converted_status'),
                'command_generation_stage': debug.get('command_generation_stage'),
                'official_discrete_selected': debug.get('official_discrete_selected'),
                'official_discrete_primitive': debug.get('official_discrete_primitive'),
                'primitive_interface': debug.get('primitive_interface'),
                'primitive_forward_speed': debug.get('primitive_forward_speed'),
                'primitive_turn_speed': debug.get('primitive_turn_speed'),
                'arc_turn': bool(debug.get('arc_turn', False)),
                'history_tail': list(self._action_history[-24:]),
                'remaining_action_queue': debug.get('remaining_action_queue'),
                'queued_action': debug.get('queued_action'),
                'queued_action_sequence_tail': debug.get('queued_action_sequence_tail'),
                'dropped_action_sequence_tail': debug.get('dropped_action_sequence_tail'),
                'official_discrete_action_tail_limit': debug.get('official_discrete_action_tail_limit'),
                'cached_selected_action': debug.get('cached_selected_action'),
            },
            'action_effect': debug.get('action_effect'),
            'goal': geometry,
            'observation': {
                'rgb_available': bool(observation is not None and observation.rgb_image is not None),
                'depth_available': bool(observation is not None and observation.depth_image is not None),
                'camera_info_available': bool(observation is not None and observation.camera_intrinsics is not None),
                'rgb_shape': debug.get('rgb_shape'),
                'depth_shape': debug.get('depth_shape'),
                'camera_frame_id': observation.camera_frame_id if observation is not None else '',
                'sensor_ages_sec': self._to_jsonable(self._camera_sensor_ages()),
                'stale_after_sec': float(self._params.get('camera_stale_after_sec', 0.0)),
                'look_down': bool(observation.look_down) if observation is not None else bool(debug.get('look_down', False)),
            },
            'instruction': {
                'length': len(observation.instruction) if observation is not None else None,
                'preview': observation.instruction[:220] if observation is not None else '',
                'quality': debug.get('instruction_quality'),
                'warning': debug.get('instruction_warning'),
            },
            'timing': {
                'infer_time_sec': debug.get('infer_time_sec'),
                'subprocess_compute_sec': debug.get('subprocess_compute_sec'),
            },
            'llm': {
                'raw_output_text': debug.get('raw_output_text') or debug.get('subprocess_llm_output') or debug.get('adapter_llm_output') or debug.get('llm_output'),
                'llm_digits': debug.get('llm_digits', debug.get('digit_groups', [])),
                'digit_groups': debug.get('digit_groups', debug.get('llm_digits', [])),
                'output_mode': debug.get('model_generation_output_mode'),
                'pixel_goal': debug.get('pixel_goal') or debug.get('target_pixel'),
                'symbolic_action_seq': debug.get('symbolic_action_seq'),
            },
            'debug': debug,
        }

        message = String()
        message.data = json.dumps(self._to_jsonable(record), ensure_ascii=False, sort_keys=True)
        self._model_output_publisher.publish(message)

    def _should_compute(self, now: float) -> bool:
        if getattr(self, '_backend', None) is None:
            return False
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

        self._annotate_decision_for_diagnostics(observation, decision)
        self._write_trace_record(observation, decision, event_type='model_result')

        with self._state_lock:
            self._last_decision = decision
            self._last_cmd = self._decision_to_twist(decision)
            if not decision.degraded:
                self._last_model_decision = deepcopy(decision)
            self._compute_in_progress = False

        self._publish_status(decision)
        self._publish_model_output(observation, decision, event_type='model_result')
        self._maybe_publish_visualization(observation, decision)

    def _publish_fallback_while_computing(self, observation: ModelSimObservation) -> None:
        try:
            decision = self._cached_model_decision_while_computing(observation)
            if decision is None:
                decision = self._fallback_backend.compute(observation)
                decision.status = 'inference_in_progress_cached_' + decision.status
                decision.degraded = True
                decision.debug['background_compute_in_progress'] = True
                decision.debug['cached_previous_model_command'] = False
            self._annotate_decision_for_diagnostics(observation, decision)
            self._write_trace_record(observation, decision, event_type='fallback_command')
            self._set_last_decision(decision)
            self._publish_status(decision)
            self._publish_model_output(observation, decision, event_type='fallback_command')
        except Exception as exc:
            self.get_logger().warn(f'fallback command while model inference is running failed: {exc}')

    def _maybe_publish_visualization(self, observation: ModelSimObservation, decision: ModelSimDecision) -> None:
        if not self._visualization_enabled or (
            self._visualization_publisher is None and self._action_visualization_publisher is None
        ):
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
                backend_name=getattr(self._backend, 'backend_type', ''),
            )
            image_msg = numpy_to_image_msg(image, self._latest_rgb_msg)
            if self._visualization_publisher is not None:
                self._visualization_publisher.publish(image_msg)
            if self._action_visualization_publisher is not None:
                self._action_visualization_publisher.publish(image_msg)
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
        self._learn_tf_frames_from_odom(msg)
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
        self._latest_rgb_ts = time.monotonic()

    def _on_depth(self, msg: Image) -> None:
        depth = image_msg_to_numpy(msg)
        if depth is None:
            return
        self._latest_depth = depth
        self._latest_depth_msg = msg
        self._latest_depth_ts = time.monotonic()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_intrinsics = tuple(float(value) for value in msg.k)
        self._camera_info_ts = time.monotonic()

    def _on_get_command(self, request: GetCommand.Request, response: GetCommand.Response) -> GetCommand.Response:
        del request
        now = time.monotonic()
        self.get_logger().info(f'{self.SERVER_LABEL} get_command request received')

        try:
            camera_gate_decision = self._camera_gate_decision()
            if camera_gate_decision is not None:
                observation = self._build_observation()
                self._annotate_decision_for_diagnostics(observation, camera_gate_decision)
                self._write_trace_record(observation, camera_gate_decision, event_type='camera_gate')
                self._set_last_decision(camera_gate_decision)
                self._publish_status(camera_gate_decision)
                self._publish_model_output(observation, camera_gate_decision, event_type='camera_gate')
                response.twist = self._get_last_cmd()
                self.get_logger().info(
                    f'{self.SERVER_LABEL} get_command camera gate response: '
                    f'linear={response.twist.linear.x:.3f} angular={response.twist.angular.z:.3f}'
                )
                return response
            observation = self._build_observation()
            instruction_gate_decision = self._instruction_gate_decision(observation)
            if instruction_gate_decision is not None:
                self._annotate_decision_for_diagnostics(observation, instruction_gate_decision)
                self._write_trace_record(observation, instruction_gate_decision, event_type='instruction_gate')
                self._set_last_decision(instruction_gate_decision)
                self._publish_status(instruction_gate_decision)
                self._publish_model_output(observation, instruction_gate_decision, event_type='instruction_gate')
                response.twist = self._get_last_cmd()
                self.get_logger().info(
                    f'{self.SERVER_LABEL} get_command instruction gate response: '
                    f'linear={response.twist.linear.x:.3f} angular={response.twist.angular.z:.3f}'
                )
                return response

            if getattr(self, '_backend', None) is None and not self._try_create_backend_if_possible():
                decision = self._backend_unavailable_decision()
                self._set_last_decision(decision)
                self._publish_status(decision)
                self._publish_model_output(None, decision, event_type='backend_unavailable')
                response.twist = self._get_last_cmd()
                self.get_logger().info(
                    f'{self.SERVER_LABEL} get_command backend unavailable response: '
                    f'linear={response.twist.linear.x:.3f} angular={response.twist.angular.z:.3f}'
                )
                return response
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
                    self._annotate_decision_for_diagnostics(observation, decision)
                    self._write_trace_record(observation, decision, event_type='model_result')
                    self._set_last_decision(decision)
                    self._last_compute_ts = now
                    self._publish_status(decision)
                    self._publish_model_output(observation, decision, event_type='model_result')
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
            self._publish_model_output(None, decision, event_type='exception')

        response.twist = self._get_last_cmd()
        self.get_logger().info(
            f'{self.SERVER_LABEL} get_command response: '
            f'linear={response.twist.linear.x:.3f} angular={response.twist.angular.z:.3f}'
        )
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
    DEFAULT_ACTION_VISUALIZATION_TOPIC = 'internnav/action_image'
    DEFAULT_STATUS_TOPIC = 'internnav/status'
    DEFAULT_MODEL_OUTPUT_TOPIC = 'internnav/model_output'
    COMPUTE_THREAD_NAME = 'internnav_compute_worker'

    def _requires_initial_camera(self, *, mode: str, rgb_topic: str, depth_topic: str, camera_info_topic: str) -> bool:
        mode_lower = str(mode or '').strip().lower()
        model_path_lower = str(self._params.get('model_path', '')).lower()
        adapter_lower = str(self._params.get('adapter_target', '')).lower()
        camera_topics_configured = bool(rgb_topic and depth_topic)
        internnav_like = (
            mode_lower in {'internnav', 'model'}
            or 'internnav' in adapter_lower
            or 'internvla' in adapter_lower
            or 'internnav' in model_path_lower
            or 'internvla' in model_path_lower
        )
        return camera_topics_configured and (internnav_like or bool(self._params.get('require_real_backend', False)))

    def _camera_missing_inputs(self) -> list[str]:
        missing, _stale = self._required_input_issues(require_fresh=False)
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
            'Waiting for real TF/odom/camera inputs before loading InternNav adapter: '
            f'rgb={rgb_topic}, depth={depth_topic}, camera_info={camera_info_topic}, '
            f'odom={self._required_readiness_topics.get("odom", "")}, '
            f'tf={self._required_tf_frames.get("odom", "")}->{self._required_tf_frames.get("base", "")}, '
            f'timeout={timeout_sec:.1f}s'
        )
        waiting = ModelSimDecision(
            status='waiting_for_camera',
            degraded=True,
            debug={
                'rgb_topic': rgb_topic,
                'depth_topic': depth_topic,
                'camera_info_topic': camera_info_topic,
                'odom_topic': self._required_readiness_topics.get('odom', ''),
                'topics': self._required_readiness_topics,
                'tf_frames': self._required_tf_frames,
                'tf_ready': self._tf_tree_ready(),
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
            self._initial_camera_timed_out = False
            self.get_logger().info('Initial TF/odom/camera readiness barrier passed; loading InternNav adapter now.')
            return

        self._initial_camera_timed_out = True
        failure = ModelSimDecision(
            status='camera_timeout',
            degraded=True,
            debug={
                'safe_stop': True,
                'missing_inputs': missing,
                'sensor_ages_sec': self._camera_sensor_ages(),
                'topics': self._required_readiness_topics,
                'tf_frames': self._required_tf_frames,
                'tf_ready': self._tf_tree_ready(),
                'timeout_sec': timeout_sec,
            },
        )
        self._publish_status(failure)
        self._publish_model_output(None, failure, event_type='startup_failure')
        message = (
            'Timed out waiting for required real TF/odom/camera inputs before InternNav adapter load; missing '
            + ', '.join(missing)
        )
        if self._params['require_real_backend']:
            self.get_logger().error(
                message
                + '. Keeping the server alive in safe-stop mode so it can recover to backend_ready '
                + 'once real TF/odom/camera inputs eventually arrive.'
            )
            return
        self.get_logger().warn(message + '. Continuing in degraded mode because require_real_backend=false.')

def main() -> None:
    rclpy.init()
    node = InternNavServer()
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
