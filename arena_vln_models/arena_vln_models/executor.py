"""ROS 2 executor for the unified VLN model adapters.

This is the only module in ``arena_vln_models`` that creates a
``nav2_msgs/action/FollowPath`` client or sends FollowPath goals.  It never
creates a velocity publisher or velocity message.  The velocity smoother owned by
``follow_path.launch.py`` remains the only final velocity publisher.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path as FilePath
from typing import Any, Optional

import numpy as np

from .core import ModelObservation, Pose2D
from .execution import (
    AsyncInferenceWorker,
    ExecutionConfig,
    ExecutionCore,
    ExecutionDecision,
    SessionTransition,
)
from .goal_state import IDLE, GoalStateMachine
from .model_adapter import DEFAULT_CONTROL_MODE, create_adapter, resolve_model_spec

try:  # ROS is available in the Arena container; tests only read this file.
    import rclpy
    import tf2_ros
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry, Path
    from nav2_msgs.action import FollowPath
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rclpy.time import Time
    from std_msgs.msg import Bool
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Int16, String
except Exception:  # pragma: no cover - import guard for source-only inspection
    rclpy = None
    tf2_ros = None
    GoalStatus = None
    PoseStamped = None
    Odometry = None
    Path = None
    FollowPath = None
    ActionClient = None
    Duration = None
    Node = object
    DurabilityPolicy = ReliabilityPolicy = None
    QoSProfile = None
    qos_profile_sensor_data = None
    Time = None
    CameraInfo = Image = None
    Int16 = String = None


def _yaw_from_quaternion(q: Any) -> float:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-9:
        return 0.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _row_view(data: bytes, *, height: int, width: int, channels: int, step: int) -> np.ndarray:
    row_bytes = int(width) * int(channels)
    stride = int(step) or row_bytes
    if height <= 0 or width <= 0 or stride < row_bytes:
        raise ValueError('invalid image geometry/step')
    raw = np.frombuffer(bytes(data), dtype=np.uint8, count=stride * height)
    return raw.reshape(height, stride)[:, :row_bytes].reshape(height, width, channels)


def _image_to_rgb(msg: Any) -> Optional[np.ndarray]:
    if msg is None:
        return None
    height, width = int(msg.height), int(msg.width)
    encoding = str(getattr(msg, 'encoding', 'rgb8') or 'rgb8').lower()
    step = int(getattr(msg, 'step', 0) or 0)
    if height <= 0 or width <= 0:
        return None
    try:
        if encoding in {'rgb8', 'bgr8'}:
            array = _row_view(msg.data, height=height, width=width, channels=3, step=step)
            if encoding == 'bgr8':
                array = array[:, :, ::-1]
            return array.copy()
        if encoding in {'rgba8', 'bgra8'}:
            array = _row_view(msg.data, height=height, width=width, channels=4, step=step)[:, :, :3]
            if encoding == 'bgra8':
                array = array[:, :, ::-1]
            return array.copy()
        if encoding in {'mono8', '8uc1'}:
            array = _row_view(msg.data, height=height, width=width, channels=1, step=step)
            return np.repeat(array, 3, axis=2)
    except (TypeError, ValueError):
        return None
    return None


def _depth_to_metric(msg: Any) -> Optional[np.ndarray]:
    """Decode ROS Image depth respecting ``step`` and ``is_bigendian``."""

    if msg is None:
        return None
    height, width = int(msg.height), int(msg.width)
    if height <= 0 or width <= 0:
        return None
    encoding = str(getattr(msg, 'encoding', '32FC1') or '32FC1').lower()
    step = int(getattr(msg, 'step', 0) or 0)
    big_endian = bool(getattr(msg, 'is_bigendian', 0))
    try:
        if encoding in {'32fc1', '64fc1'}:
            itemsize = 4 if encoding == '32fc1' else 8
            stride = step or width * itemsize
            if stride < width * itemsize:
                return None
            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8, count=stride * height)
            rows = raw.reshape(height, stride)[:, :width * itemsize]
            dtype = '<f4' if itemsize == 4 else '<f8'
            array = np.frombuffer(rows.tobytes(), dtype=dtype).reshape(height, width)
            if big_endian:
                array = array.byteswap()
            return array.astype(np.float32)
        if encoding in {'16uc1', 'mono16'}:
            stride = step or width * 2
            if stride < width * 2:
                return None
            raw = np.frombuffer(bytes(msg.data), dtype=np.uint8, count=stride * height)
            rows = raw.reshape(height, stride)[:, :width * 2]
            array = np.frombuffer(rows.tobytes(), dtype='<u2').reshape(height, width)
            if big_endian:
                array = array.byteswap()
            return array.astype(np.float32) / 1000.0
    except (TypeError, ValueError):
        return None
    return None


def grounded_goal_identity(decision: ExecutionDecision) -> Optional[tuple[str, int]]:
    """Return the grounded map identity, or None for non-LISN decisions."""
    if decision.debug.get('source') != 'lisn_grounding_live':
        return None
    frame_id = str(decision.debug.get('frame_id') or '')
    stamp_ns = int(decision.debug.get('stamp_ns') or 0)
    if frame_id != 'map' or stamp_ns <= 0:
        raise ValueError('lisn_grounded_goal_missing_map_identity')
    return frame_id, stamp_ns


def grounded_goal_decision(msg: Any) -> ExecutionDecision:
    """Convert one grounded PoseStamped into the common FollowPath decision."""
    header = getattr(msg, 'header', None)
    pose = getattr(msg, 'pose', None)
    position = getattr(pose, 'position', None)
    if header is None or position is None:
        raise ValueError('grounded_goal_missing_header_or_pose')
    frame_id = str(getattr(header, 'frame_id', '') or '')
    if frame_id != 'map':
        raise ValueError(f'grounded_goal_frame_must_be_map:{frame_id}')
    x = float(position.x)
    y = float(position.y)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError('grounded_goal_not_finite')
    stamp_ns = int(getattr(header.stamp, 'sec', 0)) * 1_000_000_000 + int(
        getattr(header.stamp, 'nanosec', 0)
    )
    if stamp_ns <= 0:
        raise ValueError('grounded_goal_stamp_missing')
    return ExecutionDecision(
        action='follow_path',
        status='running',
        reason='lisn_grounding_live',
        poses=[(x, y, 0.0)],
        debug={'source': 'lisn_grounding_live', 'frame_id': 'map', 'stamp_ns': stamp_ns},
    )


class ArenaModelExecutor(Node):
    """One episode-aware model executor; FollowPath is its only output."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError('rclpy is unavailable; ArenaModelExecutor requires the Arena ROS environment')
        super().__init__('arena_model_executor')
        self.declare_parameter('model_adapter', '')
        self.declare_parameter('control_mode', DEFAULT_CONTROL_MODE)
        self.declare_parameter('task_namespace', '/task_generator_node')
        self.declare_parameter('rgb_topic', 'head_camera/image')
        self.declare_parameter('depth_topic', 'head_camera/depth')
        self.declare_parameter('camera_info_topic', 'head_camera/camera_info')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('eval_ready_topic', '')
        self.declare_parameter('task_reset_topic', '')
        self.declare_parameter('instruction_topic', '')
        self.declare_parameter('model_status_topic', 'model/status')
        self.declare_parameter('follow_path_action', 'follow_path')
        self.declare_parameter('inference_hz', 4.0)
        self.declare_parameter('request_timeout_sec', 10.0)
        self.declare_parameter('observation_max_age_sec', 2.0)
        self.declare_parameter('server_url', '')
        self.declare_parameter('trace_dir', '')
        self.declare_parameter('model_params_json', '{}')
        self.declare_parameter('cancel_timeout_sec', 5.0)
        self.declare_parameter('odom_frame', '')
        self.declare_parameter('base_frame', '')
        self.declare_parameter('use_tf_exact_stamp', False)
        self.declare_parameter('lisn_grounding_live', False)
        self.declare_parameter('lisn_grounded_goal_topic', '')
        # Strict RGB-only B fast guard.  Empty topic keeps the existing A/B
        # behavior unchanged; the guard node publishes Bool vetoes.
        self.declare_parameter('fast_guard_veto_topic', '')
        self.declare_parameter('fast_guard_veto_timeout_sec', 1.0)
        self.declare_parameter('fast_guard_required', False)

        model = str(self.get_parameter('model_adapter').value or '').strip().lower()
        if not model:
            raise RuntimeError('model_adapter parameter is required')
        resolve_model_spec(model)
        control_mode = str(self.get_parameter('control_mode').value or DEFAULT_CONTROL_MODE)
        if control_mode != DEFAULT_CONTROL_MODE:
            raise RuntimeError(
                f"unified ArenaModelExecutor only supports control_mode='{DEFAULT_CONTROL_MODE}'; "
                f"got '{control_mode}'. Native velocity belongs to the isolated legacy entry."
            )
        params = self._adapter_params(model)
        self.adapter = create_adapter(model, params=params, logger=self.get_logger())
        doctor = getattr(self.adapter, 'doctor', None)
        if callable(doctor):
            doctor()
        self.core = ExecutionCore(self.adapter, ExecutionConfig(
            model=model,
            control_mode=control_mode,
            inference_hz=float(self.get_parameter('inference_hz').value),
            request_timeout_sec=float(self.get_parameter('request_timeout_sec').value),
            observation_max_age_sec=float(self.get_parameter('observation_max_age_sec').value),
        ))
        self.worker = AsyncInferenceWorker(self.core)
        self.goals = GoalStateMachine(
            cancel_timeout_sec=float(self.get_parameter('cancel_timeout_sec').value)
        )
        self._session_transition = SessionTransition()
        self._odom_frame = str(self.get_parameter('odom_frame').value or '').strip()
        self._base_frame = str(self.get_parameter('base_frame').value or '').strip()
        self._use_tf = bool(self.get_parameter('use_tf_exact_stamp').value)
        self._tf_buffer = None
        self._tf_listener = None
        if self._use_tf:
            if tf2_ros is None:
                raise RuntimeError('tf2_ros is unavailable; cannot guarantee exact-stamp pose')
            self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=15.0))
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.trace_path = self._trace_path()

        task_ns = str(self.get_parameter('task_namespace').value).rstrip('/')
        eval_ready = str(self.get_parameter('eval_ready_topic').value or f'{task_ns}/eval_ready')
        task_reset = str(self.get_parameter('task_reset_topic').value or f'{task_ns}/task_reset')
        instruction = str(self.get_parameter('instruction_topic').value or f'{task_ns}/vln_instruction')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, eval_ready, self._on_episode, latched)
        self.create_subscription(Int16, task_reset, self._on_reset, latched)
        self.create_subscription(String, instruction, self._on_instruction, 10)
        self.create_subscription(Image, str(self.get_parameter('rgb_topic').value), self._on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, str(self.get_parameter('depth_topic').value), self._on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, str(self.get_parameter('camera_info_topic').value),
                                 self._on_camera_info, qos_profile_sensor_data)
        self._lisn_grounding_live = bool(self.get_parameter('lisn_grounding_live').value)
        self._lisn_pending_goal = None
        if self._lisn_grounding_live:
            topic = str(self.get_parameter('lisn_grounded_goal_topic').value or '').strip()
            if not topic:
                raise RuntimeError('lisn_grounding_live requires lisn_grounded_goal_topic')
            if PoseStamped is None:
                raise RuntimeError('PoseStamped is unavailable; cannot consume grounded goals')
            self.create_subscription(PoseStamped, topic, self._on_lisn_grounded_goal, 10)
        self.create_subscription(Odometry, str(self.get_parameter('odom_topic').value), self._on_odom,
                                 qos_profile_sensor_data)
        self._fast_guard_veto = False
        self._fast_guard_veto_at = float('-inf')
        self._fast_guard_required = bool(self.get_parameter('fast_guard_required').value)
        self._fast_guard_veto_topic = str(
            self.get_parameter('fast_guard_veto_topic').value or '').strip()
        if self._fast_guard_veto_topic:
            self.create_subscription(Bool, self._fast_guard_veto_topic,
                                     self._on_fast_guard_veto, 10)
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter('model_status_topic').value), 10
        )
        # The task_generator holds the episode until the harness-owned
        # pre-episode gate publishes its session-scoped READY.  In the unified
        # in-process path this executor is that harness client, so it must
        # publish the gate; otherwise the barrier can never be satisfied.
        self.pre_episode_gate_pub = self.create_publisher(
            Bool, '/task_generator_node/pre_episode_gate_ready',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._pre_episode_gate_ready = False
        self.action = ActionClient(
            self, FollowPath, str(self.get_parameter('follow_path_action').value)
        )
        self._handles: dict[int, Any] = {}
        self._image = None
        self._image_stamp_ns = 0
        self._image_frame_id = ''
        self._image_at = float('-inf')
        self._depth = None
        self._depth_stamp_ns = 0
        self._camera_info = None
        self._camera_info_stamp_ns = 0
        # Bounded exact-stamp RGB/depth pairing.  RGB and depth arrive on
        # separate subscriptions; a timer tick may fall in the arrival gap.
        self._rgb_buffer: dict[int, tuple[Any, float, str]] = {}
        self._depth_buffer: dict[int, tuple[Any, float]] = {}
        self._sync_hold_reason = ''
        self._odom_pose: Optional[Pose2D] = None
        self._odom_frame_id = ''
        self._odom_stamp_ns = 0
        self._odom_at = float('-inf')
        self.create_timer(0.05, self._tick)
        self.create_timer(1.0, self._heartbeat)
        self._record('executor_started', model=model, adapter=self.adapter.capability.name)

    def _adapter_params(self, model: str) -> dict[str, Any]:
        try:
            params = json.loads(str(self.get_parameter('model_params_json').value or '{}'))
        except (TypeError, ValueError):
            params = {}
        if not isinstance(params, dict):
            params = {}
        server_url = str(self.get_parameter('server_url').value or '').strip()
        if server_url:
            params['server_url'] = server_url
        params.setdefault('request_timeout_sec', float(self.get_parameter('request_timeout_sec').value))
        spec = resolve_model_spec(model)
        if spec.server_default and not params.get('server_url'):
            params['server_url'] = spec.server_default
        return params

    def _trace_path(self) -> Optional[FilePath]:
        trace_dir = str(self.get_parameter('trace_dir').value or '').strip()
        if not trace_dir:
            return None
        root = FilePath(trace_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root / f'executor_{time.time_ns()}_{self.get_name()}.jsonl'

    def _record(self, event: str, **fields: Any) -> None:
        if self.trace_path is None:
            return
        payload = {
            'event': event,
            'wall_time': time.time(),
            'session_id': self.core.session_id,
            'episode': self.core.episode,
            'status': self.core.status,
            'reason': self.core.reason,
            **fields,
        }
        try:
            with self.trace_path.open('a') as stream:
                stream.write(json.dumps(payload, default=str) + '\n')
        except Exception:
            pass

    def _publish_status(self, status: str, reason: str = '') -> None:
        if not self.core.session_id:
            return
        payload = {
            'session_id': self.core.session_id,
            'episode': self.core.episode,
            'status': status,
        }
        if reason:
            payload['reason'] = reason
        self.status_pub.publish(String(data=json.dumps(payload)))

    def _publish_pre_episode_gate(self, ready: bool) -> None:
        """Publish the session-scoped harness gate the barrier waits on."""
        ready = bool(ready)
        if ready == self._pre_episode_gate_ready:
            return
        self._pre_episode_gate_ready = ready
        self.pre_episode_gate_pub.publish(Bool(data=ready))

    def _heartbeat(self) -> None:
        if self.core.session_id:
            self._publish_status(self.core.status, self.core.reason)

    def _on_episode(self, msg: Any) -> None:
        try:
            data = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if (not isinstance(data, dict) or data.get('stage') != 'episode'
                or not isinstance(data.get('session_id'), str) or not data['session_id']
                or type(data.get('episode')) is not int or type(data.get('ready')) is not bool):
            return
        identity = (data['session_id'], data['episode'])
        if identity != (self.core.session_id, self.core.episode):
            if self._session_transition.pending != identity:
                self._session_transition.request(identity)
                self._cancel_all('new_session')
                self.worker.invalidate()
                self.core.set_hold('session_transition')
                self._publish_status('hold', 'session_transition')
                self._record('session_transition_pending', session_id=identity[0], episode=identity[1])
            # The new session is not active yet.  Never reset the core/adapter or
            # the goal state machine while the old goal may still execute or the
            # old inference is using the shared transport/history.  The timer
            # completes the transition after both are quiescent.
            return
        if data['ready']:
            self.core.release()
            self.core.status = 'ready'
            self.core.reason = ''
            self._record('episode_released')
            self._publish_status('ready')
        elif self.core.released:
            self.core.set_hold('episode_not_ready')
            self._cancel_goal('episode_not_ready')
            self._publish_status('hold', 'episode_not_ready')

    def _on_reset(self, msg: Any) -> None:
        if not self.core.session_id:
            return
        try:
            episode = int(msg.data)
        except (TypeError, ValueError):
            return
        if episode > self.core.episode and self.core.released:
            self.worker.invalidate()
            self._lisn_pending_goal = None
            self._cancel_all('task_reset')
            self.core.released = False
            self.core.set_hold('task_reset')
            self._publish_status('hold', 'task_reset')
            self._publish_pre_episode_gate(False)

    def _on_instruction(self, msg: Any) -> None:
        if not self.core.released or self.core.terminal:
            return
        self.core.set_instruction(str(msg.data or ''))
        self._record('instruction_received', instruction_preview=str(msg.data or '')[:160])

    @staticmethod
    def _remember(buffer: dict, stamp: int, value: Any, *, limit: int = 16) -> None:
        buffer[int(stamp)] = value
        if len(buffer) > limit:
            for old in sorted(buffer)[:-limit]:
                buffer.pop(old, None)

    def _clear_observation_buffers(self) -> None:
        self._image = None
        self._depth = None
        self._camera_info = None
        self._image_stamp_ns = 0
        self._depth_stamp_ns = 0
        self._camera_info_stamp_ns = 0
        self._rgb_buffer.clear()
        self._depth_buffer.clear()

    def _on_image(self, msg: Any) -> None:
        image = _image_to_rgb(msg)
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if image is None or stamp <= 0:
            return
        self._remember(self._rgb_buffer, stamp, (image, time.monotonic(), str(msg.header.frame_id or '')))
        self._image = image
        self._image_stamp_ns = stamp
        self._image_frame_id = str(msg.header.frame_id or '')
        self._image_at = time.monotonic()

    def _on_depth(self, msg: Any) -> None:
        depth = _depth_to_metric(msg)
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if depth is None or stamp <= 0:
            return
        self._remember(self._depth_buffer, stamp, (depth, time.monotonic()))
        self._depth = depth
        self._depth_stamp_ns = stamp

    def _on_camera_info(self, msg: Any) -> None:
        self._camera_info = tuple(float(value) for value in msg.k)
        self._camera_info_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _on_lisn_grounded_goal(self, msg: Any) -> None:
        """Queue one grounded map goal for the common FollowPath sender."""
        try:
            self._lisn_pending_goal = grounded_goal_decision(msg)
        except (TypeError, ValueError) as exc:
            self._record('lisn_grounded_goal_rejected', reason=str(exc))
            return
        self._record('lisn_grounded_goal_received',
                     poses=self._lisn_pending_goal.poses,
                     stamp_ns=self._lisn_pending_goal.debug.get('stamp_ns'))

    def _on_odom(self, msg: Any) -> None:
        pose = msg.pose.pose
        self._odom_pose = Pose2D(
            float(pose.position.x), float(pose.position.y), _yaw_from_quaternion(pose.orientation)
        )
        self._odom_frame_id = str(msg.header.frame_id or '')
        self._odom_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self._odom_at = time.monotonic()

    def _pose_at_stamp(self, stamp_ns: int) -> Optional[Pose2D]:
        if not self._use_tf:
            if self._odom_pose is None or self._odom_stamp_ns != stamp_ns:
                return None
            return self._odom_pose
        if not self._odom_frame or not self._base_frame or self._tf_buffer is None:
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                self._odom_frame,
                self._base_frame,
                Time(nanoseconds=stamp_ns),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return Pose2D(float(translation.x), float(translation.y), _yaw_from_quaternion(rotation))

    def _map_pose_at_stamp(self, stamp_ns: int) -> Optional[Pose2D]:
        """Robot pose in ``map`` for a path start.

        ``stamp_ns <= 0`` means the latest transform: this is control state
        for the controller, not the source-observation identity.  The grounded
        goal pose itself always keeps its immutable source stamp.
        """
        if self._tf_buffer is None or not self._base_frame:
            return None
        try:
            when = Time() if int(stamp_ns) <= 0 else Time(nanoseconds=int(stamp_ns))
            transform = self._tf_buffer.lookup_transform(
                'map',
                self._base_frame,
                when,
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return Pose2D(float(translation.x), float(translation.y),
                      _yaw_from_quaternion(rotation))

    def _observation(self) -> Optional[ModelObservation]:
        self._sync_hold_reason = ''
        requires_depth = 'depth' in tuple(getattr(self.adapter.capability, 'required_inputs', ()))
        if requires_depth:
            common = sorted(set(self._rgb_buffer) & set(self._depth_buffer))
            if not common:
                self._sync_hold_reason = 'waiting_for_synchronized_rgb_depth'
                return None
            stamp = common[-1]
        else:
            if not self._rgb_buffer:
                self._sync_hold_reason = 'waiting_for_observation'
                return None
            stamp = max(self._rgb_buffer)
        image, image_at, _image_frame = self._rgb_buffer[stamp]
        depth = self._depth_buffer.get(stamp, (None, None))[0]
        pose = self._pose_at_stamp(stamp)
        if pose is None:
            self._sync_hold_reason = 'waiting_for_exact_stamp_pose'
            return None
        ci_stamp = self._camera_info_stamp_ns
        intrinsics = None
        if self._camera_info is not None and (
            ci_stamp == 0 or (ci_stamp <= stamp and stamp - ci_stamp <= 1_000_000_000)
        ):
            intrinsics = self._camera_info
        frame = self._odom_frame or self._odom_frame_id
        return ModelObservation(
            pose=pose,
            instruction=self.core.instruction,
            rgb_image=image,
            depth_image=depth,
            camera_intrinsics=intrinsics,
            frame_id=frame,
            stamp_ns=stamp,
            received_at=max(image_at, self._odom_at),
            session_id=self.core.session_id,
            episode=self.core.episode,
            metadata={
                'image_frame_id': self._image_frame_id,
                'odom_frame_id': frame,
                'depth_available': depth is not None,
                'camera_info_available': intrinsics is not None,
            },
        )

    def _maybe_complete_session_transition(self) -> None:
        if self._session_transition.pending is None:
            return
        if not self._session_transition.can_complete(
            worker_idle=not self.worker.in_flight,
            goals_state=self.goals.state,
        ):
            return
        identity = self._session_transition.complete()
        if identity is None:
            return
        # The old goal is terminal and the old inference is idle: it is now safe
        # to reset the shared core/adapter and start the pending session.
        self.core.reset(*identity)
        self.goals.reset(identity[0])
        self._clear_observation_buffers()
        self._odom_pose = None
        self._record('session_started', session_id=identity[0], episode=identity[1])
        self._publish_status('hold', 'waiting_for_release')
        self._publish_pre_episode_gate(True)

    def _tick(self) -> None:
        now = time.monotonic()
        self._maybe_complete_session_transition()
        actions = self.goals.cancel_timeout(now)
        if actions:
            self._execute_goal_actions(actions)
        if self._lisn_grounding_live:
            # The LISN live VLM is the sole goal source; the common FollowPath
            # sender remains the only consumer.  Do not run the fast model in
            # parallel, which would create a second goal producer.
            # released=True means the episode is active, so only a terminal
            # episode, an unreleased one, or a pending session transition may
            # discard the pending grounded goal.
            if (
                self.core.terminal
                or not self.core.released
                or self._session_transition.pending is not None
            ):
                self._lisn_pending_goal = None
                return
            if self._lisn_pending_goal is not None and not self.worker.in_flight:
                decision, self._lisn_pending_goal = self._lisn_pending_goal, None
                # GoalStateMachine.request() keeps the latest path pending and
                # cancels the active one, so a new slow-loop goal replaces a
                # still-running path instead of waiting for it to finish.
                self._handle_decision(decision)
            return
        decision = self.worker.harvest()
        if decision is not None:
            self._handle_decision(decision)
        if self.worker.in_flight:
            return
        observation = self._observation()
        if observation is None:
            if self.core.released and not self.core.terminal:
                self.core.set_hold(self._sync_hold_reason or 'waiting_for_exact_stamp_pose')
            return
        eligible, reason = self.core.eligible(observation, now)
        if not eligible:
            if reason in {'waiting_for_observation', 'stale_observation', 'waiting_for_instruction'}:
                self.core.set_hold(reason)
                self._publish_status('hold', reason)
            return
        if self.worker.submit(observation, now):
            self._record('inference_submitted', stamp_ns=observation.stamp_ns)

    def _on_fast_guard_veto(self, msg: Any) -> None:
        self._fast_guard_veto = bool(msg.data)
        self._fast_guard_veto_at = time.monotonic()
        self._record('bfast_veto_state', veto=self._fast_guard_veto,
                     topic=self._fast_guard_veto_topic)

    def _fast_guard_veto_reason(self) -> str:
        if not self._fast_guard_veto_topic:
            return ''
        timeout = float(self.get_parameter('fast_guard_veto_timeout_sec').value)
        age = time.monotonic() - self._fast_guard_veto_at
        if self._fast_guard_veto and age <= timeout:
            return 'bfast_veto'
        if self._fast_guard_required and age > timeout:
            return 'bfast_guard_stale'
        return ''

    def _fast_guard_veto_active(self) -> bool:
        return bool(self._fast_guard_veto_reason())

    def _handle_decision(self, decision: ExecutionDecision) -> None:
        self._record('decision', decision=decision.to_manifest())
        guard_reason = self._fast_guard_veto_reason()
        if decision.action == 'follow_path' and guard_reason:
            self.core.set_stop(guard_reason)
            self._record('bfast_veto', reason=guard_reason,
                         decision=decision.to_manifest())
            self._publish_status('stop', guard_reason)
            return
        if decision.action == 'follow_path':
            self._request_goal(decision)
            self._publish_status('running')
        elif decision.action == 'stop':
            self.worker.invalidate()
            self._cancel_all('model_stop')
            self._publish_status('stop', 'model_stop')
        elif decision.action == 'error':
            self.worker.invalidate()
            self._cancel_all(decision.error_type or 'error')
            self._publish_status('error', decision.reason)
        elif decision.action == 'hold':
            self._publish_status('hold', decision.reason)

    def _cancel_all(self, reason: str) -> None:
        actions = self.goals.stop()
        self._execute_goal_actions(actions)
        self._record('goal_cancel_all', reason=reason, actions=actions)

    def _request_goal(self, decision: ExecutionDecision) -> None:
        try:
            identity = grounded_goal_identity(decision)
        except ValueError as exc:
            self._record('goal_rejected', reason=str(exc))
            return
        poses = list(decision.poses)
        if identity is not None:
            # Grounded map coordinates carry their own frame/stamp; never
            # rebind them to a fresh odom observation.
            frame_id, stamp_ns = identity
            if len(poses) == 1:
                # SocialMPC refuses to trajectorize a single-pose path.  The
                # first pose is the robot's current map pose (control state),
                # not a fresh source observation and not an eval ground-truth
                # goal.  The goal pose keeps its immutable source stamp.
                start = self._map_pose_at_stamp(0)
                if start is None:
                    self._record('goal_rejected',
                                 reason='grounded_goal_start_pose_unavailable')
                    return
                goal_x, goal_y = poses[0][0], poses[0][1]
                # A VLM point has no heading.  Represent the path as a straight
                # segment with the path direction on both poses.  Using the
                # robot's current yaw on the first pose made SocialMPC spend
                # the whole 10s slow-loop window rotating instead of moving.
                heading = math.atan2(goal_y - start.y, goal_x - start.x)
                poses = [(start.x, start.y, heading), (goal_x, goal_y, heading)]
        else:
            observation = self._observation()
            if observation is None:
                return
            frame_id = observation.frame_id
            stamp_ns = observation.stamp_ns
        goal = self.goals.next_goal(
            poses=poses,
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            metadata={
                'source': decision.prediction.source if decision.prediction else '',
                'adapter': self.adapter.name,
            },
        )
        actions = self.goals.request(goal, time.monotonic())
        self._execute_goal_actions(actions)

    def _execute_goal_actions(self, actions: list[str]) -> None:
        for action in actions:
            if action.startswith('send:'):
                self._send_goal(int(action.split(':', 1)[1]))
            elif action.startswith('cancel:'):
                self._cancel_goal(int(action.split(':', 1)[1]))
            elif action.startswith('error:'):
                parts = action.split(':')
                reason = parts[1] if len(parts) > 1 else 'goal_error'
                serial = parts[2] if len(parts) > 2 else ''
                self.core.set_error(reason, error_type=reason)
                self._publish_status('error', self.core.reason)
                self._record('goal_error', action=action, serial=serial)

    def _send_goal(self, serial: int) -> None:
        if not self.action.server_is_ready():
            self.core.set_hold('follow_path_server_unavailable')
            self._publish_status('hold', 'follow_path_server_unavailable')
            return
        goal = self.goals.active
        if goal is None or goal.serial != serial:
            return
        path = Path()
        path.header.frame_id = goal.frame_id
        path.header.stamp.sec = goal.stamp_ns // 1_000_000_000
        path.header.stamp.nanosec = goal.stamp_ns % 1_000_000_000
        for x, y, yaw in goal.poses:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        follow_goal = FollowPath.Goal()
        follow_goal.path = path
        follow_goal.controller_id = 'FollowPath'
        follow_goal.goal_checker_id = 'goal_checker'
        follow_goal.progress_checker_id = 'progress_checker'
        self._record('follow_path_goal', goal_serial=serial, poses=len(path.poses))
        future = self.action.send_goal_async(follow_goal)
        future.add_done_callback(lambda completed, serial=serial: self._goal_response(serial, completed))

    def _goal_response(self, serial: int, future: Any) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._record('goal_response_error', serial=serial, error=str(exc))
            return
        accepted = goal_handle is not None and bool(goal_handle.accepted)
        actions = self.goals.goal_response(serial, accepted)
        if accepted:
            self._handles[serial] = goal_handle
            self._record('goal_accepted', serial=serial, goal_uuid=bytes(goal_handle.goal_id.uuid).hex())
            goal_handle.get_result_async().add_done_callback(
                lambda completed, serial=serial: self._goal_result(serial, completed)
            )
        else:
            self._record('goal_rejected', serial=serial)
        self._execute_goal_actions(actions)

    def _goal_result(self, serial: int, future: Any) -> None:
        try:
            wrapped = future.result()
            status = wrapped.status
        except Exception as exc:
            self._record('goal_result_error', serial=serial, error=str(exc))
            return
        self._handles.pop(serial, None)
        actions = self.goals.goal_result(serial, status)
        self._record('goal_result', serial=serial, status=int(status), actions=actions)
        if status == GoalStatus.STATUS_SUCCEEDED and not self.core.terminal:
            self.core.status = 'running'
            self.core.reason = ''
            self._publish_status('running')
        elif status in {GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED} and not self.core.terminal:
            self.core.set_hold(f'follow_path_{status}')
            self._publish_status('hold', self.core.reason)
        self._execute_goal_actions(actions)

    def _cancel_goal(self, serial: int) -> None:
        handle = self._handles.get(serial)
        if handle is None:
            return
        self._record('goal_cancel_requested', serial=serial)
        future = handle.cancel_goal_async()
        future.add_done_callback(lambda completed, serial=serial: self._cancel_response(serial, completed))

    def _cancel_response(self, serial: int, future: Any) -> None:
        try:
            future.result()
        except Exception as exc:
            self._record('goal_cancel_error', serial=serial, error=str(exc))
        self._handles.pop(serial, None)
        actions = self.goals.cancel_response(serial, time.monotonic())
        self._record('goal_cancel_response', serial=serial, actions=actions)
        self._execute_goal_actions(actions)

    def destroy_node(self) -> None:
        self.worker.shutdown()
        super().destroy_node()


def main(argv: Optional[list[str]] = None) -> int:
    if rclpy is None:
        raise SystemExit('rclpy is unavailable; run ArenaModelExecutor inside the Arena container')
    rclpy.init(args=argv)
    node = None
    try:
        node = ArenaModelExecutor()
        rclpy.spin(node)
        return 0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
