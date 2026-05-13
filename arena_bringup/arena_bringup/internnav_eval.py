import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime

import yaml
from ament_index_python.packages import get_package_share_directory


def _write_yaml(path: str, data) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_text(path: str, data: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


def _copy_if_exists(src: str, dst: str) -> str | None:
    if not src or not os.path.exists(src):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _workspace_root_from_share(package_share_dir: str) -> str:
    """Return the Arena workspace root used for human-facing eval outputs.

    ROS package share directories live under ``<ws>/install/<pkg>/share/<pkg>``.
    Historical Arena evaluation code used that share path as its data root, which
    made generated videos appear inside ``install/``.  Eval artifacts are user
    outputs, not installed package resources, so default them to ``<ws>/outputs``.
    """
    for env_name in ('ARENA_OUTPUT_WORKSPACE', 'ARENA_WS_DIR', 'HOST_ARENA_WS_DIR', 'COLCON_PREFIX_PATH'):
        value = os.environ.get(env_name, '').strip()
        if not value:
            continue
        candidate = value.split(os.pathsep)[0] if env_name == 'COLCON_PREFIX_PATH' else value
        if env_name == 'COLCON_PREFIX_PATH' and os.path.basename(candidate) == 'install':
            candidate = os.path.dirname(candidate)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    marker = os.path.join(os.sep, 'install', 'arena_evaluation', 'share', 'arena_evaluation')
    abs_share = os.path.abspath(package_share_dir)
    if abs_share.endswith(marker):
        return abs_share[:-len(marker)] or os.path.sep

    return os.getcwd()


def _resolve_output_root(output_root_arg: str, package_share_dir: str) -> str:
    if output_root_arg:
        root = output_root_arg
        if not os.path.isabs(root):
            root = os.path.join(_workspace_root_from_share(package_share_dir), root)
        return os.path.abspath(root)

    return os.path.join(_workspace_root_from_share(package_share_dir), 'outputs')


def _first_env_value(*names: str) -> tuple[str, str]:
    for name in names:
        value = str(os.environ.get(name, '')).strip()
        if value:
            return value, name
    return '', ''


def _eval_python_executable(env: dict[str, str]) -> str:
    for candidate in (
        str(env.get('ARENA_EVAL_PYTHON', '')).strip(),
        str(sys.executable).strip(),
        '/usr/bin/python3',
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return '/usr/bin/python3'


def _start_finished_watcher(
    env: dict[str, str],
    topic: str,
    task_reset_topic: str,
    scenario_reset_topic: str,
) -> subprocess.Popen:
    python_bin = _eval_python_executable(env)
    watcher_code = r'''
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Empty
from std_msgs.msg import Int16

topic = sys.argv[1]
task_reset_topic = sys.argv[2]
scenario_reset_topic = sys.argv[3]
rclpy.init()
node = Node('internnav_eval_finished_watcher')
qos = QoSProfile(depth=1)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
done = {'seen': False, 'reset_seen': False}

def _cb(_msg):
    if done['reset_seen']:
        done['seen'] = True

def _on_reset(_msg):
    done['reset_seen'] = True

node.create_subscription(Empty, topic, _cb, qos)
node.create_subscription(Int16, task_reset_topic, _on_reset, 10)
node.create_subscription(Int16, scenario_reset_topic, _on_reset, 10)
try:
    while rclpy.ok() and not done['seen']:
        rclpy.spin_once(node, timeout_sec=0.5)
finally:
    node.destroy_node()
    rclpy.shutdown()

raise SystemExit(0 if done['seen'] else 1)
'''

    return subprocess.Popen(
        [python_bin, '-c', watcher_code, topic, task_reset_topic, scenario_reset_topic],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _start_status_watcher(env: dict[str, str], topic: str, output_path: str) -> subprocess.Popen:
    python_bin = _eval_python_executable(env)
    watcher_code = r'''
import json
import sys
import rclpy
from pathlib import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

topic = sys.argv[1]
output_path = Path(sys.argv[2])
rclpy.init()
node = Node('internnav_eval_status_watcher')
qos = QoSProfile(depth=1)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
state = {'last': None}

def _cb(msg):
    state['last'] = msg.data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(msg.data, encoding='utf-8')

node.create_subscription(String, topic, _cb, qos)
try:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()

raise SystemExit(0)
'''

    return subprocess.Popen(
        [python_bin, '-c', watcher_code, topic, output_path],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _read_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _read_text_if_exists(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def _is_internnav_run(args) -> bool:
    adapter_target = str(args.dual_vln_adapter_target or '').lower()
    mode = str(args.dual_vln_mode or '').lower()
    model_path = str(args.dual_vln_model_path or '').lower()
    return (
        'internnav' in adapter_target
        or mode == 'internnav'
        or ('internvla' in model_path or 'internnav' in model_path)
    )


def _default_vision_topics(robot: str) -> tuple[str, str, str] | None:
    normalized = str(robot or '').strip().lower()
    if normalized == 'turtlebot':
        return ('rgbd_camera/image', 'rgbd_camera/depth_image', 'rgbd_camera/camera_info')
    if normalized == 'ai2_bot2':
        return ('head_camera/image', 'head_camera/depth', 'head_camera/camera_info')
    if normalized == 'linkhou_s2':
        return ('head_camera/image', 'head_camera/depth', 'head_camera/camera_info')
    return None


def _default_eval_video_sim_top_down_topic(robot: str) -> str:
    normalized = str(robot or '').strip().lower()
    if normalized in {'turtlebot', 'ai2_bot2', 'linkhou_s2'}:
        return 'top_down_camera/image'
    return ''


def _default_eval_video_debug_overlay_topic() -> str:
    return 'internnav/debug_image'


def _task_root_from_topic(topic: str) -> str:
    normalized = '/' + str(topic or '').strip().strip('/')
    if normalized.endswith('/task_reset'):
        return normalized[: -len('/task_reset')]
    if normalized.endswith('/finished'):
        return normalized[: -len('/finished')]
    return normalized.rstrip('/')


def _robot_topic(task_reset_topic: str, robot: str, topic: str) -> str:
    normalized_topic = str(topic or '').strip()
    if not normalized_topic:
        return ''
    if normalized_topic.startswith('/'):
        return normalized_topic
    root = _task_root_from_topic(task_reset_topic)
    return f'{root}/{robot}/{normalized_topic.strip("/")}'


def _scenario_reset_topic(task_reset_topic: str, robot: str) -> str:
    return _robot_topic(task_reset_topic, robot, 'scenario_reset')


def _world_map_yaml_path(sim_setup_share: str, world: str) -> str:
    return os.path.join(sim_setup_share, 'worlds', world, 'map', 'map.yaml')


def _start_eval_video_recorder(
    env: dict[str, str],
    *,
    output_dir: str,
    map_yaml_path: str,
    task_reset_topic: str,
    scenario_reset_topic: str,
    finished_topic: str,
    ego_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    debug_overlay_topic: str,
    sim_top_down_topic: str,
    odom_topic: str,
    goal_topic: str,
    scan_topic: str,
    fps: float,
    top_down_size_px: int,
    top_down_window_m: float,
) -> subprocess.Popen:
    python_bin = _eval_python_executable(env)
    recorder_code = r'''
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from PIL import ImageDraw
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Empty, Int16


try:
    _PIL_BILINEAR = PILImage.Resampling.BILINEAR
except AttributeError:
    _PIL_BILINEAR = PILImage.BILINEAR


def _load_video_backend():
    try:
        import imageio.v2 as imageio  # type: ignore
        return 'imageio', imageio
    except Exception:
        pass
    try:
        import imageio  # type: ignore
        return 'imageio', imageio
    except Exception:
        pass
    try:
        import cv2  # type: ignore
        return 'cv2', cv2
    except Exception:
        return None, None


BACKEND_NAME, BACKEND_MODULE = _load_video_backend()


def _codec_is_h264(codec_name):
    if not codec_name:
        return False
    normalized = str(codec_name).strip().lower().replace('.', '')
    return normalized in {'h264', 'avc1', 'x264', 'libx264'}


def _probe_video_codec(path: Path):
    ffprobe_bin = shutil.which('ffprobe')
    if ffprobe_bin is None or not path.exists():
        return None
    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    for line in result.stdout.splitlines():
        value = line.strip()
        if value:
            return value
    return None


def _transcode_to_h264(path: Path):
    ffmpeg_bin = shutil.which('ffmpeg')
    if ffmpeg_bin is None:
        return False, 'ffmpeg not found on PATH'
    if not path.exists():
        return False, f'video file does not exist: {path}'
    temp_path = path.with_name(f'{path.stem}.h264tmp{path.suffix}')
    if temp_path.exists():
        temp_path.unlink()
    try:
        result = subprocess.run(
            [
                ffmpeg_bin,
                '-y',
                '-i', str(path),
                '-an',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                str(temp_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or '').strip()
        return False, stderr or f'ffmpeg failed with return code {exc.returncode}'
    except Exception as exc:
        return False, str(exc)

    if not temp_path.exists():
        return False, 'ffmpeg did not create transcoded output'

    temp_path.replace(path)
    detected = _probe_video_codec(path)
    if _codec_is_h264(detected):
        return True, detected or 'h264'

    stderr = (result.stderr or '').strip()
    return False, stderr or f'transcoded file codec is {detected!r}, expected h264'


class VideoWriterWrapper:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = max(float(fps), 1.0)
        self._writer = None
        self._size = None
        self.codec = 'libx264' if BACKEND_NAME == 'imageio' else None
        self.actual_codec = None
        self.transcode_error = None

    def _ensure_writer(self, frame: np.ndarray):
        if self._writer is not None:
            return
        height, width = frame.shape[:2]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._size = (width, height)
        if BACKEND_NAME == 'imageio':
            self._writer = BACKEND_MODULE.get_writer(
                str(self.path),
                fps=self.fps,
                codec='libx264',
                macro_block_size=1,
                ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'],
            )
            return
        if BACKEND_NAME == 'cv2':
            for codec in ('avc1', 'H264', 'X264', 'mp4v'):
                fourcc = BACKEND_MODULE.VideoWriter_fourcc(*codec)
                writer = BACKEND_MODULE.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
                if writer.isOpened():
                    self._writer = writer
                    self.codec = codec
                    return
                writer.release()
            raise RuntimeError(f'failed to open cv2 mp4 writer for {self.path}')
            return
        raise RuntimeError('no supported video backend available (need cv2 or imageio[ffmpeg])')

    def write(self, frame_rgb: np.ndarray):
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('expected RGB frame with shape HxWx3')
        self._ensure_writer(frame)
        if BACKEND_NAME == 'imageio':
            self._writer.append_data(frame)
            return
        self._writer.write(frame[:, :, ::-1])

    def close(self):
        if self._writer is None:
            return
        if BACKEND_NAME == 'cv2':
            self._writer.release()
        else:
            self._writer.close()
        self._writer = None
        detected_codec = _probe_video_codec(self.path) or self.codec
        self.actual_codec = detected_codec
        self.codec = detected_codec
        self.transcode_error = None
        if not _codec_is_h264(detected_codec):
            ok, detail = _transcode_to_h264(self.path)
            if ok:
                self.actual_codec = _probe_video_codec(self.path) or 'h264'
                self.codec = self.actual_codec
            else:
                self.transcode_error = (
                    f'failed to transcode {self.path.name} to h264 from codec={detected_codec!r}: {detail}'
                )


def _is_static_fallback_gradient(image: np.ndarray) -> bool:
    """Detect task_generator's old synthetic Isaac fallback RGB image.

    The deprecated fallback publisher produced a fixed 640x480 RGB test pattern:
    R=x%256, G=y%256, B=96. Recording that pattern makes the ego video look like
    meaningless 2x3 color blocks, so treat it as invalid camera input.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return False
    height, width = image.shape[:2]
    if height < 16 or width < 16:
        return False
    sample = image[: min(height, 480), : min(width, 640), :]
    yy, xx = np.mgrid[0:sample.shape[0], 0:sample.shape[1]]
    expected_r = (xx % 256).astype(np.uint8)
    expected_g = (yy % 256).astype(np.uint8)
    blue = sample[..., 2]
    return (
        np.array_equal(sample[..., 0], expected_r)
        and np.array_equal(sample[..., 1], expected_g)
        and bool(np.all((blue >= 94) & (blue <= 98)))
    )


def image_msg_to_numpy(message: Image):
    data = np.frombuffer(message.data, dtype=np.uint8)
    if message.encoding in ('rgb8', 'bgr8'):
        image = data.reshape((message.height, message.step // 3, 3))[:, : message.width, :].copy()
        if message.encoding == 'bgr8':
            image = image[:, :, ::-1]
        return image
    if message.encoding in ('rgba8', 'bgra8'):
        image = data.reshape((message.height, message.step // 4, 4))[:, : message.width, :4].copy()
        if message.encoding == 'bgra8':
            image = image[:, :, [2, 1, 0, 3]]
        return image[:, :, :3]
    if message.encoding == 'mono8':
        mono = data.reshape((message.height, message.step))[:, : message.width].copy()
        return np.repeat(mono[:, :, None], 3, axis=2)
    return None


def _yaw_from_quat(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_scan_ranges(scan: LaserScan):
    points = []
    angle = float(scan.angle_min)
    for raw in scan.ranges:
        rng = float(raw)
        if math.isfinite(rng) and scan.range_min <= rng <= scan.range_max:
            points.append((rng, angle))
        angle += float(scan.angle_increment)
    return points


class EvalVideoRecorder(Node):
    def __init__(self, *, output_dir, map_yaml_path, task_reset_topic, scenario_reset_topic, finished_topic, ego_topic, depth_topic, camera_info_topic, debug_overlay_topic, sim_top_down_topic, odom_topic, goal_topic, scan_topic, fps, top_down_size_px, top_down_window_m):
        super().__init__('internnav_eval_video_recorder')
        self.output_dir = Path(output_dir)
        self.videos_dir = self.output_dir / 'videos'
        self.index_path = self.output_dir / 'video_index.json'
        self.error_path = self.output_dir / 'video_recording_error.txt'
        self.fps = max(float(fps), 1.0)
        self.frame_period = 1.0 / self.fps
        self.top_down_size_px = max(int(top_down_size_px), 128)
        self.top_down_window_m = max(float(top_down_window_m), 2.0)
        self.map_yaml_path = str(map_yaml_path)
        self.ego_topic = ego_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.debug_overlay_topic = debug_overlay_topic
        self.sim_top_down_topic = sim_top_down_topic
        self.odom_topic = odom_topic
        self.goal_topic = goal_topic
        self.scan_topic = scan_topic
        self.finished = False
        self.current_episode = None
        self.current_episode_info = None
        self.ego_writer = None
        self.top_writer = None
        self.debug_overlay_writer = None
        self.sim_top_down_writer = None
        self.last_frame_time = 0.0
        self.latest_rgb = None
        self.depth_ready = False
        self.camera_info_ready = False
        self.latest_debug_overlay = None
        self.latest_sim_top_down = None
        self.latest_pose = None
        self.latest_goal = None
        self.latest_scan = []
        self.trajectory_world = []
        self.reset_seen = False
        self.last_reset_wall_time = 0.0
        self.index = {
            'video_backend': BACKEND_NAME,
            'config': {
                'map_yaml_path': self.map_yaml_path,
                'task_reset_topic': task_reset_topic,
                'scenario_reset_topic': scenario_reset_topic,
                'ego_topic': ego_topic,
                'depth_topic': depth_topic,
                'camera_info_topic': camera_info_topic,
                'debug_overlay_topic': debug_overlay_topic,
                'sim_top_down_topic': sim_top_down_topic,
                'odom_topic': odom_topic,
                'goal_topic': goal_topic,
                'scan_topic': scan_topic,
                'fps': self.fps,
                'top_down_size_px': self.top_down_size_px,
                'top_down_window_m': self.top_down_window_m,
            },
            'format': {
                'container': 'mp4',
                'preferred_codec': 'libx264',
                'file_extension': '.mp4',
            },
            'episodes': [],
        }

        self.map_image, self.map_resolution, self.map_origin = self._load_map(self.map_yaml_path)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self._write_index()

        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        event_qos = QoSProfile(depth=1)
        event_qos.reliability = ReliabilityPolicy.RELIABLE
        event_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.create_subscription(Int16, task_reset_topic, self._on_task_reset, 10)
        if scenario_reset_topic and scenario_reset_topic != task_reset_topic:
            self.create_subscription(Int16, scenario_reset_topic, self._on_task_reset, 10)
        self.create_subscription(Empty, finished_topic, self._on_finished, event_qos)
        self.create_subscription(Image, ego_topic, self._on_ego_image, sensor_qos)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self._on_depth_image, sensor_qos)
        if camera_info_topic:
            self.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, sensor_qos)
        if debug_overlay_topic:
            self.create_subscription(Image, debug_overlay_topic, self._on_debug_overlay_image, sensor_qos)
        if sim_top_down_topic:
            self.create_subscription(Image, sim_top_down_topic, self._on_sim_top_down_image, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, sensor_qos)

    def _load_map(self, map_yaml_path):
        try:
            map_yaml = Path(map_yaml_path)
            if not map_yaml.exists():
                return None, 0.05, (-0.0, -0.0)
            metadata = yaml.safe_load(map_yaml.read_text(encoding='utf-8')) or {}
            image_path = map_yaml.parent / str(metadata.get('image', ''))
            if not image_path.exists():
                return None, float(metadata.get('resolution', 0.05)), tuple(metadata.get('origin', [0.0, 0.0])[:2])
            image = PILImage.open(image_path).convert('L')
            rgb = PILImage.merge('RGB', (image, image, image))
            return np.asarray(rgb), float(metadata.get('resolution', 0.05)), tuple(metadata.get('origin', [0.0, 0.0])[:2])
        except Exception as exc:
            self._record_error(f'failed to load map: {exc}')
            return None, 0.05, (0.0, 0.0)

    def _record_error(self, message):
        self.error_path.write_text(str(message), encoding='utf-8')

    def _write_index(self):
        self.index_path.write_text(json.dumps(self.index, ensure_ascii=False, indent=2), encoding='utf-8')

    def _episode_dir(self, episode):
        return self.videos_dir / f'episode_{int(episode):04d}'

    def _close_episode(self, *, reason):
        ego_writer = self.ego_writer
        top_writer = self.top_writer
        debug_overlay_writer = self.debug_overlay_writer
        sim_top_down_writer = self.sim_top_down_writer
        if ego_writer is not None:
            ego_writer.close()
            self.ego_writer = None
        if top_writer is not None:
            top_writer.close()
            self.top_writer = None
        if debug_overlay_writer is not None:
            debug_overlay_writer.close()
            self.debug_overlay_writer = None
        if sim_top_down_writer is not None:
            sim_top_down_writer.close()
            self.sim_top_down_writer = None
        if self.current_episode_info is not None:
            if ego_writer is not None:
                self.current_episode_info['ego_video_codec'] = getattr(ego_writer, 'codec', None)
                self.current_episode_info['ego_video_codec_detected'] = getattr(ego_writer, 'actual_codec', None)
            if top_writer is not None:
                self.current_episode_info['top_down_video_codec'] = getattr(top_writer, 'codec', None)
                self.current_episode_info['top_down_video_codec_detected'] = getattr(top_writer, 'actual_codec', None)
                self.current_episode_info['map_top_down_video_codec'] = getattr(top_writer, 'codec', None)
            if debug_overlay_writer is not None:
                self.current_episode_info['debug_overlay_video_codec'] = getattr(debug_overlay_writer, 'codec', None)
                self.current_episode_info['debug_overlay_video_codec_detected'] = getattr(debug_overlay_writer, 'actual_codec', None)
            if sim_top_down_writer is not None:
                self.current_episode_info['sim_top_down_video_codec'] = getattr(sim_top_down_writer, 'codec', None)
                self.current_episode_info['sim_top_down_video_codec_detected'] = getattr(sim_top_down_writer, 'actual_codec', None)

            transcode_errors = [
                getattr(writer, 'transcode_error', None)
                for writer in (ego_writer, top_writer, debug_overlay_writer, sim_top_down_writer)
                if writer is not None and getattr(writer, 'transcode_error', None)
            ]
            if transcode_errors:
                self.current_episode_info['video_transcode_errors'] = transcode_errors
                self._record_error('\n'.join(transcode_errors))
            self.current_episode_info['close_reason'] = reason
            self.current_episode_info['finished_at_wall_time'] = time.time()
            self._write_index()
        self.current_episode = None
        self.current_episode_info = None
        self.trajectory_world = []

    def _ensure_episode(self):
        if self.current_episode is None:
            if not self._camera_ready():
                return False
            next_episode = 0
            if self.index['episodes']:
                try:
                    next_episode = int(self.index['episodes'][-1].get('episode', -1)) + 1
                except Exception:
                    next_episode = 0
            self.current_episode = next_episode
        if not self._camera_ready():
            self._record_error(
                'waiting for first real camera messages before recording episode '
                f'{self.current_episode}: ego={self.ego_topic}, depth={self.depth_topic or "<disabled>"}, '
                f'camera_info={self.camera_info_topic or "<disabled>"}'
            )
            return False
        if self.current_episode_info is not None:
            return True
        episode_dir = self._episode_dir(self.current_episode)
        episode_dir.mkdir(parents=True, exist_ok=True)
        ego_path = episode_dir / 'ego_observation.mp4'
        top_path = episode_dir / 'map_top_down_follow.mp4'
        debug_overlay_path = episode_dir / 'ego_debug_overlay.mp4'
        sim_top_down_path = episode_dir / 'sim_top_down.mp4'
        self.ego_writer = VideoWriterWrapper(ego_path, self.fps)
        self.top_writer = VideoWriterWrapper(top_path, self.fps)
        self.debug_overlay_writer = VideoWriterWrapper(debug_overlay_path, self.fps) if self.debug_overlay_topic else None
        self.sim_top_down_writer = VideoWriterWrapper(sim_top_down_path, self.fps) if self.sim_top_down_topic else None
        self.current_episode_info = {
            'episode': int(self.current_episode),
            'directory': str(episode_dir),
            'ego_video': str(ego_path),
            'top_down_video': str(top_path),
            'map_top_down_video': str(top_path),
            'debug_overlay_video': str(debug_overlay_path) if self.debug_overlay_topic else None,
            'sim_top_down_video': str(sim_top_down_path) if self.sim_top_down_topic else None,
            'ego_frames': 0,
            'top_down_frames': 0,
            'debug_overlay_frames': 0,
            'sim_top_down_frames': 0,
            'container': 'mp4',
            'started_at_wall_time': time.time(),
        }
        self.index['episodes'].append(self.current_episode_info)
        self._write_index()
        return True

    def _camera_ready(self):
        return (
            self.latest_rgb is not None
            and (not self.depth_topic or self.depth_ready)
            and (not self.camera_info_topic or self.camera_info_ready)
        )

    def _map_world_to_pixel(self, x, y):
        if self.map_image is None:
            return None
        width = self.map_image.shape[1]
        height = self.map_image.shape[0]
        px = int(round((float(x) - float(self.map_origin[0])) / self.map_resolution))
        py = int(round(height - ((float(y) - float(self.map_origin[1])) / self.map_resolution)))
        return px, py

    def _pose_pixel_to_crop(self, pixel, center_pixel, crop_size_px):
        if pixel is None:
            return None
        left = center_pixel[0] - crop_size_px
        top = center_pixel[1] - crop_size_px
        rel_x = (pixel[0] - left) * (self.top_down_size_px / (crop_size_px * 2))
        rel_y = (pixel[1] - top) * (self.top_down_size_px / (crop_size_px * 2))
        return int(rel_x), int(rel_y)

    def _render_top_down(self):
        size = self.top_down_size_px
        if self.latest_pose is None:
            return np.zeros((size, size, 3), dtype=np.uint8)

        if self.map_image is None:
            canvas = PILImage.new('RGB', (size, size), color=(25, 25, 25))
        else:
            center_pixel = self._map_world_to_pixel(self.latest_pose['x'], self.latest_pose['y'])
            crop_radius_px = max(int(round((self.top_down_window_m / self.map_resolution) / 2.0)), 32)
            left = center_pixel[0] - crop_radius_px
            top = center_pixel[1] - crop_radius_px
            right = center_pixel[0] + crop_radius_px
            bottom = center_pixel[1] + crop_radius_px
            map_image = PILImage.fromarray(self.map_image)
            if left < 0 or top < 0 or right > map_image.width or bottom > map_image.height:
                padded = PILImage.new('RGB', (max(right, map_image.width) - min(left, 0), max(bottom, map_image.height) - min(top, 0)), color=(180, 180, 180))
                padded.paste(map_image, (-min(left, 0), -min(top, 0)))
                crop = padded.crop((left + min(left, 0), top + min(top, 0), right + min(left, 0), bottom + min(top, 0)))
            else:
                crop = map_image.crop((left, top, right, bottom))
            canvas = crop.resize((size, size), _PIL_BILINEAR)

        draw = ImageDraw.Draw(canvas)
        center = (size // 2, size // 2)
        draw.ellipse((center[0] - 8, center[1] - 8, center[0] + 8, center[1] + 8), fill=(255, 80, 80), outline=(255, 255, 255), width=2)

        heading = float(self.latest_pose['yaw'])
        arrow_length = max(size // 8, 24)
        arrow_end = (
            int(center[0] + math.cos(heading) * arrow_length),
            int(center[1] - math.sin(heading) * arrow_length),
        )
        draw.line([center, arrow_end], fill=(255, 80, 80), width=4)

        if self.latest_goal is not None and self.map_image is not None:
            center_pixel = self._map_world_to_pixel(self.latest_pose['x'], self.latest_pose['y'])
            crop_radius_px = max(int(round((self.top_down_window_m / self.map_resolution) / 2.0)), 32)
            goal_pixel = self._map_world_to_pixel(self.latest_goal['x'], self.latest_goal['y'])
            goal_crop = self._pose_pixel_to_crop(goal_pixel, center_pixel, crop_radius_px)
            if goal_crop is not None:
                gx, gy = goal_crop
                draw.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), outline=(80, 255, 80), width=3)
                draw.text((gx + 10, gy - 16), 'goal', fill=(80, 255, 80))

        if self.trajectory_world and self.map_image is not None:
            center_pixel = self._map_world_to_pixel(self.latest_pose['x'], self.latest_pose['y'])
            crop_radius_px = max(int(round((self.top_down_window_m / self.map_resolution) / 2.0)), 32)
            traj = []
            for x, y in self.trajectory_world[-80:]:
                pixel = self._map_world_to_pixel(x, y)
                crop_pixel = self._pose_pixel_to_crop(pixel, center_pixel, crop_radius_px)
                if crop_pixel is not None:
                    traj.append(crop_pixel)
            if len(traj) >= 2:
                draw.line(traj, fill=(64, 200, 255), width=3)

        for rng, angle in self.latest_scan[:720]:
            rel_angle = heading + angle
            wx = self.latest_pose['x'] + math.cos(rel_angle) * rng
            wy = self.latest_pose['y'] + math.sin(rel_angle) * rng
            if self.map_image is not None:
                center_pixel = self._map_world_to_pixel(self.latest_pose['x'], self.latest_pose['y'])
                crop_radius_px = max(int(round((self.top_down_window_m / self.map_resolution) / 2.0)), 32)
                pixel = self._map_world_to_pixel(wx, wy)
                scan_crop = self._pose_pixel_to_crop(pixel, center_pixel, crop_radius_px)
            else:
                scale = self.top_down_size_px / self.top_down_window_m
                scan_crop = (
                    int(center[0] + math.cos(rel_angle) * rng * scale),
                    int(center[1] - math.sin(rel_angle) * rng * scale),
                )
            if scan_crop is not None:
                draw.point(scan_crop, fill=(255, 220, 80))

        draw.rectangle((8, 8, size - 8, 58), outline=(0, 255, 0), width=2)
        draw.text((16, 16), f'episode: {self.current_episode if self.current_episode is not None else "idle"}', fill=(255, 255, 255))
        draw.text((16, 34), f'pose: x={self.latest_pose["x"]:.2f} y={self.latest_pose["y"]:.2f}', fill=(255, 255, 255))
        return np.asarray(canvas, dtype=np.uint8)

    def _maybe_write_frame(self):
        if self.latest_rgb is None or not self._ensure_episode():
            return
        now = time.monotonic()
        if (now - self.last_frame_time) < self.frame_period:
            return
        ego_frame = np.asarray(self.latest_rgb, dtype=np.uint8)
        if _is_static_fallback_gradient(ego_frame):
            self._record_error(f'skipped synthetic fallback gradient frame from {self.ego_topic}; waiting for real Isaac camera frames')
            return
        top_frame = self._render_top_down()
        self.ego_writer.write(ego_frame)
        self.top_writer.write(top_frame)
        self.current_episode_info['ego_frames'] += 1
        self.current_episode_info['top_down_frames'] += 1
        self.current_episode_info['ego_video_codec'] = getattr(self.ego_writer, 'codec', None)
        self.current_episode_info['top_down_video_codec'] = getattr(self.top_writer, 'codec', None)
        if self.debug_overlay_writer is not None and self.latest_debug_overlay is not None:
            debug_overlay_frame = np.asarray(self.latest_debug_overlay, dtype=np.uint8)
            if not _is_static_fallback_gradient(debug_overlay_frame):
                self.debug_overlay_writer.write(debug_overlay_frame)
                self.current_episode_info['debug_overlay_frames'] += 1
                self.current_episode_info['debug_overlay_video_codec'] = getattr(self.debug_overlay_writer, 'codec', None)
        if self.sim_top_down_writer is not None and self.latest_sim_top_down is not None:
            sim_top_down_frame = np.asarray(self.latest_sim_top_down, dtype=np.uint8)
            self.sim_top_down_writer.write(sim_top_down_frame)
            self.current_episode_info['sim_top_down_frames'] += 1
        self.current_episode_info['last_frame_wall_time'] = time.time()
        self.last_frame_time = now
        self._write_index()

    def _on_task_reset(self, msg: Int16):
        episode = int(msg.data)
        self.reset_seen = True
        self.last_reset_wall_time = time.time()
        if self.current_episode == episode and self.current_episode_info is not None:
            return
        if self.current_episode_info is not None:
            self._close_episode(reason='task_reset')
        self.current_episode = episode
        self.current_episode_info = None
        self.last_frame_time = 0.0
        self.trajectory_world = []
        if self._camera_ready():
            self._ensure_episode()

    def _on_finished(self, _msg: Empty):
        # /finished uses transient-local QoS in the task generator.  A recorder
        # that starts after a previous run can receive that stale latched finish
        # near the next task_reset and exit before any navigation frames are
        # written.  Only accept finish after this process has observed a reset
        # and the episode has been alive long enough to be from the current run.
        if self.reset_seen:
            if (time.time() - self.last_reset_wall_time) < 2.0:
                return
        else:
            if self.current_episode_info is None:
                return
            if int(self.current_episode_info.get('ego_frames', 0) or 0) <= 0:
                return
        self.finished = True
        self._close_episode(reason='finished')

    def _on_ego_image(self, msg: Image):
        image = image_msg_to_numpy(msg)
        if image is None:
            return
        self.latest_rgb = image
        self._maybe_write_frame()

    def _on_depth_image(self, _msg: Image):
        self.depth_ready = True
        self._maybe_write_frame()

    def _on_camera_info(self, _msg: CameraInfo):
        self.camera_info_ready = True
        self._maybe_write_frame()

    def _on_debug_overlay_image(self, msg: Image):
        image = image_msg_to_numpy(msg)
        if image is None:
            return
        self.latest_debug_overlay = image

    def _on_sim_top_down_image(self, msg: Image):
        image = image_msg_to_numpy(msg)
        if image is None:
            return
        self.latest_sim_top_down = image

    def _on_odom(self, msg: Odometry):
        pose = msg.pose.pose
        quat = pose.orientation
        self.latest_pose = {
            'x': float(pose.position.x),
            'y': float(pose.position.y),
            'yaw': _yaw_from_quat(float(quat.x), float(quat.y), float(quat.z), float(quat.w)),
        }
        self.trajectory_world.append((self.latest_pose['x'], self.latest_pose['y']))
        if len(self.trajectory_world) > 512:
            self.trajectory_world = self.trajectory_world[-512:]
        # Camera frames in Isaac can arrive before the episode reset marker and
        # then stay latched/static for a while.  Odom is the reliable stream
        # during navigation, so use it to drive video frame capture once the
        # first valid camera/depth/camera_info sample has made the recorder
        # ready.  This guarantees a top-down trajectory video even if no new
        # RGB callback occurs after task_reset.
        self._maybe_write_frame()

    def _on_goal(self, msg: PoseStamped):
        self.latest_goal = {
            'x': float(msg.pose.position.x),
            'y': float(msg.pose.position.y),
        }

    def _on_scan(self, msg: LaserScan):
        self.latest_scan = _normalize_scan_ranges(msg)


OUTPUT_DIR = sys.argv[1]
MAP_YAML_PATH = sys.argv[2]
TASK_RESET_TOPIC = sys.argv[3]
SCENARIO_RESET_TOPIC = sys.argv[4]
FINISHED_TOPIC = sys.argv[5]
EGO_TOPIC = sys.argv[6]
DEPTH_TOPIC = sys.argv[7]
CAMERA_INFO_TOPIC = sys.argv[8]
DEBUG_OVERLAY_TOPIC = sys.argv[9]
SIM_TOP_DOWN_TOPIC = sys.argv[10]
ODOM_TOPIC = sys.argv[11]
GOAL_TOPIC = sys.argv[12]
SCAN_TOPIC = sys.argv[13]
FPS = float(sys.argv[14])
TOP_DOWN_SIZE_PX = int(sys.argv[15])
TOP_DOWN_WINDOW_M = float(sys.argv[16])

rclpy.init()
node = EvalVideoRecorder(
    output_dir=OUTPUT_DIR,
    map_yaml_path=MAP_YAML_PATH,
    task_reset_topic=TASK_RESET_TOPIC,
    scenario_reset_topic=SCENARIO_RESET_TOPIC,
    finished_topic=FINISHED_TOPIC,
    ego_topic=EGO_TOPIC,
    depth_topic=DEPTH_TOPIC,
    camera_info_topic=CAMERA_INFO_TOPIC,
    debug_overlay_topic=DEBUG_OVERLAY_TOPIC,
    sim_top_down_topic=SIM_TOP_DOWN_TOPIC,
    odom_topic=ODOM_TOPIC,
    goal_topic=GOAL_TOPIC,
    scan_topic=SCAN_TOPIC,
    fps=FPS,
    top_down_size_px=TOP_DOWN_SIZE_PX,
    top_down_window_m=TOP_DOWN_WINDOW_M,
)


def _shutdown(*_args):
    try:
        node._close_episode(reason='shutdown')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

exit_code = 0
try:
    while rclpy.ok() and not node.finished:
        try:
            rclpy.spin_once(node, timeout_sec=0.5)
        except KeyboardInterrupt:
            break
        except BaseException:
            if not rclpy.ok():
                break
            raise
except BaseException:
    node._record_error(traceback.format_exc())
    exit_code = 1
finally:
    node._close_episode(reason='process_exit' if exit_code == 0 else 'exception')
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

raise SystemExit(exit_code)
'''

    return subprocess.Popen(
        [
            python_bin,
            '-c',
            recorder_code,
            output_dir,
            map_yaml_path,
            task_reset_topic,
            scenario_reset_topic,
            finished_topic,
            ego_topic,
            depth_topic,
            camera_info_topic,
            debug_overlay_topic,
            sim_top_down_topic,
            odom_topic,
            goal_topic,
            scan_topic,
            str(fps),
            str(top_down_size_px),
            str(top_down_window_m),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _apply_runtime_defaults(args) -> dict:
    adjustments = {}

    env_python, env_python_name = _first_env_value(
        'ARENA_VLN_MODEL_PYTHON',
        'ARENA_INTERNNAV_PYTHON',
        'ARENA_PYTHON',
    )
    if env_python and not getattr(args, 'dual_vln_python_executable', ''):
        args.dual_vln_python_executable = env_python
        adjustments['dual_vln_python_executable'] = f'{env_python} ({env_python_name})'

    if getattr(args, 'dual_vln_status_topic', '') in {
        '/task_generator_node/dual_vln/status',
        '/task_generator_node/internnav/status',
    }:
        args.dual_vln_status_topic = f'/task_generator_node/{args.robot}/internnav/status'
        adjustments['dual_vln_status_topic'] = args.dual_vln_status_topic

    if _is_internnav_run(args):
        env_model_path, env_model_path_name = _first_env_value(
            'ARENA_INTERNNAV_MODEL_PATH',
            'INTERNNAV_MODEL_PATH',
            'ARENA_VLN_MODEL_PATH',
        )
        if env_model_path and not getattr(args, 'dual_vln_model_path', ''):
            args.dual_vln_model_path = env_model_path
            adjustments['dual_vln_model_path'] = f'{env_model_path} ({env_model_path_name})'
        default_topics = _default_vision_topics(args.robot)
        if default_topics is not None:
            rgb_topic, depth_topic, camera_info_topic = default_topics
            if not args.dual_vln_rgb_topic:
                args.dual_vln_rgb_topic = rgb_topic
                adjustments['dual_vln_rgb_topic'] = rgb_topic
            if not args.dual_vln_depth_topic:
                args.dual_vln_depth_topic = depth_topic
                adjustments['dual_vln_depth_topic'] = depth_topic
            if not args.dual_vln_camera_info_topic:
                args.dual_vln_camera_info_topic = camera_info_topic
                adjustments['dual_vln_camera_info_topic'] = camera_info_topic

    if not getattr(args, 'eval_video_sim_top_down_topic', ''):
        sim_top_down_topic = _default_eval_video_sim_top_down_topic(args.robot)
        if sim_top_down_topic:
            args.eval_video_sim_top_down_topic = sim_top_down_topic
            adjustments['eval_video_sim_top_down_topic'] = sim_top_down_topic

    if not getattr(args, 'eval_video_debug_overlay_topic', ''):
        debug_overlay_topic = _default_eval_video_debug_overlay_topic()
        if debug_overlay_topic:
            args.eval_video_debug_overlay_topic = debug_overlay_topic
            adjustments['eval_video_debug_overlay_topic'] = debug_overlay_topic

    if getattr(args, 'save_eval_video', False):
        args.dual_vln_enable_visualization = True
        adjustments.setdefault('dual_vln_enable_visualization', True)

    if str(args.dual_vln_device).strip().lower() == 'cpu' and args.dual_vln_inference_timeout_sec <= 0.2:
        args.dual_vln_inference_timeout_sec = 120.0
        adjustments['dual_vln_inference_timeout_sec'] = 120.0
    if str(args.dual_vln_device).strip().lower() == 'cpu' and args.dual_vln_inference_rate_hz >= 10.0:
        args.dual_vln_inference_rate_hz = 0.5
        adjustments['dual_vln_inference_rate_hz'] = 0.5

    return adjustments


def _classify_end_reason(*, finished_observed: bool, launch_returncode: int | None, timed_out: bool, internnav_status):
    if isinstance(internnav_status, dict):
        status = str(internnav_status.get('status', ''))
        degraded = bool(internnav_status.get('degraded', False))
        if status in {
            'adapter_exception',
            'invalid_adapter_output',
            'model_unavailable',
            'internnav_missing_rgb',
            'internnav_missing_depth',
            'internnav_empty_output',
        }:
            return 'adapter_failure'
        if degraded and status in {'inference_timeout', 'exception'}:
            return 'infrastructure_exception'
        if internnav_status.get('debug', {}).get('safe_stop'):
            return 'safe_stop'

    if finished_observed:
        return 'finished'
    if timed_out:
        return 'timeout'
    if launch_returncode not in (None, 0):
        return 'infrastructure_exception'
    return 'completed_without_finished_topic'


def _terminate_process_tree(proc: subprocess.Popen, *, grace_period_sec: float = 20.0) -> int:
    if proc.poll() is not None:
        return proc.returncode

    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return proc.wait(timeout=1.0)

    deadline = time.monotonic() + max(grace_period_sec, 0.0)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(0.25)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.wait(timeout=1.0)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(0.25)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return proc.wait(timeout=5.0)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run a reproducible Arena InternNav eval.')
    parser.add_argument('--sim', default='isaac')
    parser.add_argument('--human', default='hunav')
    parser.add_argument('--world', default='map_empty')
    parser.add_argument('--robot', default='jackal')
    parser.add_argument('--local-planner', default='dual_vln')
    parser.add_argument('--inter-planner', default='navigate_to_pose_w_replanning_and_recovery')
    parser.add_argument('--global-planner', default='navfn')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--tm-robots', default='random')
    parser.add_argument('--headless', default='2')
    parser.add_argument('--log-level', default='warn')
    parser.add_argument('--vln-instruction', default='navigate')
    parser.add_argument('--vln-instruction-file', default='')
    parser.add_argument('--internnav-mode', '--dual-vln-mode', dest='dual_vln_mode', default='heuristic')
    parser.add_argument('--internnav-model-path', '--dual-vln-model-path', dest='dual_vln_model_path', default='')
    parser.add_argument('--internnav-device', '--dual-vln-device', dest='dual_vln_device', default='cpu')
    parser.add_argument('--internnav-inference-rate-hz', '--dual-vln-inference-rate-hz', dest='dual_vln_inference_rate_hz', type=float, default=10.0)
    parser.add_argument('--internnav-inference-timeout-sec', '--dual-vln-inference-timeout-sec', dest='dual_vln_inference_timeout_sec', type=float, default=0.2)
    parser.add_argument('--internnav-rgb-topic', '--dual-vln-rgb-topic', dest='dual_vln_rgb_topic', default='')
    parser.add_argument('--internnav-depth-topic', '--dual-vln-depth-topic', dest='dual_vln_depth_topic', default='')
    parser.add_argument('--internnav-camera-info-topic', '--dual-vln-camera-info-topic', dest='dual_vln_camera_info_topic', default='')
    parser.add_argument('--internnav-python-executable', '--dual-vln-python-executable', dest='dual_vln_python_executable', default='')
    parser.add_argument('--internnav-adapter-target', '--dual-vln-adapter-target', dest='dual_vln_adapter_target', default='')
    parser.add_argument('--internnav-require-real-backend', '--dual-vln-require-real-backend', dest='dual_vln_require_real_backend', action='store_true')
    parser.add_argument('--internnav-strict-device', '--dual-vln-strict-device', dest='dual_vln_strict_device', action='store_true')
    parser.add_argument('--internnav-look-down', '--dual-vln-look-down', dest='dual_vln_look_down', action='store_true')
    parser.add_argument('--internnav-enable-visualization', '--dual-vln-enable-visualization', dest='dual_vln_enable_visualization', action='store_true')
    parser.add_argument('--internnav-visualization-topic', '--dual-vln-visualization-topic', dest='dual_vln_visualization_topic', default='internnav/debug_image')
    parser.add_argument('--internnav-visualization-rate-hz', '--dual-vln-visualization-rate-hz', dest='dual_vln_visualization_rate_hz', type=float, default=5.0)
    parser.add_argument('--save-eval-video', action='store_true')
    parser.add_argument('--eval-video-fps', type=float, default=10.0)
    parser.add_argument('--eval-video-top-down-size-px', type=int, default=640)
    parser.add_argument('--eval-video-top-down-window-m', type=float, default=10.0)
    parser.add_argument('--eval-video-sim-top-down-topic', default='')
    parser.add_argument('--eval-video-debug-overlay-topic', default='')
    parser.add_argument('--internnav-status-topic', '--dual-vln-status-topic', dest='dual_vln_status_topic', default='/task_generator_node/internnav/status')
    parser.add_argument('--finished-topic', default='/task_generator_node/finished')
    parser.add_argument('--task-reset-topic', default='/task_generator_node/task_reset')
    parser.add_argument('--launch-timeout-sec', type=float, default=0.0)
    parser.add_argument('--shutdown-grace-period-sec', type=float, default=20.0)
    parser.add_argument('--output-prefix', default='internnav_eval')
    parser.add_argument(
        '--output-root',
        default='',
        help=(
            'Root directory for eval artifacts. Defaults to <workspace>/outputs. '
            'Use an absolute path to write elsewhere, or a relative path under the workspace root.'
        ),
    )
    parser.add_argument('--skip-metrics', action='store_true')
    parser.add_argument('extra_launch_args', nargs='*', help='Additional KEY:=VALUE launch arguments')
    args = parser.parse_args()
    runtime_adjustments = _apply_runtime_defaults(args)

    arena_eval_share = get_package_share_directory('arena_evaluation')
    bringup_share = get_package_share_directory('arena_bringup')
    sim_setup_share = get_package_share_directory('arena_simulation_setup')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{timestamp}_{args.world}_{args.robot}_{args.dual_vln_mode}'
    relative_dir = os.path.join(args.output_prefix, run_name)
    output_root = _resolve_output_root(args.output_root, arena_eval_share)
    output_dir = os.path.join(output_root, relative_dir)
    snapshots_dir = os.path.join(output_dir, 'snapshots')
    os.makedirs(snapshots_dir, exist_ok=True)
    dual_vln_status_path = os.path.join(output_dir, 'internnav_status.json')
    postprocess_commands_path = os.path.join(output_dir, 'postprocess_commands.txt')
    videos_dir = os.path.join(output_dir, 'videos')
    video_index_path = os.path.join(output_dir, 'video_index.json')
    video_error_path = os.path.join(output_dir, 'video_recording_error.txt')

    if args.save_eval_video and not args.dual_vln_rgb_topic:
        raise SystemExit(
            'save_eval_video requires a resolvable RGB topic. '
            'Pass --internnav-rgb-topic explicitly or use a robot/mode with runtime defaults.'
        )

    robot_ego_topic = _robot_topic(args.task_reset_topic, args.robot, args.dual_vln_rgb_topic)
    robot_depth_topic = _robot_topic(args.task_reset_topic, args.robot, args.dual_vln_depth_topic)
    robot_camera_info_topic = _robot_topic(args.task_reset_topic, args.robot, args.dual_vln_camera_info_topic)
    robot_debug_overlay_topic = _robot_topic(args.task_reset_topic, args.robot, args.eval_video_debug_overlay_topic)
    robot_sim_top_down_topic = _robot_topic(args.task_reset_topic, args.robot, args.eval_video_sim_top_down_topic)
    robot_odom_topic = _robot_topic(args.task_reset_topic, args.robot, 'odom')
    robot_goal_topic = _robot_topic(args.task_reset_topic, args.robot, 'episode_goal_pose')
    robot_scan_topic = _robot_topic(args.task_reset_topic, args.robot, 'scan')
    robot_scenario_reset_topic = _scenario_reset_topic(args.task_reset_topic, args.robot)
    map_yaml_path = _world_map_yaml_path(sim_setup_share, args.world)

    snapshot_files = {}
    for label, src in {
        'task_generator': os.path.join(bringup_share, 'configs', 'task_generator.yaml'),
        'internnav_controller': os.path.join(sim_setup_share, 'configs', 'nav2', 'controllers', 'dual_vln', 'controller_config.yaml'),
    }.items():
        copied = _copy_if_exists(src, os.path.join(snapshots_dir, os.path.basename(src)))
        if copied is not None:
            snapshot_files[label] = copied

    launch_cmd = [
        'ros2', 'launch', 'arena_bringup', 'arena.launch.py',
        f'sim:={args.sim}',
        f'human:={args.human}',
        f'world:={args.world}',
        f'robot:={args.robot}',
        f'local_planner:={args.local_planner}',
        f'inter_planner:={args.inter_planner}',
        f'global_planner:={args.global_planner}',
        f'record_data_dir:={output_dir}',
        f'episodes:={args.episodes}',
        'auto_reset:=true',
        f'tm_robots:={args.tm_robots}',
        f'timeout:={args.timeout}',
        f'headless:={args.headless}',
        f'log_level:={args.log_level}',
        f'vln_instruction:={args.vln_instruction}',
        f'dual_vln_mode:={args.dual_vln_mode}',
        f'dual_vln_device:={args.dual_vln_device}',
        f'dual_vln_inference_rate_hz:={args.dual_vln_inference_rate_hz}',
        f'dual_vln_inference_timeout_sec:={args.dual_vln_inference_timeout_sec}',
        f'dual_vln_enable_visualization:={str(args.dual_vln_enable_visualization).lower()}',
        f'dual_vln_require_real_backend:={str(args.dual_vln_require_real_backend).lower()}',
        f'dual_vln_strict_device:={str(args.dual_vln_strict_device).lower()}',
        f'dual_vln_visualization_topic:={args.dual_vln_visualization_topic}',
        f'dual_vln_visualization_rate_hz:={args.dual_vln_visualization_rate_hz}',
    ]
    if args.dual_vln_rgb_topic:
        launch_cmd.append(f'dual_vln_rgb_topic:={args.dual_vln_rgb_topic}')
    if args.dual_vln_depth_topic:
        launch_cmd.append(f'dual_vln_depth_topic:={args.dual_vln_depth_topic}')
    if args.dual_vln_camera_info_topic:
        launch_cmd.append(f'dual_vln_camera_info_topic:={args.dual_vln_camera_info_topic}')
    if args.dual_vln_python_executable:
        launch_cmd.append(f'dual_vln_python_executable:={args.dual_vln_python_executable}')
    if args.dual_vln_adapter_target:
        launch_cmd.append(f'dual_vln_adapter_target:={args.dual_vln_adapter_target}')
    if args.dual_vln_look_down:
        launch_cmd.append('dual_vln_look_down:=true')
    if args.vln_instruction_file:
        launch_cmd.append(f'vln_instruction_file:={args.vln_instruction_file}')
    if args.dual_vln_model_path:
        launch_cmd.append(f'dual_vln_model_path:={args.dual_vln_model_path}')
    launch_cmd.extend(args.extra_launch_args)

    metrics_cmd = ['ros2', 'run', 'arena_evaluation', 'metrics', '--dir', output_dir]
    postprocess_commands = [
        ' '.join(launch_cmd),
        ' '.join(metrics_cmd),
    ]
    _write_text(postprocess_commands_path, '\n'.join(postprocess_commands) + '\n')

    manifest = {
        'timestamp': timestamp,
        'result_dir_relative': relative_dir,
        'result_dir_absolute': output_dir,
        'output_root': output_root,
        'launch_command': launch_cmd,
        'metrics_command': None if args.skip_metrics else metrics_cmd,
        'postprocess_commands_file': postprocess_commands_path,
        'parameters': {
            'sim': args.sim,
            'human': args.human,
            'world': args.world,
            'robot': args.robot,
            'local_planner': args.local_planner,
            'inter_planner': args.inter_planner,
            'global_planner': args.global_planner,
            'episodes': args.episodes,
            'timeout': args.timeout,
            'tm_robots': args.tm_robots,
            'vln_instruction': args.vln_instruction,
            'vln_instruction_file': args.vln_instruction_file,
            'dual_vln_mode': args.dual_vln_mode,
            'dual_vln_model_path': args.dual_vln_model_path,
            'dual_vln_device': args.dual_vln_device,
            'dual_vln_inference_rate_hz': args.dual_vln_inference_rate_hz,
            'dual_vln_inference_timeout_sec': args.dual_vln_inference_timeout_sec,
            'dual_vln_rgb_topic': args.dual_vln_rgb_topic,
            'dual_vln_depth_topic': args.dual_vln_depth_topic,
            'dual_vln_camera_info_topic': args.dual_vln_camera_info_topic,
            'dual_vln_python_executable': args.dual_vln_python_executable,
            'eval_python_executable': sys.executable,
            'dual_vln_adapter_target': args.dual_vln_adapter_target,
            'dual_vln_require_real_backend': args.dual_vln_require_real_backend,
            'dual_vln_strict_device': args.dual_vln_strict_device,
            'dual_vln_look_down': args.dual_vln_look_down,
            'dual_vln_enable_visualization': args.dual_vln_enable_visualization,
            'dual_vln_visualization_topic': args.dual_vln_visualization_topic,
            'dual_vln_visualization_rate_hz': args.dual_vln_visualization_rate_hz,
            'save_eval_video': args.save_eval_video,
            'eval_video_fps': args.eval_video_fps,
            'eval_video_top_down_size_px': args.eval_video_top_down_size_px,
            'eval_video_top_down_window_m': args.eval_video_top_down_window_m,
            'eval_video_sim_top_down_topic': robot_sim_top_down_topic if args.save_eval_video else None,
            'eval_video_ego_topic': robot_ego_topic if args.save_eval_video else None,
            'eval_video_depth_topic': robot_depth_topic if args.save_eval_video else None,
            'eval_video_camera_info_topic': robot_camera_info_topic if args.save_eval_video else None,
            'eval_video_debug_overlay_topic': robot_debug_overlay_topic if args.save_eval_video else None,
            'eval_video_odom_topic': robot_odom_topic if args.save_eval_video else None,
            'eval_video_goal_topic': robot_goal_topic if args.save_eval_video else None,
            'eval_video_scan_topic': robot_scan_topic if args.save_eval_video else None,
            'eval_video_scenario_reset_topic': robot_scenario_reset_topic if args.save_eval_video else None,
            'eval_video_map_yaml_path': map_yaml_path if args.save_eval_video else None,
            'dual_vln_status_topic': args.dual_vln_status_topic,
            'finished_topic': args.finished_topic,
            'task_reset_topic': args.task_reset_topic,
            'launch_timeout_sec': args.launch_timeout_sec,
            'shutdown_grace_period_sec': args.shutdown_grace_period_sec,
            'output_prefix': args.output_prefix,
            'output_root': output_root,
        },
        'runtime_adjustments': runtime_adjustments,
        'snapshots': snapshot_files,
        'artifacts': {
            'snapshots_dir': snapshots_dir,
            'dual_vln_status_path': dual_vln_status_path,
            'postprocess_commands_file': postprocess_commands_path,
            'videos_dir': videos_dir if args.save_eval_video else None,
            'video_index_path': video_index_path if args.save_eval_video else None,
            'video_recording_error_path': video_error_path if args.save_eval_video else None,
        },
        'result': {
            'finished_observed': False,
            'launch_returncode': None,
            'metrics_returncode': None,
            'end_reason': 'running',
            'video_recorder_returncode': None,
        },
    }

    manifest_path = os.path.join(output_dir, 'run_manifest.yaml')
    _write_yaml(manifest_path, manifest)

    env = os.environ.copy()
    env.setdefault('RCUTILS_LOGGING_BUFFERED_STREAM', '1')
    env.setdefault('ARENA_EVAL_PYTHON', sys.executable)
    launch_timeout_sec = args.launch_timeout_sec
    if launch_timeout_sec <= 0.0:
        launch_timeout_sec = max(float(args.timeout) * max(args.episodes, 1) + 120.0, 180.0)

    video_proc = None
    if args.save_eval_video:
        video_proc = _start_eval_video_recorder(
            env,
            output_dir=output_dir,
            map_yaml_path=map_yaml_path,
            task_reset_topic=args.task_reset_topic,
            scenario_reset_topic=robot_scenario_reset_topic,
            finished_topic=args.finished_topic,
            ego_topic=robot_ego_topic,
            depth_topic=robot_depth_topic,
            camera_info_topic=robot_camera_info_topic,
            debug_overlay_topic=robot_debug_overlay_topic,
            sim_top_down_topic=robot_sim_top_down_topic,
            odom_topic=robot_odom_topic,
            goal_topic=robot_goal_topic,
            scan_topic=robot_scan_topic,
            fps=args.eval_video_fps,
            top_down_size_px=args.eval_video_top_down_size_px,
            top_down_window_m=args.eval_video_top_down_window_m,
        )

    launch_proc = subprocess.Popen(
        launch_cmd,
        env=env,
        start_new_session=True,
    )
    finished_proc = _start_finished_watcher(
        env,
        args.finished_topic,
        args.task_reset_topic,
        robot_scenario_reset_topic,
    )
    status_proc = _start_status_watcher(env, args.dual_vln_status_topic, dual_vln_status_path)

    deadline = time.monotonic() + launch_timeout_sec
    launch_returncode = None
    metrics_returncode = None
    finished_observed = False
    timed_out = False

    try:
        while True:
            launch_returncode = launch_proc.poll()
            if launch_returncode is not None:
                break

            finished_returncode = finished_proc.poll()
            if finished_returncode == 0:
                finished_observed = True
                launch_returncode = _terminate_process_tree(
                    launch_proc,
                    grace_period_sec=args.shutdown_grace_period_sec,
                )
                break

            if time.monotonic() >= deadline:
                timed_out = True
                launch_returncode = _terminate_process_tree(
                    launch_proc,
                    grace_period_sec=args.shutdown_grace_period_sec,
                )
                break

            time.sleep(1.0)
    finally:
        if finished_proc.poll() is None:
            _terminate_process_tree(finished_proc, grace_period_sec=2.0)
        if status_proc.poll() is None:
            _terminate_process_tree(status_proc, grace_period_sec=2.0)
        if video_proc is not None and video_proc.poll() is None:
            _terminate_process_tree(video_proc, grace_period_sec=5.0)

    dual_vln_status = _read_json_if_exists(dual_vln_status_path)
    video_index = _read_json_if_exists(video_index_path) if args.save_eval_video else None
    video_error = _read_text_if_exists(video_error_path) if args.save_eval_video else None
    video_returncode = video_proc.returncode if video_proc is not None else None
    end_reason = _classify_end_reason(
        finished_observed=finished_observed,
        launch_returncode=launch_returncode,
        timed_out=timed_out,
        internnav_status=dual_vln_status,
    )

    manifest['artifacts']['internnav_status_present'] = dual_vln_status is not None
    manifest['artifacts']['snapshot_files'] = sorted(snapshot_files.values())
    manifest['artifacts']['video_index_present'] = video_index is not None
    manifest['artifacts']['video_index'] = video_index
    manifest['artifacts']['video_recording_error'] = video_error
    manifest['result'].update(
        {
            'finished_observed': finished_observed,
            'launch_returncode': launch_returncode,
            'metrics_returncode': metrics_returncode,
            'timed_out': timed_out,
            'end_reason': end_reason,
            'dual_vln_status': dual_vln_status,
            'video_recorder_returncode': video_returncode,
        }
    )
    _write_yaml(manifest_path, manifest)

    if finished_observed and launch_returncode not in (None, 0):
        launch_returncode = 0

    if timed_out:
        manifest['result']['launch_returncode'] = launch_returncode
        _write_yaml(manifest_path, manifest)
        return 124 if launch_returncode == 0 else launch_returncode

    if launch_returncode != 0:
        manifest['result']['launch_returncode'] = launch_returncode
        _write_yaml(manifest_path, manifest)
        return launch_returncode

    if args.skip_metrics:
        _write_yaml(manifest_path, manifest)
        return 0

    metrics_result = subprocess.run(metrics_cmd, env=env)
    metrics_returncode = metrics_result.returncode
    manifest['result']['metrics_returncode'] = metrics_returncode
    if metrics_returncode != 0 and manifest['result']['end_reason'] == 'finished':
        manifest['result']['end_reason'] = 'metrics_failed'
    _write_yaml(manifest_path, manifest)
    return metrics_returncode


if __name__ == '__main__':
    raise SystemExit(main())
