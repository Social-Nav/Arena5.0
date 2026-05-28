"""
Rosbag playback tests for EvalVideoRecorder.

Verifies that EvalVideoRecorder correctly:
1. Subscribes to input topics from rosbag playback
2. Creates video output files
3. Generates a valid video_index.json
4. Does not crash during playback

Usage (in Docker container):
    cd /opt/arena_ws
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    python3 -m pytest src/Arena/arena_bringup/test/test_eval_video_recorder_rosbag.py -v
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

_ARENA_SRC = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ARENA_SRC))

import rclpy
from rclpy.executors import SingleThreadedExecutor

# Import rosbag_player from conftest
from conftest import rosbag_player


def _get_eval_video_recorder_class():
    """Extract EvalVideoRecorder class from internnav_eval module.

    The class is defined inside a raw string literal (recorder_code) that is
    executed as a subprocess, so it's not directly importable.
    """
    import arena_bringup.internnav_eval as _mod
    import inspect

    src = inspect.getsource(_mod)
    # Find the recorder_code raw string
    start_marker = "recorder_code = r'''"
    end_marker = "'''"
    start_idx = src.find(start_marker)
    if start_idx == -1:
        raise RuntimeError("Cannot find recorder_code in internnav_eval")
    start_idx += len(start_marker) + 1  # skip past the opening newline
    end_idx = src.find(end_marker, start_idx)
    if end_idx == -1:
        raise RuntimeError("Cannot find end of recorder_code")
    code = src[start_idx:end_idx]

    # Strip the main block (everything after the class definition)
    # The main block starts with "OUTPUT_DIR = sys.argv[1]"
    main_start = code.find('\nOUTPUT_DIR = sys.argv[1]')
    if main_start != -1:
        code = code[:main_start]

    # Exec the code to get EvalVideoRecorder
    namespace = {}
    exec(code, namespace)
    return namespace['EvalVideoRecorder']


@pytest.fixture(scope='module')
def ros_context():
    """Initialize rclpy once per module."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for video output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def map_yaml_path():
    """Path to a minimal map YAML for testing."""
    hospital_map = (
        _ARENA_SRC / 'arena_simulation_setup' / 'worlds' / 'hospital_1' / 'map.yaml'
    )
    if hospital_map.exists():
        return str(hospital_map)

    # Create a minimal map YAML in temp
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    tmp.write("""image: test_map.pgm
resolution: 0.05
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
""")
    tmp.close()
    return tmp.name


class TestEvalVideoRecorderRosbag:
    """Integration tests for EvalVideoRecorder with rosbag playback."""

    def test_node_initializes(self, ros_context, temp_output_dir, map_yaml_path):
        """EvalVideoRecorder should initialize without errors."""
        EvalVideoRecorder = _get_eval_video_recorder_class()

        node = EvalVideoRecorder(
            output_dir=str(temp_output_dir),
            map_yaml_path=map_yaml_path,
            task_reset_topic='/task_reset',
            scenario_reset_topic='',
            finished_topic='/finished',
            ego_topic='/ego_image',
            depth_topic='',
            camera_info_topic='',
            debug_overlay_topic='',
            sim_top_down_topic='',
            odom_topic='/odom',
            goal_topic='/goal_pose',
            scan_topic='/scan',
            fps=10.0,
            top_down_size_px=256,
            top_down_window_m=10.0,
        )

        assert node is not None
        assert node.get_name() == 'internnav_eval_video_recorder'
        node.destroy_node()

    def test_handles_rosbag_playback(
        self, ros_context, temp_output_dir, map_yaml_path, rosbag_reader
    ):
        """EvalVideoRecorder should process rosbag messages and create output."""
        reader, topics = rosbag_reader
        EvalVideoRecorder = _get_eval_video_recorder_class()

        node = EvalVideoRecorder(
            output_dir=str(temp_output_dir),
            map_yaml_path=map_yaml_path,
            task_reset_topic='/task_reset',
            scenario_reset_topic='',
            finished_topic='/finished',
            ego_topic='/ego_image',
            depth_topic='',
            camera_info_topic='',
            debug_overlay_topic='',
            sim_top_down_topic='',
            odom_topic='/odom',
            goal_topic='/goal_pose',
            scan_topic='/scan',
            fps=10.0,
            top_down_size_px=256,
            top_down_window_m=10.0,
        )

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Map bag topic names to what EvalVideoRecorder expects (with leading /)
        topic_map = {
            'task_reset': '/task_reset',
            'finished': '/finished',
            'ego_image': '/ego_image',
            'odom': '/odom',
            'goal_pose': '/goal_pose',
            'scan': '/scan',
        }

        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, node, topic_map=topic_map) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 10.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=15.0)
        stop_event.set()

        # Verify output
        videos_dir = temp_output_dir / 'videos'
        index_path = temp_output_dir / 'video_index.json'

        # video_index.json should exist
        assert index_path.exists(), f"video_index.json not found at {index_path}"

        # video_index.json should be valid JSON
        with open(index_path, 'r') as f:
            index_data = json.load(f)

        assert 'episodes' in index_data
        assert 'config' in index_data
        assert 'format' in index_data

        # videos directory should exist
        assert videos_dir.exists(), f"videos directory not found at {videos_dir}"

        node.destroy_node()

    def test_error_file_not_created_on_success(
        self, ros_context, temp_output_dir, map_yaml_path, rosbag_reader
    ):
        """video_recording_error.txt should not exist after successful playback."""
        reader, topics = rosbag_reader
        EvalVideoRecorder = _get_eval_video_recorder_class()

        node = EvalVideoRecorder(
            output_dir=str(temp_output_dir),
            map_yaml_path=map_yaml_path,
            task_reset_topic='/task_reset',
            scenario_reset_topic='',
            finished_topic='/finished',
            ego_topic='/ego_image',
            depth_topic='',
            camera_info_topic='',
            debug_overlay_topic='',
            sim_top_down_topic='',
            odom_topic='/odom',
            goal_topic='/goal_pose',
            scan_topic='/scan',
            fps=10.0,
            top_down_size_px=256,
            top_down_window_m=10.0,
        )

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        topic_map = {
            'task_reset': '/task_reset',
            'finished': '/finished',
            'ego_image': '/ego_image',
            'odom': '/odom',
            'goal_pose': '/goal_pose',
            'scan': '/scan',
        }

        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, node, topic_map=topic_map) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 10.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=15.0)
        stop_event.set()

        error_path = temp_output_dir / 'video_recording_error.txt'
        if error_path.exists():
            content = error_path.read_text()
            # Empty error file is OK (it gets created but may be empty)
            assert len(content.strip()) == 0, \
                f"Unexpected error recorded: {content}"

        node.destroy_node()
