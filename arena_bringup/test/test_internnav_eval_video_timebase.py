"""CPU-only tests for the Task34 sim-time video timebase.

The recorder lives inside ``internnav_eval.py``, which imports rclpy at module
scope.  These tests therefore extract the relevant definitions with ``ast`` and
exercise them against a bare object, so they run without a ROS graph.
"""
import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw


SOURCE_PATH = Path(__file__).parents[1] / 'arena_bringup' / 'internnav_eval.py'


def _recorder_source():
    tree = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'))
    start = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == '_start_eval_video_recorder'
    )
    assignment = next(
        node
        for node in start.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'recorder_code' for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def _namespace():
    tree = ast.parse(_recorder_source())
    wanted = {'_message_stamp_ns', 'EvalVideoRecorder'}
    selected = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted)
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    message_type = type('Message', (), {})
    namespace = {
        'json': json,
        'math': __import__('math'),
        'os': __import__('os'),
        'time': time,
        'Path': Path,
        'np': np,
        'PILImage': PILImage,
        'ImageDraw': ImageDraw,
        'Node': object,
        'Int16': message_type,
        'Empty': message_type,
        'Image': message_type,
        'CameraInfo': message_type,
        'Odometry': message_type,
        'PoseStamped': message_type,
        'LaserScan': message_type,
        'BACKEND_NAME': 'cv2',
        'BACKEND_MODULE': None,
        'EGO_RENDER_SETTLING_PREFIX': 'ego_render_settling',
        'EPISODE_START_TIMEOUT_PREFIX': 'episode_start_timeout',
        'VIDEO_RECORDER_FINALIZATION_DEADLINE_SEC': 10.0,
    }
    exec(compile(module, '<video-timebase-definitions>', 'exec'), namespace)
    return namespace


def _recorder(namespace):
    recorder = object.__new__(namespace['EvalVideoRecorder'])
    recorder.latest_ego_stamp_ns = None
    recorder.latest_odom_stamp_ns = None
    recorder.sim_clock_fallback_count = 0
    recorder.latest_pose = None
    recorder.latest_goal = None
    recorder.latest_scan = []
    recorder.fps = 20.0
    recorder.frame_pts = {}
    return recorder


def _stamped(sec, nanosec=0):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)))


def test_message_stamp_ns_handles_missing_and_zero():
    namespace = _namespace()
    stamp_ns = namespace['_message_stamp_ns']
    assert stamp_ns(SimpleNamespace()) is None
    assert stamp_ns(SimpleNamespace(header=None)) is None
    assert stamp_ns(_stamped(0, 0)) is None
    assert stamp_ns(_stamped(12, 500_000_000)) == 12_500_000_000


def test_sim_clock_prefers_ego_then_odom_then_wall():
    namespace = _namespace()
    recorder = _recorder(namespace)

    ego_ns, source = recorder._sample_sim_clock_ns()
    assert source == 'wall_monotonic_fallback'
    assert ego_ns > 0

    recorder.latest_odom_stamp_ns = 5_000_000_000
    odom_ns, source = recorder._sample_sim_clock_ns()
    assert (odom_ns, source) == (5_000_000_000, 'odom_header_stamp')

    recorder.latest_ego_stamp_ns = 9_000_000_000
    ego_ns, source = recorder._sample_sim_clock_ns()
    assert (ego_ns, source) == (9_000_000_000, 'ego_header_stamp')


def test_map_signature_changes_only_with_new_map_data():
    namespace = _namespace()
    recorder = _recorder(namespace)
    recorder.latest_pose = {'x': 1.0, 'y': 2.0, 'yaw': 0.5}
    recorder.latest_goal = {'x': 3.0, 'y': 4.0}
    recorder.latest_scan = [(1.0, 0.0), (2.0, 0.1)]

    first = recorder._map_signature()
    assert recorder._map_signature() == first

    recorder.latest_scan = [(1.0, 0.0), (2.0, 0.2)]
    assert recorder._map_signature() != first

    second = recorder._map_signature()
    recorder.latest_pose = {'x': 1.5, 'y': 2.0, 'yaw': 0.5}
    assert recorder._map_signature() != second


def test_annotate_timebase_stamps_top_right_corner():
    namespace = _namespace()
    recorder = _recorder(namespace)
    frame = np.zeros((90, 160, 3), dtype=np.uint8)

    annotated = recorder._annotate_timebase(frame, sim_elapsed_sec=2.0, video_pts_sec=1.0, x_ratio=2.0)

    assert annotated.dtype == np.uint8
    assert annotated.shape == frame.shape
    # The yellow timebase text must land in the top-right corner ...
    assert int(annotated[4:26, -80:].sum()) > 0
    # ... and the untouched bottom-left must stay black.
    assert int(annotated[-20:, :20].sum()) == 0


def test_record_frame_pts_appends_rounded_entries():
    namespace = _namespace()
    recorder = _recorder(namespace)

    recorder._record_frame_pts('ego', 3, 0.1234567, 0.15, 0.823045)
    recorder._record_frame_pts('ego', 4, 0.1734567, 0.20, 0.867283)

    assert len(recorder.frame_pts['ego']) == 2
    assert recorder.frame_pts['ego'][0] == {
        'frame': 3,
        'sim_elapsed_sec': 0.123457,
        'video_pts_sec': 0.15,
        'xN': 0.823045,
    }
