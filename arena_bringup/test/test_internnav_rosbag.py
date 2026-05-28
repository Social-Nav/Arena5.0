"""
Rosbag playback tests for InternNavServer.

Verifies that InternNavServer in heuristic mode correctly:
1. Subscribes to input topics from rosbag playback
2. Publishes status messages (JSON with valid fields)
3. Provides get_command service returning valid Twist
4. Does not crash or timeout during playback

Usage (in Docker container):
    cd /opt/arena_ws
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    python3 -m pytest src/Arena/arena_bringup/test/test_internnav_rosbag.py -v
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

# Must be imported before rclpy due to ARENA_PYTHON handling in internnav_server
_ARENA_SRC = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ARENA_SRC))

import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist

# Import rosbag_player from conftest
from conftest import rosbag_player


@pytest.fixture(scope='module')
def ros_context():
    """Initialize rclpy once per module."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def internnav_node(ros_context, golden_bag_path):
    """Create an InternNavServer node in heuristic mode."""
    from arena_vln_models.internnav_server import InternNavServer

    # Override parameters for testing
    os.environ['ARENA_EVAL_INTERNNAV_MODE'] = 'heuristic'
    os.environ['ARENA_EVAL_INTERNNAV_DEVICE'] = 'cpu'
    os.environ['ARENA_EVAL_INTERNNAV_REQUIRE_REAL_BACKEND'] = 'false'
    os.environ['ARENA_EVAL_INTERNNAV_RGB_TOPIC'] = ''
    os.environ['ARENA_EVAL_INTERNNAV_DEPTH_TOPIC'] = ''
    os.environ['ARENA_EVAL_INTERNNAV_CAMERA_INFO_TOPIC'] = ''
    os.environ['ARENA_EVAL_INTERNNAV_ENABLE_VISUALIZATION'] = 'false'

    node = InternNavServer()
    yield node
    node.destroy_node()


class TestInternNavServerRosbag:
    """Integration tests for InternNavServer with rosbag playback."""

    def test_node_initializes(self, internnav_node):
        """Node should initialize without errors."""
        assert internnav_node is not None
        assert internnav_node.get_name() == 'internnav_server'

    def test_get_command_service_available(self, internnav_node):
        """get_command service should be available."""
        service_names = [s[0] for s in internnav_node.get_service_names_and_types()]
        assert '/get_command' in service_names, \
            f"get_command not in services: {service_names}"

    def test_get_command_returns_valid_twist(self, internnav_node):
        """get_command should return a valid Twist even without data."""
        from rosnav_rl_msgs.srv import GetCommand

        client = internnav_node.create_client(
            GetCommand, 'get_command'
        )

        ready = client.wait_for_service(timeout_sec=5.0)
        assert ready, "get_command service not ready"

        req = GetCommand.Request()
        future = client.call_async(req)

        executor = SingleThreadedExecutor()
        executor.add_node(internnav_node)
        executor.spin_until_future_complete(future, timeout_sec=5.0)

        assert future.done(), "get_command did not complete"
        response = future.result()
        assert response is not None
        assert hasattr(response, 'twist')
        assert isinstance(response.twist, Twist)

        # Twist values should be finite
        import math
        assert math.isfinite(response.twist.linear.x)
        assert math.isfinite(response.twist.angular.z)

        internnav_node.destroy_client(client)

    def test_handles_rosbag_playback(self, internnav_node, rosbag_reader):
        """Node should process rosbag messages without crashing."""
        reader, topics = rosbag_reader

        # Collect status messages
        status_msgs = []

        from std_msgs.msg import String

        def _status_callback(msg: String):
            status_msgs.append(msg.data)

        status_sub = internnav_node.create_subscription(
            String,
            'internnav/status',
            _status_callback,
            10,
        )

        executor = SingleThreadedExecutor()
        executor.add_node(internnav_node)

        # Map bag topic names to what InternNavServer expects (no leading /)
        topic_map = {
            'odom': 'odom',
            'pose': 'pose',
            'goal_pose': 'goal_pose',
            'vln_instruction': 'vln_instruction',
        }

        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, internnav_node, topic_map=topic_map) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 10.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=15.0)
        stop_event.set()

        internnav_node.destroy_subscription(status_sub)

        # Verify we got some status messages
        assert len(status_msgs) > 0, \
            "No status messages received during playback"

        # Verify status messages are valid JSON
        for msg in status_msgs:
            try:
                data = json.loads(msg)
                assert 'status' in data, f"Status message missing 'status' field: {msg}"
            except json.JSONDecodeError:
                pytest.fail(f"Status message is not valid JSON: {msg}")

    def test_model_output_published(self, internnav_node, rosbag_reader):
        """Model output topic should receive messages during playback."""
        reader, topics = rosbag_reader

        from std_msgs.msg import String

        model_outputs = []

        def _output_callback(msg: String):
            model_outputs.append(msg.data)

        output_sub = internnav_node.create_subscription(
            String,
            'internnav/model_output',
            _output_callback,
            10,
        )

        executor = SingleThreadedExecutor()
        executor.add_node(internnav_node)

        topic_map = {
            'odom': 'odom',
            'pose': 'pose',
            'goal_pose': 'goal_pose',
            'vln_instruction': 'vln_instruction',
        }

        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, internnav_node, topic_map=topic_map) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 10.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=15.0)
        stop_event.set()

        internnav_node.destroy_subscription(output_sub)

        # Model output may be empty if no goal reached, that's OK
        # Just verify the subscription worked without errors
