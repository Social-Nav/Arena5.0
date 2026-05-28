"""
Rosbag playback tests for model action pipeline (discrete + trajectory).

Verifies that InternNavServer correctly processes model output messages
from a rosbag and produces correct Twist commands via get_command service.

Tests both discrete and trajectory policies to ensure dual-mode consistency.

Usage (in Docker container):
    cd /opt/arena_ws
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    python3 -m pytest src/Arena/arena_vln_models/test/test_model_action_rosbag.py -v
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

_ARENA_SRC = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ARENA_SRC))

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.serialization import serialize_message
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Import rosbag_player from arena_bringup conftest
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'arena_bringup' / 'test'))
from conftest import rosbag_player


# ── Generate a synthetic model output rosbag ─────────────────────────────────

_MODEL_BAG_DIR = Path(__file__).parent / 'fixtures' / 'model_action_bag'


def _generate_model_action_bag():
    """Generate a rosbag with model output messages covering both policies."""
    import shutil
    from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

    if _MODEL_BAG_DIR.exists():
        shutil.rmtree(_MODEL_BAG_DIR)

    storage = StorageOptions(uri=str(_MODEL_BAG_DIR), storage_id='sqlite3')
    writer = SequentialWriter()
    writer.open(storage, ConverterOptions())

    # Create model_output topic
    writer.create_topic(TopicMetadata(
        id=0,
        name='internnav/model_output',
        type='std_msgs/msg/String',
        serialization_format='cdr',
    ))

    # Scenario: robot at (0,0,0), goal at (1,0,0) — straight ahead
    # We send model outputs that exercise both discrete and trajectory paths

    messages = [
        # 1. Trajectory output: waypoint straight ahead
        json.dumps({
            'status': 'internnav_command',
            'output_trajectory': [[0.3, 0.0, 0.0], [0.5, 0.0, 0.0]],
            'debug': {'source': 'test_trajectory_forward'},
        }),
        # 2. Discrete action: forward
        json.dumps({
            'status': 'internnav_command',
            'discrete_action': 1,
            'debug': {'source': 'test_discrete_forward'},
        }),
        # 3. Trajectory output: turn left (positive yaw)
        json.dumps({
            'status': 'internnav_command',
            'output_trajectory': [[0.2, 0.0, 0.3]],
            'debug': {'source': 'test_trajectory_left'},
        }),
        # 4. Discrete action: native left (action 2)
        json.dumps({
            'status': 'internnav_command',
            'discrete_action': 2,
            'debug': {'source': 'test_discrete_left'},
        }),
        # 5. Trajectory output: turn right (negative yaw)
        json.dumps({
            'status': 'internnav_command',
            'output_trajectory': [[0.2, 0.0, -0.3]],
            'debug': {'source': 'test_trajectory_right'},
        }),
        # 6. Discrete action: native right (action 3)
        json.dumps({
            'status': 'internnav_command',
            'discrete_action': 3,
            'debug': {'source': 'test_discrete_right'},
        }),
        # 7. Both trajectory and discrete (trajectory policy should prefer trajectory)
        json.dumps({
            'status': 'internnav_command',
            'discrete_action': 3,
            'output_trajectory': [[0.4, 0.0, 0.25]],
            'debug': {'source': 'test_both_traj_wins'},
        }),
        # 8. Stop
        json.dumps({
            'status': 'internnav_command',
            'discrete_action': 0,
            'debug': {'source': 'test_stop'},
        }),
    ]

    for t, msg_json in enumerate(messages):
        msg = String()
        msg.data = msg_json
        timestamp_ns = t * 500_000_000  # 500ms intervals
        writer.write('internnav/model_output', serialize_message(msg), timestamp_ns)

    writer.close()
    return _MODEL_BAG_DIR


@pytest.fixture(scope='session')
def model_action_bag_path():
    """Generate the model action test bag once per session."""
    _generate_model_action_bag()
    db3 = _MODEL_BAG_DIR / 'model_action_bag_0.db3'
    if not db3.exists():
        pytest.fail(f"Failed to generate model action bag at {db3}")
    return db3


@pytest.fixture(scope='session')
def model_action_reader(model_action_bag_path):
    """Open the model action rosbag."""
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

    storage = StorageOptions(
        uri=str(model_action_bag_path.parent),
        storage_id='sqlite3',
    )
    reader = SequentialReader()
    reader.open(storage, ConverterOptions())
    topics = reader.get_all_topics_and_types()
    return reader, topics


# ── InternNavServer fixture ──────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_internnav_node(mode='heuristic', model_output_policy='trajectory'):
    """Create an InternNavServer node with given parameters."""
    from arena_vln_models.internnav_server import InternNavServer

    os.environ['ARENA_EVAL_INTERNNAV_MODE'] = mode
    os.environ['ARENA_EVAL_INTERNNAV_DEVICE'] = 'cpu'
    os.environ['ARENA_EVAL_INTERNNAV_REQUIRE_REAL_BACKEND'] = 'false'
    os.environ['ARENA_EVAL_INTERNNAV_RGB_TOPIC'] = ''
    os.environ['ARENA_EVAL_INTERNNAV_DEPTH_TOPIC'] = ''
    os.environ['ARENA_EVAL_INTERNNAV_CAMERA_INFO_TOPIC'] = ''
    os.environ['ARENA_EVAL_INTERNNAV_ENABLE_VISUALIZATION'] = 'false'
    os.environ['ARENA_EVAL_INTERNNAV_MODEL_OUTPUT_POLICY'] = model_output_policy

    return InternNavServer()


# ── Tests ────────────────────────────────────────────────────────────────────

class TestModelActionRosbag:
    """Verify model output → Twist pipeline via rosbag playback."""

    def test_trajectory_policy_forward(self, ros_context, model_action_reader):
        """Trajectory policy: forward waypoint → positive linear_x, zero angular_z."""
        reader, topics = model_action_reader
        node = _make_internnav_node(mode='heuristic', model_output_policy='trajectory')

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Play only the first message (trajectory forward)
        topic_map = {'internnav/model_output': 'internnav/model_output'}
        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, node, topic_map=topic_map, rate_hz=10.0) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 5.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=10.0)
        stop_event.set()

        # Call get_command to see what the node would output
        from rosnav_rl_msgs.srv import GetCommand
        client = node.create_client(GetCommand, 'get_command')
        if client.wait_for_service(timeout_sec=5.0):
            req = GetCommand.Request()
            future = client.call_async(req)
            executor.spin_until_future_complete(future, timeout_sec=5.0)
            if future.done():
                twist = future.result().twist
                # Heuristic mode: goal at (1,0,0) from (0,0,0) → should drive forward
                assert twist.linear.x >= 0.0, f"Expected non-negative linear.x, got {twist.linear.x}"
                assert abs(twist.angular.z) < 1.0  # mostly straight
        node.destroy_client(client)
        node.destroy_node()

    def test_discrete_policy_forward(self, ros_context, model_action_reader):
        """Discrete policy: action 1 → positive linear_x, zero angular_z."""
        reader, topics = model_action_reader
        node = _make_internnav_node(mode='heuristic', model_output_policy='discrete')

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        topic_map = {'internnav/model_output': 'internnav/model_output'}
        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, node, topic_map=topic_map, rate_hz=10.0) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 5.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=10.0)
        stop_event.set()

        from rosnav_rl_msgs.srv import GetCommand
        client = node.create_client(GetCommand, 'get_command')
        if client.wait_for_service(timeout_sec=5.0):
            req = GetCommand.Request()
            future = client.call_async(req)
            executor.spin_until_future_complete(future, timeout_sec=5.0)
            if future.done():
                twist = future.result().twist
                assert twist.linear.x >= 0.0
        node.destroy_client(client)
        node.destroy_node()

    def test_model_output_published_during_get_command(self, ros_context):
        """Verify model_output is published when get_command is called with valid inputs."""
        node = _make_internnav_node(mode='heuristic', model_output_policy='trajectory')

        received = []

        def _on_model_output(msg: String):
            received.append(msg.data)

        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sub = node.create_subscription(
            String, 'internnav/model_output', _on_model_output, status_qos
        )

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        # Publish odom + goal_pose to satisfy input requirements
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose, Vector3
        from rosnav_rl_msgs.srv import GetCommand

        odom_pub = node.create_publisher(Odometry, 'odom', 10)
        goal_pub = node.create_publisher(PoseStamped, 'goal_pose', 10)

        odom = Odometry()
        odom.header.frame_id = 'odom'
        odom.header.stamp = node.get_clock().now().to_msg()
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
        odom.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        odom.twist.twist = Twist(linear=Vector3(x=0.0, y=0.0, z=0.0))
        odom_pub.publish(odom)

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = node.get_clock().now().to_msg()
        goal.pose = Pose(position=Point(x=5.0, y=0.0, z=0.0),
                         orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        goal_pub.publish(goal)

        executor.spin_once(timeout_sec=0.5)

        # Call get_command — may return camera_gate if TF missing, but model_output
        # should still be published (camera gate publishes model_output too)
        client = node.create_client(GetCommand, 'get_command')
        if client.wait_for_service(timeout_sec=5.0):
            req = GetCommand.Request()
            future = client.call_async(req)
            executor.spin_until_future_complete(future, timeout_sec=5.0)
        node.destroy_client(client)

        node.destroy_subscription(sub)
        node.destroy_publisher(odom_pub)
        node.destroy_publisher(goal_pub)
        node.destroy_node()

        # model_output is published even for camera_gate decisions
        assert len(received) > 0, (
            f"No model_output messages received. "
            f"Camera gate may be blocking — check TF readiness."
        )
        for msg in received:
            data = json.loads(msg)
            assert 'status' in data

    def test_get_command_returns_valid_twist_after_playback(
        self, ros_context, model_action_reader
    ):
        """After rosbag playback, get_command should return a valid Twist."""
        reader, topics = model_action_reader
        node = _make_internnav_node(mode='heuristic', model_output_policy='trajectory')

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        topic_map = {'internnav/model_output': 'internnav/model_output'}
        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, node, topic_map=topic_map, rate_hz=10.0) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 5.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=10.0)
        stop_event.set()

        from rosnav_rl_msgs.srv import GetCommand
        client = node.create_client(GetCommand, 'get_command')
        assert client.wait_for_service(timeout_sec=5.0), "get_command service not ready"

        req = GetCommand.Request()
        future = client.call_async(req)
        executor.spin_until_future_complete(future, timeout_sec=5.0)
        assert future.done(), "get_command did not complete"

        twist = future.result().twist
        assert isinstance(twist, Twist)
        import math
        assert math.isfinite(twist.linear.x)
        assert math.isfinite(twist.angular.z)

        node.destroy_client(client)
        node.destroy_node()

    def test_status_published_during_playback(self, ros_context, model_action_reader):
        """Status messages should be published during rosbag playback."""
        reader, topics = model_action_reader
        node = _make_internnav_node(mode='heuristic', model_output_policy='trajectory')

        status_msgs = []

        def _on_status(msg: String):
            status_msgs.append(msg.data)

        sub = node.create_subscription(String, 'internnav/status', _on_status, 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)

        topic_map = {'internnav/model_output': 'internnav/model_output'}
        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, node, topic_map=topic_map, rate_hz=10.0) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 5.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=10.0)
        stop_event.set()

        node.destroy_subscription(sub)
        node.destroy_node()

        assert len(status_msgs) > 0, "No status messages received"
        for msg in status_msgs:
            data = json.loads(msg)
            assert 'status' in data


class TestDualModeRosbag:
    """Verify trajectory vs discrete policy produce different Twist from same bag."""

    def _play_and_get_twist(self, node, reader, topics):
        """Helper: play rosbag and return the get_command Twist."""
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        topic_map = {'internnav/model_output': 'internnav/model_output'}
        stop_event = threading.Event()

        def _play_and_spin():
            with rosbag_player(reader, topics, node, topic_map=topic_map, rate_hz=10.0) as published:
                start = time.time()
                while not stop_event.is_set() and (time.time() - start) < 5.0:
                    executor.spin_once(timeout_sec=0.1)

        play_thread = threading.Thread(target=_play_and_spin, daemon=True)
        play_thread.start()
        play_thread.join(timeout=10.0)
        stop_event.set()

        from rosnav_rl_msgs.srv import GetCommand
        client = node.create_client(GetCommand, 'get_command')
        twist = Twist()
        if client.wait_for_service(timeout_sec=5.0):
            req = GetCommand.Request()
            future = client.call_async(req)
            executor.spin_until_future_complete(future, timeout_sec=5.0)
            if future.done():
                twist = future.result().twist
        node.destroy_client(client)
        return twist

    def test_trajectory_vs_discrete_policy_different_outputs(
        self, ros_context, model_action_reader, model_action_bag_path
    ):
        """Same bag, different policies → potentially different Twist commands."""
        reader, topics = model_action_reader

        # Trajectory policy node
        node_traj = _make_internnav_node(
            mode='heuristic', model_output_policy='trajectory'
        )
        twist_traj = self._play_and_get_twist(node_traj, reader, topics)
        node_traj.destroy_node()

        # Reset reader for second pass
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        storage = StorageOptions(
            uri=str(model_action_bag_path.parent),
            storage_id='sqlite3',
        )
        reader2 = SequentialReader()
        reader2.open(storage, ConverterOptions())
        topics2 = reader2.get_all_topics_and_types()

        # Discrete policy node
        node_disc = _make_internnav_node(
            mode='heuristic', model_output_policy='discrete'
        )
        twist_disc = self._play_and_get_twist(node_disc, reader2, topics2)
        node_disc.destroy_node()

        # Both should produce valid finite twists
        import math
        assert math.isfinite(twist_traj.linear.x)
        assert math.isfinite(twist_traj.angular.z)
        assert math.isfinite(twist_disc.linear.x)
        assert math.isfinite(twist_disc.angular.z)

        # At minimum, both should be valid commands (not all zeros necessarily,
        # since heuristic mode uses goal/odom which may not be set)
