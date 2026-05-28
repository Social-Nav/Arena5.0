"""
Shared fixtures for rosbag-based node testing.

Provides:
- golden_bag_path: resolves path to the golden test rosbag
- rosbag_reader: opens a rosbag and returns (reader, topic_list)
- rosbag_player: context manager that plays bag messages on a simulated clock
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import pytest

# Path to test fixtures directory
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
GOLDEN_BAG_DIR = FIXTURES_DIR / 'test_bag'
GOLDEN_BAG_PATH = GOLDEN_BAG_DIR / 'test_bag_0.db3'


@pytest.fixture(scope='session')
def golden_bag_path() -> Path:
    """Path to the golden test rosbag.

    Returns the path if it exists, otherwise skips tests that depend on it.
    """
    if not GOLDEN_BAG_PATH.exists():
        pytest.skip(
            f"Golden test bag not found at {GOLDEN_BAG_PATH}. "
            f"Run 'python generate_test_bag.py' in Docker to create it."
        )
    return GOLDEN_BAG_PATH


@pytest.fixture(scope='session')
def rosbag_reader(golden_bag_path: Path):
    """Open the golden rosbag and return (reader, topic_list).

    Returns:
        Tuple of (SequentialReader, list of TopicMetadata)
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

    storage = StorageOptions(
        uri=str(golden_bag_path.parent),  # rosbag2 needs the directory, not the .db3 file
        storage_id='sqlite3',
    )
    reader = SequentialReader()
    reader.open(storage, ConverterOptions())
    topics = reader.get_all_topics_and_types()
    return reader, topics


@contextmanager
def rosbag_player(reader, topics, node, topic_map=None, rate_hz=100.0):
    """Context manager that plays rosbag messages to a node's input topics.

    Args:
        reader: SequentialReader opened on the bag
        topics: list of TopicMetadata from get_all_topics_and_types()
        node: rclpy Node to publish to
        topic_map: optional dict mapping bag topic names to node input topic names
        rate_hz: playback rate in Hz

    Yields:
        dict of topic_name -> list of published messages
    """
    import threading
    import time

    import rclpy
    from rclpy.serialization import deserialize_message
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

    if topic_map is None:
        topic_map = {}

    # Create publishers for each topic in the bag
    publishers = {}
    published_msgs = {}
    for topic in topics:
        output_topic = topic_map.get(topic.name, topic.name)
        try:
            msg_type = _import_msg_type(topic.type)
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            pub = node.create_publisher(msg_type, output_topic, qos)
            publishers[topic.name] = pub
            published_msgs[topic.name] = []
        except (ImportError, AttributeError) as e:
            node.get_logger().warn(f"Cannot create publisher for {topic.name}: {e}")

    stop_event = threading.Event()
    play_thread = None

    def _play():
        while not stop_event.is_set() and reader.has_next():
            topic_name, serialized_data, timestamp = reader.read_next()
            if topic_name in publishers:
                pub = publishers[topic_name]
                msg = deserialize_message(serialized_data, pub.msg_type)
                pub.publish(msg)
                published_msgs[topic_name].append(msg)
            time.sleep(1.0 / rate_hz)

    play_thread = threading.Thread(target=_play, daemon=True)
    play_thread.start()

    try:
        yield published_msgs
    finally:
        stop_event.set()
        if play_thread:
            play_thread.join(timeout=5.0)
        for pub in publishers.values():
            node.destroy_publisher(pub)


def _import_msg_type(type_str: str):
    """Import a ROS message type from its string representation.

    'sensor_msgs/msg/Image' -> sensor_msgs.msg.Image
    'nav_msgs/msg/Odometry' -> nav_msgs.msg.Odometry
    """
    parts = type_str.split('/')
    if len(parts) == 3:
        package, subfolder, msg_name = parts
        module_name = f"{package}.{subfolder}"
    elif len(parts) == 2:
        package, msg_name = parts
        module_name = f"{package}.msg"
    else:
        raise ValueError(f"Invalid message type: {type_str}")

    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, msg_name)
