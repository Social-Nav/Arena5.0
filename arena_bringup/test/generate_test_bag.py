#!/usr/bin/env python3
"""
Generate a synthetic golden rosbag for node testing.

Creates a rosbag with known test data covering the topics needed by:
- InternNavServer (heuristic mode): odom, pose, goal_pose, vln_instruction
- EvalVideoRecorder: task_reset, ego (RGB), odom, goal, scan

Usage (in Docker container):
    cd /opt/arena_ws
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    python3 src/Arena/arena_bringup/test/generate_test_bag.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# Add Arena source to path for message imports
_ARENA_SRC = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ARENA_SRC))

import rclpy
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion, Pose, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Int16, Empty, Header
from sensor_msgs.msg import Image, LaserScan
from builtin_interfaces.msg import Time as RosTime


OUTPUT_DIR = Path(__file__).parent / 'fixtures' / 'test_bag'
NUM_MESSAGES = 100  # ~10 seconds at 10Hz


def _make_header(frame_id: str, sec: int, nanosec: int = 0) -> Header:
    h = Header()
    h.frame_id = frame_id
    h.stamp = RosTime(sec=sec, nanosec=nanosec)
    return h


def _make_pose(x: float, y: float, yaw: float) -> Pose:
    p = Pose()
    p.position = Point(x=x, y=y, z=0.0)
    # Simple yaw-only quaternion
    p.orientation = Quaternion(
        x=0.0, y=0.0, z=np.sin(yaw / 2.0), w=np.cos(yaw / 2.0)
    )
    return p


def generate_odometry(t: int) -> Odometry:
    """Generate a simple forward-moving odometry message."""
    msg = Odometry()
    msg.header = _make_header('odom', t, 0)
    msg.child_frame_id = 'base_link'
    x = t * 0.05  # 0.05 m/s forward
    msg.pose.pose = _make_pose(x, 0.0, 0.0)
    msg.twist.twist = Twist(linear=Vector3(x=0.05, y=0.0, z=0.0))
    return msg


def generate_pose(t: int) -> PoseStamped:
    """Generate a pose matching the odometry."""
    msg = PoseStamped()
    msg.header = _make_header('map', t, 0)
    x = t * 0.05
    msg.pose = _make_pose(x, 0.0, 0.0)
    return msg


def generate_goal_pose() -> PoseStamped:
    """Generate a goal pose 5m ahead."""
    msg = PoseStamped()
    msg.header = _make_header('map', 0, 0)
    msg.pose = _make_pose(5.0, 0.0, 0.0)
    return msg


def generate_instruction() -> String:
    """Generate a VLN instruction."""
    msg = String()
    msg.data = 'navigate forward 5 meters'
    return msg


def generate_task_reset(episode: int = 0) -> Int16:
    """Generate a task reset signal."""
    msg = Int16()
    msg.data = episode
    return msg


def generate_finished() -> Empty:
    """Generate a finished signal."""
    return Empty()


def generate_rgb_image(t: int) -> Image:
    """Generate a synthetic 64x64 RGB image."""
    msg = Image()
    msg.header = _make_header('head_camera_link', t, 0)
    msg.height = 64
    msg.width = 64
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = 64 * 3
    # Simple gradient pattern
    data = np.zeros((64, 64, 3), dtype=np.uint8)
    data[:, :, 0] = (t * 2) % 256  # R channel varies with time
    data[:, :, 1] = 128
    data[:, :, 2] = 64
    msg.data = data.tobytes()
    return msg


def generate_scan(t: int) -> LaserScan:
    """Generate a synthetic laser scan."""
    msg = LaserScan()
    msg.header = _make_header('base_link', t, 0)
    msg.angle_min = -np.pi
    msg.angle_max = np.pi
    msg.angle_increment = np.pi / 180.0  # 1 degree
    msg.time_increment = 0.0
    msg.scan_time = 0.1
    msg.range_min = 0.1
    msg.range_max = 30.0
    num_readings = 360
    ranges = np.ones(num_readings) * 10.0  # 10m in all directions
    # Add a "wall" in front
    ranges[175:185] = 1.0
    msg.ranges = ranges.tolist()
    msg.intensities = [0.0] * num_readings
    return msg


def main():
    rclpy.init()

    # Remove existing bag directory (rosbag2 won't overwrite)
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    # Don't create the directory — rosbag2 will create it

    storage = StorageOptions(
        uri=str(OUTPUT_DIR),
        storage_id='sqlite3',
    )
    writer = SequentialWriter()
    writer.open(storage, ConverterOptions())

    # Define topics
    topic_defs = [
        ('odom', 'nav_msgs/msg/Odometry', generate_odometry),
        ('pose', 'geometry_msgs/msg/PoseStamped', generate_pose),
        ('goal_pose', 'geometry_msgs/msg/PoseStamped', lambda t: generate_goal_pose()),
        ('vln_instruction', 'std_msgs/msg/String', lambda t: generate_instruction()),
        ('task_reset', 'std_msgs/msg/Int16', lambda t: generate_task_reset(0)),
        ('finished', 'std_msgs/msg/Empty', lambda t: generate_finished()),
        ('ego_image', 'sensor_msgs/msg/Image', generate_rgb_image),
        ('scan', 'sensor_msgs/msg/LaserScan', generate_scan),
    ]

    # Create topics
    for i, (topic_name, topic_type, _) in enumerate(topic_defs):
        writer.create_topic(TopicMetadata(
            id=i,
            name=topic_name,
            type=topic_type,
            serialization_format='cdr',
        ))

    # Write messages
    for t in range(NUM_MESSAGES):
        timestamp_ns = t * 100_000_000  # 100ms intervals = 10Hz
        for topic_name, _, generator in topic_defs:
            msg = generator(t)
            writer.write(topic_name, serialize_message(msg), timestamp_ns)

    writer.close()
    print(f"Golden test bag written to {OUTPUT_DIR}")
    print(f"Topics: {[t[0] for t in topic_defs]}")
    print(f"Messages per topic: {NUM_MESSAGES}")


if __name__ == '__main__':
    main()
