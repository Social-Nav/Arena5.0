#!/usr/bin/env python3
"""
Offline InternNav inference replay from rosbag.

Reads sensor data (RGB, depth, camera_info, odom, goal, instruction) from a
rosbag, replays each frame through the InternVLA-N1-DualVLN model, and
compares the offline actions against the online rollout actions recorded
in the same bag.

Usage (inside internnav container):
    python3 offline_inference_replay.py \
        --rosbag /path/to/rosbag2 \
        --model-path /opt/arena_ws/deps/models/InternVLA-N1-DualVLN \
        --device cuda:0 \
        --output /tmp/offline_replay_result.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── ROS bag reading ──────────────────────────────────────────────────────────

def read_bag(bag_path: str):
    """Read messages from a ros2 bag (mcap format) using rosbag2_py."""
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    except ImportError:
        print("ERROR: rosbag2_py not available. Run inside the arena container.", file=sys.stderr)
        sys.exit(1)

    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_opts = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_opts = ConverterOptions("", "")
    reader = SequentialReader()
    reader.open(storage_opts, converter_opts)

    topic_types = {}
    for t in reader.get_all_topics_and_types():
        topic_types[t.name] = t.type

    messages = {}
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)
        messages.setdefault(topic, []).append((timestamp, msg))

    return messages, topic_types


# ── Message helpers ──────────────────────────────────────────────────────────

def image_msg_to_numpy(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to numpy array."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        image = data.reshape((msg.height, msg.step // 3, 3))[:, :msg.width, :].copy()
        if msg.encoding == "bgr8":
            image = image[:, :, ::-1]
        return image
    if msg.encoding in ("rgba8", "bgra8"):
        image = data.reshape((msg.height, msg.step // 4, 4))[:, :msg.width, :4].copy()
        if msg.encoding == "bgra8":
            image = image[:, :, [2, 1, 0, 3]]
        return image[:, :, :3]
    if msg.encoding == "mono8":
        mono = data.reshape((msg.height, msg.step))[:, :msg.width].copy()
        return np.repeat(mono[:, :, None], 3, axis=2)
    if msg.encoding in ("16UC1", "16SC1", "mono16"):
        depth = data.reshape((msg.height, msg.step // 2)).astype(np.float32)
        # Convert to meters (assuming mm or raw depth units)
        if msg.encoding in ("16UC1", "mono16"):
            depth = depth / 1000.0
        return depth
    if msg.encoding in ("32FC1",):
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width)).copy()
        return depth
    return None


def camera_info_to_intrinsic(msg) -> np.ndarray:
    """Extract 3x3 intrinsic matrix from CameraInfo."""
    K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
    return K


def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_to_pose(msg):
    """Extract (x, y, yaw) from Odometry."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w)


def goal_from_pose_stamped(msg):
    """Extract (x, y, yaw) from PoseStamped."""
    p = msg.pose.position
    q = msg.pose.orientation
    return p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w)


# ── Align messages by timestamp ──────────────────────────────────────────────

def align_frames(messages):
    """
    Build a list of aligned frames, each containing the closest
    rgb, depth, camera_info, odom, and goal to each rgb timestamp.
    """
    rgb_msgs = messages.get("/task_generator_node/Ai2_Bot2/head_camera/image", [])
    depth_msgs = messages.get("/task_generator_node/Ai2_Bot2/head_camera/depth", [])
    info_msgs = messages.get("/task_generator_node/Ai2_Bot2/head_camera/camera_info", [])
    odom_msgs = messages.get("/task_generator_node/Ai2_Bot2/odom", [])
    goal_msgs = messages.get("/task_generator_node/Ai2_Bot2/episode_goal_pose", [])
    instr_msgs = messages.get("/task_generator_node/vln_instruction", [])
    status_msgs = messages.get("/task_generator_node/Ai2_Bot2/internnav/status", [])
    model_output_msgs = messages.get("/task_generator_node/Ai2_Bot2/internnav/model_output", [])
    cmd_vel_msgs = messages.get("/task_generator_node/Ai2_Bot2/cmd_vel", [])

    if not rgb_msgs:
        print("ERROR: No RGB images found in bag", file=sys.stderr)
        return []

    # Get instruction (should be just one)
    instruction = ""
    if instr_msgs:
        instruction = instr_msgs[0][1].data

    # Get goal (use the first one after task_reset)
    goal_x, goal_y, goal_yaw = 0.0, 0.0, 0.0
    if goal_msgs:
        goal_x, goal_y, goal_yaw = goal_from_pose_stamped(goal_msgs[0][1])

    def find_closest(msgs, target_ts):
        if not msgs:
            return None
        best = None
        best_dt = float("inf")
        for ts, msg in msgs:
            dt = abs(ts - target_ts)
            if dt < best_dt:
                best_dt = dt
                best = msg
        return best

    # Build online action trace from status messages
    online_actions = []  # list of (timestamp, discrete_action, linear, angular)
    for ts, msg in status_msgs:
        try:
            data = json.loads(msg.data)
            status = data.get("status", "")
            if status == "internnav_command":
                action = data.get("debug", {}).get("selected_action", -1)
                if action is None:
                    action = data.get("debug", {}).get("discrete_action", -1)
                linear = data.get("linear_x", 0.0)
                angular = data.get("angular_z", 0.0)
                online_actions.append((ts, action, linear, angular))
        except (json.JSONDecodeError, AttributeError):
            pass

    # Build model output trace
    model_outputs = []  # list of (timestamp, parsed dict)
    for ts, msg in model_output_msgs:
        try:
            data = json.loads(msg.data)
            model_outputs.append((ts, data))
        except (json.JSONDecodeError, AttributeError):
            pass

    # Build cmd_vel trace
    cmd_vel_trace = []
    for ts, msg in cmd_vel_msgs:
        cmd_vel_trace.append((ts, msg.linear.x, msg.angular.z))

    frames = []
    for ts, rgb_msg in rgb_msgs:
        depth_msg = find_closest(depth_msgs, ts)
        info_msg = find_closest(info_msgs, ts)
        odom_msg = find_closest(odom_msgs, ts)

        rgb = image_msg_to_numpy(rgb_msg)
        depth = image_msg_to_numpy(depth_msg) if depth_msg else None
        intrinsic = camera_info_to_intrinsic(info_msg) if info_msg else None

        x, y, yaw = 0.0, 0.0, 0.0
        if odom_msg:
            x, y, yaw = odom_to_pose(odom_msg)

        frames.append({
            "timestamp": ts,
            "rgb": rgb,
            "depth": depth,
            "intrinsic": intrinsic,
            "pose": (x, y, yaw),
            "goal": (goal_x, goal_y, goal_yaw),
            "instruction": instruction,
        })

    return frames, online_actions, model_outputs, cmd_vel_trace


# ── Offline inference ────────────────────────────────────────────────────────

def run_offline_inference(frames, model_path, device, inference_rate_hz=0.2):
    """
    Run InternNav model on each frame at the given inference rate.
    Returns list of (frame_idx, discrete_action, infer_time).
    """
    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
    from types import SimpleNamespace

    print(f"Loading InternVLA-N1 model from {model_path} on {device}...")
    agent = InternVLAN1AsyncAgent(SimpleNamespace(
        device=device,
        model_path=model_path,
        resize_w=336,
        resize_h=336,
        num_history=0,
        plan_step_gap=12,
    ))
    print("Model loaded.")

    step_interval = 1.0 / max(inference_rate_hz, 0.01)
    offline_actions = []
    last_infer_time = -step_interval  # ensure first frame is inferred

    for i, frame in enumerate(frames):
        rgb = frame["rgb"]
        depth = frame["depth"]
        intrinsic = frame["intrinsic"]
        pose = list(frame["pose"])
        instruction = frame["instruction"]

        if rgb is None or depth is None or intrinsic is None:
            continue

        # Ensure depth is HxW float32 in meters
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        t0 = time.monotonic()
        try:
            output = agent.step(
                rgb,
                depth,
                pose,
                instruction,
                intrinsic,
                look_down=False,
            )
            # Extract discrete action
            if output.output_action is not None:
                action = int(output.output_action[0]) if len(output.output_action) > 0 else -1
            else:
                action = -1

            trajectory = None
            if output.output_trajectory is not None:
                trajectory = output.output_trajectory.tolist()

        except Exception as e:
            print(f"  Frame {i}: inference error: {e}", file=sys.stderr)
            action = -1
            trajectory = None

        infer_time = time.monotonic() - t0
        offline_actions.append({
            "frame_idx": i,
            "timestamp": frame["timestamp"],
            "discrete_action": action,
            "trajectory": trajectory,
            "pose": pose,
            "infer_time_sec": infer_time,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames, last action={action}, infer_time={infer_time:.3f}s")

    return offline_actions


# ── Comparison ───────────────────────────────────────────────────────────────

def compare_actions(offline_actions, online_actions, model_outputs):
    """
    Compare offline inference actions with online rollout actions.
    """
    # Build a map from timestamp to online action
    online_by_ts = {}
    for ts, action, linear, angular in online_actions:
        online_by_ts[ts] = {"action": action, "linear": linear, "angular": angular}

    # Build a map from timestamp to model output
    model_by_ts = {}
    for ts, data in model_outputs:
        model_by_ts[ts] = data

    # Match offline actions to closest online actions
    online_timestamps = sorted(online_by_ts.keys())

    def find_closest_online(ts):
        if not online_timestamps:
            return None
        best = min(online_timestamps, key=lambda t: abs(t - ts))
        if abs(best - ts) > 1e9:  # > 1 second apart
            return None
        return online_by_ts[best]

    matches = []
    action_counts = {"match": 0, "mismatch": 0, "no_online": 0}
    offline_action_dist = {}
    online_action_dist = {}

    for oa in offline_actions:
        ts = oa["timestamp"]
        offline_action = oa["discrete_action"]
        offline_action_dist[offline_action] = offline_action_dist.get(offline_action, 0) + 1

        online = find_closest_online(ts)
        if online is None:
            matches.append({
                "frame_idx": oa["frame_idx"],
                "offline_action": offline_action,
                "online_action": None,
                "match": False,
                "reason": "no_online",
            })
            action_counts["no_online"] += 1
            continue

        online_action = online["action"]
        online_action_dist[online_action] = online_action_dist.get(online_action, 0) + 1

        match = offline_action == online_action
        if match:
            action_counts["match"] += 1
        else:
            action_counts["mismatch"] += 1

        matches.append({
            "frame_idx": oa["frame_idx"],
            "offline_action": offline_action,
            "online_action": online_action,
            "online_linear": online["linear"],
            "online_angular": online["angular"],
            "match": match,
        })

    total = len(offline_actions)
    result = {
        "total_offline_frames": total,
        "action_comparison": action_counts,
        "match_rate": action_counts["match"] / max(total, 1),
        "offline_action_distribution": {str(k): v for k, v in sorted(offline_action_dist.items())},
        "online_action_distribution": {str(k): v for k, v in sorted(online_action_dist.items())},
        "matches_sample": matches[:50],  # first 50 for inspection
        "all_matches": matches,
    }
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offline InternNav inference replay from rosbag")
    parser.add_argument("--rosbag", required=True, help="Path to ros2 bag directory")
    parser.add_argument("--model-path", default="/opt/arena_ws/deps/models/InternVLA-N1-DualVLN")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-rate-hz", type=float, default=0.2)
    parser.add_argument("--output", default="/tmp/offline_replay_result.json")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0=all)")
    args = parser.parse_args()

    print(f"Reading rosbag from {args.rosbag}...")
    messages, topic_types = read_bag(args.rosbag)

    print("Topic counts:")
    for topic, msgs in sorted(messages.items()):
        print(f"  {topic}: {len(msgs)} messages")

    print("Aligning frames...")
    frames, online_actions, model_outputs, cmd_vel_trace = align_frames(messages)
    print(f"  {len(frames)} frames, {len(online_actions)} online actions, "
          f"{len(model_outputs)} model outputs, {len(cmd_vel_trace)} cmd_vel")

    if args.max_frames > 0:
        frames = frames[:args.max_frames]
        print(f"  Limited to {args.max_frames} frames")

    print(f"\nRunning offline inference on {len(frames)} frames...")
    offline_actions = run_offline_inference(
        frames, args.model_path, args.device, args.inference_rate_hz
    )

    print("\nComparing offline vs online actions...")
    result = compare_actions(offline_actions, online_actions, model_outputs)

    # Add summary info
    result["config"] = {
        "rosbag": args.rosbag,
        "model_path": args.model_path,
        "device": args.device,
        "inference_rate_hz": args.inference_rate_hz,
        "total_frames_in_bag": len(frames),
        "total_online_status_msgs": len(online_actions),
        "total_model_output_msgs": len(model_outputs),
    }

    # Save full result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResult saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Offline inference frames: {result['total_offline_frames']}")
    print(f"  Action matches:          {result['action_comparison']['match']}")
    print(f"  Action mismatches:       {result['action_comparison']['mismatch']}")
    print(f"  No online counterpart:   {result['action_comparison']['no_online']}")
    print(f"  Match rate:              {result['match_rate']:.2%}")
    print(f"  Offline action dist:     {result['offline_action_distribution']}")
    print(f"  Online action dist:      {result['online_action_distribution']}")


if __name__ == "__main__":
    main()
