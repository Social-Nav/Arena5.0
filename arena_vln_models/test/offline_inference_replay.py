#!/usr/bin/env python3
"""
Offline InternNav inference replay and ego-action overlay from rosbag/eval artifacts.

Reads sensor data (RGB, depth, camera_info, odom, goal, instruction) from a
rosbag when those topics are present, replays frames through the
InternVLA-N1-DualVLN model, and compares offline actions against online rollout
actions recorded in the same bag.  For Arena eval directories whose rosbag only
contains low-bandwidth topics, it can still pair `ego_observation.mp4` with
`internnav_trace.jsonl` to render an audit video that overlays System-2/System-1
state and selected action on the ego image.

Usage (inside internnav container):
    python3 offline_inference_replay.py \
        --rosbag /path/to/rosbag2 \
        --model-path /opt/arena_ws/deps/models/InternVLA-N1-DualVLN \
        --device cuda:0 \
        --output /tmp/offline_replay_result.json
"""

import argparse
import bisect
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# ── ROS bag reading ──────────────────────────────────────────────────────────

def _storage_id_from_metadata(bag_path: Path) -> str:
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.exists():
        return "mcap"
    text = metadata_path.read_text(errors="ignore")
    match = re.search(r"storage_identifier:\s*([^\s]+)", text)
    return match.group(1).strip() if match else "mcap"


def read_bag(bag_path: str, storage_id: str = "auto"):
    """Read messages from a ros2 bag using rosbag2_py.

    Supports Arena's sqlite3 data-recorder bags and MCAP VLN dataset bags.
    """
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    except ImportError:
        print("ERROR: rosbag2_py not available. Run inside the arena container.", file=sys.stderr)
        sys.exit(1)

    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag_dir = Path(bag_path)
    if bag_dir.is_file():
        bag_dir = bag_dir.parent
    resolved_storage = _storage_id_from_metadata(bag_dir) if storage_id == "auto" else storage_id
    storage_opts = StorageOptions(uri=str(bag_dir), storage_id=resolved_storage)
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

    return messages, topic_types, resolved_storage


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

def _topic_suffix(topic: str) -> str:
    return topic.lstrip("/")


def _find_topic(messages, candidates):
    by_suffix = {_topic_suffix(topic): topic for topic in messages}
    for candidate in candidates:
        if candidate in messages:
            return candidate
        stripped = _topic_suffix(candidate)
        if stripped in by_suffix:
            return by_suffix[stripped]
    for topic in messages:
        suffix = _topic_suffix(topic)
        for candidate in candidates:
            if suffix.endswith(_topic_suffix(candidate)):
                return topic
    return None


def align_frames(messages):
    """
    Build a list of aligned frames, each containing the closest
    rgb, depth, camera_info, odom, and goal to each rgb timestamp.
    """
    topic_map = {
        "rgb": _find_topic(messages, [
            "/task_generator_node/Ai2_Bot2/head_camera/image",
            "task_generator_node/Ai2_Bot2/head_camera/image",
            "ego_image",
            "head_camera/image",
        ]),
        "depth": _find_topic(messages, [
            "/task_generator_node/Ai2_Bot2/head_camera/depth",
            "task_generator_node/Ai2_Bot2/head_camera/depth",
            "head_camera/depth",
        ]),
        "camera_info": _find_topic(messages, [
            "/task_generator_node/Ai2_Bot2/head_camera/camera_info",
            "task_generator_node/Ai2_Bot2/head_camera/camera_info",
            "head_camera/camera_info",
        ]),
        "odom": _find_topic(messages, [
            "/task_generator_node/Ai2_Bot2/odom",
            "task_generator_node/Ai2_Bot2/odom",
            "odom",
        ]),
        "goal": _find_topic(messages, [
            "/task_generator_node/Ai2_Bot2/episode_goal_pose",
            "task_generator_node/Ai2_Bot2/episode_goal_pose",
            "goal_pose",
            "episode_goal_pose",
        ]),
        "instruction": _find_topic(messages, [
            "/task_generator_node/vln_instruction",
            "task_generator_node/vln_instruction",
            "vln_instruction",
        ]),
        "status": _find_topic(messages, [
            "/task_generator_node/Ai2_Bot2/internnav/status",
            "task_generator_node/Ai2_Bot2/internnav/status",
            "internnav/status",
        ]),
        "model_output": _find_topic(messages, [
            "/task_generator_node/Ai2_Bot2/internnav/model_output",
            "task_generator_node/Ai2_Bot2/internnav/model_output",
            "internnav/model_output",
        ]),
        "cmd_vel": _find_topic(messages, [
            "/task_generator_node/Ai2_Bot2/cmd_vel",
            "task_generator_node/Ai2_Bot2/cmd_vel",
            "cmd_vel",
        ]),
    }

    rgb_msgs = messages.get(topic_map["rgb"], []) if topic_map["rgb"] else []
    depth_msgs = messages.get(topic_map["depth"], []) if topic_map["depth"] else []
    info_msgs = messages.get(topic_map["camera_info"], []) if topic_map["camera_info"] else []
    odom_msgs = messages.get(topic_map["odom"], []) if topic_map["odom"] else []
    goal_msgs = messages.get(topic_map["goal"], []) if topic_map["goal"] else []
    instr_msgs = messages.get(topic_map["instruction"], []) if topic_map["instruction"] else []
    status_msgs = messages.get(topic_map["status"], []) if topic_map["status"] else []
    model_output_msgs = messages.get(topic_map["model_output"], []) if topic_map["model_output"] else []
    cmd_vel_msgs = messages.get(topic_map["cmd_vel"], []) if topic_map["cmd_vel"] else []

    if not rgb_msgs:
        return [], online_actions if 'online_actions' in locals() else [], [], [], topic_map

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

    return frames, online_actions, model_outputs, cmd_vel_trace, topic_map


# ── Trace/video overlay helpers ──────────────────────────────────────────────

ACTION_LABELS = {
    0: "STOP",
    1: "FWD",
    2: "LEFT",
    3: "RIGHT",
    4: "BACK",
    5: "LOOK_DOWN",
}

ACTION_NAME_TO_ID = {
    "stop": 0,
    "forward": 1,
    "fwd": 1,
    "turn_left": 2,
    "left": 2,
    "turn_right": 3,
    "right": 3,
    "back": 4,
    "backward": 4,
    "look_down": 5,
}


def _safe_json_loads(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def read_trace(trace_path: str):
    records = []
    path = Path(trace_path)
    if not path.exists():
        return records
    with path.open(errors="ignore") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = record.get("parsed") or _safe_json_loads(record.get("raw"))
            if not isinstance(parsed, dict):
                parsed = {}
            record["parsed"] = parsed
            if "wall_time" in record:
                records.append(record)
    records.sort(key=lambda item: float(item.get("wall_time", 0.0)))
    return records


def _trace_for_time(records, ts):
    if not records:
        return None
    times = [float(item.get("wall_time", 0.0)) for item in records]
    idx = bisect.bisect_left(times, ts)
    candidates = []
    if idx < len(records):
        candidates.append(records[idx])
    if idx > 0:
        candidates.append(records[idx - 1])
    return min(candidates, key=lambda item: abs(float(item.get("wall_time", 0.0)) - ts)) if candidates else records[-1]


def _extract_action(parsed):
    debug = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    action_info = parsed.get("action") if isinstance(parsed.get("action"), dict) else {}
    for key in ("selected_action", "discrete_action", "output_action"):
        if key in debug and debug[key] is not None:
            try:
                return int(debug[key])
            except (TypeError, ValueError):
                value = str(debug[key]).strip().lower()
                return ACTION_NAME_TO_ID.get(value, debug[key])
    for key in ("effective_action_label", "native_action_label", "action_label"):
        value = str(debug.get(key, "")).strip().lower()
        if value in ACTION_NAME_TO_ID:
            return ACTION_NAME_TO_ID[value]
    if "selected" in action_info:
        return action_info.get("selected")
    raw_model = debug.get("raw_model_output")
    if isinstance(raw_model, dict):
        value = raw_model.get("discrete_action", raw_model.get("output_action"))
        if isinstance(value, list) and value:
            value = value[0]
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
    return None


def _action_display(parsed, action):
    debug = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    label = debug.get("effective_action_label") or debug.get("native_action_label") or debug.get("action_label")
    if isinstance(label, str) and label.strip():
        if isinstance(action, int):
            return f"{ACTION_LABELS.get(action, action)} / {label.strip()}"
        return label.strip()
    if isinstance(action, int):
        return ACTION_LABELS.get(action, str(action))
    return str(action) if action is not None else "n/a"


def _extract_pixel(parsed):
    debug = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    raw_model = debug.get("raw_model_output") if isinstance(debug.get("raw_model_output"), dict) else {}
    value = parsed.get("output_pixel", raw_model.get("output_pixel", debug.get("target_pixel")))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(round(float(value[0]))), int(round(float(value[1])))
        except (TypeError, ValueError):
            return None
    return None


def _draw_text_block(frame, lines, *, origin=(12, 12), scale=0.52):
    try:
        import cv2
    except ImportError:
        return frame
    x, y = origin
    line_h = int(22 * scale / 0.52)
    max_chars = max((len(line) for line in lines), default=0)
    w = min(frame.shape[1] - x - 8, max(360, int(max_chars * 8 * scale / 0.52)))
    h = min(frame.shape[0] - y - 8, 14 + line_h * len(lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 255), 2)
    yy = y + 20
    for line in lines:
        cv2.putText(frame, line[:120], (x + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        yy += line_h
        if yy > y + h - 4:
            break
    return frame


def _draw_action_glyph_cv(frame, action, linear, angular, display_label=None):
    try:
        import cv2
    except ImportError:
        return frame
    h, w = frame.shape[:2]
    center = (w // 2, h - 48)
    color = (80, 255, 80)
    cv2.circle(frame, center, 7, (255, 255, 255), -1)
    label = display_label or ACTION_LABELS.get(action, str(action) if action is not None else "n/a")
    if action == 0 or (abs(linear) < 1e-6 and abs(angular) < 1e-6):
        cv2.circle(frame, (w - 78, h - 72), 32, (0, 0, 255), 4)
        cv2.line(frame, (w - 98, h - 92), (w - 58, h - 52), (0, 0, 255), 4)
    elif angular > 1e-6:
        cv2.arrowedLine(frame, center, (center[0] - 70, center[1] - 45), color, 5, tipLength=0.35)
    elif angular < -1e-6:
        cv2.arrowedLine(frame, center, (center[0] + 70, center[1] - 45), color, 5, tipLength=0.35)
    elif linear > 1e-6 or action == 1:
        cv2.arrowedLine(frame, center, (center[0], center[1] - 90), color, 6, tipLength=0.3)
    cv2.putText(frame, f"ACTION {str(label)[:24]}", (w - 260, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return frame


def render_trace_overlay_video(ego_video_path: str, trace_path: str, output_video_path: str, *, max_frames: int = 0):
    import cv2

    records = read_trace(trace_path)
    cap = cv2.VideoCapture(str(ego_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open ego video: {ego_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output = Path(output_video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open overlay writer: {output}")

    start_time = float(records[0].get("wall_time", 0.0)) if records else 0.0
    end_time = float(records[-1].get("wall_time", start_time)) if records else start_time
    duration = max(end_time - start_time, frame_count / max(fps, 1e-3), 1e-3)
    written = 0
    summary_counts = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and written >= max_frames:
            break
        ratio = written / max(frame_count - 1, 1) if frame_count else written / max(fps * duration, 1.0)
        trace_ts = start_time + ratio * duration
        record = _trace_for_time(records, trace_ts)
        parsed = record.get("parsed", {}) if record else {}
        debug = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
        action = _extract_action(parsed)
        action_display = _action_display(parsed, action)
        pixel = _extract_pixel(parsed)
        linear = float(parsed.get("linear_x", 0.0) or 0.0)
        angular = float(parsed.get("angular_z", 0.0) or 0.0)
        status = str(parsed.get("status", "n/a"))
        summary_counts[status] = summary_counts.get(status, 0) + 1
        if pixel is not None:
            cv2.drawMarker(frame, pixel, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
            cv2.putText(frame, "System2 pixel goal", (pixel[0] + 8, pixel[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        frame = _draw_action_glyph_cv(frame, action if isinstance(action, int) else None, linear, angular, action_display)
        raw_llm = str(debug.get("subprocess_llm_output") or debug.get("adapter_llm_output") or debug.get("llm_output") or "")
        raw_model = debug.get("raw_model_output") if isinstance(debug.get("raw_model_output"), dict) else {}
        if not raw_llm and isinstance(raw_model.get("debug"), dict):
            raw_llm = str(raw_model["debug"].get("adapter_llm_output") or raw_model["debug"].get("subprocess_llm_output") or "")
        trajectory = raw_model.get("output_trajectory") or parsed.get("output_trajectory") or debug.get("trajectory_preview")
        tail = debug.get("action_sequence_tail") or debug.get("dropped_action_tail") or debug.get("official_discrete_dropped_action_tail")
        lines = [
            f"frame={written} status={status} action={action_display} vx={linear:.3f} wz={angular:.3f}",
            f"System2 input: rgb={not debug.get('missing_rgb', False)} depth={not debug.get('missing_depth', False)} instr={str(debug.get('instruction_normalized', ''))[:32]}",
            f"System2 output: llm={raw_llm[:72] if raw_llm else 'n/a'} pixel={pixel} action_tail={tail if tail is not None else 'n/a'}",
            f"System1 input: latent_pending={debug.get('adapter_output_latent_pending', debug.get('subprocess_output_latent_pending', 'n/a'))} pixel_goal={pixel}",
            f"System1 output: traj={'yes' if trajectory is not None else 'no'} mode={debug.get('selected_output_mode', debug.get('model_output_policy', 'n/a'))}",
            f"goal_dist={debug.get('goal_distance', 'n/a')} yaw_err={debug.get('yaw_error', 'n/a')}",
        ]
        frame = _draw_text_block(frame, lines)
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    return {
        "overlay_video": str(output),
        "ego_video": str(ego_video_path),
        "trace_path": str(trace_path),
        "trace_records": len(records),
        "frames_written": written,
        "fps": fps,
        "status_counts_on_overlay": summary_counts,
    }


# ── Offline inference ────────────────────────────────────────────────────────

def run_offline_inference(frames, model_path, device, inference_rate_hz=0.2):
    """
    Run InternNav model on each frame at the given inference rate.
    Returns list of (frame_idx, discrete_action, infer_time).
    """
    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
    from types import SimpleNamespace

    try:
        from arena_vln_models.internnav import (
            HABITAT_VLN_NUM_HISTORY,
            HABITAT_VLN_PLAN_STEP_GAP,
            HABITAT_VLN_RESIZE_H,
            HABITAT_VLN_RESIZE_W,
            _llm_output_trace_from_agent,
        )
    except Exception:
        HABITAT_VLN_RESIZE_W = 384
        HABITAT_VLN_RESIZE_H = 384
        HABITAT_VLN_NUM_HISTORY = 8
        HABITAT_VLN_PLAN_STEP_GAP = 4

        def _llm_output_trace_from_agent(agent):
            raw_output = str(getattr(agent, "llm_output", "") or "")
            digit_groups = [int(c) for c in re.findall(r"\d+", raw_output)]
            return {
                "raw_output_text": raw_output,
                "llm_output": raw_output,
                "llm_digits": digit_groups,
                "digit_groups": digit_groups,
                "model_generation_output_mode": "pixel_goal" if digit_groups else "symbolic_action",
                "symbolic_action_seq": list(getattr(agent, "last_symbolic_action_seq", []) or []),
                "generated_token_ids": list(getattr(agent, "last_generated_token_ids", []) or []),
            }

    print(f"Loading InternVLA-N1 model from {model_path} on {device}...")
    agent = InternVLAN1AsyncAgent(SimpleNamespace(
        device=device,
        model_path=model_path,
        resize_w=HABITAT_VLN_RESIZE_W,
        resize_h=HABITAT_VLN_RESIZE_H,
        num_history=HABITAT_VLN_NUM_HISTORY,
        plan_step_gap=HABITAT_VLN_PLAN_STEP_GAP,
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
        llm_trace = {}
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

            llm_trace = _llm_output_trace_from_agent(agent)

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
            "llm": llm_trace,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames, last action={action}, infer_time={infer_time:.3f}s")

    return offline_actions


def write_offline_overlay(frames, offline_actions, output_video_path: str, *, fps: float = 5.0):
    """Render overlay video from offline inference results and bag RGB frames."""
    import cv2

    if not frames:
        return {"overlay_video": None, "frames_written": 0, "reason": "no_frames"}
    action_by_frame = {int(item["frame_idx"]): item for item in offline_actions}
    first_rgb = next((frame.get("rgb") for frame in frames if frame.get("rgb") is not None), None)
    if first_rgb is None:
        return {"overlay_video": None, "frames_written": 0, "reason": "no_rgb"}
    height, width = first_rgb.shape[:2]
    output = Path(output_video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open offline overlay writer: {output}")
    written = 0
    for idx, frame_data in enumerate(frames):
        rgb = frame_data.get("rgb")
        if rgb is None:
            continue
        frame = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        action_info = action_by_frame.get(idx, {})
        action = action_info.get("discrete_action")
        trajectory = action_info.get("trajectory")
        linear = 0.0
        angular = 0.0
        if trajectory:
            try:
                subgoal = trajectory[-1]
                linear = float(math.hypot(float(subgoal[0]), float(subgoal[1])))
                angular = float(subgoal[2]) if len(subgoal) > 2 else 0.0
            except (TypeError, ValueError, IndexError):
                pass
        frame = _draw_action_glyph_cv(frame, action if isinstance(action, int) else None, linear, angular)
        pose = frame_data.get("pose") or (0.0, 0.0, 0.0)
        goal = frame_data.get("goal") or (0.0, 0.0, 0.0)
        lines = [
            f"offline frame={idx} action={ACTION_LABELS.get(action, action)} infer={action_info.get('infer_time_sec', 0.0):.2f}s",
            f"System2 input: rgb=yes depth={frame_data.get('depth') is not None} K={frame_data.get('intrinsic') is not None}",
            f"System2 input: pose=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}) instruction={str(frame_data.get('instruction', ''))[:48]}",
            f"System2 output: discrete_action={action}",
            f"System1 input: latent={'yes' if trajectory else 'n/a'} current_rgb/depth={'yes' if frame_data.get('depth') is not None else 'no'}",
            f"System1 output: trajectory={'yes' if trajectory else 'no'} goal=({goal[0]:.2f},{goal[1]:.2f})",
        ]
        frame = _draw_text_block(frame, lines)
        writer.write(frame)
        written += 1
    writer.release()
    return {"overlay_video": str(output), "frames_written": written, "fps": fps}


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
    parser.add_argument("--storage-id", default="auto", choices=["auto", "sqlite3", "mcap"], help="rosbag storage id")
    parser.add_argument("--ego-video", default="", help="Existing ego_observation.mp4 for trace-only overlay")
    parser.add_argument("--trace", default="", help="internnav_trace.jsonl for trace-only overlay")
    parser.add_argument("--overlay-video", default="", help="Path to write ego action overlay mp4")
    parser.add_argument("--skip-model", action="store_true", help="Only inspect bag and/or render trace overlay; do not run model")
    args = parser.parse_args()

    if args.ego_video and args.trace:
        overlay_path = args.overlay_video or str(Path(args.output).with_suffix(".overlay.mp4"))
        overlay_summary = render_trace_overlay_video(
            args.ego_video,
            args.trace,
            overlay_path,
            max_frames=args.max_frames,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "mode": "trace_video_overlay",
            "config": {
                "ego_video": args.ego_video,
                "trace": args.trace,
                "overlay_video": overlay_path,
                "max_frames": args.max_frames,
            },
            "overlay": overlay_summary,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"Reading rosbag from {args.rosbag}...")
    messages, topic_types, storage_id = read_bag(args.rosbag, storage_id=args.storage_id)

    print("Topic counts:")
    for topic, msgs in sorted(messages.items()):
        print(f"  {topic}: {len(msgs)} messages")

    print("Aligning frames...")
    frames, online_actions, model_outputs, cmd_vel_trace, topic_map = align_frames(messages)
    print(f"  {len(frames)} frames, {len(online_actions)} online actions, "
          f"{len(model_outputs)} model outputs, {len(cmd_vel_trace)} cmd_vel")
    print(f"  topic map: {topic_map}")

    if args.max_frames > 0:
        frames = frames[:args.max_frames]
        print(f"  Limited to {args.max_frames} frames")

    offline_actions = []
    if args.skip_model or not frames:
        print("\nSkipping offline model inference.")
    else:
        print(f"\nRunning offline inference on {len(frames)} frames...")
        offline_actions = run_offline_inference(
            frames, args.model_path, args.device, args.inference_rate_hz
        )

    print("\nComparing offline vs online actions...")
    result = compare_actions(offline_actions, online_actions, model_outputs) if offline_actions else {
        "total_offline_frames": 0,
        "action_comparison": {"match": 0, "mismatch": 0, "no_online": 0},
        "match_rate": 0.0,
        "offline_action_distribution": {},
        "online_action_distribution": {},
        "matches_sample": [],
        "all_matches": [],
    }
    overlay_summary = None
    if args.overlay_video and frames and offline_actions:
        overlay_summary = write_offline_overlay(frames, offline_actions, args.overlay_video)

    # Add summary info
    result["config"] = {
        "rosbag": args.rosbag,
        "model_path": args.model_path,
        "device": args.device,
        "inference_rate_hz": args.inference_rate_hz,
        "total_frames_in_bag": len(frames),
        "total_online_status_msgs": len(online_actions),
        "total_model_output_msgs": len(model_outputs),
        "storage_id": storage_id,
        "topic_map": topic_map,
        "skip_model": bool(args.skip_model),
    }
    if overlay_summary is not None:
        result["overlay"] = overlay_summary

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
