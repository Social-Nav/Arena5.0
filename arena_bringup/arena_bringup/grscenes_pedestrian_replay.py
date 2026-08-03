"""Replay GRScenes pedestrian GT tracks on ROS topics for repeatable tests."""

from __future__ import annotations

import bisect
import math
from pathlib import Path

from .grscenes_pedestrian_gt import PedestrianFrame, PedestrianTrack, load_tracks_from_parquet


def _interpolate_angle(a: float, b: float, ratio: float) -> float:
    delta = math.atan2(math.sin(b - a), math.cos(b - a))
    return a + delta * ratio


def sample_tracks(tracks: list[PedestrianTrack], time_sec: float) -> list[tuple[PedestrianTrack, PedestrianFrame]]:
    samples = []
    for track in tracks:
        frames = track.frames
        if not frames:
            continue
        if time_sec <= frames[0].time_sec:
            samples.append((track, frames[0]))
            continue
        if time_sec >= frames[-1].time_sec:
            samples.append((track, frames[-1]))
            continue
        times = [frame.time_sec for frame in frames]
        right = bisect.bisect_right(times, time_sec)
        left_frame = frames[right - 1]
        right_frame = frames[right]
        span = max(right_frame.time_sec - left_frame.time_sec, 1e-9)
        ratio = (time_sec - left_frame.time_sec) / span
        samples.append(
            (
                track,
                PedestrianFrame(
                    frame_index=left_frame.frame_index,
                    time_sec=time_sec,
                    x=left_frame.x + (right_frame.x - left_frame.x) * ratio,
                    y=left_frame.y + (right_frame.y - left_frame.y) * ratio,
                    yaw_rad=_interpolate_angle(left_frame.yaw_rad, right_frame.yaw_rad, ratio),
                ),
            )
        )
    return samples


def quaternion_from_yaw(yaw: float):
    try:
        from geometry_msgs.msg import Quaternion
    except Exception:  # pragma: no cover - ROS import guard
        raise RuntimeError("geometry_msgs is required for ROS replay")
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class GrscenesPedestrianReplayNode:
    def __init__(self):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from std_msgs.msg import Header
            from hunav_msgs.msg import Agent, Agents
            from arena_people_msgs.msg import Pedestrian, Pedestrians
            from geometry_msgs.msg import Pose, Twist
        except Exception as exc:  # pragma: no cover - ROS import guard
            raise RuntimeError("ROS 2 Python message packages are required for GT replay") from exc

        class _Node(Node):
            pass

        self.rclpy = rclpy
        self.Header = Header
        self.Agent = Agent
        self.Agents = Agents
        self.Pedestrian = Pedestrian
        self.Pedestrians = Pedestrians
        self.Pose = Pose
        self.Twist = Twist

        self.node = _Node("grscenes_pedestrian_replay")
        self.node.declare_parameter("params_path", "")
        self.node.declare_parameter("frame_dt_sec", 0.1)
        self.node.declare_parameter("rate_hz", 10.0)
        self.node.declare_parameter("loop", False)
        self.node.declare_parameter("frame_id", "map")
        self.node.declare_parameter("human_states_topic", "/task_generator_node/human_states")
        self.node.declare_parameter("arena_peds_topic", "/task_generator_node/arena_peds")
        self.node.declare_parameter("publish_arena_peds", True)
        self.node.declare_parameter("name_prefix", "hunav")
        self.node.declare_parameter("z", 1.25)

        params_path = str(self.node.get_parameter("params_path").value or "").strip()
        if not params_path:
            raise ValueError("params_path parameter is required")
        frame_dt_sec = float(self.node.get_parameter("frame_dt_sec").value)
        name_prefix = str(self.node.get_parameter("name_prefix").value or "hunav")
        self.tracks = load_tracks_from_parquet(params_path, frame_dt_sec=frame_dt_sec, name_prefix=name_prefix)
        if not self.tracks:
            raise ValueError(f"No pedestrian GT tracks found in {params_path}")

        self.rate_hz = float(self.node.get_parameter("rate_hz").value)
        self.loop = bool(self.node.get_parameter("loop").value)
        self.frame_id = str(self.node.get_parameter("frame_id").value or "map")
        self.z = float(self.node.get_parameter("z").value)
        self.started_ns = self.node.get_clock().now().nanoseconds
        self.duration_sec = max(track.frames[-1].time_sec for track in self.tracks if track.frames)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.human_pub = self.node.create_publisher(
            Agents,
            str(self.node.get_parameter("human_states_topic").value),
            qos,
        )
        self.arena_pub = None
        if bool(self.node.get_parameter("publish_arena_peds").value):
            self.arena_pub = self.node.create_publisher(
                Pedestrians,
                str(self.node.get_parameter("arena_peds_topic").value),
                qos,
            )
        self.timer = self.node.create_timer(1.0 / max(self.rate_hz, 1e-6), self._tick)
        self.node.get_logger().info(
            f"Loaded {len(self.tracks)} GT pedestrian track(s) from {Path(params_path)} duration={self.duration_sec:.2f}s"
        )

    def _elapsed_sec(self) -> float:
        elapsed = (self.node.get_clock().now().nanoseconds - self.started_ns) / 1e9
        if self.loop and self.duration_sec > 0.0:
            return elapsed % self.duration_sec
        return min(elapsed, self.duration_sec)

    def _header(self):
        header = self.Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        return header

    def _tick(self) -> None:
        samples = sample_tracks(self.tracks, self._elapsed_sec())
        agents_msg = self.Agents()
        agents_msg.header = self._header()
        peds_msg = self.Pedestrians()
        peds_msg.header = agents_msg.header

        for index, (track, frame) in enumerate(samples, start=1):
            pose = self.Pose()
            pose.position.x = float(frame.x)
            pose.position.y = float(frame.y)
            pose.position.z = self.z
            pose.orientation = quaternion_from_yaw(frame.yaw_rad)

            agent = self.Agent()
            agent.id = index
            agent.type = self.Agent.PERSON
            agent.name = track.name
            agent.position = pose
            agent.yaw = float(frame.yaw_rad)
            agent.radius = 0.3
            agent.desired_velocity = 0.0
            agents_msg.agents.append(agent)

            ped = self.Pedestrian()
            ped.id = index
            ped.name = track.name
            ped.pose = pose
            ped.twist = self.Twist()
            ped.animation_state = self.Pedestrian.WALKING
            peds_msg.pedestrians.append(ped)

        self.human_pub.publish(agents_msg)
        if self.arena_pub is not None:
            self.arena_pub.publish(peds_msg)


def main(args=None) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    replay = None
    try:
        replay = GrscenesPedestrianReplayNode()
        try:
            rclpy.spin(replay.node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
    finally:
        if replay is not None:
            try:
                replay.node.destroy_node()
            except (KeyboardInterrupt, ExternalShutdownException):
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except (KeyboardInterrupt, ExternalShutdownException):
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
