"""Task34 three-segment speed trace: raw controller -> delivered -> odom.

The existing recorder stores only the delivered ``cmd_vel`` and ``odom``.  This
module adds the missing raw ``cmd_vel_nav`` segment and a per-control-tick
dictionary with human speed and turn-condition fields, so a later analysis can
attribute a speed change to the controller, the velocity smoother, or the
plant without re-deriving it from separate CSVs.

The pure ``trace_record`` helper is CPU-testable.  The ROS node is imported
lazily so the module stays importable outside a sourced Jazzy environment.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

TRACE_SCHEMA = 'arena.task34.cmd_vel_trace.v1'
TRACE_FILE = 'cmd_vel_trace.jsonl'
TURN_THRESHOLD_RAD = 0.5
SAFE_SLOWDOWN_LINEAR_MPS = 0.2


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _person_speeds(agents: Iterable[Mapping[str, Any]] | None) -> tuple[int, float, float]:
    """Return (count, max_speed, min_distance) over PERSON agents only."""
    speeds: list[float] = []
    distances: list[float] = []
    for agent in agents or ():
        try:
            if int(agent.get('type', 1)) != 1:  # hunav_msgs Agent.PERSON
                continue
            vel = agent.get('velocity') or {}
            linear = vel.get('linear') or {}
            speeds.append(math.hypot(_finite(linear.get('x')), _finite(linear.get('y'))))
            pos = (agent.get('position') or {}).get('position') or {}
            distances.append(math.hypot(_finite(pos.get('x')), _finite(pos.get('y'))))
        except (TypeError, ValueError, AttributeError):
            continue
    return len(speeds), (max(speeds) if speeds else 0.0), (min(distances) if distances else 0.0)


def trace_record(*, wall_time: float, monotonic_ns: int, episode: int,
                 raw_linear_x: float, raw_angular_z: float,
                 delivered_linear_x: float, delivered_angular_z: float,
                 odom_linear_x: float, odom_angular_z: float,
                 odom_pose_xy: tuple[float, float] | None,
                 goal_xy: tuple[float, float] | None,
                 agents: Iterable[Mapping[str, Any]] | None,
                 sim_time_sec: float | None = None) -> dict[str, Any]:
    """Build one trace record with the three speed segments and condition flags."""
    human_count, human_max_speed, human_min_distance = _person_speeds(agents)
    goal_distance = None
    goal_bearing = None
    turn_condition = None
    if odom_pose_xy is not None and goal_xy is not None:
        dx = float(goal_xy[0]) - float(odom_pose_xy[0])
        dy = float(goal_xy[1]) - float(odom_pose_xy[1])
        goal_distance = math.hypot(dx, dy)
        goal_bearing = math.atan2(dy, dx)
        turn_condition = abs(goal_bearing) > TURN_THRESHOLD_RAD
    delivered = _finite(delivered_linear_x)
    return {
        'schema': TRACE_SCHEMA,
        'wall_time': float(wall_time),
        'monotonic_ns': int(monotonic_ns),
        'sim_time_sec': None if sim_time_sec is None else round(float(sim_time_sec), 6),
        'episode': int(episode),
        'raw': {'linear_x': round(_finite(raw_linear_x), 6),
                'angular_z': round(_finite(raw_angular_z), 6)},
        'delivered': {'linear_x': round(delivered, 6),
                      'angular_z': round(_finite(delivered_angular_z), 6)},
        'odom': {'linear_x': round(_finite(odom_linear_x), 6),
                 'angular_z': round(_finite(odom_angular_z), 6)},
        'human': {'count': human_count,
                  'max_speed_mps': round(human_max_speed, 6),
                  'min_distance_m': round(human_min_distance, 6)},
        'goal': {'distance_m': None if goal_distance is None else round(goal_distance, 6),
                 'bearing_rad': None if goal_bearing is None else round(goal_bearing, 6),
                 'turn_condition': turn_condition},
        'condition': {
            'safe_slowdown': delivered <= SAFE_SLOWDOWN_LINEAR_MPS,
            'stop': abs(delivered) <= 1e-6,
        },
        'smoother_delta_linear_x': round(delivered - _finite(raw_linear_x), 6),
        'plant_tracking_delta_linear_x': round(_finite(odom_linear_x) - delivered, 6),
    }


def write_trace(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')


def run_node(argv: list[str] | None = None) -> int:  # pragma: no cover - ROS runtime
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from hunav_msgs.msg import Agents
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Int16

    class CmdVelTrace(Node):
        def __init__(self) -> None:
            super().__init__('task34_cmd_vel_trace')
            self.declare_parameter('record_data_dir', '')
            self.declare_parameter('namespace', '')
            record_dir = str(self.get_parameter('record_data_dir').value or '').strip()
            namespace = str(self.get_parameter('namespace').value or '').rstrip('/')
            self.output = Path(record_dir) / TRACE_FILE if record_dir else None
            self.episode = 0
            self.raw = (0.0, 0.0)
            self.odom = (0.0, 0.0)
            self.pose = None
            self.goal = None
            self.agents = []
            self.create_subscription(Twist, f'{namespace}/cmd_vel_nav', self._raw, 10)
            self.create_subscription(Twist, f'{namespace}/cmd_vel', self._delivered, 10)
            self.create_subscription(Odometry, f'{namespace}/odom', self._odom, qos_profile_sensor_data)
            self.create_subscription(Agents, '/task_generator_node/human_states', self._agents, 10)
            self.create_subscription(PoseStamped, f'{namespace}/episode_goal_pose_metadata', self._goal, 10)
            self.create_subscription(Int16, '/task_generator_node/task_reset', self._reset, 10)

        def _raw(self, msg):
            self.raw = (msg.linear.x, msg.angular.z)

        def _odom(self, msg):
            self.odom = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
            p = msg.pose.pose.position
            self.pose = (p.x, p.y)

        def _agents(self, msg):
            self.agents = [
                {'type': a.type, 'position': {'position': {'x': a.position.position.x,
                                                           'y': a.position.position.y}},
                 'velocity': {'linear': {'x': a.velocity.linear.x, 'y': a.velocity.linear.y}}}
                for a in msg.agents
            ]

        def _goal(self, msg):
            self.goal = (msg.pose.position.x, msg.pose.position.y)

        def _reset(self, msg):
            try:
                self.episode = int(msg.data)
            except (TypeError, ValueError):
                pass

        def _delivered(self, msg):
            if self.output is None:
                return
            record = trace_record(
                wall_time=time.time(), monotonic_ns=time.monotonic_ns(), episode=self.episode,
                raw_linear_x=self.raw[0], raw_angular_z=self.raw[1],
                delivered_linear_x=msg.linear.x, delivered_angular_z=msg.angular.z,
                odom_linear_x=self.odom[0], odom_angular_z=self.odom[1],
                odom_pose_xy=self.pose, goal_xy=self.goal, agents=self.agents,
            )
            write_trace(self.output, record)

    rclpy.init(args=argv)
    node = CmdVelTrace()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(run_node())
