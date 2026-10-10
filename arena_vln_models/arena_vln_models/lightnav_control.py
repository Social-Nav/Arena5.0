"""ROS-independent episode and freshness rules for the native LightNav MPC."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SimCadence:
    """Consume at most one control slot; never replay missed simulation ticks."""

    period: float
    next_sim: float | None = None

    def reset(self):
        self.next_sim = None

    def due(self, sim: float) -> bool:
        if self.next_sim is not None and sim + 1e-9 < self.next_sim:
            return False
        self.next_sim = sim + self.period
        return True


@dataclass
class FrameSequence:
    """Consume each new camera stamp once, including frames dropped when busy."""

    last_stamp_ns: int | None = None

    def observe(self, stamp_ns: int) -> bool:
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            return False
        self.last_stamp_ns = stamp_ns
        return True

    def reset(self):
        self.last_stamp_ns = None


@dataclass
class ControlState:
    episode: int | None = None
    ready_episode: int | None = None
    generation: int = 0
    sequence: int = 0
    active: bool = False
    terminal: bool = False
    path_capture_wall: float = float('-inf')
    path_capture_sim: float = float('-inf')
    odom_wall: float = float('-inf')
    rgb_wall: float = float('-inf')
    clock_wall: float = float('-inf')
    command_wall: float = float('-inf')
    path_timeout: float = 2.0
    command_timeout: float = 0.5

    def invalidate(self):
        self.generation += 1
        self.path_capture_wall = self.path_capture_sim = float('-inf')
        self.command_wall = float('-inf')
        self.sequence = 0

    def reset(self, episode):
        # DDS may deliver the matching task_reset after episode-ready.
        if episode == self.episode or (self.episode is not None and episode < self.episode):
            return False
        self.invalidate()
        self.episode = episode
        self.ready_episode = None
        self.active = self.terminal = False
        return True

    def ready(self, episode):
        if self.episode is not None and episode < self.episode:
            return False
        changed = self.reset(episode)
        self.ready_episode = episode
        return changed

    def block_reason(self, wall, sim, *, require_command=False):
        if not self.active or self.terminal or self.ready_episode != self.episode:
            return 'episode_not_active'
        for name, stamp, limit in (
            ('odom_stale', self.odom_wall, self.command_timeout),
            ('clock_stalled', self.clock_wall, self.command_timeout),
            ('camera_stale', self.rgb_wall, self.path_timeout),
            ('path_expired', self.path_capture_wall, self.path_timeout),
        ):
            if wall - stamp > limit:
                return name
        age = sim - self.path_capture_sim
        if age < -0.01 or age > self.path_timeout:
            return 'path_sim_time_invalid'
        if require_command and wall - self.command_wall > self.command_timeout:
            return 'control_timeout'
        return ''


def valid_waypoints(points):
    return (isinstance(points, (list, tuple)) and 0 < len(points) <= 4096
            and all(isinstance(p, (list, tuple)) and len(p) == 3
                    and all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                            for v in p) for p in points))
