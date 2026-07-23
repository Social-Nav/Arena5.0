#!/usr/bin/env python3
"""Print the staged velocity command at EVERY layer of the nav2 pipeline, so you can
see which layer a motion problem (spin / stuck / sway) comes from.

It AUTO-DISCOVERS every geometry_msgs/Twist topic under the robot namespace and prints
each one's latest vx/wz per tick (so it works regardless of the exact topic names:
controller output, smoother output, collision_monitor output, ...). It also prints the
global-plan vs local-plan heading divergence, and the measured odom twist.

Typical pipeline order (left -> right in the printout, once discovered):
  cmd_vel_nav       controller_server raw output   (MPC / RotationShim decision)
  cmd_vel_smoothed  after velocity_smoother        (accel/vel/deadband limiting)
  cmd_vel           after collision_monitor        (what the robot actually receives)
  odom              measured / actually executed

How to read the three failure modes:
  * SPIN   : a layer shows large |wz| with vx~0. If cmd_vel_nav already spins -> controller
             (MPC/RotationShim). If only odom spins but cmd_vel~0 -> actuation/sim.
  * STUCK  : every layer ~0. Controller isn't commanding (check G/L/Δ: is the reference sane?).
  * SWAY   : wz flips sign repeatedly. See the FIRST layer that shows it (that layer is the source).
  * SLOW / open-path-but-crawls: vx tiny at cmd_vel_nav -> controller; or G-L Δ large -> bad reference.
  * WRONG-TURN AT STARTUP: if the bad wz is on cmd_vel but NOT on cmd_vel_nav, it's a
             behavior_server RECOVERY SPIN (it bypasses the controller) — watch the ">> [E]/[W]/[I]"
             rosout timeline for  costmap-timeout -> Aborting -> spin. If it's already on
             cmd_vel_nav, it's the controller (RotationShim/MPC): compare G (global heading) vs wz sign.

Run inside the arena container (env sourced):
  python3 /opt/arena_ws/src/Arena/_meta/tools/motion_layers.py [ROBOT_NS] [--always]

social_yielding mode (default): the tool watches `isaac/sim_running`. While the sim is
PAUSED (blocked -> snapshot -> LLM) it stays quiet; on each paused->running edge (the
orchestrator published the new yielding goal and unpaused) it prints a "YIELD RESUMED"
banner + the new goal, then streams the per-layer cmd_vel lines until the next pause.
Use --always to stream unconditionally regardless of sim_running (old behavior).
"""
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool

# args: [ROBOT_NS] [--always]
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
_flags = {a for a in sys.argv[1:] if a.startswith("-")}
NS = _args[0] if _args else "/task_generator_node/Ai2_Bot2"
# Default (social_yielding-aware): only stream while sim is RUNNING, and print a banner
# on each paused->running edge (i.e. right after the orchestrator sends the new yielding
# goal and unpauses). Pass --always to stream unconditionally (ignore sim_running gating).
ALWAYS = "--always" in _flags


def heading(path, ahead=0.5):
    """Direction (deg) from the path's first pose to the pose ~`ahead` m along it."""
    if path is None or len(path.poses) < 2:
        return None
    p0 = path.poses[0].pose.position
    for ps in path.poses[1:]:
        p = ps.pose.position
        if math.hypot(p.x - p0.x, p.y - p0.y) >= ahead:
            return math.degrees(math.atan2(p.y - p0.y, p.x - p0.x))
    p = path.poses[-1].pose.position
    if p.x == p0.x and p.y == p0.y:
        return None
    return math.degrees(math.atan2(p.y - p0.y, p.x - p0.x))


def angdiff(a, b):
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


# Preferred left->right ordering of known layers; unknown twist topics appended alphabetically.
ORDER = ["cmd_vel_nav", "cmd_vel_raw", "cmd_vel_smoothed", "cmd_vel_collision", "cmd_vel"]

# nav nodes whose /rosout events we surface as an inline timeline
# (catches the costmap-timeout -> follow_path abort -> recovery-spin chain that turns the robot at startup)
NAV_NODES = ("controller_server", "behavior_server", "bt_navigator", "planner_server", "smoother_server")
LEVELS = {10: "D", 20: "I", 30: "W", 40: "E", 50: "F"}


class Mon(Node):
    def __init__(self):
        super().__init__("motion_layers")
        self.twists = {}   # short topic name -> latest Twist
        self.subbed = set()
        self.g = None
        self.l = None
        self.od = None
        self.goal = None            # latest yielding/nav goal (PoseStamped)
        self.running = None         # sim_running state (None=unknown yet)
        self.create_subscription(Path, f"{NS}/received_global_plan", lambda m: setattr(self, "g", m), 10)
        self.create_subscription(Path, f"{NS}/local_plan", lambda m: setattr(self, "l", m), 10)
        self.create_subscription(Odometry, f"{NS}/odom", lambda m: setattr(self, "od", m), 10)
        self.create_subscription(Log, "/rosout", self.on_log, 100)
        self.create_subscription(PoseStamped, f"{NS}/goal_pose", lambda m: setattr(self, "goal", m), 10)
        # sim_running is published latched (TRANSIENT_LOCAL) by run_isaacsim; match that QoS.
        _state_qos = QoSProfile(depth=1)
        _state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Bool, "isaac/sim_running", self.on_sim_running, _state_qos)
        self.create_timer(1.0, self.discover)
        self.create_timer(0.5, self.tick)

    def on_sim_running(self, m):
        prev = self.running
        self.running = bool(m.data)
        if prev is None:
            print(f"  == sim_running = {self.running} ==", flush=True)
            return
        if self.running and not prev:
            # paused -> running edge: this is the yield resume (new goal published + unpause)
            g = self.goal
            gs = (f"({g.pose.position.x:+.2f}, {g.pose.position.y:+.2f})"
                  if g is not None else "unknown")
            print("\n" + "=" * 62, flush=True)
            print(f"  YIELD RESUMED  (sim unpaused) -> new goal = {gs}", flush=True)
            print("  streaming per-layer cmd_vel until next pause ...", flush=True)
            print("=" * 62, flush=True)
        elif prev and not self.running:
            print("\n  -- sim PAUSED (blocked -> snapshot -> LLM); streaming halted --", flush=True)

    def on_log(self, m):
        # Surface nav-node events inline so a wrong-turn's CAUSE shows in the same stream:
        #   behavior_server INFO => a RECOVERY (e.g. spin) is running; WARN/ERROR => abort / costmap timeout.
        if not any(n in m.name for n in NAV_NODES):
            return
        if m.level < 20:                                                   # drop DEBUG
            return
        if m.level < 30 and not ("behavior" in m.name or "bt_navigator" in m.name):
            return                                                         # non-recovery nodes: keep WARN+ only
        node = m.name.split(".")[-1]
        print(f"  >> [{LEVELS.get(m.level, '?')}][{node}] {m.msg}", flush=True)

    def discover(self):
        for name, types in self.get_topic_names_and_types():
            if (name.startswith(NS + "/") and "geometry_msgs/msg/Twist" in types
                    and name not in self.subbed):
                self.subbed.add(name)
                short = name[len(NS) + 1:]
                self.create_subscription(Twist, name, lambda m, s=short: self.twists.__setitem__(s, m), 10)
                print(f"  + velocity layer discovered: {short}", flush=True)

    def _ordered(self):
        keys = list(self.twists)
        known = [k for k in ORDER if k in keys]
        rest = sorted(k for k in keys if k not in ORDER)
        return known + rest

    def tick(self):
        # In social_yielding mode, stay quiet while paused (or before sim_running is known);
        # --always streams unconditionally.
        if not ALWAYS and not self.running:
            return

        gh = heading(self.g)
        lh = heading(self.l)
        dd = angdiff(gh, lh)

        def a(x):
            return f"{x:+6.1f}" if x is not None else "   n/a"

        parts = []
        for s in self._ordered():
            t = self.twists[s]
            parts.append(f"{s}[vx={t.linear.x:+.2f} wz={t.angular.z:+.2f}]")
        od = self.od.twist.twist if self.od else None
        odom = f"odom[vx={od.linear.x:+.2f} wz={od.angular.z:+.2f}]" if od else "odom[--]"
        line = f"G={a(gh)} L={a(lh)} Δ={a(dd)} | " + "  ".join(parts) + " | " + odom
        print(line, flush=True)


def main():
    rclpy.init()
    print(f"motion_layers on {NS}  (Ctrl-C to stop)")
    mode = ("ALWAYS (stream regardless of sim_running)" if ALWAYS
            else "social_yielding (quiet while paused; auto-print on each unpause/yield-resume)")
    print(f"mode: {mode}")
    print("G/L/Δ = global/local plan heading & divergence (deg) | then each Twist layer | odom = measured")
    node = Mon()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
