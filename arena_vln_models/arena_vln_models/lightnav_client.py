"""Arena ROS lifecycle around the unmodified LightNav client and native MPC.

No robot hardware launch, Nav2 controller, or InternNav trajectory conversion
is used here. The upstream geometric reference and CasADi solver are imported
from the versioned runtime staging directory.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from arena_vln_models.lightnav_control import ControlState, FrameSequence, SimCadence, valid_waypoints


def load_native():
    root = Path(os.environ.get('ARENA_WORKSPACE', '/opt/arena_ws')) / 'deps/lightnav-client'
    record = json.loads((root / 'runtime.json').read_text())
    stage = Path(record['stage'])
    provenance = json.loads((stage / 'source.json').read_text())
    for name, digest in provenance['files'].items():
        if hashlib.sha256((stage / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'LightNav client source hash mismatch: {name}')
    for package in ('vln_client', 'vln_mpc'):
        sys.path.insert(0, str(stage / package))
    from vln_client.vln_client import VlnClient
    from vln_mpc import geometry, mpc
    return VlnClient, geometry, mpc, provenance


def controller_for(robot, mpc):
    return mpc.MPCController(horizon=5, dt_s=.1, w_max=float(robot['max_vel_theta']),
        a_max_v=min(abs(float(robot['velocity_smoother_max_accel'][0])),
                    abs(float(robot['velocity_smoother_max_decel'][0]))),
        a_max_w=min(abs(float(robot['velocity_smoother_max_accel'][2])),
                    abs(float(robot['velocity_smoother_max_decel'][2]))),
        q_weights=(10., 10., 1.), r_weights=(.1, .1))


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--robot', default='Ai2_Bot2')
    parser.add_argument('--run-token', default='manual')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--probe-control', action='store_true')
    parser.add_argument('--ready-probe', action='store_true')
    args = parser.parse_args(argv)
    profile = yaml.safe_load(Path(args.profile).read_text())
    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from ament_index_python.packages import get_package_share_directory
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock as ClockMessage
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Int16, Empty
    robot = yaml.safe_load((Path(get_package_share_directory('arena_robots')) /
                           'robots' / args.robot / 'model_params.yaml').read_text())
    namespace = f'/task_generator_node/{args.robot}'
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    if args.probe_control or args.ready_probe:
        probe = Node('lightnav_control_probe')
        observed = []
        if args.ready_probe:
            probe.create_subscription(String, namespace + '/lightnav/status',
                                      lambda msg: observed.append(json.loads(msg.data)), latched)
        deadline = time.monotonic() + (15 if args.ready_probe else 3)
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=.1)
            if args.ready_probe and any(x.get('backend_ready') and x.get('backend') == 'lightnav'
                    and (args.run_token == 'manual' or x.get('run_token') == args.run_token)
                    and 0 <= time.time() - x.get('wall_time', 0) < 2.5 for x in observed):
                probe.destroy_node(); rclpy.shutdown()
                print('LightNav native client ready')
                return 0
        publishers = probe.get_publishers_info_by_topic(namespace + '/cmd_vel')
        probe.destroy_node(); rclpy.shutdown()
        if args.ready_probe:
            raise RuntimeError('LightNav client status was not observed')
        if publishers:
            raise RuntimeError('cmd_vel already owned by: ' + ', '.join(p.node_namespace + '/' + p.node_name for p in publishers))
        print('Robot command channel is available')
        return 0
    VlnClient, geometry, mpc, provenance = load_native()
    controller = controller_for(robot, mpc)
    if args.smoke:
        reference = np.array([[.05 * (i + 1), 0., 0.] for i in range(5)])
        command, _ = controller.solve(np.zeros(3), reference, (0., 0.), float(robot['max_vel_x']))
        if not all(math.isfinite(v) for v in command) or command[0] <= 0:
            raise RuntimeError(f'Native MPC smoke failed: {command}')
        print(json.dumps({'native_mpc_smoke': 'pass', 'command': list(command), 'source': provenance['revision'],
                          'python': sys.version, 'numpy': np.__version__}))
        rclpy.shutdown()
        return 0

    class LightNavClient(Node):
        def __init__(self):
            super().__init__('lightnav_client', namespace=namespace)
            self.native = VlnClient(logger=self.get_logger(), server_url=profile['server']['endpoint'])
            self.state = ControlState(path_timeout=profile['client']['path_timeout_sec'],
                                      command_timeout=profile['client']['command_timeout_sec'])
            self.history = deque(maxlen=1000)
            self.frames = {}
            self.image_sequence = FrameSequence()
            self.pose = None
            self.path = None
            self.sim = 0.
            self.instruction = ''
            self.native_episode = None
            self.previous = (0., 0.)
            self.command_request = None
            self.future = None
            self.future_generation = None
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lightnav-mpc')
            self.solve_cadence = SimCadence(.1)
            self.publish_cadence = SimCadence(.05)
            self.last_status = self.last_heartbeat = 0.
            self.last_zero_wall = float('-inf')
            self.termination_reason = ""
            self.last_clock_ns = None
            self.ready_announced = False
            self.base_frame = args.robot + '/' + robot.get('robot_base_frame', 'base_link')
            self.odom_frame = args.robot + '/' + robot.get('robot_odom_frame', 'odom')
            self.cmd = self.create_publisher(Twist, 'cmd_vel', 10)
            status_qos = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE,
                                    durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.status = self.create_publisher(String, 'lightnav/status', status_qos)
            self.output = self.create_publisher(String, 'lightnav/model_output', 10)
            self.debug_image = self.create_publisher(Image, 'lightnav/debug_image', qos_profile_sensor_data)
            self.create_subscription(Image, profile['client']['rgb_topic'], self.on_image, qos_profile_sensor_data)
            self.create_subscription(Odometry, profile['client']['odom_topic'], self.on_odom, qos_profile_sensor_data)
            self.create_subscription(ClockMessage, '/clock', self.on_clock, qos_profile_sensor_data)
            self.create_subscription(Int16, '/task_generator_node/task_reset', self.on_reset, 10)
            self.create_subscription(String, '/task_generator_node/eval_ready', self.on_ready, latched)
            self.create_subscription(String, '/task_generator_node/vln_instruction', self.on_instruction, latched)
            self.create_subscription(String, '/task_generator_node/episode_outcome', self.on_outcome, latched)
            self.create_subscription(Empty, '/task_generator_node/finished', self.on_finished, 10)
            self.create_timer(.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.get_logger().info('LightNav native client/MPC ready; waiting for Arena episode and sensors')

        def emit(self, status, **fields):
            now = time.monotonic()
            payload = dict(status=status, backend='lightnav', backend_ready=True,
                execution_mode='external_direct_cmd_vel', controller='lightnav_mpc',
                episode=self.state.episode, episode_started=self.state.active,
                native_episode=self.native_episode, generation=self.state.generation,
                request_id=self.state.sequence, run_token=args.run_token, sim_time=self.sim,
                wall_time=time.time(), ready=True, control_timebase='simulation',
                solver_dt_sec=.1, solver_rate_hz=10., command_rate_hz=20.,
                debug={'topics': {'rgb': profile['client']['rgb_topic'], 'odom': profile['client']['odom_topic']},
                       'sensor_ages_sec': {'rgb': min(1e9, now - self.state.rgb_wall),
                                           'odom': min(1e9, now - self.state.odom_wall)},
                       'stale_after_sec': self.state.path_timeout, 'tf_ready': self.pose is not None})
            payload.update(fields)
            self.status.publish(String(data=json.dumps(payload, allow_nan=False)))

        def zero(self, reason):
            now = time.monotonic()
            was_moving = self.previous != (0., 0.) or self.command_request is not None
            self.previous = (0., 0.)
            self.command_request = None
            self.state.command_wall = float('-inf')
            if was_moving or now - self.last_zero_wall >= .05:
                self.cmd.publish(Twist())
                self.last_zero_wall = now
            if time.monotonic() - self.last_status > .2:
                self.emit('hold', reason=reason, linear_x=0., angular_z=0.)
                self.last_status = time.monotonic()

        def clear(self, reason, *, terminal=False):
            self.native.stop()
            self.native_episode = None
            self.image_sequence.reset()
            self.frames.clear()
            self.state.invalidate()
            self.state.active = False
            self.state.terminal = terminal
            self.termination_reason = reason if terminal else ""
            self.path = None
            self.ready_announced = False
            self.solve_cadence.reset(); self.publish_cadence.reset()
            self.zero(reason)

        def on_reset(self, msg):
            if self.state.reset(msg.data):
                self.clear('task_reset')
                self.history.clear(); self.frames.clear()
                self.pose = None
                self.instruction = ''
                self.emit('resetting', ready=False, episode_started=False)

        def on_ready(self, msg):
            try:
                value = json.loads(msg.data)
            except ValueError:
                return
            if value.get('stage') != 'episode':
                return
            episode = value.get('episode')
            if type(episode) is not int or (self.state.episode is not None and episode < self.state.episode):
                return
            if value.get('ready'):
                if self.state.ready(episode):
                    self.clear('new_ready_episode')
                    self.instruction = ''
                self.state.ready_episode = episode
                self.maybe_start()
            else:
                finished = (value.get('details') or {}).get('reason') == 'episode_finished'
                self.clear('episode_not_ready', terminal=finished or self.state.terminal)
                self.state.ready_episode = None

        def on_instruction(self, msg):
            instruction = msg.data.strip()
            if instruction and instruction != self.instruction:
                if self.state.active:
                    self.clear('instruction_changed')
                self.instruction = instruction
            self.maybe_start()

        def maybe_start(self):
            if (not self.state.active and not self.state.terminal and self.state.episode is not None
                    and self.state.ready_episode == self.state.episode and self.instruction):
                self.native_episode = self.native.start(self.instruction)
                self.state.active = True
                self.emit('episode_ready', episode_started=True, instruction=self.instruction)

        def on_outcome(self, msg):
            try:
                value = json.loads(msg.data)
            except ValueError:
                return
            if value.get('episode_index') == self.state.episode:
                self.clear('episode_outcome', terminal=True)

        def on_finished(self, msg):
            self.clear('task_finished', terminal=True)

        def on_clock(self, msg):
            ns = stamp_ns(msg.clock)
            if self.last_clock_ns is not None and ns < self.last_clock_ns:
                self.clear('clock_rewind')
                self.state.episode = None
                self.history.clear(); self.frames.clear()
                self.state.ready_episode = None
                self.instruction = ''
            if ns != self.last_clock_ns:
                self.state.clock_wall = time.monotonic()
            advanced = self.last_clock_ns is None or ns != self.last_clock_ns
            self.last_clock_ns = ns
            self.sim = ns / 1e9
            if advanced:
                self.tick()

        def on_image(self, msg):
            if msg.encoding not in ('rgb8', 'rgba8', 'bgr8', 'bgra8'):
                return
            ns = stamp_ns(msg.header.stamp)
            if not self.image_sequence.observe(ns):
                return
            channels = 4 if 'a' in msg.encoding else 3
            rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
            rgb = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)[:, :, :3]
            if msg.encoding.startswith('bgr'):
                rgb = rgb[:, :, ::-1]
            now = time.monotonic()
            self.state.rgb_wall = now
            self.frames.setdefault(ns, now)
            while len(self.frames) > 256:
                self.frames.pop(next(iter(self.frames)))
            rgb_bytes = np.ascontiguousarray(rgb).tobytes()
            # Match the native ROS client: offer only at camera arrival. If the
            # client is busy, drop this frame and wait for the next camera frame.
            if self.state.active and self.native.offer_frame(
                    stamp_sec=msg.header.stamp.sec, stamp_nanosec=msg.header.stamp.nanosec,
                    width=msg.width, height=msg.height, step=msg.width*3, rgb=rgb_bytes):
                self.emit('lightnav_frame_submitted', capture_stamp_ns=ns)
            # Actual input image, with no synthesized camera or model output.
            debug = Image()
            debug.header = msg.header
            debug.height, debug.width = msg.height, msg.width
            debug.encoding, debug.step, debug.data = 'rgb8', msg.width * 3, rgb_bytes
            self.debug_image.publish(debug)

        def on_odom(self, msg):
            if msg.header.frame_id != self.odom_frame or msg.child_frame_id != self.base_frame:
                return
            q = msg.pose.pose.orientation
            yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
            pose = geometry.TimedPose(stamp_ns(msg.header.stamp), msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
            if self.history and pose.stamp_ns <= self.history[-1].stamp_ns:
                return
            self.pose = pose
            self.history.append(pose)
            self.state.odom_wall = time.monotonic()

        def accept(self, result):
            if not self.state.active or result.episode != self.native_episode or result.sequence <= self.state.sequence:
                return
            if type(result.stop) is not bool:
                self.clear('invalid_stop')
                return
            ns = result.frame.stamp_sec * 1_000_000_000 + result.frame.stamp_nanosec
            captured = self.frames.get(ns, float('-inf'))
            if time.monotonic() - captured > self.state.path_timeout or self.sim - ns/1e9 > self.state.path_timeout:
                self.emit('path_rejected', reason='expired_observation', response_sequence=result.sequence)
                return
            self.state.sequence = result.sequence
            response = dict(waypoints=result.waypoints, stop=result.stop, latency_ms=result.latency_ms,
                            capture_stamp_ns=ns, instruction=self.instruction)
            self.emit('lightnav_inference', response=response)
            self.output.publish(String(data=json.dumps(response)))
            if result.stop:
                self.clear('model_stop', terminal=True)
                self.emit('stop', terminal=True, termination_reason='model_stop', response=response)
                return
            match = geometry.pose_at_stamp(self.history, ns, .3)
            if match is None or not valid_waypoints(result.waypoints):
                self.path = None
                self.zero('invalid_path_or_capture_pose')
                self.emit('path_rejected', reason='invalid_path_or_capture_pose')
                return
            self.path = np.asarray(geometry.project_body_to_odom(result.waypoints, match[0]), dtype=np.float64)
            self.state.path_capture_wall = captured
            self.state.path_capture_sim = ns / 1e9
            self.emit('path_accepted', capture_stamp_ns=ns, odom_match_gap_sec=match[1],
                      point_count=len(self.path), frame=self.odom_frame)

        def solve(self, generation, request, pose, path, previous):
            reference = mpc.build_pose_aligned_reference(path, np.array([pose.x, pose.y, pose.yaw]),
                                                         horizon=5, weights=(10., 10., 1.))
            local = np.asarray(geometry.project_odom_to_local(reference, pose))
            command, _ = controller.solve(np.zeros(3), local, previous, float(robot['max_vel_x']))
            return generation, request, command

        def tick(self):
            now = time.monotonic()
            if self.count_publishers('cmd_vel') > 1:
                self.clear('competing_cmd_vel_publisher')
                raise RuntimeError('another cmd_vel publisher appeared')
            for result in self.native.take_results():
                self.accept(result)
            snap = self.native.snapshot()
            if self.state.active and snap.state == 'ERROR':
                self.path = None
                self.state.invalidate()
                self.zero('client_error')
            reason = self.state.block_reason(now, self.sim)
            if reason in {'odom_stale', 'clock_stalled', 'camera_stale', 'path_expired', 'path_sim_time_invalid'} and self.path is not None:
                self.path = None
                self.state.invalidate()
            if self.future is not None and self.future.done():
                future, self.future = self.future, None
                future_generation = self.future_generation
                try:
                    generation, request, command = future.result()
                    if (not reason and generation == self.state.generation and request == self.state.sequence
                            and all(math.isfinite(float(v)) for v in command)):
                        self.previous = (max(0., min(float(robot['max_vel_x']), float(command[0]))),
                            max(-float(robot['max_vel_theta']), min(float(robot['max_vel_theta']), float(command[1]))))
                        self.state.command_wall = now
                        self.command_request = request
                except Exception as exc:
                    if future_generation == self.state.generation:
                        self.path = None
                        self.state.invalidate()
                        self.zero('solver_error')
                        self.emit('error', error=str(exc))
                        reason = 'solver_error'
            if not reason and self.path is not None and self.pose is not None and self.future is None and self.solve_cadence.due(self.sim):
                self.future_generation = self.state.generation
                self.future = self.pool.submit(self.solve, self.state.generation, self.state.sequence,
                                              self.pose, self.path.copy(), self.previous)
            reason = self.state.block_reason(now, self.sim, require_command=True)
            if reason:
                self.zero(reason)
            elif self.publish_cadence.due(self.sim):
                command = Twist()
                command.linear.x, command.angular.z = self.previous
                self.cmd.publish(command)
                self.emit('lightnav_command', request_id=self.command_request,
                          linear_x=command.linear.x, angular_z=command.angular.z)
            if now - self.last_heartbeat >= .5:
                if self.state.terminal:
                    self.emit('stop', terminal=True, termination_reason=self.termination_reason)
                elif not self.state.active:
                    self.emit('backend_ready')
                else:
                    self.emit('episode_ready', episode_started=True)
                self.last_heartbeat = now

        def close(self):
            self.clear('shutdown')
            self.native.close()
            self.pool.shutdown(wait=False, cancel_futures=True)

    node = LightNavClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
