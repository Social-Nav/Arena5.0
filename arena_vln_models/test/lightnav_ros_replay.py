"""Real ROS/native-MPC replay with a deterministic WebSocket peer (no GPU).

Run with the LightNav client venv and a separate ROS_DOMAIN_ID. This exercises
the production adapter process; it is not a claim about model navigation.
"""
import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from aiohttp import web
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int16, Empty
import yaml


def main():
    if os.environ.get('ROS_DOMAIN_ID') in (None, '', '0', '1'):
        raise RuntimeError('Replay must run in a dedicated ROS domain, such as 72')
    peer = {'mode': 'forward', 'delay': 0., 'connections': 0, 'requests': 0}
    loop = asyncio.new_event_loop()

    async def ws(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        peer['connections'] += 1
        async for message in socket:
            if message.type != web.WSMsgType.TEXT:
                continue
            payload = json.loads(message.data)
            action = payload['action']
            data = {'rc': 0}
            if action == 'next':
                peer['requests'] += 1
                mode, delay = peer['mode'], peer['delay']
                await asyncio.sleep(delay)
                if mode == 'disconnect':
                    await socket.close()
                    continue
                points = ([[0., 0., .5]] if mode == 'rotate' else
                          [[.08 * (i + 1), 0., 0.] for i in range(10)])
                data.update(seq=payload['data']['seq'], stop=mode == 'stop',
                            actions={'actions': points}, pointing={})
            if socket.closed:
                break
            try:
                await socket.send_json({'action': action, 'data': data})
            except ConnectionResetError:
                break
        return socket

    async def start():
        app = web.Application(); app.router.add_get('/', ws)
        runner = web.AppRunner(app); await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', 18050).start()
    loop.run_until_complete(start())
    thread = threading.Thread(target=loop.run_forever, daemon=True); thread.start()
    rclpy.init()
    node = Node('lightnav_replay')
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    namespace = '/task_generator_node/Ai2_Bot2'
    image_pub = node.create_publisher(Image, namespace + '/head_camera/image', qos_profile_sensor_data)
    odom_pub = node.create_publisher(Odometry, namespace + '/odom', qos_profile_sensor_data)
    clock_pub = node.create_publisher(Clock, '/clock', qos_profile_sensor_data)
    ready_pub = node.create_publisher(String, '/task_generator_node/eval_ready', latched)
    instruction_pub = node.create_publisher(String, '/task_generator_node/vln_instruction', latched)
    reset_pub = node.create_publisher(Int16, '/task_generator_node/task_reset', 10)
    finished_pub = node.create_publisher(Empty, '/task_generator_node/finished', 10)
    commands, statuses = [], []
    velocity = [0., 0.]
    def command(msg):
        velocity[:] = [msg.linear.x, msg.angular.z]
        commands.append((time.monotonic(), msg.linear.x, msg.angular.z))
    node.create_subscription(Twist, namespace + '/cmd_vel', command, 100)
    node.create_subscription(String, namespace + '/lightnav/status', lambda m: statuses.append(json.loads(m.data)), 100)
    sim = 10.
    pose = [0., 0., 0.]
    pause = False
    sensors = True
    last_camera_wall = 0.
    def advance(seconds, *, image_stamp=None, images=True, copies=1):
        nonlocal sim, last_camera_wall
        deadline = time.monotonic() + seconds
        last = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic(); dt = now - last; last = now
            if not pause:
                sim += dt
                pose[0] += velocity[0] * math.cos(pose[2]) * dt
                pose[1] += velocity[0] * math.sin(pose[2]) * dt
                pose[2] += velocity[1] * dt
            stamp = Clock(); stamp.clock.sec = int(sim); stamp.clock.nanosec = int(sim % 1 * 1e9)
            clock_pub.publish(stamp)
            if sensors:
                odom = Odometry(); odom.header.stamp = stamp.clock
                odom.header.frame_id = 'Ai2_Bot2/odom'; odom.child_frame_id = 'Ai2_Bot2/base_footprint'
                odom.pose.pose.position.x, odom.pose.pose.position.y = pose[0], pose[1]
                odom.pose.pose.orientation.z = math.sin(pose[2]/2); odom.pose.pose.orientation.w = math.cos(pose[2]/2)
                odom_pub.publish(odom)
                # Isaac's camera runs at roughly 4 Hz in this setup. Avoid a
                # zero-latency peer driving new paths faster than the 10 Hz MPC.
                if images and now - last_camera_wall >= .25:
                    last_camera_wall = now
                    image = Image()
                    camera_time = sim if image_stamp is None else image_stamp
                    image.header.stamp.sec = int(camera_time)
                    image.header.stamp.nanosec = int(camera_time % 1 * 1e9)
                    image.height, image.width, image.step = 24, 32, 96
                    image.encoding = 'rgb8'; image.data = bytes([90, 140, 190] * 32 * 24)
                    for _ in range(copies):
                        image_pub.publish(image)
            rclpy.spin_once(node, timeout_sec=.025)

    def nonzero_since(index):
        return any(abs(v) + abs(w) > 1e-5 for _,v,w in commands[index:])

    def ready(episode):
        ready_pub.publish(String(data=json.dumps({'stage': 'episode', 'ready': True, 'episode': episode})))
        instruction_pub.publish(String(data='Move forward through the room.'))

    profile_path = Path(__file__).resolve().parents[2] / 'arena_bringup/configs/benchmark/profiles/lightnav_grscenes.yaml'
    profile = yaml.safe_load(profile_path.read_text()); profile['server']['endpoint'] = 'ws://127.0.0.1:18050'
    report = {}
    with tempfile.TemporaryDirectory(prefix='lightnav-replay-') as tmp:
        local_profile = Path(tmp) / 'profile.yaml'; local_profile.write_text(yaml.safe_dump(profile))
        log = Path(tmp) / 'client.log'
        failure_marker = Path(tmp) / 'fail-solver'
        # Inject a solver exception only in this replay subprocess. The production
        # adapter and staged upstream source remain unchanged.
        runner = '''
import sys
from pathlib import Path
import arena_vln_models.lightnav_client as adapter
original_factory = adapter.controller_for
def factory(*args):
    controller = original_factory(*args)
    original_solve = controller.solve
    def solve(*args, **kwargs):
        if Path(sys.argv[1]).exists():
            raise RuntimeError('replay injected solver failure')
        return original_solve(*args, **kwargs)
    controller.solve = solve
    return controller
adapter.controller_for = factory
raise SystemExit(adapter.main(sys.argv[2:]))
'''
        with log.open('w') as stream:
            process = subprocess.Popen([sys.executable, '-c', runner, str(failure_marker),
                '--profile', str(local_profile), '--run-token', 'ros-replay'], stdout=stream, stderr=subprocess.STDOUT)
            try:
                advance(3)
                if process.poll() is not None:
                    raise RuntimeError(log.read_text())
                assert not nonzero_since(0), 'motion before episode authorization'
                reset_pub.publish(Int16(data=0)); advance(.2); ready(0); advance(3)
                assert nonzero_since(0), 'no native MPC motion'
                report['forward'] = True
                frozen = sim
                advance(.5, image_stamp=frozen, copies=3)
                requests = peer['requests']
                advance(.7, image_stamp=frozen, copies=3)
                assert peer['requests'] == requests, 'duplicate camera stamp submitted again'
                advance(.3, image_stamp=frozen-.2)
                assert peer['requests'] == requests, 'out-of-order camera frame submitted'
                advance(1.2, images=False)
                assert peer['requests'] == requests, 'timer resent old image without a camera callback'
                assert not nonzero_since(len(commands)-5), 'frozen camera did not stop motion'
                report['duplicate_and_reordered_camera_frames'] = True
                advance(1)
                peer['delay'] = .6
                advance(1.5, copies=3)
                advance(1, images=False)
                requests = peer['requests']
                advance(.4, images=False)
                assert peer['requests'] == requests, 'busy frame replayed after response'
                peer['delay'] = 0.
                report['busy_frames_dropped'] = True
                advance(1)
                index = len(commands); pause = True; advance(.8)
                cutoff = time.monotonic(); advance(.4)
                assert not any(abs(v)+abs(w)>1e-5 for t,v,w in commands[index:] if t>cutoff), 'clock pause did not stop'
                pause = False; advance(1)
                report['clock_pause'] = True
                peer['delay'] = 2.4; advance(3)
                index = len(commands); advance(.4)
                assert not nonzero_since(index), 'expired model path did not stop'
                report['path_expiry'] = True
                peer['delay'] = 0.; advance(3)
                failure_marker.touch(); advance(.5)
                index = len(commands); advance(.4)
                assert not nonzero_since(index), 'solver failure did not stop'
                assert any(s.get('status') == 'error' for s in statuses)
                report['solver_failure'] = True
                failure_marker.unlink(); advance(1)
                sensors = False; advance(.7)
                index = len(commands); advance(.4)
                assert not nonzero_since(index), 'missing odometry did not stop'
                report['odometry_watchdog'] = True
                sensors = True; advance(1)
                peer['mode'] = 'rotate'; advance(2)
                assert any(abs(r[2])>.05 for r in commands[-25:]), 'pure rotation lost yaw'
                report['pure_rotation'] = True
                peer['mode'] = 'stop'; advance(1)
                index = len(commands); advance(.5)
                assert not nonzero_since(index), 'model STOP did not stop'
                report['stop'] = True
                peer.update(mode='forward', delay=.8)
                reset_pub.publish(Int16(data=1)); advance(.2); ready(1); advance(.3)
                # Reset while the native client has an outstanding response.
                reset_pub.publish(Int16(data=2)); advance(.2)
                index = len(commands); advance(1.1)
                assert not nonzero_since(index), 'late previous-episode result moved robot'
                peer['delay'] = 0.
                ready(2); advance(2)
                assert nonzero_since(index), 'second episode did not resume'
                report['reset_and_late_response'] = True
                connections = peer['connections']
                instruction_pub.publish(String(data='Move forward through the room.')); advance(.5)
                assert peer['connections'] == connections, 'duplicate instruction reset model history'
                # Matching reset delivered after readiness must not disarm the episode.
                reset_pub.publish(Int16(data=2)); advance(.5)
                assert any(s.get('episode') == 2 and s.get('status') == 'lightnav_command' for s in statuses[-30:])
                report['duplicate_instruction_and_late_reset'] = True
                reset_pub.publish(Int16(data=1)); ready(1); advance(.5)
                assert statuses[-1]['episode'] == 2, 'old episode notification rolled state backwards'
                report['old_episode_ignored'] = True
                old_sim = sim
                sim = 1.; advance(.5)
                index = len(commands); advance(.3)
                assert not nonzero_since(index), 'clock rewind did not stop old episode'
                reset_pub.publish(Int16(data=0)); advance(.2); ready(0); advance(2)
                assert any(s.get('status') == 'lightnav_inference' and
                           s['response']['capture_stamp_ns'] < old_sim*1e9 for s in statuses[-80:])
                report['clock_rewind_new_session'] = True
                peer['mode'] = 'disconnect'; advance(1.1)
                index = len(commands); advance(.7)
                assert not nonzero_since(index), 'disconnect did not stop'
                report['disconnect'] = True
                assert len(node.get_publishers_info_by_topic(namespace + '/cmd_vel')) == 1
                report['single_publisher'] = True
                assert all(abs(v) <= .40001 and abs(w) <= 1.50001 for _,v,w in commands)
                report['robot_limits'] = True
                key = lambda s: (s['episode'], s['native_episode'], s['request_id'])
                predicted = {key(s) for s in statuses if s.get('status') == 'lightnav_inference'}
                accepted = {key(s) for s in statuses if s.get('status') == 'path_accepted'}
                controls = [s for s in statuses if s.get('status') == 'lightnav_command']
                assert controls and all(key(s) in predicted & accepted for s in controls)
                report['control_prediction_association'] = True
                frame_keys = [(s['episode'], s['native_episode'], s['response']['capture_stamp_ns'])
                              for s in statuses if s.get('status') == 'lightnav_inference']
                assert len(frame_keys) == len(set(frame_keys)), 'duplicate model input in one session'
                report['unique_inference_frames'] = True
                peer['mode'] = 'forward'; advance(1.5)
                finished_pub.publish(Empty()); advance(.3)
                index = len(commands); advance(.4)
                assert not nonzero_since(index), 'Arena finished event did not stop'
                assert any(s.get('termination_reason') == 'task_finished' for s in statuses[-15:])
                report['arena_finished_stop'] = True
                connections = peer['connections']
                ready_pub.publish(String(data=json.dumps({
                    'stage': 'episode', 'ready': False, 'episode': 0,
                    'details': {'reason': 'episode_finished'},
                })))
                advance(.3)
                ready(0); advance(.5)
                assert not nonzero_since(index), 'late readiness restarted a finished episode'
                assert peer['connections'] == connections, 'finished barrier reopened model session'
                report['finished_barrier_preserves_terminal_state'] = True
                report.update(requests=peer['requests'], connections=peer['connections'], command_samples=len(commands))
            except BaseException:
                print(log.read_text()[-10000:], file=sys.stderr)
                print(json.dumps(statuses[-12:], indent=2), file=sys.stderr)
                raise
            finally:
                process.send_signal(2)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                if process.returncode:
                    print(log.read_text()[-10000:], file=sys.stderr)
    node.destroy_node(); rclpy.shutdown()
    loop.call_soon_threadsafe(loop.stop)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
