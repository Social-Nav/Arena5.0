"""Isolated ROS replay of the production NaVILA adapter with a synthetic TCP peer.

This validates lifecycle/control, not model navigation. Use a non-production DDS domain.
"""
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int16
import yaml
from arena_vln_models.navila_control import TCPModel


def main():
    assert os.environ.get('ROS_DOMAIN_ID') not in (None, '', '0', '1')
    peer = {'text': 'move forward 25 cm', 'delay': .05, 'requests': 0, 'images': []}
    server = socket.socket(); server.bind(('127.0.0.1', 0)); server.listen()
    port = server.getsockname()[1]
    def serve():
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            with connection:
                try:
                    n = int.from_bytes(TCPModel.receive(connection, 8), 'big')
                    request = json.loads(TCPModel.receive(connection, n))
                    peer['requests'] += 1
                    peer['images'].append(len(request['images']))
                    text, delay = peer['text'], peer['delay']
                    time.sleep(delay)
                    if text == 'disconnect':
                        continue
                    response = json.dumps(text).encode()
                    connection.sendall(len(response).to_bytes(8, 'big') + response)
                except (OSError, ValueError):
                    pass
    threading.Thread(target=serve, daemon=True).start()
    rclpy.init(); node = Node('navila_replay')
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ns = '/task_generator_node/Ai2_Bot2'
    rgb = node.create_publisher(Image, ns+'/head_camera/image', qos_profile_sensor_data)
    odom = node.create_publisher(Odometry, ns+'/odom', qos_profile_sensor_data)
    clock = node.create_publisher(Clock, '/clock', qos_profile_sensor_data)
    ready = node.create_publisher(String, '/task_generator_node/eval_ready', latched)
    instruction = node.create_publisher(String, '/task_generator_node/vln_instruction', latched)
    reset = node.create_publisher(Int16, '/task_generator_node/task_reset', 10)
    commands, statuses = [], []
    velocity = [0., 0.]; pose = [0., 0., math.radians(175)]
    def command(msg):
        velocity[:] = [msg.linear.x, msg.angular.z]
        commands.append((time.monotonic(), msg.linear.x, msg.angular.z))
    node.create_subscription(Twist, ns+'/cmd_vel', command, 200)
    node.create_subscription(String, ns+'/navila/status', lambda m: statuses.append(json.loads(m.data)), 1000)
    sim = 10.; speed = 1.; paused = False; sensors = True; last_image_wall = 0.
    def advance(seconds, image_stamp=None, images=True, copies=1):
        nonlocal sim, last_image_wall
        end = time.monotonic()+seconds; previous = time.monotonic()
        while time.monotonic() < end:
            now = time.monotonic(); dt = now-previous; previous = now
            if not paused:
                sim += dt*speed
                pose[0] += velocity[0]*math.cos(pose[2])*dt*speed
                pose[1] += velocity[0]*math.sin(pose[2])*dt*speed
                pose[2] += velocity[1]*dt*speed
            c = Clock(); c.clock.sec = int(sim); c.clock.nanosec = int(sim%1*1e9); clock.publish(c)
            if sensors:
                o = Odometry(); o.header.stamp = c.clock
                o.header.frame_id = 'Ai2_Bot2/odom'; o.child_frame_id = 'Ai2_Bot2/base_footprint'
                o.pose.pose.position.x, o.pose.pose.position.y = pose[:2]
                o.pose.pose.orientation.z = math.sin(pose[2]/2); o.pose.pose.orientation.w = math.cos(pose[2]/2)
                odom.publish(o)
                if images and now-last_image_wall > .07:
                    last_image_wall = now
                    stamp = sim if image_stamp is None else image_stamp
                    image = Image(); image.header.stamp.sec = int(stamp); image.header.stamp.nanosec = int(stamp%1*1e9)
                    image.width, image.height, image.step = 32, 24, 96
                    image.encoding = 'rgb8'; image.data = bytes([110, 120, 130]*32*24)
                    for _ in range(copies): rgb.publish(image)
            rclpy.spin_once(node, timeout_sec=.01)
    def begin(episode):
        reset.publish(Int16(data=episode)); advance(.1)
        ready.publish(String(data=json.dumps({'stage':'episode','episode':episode,'ready':True})))
        instruction.publish(String(data='Walk through the room.'))
    def moving(index): return any(abs(v)+abs(w)>1e-6 for _,v,w in commands[index:])
    profile = yaml.safe_load((Path(__file__).resolve().parents[2]/'arena_bringup/configs/benchmark/profiles/navila_grscenes.yaml').read_text())
    profile['server']['endpoint'] = f'tcp://127.0.0.1:{port}'
    report = {}
    with tempfile.TemporaryDirectory(prefix='navila-replay-') as tmp:
        path = Path(tmp)/'profile.yaml'; path.write_text(yaml.safe_dump(profile))
        log = Path(tmp)/'client.log'
        with log.open('w') as stream:
            proc = subprocess.Popen([sys.executable,'-m','arena_vln_models.navila_client','--profile',str(path),'--run-token','replay'],stdout=stream,stderr=subprocess.STDOUT)
            try:
                advance(2); assert proc.poll() is None, log.read_text()
                assert not moving(0)
                begin(0); advance(2)
                assert moving(0) and any(r['status']=='action_completed' for r in statuses)
                report['forward_and_start_barrier'] = True
                assert sum(r['status']=='episode_ready' and r['episode']==0 for r in statuses) >= 2
                late = []
                late_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
                sub = node.create_subscription(String, ns+'/navila/status', lambda m: late.append(json.loads(m.data)), late_qos)
                advance(.8)
                assert any(r['status']=='episode_ready' and r['episode_started'] for r in late)
                node.destroy_subscription(sub)
                report['late_subscriber_episode_state'] = True
                frozen = sim; advance(1, image_stamp=frozen, copies=3)
                count=peer['requests']; advance(.8,image_stamp=frozen,copies=3)
                assert peer['requests']==count
                advance(.3,image_stamp=frozen-.2); assert peer['requests']==count
                report['duplicate_and_out_of_order'] = True
                peer['text']='turn left 45 degrees'; begin(1); advance(2)
                assert any(r['status']=='action_completed' and r['action']['kind']=='left' and r['actual_yaw_rad']>0 for r in statuses)
                report['left_across_pi'] = True
                peer['text']='turn right 30 degrees'; begin(2); advance(1.5)
                assert any(r['status']=='action_completed' and r['action']['kind']=='right' and r['actual_yaw_rad']<0 for r in statuses)
                report['right'] = True
                paused=True; advance(.65); mark=len(commands); advance(.25); assert not moving(mark)
                paused=False; advance(1.5); assert moving(mark)
                report['pause_wall_watchdog_resume'] = True
                sensors=False; advance(.65); mark=len(commands); advance(.2); assert not moving(mark)
                sensors=True; advance(1.5)
                report['odom_watchdog'] = True
                peer['delay']=.8; peer['text']='move forward 75 cm'; begin(3); advance(.3)
                peer['text']='stop'; begin(4); mark=len(commands); advance(1.8)
                assert not moving(mark) and any(r['status']=='response_discarded' for r in statuses)
                report['busy_request_epoch_isolation'] = True
                peer['delay']=.05; sim=1.; advance(.3); peer['text']='move forward 25 cm'; begin(0); mark=len(commands); advance(1.5)
                assert moving(mark)
                report['clock_rewind_and_new_episode'] = True
                for rate in [.5, 2.]:
                    speed=rate; begin(1 if rate==.5 else 2); advance(2.5 if rate==.5 else 1.5)
                completed=[r for r in statuses if r['status']=='action_completed']
                assert all(abs(r['elapsed_sim']-r['action']['duration_sec'])<.09 for r in completed)
                report['different_simulation_rates'] = True
                speed=1.; peer['text']='I think I should stop because I have finished the instruction.'; begin(3); advance(.7); mark=len(commands); advance(.3); assert not moving(mark)
                assert any(r['status']=='stop' and r.get('termination_reason')=='model_stop' for r in statuses)
                late_stop = []
                sub = node.create_subscription(String, ns+'/navila/status', lambda m: late_stop.append(json.loads(m.data)), late_qos)
                advance(.8)
                assert any(r['status']=='stop' and r.get('terminal') and r.get('termination_reason')=='model_stop' for r in late_stop)
                node.destroy_subscription(sub)
                report['late_subscriber_terminal_state'] = True
                report['model_stop'] = True
                for episode, text in [(4,'disconnect'),(5,'move forward 100 cm')]:
                    peer['text']=text; begin(episode); mark=len(commands); advance(.7); assert not moving(mark)
                report['disconnect_and_invalid_output_stop'] = True
                assert set(peer['images'])=={8}
                report['eight_frame_requests'] = True
                assert proc.poll() is None, log.read_text()
                print(json.dumps({'pass':True,'checks':report,'requests':peer['requests'],'completed_actions':len(completed)},indent=2))
            except Exception:
                print(log.read_text()); print(json.dumps(statuses[-12:],indent=2)); raise
            finally:
                proc.send_signal(signal.SIGINT); proc.wait(timeout=15)
    server.close(); node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__': main()
