"""Real Jazzy RPP and kinematic odometry, in a dedicated ROS domain.

No model server, Isaac process, GPU, private ROS messages or custom controller.
"""
import asyncio
import json
import math
import os
import signal
import subprocess
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('ARENA_RUN_RPP_TESTS') != '1',
                               reason='requires sourced Jazzy and an isolated ROS domain')


def test_native_rpp_motion_preemption_rotation_reset_and_errors(tmp_path):
    assert int(os.environ['ROS_DOMAIN_ID']) >= 70, 'Use a dedicated test domain, never benchmark domain 1'
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
    from nav_msgs.msg import Odometry, Path
    from nav2_msgs.action import FollowPath
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster
    from arena_vln_models.lifecycle import WaypointLifecycle

    log_path = tmp_path / 'nav2.log'
    with log_path.open('w') as log:
        proc = subprocess.Popen(['ros2', 'launch', 'arena_vln_models', 'follow_path.launch.py',
                                 'robot:=Ai2_Bot2', 'namespace:=/rpp_test', 'frame:=rpp/',
                                 f'record_data_dir:={tmp_path}'],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    node = Node('rpp_test_plant')
    lifecycle = WaypointLifecycle(node, '/rpp_test', transition_timeout=15., record_dir=str(tmp_path))
    action = ActionClient(node, FollowPath, '/rpp_test/follow_path')
    clock = node.create_publisher(Clock, '/clock', 10)
    odom = node.create_publisher(Odometry, '/rpp_test/odom', 10)
    status = node.create_publisher(String, '/rpp_test/model/status', 10)
    ready = node.create_publisher(String, '/example_task/eval_ready',
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    clients = []
    tf = TransformBroadcaster(node)
    state = dict(x=0., y=0., yaw=0., sim=10., cmd=Twist(), freeze=False, peak_v=0., peak_w=0.)

    def velocity(msg):
        state['cmd'] = msg
        state['peak_v'] = max(state['peak_v'], abs(msg.linear.x))
        state['peak_w'] = max(state['peak_w'], abs(msg.angular.z))
    node.create_subscription(Twist, '/rpp_test/cmd_vel', velocity, 10)

    async def pump():
        last = time.monotonic()
        while True:
            assert proc.poll() is None, log_path.read_text()[-6000:]
            now = time.monotonic(); dt = min(now-last, .1); last = now
            if not state['freeze']:
                state['sim'] += dt
                state['x'] += dt * state['cmd'].linear.x * math.cos(state['yaw'])
                state['y'] += dt * state['cmd'].linear.x * math.sin(state['yaw'])
                state['yaw'] += dt * state['cmd'].angular.z
                msg = Odometry()
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(state['sim'] * 1e9), 10**9)
                msg.header.frame_id, msg.child_frame_id = 'rpp/odom', 'rpp/base_footprint'
                msg.pose.pose.position.x, msg.pose.pose.position.y = state['x'], state['y']
                msg.pose.pose.orientation.z = math.sin(state['yaw']/2)
                msg.pose.pose.orientation.w = math.cos(state['yaw']/2)
                msg.twist.twist = state['cmd']
                clock.publish(Clock(clock=msg.header.stamp)); odom.publish(msg)
                trans = TransformStamped(header=msg.header, child_frame_id=msg.child_frame_id)
                trans.transform.translation.x, trans.transform.translation.y = state['x'], state['y']
                trans.transform.rotation = msg.pose.pose.orientation
                tf.sendTransform(trans)
            rclpy.spin_once(node, timeout_sec=0.)
            await asyncio.sleep(.01)

    async def until(predicate, timeout=15.):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, log_path.read_text()[-6000:]
            await asyncio.sleep(.02)

    async def result(future, timeout=15.):
        await until(future.done, timeout)
        return future.result()

    async def send(offsets):
        path = Path()
        path.header.frame_id = 'rpp/odom'
        path.header.stamp.sec, path.header.stamp.nanosec = divmod(int(state['sim']*1e9), 10**9)
        x0, y0, yaw = state['x'], state['y'], state['yaw']
        for x, y, angle in offsets:
            pose = PoseStamped(header=path.header)
            pose.pose.position.x = x0 + math.cos(yaw)*x - math.sin(yaw)*y
            pose.pose.position.y = y0 + math.sin(yaw)*x + math.cos(yaw)*y
            pose.pose.orientation.z = math.sin((yaw+angle)/2)
            pose.pose.orientation.w = math.cos((yaw+angle)/2)
            path.poses.append(pose)
        return await result(action.send_goal_async(FollowPath.Goal(
            path=path, controller_id='FollowPath', goal_checker_id='goal_checker', progress_checker_id='progress_checker')))

    async def exercise():
        pumping = asyncio.create_task(pump())
        try:
            await lifecycle.begin('first', 0)
            await until(action.server_is_ready)
            rejected = await send([(0., 0., 0.), (.2, 0., 0.)])
            assert not rejected.accepted
            await lifecycle.activate()
            lifecycle.session.released = True
            status.publish(String(data=json.dumps(dict(session_id='first', episode=0, status='ready'))))
            await until(lambda: lifecycle.session.ready(time.monotonic()))
            handle = await send([(0., 0., 0.), (.6, 0., 0.)])
            assert handle.accepted
            assert (await result(handle.get_result_async())).status == GoalStatus.STATUS_SUCCEEDED
            assert state['x'] > .5 and abs(state['y']) < .05
            assert not lifecycle.session.terminal, 'A local path result must not declare model STOP'
            # Native preemption replaces a longer path, then stop cancels/drains it.
            long = await send([(0., 0., 0.), (2., 0., 0.)])
            await until(lambda: state['cmd'].linear.x > .1)
            before = state['yaw']
            turn = await send([(0., 0., 0.), (.2, .1, .4), (.5, .4, .6)])
            assert turn.accepted
            await until(lambda: state['yaw'] > before+.1)
            await lifecycle.stop()
            assert not lifecycle.active_goals and abs(state['cmd'].linear.x) < 1e-5
            assert (await result(long.get_result_async())).status in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED)
            await lifecycle.activate()
            before = state['yaw']
            await send([(0., 0., 0.), (.2, -.1, -.4), (.5, -.4, -.6)])
            await until(lambda: state['yaw'] < before-.1)
            await lifecycle.stop()
            # Strict terminal yaw permits in-place goals in both directions.
            for angle in (-.5, .5):
                await lifecycle.activate()
                x, y, yaw = state['x'], state['y'], state['yaw']
                handle = await send([(0., 0., angle)])
                assert (await result(handle.get_result_async())).status == GoalStatus.STATUS_SUCCEEDED
                assert abs(state['yaw']-yaw-angle) < .05
                assert math.hypot(state['x']-x, state['y']-y) < .02
                await lifecycle.stop()
            await lifecycle.activate()
            x, y = state['x'], state['y']
            handle = await send([(0., 0., 0.), (.12, 0., 0.)])
            assert (await result(handle.get_result_async())).status == GoalStatus.STATUS_SUCCEEDED
            assert math.hypot(state['x']-x, state['y']-y) > .08
            empty = await send([])
            empty_result = await result(empty.get_result_async())
            assert empty_result.status == GoalStatus.STATUS_ABORTED
            assert empty_result.result.error_code != 0
            # Native client cancellation and heartbeat hold use the same server.
            moving = await send([(0., 0., 0.), (2., 0., 0.)])
            await until(lambda: state['cmd'].linear.x > .1)
            await result(moving.cancel_goal_async())
            assert (await result(moving.get_result_async())).status == GoalStatus.STATUS_CANCELED
            lifecycle.session.received_at = time.monotonic() - 3.
            assert await lifecycle.poll() == 'model_heartbeat_timeout'
            assert not lifecycle.controller_active and abs(state['cmd'].linear.x) < 1e-5
            lifecycle.session.receive(dict(session_id='first', episode=0, status='ready'), time.monotonic())
            assert await lifecycle.poll() == ''
            assert lifecycle.controller_active
            lifecycle.session.receive(dict(session_id='first', episode=0, status='hold'), time.monotonic())
            assert await lifecycle.poll() == 'model_hold'
            assert not lifecycle.controller_active
            lifecycle.session.receive(dict(session_id='first', episode=0, status='running'), time.monotonic())
            assert await lifecycle.poll() == ''
            # Wall-clock smoother drains during a normal cancellation even if sim time pauses.
            await send([(0., 0., 0.), (2., 0., 0.)])
            await until(lambda: state['cmd'].linear.x > .1)
            state['freeze'] = True
            await lifecycle.begin('second', 1)
            assert abs(state['cmd'].linear.x) < 1e-5 and not lifecycle.active_goals
            assert not lifecycle.session.receive(dict(session_id='first', episode=0, status='stop'), time.monotonic())
            assert not lifecycle.session.terminal
            state['freeze'] = False
            await lifecycle.activate()
            lifecycle.session.released = True
            lifecycle.session.receive(dict(session_id='second', episode=1, status='stop'), time.monotonic())
            assert await lifecycle.poll() == 'model_stop'
            assert not lifecycle.controller_active
            await lifecycle.begin('third', 2)
            lifecycle.session.released = True
            lifecycle.session.receive(dict(session_id='third', episode=2, status='error', reason='client failed'), time.monotonic())
            assert await lifecycle.poll() == 'model_error'
            assert lifecycle.session.reason == 'client failed'
            assert node.count_publishers('/rpp_test/cmd_vel') == 1
            assert state['peak_v'] <= .40001 and state['peak_w'] <= 1.50001
            # Exercise the shipped example's readiness, odom conversion and STOP.
            await lifecycle.begin('example', 3)
            with (tmp_path / 'example.log').open('w') as output:
                clients.append(subprocess.Popen([
                    'ros2', 'run', 'arena_vln_models', 'example_client',
                    '--namespace', '/rpp_test', '--task-namespace', '/example_task',
                    '--points', '[[0,0,0],[0.15,0,0]]'],
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True))
            def release(value):
                ready.publish(String(data=json.dumps(dict(
                    stage='episode', ready=value, session_id='example', episode=3))))
            release(False)
            await until(lambda: lifecycle.session.ready(time.monotonic()))
            await lifecycle.activate()
            lifecycle.session.released = True
            release(True)
            await until(lambda: lifecycle.session.status == 'stop')
            assert await lifecycle.poll() == 'model_stop'
        finally:
            pumping.cancel()
            try:
                await pumping
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(exercise())
        assert 'process has died' not in log_path.read_text(), log_path.read_text()[-6000:]
    finally:
        for client in clients:
            os.killpg(client.pid, signal.SIGINT)
            try:
                client.wait(timeout=8.)
            except subprocess.TimeoutExpired:
                os.killpg(client.pid, signal.SIGKILL); client.wait()
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10.)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        lifecycle.destroy(); node.destroy_node(); rclpy.try_shutdown()
    assert (tmp_path / 'rtf.csv').is_file()
    for name in ('odom.csv', 'cmd_vel.csv'):
        assert len((tmp_path / name).read_text().splitlines()) > 2
