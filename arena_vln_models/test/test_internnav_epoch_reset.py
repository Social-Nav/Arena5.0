"""Exercise the installed client reset code against ROS message_filters."""
import ast
from pathlib import Path
import threading
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip('message_filters')
from message_filters import ApproximateTimeSynchronizer, SimpleFilter
from sensor_msgs.msg import Image


class Lock:
    def acquire_write(self):
        pass

    def release_write(self):
        pass


def client():
    path = Path('/opt/arena_ws/deps/InternNav/scripts/realworld/http_internvla_client.py')
    if not path.exists():
        pytest.skip('InternNav checkout not mounted')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Go2Manager')
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {'reset_runtime_state', 'clock_callback'}]
    context = {'_reset_policy_state': lambda reason: None,
               'rgb_depth_rw_lock': Lock(), 'odom_rw_lock': Lock()}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), context)
    node = context['Go2Manager']()
    a, b = SimpleFilter(), SimpleFilter()
    node.syncronizer = ApproximateTimeSynchronizer([a, b], 1, .1)
    paired = []
    node.syncronizer.registerCallback(lambda x, y: paired.append(x.header.stamp.sec))
    node.runtime_lock = threading.RLock()
    node.runtime_epoch = 0
    node.odom_queue = []
    node.stop_motion = lambda reason: None
    node.episode_started = True
    node.eval_ready_episode = 0
    node.last_clock_ns = 100 * 10**9
    return node, a, b, paired


def msg(sec):
    m = Image()
    m.header.stamp.sec = sec
    return m


@pytest.mark.parametrize('reset', ['episode', 'clock'])
def test_high_timestamp_queue_no_longer_blocks_new_epoch(reset):
    node, rgb, depth, paired = client()
    rgb.signalMessage(msg(100))
    rgb.signalMessage(msg(1))
    depth.signalMessage(msg(1))
    assert paired == []  # The original queue-size-one failure is reproduced.
    if reset == 'episode':
        node.reset_runtime_state('task_reset')
    else:
        node.clock_callback(SimpleNamespace(clock=SimpleNamespace(sec=1, nanosec=0)))
        assert not node.episode_started
    assert node.runtime_epoch == 1
    assert all(not q for q in node.syncronizer.queues)
    assert node.rgb_time == node.depth_time == node.last_rgb_time == node.last_depth_time == 0.
    rgb.signalMessage(msg(2)); depth.signalMessage(msg(2))
    assert paired == [2]
    node.reset_runtime_state('next_episode')
    rgb.signalMessage(msg(3)); depth.signalMessage(msg(3))
    assert paired == [2, 3]


def test_reset_serializes_with_synchronizer_pairing():
    node, rgb, _, _ = client()
    rgb.signalMessage(msg(100))
    node.syncronizer.lock.acquire()
    worker = threading.Thread(target=node.reset_runtime_state, args=('concurrent_reset',))
    try:
        worker.start()
        worker.join(.05)
        assert worker.is_alive()
        assert any(node.syncronizer.queues)
    finally:
        node.syncronizer.lock.release()
        worker.join(1.)
    assert not worker.is_alive()
    assert all(not q for q in node.syncronizer.queues)


def test_late_http_response_cannot_update_new_epoch_counters():
    node, _, _, _ = client()
    path = Path('/opt/arena_ws/deps/InternNav/scripts/realworld/http_internvla_client.py')
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in {'dual_sys_eval', 'StaleEpisodeResult'}]
    context = dict(manager=node, json=json, hashlib=hashlib, time=time,
                   policy_init=True, http_idx=-1, first_running_time=0., _write_trace=lambda *a, **k: None)

    def post(*args, **kwargs):
        node.reset_runtime_state('new_episode_during_http')
        return SimpleNamespace(text='{"trajectory": [[1,0,0]]}', raise_for_status=lambda: None)

    context['requests'] = SimpleNamespace(post=post)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), context)
    with pytest.raises(context['StaleEpisodeResult']):
        context['dual_sys_eval'](None, None, None, expected_epoch=0)
    assert node.runtime_epoch == 1
    assert context['http_idx'] == -1


def test_old_dds_sample_after_rewind_cannot_poison_fresh_sync_queue():
    node, _, _, _ = client()
    path = Path('/opt/arena_ws/deps/InternNav/scripts/realworld/http_internvla_client.py')
    definition = next(n for n in ast.parse(path.read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'EpochImageSynchronizer')
    context = {'ApproximateTimeSynchronizer': ApproximateTimeSynchronizer}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), context)
    rgb, depth = SimpleFilter(), SimpleFilter()
    node.syncronizer = context['EpochImageSynchronizer']([rgb, depth], node)
    paired = []
    node.syncronizer.registerCallback(lambda a, b: paired.append(a.header.stamp.sec))
    node.clock_callback(SimpleNamespace(clock=SimpleNamespace(sec=1, nanosec=0)))
    rgb.signalMessage(msg(100)); depth.signalMessage(msg(100))
    rgb.signalMessage(msg(1)); depth.signalMessage(msg(1))
    assert paired == [1]
