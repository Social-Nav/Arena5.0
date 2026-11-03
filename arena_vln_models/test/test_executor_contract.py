"""Pure contracts for async execution, goal replacement, decoding and wiring."""
from __future__ import annotations

import json
import math
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from arena_vln_models.adapters.internnav import INTERNNAV_CAPABILITY
from arena_vln_models.core import ModelObservation, Pose2D
from arena_vln_models.execution import (
    AsyncInferenceWorker,
    ExecutionConfig,
    ExecutionCore,
)
from arena_vln_models.goal_state import GoalStateMachine
from arena_vln_models.model_adapter import ModelRawOutput, prediction_from_raw
from arena_vln_models.sensor_decode import decode_depth, decode_rgb

ROOT = Path(__file__).resolve().parents[2]


def _observation(*, stamp_ns=1, received_at=None, instruction='go'):
    return ModelObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        instruction=instruction,
        rgb_image=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_image=np.ones((8, 8), dtype=np.float32),
        camera_intrinsics=(1.0, 0.0, 4.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0),
        frame_id='odom',
        stamp_ns=stamp_ns,
        received_at=time.monotonic() if received_at is None else received_at,
        session_id='s',
        episode=0,
    )


class _BlockingAdapter:
    name = 'blocking'
    capability = INTERNNAV_CAPABILITY

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def reset(self, **kwargs):
        return None

    def infer(self, observation):
        self.started.set()
        self.release.wait(5.0)
        return ModelRawOutput(adapter=self.name, local_waypoints=[(0.5, 0.0, 0.0)], stop=False)

    def to_prediction(self, raw, observation):
        return prediction_from_raw(raw, observation, adapter_name=self.name)


def test_async_worker_one_in_flight_heartbeat_and_stale_discard():
    adapter = _BlockingAdapter()
    core = ExecutionCore(adapter, ExecutionConfig(model='blocking', observation_max_age_sec=30.0))
    core.reset('s', 0)
    core.release()
    core.set_instruction('go')
    worker = AsyncInferenceWorker(core)
    observation = _observation(received_at=time.monotonic())
    assert worker.submit(observation, time.monotonic()) is True
    assert adapter.started.wait(1.0)
    assert worker.in_flight is True
    # The ROS thread can still run heartbeats/resets while inference is blocked.
    assert worker.submit(observation, time.monotonic()) is False
    worker.invalidate()
    adapter.release.set()
    deadline = time.monotonic() + 2.0
    decision = None
    while time.monotonic() < deadline:
        decision = worker.harvest()
        if decision is not None:
            break
        time.sleep(0.01)
    assert decision is not None
    assert decision.action == 'skip'
    assert decision.reason == 'stale_inference'
    worker.shutdown()


def test_goal_state_machine_replaces_only_after_confirmation_and_ignores_stale_callbacks():
    machine = GoalStateMachine(cancel_timeout_sec=1.0)
    machine.reset('s')
    first = machine.next_goal(poses=[(0.0, 0.0, 0.0)], frame_id='odom', stamp_ns=1)
    assert machine.request(first, now=0.0) == ['send:1']
    assert machine.goal_response(1, True) == []
    assert machine.state == 'active'

    second = machine.next_goal(poses=[(0.5, 0.0, 0.0)], frame_id='odom', stamp_ns=2)
    assert machine.request(second, now=0.1) == ['cancel:1']
    assert machine.state == 'canceling'
    # Out-of-order stale callback must not replace the active goal.
    assert machine.goal_response(99, True) == []
    assert machine.goal_result(99, 4) == []
    # Cancel ACK alone must not release the pending path.
    assert machine.cancel_response(1, now=0.2) == []
    assert machine.state == 'canceling'
    # A non-terminal status must not release it either.
    assert machine.goal_result(1, 2) == []
    # Only the terminal result for the old serial releases the pending path.
    assert machine.goal_result(1, 5) == ['send:2']
    assert machine.state == 'sending'
    assert machine.goal_response(2, True) == []
    # A late result for the old serial is ignored.
    assert machine.goal_result(1, 5) == []
    assert machine.active.serial == 2


def test_goal_state_machine_cancel_timeout_and_stop():
    machine = GoalStateMachine(cancel_timeout_sec=1.0)
    machine.reset('s')
    first = machine.next_goal(poses=[(0.0, 0.0, 0.0)], frame_id='odom', stamp_ns=1)
    machine.request(first, now=0.0)
    machine.goal_response(1, True)
    second = machine.next_goal(poses=[(1.0, 0.0, 0.0)], frame_id='odom', stamp_ns=2)
    assert machine.request(second, now=0.1) == ['cancel:1']
    assert machine.cancel_timeout(0.5) == []
    # Timeout without a terminal result fails closed and never sends a second goal.
    assert machine.cancel_timeout(1.5) == ['error:cancel_timeout:1']
    assert machine.state == 'error'
    assert machine.pending is not None
    assert machine.stop() == ['cancel:1']
    assert machine.pending is None


class _Msg:
    def __init__(self, *, data, width, height, step, encoding, is_bigendian=0):
        self.data = data
        self.width = width
        self.height = height
        self.step = step
        self.encoding = encoding
        self.is_bigendian = is_bigendian


def test_decode_rgb_respects_padded_step():
    # 2x2 RGB with 2 padding bytes per row (6 bytes data + 2 pad).
    rows = []
    for y in range(2):
        row = bytes([
            y * 10 + 1, y * 10 + 2, y * 10 + 3,
            y * 10 + 4, y * 10 + 5, y * 10 + 6,
            0, 0,
        ])
        rows.append(row)
    msg = _Msg(data=b''.join(rows), width=2, height=2, step=8, encoding='rgb8')
    image = decode_rgb(msg)
    assert image.shape == (2, 2, 3)
    assert image[0, 0].tolist() == [1, 2, 3]
    assert image[1, 1].tolist() == [14, 15, 16]


def test_decode_depth_respects_endianness_and_step():
    values = [1000, 2000, 3000, 4000]
    little = b''.join(struct.pack('<H', value) for value in values)
    big = b''.join(struct.pack('>H', value) for value in values)
    msg_little = _Msg(data=little, width=2, height=2, step=4, encoding='16UC1', is_bigendian=0)
    msg_big = _Msg(data=big, width=2, height=2, step=4, encoding='16UC1', is_bigendian=1)
    np.testing.assert_allclose(decode_depth(msg_little), [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(decode_depth(msg_big), [[1.0, 2.0], [3.0, 4.0]])

    floats = [1.5, 2.5]
    little_f = b''.join(struct.pack('<f', value) for value in floats)
    big_f = b''.join(struct.pack('>f', value) for value in floats)
    np.testing.assert_allclose(
        decode_depth(_Msg(data=little_f, width=2, height=1, step=8, encoding='32FC1', is_bigendian=0)),
        [[1.5, 2.5]],
    )
    np.testing.assert_allclose(
        decode_depth(_Msg(data=big_f, width=2, height=1, step=8, encoding='32FC1', is_bigendian=1)),
        [[1.5, 2.5]],
    )


def test_lightnav_startup_requires_runtime_websocket_client(monkeypatch):
    import arena_vln_models.adapters.lightnav as lightnav

    monkeypatch.setattr(lightnav, 'websocket_transport_available', lambda: False)
    adapter = lightnav.LightNavAdapter(params={'server_url': 'ws://fixture'})
    with pytest.raises(Exception, match='websockets'):
        adapter.doctor()


def test_follow_path_launch_has_one_conditional_executor():
    text = (ROOT / 'arena_vln_models/launch/follow_path.launch.py').read_text()
    assert "if get('model_adapter')" in text
    assert "executable='model_executor'" in text
    assert "namespace=get('namespace')" in text
    assert "model_server_url" in text
    assert "inference_hz" in text
    assert "request_timeout_sec" in text
    assert "model_trace_dir" in text


def test_executor_uses_exact_stamp_tf_not_latest_odom():
    text = (ROOT / 'arena_vln_models/arena_vln_models/executor.py').read_text()
    assert 'lookup_transform' in text
    assert 'Time(nanoseconds=stamp_ns)' in text
    assert 'waiting_for_exact_stamp_pose' in text
    assert 'pose=self._odom_pose' not in text
    assert 'AsyncInferenceWorker' in text
    assert 'GoalStateMachine' in text


def test_evaluator_propagates_unified_model_parameters(tmp_path, monkeypatch, capsys):
    from arena_vln_models.runner import main

    for name in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'FASTDDS_BUILTIN_TRANSPORTS',
                 'ROS_LOCALHOST_ONLY', 'ROS_AUTOMATIC_DISCOVERY_RANGE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('ARENA_WS_DIR', str(tmp_path))
    assert main(['plan', 'run', '--model', 'navila']) == 0
    plan = json.loads(capsys.readouterr().out)
    command = ' '.join(plan['commands'][0])
    assert '--model-server-url' in command
    assert 'tcp://127.0.0.1:54321' in command
    assert '--model-inference-hz' in command
    assert '--model-request-timeout-sec' in command
    evaluator = (ROOT / 'arena_bringup/arena_bringup/internnav_eval.py').read_text()
    for token in ('--model-server-url', '--model-inference-hz', '--model-request-timeout-sec',
                  'model_trace_dir', 'model_server_url', 'model_inference_hz'):
        assert token in evaluator
