import json
import math
import socket
import threading
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena_vln_models.navila_control import FrameHistory, TCPModel, TimedAction, parse_action, sample_indices


@pytest.mark.parametrize('side,values,speed', [('move forward', [25, 50, 75], .5),
    ('turn left', [15, 30, 45], math.pi/6), ('turn right', [15, 30, 45], -math.pi/6)])
def test_native_actions_integrate_to_requested_distance_or_angle(side, values, speed):
    for value in values:
        action = parse_action(f'The next action is {side} {value} ' + ('cm.' if side == 'move forward' else 'degrees.'))
        assert (action.linear_x or action.angular_z) == speed
        assert abs(speed) * action.duration_sec == pytest.approx(value / 100 if side == 'move forward' else math.radians(value))
        run = TimedAction(action, 17.)
        assert run.command(17.) == (action.linear_x, action.angular_z)
        assert run.command(17.01) is None
        assert run.command(17.3) == (action.linear_x, action.angular_z)
        assert run.command(17.3) is None  # no backlog replay
        assert run.command(run.end) == (0., 0.)
        assert run.command(1.) == (0., 0.)  # rewind cannot extend motion


@pytest.mark.parametrize('text', ['server error: move forward 25 cm', 'move forward 100 cm',
    'turn left 90 degree', 'turn left -15 degree', 'stop then move forward 25 cm',
    'The next action is move forward 25 cm and turn left 15 degree.', '', 'nothing', None])
def test_bad_output_never_becomes_motion(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_stop_is_zero_and_sampling_matches_native_uniform_history():
    assert parse_action('The next action is STOP.').duration_sec == 0
    assert parse_action('I think I should stop because I have finished the instruction.').kind == 'stop'
    assert sample_indices(1) == [None]*7 + [0]
    assert sample_indices(8) == list(range(8))
    assert sample_indices(20) == [0, 2, 5, 8, 10, 13, 16, 19]


def test_new_frames_sampling_and_reset():
    h = FrameHistory(); image = Image.new('RGB', (8, 8))
    for ns in [0, 100_000_000, 500_000_000, 600_000_000]:
        assert h.offer(ns, image)
    assert not h.offer(500_000_000, image)
    assert not h.offer(600_000_000, image)
    assert len(h.frames) == 2
    encoded, stamps = h.sample()
    assert len(encoded) == 8 and stamps[-1] == 600_000_000
    h.reset()
    assert h.offer(10, image)
    assert h.sample()[1] == [None]*7 + [10]


@pytest.mark.parametrize('response,success', [((json.dumps('stop').encode()), True),
    (json.dumps('server error: broken').encode(), False), (b'not json', False)])
def test_tcp_fragmented_header_and_payload(response, success):
    server = socket.socket(); server.bind(('127.0.0.1', 0)); server.listen()
    def serve():
        with server, server.accept()[0] as connection:
            n = int.from_bytes(TCPModel.receive(connection, 8), 'big')
            request = json.loads(TCPModel.receive(connection, n))
            assert len(request['images']) == 8
            for byte in len(response).to_bytes(8, 'big') + response:
                connection.sendall(bytes([byte]))
    worker = threading.Thread(target=serve); worker.start()
    model = TCPModel(*server.getsockname())
    try:
        if success:
            assert model.generate(['a']*8, 'stop') == 'stop'
        else:
            with pytest.raises(ValueError): model.generate(['a']*8, 'stop')
    finally:
        worker.join(timeout=3)
