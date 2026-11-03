"""401/403 first-response latch, node loop and redaction tests.

Only a loopback fake HTTP server and dummy credentials are used.  No external
API is contacted and no real credential exists in this file.
"""
import json
import threading
import struct
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import pytest

from arena_bringup.lisn_slow_loop_ros import (
    HttpVlmProducer,
    LisnSlowLoopNode,
    ProducerAuthGate,
    ProducerConfig,
    ProducerUnavailableError,
    SubprocessHttpVlmProducer,
)

DUMMY_KEY = 'DUMMY-AUTH-KEY-NOT-A-SECRET'
DUMMY_SECRET = 'DUMMY-SECRET-IN-ERROR-BODY-OR-QUERY'
PROPOSAL = {
    'controller_mode': 'follow',
    'controller_parameters': {
        'max_linear_velocity_mps': 0.3,
        'max_angular_velocity_rps': 1.0,
        'goal_weight': 1.0,
        'obstacle_weight': 0.05,
        'people_weight': 1.0,
    },
    'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1,
             'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
    'semantic_entities': [],
    'provenance': {'model_version': 'dummy', 'prompt_version': 'v1',
                   'tool_version': 'v1', 'reasoning_latency_ms': 1},
}


class _Handler(BaseHTTPRequestHandler):
    status = 200
    requests = 0
    body = b'{}'
    reason = 'OK'
    delay = 0.0
    body_delay = 0.0

    def do_POST(self):  # noqa: N802
        import time
        type(self).requests += 1
        if type(self).delay:
            time.sleep(type(self).delay)
        self.send_response(type(self).status, self.reason)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(type(self).body)))
        self.end_headers()
        if type(self).body_delay:
            time.sleep(type(self).body_delay)
        self.wfile.write(type(self).body)

    def log_message(self, *args):
        return


def _server(status, body, reason='OK', delay=0.0, body_delay=0.0):
    handler = type('Handler', (_Handler,), {
        'status': status, 'requests': 0, 'body': body, 'reason': reason,
        'delay': delay, 'body_delay': body_delay,
    })
    server = HTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler, thread


def _config(endpoint, *, api_key=DUMMY_KEY, provider='generic'):
    return ProducerConfig(
        endpoint=endpoint,
        api_key=api_key,
        api_key_env='DUMMY_API_KEY_ENV',
        model='dummy-model',
        provider=provider,
        timeout_sec=2.0,
    )


def _produce(producer):
    return producer.produce(
        instruction='walk forward',
        image_bytes=bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]),
        frame_id='map',
        stamp_ns=1,
        image_encoding='rgb8',
        width=2,
        height=2,
        channels=3,
    )


class _FakeLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(str(message))


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _FakeRosNode:
    def __init__(self):
        self.logger = _FakeLogger()

    def get_logger(self):
        return self.logger

    def get_clock(self):
        node = self

        class Clock:
            def now(self):
                class Stamp:
                    nanoseconds = 0
                return Stamp()
        return Clock()


def _node_for_auth_test():
    node = object.__new__(LisnSlowLoopNode)
    node.node = _FakeRosNode()
    node._auth_gate = ProducerAuthGate()
    node._status_pub = _FakePublisher()
    node._session_id = 'session-dummy'
    node._session_episode = 0
    node.engineering = True
    node.wiring_live = False
    node._commit_future = None
    node._commit_meta = None
    node._pending = None
    node._pending_meta = None
    node.producer = None
    return node


class _FakeImage:
    def __init__(self):
        class Stamp:
            sec = 0
            nanosec = 0

        class Header:
            stamp = Stamp()
            frame_id = 'map'

        self.header = Header()
        self.data = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        self.encoding = 'rgb8'
        self.width = 2
        self.height = 2


class _FakeTick:
    def __init__(self, producer):
        self.producer = producer

    def produce(self, **kwargs):
        return self.producer.produce(**kwargs)


class _FakeRunner:
    def scheduled_for_ns(self):
        return 0


@pytest.mark.parametrize('status', [401, 403])
def test_auth_failure_latches_after_one_request(status):
    server, handler, thread = _server(status, b'{"error":"denied"}')
    try:
        producer = HttpVlmProducer(config=_config(
            f'http://127.0.0.1:{server.server_port}/v1/chat/completions'))
        with pytest.raises(ProducerUnavailableError) as first:
            _produce(producer)
        assert first.value.code == 'auth_denied'
        assert first.value.status == status
        assert producer.auth_latched is True
        assert producer.request_count == 1
        assert handler.requests == 1
        for _ in range(3):
            with pytest.raises(ProducerUnavailableError) as later:
                _produce(producer)
            assert later.value.code == 'auth_latched'
        assert producer.request_count == 1
        assert handler.requests == 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_node_multi_tick_latches_one_error_and_zero_extra_requests():
    node = _node_for_auth_test()

    class _FakeProducer:
        auth_latched = True
        produce_calls = 0

        def produce(self, **kwargs):
            type(self).produce_calls += 1
            raise ProducerUnavailableError('auth_denied', status=401)

    node.producer = _FakeProducer()
    error = ProducerUnavailableError('auth_denied', status=401)
    for _ in range(5):
        node._handle_producer_failure(error)
    assert node._auth_gate.error_count == 1
    assert node._auth_gate.last_code == 'auth_denied'
    assert len(node.node.logger.errors) == 1
    assert len(node._status_pub.messages) == 1
    for _ in range(5):
        node._tick_producer(0)
    assert _FakeProducer.produce_calls == 0


def test_non_auth_failure_does_not_latch():
    node = _node_for_auth_test()
    for _ in range(3):
        node._handle_producer_failure(ProducerUnavailableError('transport_unavailable'))
    assert node._auth_gate.latched is False
    assert node._auth_gate.error_count == 0
    assert len(node.node.logger.errors) == 3
    assert node._status_pub.messages == []


def test_http_200_strict_json_still_works():
    body = json.dumps({
        'model': 'dummy-model',
        'choices': [{'message': {'content': json.dumps(PROPOSAL)}, 'finish_reason': 'stop'}],
    }).encode()
    server, handler, thread = _server(200, body)
    try:
        producer = HttpVlmProducer(config=_config(
            f'http://127.0.0.1:{server.server_port}/v1/chat/completions'))
        result = _produce(producer)
        assert result['controller_parameters']['max_linear_velocity_mps'] == 0.3
        assert producer.auth_latched is False
        assert producer.request_count == 1
        assert handler.requests == 1
        assert producer.last_response_meta['status'] == 200
        assert producer.last_response_meta['response_model'] == 'dummy-model'
        assert producer.last_response_meta['finish_reason'] == 'stop'
        assert DUMMY_KEY not in json.dumps(producer.redacted_trace())
        assert DUMMY_KEY not in json.dumps(producer.redacted_config())
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.mark.parametrize('env', [
    {},
    {'GPT_OPEN_API_KEY': '', 'GPT_OPEN_API_VERSION': ''},
    {'ARK_API_KEY': ''},
    {'ARENA_LISN_GEMINI_AK': ''},
])
def test_missing_credential_issues_zero_requests(env):
    server, handler, thread = _server(200, b'{}')
    try:
        config = ProducerConfig.from_env(env)
        assert config.provider == 'none'
        config = replace(
            config,
            endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
        )
        producer = HttpVlmProducer(config=config)
        with pytest.raises(ProducerUnavailableError) as exc:
            _produce(producer)
        assert exc.value.code == 'credential_missing'
        assert producer.request_count == 0
        assert handler.requests == 0
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_endpoint_userinfo_path_query_never_leaks():
    server, handler, thread = _server(401, b'{"error":"denied"}')
    try:
        endpoint = (
            f'http://user:{DUMMY_SECRET}@127.0.0.1:{server.server_port}'
            f'/v1/chat/completions?token={DUMMY_SECRET}'
        )
        producer = HttpVlmProducer(config=_config(endpoint))
        with pytest.raises(ProducerUnavailableError):
            _produce(producer)
        text = json.dumps(producer.redacted_trace()) + json.dumps(producer.redacted_config())
        assert DUMMY_SECRET not in text
        assert DUMMY_KEY not in text
        assert 'user:' not in text
        assert 'token=' not in text
        assert '/v1/' not in text
        assert producer.redacted_config()['endpoint_label'] == 'loopback'
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_exception_and_traceback_do_not_echo_error_body_or_reason():
    server, handler, thread = _server(
        401,
        json.dumps({'error': DUMMY_SECRET, 'key': DUMMY_KEY}).encode(),
        reason=DUMMY_SECRET,
    )
    try:
        producer = HttpVlmProducer(config=_config(
            f'http://127.0.0.1:{server.server_port}/v1/chat/completions'))
        try:
            _produce(producer)
        except ProducerUnavailableError as exc:
            rendered = traceback.format_exc()
            cause = exc.__cause__
        else:
            raise AssertionError('expected auth failure')
        assert DUMMY_SECRET not in rendered
        assert DUMMY_KEY not in rendered
        assert 'Unauthorized' not in rendered
        assert cause is None
        trace_text = json.dumps(producer.redacted_trace())
        assert DUMMY_SECRET not in trace_text
        assert DUMMY_KEY not in trace_text
        assert 'preview' not in trace_text
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_dummy_credential_never_enters_process_environment():
    import os

    before = dict(os.environ)
    server, handler, thread = _server(200, b'{}')
    try:
        producer = HttpVlmProducer(config=_config(
            f'http://127.0.0.1:{server.server_port}/v1/chat/completions'))
        with pytest.raises(ProducerUnavailableError) as exc:
            _produce(producer)
        assert exc.value.code == 'response_schema_invalid'
        assert DUMMY_KEY not in json.dumps(producer.redacted_trace())
        assert DUMMY_KEY not in json.dumps(producer.redacted_config())
        assert os.environ == before
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_default_rpp_and_lisn_social_mpc_boundary_is_unchanged():
    from arena_vln_models.tracking_config import tracking_parameters

    robot = {
        'robot_base_frame': 'base_footprint',
        'robot_odom_frame': 'odom',
        'max_vel_x': 0.4,
        'max_vel_theta': 1.5,
        'velocity_smoother_max_velocity': [0.8, 0.0, 1.5],
        'velocity_smoother_min_velocity': [0.0, 0.0, -1.5],
        'velocity_smoother_max_accel': [2.5, 0.0, 3.2],
        'velocity_smoother_max_decel': [-2.5, 0.0, -3.2],
        'footprint': '[[0.42, 0.48], [0.42, -0.48], [-0.42, -0.48], [-0.42, 0.48]]',
    }
    rpp = tracking_parameters(robot, 'Ai2_Bot2/')['controller_server']['ros__parameters']
    mpc = tracking_parameters(robot, 'Ai2_Bot2/', 'social_mpc')['controller_server']['ros__parameters']
    assert rpp['FollowPath']['plugin'].endswith('RegulatedPurePursuitController')
    assert mpc['FollowPath']['primary_controller'].endswith('SocialMPCController')


def test_error_code_allowlist_and_status_normalization():
    assert ProducerUnavailableError('DUMMY_SECRET').code == 'unknown'
    assert str(ProducerUnavailableError('DUMMY_SECRET')) == 'unknown'
    accepted = ProducerUnavailableError('auth_denied', status='401')
    assert accepted.code == 'auth_denied'
    assert accepted.status == 401
    assert ProducerUnavailableError('auth_denied', status=999).status is None
    assert ProducerUnavailableError('auth_denied', status=True).status is None


def test_node_ignores_third_party_code_attribute():
    node = _node_for_auth_test()

    class Weird(Exception):
        code = 'DUMMY_SECRET_IN_CODE'

    node._handle_producer_failure(Weird())
    joined = ' '.join(node.node.logger.errors)
    assert 'DUMMY_SECRET_IN_CODE' not in joined
    assert 'code=unknown' in joined
    assert node._auth_gate.latched is False


@pytest.mark.parametrize('status', [401, 403])
def test_node_future_then_multi_tick_latches_one_request(status):
    server, handler, thread = _server(status, b'{"error":"denied"}')
    executor = None
    try:
        node = _node_for_auth_test()
        producer = HttpVlmProducer(config=_config(
            f'http://127.0.0.1:{server.server_port}/v1/chat/completions'))
        node.producer = producer
        node.producer_tick = _FakeTick(producer)
        executor = ThreadPoolExecutor(max_workers=1)
        node._executor = executor
        node._commit_executor = None
        node._image = _FakeImage()
        node._instruction = 'walk'
        node._image_received_ns = 0
        node.runner = _FakeRunner()

        node._tick_producer(0)
        assert node._pending is not None
        with pytest.raises(ProducerUnavailableError):
            node._pending.result()
        node._tick_producer(0)
        for _ in range(4):
            node._tick_producer(0)

        assert producer.request_count == 1
        assert handler.requests == 1
        assert node._auth_gate.error_count == 1
        assert len(node.node.logger.errors) == 1
        assert len(node._status_pub.messages) == 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.mark.parametrize('provider', ['gemini', 'modelhub', 'ark', 'dmxapi'])
def test_named_provider_without_key_issues_zero_requests(provider):
    server, handler, thread = _server(401, b'{}')
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
        config = replace(
            _config(endpoint, api_key='', provider=provider),
            models_endpoint=f'http://127.0.0.1:{server.server_port}/v1/models',
        )
        producer = HttpVlmProducer(config=config)
        with pytest.raises(ProducerUnavailableError) as produce_exc:
            _produce(producer)
        assert produce_exc.value.code == 'credential_missing'
        with pytest.raises(ProducerUnavailableError) as list_exc:
            producer.list_models()
        assert list_exc.value.code == 'credential_missing'
        assert producer.request_count == 0
        assert handler.requests == 0
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_generic_without_key_requires_explicit_unauthenticated_policy():
    server, handler, thread = _server(401, b'{}')
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
        producer = HttpVlmProducer(config=_config(endpoint, api_key='', provider='generic'))
        with pytest.raises(ProducerUnavailableError) as exc:
            _produce(producer)
        assert exc.value.code == 'credential_missing'
        assert handler.requests == 0

        allowed = HttpVlmProducer(config=replace(
            _config(endpoint, api_key='', provider='generic'),
            allow_unauthenticated=True,
        ))
        with pytest.raises(ProducerUnavailableError) as auth_exc:
            _produce(allowed)
        assert auth_exc.value.code == 'auth_denied'
        assert handler.requests == 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_producer_timeout_from_env_and_cli():
    from arena_bringup.lisn_slow_loop_ros import build_parser

    config = ProducerConfig.from_env({'ARENA_LISN_VLM_TIMEOUT_SEC': '120'})
    assert config.timeout_sec == 120.0
    args = build_parser().parse_args(['--evidence', '/tmp/e.jsonl', '--producer-timeout-sec', '120'])
    assert args.producer_timeout_sec == 120.0


def test_node_timeout_latches_after_one_error():
    node = _node_for_auth_test()
    for _ in range(3):
        node._handle_producer_failure(ProducerUnavailableError('timeout'))
    assert node._auth_gate.latched is True
    assert node._auth_gate.error_count == 1
    assert node._auth_gate.last_code == 'timeout'
    assert len(node.node.logger.errors) == 1
    assert len(node._status_pub.messages) == 1


def test_delayed_http_timeout_issues_one_request():
    server, handler, thread = _server(200, b'{}', delay=0.5)
    try:
        producer = HttpVlmProducer(config=replace(
            _config(f'http://127.0.0.1:{server.server_port}/v1/chat/completions'),
            timeout_sec=0.1,
        ))
        with pytest.raises(ProducerUnavailableError) as exc:
            _produce(producer)
        assert exc.value.code == 'timeout'
        assert producer.request_count == 1
        assert handler.requests == 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_safe_error_category_is_static_only():
    from arena_bringup.lisn_slow_loop_ros import safe_error_category

    assert safe_error_category(ProducerUnavailableError('auth_denied')) == 'producer'
    assert safe_error_category(TimeoutError()) == 'timeout'
    assert safe_error_category(ValueError('DUMMY_SECRET')) == 'value_error'
    assert safe_error_category(RuntimeError('DUMMY_SECRET')) == 'runtime_error'
    assert safe_error_category(KeyError('DUMMY_SECRET')) == 'key_error'
    assert safe_error_category(OSError('DUMMY_SECRET')) == 'os_error'

    class Weird(Exception):
        code = 'DUMMY_SECRET'

    assert safe_error_category(Weird('DUMMY_SECRET')) == 'unknown'


def test_node_ignores_third_party_message_and_code():
    node = _node_for_auth_test()

    class Weird(Exception):
        code = 'DUMMY_SECRET_IN_CODE'

    node._handle_producer_failure(Weird('DUMMY_SECRET_IN_MESSAGE'))
    joined = ' '.join(node.node.logger.errors)
    assert 'DUMMY_SECRET_IN_CODE' not in joined
    assert 'DUMMY_SECRET_IN_MESSAGE' not in joined
    assert 'code=unknown' in joined
    assert 'category=unknown' in joined


def test_node_global_deadline_latches_and_ignores_late_result():
    from concurrent.futures import Future

    node = _node_for_auth_test()
    node._producer_deadline_ns = 1
    node._pending_started_ns = 0
    node._pending = Future()
    node._pending_meta = ('image', 'instruction', 'obs', 'map', 1, 1)
    node._tick_producer(10)
    assert node._auth_gate.latched is True
    assert node._auth_gate.last_code == 'deadline_exceeded'
    assert len(node._status_pub.messages) == 1
    assert len(node.node.logger.errors) == 1
    node._pending.set_result({'decision': 'accepted'})
    node._tick_producer(20)
    assert len(node._status_pub.messages) == 1
    assert len(node.node.logger.errors) == 1


@pytest.mark.parametrize('mode', ['headers_delay', 'body_delay'])
def test_delayed_headers_or_body_times_out_with_one_request(mode):
    kwargs = {'delay': 0.5} if mode == 'headers_delay' else {'body_delay': 0.5}
    server, handler, thread = _server(200, b'{}', **kwargs)
    try:
        producer = HttpVlmProducer(config=replace(
            _config(f'http://127.0.0.1:{server.server_port}/v1/chat/completions'),
            timeout_sec=0.1,
        ))
        with pytest.raises(ProducerUnavailableError) as exc:
            _produce(producer)
        assert exc.value.code == 'timeout'
        assert producer.request_count == 1
        assert handler.requests == 1
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_producer_deadline_cli_is_parsed():
    from arena_bringup.lisn_slow_loop_ros import build_parser

    args = build_parser().parse_args(['--evidence', '/tmp/e.jsonl', '--producer-deadline-sec', '120'])
    assert args.producer_deadline_sec == 120.0


class _StreamHandler(BaseHTTPRequestHandler):
    status = 200
    requests = 0
    body = b'{}'
    delay = 0.0
    body_delay = 0.0
    stream_chunks = 0
    stream_interval = 0.05

    def do_POST(self):  # noqa: N802
        import time
        type(self).requests += 1
        if type(self).delay:
            time.sleep(type(self).delay)
        self.send_response(type(self).status)
        self.send_header('Content-Type', 'application/json')
        length = type(self).stream_chunks if type(self).stream_chunks else len(type(self).body)
        self.send_header('Content-Length', str(length))
        self.end_headers()
        if type(self).stream_chunks:
            for _ in range(type(self).stream_chunks):
                try:
                    self.wfile.write(b' ')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(type(self).stream_interval)
            return
        if type(self).body_delay:
            time.sleep(type(self).body_delay)
        try:
            self.wfile.write(type(self).body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        return


def _threaded_server(status, body, **kwargs):
    handler = type('Handler', (_StreamHandler,), {
        'status': status, 'requests': 0, 'body': body, **kwargs,
    })
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler, thread


def _subprocess_config(endpoint, deadline_sec):
    return replace(_config(endpoint), deadline_sec=deadline_sec)


@pytest.mark.parametrize('mode', ['headers_delay', 'streaming_body'])
def test_subprocess_deadline_aborts_owned_child(mode):
    kwargs = (
        {'delay': 3.0} if mode == 'headers_delay'
        else {'stream_chunks': 60, 'stream_interval': 0.05}
    )
    server, handler, thread = _threaded_server(200, b'{}', **kwargs)
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
        producer = SubprocessHttpVlmProducer(config=_subprocess_config(endpoint, 1.5))
        started = time.monotonic()
        with pytest.raises(ProducerUnavailableError) as exc:
            _produce(producer)
        elapsed = time.monotonic() - started
        assert exc.value.code == 'deadline_exceeded'
        assert producer.request_count == 1
        assert handler.requests == 1
        assert producer.last_worker_alive is False
        assert elapsed < 4.5, elapsed
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_subprocess_200_returns_proposal_and_meta():
    body = json.dumps({
        'model': 'dummy-model',
        'choices': [{'message': {'content': json.dumps(PROPOSAL)}, 'finish_reason': 'stop'}],
    }).encode()
    server, handler, thread = _threaded_server(200, body)
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
        producer = SubprocessHttpVlmProducer(config=_subprocess_config(endpoint, 3.0))
        result = _produce(producer)
        assert result['controller_parameters']['max_linear_velocity_mps'] == 0.3
        assert producer.request_count == 1
        assert producer.auth_latched is False
        assert producer.last_worker_alive is False
        assert producer.last_response_meta['finish_reason'] == 'stop'
        assert producer.last_response_meta['response_model'] == 'dummy-model'
        assert handler.requests == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_subprocess_401_latches_and_does_not_reenter():
    server, handler, thread = _threaded_server(401, b'{"error":"denied"}')
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
        producer = SubprocessHttpVlmProducer(config=_subprocess_config(endpoint, 3.0))
        with pytest.raises(ProducerUnavailableError) as exc:
            _produce(producer)
        assert exc.value.code == 'auth_denied'
        assert producer.auth_latched is True
        assert producer.request_count == 1
        with pytest.raises(ProducerUnavailableError) as later:
            _produce(producer)
        assert later.value.code == 'auth_latched'
        assert producer.request_count == 1
        assert handler.requests == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_subprocess_rejects_non_positive_deadline():
    config = replace(_config('http://127.0.0.1:9/v1/chat/completions'), deadline_sec=0.0)
    with pytest.raises(ProducerUnavailableError) as exc:
        SubprocessHttpVlmProducer(config=config)
    assert exc.value.code == 'invalid_deadline'


def test_bridge_cli_argv_includes_producer_deadline_flags():
    from arena_bringup.lisn_slow_loop_ros import bridge_cli_argv

    argv = bridge_cli_argv(
        policy='p', evidence='e', controller_node='c', proposal_topic='t',
        target_frame='map', episode_id='id', producer_endpoint='http://127.0.0.1:9/x',
        rgb_topic='rgb', instruction_topic='i', wiring_live='false',
        producer_timeout_sec='125', producer_deadline_sec='120',
        model_status_topic='s', eval_ready_topic='r', status_heartbeat_sec='1.0',
    )
    assert '--producer-timeout-sec' in argv
    assert argv[argv.index('--producer-timeout-sec') + 1] == '125'
    assert '--producer-deadline-sec' in argv
    assert argv[argv.index('--producer-deadline-sec') + 1] == '120'
    assert '--model-status-topic' in argv
    assert '--eval-ready-topic' in argv


def test_arena_launch_declares_lisn_producer_deadline_args():
    import importlib.util
    import os
    from launch.actions import DeclareLaunchArgument

    prefix = '/opt/arena_ws/target_install:/opt/arena_ws/install'
    os.environ['AMENT_PREFIX_PATH'] = prefix + ':' + os.environ.get('AMENT_PREFIX_PATH', '')
    path = Path(__file__).resolve().parents[2] / 'arena_bringup/launch/arena.launch.py'
    spec = importlib.util.spec_from_file_location('arena_launch_parse_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    names = {entity.name for entity in description.entities if isinstance(entity, DeclareLaunchArgument)}
    assert 'lisn_slow_loop_producer_timeout_sec' in names
    assert 'lisn_slow_loop_producer_deadline_sec' in names




def _supervisor(code):
    return [sys.executable, '-c', code]


def test_supervisor_partial_stdout_times_out_and_reaps_group():
    producer = SubprocessHttpVlmProducer(
        config=_subprocess_config('http://127.0.0.1:9/v1/chat/completions', 1.5),
        supervisor_cmd=_supervisor(
            "import sys,time; sys.stdout.write('{\"ok\":'); sys.stdout.flush(); time.sleep(30)"
        ),
    )
    started = time.monotonic()
    with pytest.raises(ProducerUnavailableError) as exc:
        _produce(producer)
    elapsed = time.monotonic() - started
    assert exc.value.code == 'deadline_exceeded'
    assert producer.request_count == 1
    assert producer.last_worker_alive is False
    assert producer._terminal_latched is True
    assert elapsed < 4.5, elapsed


def test_supervisor_slow_bootstrap_times_out_and_reaps_group():
    producer = SubprocessHttpVlmProducer(
        config=_subprocess_config('http://127.0.0.1:9/v1/chat/completions', 0.5),
        supervisor_cmd=_supervisor(
            'import time; time.sleep(2.0); '
            'from arena_bringup.lisn_producer_supervisor import main; main()'
        ),
    )
    started = time.monotonic()
    with pytest.raises(ProducerUnavailableError) as exc:
        _produce(producer)
    elapsed = time.monotonic() - started
    assert exc.value.code == 'deadline_exceeded'
    assert producer.request_count == 1
    assert producer.last_worker_alive is False
    assert producer._terminal_latched is True
    assert elapsed < 4.5, elapsed


def test_supervisor_slow_decode_times_out_and_reaps_group():
    body = json.dumps({
        'model': 'dummy-model',
        'choices': [{'message': {'content': json.dumps(PROPOSAL)}, 'finish_reason': 'stop'}],
    }).encode()
    server, handler, thread = _threaded_server(200, body)
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
        producer = SubprocessHttpVlmProducer(
            config=_subprocess_config(endpoint, 1.5),
            supervisor_cmd=_supervisor(
                'import time; import arena_bringup.lisn_slow_loop_ros as m; '
                'real=m.json.loads; '
                "m.json.loads=lambda data,*a,**k: (time.sleep(3.0) if 'choices' in str(data)[:500] else None, real(data,*a,**k))[1]; "
                'from arena_bringup.lisn_producer_supervisor import main; main()'
            ),
        )
        started = time.monotonic()
        with pytest.raises(ProducerUnavailableError) as exc:
            _produce(producer)
        elapsed = time.monotonic() - started
        assert exc.value.code == 'deadline_exceeded'
        assert producer.request_count == 1
        assert handler.requests == 1
        assert producer.last_worker_alive is False
        assert producer._terminal_latched is True
        assert elapsed < 4.5, elapsed
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_supervisor_oversize_stdout_fails_closed():
    producer = SubprocessHttpVlmProducer(
        config=_subprocess_config('http://127.0.0.1:9/v1/chat/completions', 5.0),
        supervisor_cmd=_supervisor(
            "import sys; sys.stdout.write('x'*(5*1024*1024)); sys.stdout.flush()"
        ),
    )
    with pytest.raises(ProducerUnavailableError) as exc:
        _produce(producer)
    assert exc.value.code == 'oversize_result'
    assert producer.last_worker_alive is False


def test_supervisor_pickle_stdout_is_not_unpickled():
    producer = SubprocessHttpVlmProducer(
        config=_subprocess_config('http://127.0.0.1:9/v1/chat/completions', 5.0),
        supervisor_cmd=_supervisor(
            "import sys; sys.stdout.buffer.write(b\"cos\\nsystem\\n(S'echo DUMMY_SECRET'\\ntR.\"); sys.stdout.buffer.flush()"
        ),
    )
    with pytest.raises(ProducerUnavailableError) as exc:
        _produce(producer)
    assert exc.value.code == 'response_not_json'
    assert producer._terminal_latched is True


def test_node_tick_is_non_blocking_while_the_producer_is_slow():
    from concurrent.futures import ThreadPoolExecutor

    node = _node_for_auth_test()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        node._executor = executor
        node._commit_executor = None
        node._image = _FakeImage()
        node._instruction = 'walk'
        node._image_received_ns = 0
        node.runner = _FakeRunner()
        node.producer = type('P', (), {
            'auth_latched': False, 'request_count': 0,
        })()

        class SlowTick:
            def produce(self, **kwargs):
                time.sleep(2.0)
                return dict(PROPOSAL)

        node.producer_tick = SlowTick()
        started = time.monotonic()
        node._tick_producer(0)
        elapsed = time.monotonic() - started
        assert elapsed < 0.2, elapsed
        assert node._pending is not None
    finally:
        executor.shutdown(wait=True)
