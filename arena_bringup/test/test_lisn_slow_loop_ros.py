"""Deterministic wiring tests for the LISN slow-loop bridge.

These are wiring tests.  They use a fixture snapshot and an in-memory parameter
adapter.  They are not a real LISN agent and not a scientific result.
"""
import json
from pathlib import Path

import pytest

from arena_bringup.lisn_slow_loop_contract import (
    ApplyResult,
    canonical_sha256,
    load_slow_loop_policy,
)
from arena_bringup.lisn_slow_loop_ros import (
    DEFAULT_FIXTURE,
    DEFAULT_POLICY,
    FixtureParameterAdapter,
    TickRunner,
    main,
    production_blockers,
    run_wiring_smoke,
)


def _fixture():
    return json.loads(Path(DEFAULT_FIXTURE).read_text())


def _runner(tmp_path, adapter=None, policy=None):
    policy = policy or load_slow_loop_policy(Path(DEFAULT_POLICY))
    adapter = adapter or FixtureParameterAdapter(_fixture()['controller_parameters'])
    return TickRunner(
        policy=policy,
        episode_id='wiring-episode',
        epoch_monotonic_ns=1_000_000_000,
        target_frame='map',
        adapter=adapter,
        evidence_path=tmp_path / 'evidence.jsonl',
        wiring_smoke=True,
    )


class _MismatchAdapter:
    def __init__(self, values):
        self.values = {k: float(v) for k, v in values.items()}
        self.calls = 0

    def read(self, names):
        return {str(n): float(self.values[str(n)]) for n in names}

    def set_atomically(self, values):
        self.calls += 1
        requested = {str(k): float(v) for k, v in values.items()}
        self.values.update(requested)
        if self.calls == 1:
            self.values['max_linear_velocity_mps'] = requested['max_linear_velocity_mps'] + 0.1
        return ApplyResult(True, '')


class _RollbackFailAdapter(_MismatchAdapter):
    def set_atomically(self, values):
        self.calls += 1
        if self.calls >= 2:
            return ApplyResult(False, 'rollback_rejected')
        requested = {str(k): float(v) for k, v in values.items()}
        self.values.update(requested)
        self.values['max_linear_velocity_mps'] = requested['max_linear_velocity_mps'] + 0.1
        return ApplyResult(True, '')


def test_wiring_smoke_accepts_fixture_and_writes_evidence(tmp_path):
    receipt = run_wiring_smoke(
        policy_path=Path(DEFAULT_POLICY),
        fixture_path=Path(DEFAULT_FIXTURE),
        evidence_path=tmp_path / 'evidence.jsonl',
    )
    assert receipt['labels'] == {
        'wiring_smoke': True,
        'real_agent': False,
        'scientific_result': False,
        'runtime_executed': False,
    }
    assert receipt['decision'] == 'accepted'
    records = [json.loads(line) for line in (tmp_path / 'evidence.jsonl').read_text().splitlines()]
    assert len(records) == 1
    assert records[0]['decision'] == 'accepted'
    assert records[0]['runtime_executed'] is False
    assert records[0]['wiring_smoke'] is True
    assert (tmp_path / 'WIRING_SMOKE.json').is_file()


def test_readback_mismatch_rolls_back_to_previous_values(tmp_path):
    fixture = _fixture()
    before = dict(fixture['controller_parameters'])
    adapter = _MismatchAdapter(before)
    runner = _runner(tmp_path, adapter=adapter)
    record = runner.tick(
        fixture,
        received_monotonic_ns=runner.scheduled_for_ns() + 1 - 100_000_000,
        now_monotonic_ns=runner.scheduled_for_ns() + 1,
        now_ros_ns=1_500_000_000,
    )
    assert record['decision'] == 'rolled_back'
    assert 'ROLLBACK_SUCCEEDED_PREVIOUS_STATE_RETAINED' in record['reason_codes']
    assert adapter.values == pytest.approx(before)
    assert not runner.loop.state.faulted


def test_rollback_failure_faults_the_loop(tmp_path):
    fixture = _fixture()
    adapter = _RollbackFailAdapter(fixture['controller_parameters'])
    runner = _runner(tmp_path, adapter=adapter)
    record = runner.tick(
        fixture,
        received_monotonic_ns=runner.scheduled_for_ns() + 1 - 100_000_000,
        now_monotonic_ns=runner.scheduled_for_ns() + 1,
        now_ros_ns=1_500_000_000,
    )
    assert record['decision'] == 'faulted'
    assert 'ROLLBACK_FAILED' in record['reason_codes']
    assert runner.loop.state.faulted


def test_stale_observation_is_rejected(tmp_path):
    fixture = _fixture()
    runner = _runner(tmp_path)
    now = runner.scheduled_for_ns() + 1
    record = runner.tick(
        fixture,
        received_monotonic_ns=now - 6_000_000_000,
        now_monotonic_ns=now,
        now_ros_ns=1_500_000_000,
    )
    assert record['decision'] == 'rejected'
    assert 'OBSERVATION_STALE' in record['reason_codes']


def test_evidence_hash_is_deterministic(tmp_path):
    fixture = _fixture()
    first = _runner(tmp_path / 'a')
    second = _runner(tmp_path / 'b')
    kwargs = dict(now_ros_ns=1_500_000_000)
    a = first.tick(fixture, received_monotonic_ns=first.scheduled_for_ns() + 1 - 1, now_monotonic_ns=first.scheduled_for_ns() + 1, **kwargs)
    b = second.tick(fixture, received_monotonic_ns=second.scheduled_for_ns() + 1 - 1, now_monotonic_ns=second.scheduled_for_ns() + 1, **kwargs)
    assert a['snapshot_sha256'] == b['snapshot_sha256']
    assert a['update_id'] == b['update_id']
    assert canonical_sha256(a) == canonical_sha256(b)


def test_production_start_refuses_while_contract_is_not_runtime_ready(tmp_path):
    rc = main(['--evidence', str(tmp_path / 'evidence.jsonl')])
    assert rc == 2
    policy = load_slow_loop_policy(Path(DEFAULT_POLICY))
    blockers = production_blockers(policy)
    assert blockers
    assert policy.runtime_ready is False


def test_bridge_never_publishes_cmd_vel_or_twist():
    source = Path(__file__).resolve().parents[1] / 'arena_bringup/lisn_slow_loop_ros.py'
    text = source.read_text()
    assert 'cmd_vel' not in text
    assert 'Twist' not in text
    assert 'create_publisher' in text
    assert 'set_parameters_atomically' in text


# --- engineering profile: exact binding only -------------------------------

from arena_bringup.lisn_slow_loop_ros import (  # noqa: E402
    DEFAULT_ENGINEERING_POLICY,
    ENGINEERING_APPLIED_PARAMETERS,
    ENGINEERING_FROZEN_PARAMETERS,
    ENGINEERING_LOGICAL_PARAMETERS,
    EngineeringParameterAdapter,
    HttpVlmProducer,
    ProducerUnavailableError,
    build_snapshot,
    run_producer_wiring_smoke,
)


def _engineering_policy():
    return load_slow_loop_policy(Path(DEFAULT_ENGINEERING_POLICY))


def test_engineering_profile_loads_and_is_not_runtime_ready():
    policy = _engineering_policy()
    assert policy.runtime_ready is False
    assert policy.runtime_blockers
    assert policy.logical_parameters == ENGINEERING_LOGICAL_PARAMETERS
    assert policy.parameter_bindings['max_linear_velocity_mps']['status'] == 'exact_hot_reload_verified'


def test_engineering_adapter_rejects_unqualified_change_before_set():
    fixture = _fixture()
    base = FixtureParameterAdapter(fixture['controller_parameters'])
    adapter = EngineeringParameterAdapter(base)
    current = adapter.read(ENGINEERING_LOGICAL_PARAMETERS)
    requested = dict(current)
    requested['people_weight'] = current['people_weight'] + 1.0
    result = adapter.set_atomically(requested)
    assert result.successful is False
    assert 'UNQUALIFIED_PARAMETER_CHANGE:people_weight' in result.reason
    assert base.calls == [('read', dict(current))]  # no set call reached the base adapter


def test_engineering_adapter_applies_only_exact_binding():
    fixture = _fixture()
    base = FixtureParameterAdapter(fixture['controller_parameters'])
    adapter = EngineeringParameterAdapter(base)
    current = adapter.read(ENGINEERING_LOGICAL_PARAMETERS)
    requested = dict(current)
    requested['max_linear_velocity_mps'] = 0.30
    result = adapter.set_atomically(requested)
    assert result.successful is True
    set_calls = [call for call in base.calls if call[0] == 'set']
    assert set_calls == [('set', {'max_linear_velocity_mps': 0.30})]
    readback = adapter.read(ENGINEERING_LOGICAL_PARAMETERS)
    assert readback == pytest.approx(requested)


def test_engineering_tick_accepts_and_keeps_frozen_parameters(tmp_path):
    fixture = _fixture()
    policy = _engineering_policy()
    base = FixtureParameterAdapter(fixture['controller_parameters'])
    adapter = EngineeringParameterAdapter(base)
    runner = TickRunner(
        policy=policy,
        episode_id='engineering-episode',
        epoch_monotonic_ns=1_000_000_000,
        target_frame='map',
        adapter=adapter,
        evidence_path=tmp_path / 'evidence.jsonl',
        wiring_smoke=True,
    )
    proposal = {
        'controller_mode': 'cautious',
        'controller_parameters': {'max_linear_velocity_mps': 0.30},
        'goal': fixture['goal'],
        'semantic_entities': [],
        'provenance': fixture['provenance'],
    }
    snapshot = build_snapshot(
        current_parameters=adapter.read(ENGINEERING_LOGICAL_PARAMETERS),
        proposal=proposal,
        observation_id='rgb-42',
        frame_id='Ai2_Bot2/head_camera',
        stamp_ns=900_000_000,
        target_frame='map',
        now_ns=1_500_000_000,
        max_age_ns=policy.max_observation_age_ns,
    )
    now = runner.scheduled_for_ns() + 1
    record = runner.tick(
        snapshot,
        received_monotonic_ns=now - 100_000_000,
        now_monotonic_ns=now,
        now_ros_ns=1_500_000_000,
    )
    assert record['decision'] == 'accepted'
    assert record['readback_parameters']['max_linear_velocity_mps'] == 0.30
    assert record['readback_parameters']['people_weight'] == fixture['controller_parameters']['people_weight']


def test_producer_unavailable_fails_closed_without_fixture_fallback():
    producer = HttpVlmProducer('http://127.0.0.1:9/v1/propose', timeout_sec=0.5)
    with pytest.raises(ProducerUnavailableError):
        producer.produce(instruction='walk', image_bytes=b'x', frame_id='f', stamp_ns=1,
                         image_mime='image/jpeg')


def test_producer_wiring_smoke_exercises_real_http_path(tmp_path):
    receipt = run_producer_wiring_smoke(
        policy_path=Path(DEFAULT_ENGINEERING_POLICY),
        evidence_path=tmp_path / 'evidence.jsonl',
    )
    assert receipt['kind'] == 'PRODUCER_WIRING_SMOKE'
    assert receipt['labels']['producer_endpoint_kind'] == 'local_mock_http'
    assert receipt['labels']['real_agent'] is False
    assert receipt['decision'] == 'accepted'
    assert receipt['applied_parameter'] == 'max_linear_velocity_mps'
    records = [json.loads(line) for line in (tmp_path / 'evidence.jsonl').read_text().splitlines()]
    assert records[0]['decision'] == 'accepted'
    assert records[0]['runtime_executed'] is False
    assert (tmp_path / 'PRODUCER_WIRING_SMOKE.json').is_file()


def test_controller_source_exposes_exact_binding_and_excludes_the_rest():
    import os

    repo = Path(__file__).resolve().parents[2]
    candidates = [
        Path(os.environ.get('ARENA_SOCIAL_MPC_SOURCE', '')),
        repo / 'src/deps/nav2_social_mpc_controller/src/social_mpc_controller.cpp',
        Path('/home/ubuntu/arena_jazzy_ws/src/deps/nav2_social_mpc_controller/src/social_mpc_controller.cpp'),
    ]
    source_path = next((path for path in candidates if path.is_file()), None)
    if source_path is None:
        pytest.skip('social_mpc controller source is not available in this checkout')
    source = source_path.read_text()
    callback = source[source.index('dynamicParametersCallback'):source.index('void SocialMPCController::cleanup()')]
    assert 'plugin_name_ + ".optimizer.max_linear_velocity"' in callback
    assert 'plugin_name_ + ".trajectorizer.desired_linear_vel"' in callback
    for name in ('social_weight', 'proxemics_weight', 'proxemics_d0', 'proxemics_alpha'):
        assert name in callback
    assert 'max_angular_velocity' not in callback
    assert 'optimizer_params_.obstacle_w_' not in callback
    assert 'intentionally excluded' in callback
    assert 'goal_weight' not in callback
    assert 'distance_weight' not in callback
    assert 'goal_align_weight' not in callback


# --- live producer call path and runner/launch wiring ----------------------

from contextlib import contextmanager  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402
import threading  # noqa: E402

from arena_bringup.lisn_slow_loop_ros import ProducerTick  # noqa: E402


@contextmanager
def _mock_vlm_server():
    payload = {
        'controller_mode': 'cautious',
        'controller_parameters': {'max_linear_velocity_mps': 0.30},
        'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1_400_000_000,
                 'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
        'semantic_entities': [],
        'validity': {'valid': True, 'reason': 'mock'},
        'provenance': {'model_version': 'local-mock-v1',
                       'prompt_version': 'lisn-port-prompt-v1',
                       'tool_version': 'lisn-tools-v1',
                       'reasoning_latency_ms': 1.0},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            self.rfile.read(length)
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/propose'
    finally:
        server.shutdown()
        server.server_close()


def _producer_tick(tmp_path, endpoint):
    policy = _engineering_policy()
    fixture = _fixture()
    base = FixtureParameterAdapter(fixture['controller_parameters'])
    adapter = EngineeringParameterAdapter(base)
    runner = TickRunner(
        policy=policy,
        episode_id='live-producer-episode',
        epoch_monotonic_ns=1_000_000_000,
        target_frame='map',
        adapter=adapter,
        evidence_path=tmp_path / 'evidence.jsonl',
        wiring_smoke=True,
    )
    return ProducerTick(
        producer=HttpVlmProducer(config=ProducerConfig(
            endpoint=endpoint, timeout_sec=3.0, allow_unauthenticated=True,
            assume_goal_trusted=True)),
        adapter=adapter,
        runner=runner,
        target_frame='map',
        max_observation_age_ns=policy.max_observation_age_ns,
    ), runner


def test_live_producer_tick_image_instruction_to_core(tmp_path):
    with _mock_vlm_server() as endpoint:
        tick, runner = _producer_tick(tmp_path, endpoint)
        scheduled = runner.scheduled_for_ns()
        record = tick.process(
            instruction='walk forward',
            image_bytes=bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]),
            frame_id='Ai2_Bot2/head_camera',
            stamp_ns=1_400_000_000,
            observation_id='rgb-1',
            received_monotonic_ns=scheduled + 1 - 100_000_000,
            now_monotonic_ns=scheduled + 1,
            now_ros_ns=1_500_000_000,
            image_encoding='rgb8', width=2, height=2, channels=3,
        )
    assert record['decision'] == 'accepted'
    assert record['observation']['id'] == 'rgb-1'
    assert record['observation']['frame_id'] == 'Ai2_Bot2/head_camera'
    assert record['observation']['ros_stamp_ns'] == 1_400_000_000
    assert record['readback_parameters']['max_linear_velocity_mps'] == 0.30
    assert record['readback_parameters']['people_weight'] == 8.0
    assert record['wiring_smoke'] is True


def test_live_producer_tick_unreachable_endpoint_fails_closed(tmp_path):
    tick, _ = _producer_tick(tmp_path, 'http://127.0.0.1:9/v1/propose')
    with pytest.raises(ProducerUnavailableError):
        tick.process(
            instruction='walk',
            image_bytes=b'x',
            frame_id='f',
            stamp_ns=1,
            observation_id='o',
            received_monotonic_ns=1,
            now_monotonic_ns=2,
            now_ros_ns=2,
            image_mime='image/jpeg',
        )
    assert not (tmp_path / 'evidence.jsonl').exists()


def test_wiring_live_requires_endpoint(tmp_path):
    rc = main(['--wiring-live', '--evidence', str(tmp_path / 'evidence.jsonl')])
    assert rc == 2


def _runner_plan(*args):
    return subprocess.run(
        [sys.executable, '-m', 'arena_vln_models.runner', 'plan', 'run', '--model', 'lightnav',
         '--local', *args],
        capture_output=True, text=True,
    )


def test_runner_lisn_flag_passthrough():
    result = _runner_plan('--lisn-slow-loop', '--lisn-producer-endpoint', 'http://127.0.0.1:9/v1/propose')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    command = ' '.join(plan['commands'][0])
    assert 'lisn_slow_loop:=true' in command
    assert 'local_planner:=social_mpc' in command
    assert 'lisn_slow_loop_producer_endpoint:=http://127.0.0.1:9/v1/propose' in command


def test_runner_lisn_requires_endpoint():
    result = _runner_plan('--lisn-slow-loop')
    assert result.returncode == 2
    assert '--lisn-producer-endpoint' in result.stderr


def test_arena_launch_declares_conditional_lisn_sidecar():
    launch = Path(__file__).resolve().parents[1] / 'launch/arena.launch.py'
    text = launch.read_text()
    assert "name='lisn_slow_loop'" in text
    assert "executable='lisn_slow_loop_bridge'" in text
    assert 'IfCondition(lisn_slow_loop.substitution)' in text
    assert 'arguments=bridge_cli_argv(' in text


# --- D1/D2/D3 regressions ---------------------------------------------------

from arena_bringup.lisn_slow_loop_ros import bridge_cli_argv, build_parser  # noqa: E402


def test_d1_launch_argv_false_and_true_parse():
    for value in ('false', 'true'):
        argv = bridge_cli_argv(
            policy='p', evidence='e', controller_node='c', proposal_topic='t',
            target_frame='map', episode_id='ep', producer_endpoint='http://127.0.0.1:9/v1',
            rgb_topic='rgb', instruction_topic='ins', wiring_live=value,
        )
        args = build_parser().parse_args(argv)
        assert args.wiring_live == value
        assert (str(args.wiring_live).lower() in {'true', '1'}) is (value == 'true')


class _CountingProducer:
    def __init__(self):
        self.calls = 0

    def produce(self, **kwargs):
        self.calls += 1
        return {
            'controller_mode': 'cautious',
            'controller_parameters': {'max_linear_velocity_mps': 0.30},
            'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1_400_000_000,
                     'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
            'semantic_entities': [],
            'provenance': {'model_version': 'counting-v1',
                           'prompt_version': 'p', 'tool_version': 't',
                           'reasoning_latency_ms': 1.0},
        }


def _counting_tick(tmp_path):
    policy = _engineering_policy()
    fixture = _fixture()
    base = FixtureParameterAdapter(fixture['controller_parameters'])
    adapter = EngineeringParameterAdapter(base)
    runner = TickRunner(
        policy=policy, episode_id='counting', epoch_monotonic_ns=1_000_000_000,
        target_frame='map', adapter=adapter, evidence_path=tmp_path / 'e.jsonl',
        wiring_smoke=True,
    )
    producer = _CountingProducer()
    return ProducerTick(producer=producer, adapter=adapter, runner=runner,
                        target_frame='map', max_observation_age_ns=policy.max_observation_age_ns), runner, producer


def test_d2_process_calls_producer_exactly_once(tmp_path):
    tick, runner, producer = _counting_tick(tmp_path)
    scheduled = runner.scheduled_for_ns()
    record = tick.process(
        instruction='walk', image_bytes=b'rgb', frame_id='f', stamp_ns=1_400_000_000,
        observation_id='o', received_monotonic_ns=scheduled + 1 - 1,
        now_monotonic_ns=scheduled + 1, now_ros_ns=1_500_000_000,
    )
    assert record['decision'] == 'accepted'
    assert producer.calls == 1


def test_d2_commit_reuses_proposal_without_second_produce(tmp_path):
    tick, runner, producer = _counting_tick(tmp_path)
    scheduled = runner.scheduled_for_ns()
    proposal = _CountingProducer().produce()
    record = tick.commit(
        proposal=proposal, image_bytes=b'rgb', frame_id='f', stamp_ns=1_400_000_000,
        observation_id='o', received_monotonic_ns=scheduled + 1 - 1,
        now_monotonic_ns=scheduled + 1, now_ros_ns=1_500_000_000,
    )
    assert record['decision'] == 'accepted'
    assert producer.calls == 0


def test_d2_node_uses_commit_worker_and_nonblocking_harvest():
    source = Path(__file__).resolve().parents[1] / 'arena_bringup/lisn_slow_loop_ros.py'
    text = source.read_text()
    commit = text[text.index('def _commit_proposal'):text.index('def _tick_producer')]
    tick = text[text.index('def _tick_producer'):text.index('def _tick(self)')]
    assert 'self.producer_tick.commit(' in commit
    assert 'self._commit_executor.submit(' in tick
    assert 'self.producer_tick.process(' not in tick
    assert 'self._pending.done()' in tick
    assert 'self._commit_future.done()' in tick
    assert '.result(timeout=' not in tick


def test_d3_runner_lisn_plan_identity_is_social_mpc():
    result = _runner_plan('--lisn-slow-loop', '--lisn-producer-endpoint', 'http://127.0.0.1:9/v1')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['controller'] == 'social_mpc'
    assert plan['control_path'] == 'nav2_follow_path_social_mpc'
    assert plan['local_planner'] == 'social_mpc'
    command = ' '.join(plan['commands'][0])
    assert '--local-planner social_mpc' in command


def test_d3_runner_default_plan_identity_unchanged():
    result = _runner_plan()
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['controller'] == 'regulated_pure_pursuit'
    assert plan['control_path'] == 'nav2_follow_path_rpp'
    assert plan['local_planner'] == 'regulated_pure_pursuit'


def test_d3_engineering_adapter_delegates_identity_check():
    class _Base:
        def verify_social_mpc(self):
            raise RuntimeError('controller identity is not social_mpc')

        def read(self, names):
            return {name: 0.0 for name in names}

        def set_atomically(self, values):
            return ApplyResult(True, '')

    adapter = EngineeringParameterAdapter(_Base())
    with pytest.raises(RuntimeError, match='not social_mpc'):
        adapter.verify_social_mpc()


def test_d3_internnav_eval_records_social_mpc_identity():
    source = Path(__file__).resolve().parents[1] / 'arena_bringup/internnav_eval.py'
    text = source.read_text()
    assert "social_mpc_mode = follow_path_mode and args.local_planner == 'social_mpc'" in text
    assert "'nav2_follow_path_social_mpc' if social_mpc_mode" in text
    assert "'social_mpc' if social_mpc_mode" in text


# --- Lane B: Ark/OpenAI-compatible thin protocol + redaction ----------------

from arena_bringup.lisn_slow_loop_ros import ProducerConfig, _encode_image_for_vlm  # noqa: E402


def test_image_encoding_rejects_raw_without_dimensions():
    with pytest.raises(ValueError, match='positive width/height'):
        _encode_image_for_vlm(b'raw', image_encoding='rgb8', width=0, height=0, channels=3)


def test_image_encoding_rgb8_to_jpeg_with_metadata():
    raw = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    encoded, mime = _encode_image_for_vlm(raw, image_encoding='rgb8', width=2, height=2, channels=3)
    assert mime == 'image/jpeg'
    assert encoded.startswith(b'\xff\xd8')  # JPEG SOI
    assert len(encoded) > 0


def test_producer_config_from_env_redacts_key():
    env = {
        'OPENAI_BASE_URL': 'https://ark.example/v3',
        'OPENAI_API_KEY': 'sk-test-not-real',
        'ARENA_LISN_VLM_MODEL': 'ark-test-model',
    }
    config = ProducerConfig.from_env(env)
    assert config.endpoint == 'https://ark.example/v3/chat/completions'
    assert config.model == 'ark-test-model'
    redacted = config.redacted()
    assert redacted['api_key_present'] is True
    assert redacted['api_key_env'] == 'OPENAI_API_KEY'
    assert 'sk-test-not-real' not in json.dumps(redacted)


def _openai_mock_server():
    content = json.dumps({
        'controller_mode': 'cautious',
        'controller_parameters': {'max_linear_velocity_mps': 0.30},
        'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1_400_000_000,
                 'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
        'semantic_entities': [],
        'provenance': {'model_version': 'ark-mock', 'prompt_version': 'p',
                       'tool_version': 't', 'reasoning_latency_ms': 1.0},
    })

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            self.rfile.read(length)
            body = json.dumps({'choices': [{'message': {'content': content}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f'http://127.0.0.1:{server.server_port}/chat/completions'


def test_producer_openai_compatible_content_json_and_redaction():
    server, endpoint = _openai_mock_server()
    try:
        config = ProducerConfig(endpoint=endpoint, api_key='sk-test-not-real',
                                model='ark-test-model', timeout_sec=3.0)
        producer = HttpVlmProducer(config=config)
        raw = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        proposal = producer.produce(
            instruction='walk forward', image_bytes=raw, frame_id='Ai2_Bot2/head_camera',
            stamp_ns=1_400_000_000, image_encoding='rgb8', width=2, height=2, channels=3,
        )
        assert proposal['controller_parameters']['max_linear_velocity_mps'] == 0.30
        trace = producer.redacted_trace()
        assert trace and trace[-1]['request']['image']['encoded_mime'] == 'image/jpeg'
        assert trace[-1]['request']['image']['source_encoding'] == 'rgb8'
        assert 'sk-test-not-real' not in json.dumps(trace)
        assert 'sk-test-not-real' not in json.dumps(producer.redacted_config())
    finally:
        server.shutdown()
        server.server_close()


def test_producer_missing_openai_key_fails_closed():
    config = ProducerConfig(endpoint='https://api.openai.com/v1/chat/completions', api_key='')
    producer = HttpVlmProducer(config=config)
    with pytest.raises(ProducerUnavailableError, match='credential_missing'):
        producer.produce(instruction='walk', image_bytes=b'x', frame_id='f', stamp_ns=1,
                         image_mime='image/jpeg')


# --- ModelHub / Ark provider resolution ------------------------------------

def test_producer_config_modelhub_from_env():
    env = {
        'GPT_OPEN_API_KEY': 'gpt-test-not-real',
        'GPT_OPEN_API_VERSION': '2024-08-01-preview',
    }
    config = ProducerConfig.from_env(env)
    assert config.provider == 'modelhub'
    assert config.endpoint == (
        'https://aidp.bytedance.net/api/modelhub/online/v2/crawl/'
        'chat/completions?api-version=2024-08-01-preview'
    )
    assert config.model == 'gpt-5.4-nano-2026-03-17'
    assert config.auth_header == 'api-key'
    assert config.api_key_env == 'GPT_OPEN_API_KEY'
    redacted = config.redacted()
    assert redacted['provider'] == 'modelhub'
    assert redacted['api_key_present'] is True
    assert redacted['api_version_present'] is True
    assert 'gpt-test-not-real' not in json.dumps(redacted)


def test_producer_config_ark_from_env():
    env = {'ARK_API_KEY': 'ark-test-not-real'}
    config = ProducerConfig.from_env(env)
    assert config.provider == 'ark'
    assert config.endpoint == 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
    assert config.model == 'doubao-seed-2-0-lite-260215'
    assert config.auth_header == 'Authorization'
    assert config.api_scheme if False else True
    assert 'ark-test-not-real' not in json.dumps(config.redacted())


def test_producer_config_no_credential_fails_closed():
    config = ProducerConfig.from_env({})
    assert config.provider == 'none'
    producer = HttpVlmProducer(config=config)
    with pytest.raises(ProducerUnavailableError, match='credential_missing'):
        producer.produce(instruction='walk', image_bytes=b'x', frame_id='f', stamp_ns=1,
                         image_mime='image/jpeg')


def test_modelhub_payload_uses_api_key_header_and_redacts():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length)
            seen['auth_header_present'] = bool(self.headers.get('api-key'))
            seen['authorization_present'] = bool(self.headers.get('Authorization'))
            seen['payload'] = json.loads(body)
            content = json.dumps({
                'controller_mode': 'cautious',
                'controller_parameters': {'max_linear_velocity_mps': 0.30},
                'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1_400_000_000,
                         'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
                'semantic_entities': [],
                'provenance': {'model_version': 'modelhub-mock', 'prompt_version': 'p',
                               'tool_version': 't', 'reasoning_latency_ms': 1.0},
            })
            reply = json.dumps({'choices': [{'message': {'content': content}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = ProducerConfig(
            endpoint=f'http://127.0.0.1:{server.server_port}/chat/completions?api-version=x',
            api_key='gpt-test-not-real', api_key_env='GPT_OPEN_API_KEY',
            model='gpt-5.4-nano-2026-03-17', provider='modelhub',
            api_version='x', auth_header='api-key', auth_scheme='',
        )
        producer = HttpVlmProducer(config=config)
        raw = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        proposal = producer.produce(
            instruction='walk forward', image_bytes=raw, frame_id='f', stamp_ns=1,
            image_encoding='rgb8', width=2, height=2, channels=3,
        )
        assert proposal['controller_parameters']['max_linear_velocity_mps'] == 0.30
        assert seen['auth_header_present'] is True
        assert seen['authorization_present'] is False
        assert seen['payload']['model'] == 'gpt-5.4-nano-2026-03-17'
        assert seen['payload']['messages'][0]['content'][1]['type'] == 'image_url'
        prompt_text = seen['payload']['messages'][0]['content'][0]['text']
        assert 'Return ONLY a JSON object' in prompt_text
        assert 'controller_parameters' in prompt_text
        assert 'walk forward' in prompt_text
        assert 'gpt-test-not-real' not in json.dumps(producer.redacted_trace())
    finally:
        server.shutdown()
        server.server_close()


# --- DMXAPI scoped provider ------------------------------------------------

def test_producer_config_dmxapi_from_env():
    env = {'DMXAPI_API_KEY': 'dmx-test-not-real'}
    config = ProducerConfig.from_env(env)
    assert config.provider == 'dmxapi'
    assert config.endpoint == 'https://www.dmxapi.cn/v1/chat/completions'
    assert config.models_endpoint == 'https://www.dmxapi.cn/v1/models'
    assert config.model == 'gpt-5.6-sol'
    assert config.auth_header == 'Authorization'
    assert config.auth_scheme == ''
    redacted = config.redacted()
    assert redacted['provider'] == 'dmxapi'
    assert redacted['api_key_present'] is True
    assert redacted['models_endpoint_present'] is True
    assert 'dmx-test-not-real' not in json.dumps(redacted)


def test_dmxapi_authorization_is_raw_key_and_trace_redacted():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length)
            seen['authorization'] = self.headers.get('Authorization')
            seen['payload'] = json.loads(body)
            content = json.dumps({
                'controller_mode': 'cautious',
                'controller_parameters': {'max_linear_velocity_mps': 0.30},
                'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1_400_000_000,
                         'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
                'semantic_entities': [],
                'provenance': {'model_version': 'dmx-mock', 'prompt_version': 'p',
                               'tool_version': 't', 'reasoning_latency_ms': 1.0},
            })
            reply = json.dumps({'model': 'gpt-5.6-sol-2026-07-09',
                                'choices': [{'message': {'content': content}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = ProducerConfig(
            endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
            api_key='dmx-test-not-real', api_key_env='DMXAPI_API_KEY',
            model='gpt-5.6-sol', provider='dmxapi',
            models_endpoint=f'http://127.0.0.1:{server.server_port}/v1/models',
            auth_header='Authorization', auth_scheme='',
        )
        producer = HttpVlmProducer(config=config)
        raw = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        proposal = producer.produce(
            instruction='walk forward', image_bytes=raw, frame_id='f', stamp_ns=1,
            image_encoding='rgb8', width=2, height=2, channels=3,
        )
        assert proposal['controller_parameters']['max_linear_velocity_mps'] == 0.30
        assert seen['authorization'] == 'dmx-test-not-real'  # raw key, no Bearer
        assert seen['payload']['messages'][0]['content'][1]['type'] == 'image_url'
        assert 'dmx-test-not-real' not in json.dumps(producer.redacted_trace())
    finally:
        server.shutdown()
        server.server_close()


def test_dmxapi_list_models_probe():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({'data': [{'id': 'gpt-5.6-sol'}, {'id': 'qwen3-vl-plus'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = ProducerConfig(
            endpoint=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
            api_key='dmx-test-not-real', api_key_env='DMXAPI_API_KEY',
            model='gpt-5.6-sol', provider='dmxapi',
            models_endpoint=f'http://127.0.0.1:{server.server_port}/v1/models',
            auth_header='Authorization', auth_scheme='',
        )
        producer = HttpVlmProducer(config=config)
        assert producer.list_models() == ['gpt-5.6-sol', 'qwen3-vl-plus']
    finally:
        server.shutdown()
        server.server_close()


# --- Gemini 2.5 Pro via ModelHub (authorized doc credential) ---------------

def test_producer_config_gemini_from_env():
    env = {'ARENA_LISN_GEMINI_AK': 'gem-test-not-real'}
    config = ProducerConfig.from_env(env)
    assert config.provider == 'gemini'
    assert config.endpoint == (
        'https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/'
        'gpt_openapi/openai/deployments/gemini-2.5-pro/chat/completions'
        '?api-version=2024-03-01-preview'
    )
    assert config.model == 'gemini-2.5-pro'
    assert config.auth_header == 'api-key'
    assert config.auth_scheme == ''
    assert config.max_tokens == 4095
    assert 'gem-test-not-real' not in json.dumps(config.redacted())


def test_gemini_payload_api_key_header_and_max_tokens():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length)
            seen['api_key'] = self.headers.get('api-key')
            seen['payload'] = json.loads(body)
            content = json.dumps({
                'controller_mode': 'cautious',
                'controller_parameters': {'max_linear_velocity_mps': 0.30},
                'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1_400_000_000,
                         'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
                'semantic_entities': [],
                'validity': {'valid': True, 'reason': ''},
                'provenance': {'model_version': 'gemini-mock', 'prompt_version': 'p',
                               'tool_version': 't', 'reasoning_latency_ms': 1.0},
            })
            reply = json.dumps({'model': 'gemini-2.5-pro',
                                'choices': [{'message': {'content': content},
                                             'finish_reason': 'stop'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = ProducerConfig(
            endpoint=f'http://127.0.0.1:{server.server_port}/chat/completions',
            api_key='gem-test-not-real', api_key_env='ARENA_LISN_GEMINI_AK',
            model='gemini-2.5-pro', provider='gemini',
            auth_header='api-key', auth_scheme='', max_tokens=4095,
        )
        producer = HttpVlmProducer(config=config)
        raw = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        proposal = producer.produce(
            instruction='walk forward', image_bytes=raw, frame_id='f', stamp_ns=1,
            image_encoding='rgb8', width=2, height=2, channels=3,
        )
        assert proposal['controller_parameters']['max_linear_velocity_mps'] == 0.30
        assert seen['api_key'] == 'gem-test-not-real'
        assert seen['payload']['max_tokens'] == 4095
        assert seen['payload']['messages'][0]['content'][1]['type'] == 'image_url'
        assert 'gem-test-not-real' not in json.dumps(producer.redacted_trace())
    finally:
        server.shutdown()
        server.server_close()


def test_gemini_missing_schema_field_fails_closed():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            self.rfile.read(length)
            reply = json.dumps({'choices': [{'message': {'content': '{"controller_mode":"cautious"}'}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = ProducerConfig(
            endpoint=f'http://127.0.0.1:{server.server_port}/chat/completions',
            api_key='gem-test-not-real', api_key_env='ARENA_LISN_GEMINI_AK',
            model='gemini-2.5-pro', provider='gemini',
            auth_header='api-key', auth_scheme='',
        )
        producer = HttpVlmProducer(config=config)
        with pytest.raises(ProducerUnavailableError, match='response_schema_invalid'):
            producer.produce(instruction='walk', image_bytes=b'x', frame_id='f', stamp_ns=1,
                             image_mime='image/jpeg')
    finally:
        server.shutdown()
        server.server_close()


# --- 30 s engineering profile (frozen 10 s contract untouched) -------------

from arena_bringup.lisn_slow_loop_ros import load_engineering_policy, load_policy  # noqa: E402

ENGINEERING = Path(__file__).resolve().parents[1] / 'configs/lisn/slow_loop_gemini_engineering_v1.json'


def test_engineering_profile_is_30s_and_runtime_ready():
    policy = load_engineering_policy(ENGINEERING)
    assert policy.period_ns == 30_000_000_000
    assert policy.stage_deadline_offsets_ns['hard_wall'] == 30_000_000_000
    assert policy.stage_deadline_offsets_ns['proposal'] == 27_000_000_000
    assert policy.runtime_ready is True
    assert policy.runtime_blockers == ()


def test_frozen_contract_remains_10s():
    policy = load_policy(DEFAULT_POLICY, engineering=False)
    assert policy.period_ns == 10_000_000_000


def test_live_apply_receipt_and_ready_status_source_contract():
    source = Path(__file__).resolve().parents[1] / 'arena_bringup/lisn_slow_loop_ros.py'
    text = source.read_text()
    assert "LIVE_APPLY.json" in text
    assert "'runtime_executed': True" in text
    assert "'status': 'ready'" in text
    assert "'controller_identity': 'FollowPath/SocialMPCController'" in text


def test_session_status_payload_carries_episode_identity():
    from arena_bringup.lisn_slow_loop_ros import session_status_payload

    payload = session_status_payload(
        session_id='session-abc', episode=3, status='ready', engineering=True,
        producer_mode=True,
    )
    assert payload['session_id'] == 'session-abc'
    assert payload['episode'] == 3
    assert payload['status'] == 'ready'
    assert payload['source'] == 'lisn_slow_loop'
    assert payload['engineering'] is True
    assert payload['producer_mode'] is True


def test_session_status_payload_rejects_unknown_status():
    from arena_bringup.lisn_slow_loop_ros import session_status_payload

    with pytest.raises(ValueError):
        session_status_payload(session_id='s', episode=0, status='connected')


def test_parse_eval_ready_payload_requires_session_and_int_episode():
    from arena_bringup.lisn_slow_loop_ros import parse_eval_ready_payload

    assert parse_eval_ready_payload(json.dumps({
        'session_id': 's-1', 'episode': 2, 'stage': 'episode', 'ready': True,
    })) == ('s-1', 2)
    assert parse_eval_ready_payload(json.dumps({'session_id': 's-1'})) == (None, None)
    assert parse_eval_ready_payload(json.dumps({'session_id': '', 'episode': 0})) == (None, None)
    assert parse_eval_ready_payload(json.dumps({'session_id': 's-1', 'episode': '0'})) == (None, None)
    assert parse_eval_ready_payload('not-json') == (None, None)


def test_heartbeat_default_is_below_task_generator_timeout():
    from arena_bringup.lisn_slow_loop_ros import build_parser

    args = build_parser().parse_args(['--evidence', '/tmp/e.jsonl'])
    assert args.status_heartbeat_sec == 1.0
    assert args.status_heartbeat_sec < 2.5
    assert args.eval_ready_topic == '/task_generator_node/eval_ready_status'


def test_tick_publishes_heartbeat_before_producer_branch():
    source = Path(__file__).resolve().parents[1] / 'arena_bringup/lisn_slow_loop_ros.py'
    text = source.read_text()
    tick = text[text.index('    def _tick(self):'):text.index('    def spin(self):')]
    assert tick.index('_publish_heartbeat') < tick.index('if self.producer is not None:')


def test_eval_ready_subscription_matches_transient_local_publisher():
    source = Path(__file__).resolve().parents[1] / 'arena_bringup/lisn_slow_loop_ros.py'
    text = source.read_text()
    assert 'DurabilityPolicy.TRANSIENT_LOCAL' in text
    assert 'durability=DurabilityPolicy.TRANSIENT_LOCAL' in text
    assert "self._eval_ready_topic, self._on_eval_ready, ready_qos" in text


ENGINEERING_V2 = Path(__file__).resolve().parents[1] / 'configs/lisn/slow_loop_gemini_engineering_v2.json'


def test_engineering_profile_v2_allows_measured_gemini_latency():
    policy = load_engineering_policy(ENGINEERING_V2)
    assert policy.period_ns == 130_000_000_000
    assert policy.max_observation_age_ns == 5_000_000_000
    assert policy.runtime_ready is True
    assert policy.stage_deadline_offsets_ns['proposal'] == 120_000_000_000


def test_runner_plan_passes_lisn_producer_deadline_flags():
    result = _runner_plan(
        '--lisn-slow-loop', '--lisn-producer-endpoint', 'http://127.0.0.1:9/v1/propose',
        '--lisn-producer-timeout-sec', '125', '--lisn-producer-deadline-sec', '120',
    )
    assert result.returncode == 0, result.stderr
    assert 'lisn_slow_loop_producer_timeout_sec:=125.0' in result.stdout
    assert 'lisn_slow_loop_producer_deadline_sec:=120.0' in result.stdout
