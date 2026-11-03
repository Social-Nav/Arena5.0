"""ROS 2 wiring for the LISN slow-loop contract.

The checked-in slow-loop contract is a pure, deterministic scheduler plus an
atomic parameter transaction.  It is intentionally ``runtime_ready=false``: three
of five logical parameters have no qualified one-to-one hot-reload binding, the
current controller callback ignores ``max_angular_velocity``, ``obstacle_weight``
is explicitly excluded, and the pre-set callback has no proven cross-callback
rollback contract.

This module therefore ships two clearly separated paths:

* ``run_wiring_smoke`` - deterministic, offline, fixture-backed.  It proves the
  scheduler -> validate -> apply -> readback -> evidence wiring.  It is a
  **wiring smoke, not a real agent and not a scientific result**.
* ``LisnSlowLoopNode`` - the ROS 2 transport.  It refuses to start while the
  policy is not runtime-ready and never publishes a control command; it only uses
  the controller's parameter services.

No model weights are loaded here and no control command is produced.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

from arena_bringup.lisn_vlm_doc_binding import safe_endpoint_label
from arena_bringup.hierarchical_modulation_contract import GOAL_KINDS
from arena_bringup.lisn_slow_loop_contract import (
    ApplyResult,
    LOGICAL_PARAMETERS,
    STAGE_ORDER,
    DeterministicSlowLoop,
    ObservationIdentity,
    SlowLoopPolicy,
    SlowLoopTimeline,
    canonical_sha256,
    expected_update_id,
    load_slow_loop_policy,
)


DEFAULT_POLICY = Path(__file__).resolve().parents[1] / 'configs/lisn/slow_loop_contract_v1.json'
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / 'test/fixtures/modulation_contract/valid.json'
)
EVIDENCE_SCHEMA = 'arena.lisn_slow_loop_ros_evidence'


class FixtureParameterAdapter:
    """Deterministic in-memory adapter for wiring smoke and unit tests."""

    def __init__(self, values: Mapping[str, float], *, readback_override: Mapping[str, float] | None = None,
                 fail_rollback: bool = False):
        self.values = {str(key): float(value) for key, value in values.items()}
        self.readback_override = (
            {str(key): float(value) for key, value in readback_override.items()}
            if readback_override is not None
            else None
        )
        self.fail_rollback = bool(fail_rollback)
        self.calls: list[tuple[str, dict[str, float]]] = []

    def read(self, names):
        self.calls.append(('read', dict(self.values)))
        return {str(name): float(self.values[str(name)]) for name in names}

    def verify_social_mpc(self):
        return True

    def set_atomically(self, values):
        requested = {str(key): float(value) for key, value in values.items()}
        self.calls.append(('set', requested))
        if self.fail_rollback and getattr(self, '_applied_once', False):
            return ApplyResult(False, 'fixture_rollback_rejected')
        self._applied_once = True
        self.values.update(requested)
        return ApplyResult(True, '')


class RosParameterAdapter:
    """Apply logical controller parameters through the controller node services.

    This adapter never publishes control messages.  It maps only bindings that
    the contract marks with an exact candidate; ambiguous bindings raise instead
    of guessing.
    """

    def __init__(self, node, controller_node: str, policy: SlowLoopPolicy, timeout_sec: float = 2.0):
        from rcl_interfaces.msg import ParameterType
        from rcl_interfaces.srv import GetParameters, SetParametersAtomically

        self._node = node
        self._controller = str(controller_node).rstrip('/')
        self._policy = policy
        self._timeout = float(timeout_sec)
        self._ParameterType = ParameterType
        self._get = node.create_client(GetParameters, self._controller + '/get_parameters')
        self._set = node.create_client(
            SetParametersAtomically, self._controller + '/set_parameters_atomically'
        )
        self.last_read_bindings: dict[str, str] = {}

    def _actual_name(self, logical: str) -> str:
        binding = self._policy.parameter_bindings.get(logical) or {}
        candidate = binding.get('candidate')
        if not isinstance(candidate, str) or not candidate.strip():
            raise RuntimeError(
                f"logical parameter {logical!r} has no exact runtime binding"
            )
        return candidate

    def _read_name(self, logical: str) -> str:
        """Read-only binding for frozen/ambiguous parameters.

        Writes still require an exact ``candidate``; for observation we may
        read the first declared candidate.  The receipt records the chosen
        name and the policy status so this is never presented as an apply.
        """
        binding = self._policy.parameter_bindings.get(logical) or {}
        candidate = binding.get('candidate')
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        observation = binding.get('observation_candidate')
        if isinstance(observation, str) and observation.strip():
            return observation
        candidates = binding.get('candidates')
        if isinstance(candidates, (list, tuple)):
            for name in candidates:
                if isinstance(name, str) and name.strip():
                    return name
        raise RuntimeError(
            f"logical parameter {logical!r} has no exact or candidate runtime binding"
        )

    def _wait(self, client) -> None:
        if not client.wait_for_service(timeout_sec=self._timeout):
            raise RuntimeError(f'{client.srv_name} unavailable')

    def _verify_controller(self, expected: str):
        from rcl_interfaces.srv import GetParameters

        self._wait(self._get)
        request = GetParameters.Request()
        # LISN loads the controller directly, so the controller_server's own
        # plugin parameter is the identity.  Requesting the wrapper-only
        # FollowPath.primary_controller on a direct plugin makes the server
        # log "Failed to get parameters" and would fail closed incorrectly.
        request.names = ['controller_plugins', 'FollowPath.plugin']
        result = _wait_future(self._node, self._get.call_async(request), self._timeout)
        values = list(result.values)
        if len(values) != 2:
            raise RuntimeError('controller identity parameters unavailable')
        plugins = (
            list(values[0].string_array_value)
            if values[0].type == self._ParameterType.PARAMETER_STRING_ARRAY else []
        )
        plugin = (
            str(values[1].string_value)
            if values[1].type == self._ParameterType.PARAMETER_STRING else ''
        )
        if 'FollowPath' not in plugins or expected not in plugin:
            raise RuntimeError(
                f'controller identity is not {expected}: plugins={plugins} plugin={plugin!r}'
            )
        return True

    def verify_social_mpc(self):
        """Fail closed unless the controller is the social_mpc FollowPath plugin."""
        return self._verify_controller('SocialMPCController')

    def verify_prox_mpc(self):
        """Fail closed unless the controller is the ProxMPC FollowPath plugin."""
        return self._verify_controller('ProxMpcController')

    def read(self, names):
        from rcl_interfaces.srv import GetParameters

        self._wait(self._get)
        request = GetParameters.Request()
        read_names = [self._read_name(str(name)) for name in names]
        request.names = list(read_names)
        future = self._get.call_async(request)
        result = _wait_future(self._node, future, self._timeout)
        values = {}
        self.last_read_bindings = {}
        for logical, actual, parameter in zip(names, read_names, result.values):
            if parameter.type != self._ParameterType.PARAMETER_DOUBLE:
                raise RuntimeError(f'{actual} is not a double parameter')
            values[str(logical)] = float(parameter.double_value)
            self.last_read_bindings[str(logical)] = actual
        return values

    def set_atomically(self, values):
        from rcl_interfaces.msg import Parameter, ParameterValue
        from rcl_interfaces.srv import SetParametersAtomically

        self._wait(self._set)
        request = SetParametersAtomically.Request()
        for logical, value in values.items():
            request.parameters.append(
                Parameter(
                    name=self._actual_name(str(logical)),
                    value=ParameterValue(
                        type=self._ParameterType.PARAMETER_DOUBLE,
                        double_value=float(value),
                    ),
                )
            )
        future = self._set.call_async(request)
        result = _wait_future(self._node, future, self._timeout)
        if result is None or not result.result.successful:
            return ApplyResult(False, getattr(result.result, 'reason', 'set_rejected'))
        return ApplyResult(True, '')


def _wait_future(node, future, timeout_sec: float):
    deadline = time.monotonic() + timeout_sec
    while not future.done():
        if time.monotonic() >= deadline:
            raise RuntimeError('parameter service timed out')
        time.sleep(0.01)
    return future.result()


@dataclass
class TickRunner:
    """One deterministic slow-loop tick with offline evidence output."""

    policy: SlowLoopPolicy
    episode_id: str
    epoch_monotonic_ns: int
    target_frame: str
    adapter: Any
    evidence_path: Path
    wiring_smoke: bool = False

    def __post_init__(self):
        self.loop = DeterministicSlowLoop(
            policy=self.policy,
            episode_id=self.episode_id,
            epoch_monotonic_ns=self.epoch_monotonic_ns,
            target_frame=self.target_frame,
        )
        self.evidence_path = Path(self.evidence_path)
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)

    def scheduled_for_ns(self, tick_index: int | None = None) -> int:
        return self.loop.scheduled_for_ns(tick_index)

    def tick(self, snapshot: Mapping[str, Any], *, received_monotonic_ns: int,
             now_monotonic_ns: int | None = None, now_ros_ns: int | None = None) -> dict[str, Any]:
        now_monotonic_ns = int(now_monotonic_ns if now_monotonic_ns is not None else received_monotonic_ns)
        now_ros_ns = int(now_ros_ns if now_ros_ns is not None else now_monotonic_ns)
        payload = copy.deepcopy(dict(snapshot))
        source = payload.get('source_observation') or {}
        observation = ObservationIdentity(
            episode_id=self.episode_id,
            observation_id=str(source.get('id') or ''),
            frame_id=str(source.get('frame_id') or ''),
            ros_stamp_ns=int(source.get('stamp_ns') or 0),
            received_monotonic_ns=int(received_monotonic_ns),
            content_sha256=canonical_sha256(source),
        )
        payload['update_id'] = expected_update_id(
            episode_id=self.episode_id,
            tick_index=self.loop.state.next_tick_index,
            observation_id=observation.observation_id,
            snapshot=payload,
        )
        timeline = SlowLoopTimeline(
            proposal_completed_ns=now_monotonic_ns,
            validation_completed_ns=now_monotonic_ns,
            apply_completed_ns=now_monotonic_ns,
            readback_completed_ns=now_monotonic_ns,
            evidence_completed_ns=now_monotonic_ns,
            rollback_completed_ns=now_monotonic_ns,
        )
        record = self.loop.process(
            snapshot=payload,
            observation=observation,
            now_monotonic_ns=now_monotonic_ns,
            now_ros_ns=now_ros_ns,
            timeline=timeline,
            adapter=self.adapter,
        )
        record = dict(record)
        record['wiring_smoke'] = bool(self.wiring_smoke)
        with self.evidence_path.open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
        return record


def load_engineering_policy(path) -> SlowLoopPolicy:
    """Load the separate >=30 s engineering profile; the frozen 10 s contract is untouched."""
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get('schema') != 'arena.lisn_slow_loop_engineering' or value.get('schema_version') != 1:
        raise ValueError('unsupported engineering slow-loop schema/version')
    schedule = value['schedule']
    period = int(schedule['period_ns'])
    if period < 30_000_000_000:
        raise ValueError('engineering slow-loop period must be >= 30 s')
    if int(schedule['first_tick_offset_ns']) != period:
        raise ValueError('engineering first tick must be one full period after arming')
    offsets = value['budgets']['stage_deadline_offsets_ns']
    if tuple(offsets) != STAGE_ORDER:
        raise ValueError('engineering stage deadline order changed')
    ordered = [int(offsets[name]) for name in STAGE_ORDER]
    if ordered != sorted(ordered) or ordered[0] <= 0 or ordered[-1] != period:
        raise ValueError('engineering deadlines must be positive, monotonic and end at the period')
    atomic = value['atomic_apply']
    allow = tuple(atomic['logical_parameter_allowlist'])
    if allow != LOGICAL_PARAMETERS:
        raise ValueError('engineering logical parameter allowlist changed')
    if set(atomic['parameter_bindings']) != set(LOGICAL_PARAMETERS):
        raise ValueError('engineering parameter binding coverage is incomplete')
    for flag in ('set_parameters_atomically_required', 'exact_readback_required',
                 'rollback_on_any_mismatch', 'fault_on_rollback_failure'):
        if atomic[flag] is not True:
            raise ValueError(f'engineering requirement disabled: {flag}')
    isolation = value['isolation']
    return SlowLoopPolicy(
        policy_sha256=hashlib.sha256(raw).hexdigest(),
        period_ns=period,
        max_observation_age_ns=int(schedule['max_observation_age_ns']),
        stage_deadline_offsets_ns={name: int(offsets[name]) for name in STAGE_ORDER},
        active_control_path=str(isolation['active_control_path']),
        active_execution_interface=str(isolation['active_execution_interface']),
        isolated_control_interfaces=frozenset(isolation['isolated_control_interfaces']),
        logical_parameters=allow,
        runtime_ready=bool(atomic['runtime_ready']),
        runtime_blockers=tuple(atomic.get('runtime_blockers') or ()),
        parameter_bindings=copy.deepcopy(atomic['parameter_bindings']),
    )


def load_grounding_live_policy(path) -> SlowLoopPolicy:
    """Load the LISN grounding-live profile.

    This profile removes the hard received-age rejection while keeping the
    source identity check.  It is an engineering profile, not a scientific
    contract change; the frozen 10 s / P1A profiles are untouched.
    """
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get('schema') != 'arena.lisn_grounding_live' or value.get('schema_version') != 1:
        raise ValueError('unsupported grounding-live slow-loop schema/version')
    if value.get('not_scientific') is not True:
        raise ValueError('grounding-live profile must be marked not_scientific')
    schedule = value['schedule']
    period = int(schedule['period_ns'])
    if period < 10_000_000_000:
        raise ValueError('grounding-live period must be >= 10 s')
    if int(schedule['first_tick_offset_ns']) != period:
        raise ValueError('grounding-live first tick must be one full period after arming')
    if schedule.get('catch_up') is not False:
        raise ValueError('grounding-live catch-up bursts are forbidden')
    if schedule.get('one_observation_per_tick') is not True:
        raise ValueError('grounding-live must bind exactly one observation per tick')
    if schedule.get('enforce_observation_age') is not False:
        raise ValueError('grounding-live must explicitly disable the hard age gate')
    offsets = value['budgets']['stage_deadline_offsets_ns']
    if tuple(offsets) != STAGE_ORDER:
        raise ValueError('grounding-live stage deadline order changed')
    ordered = [int(offsets[name]) for name in STAGE_ORDER]
    if ordered != sorted(ordered) or ordered[0] <= 0 or ordered[-1] != period:
        raise ValueError('grounding-live deadlines must be positive, monotonic and end at the period')
    atomic = value['atomic_apply']
    allow = tuple(atomic['logical_parameter_allowlist'])
    if allow != LOGICAL_PARAMETERS:
        raise ValueError('grounding-live logical parameter allowlist changed')
    if set(atomic['parameter_bindings']) != set(LOGICAL_PARAMETERS):
        raise ValueError('grounding-live parameter binding coverage is incomplete')
    for flag in ('set_parameters_atomically_required', 'exact_readback_required',
                 'rollback_on_any_mismatch', 'fault_on_rollback_failure'):
        if atomic[flag] is not True:
            raise ValueError(f'grounding-live requirement disabled: {flag}')
    isolation = value['isolation']
    return SlowLoopPolicy(
        policy_sha256=hashlib.sha256(raw).hexdigest(),
        period_ns=period,
        max_observation_age_ns=int(schedule['max_observation_age_ns']),
        enforce_observation_age=False,
        stage_deadline_offsets_ns={name: int(offsets[name]) for name in STAGE_ORDER},
        active_control_path=str(isolation['active_control_path']),
        active_execution_interface=str(isolation['active_execution_interface']),
        isolated_control_interfaces=frozenset(isolation['isolated_control_interfaces']),
        logical_parameters=allow,
        runtime_ready=bool(atomic['runtime_ready']),
        runtime_blockers=tuple(atomic.get('runtime_blockers') or ()),
        parameter_bindings=copy.deepcopy(atomic['parameter_bindings']),
    )


def load_policy(path, engineering: bool = False) -> SlowLoopPolicy:
    if engineering:
        raw = json.loads(Path(path).read_text())
        if raw.get('schema') == 'arena.lisn_slow_loop_engineering':
            return load_engineering_policy(path)
        if raw.get('schema') == 'arena.lisn_grounding_live':
            return load_grounding_live_policy(path)
    return load_slow_loop_policy(Path(path))


def production_blockers(policy: SlowLoopPolicy) -> list[str]:
    """Return the exact contract reasons that forbid a production start."""
    blockers = [str(item) for item in policy.runtime_blockers]
    if not policy.runtime_ready and not blockers:
        blockers = ['policy.runtime_ready is false without an explicit blocker']
    return blockers


def run_wiring_smoke(*, policy_path: Path, fixture_path: Path, evidence_path: Path,
                     target_frame: str = 'map', episode_id: str = 'wiring-smoke-episode') -> dict[str, Any]:
    """Run one deterministic offline tick. Not a real agent; not a science result."""
    policy = load_slow_loop_policy(Path(policy_path))
    fixture = json.loads(Path(fixture_path).read_text())
    adapter = FixtureParameterAdapter(fixture['controller_parameters'])
    epoch = 1_000_000_000
    runner = TickRunner(
        policy=policy,
        episode_id=episode_id,
        epoch_monotonic_ns=epoch,
        target_frame=target_frame,
        adapter=adapter,
        evidence_path=Path(evidence_path),
        wiring_smoke=True,
    )
    scheduled = runner.scheduled_for_ns()
    now_monotonic = scheduled + 1
    now_ros = 1_500_000_000
    record = runner.tick(
        fixture,
        received_monotonic_ns=now_monotonic - 100_000_000,
        now_monotonic_ns=now_monotonic,
        now_ros_ns=now_ros,
    )
    receipt = {
        'schema': EVIDENCE_SCHEMA,
        'kind': 'WIRING_SMOKE',
        'labels': {
            'wiring_smoke': True,
            'real_agent': False,
            'scientific_result': False,
            'runtime_executed': False,
        },
        'policy_path': str(Path(policy_path)),
        'policy_sha256': policy.policy_sha256,
        'policy_runtime_ready': policy.runtime_ready,
        'policy_blockers': list(policy.runtime_blockers),
        'fixture_path': str(Path(fixture_path)),
        'evidence_path': str(Path(evidence_path)),
        'decision': record.get('decision'),
        'reason_codes': record.get('reason_codes'),
        'tick_index': record.get('tick_index'),
        'runtime_executed': False,
    }
    receipt_path = Path(evidence_path).with_name('WIRING_SMOKE.json')
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt



# Exact hot-reload binding verified in the controller source:
# src/deps/nav2_social_mpc_controller/src/social_mpc_controller.cpp
# dynamicParametersCallback handles plugin_name_+".optimizer.max_linear_velocity".
ENGINEERING_APPLIED_PARAMETERS = ('max_linear_velocity_mps',)
ENGINEERING_FROZEN_PARAMETERS = (
    'max_angular_velocity_rps',
    'goal_weight',
    'obstacle_weight',
    'people_weight',
)
ENGINEERING_LOGICAL_PARAMETERS = ENGINEERING_APPLIED_PARAMETERS + ENGINEERING_FROZEN_PARAMETERS
DEFAULT_ENGINEERING_POLICY = (
    Path(__file__).resolve().parents[1] / 'configs/lisn/slow_loop_engineering_v1.json'
)


ERROR_CODES = frozenset({
    'unknown',
    'auth_denied',
    'auth_latched',
    'http_error',
    'transport_unavailable',
    'timeout',
    'response_not_json',
    'response_schema_invalid',
    'credential_missing',
    'endpoint_missing',
    'deadline_exceeded',
    'invalid_deadline',
    'oversize_result',
    'oversize_request',
    'abort_failed',
    'goal_untrusted',
    'model_alias_forbidden',
    'model_identity_mismatch',
    'grounding_sync_missing',
    'grounding_tf_missing',
    'grounding_rejected',
})

NAMED_AUTH_PROVIDERS = frozenset({'gemini', 'modelhub', 'ark', 'dmxapi', 'openai'})
TOOL_VERSION_DEFAULT = 'lisn-producer-v1'
MODEL_ALIASES = frozenset({'gpt-4o'})


def safe_error_code(error: BaseException) -> str:
    """Return only a trusted, static producer error code.

    Arbitrary exceptions may expose ``code`` attributes containing echoed
    credentials; the node and the gate never read those.
    """
    if isinstance(error, ProducerUnavailableError):
        return error.code
    return 'unknown'


def safe_error_category(error: BaseException) -> str:
    """Classify an exception into a static category without its text.

    Only built-in types are named; a custom exception class name or message
    could echo a credential and is therefore reported as ``unknown``.
    """
    if isinstance(error, ProducerUnavailableError):
        return 'producer'
    if isinstance(error, TimeoutError):
        return 'timeout'
    if isinstance(error, ValueError):
        return 'value_error'
    if isinstance(error, KeyError):
        return 'key_error'
    if isinstance(error, OSError):
        return 'os_error'
    if isinstance(error, RuntimeError):
        return 'runtime_error'
    return 'unknown'


class ProducerUnavailableError(RuntimeError):
    """Structured, redacted producer failure.

    ``code`` is a static category and ``status`` is the HTTP status when one
    exists.  The exception message never carries a reason phrase, response
    body, header, URL path/query/userinfo, or the original exception text.
    """

    def __init__(self, code: str, *, status: int | None = None):
        requested = str(code)
        self.code = requested if requested in ERROR_CODES else 'unknown'
        normalized = None
        if status is not None and not isinstance(status, bool):
            try:
                candidate = int(status)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and 100 <= candidate <= 599:
                normalized = candidate
        self.status = normalized
        super().__init__(self.code)

    def __str__(self) -> str:
        if self.status is None:
            return self.code
        return f'{self.code}:status={self.status}'


class ProducerAuthGate:
    """First-401/403 latch shared by the producer and the node loop."""

    def __init__(self) -> None:
        self.latched = False
        self.error_count = 0
        self.last_code = ''

    def should_request(self) -> bool:
        return not self.latched

    def record_failure(self, error: BaseException) -> bool:
        if self.latched:
            return False
        self.latched = True
        self.error_count += 1
        self.last_code = safe_error_code(error)
        return True


class EngineeringParameterAdapter:
    """Apply only the exact hot-reload binding; observe and freeze the rest.

    The frozen parameters are still read so the contract's all-field readback
    comparison works.  A proposal that changes any frozen parameter is rejected
    before the controller is touched.
    """

    def __init__(self, base, applied=ENGINEERING_APPLIED_PARAMETERS,
                 frozen=ENGINEERING_FROZEN_PARAMETERS):
        self.base = base
        self.applied = tuple(applied)
        self.frozen = tuple(frozen)
        self._last: dict[str, float] = {}
        self.last_read_bindings: dict[str, str] = {}

    def verify_social_mpc(self):
        verify = getattr(self.base, 'verify_social_mpc', None)
        if callable(verify):
            return verify()
        return True

    def read(self, names):
        values = {str(name): float(value) for name, value in self.base.read(names).items()}
        self._last = dict(values)
        self.last_read_bindings = dict(getattr(self.base, 'last_read_bindings', {}) or {})
        return values

    def set_atomically(self, values):
        requested = {str(name): float(value) for name, value in values.items()}
        for name in self.frozen:
            if name not in requested:
                continue
            current = self._last.get(name)
            if current is None:
                return ApplyResult(False, f'FROZEN_PARAMETER_UNREAD:{name}')
            if abs(requested[name] - float(current)) > 1e-9:
                return ApplyResult(False, f'UNQUALIFIED_PARAMETER_CHANGE:{name}')
        subset = {name: requested[name] for name in self.applied if name in requested}
        if not subset:
            return ApplyResult(False, 'NO_QUALIFIED_PARAMETER_TO_APPLY')
        return self.base.set_atomically(subset)


class ProxGoalOnlyAdapter:
    """Honest adapter for the static-parameter ProxMPC controller.

    ProxMPC declares its parameters but has no runtime ``on_set_parameters``
    callback, so a successful ``set_parameters_atomically`` would change the
    ROS parameter value without changing the solver.  This adapter therefore
    pins the host-owned parameters, accepts a no-op apply only when the
    proposal exactly matches the pinned values, and records that no dynamic
    apply exists.  The goal chain still proceeds; nothing claims a hot reload.
    """

    STATIC = {
        'max_linear_velocity_mps': 0.8,
        'max_angular_velocity_rps': 1.4,
        'goal_weight': 10.0,
        'obstacle_weight': 0.0,
        'people_weight': 0.0,
    }

    def __init__(self, base):
        self.base = base
        self.last_read_bindings: dict[str, str] = {}

    def verify_social_mpc(self):
        verify = getattr(self.base, 'verify_prox_mpc', None)
        if callable(verify):
            return verify()
        return True

    def read(self, names):
        self.last_read_bindings = {
            str(name): f'static_pinned:{self.STATIC.get(str(name))}' for name in names
        }
        return {str(name): float(self.STATIC[str(name)]) for name in names}

    def set_atomically(self, values):
        for name, value in values.items():
            expected = self.STATIC.get(str(name))
            if expected is None or abs(float(value) - float(expected)) > 1e-9:
                return ApplyResult(False, f'STATIC_PARAMETER_MISMATCH:{name}')
        return ApplyResult(True, 'STATIC_PINNED_NO_DYNAMIC_CALLBACK')


SESSION_STATUS_VALUES = ('ready', 'running', 'hold', 'stop', 'error')


def session_status_payload(*, session_id: str, episode: int, status: str = 'ready',
                           engineering: bool = False, **extra) -> dict:
    """Status payload accepted by ``WaypointLifecycle.Session.receive``.

    The task generator binds each episode to ``(session_id, episode)`` and
    rejects any status sample that does not carry the same pair.  A slow-loop
    model that only publishes ``status=ready`` is invisible to the
    episode-start barrier, so every heartbeat must be session-bound.
    """
    if str(status) not in SESSION_STATUS_VALUES:
        raise ValueError(f'unsupported session status {status!r}')
    payload = {
        'status': str(status),
        'session_id': str(session_id),
        'episode': int(episode),
        'source': 'lisn_slow_loop',
        'engineering': bool(engineering),
    }
    payload.update(extra)
    return payload


def parse_eval_ready_payload(data: str):
    """Return ``(session_id, episode)`` from ``/eval_ready_status``.

    Returns ``(None, None)`` for malformed or pre-session samples; callers must
    never invent an identity.
    """
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    session_id = str(payload.get('session_id') or '').strip()
    episode = payload.get('episode')
    if not session_id or type(episode) is not int:
        return None, None
    return session_id, episode


@dataclass(frozen=True)
class ProducerConfig:
    """Endpoint/credential resolution for the real LISN VLM producer.

    Supported providers, in resolution order:

    * ``ARENA_LISN_VLM_ENDPOINT`` - explicit generic endpoint.
    * ``modelhub`` - ModelHub/GPT Open API. Requires ``GPT_OPEN_API_KEY`` and
      ``GPT_OPEN_API_VERSION``; the deployment id is a path component.
    * ``ark`` - Ark OpenAI-compatible API. Requires ``ARK_API_KEY``.
    * ``none`` - no legal credential env present; the producer fails closed.

    Only environment references are stored.  The API key value is never
    serialized into receipts or logs; ``redacted()`` exposes only its presence.
    """

    endpoint: str
    api_key: str = ''
    api_key_env: str = 'OPENAI_API_KEY'
    model: str = 'gpt-4o'
    model_revision: str = ''
    model_image_id: str = ''
    tool_version: str = TOOL_VERSION_DEFAULT
    prompt_version: str = ''
    enforce_model_identity: bool = False
    json_response_format: bool = False
    target_frame: str = 'map'
    assume_goal_trusted: bool = False
    grounding_live: bool = False
    grounding_scan: bool = False
    provider: str = 'generic'
    api_version: str = ''
    models_endpoint: str = ''
    auth_header: str = 'Authorization'
    auth_scheme: str = 'Bearer'
    openai_compatible: bool = True
    allow_unauthenticated: bool = False
    max_tokens: int = 4095
    timeout_sec: float = 8.0
    deadline_sec: float = 120.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> 'ProducerConfig':
        env = os.environ if env is None else env
        try:
            timeout = float(env.get('ARENA_LISN_VLM_TIMEOUT_SEC', '8'))
        except (TypeError, ValueError):
            timeout = 8.0
        try:
            deadline = float(env.get('ARENA_LISN_VLM_DEADLINE_SEC', '120'))
        except (TypeError, ValueError):
            deadline = 120.0
        if deadline <= 0:
            deadline = 0.0
        explicit = str(env.get('ARENA_LISN_VLM_ENDPOINT', '') or '').strip()
        if explicit:
            key_env = str(env.get('ARENA_LISN_VLM_API_KEY_ENV', 'OPENAI_API_KEY') or 'OPENAI_API_KEY').strip()
            try:
                max_tokens = int(env.get('ARENA_LISN_VLM_MAX_TOKENS', '4095') or 4095)
            except (TypeError, ValueError):
                max_tokens = 4095
            return cls(
                endpoint=explicit, api_key=str(env.get(key_env, '') or ''),
                api_key_env=key_env,
                model=str(env.get('ARENA_LISN_VLM_MODEL', 'gpt-4o') or 'gpt-4o'),
                model_revision=str(env.get('ARENA_LISN_VLM_MODEL_REVISION', '') or ''),
                model_image_id=str(env.get('ARENA_LISN_VLM_IMAGE_ID', '') or ''),
                tool_version=str(env.get('ARENA_LISN_VLM_TOOL_VERSION', TOOL_VERSION_DEFAULT) or TOOL_VERSION_DEFAULT),
                prompt_version=str(env.get('ARENA_LISN_VLM_PROMPT_VERSION', '') or ''),
                enforce_model_identity=str(env.get('ARENA_LISN_VLM_ENFORCE_MODEL_IDENTITY', '0') or '0').strip().lower() in {'1', 'true', 'yes'},
                json_response_format=str(env.get('ARENA_LISN_VLM_JSON_FORMAT', '0') or '0').strip().lower() in {'1', 'true', 'yes'},
                target_frame=str(env.get('ARENA_LISN_VLM_TARGET_FRAME', 'map') or 'map'),
                grounding_scan=str(env.get('ARENA_LISN_GROUNDING_SCAN', '0') or '0').strip().lower() in {'1', 'true', 'yes'},
                provider='generic', timeout_sec=timeout, deadline_sec=deadline,
                max_tokens=max_tokens,
            )
        openai_key = str(env.get('OPENAI_API_KEY', '') or '')
        openai_base = str(env.get('OPENAI_BASE_URL', '') or '').strip()
        if openai_key and openai_base:
            return cls(endpoint=openai_base.rstrip('/') + '/chat/completions',
                       api_key=openai_key, api_key_env='OPENAI_API_KEY',
                       model=str(env.get('ARENA_LISN_VLM_MODEL', 'gpt-4o') or 'gpt-4o'),
                       provider='openai', timeout_sec=timeout, deadline_sec=deadline)
        gpt_key = str(env.get('GPT_OPEN_API_KEY', '') or '')
        gpt_version = str(env.get('GPT_OPEN_API_VERSION', '') or '')
        if gpt_key and gpt_version:
            # CN ModelHub official base; the search domain is an accepted
            # fallback.  The deployment path component is optional and comes
            # from the console/doc, never guessed here.
            base = str(env.get('GPT_OPEN_API_BASE', 'https://aidp.bytedance.net/api/modelhub/online/v2/crawl') or 'https://aidp.bytedance.net/api/modelhub/online/v2/crawl').strip()
            deployment = str(env.get('ARENA_LISN_VLM_DEPLOYMENT', '') or '').strip()
            model = str(env.get('ARENA_LISN_VLM_MODEL', 'gpt-5.4-nano-2026-03-17') or 'gpt-5.4-nano-2026-03-17').strip()
            path = f'{base.rstrip("/")}/{deployment}/chat/completions' if deployment else f'{base.rstrip("/")}/chat/completions'
            endpoint = f'{path}?api-version={gpt_version}'
            return cls(endpoint=endpoint, api_key=gpt_key, api_key_env='GPT_OPEN_API_KEY',
                       model=model, provider='modelhub', api_version=gpt_version,
                       auth_header='api-key', auth_scheme='', timeout_sec=timeout, deadline_sec=deadline)
        ark_key = str(env.get('ARK_API_KEY', '') or '')
        if ark_key:
            # Runtime-configured base/model win over legacy names.  The
            # endpoint and model are non-secret; the key is only ever read
            # from the environment reference and never persisted.
            base = str(env.get('ARENA_LISN_BASE_URL', '')
                       or env.get('ARK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
                       or 'https://ark.cn-beijing.volces.com/api/v3').strip()
            model = str(env.get('ARENA_LISN_MODEL', '')
                        or env.get('ARENA_LISN_VLM_MODEL', 'doubao-seed-2-0-lite-260215')
                        or 'doubao-seed-2-0-lite-260215').strip()
            return cls(endpoint=f'{base.rstrip("/")}/chat/completions', api_key=ark_key,
                       api_key_env='ARK_API_KEY', model=model, provider='ark',
                       auth_header='Authorization', auth_scheme='Bearer', timeout_sec=timeout, deadline_sec=deadline)
        gemini_key = str(env.get('ARENA_LISN_GEMINI_AK', '') or env.get('GEMINI_API_KEY', '') or '')
        if gemini_key:
            base = str(env.get('ARENA_LISN_GEMINI_BASE', 'https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi') or 'https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi').strip()
            version = str(env.get('ARENA_LISN_GEMINI_API_VERSION', '2024-03-01-preview') or '2024-03-01-preview').strip()
            model = str(env.get('ARENA_LISN_VLM_MODEL', 'gemini-2.5-pro') or 'gemini-2.5-pro').strip()
            endpoint = f'{base.rstrip("/")}/openai/deployments/{model}/chat/completions?api-version={version}'
            return cls(endpoint=endpoint, api_key=gemini_key, api_key_env='ARENA_LISN_GEMINI_AK',
                       model=model, provider='gemini', api_version=version,
                       auth_header='api-key', auth_scheme='', timeout_sec=timeout,
                       max_tokens=int(env.get('ARENA_LISN_VLM_MAX_TOKENS', '4095') or 4095))
        dmx_key = str(env.get('DMXAPI_API_KEY', '') or env.get('DMXAPI_KEY', '') or '')
        if dmx_key:
            base = str(env.get('DMXAPI_BASE_URL', 'https://www.dmxapi.cn') or 'https://www.dmxapi.cn').strip()
            model = str(env.get('DMXAPI_MODEL', '') or env.get('ARENA_LISN_VLM_MODEL', 'gpt-5.6-sol') or 'gpt-5.6-sol').strip()
            key_env = 'DMXAPI_API_KEY' if env.get('DMXAPI_API_KEY') else 'DMXAPI_KEY'
            return cls(endpoint=f'{base.rstrip("/")}/v1/chat/completions', api_key=dmx_key,
                       api_key_env=key_env, model=model, provider='dmxapi',
                       models_endpoint=f'{base.rstrip("/")}/v1/models',
                       auth_header='Authorization', auth_scheme='', timeout_sec=timeout, deadline_sec=deadline)
        return cls(endpoint='', api_key='', api_key_env='', model='', provider='none',
                   auth_header='Authorization', auth_scheme='Bearer', timeout_sec=timeout, deadline_sec=deadline)

    @classmethod
    def local(cls, endpoint: str, *, timeout_sec: float, deadline_sec: float,
              env: Mapping[str, str] | None = None,
              grounding_live: bool | None = None,
              grounding_scan: bool | None = None) -> 'ProducerConfig':
        """Explicit local endpoint with host-owned model metadata; no cloud fallback."""
        env = os.environ if env is None else env
        try:
            max_tokens = int(env.get('ARENA_LISN_VLM_MAX_TOKENS', '512') or 512)
        except (TypeError, ValueError):
            max_tokens = 512
        if grounding_live is None:
            grounding_live = str(env.get('ARENA_LISN_VLM_GROUNDING_LIVE', '0') or '0').strip().lower() in {'1', 'true', 'yes'}
        if grounding_scan is None:
            grounding_scan = str(env.get('ARENA_LISN_GROUNDING_SCAN', '0') or '0').strip().lower() in {'1', 'true', 'yes'}
        return cls(
            endpoint=str(endpoint), api_key='', api_key_env='',
            model=str(env.get('ARENA_LISN_VLM_MODEL', '') or ''),
            model_revision=str(env.get('ARENA_LISN_VLM_MODEL_REVISION', '') or ''),
            model_image_id=str(env.get('ARENA_LISN_VLM_IMAGE_ID', '') or ''),
            tool_version=str(env.get('ARENA_LISN_VLM_TOOL_VERSION', TOOL_VERSION_DEFAULT) or TOOL_VERSION_DEFAULT),
            prompt_version=str(env.get('ARENA_LISN_VLM_PROMPT_VERSION', '') or ''),
            enforce_model_identity=str(env.get('ARENA_LISN_VLM_ENFORCE_MODEL_IDENTITY', '1') or '1').strip().lower() in {'1', 'true', 'yes'},
            json_response_format=str(env.get('ARENA_LISN_VLM_JSON_FORMAT', '1') or '1').strip().lower() in {'1', 'true', 'yes'},
            target_frame=str(env.get('ARENA_LISN_VLM_TARGET_FRAME', 'map') or 'map'),
            grounding_live=bool(grounding_live),
            grounding_scan=bool(grounding_scan),
            provider='generic', openai_compatible=True, allow_unauthenticated=True,
            timeout_sec=float(timeout_sec), deadline_sec=float(deadline_sec),
            max_tokens=max_tokens,
        )

    def redacted(self) -> dict[str, Any]:
        return {
            'provider': self.provider,
            'endpoint_label': safe_endpoint_label(self.endpoint),
            'model': self.model,
            'api_key_present': bool(self.api_key),
            'api_key_env': self.api_key_env,
            'api_version_present': bool(self.api_version),
            'models_endpoint_present': bool(self.models_endpoint),
        }


def _encode_image_for_vlm(image_bytes: bytes, *, image_encoding: str, width: int,
                          height: int, channels: int, image_mime: str = '') -> tuple[bytes, str]:
    """Return (encoded_bytes, mime).  Raw sensor bytes are never labelled JPEG."""
    mime = str(image_mime or '').strip().lower()
    if mime in {'image/jpeg', 'image/jpg', 'image/png'}:
        return bytes(image_bytes), mime
    encoding = str(image_encoding or '').strip().lower()
    if width <= 0 or height <= 0:
        raise ValueError('raw image requires positive width/height')
    from PIL import Image
    import io

    if encoding in {'rgb8', 'bgr8'}:
        expected = int(width) * int(height) * 3
        if len(image_bytes) != expected:
            raise ValueError(
                f'{encoding} payload has {len(image_bytes)} bytes, expected {expected}'
            )
        raw_mode = 'RGB' if encoding == 'rgb8' else 'BGR'
        image = Image.frombytes('RGB', (int(width), int(height)), bytes(image_bytes), 'raw', raw_mode)
    elif encoding in {'mono8', '8uc1'}:
        expected = int(width) * int(height)
        if len(image_bytes) != expected:
            raise ValueError(
                f'{encoding} payload has {len(image_bytes)} bytes, expected {expected}'
            )
        image = Image.frombytes('L', (int(width), int(height)), bytes(image_bytes)).convert('RGB')
    else:
        raise ValueError(f'unsupported image encoding {encoding!r}')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    return buffer.getvalue(), 'image/jpeg'


LISN_PROPOSAL_PROMPT = (
    'You are the slow-loop VLM for a social navigation robot. Return ONLY a JSON object, no markdown. '
    'Required keys: controller_mode (goal|follow|avoid|wait|cautious), '
    'controller_parameters {max_linear_velocity_mps, max_angular_velocity_rps, goal_weight, '
    'obstacle_weight, people_weight}, goal {kind, frame_id, stamp_ns, value{x,y,yaw}}, '
    'semantic_entities (array), validity {valid:boolean, reason:string}, '
    'provenance {model_version, prompt_version, tool_version, reasoning_latency_ms}. '
    'Use conservative values; max_linear_velocity_mps between 0.2 and 0.5.'
)

LISN_GROUNDING_PROMPT = (
    'Return ONLY one compact JSON object. No explanation, no thinking, no markdown. '
    'Keys: controller_mode (goal|follow|avoid|wait|cautious), '
    'controller_parameters {max_linear_velocity_mps, max_angular_velocity_rps, goal_weight, '
    'obstacle_weight, people_weight}, goal {kind:pixel_goal, frame_id, stamp_ns, '
    'value{x,y}, units:normalized|pixel, confidence}, '
    'semantic_entities (0-3 items; each {class_name, attribute, instance_id, confidence, '
    'cost_value 0-254, inflation_radius_m 0-5, decay_rate 0-10, '
    'grounding {kind:pixel, pixel{x,y}, units:normalized}}), '
    'validity {valid:boolean, reason:string}. '
    'Allowed classes: table,booth,chair,shelf,wall,obstacle,door,plant,counter,cabinet,trash_bin,display; '
    'never person/unknown. Use Source frame_id/stamp_ns verbatim; normalized x,y in [0,1]. '
    'No host provenance. For goal/follow with no obstacle ahead set max_linear_velocity_mps=0.8; '
    'reduce only for avoid/cautious/wait, turns, obstacles, pedestrians or STOP.'
)


LISN_SCAN_GROUNDING_PROMPT = (
    'Return ONLY one compact JSON object. No explanation, no thinking, no markdown. '
    'Keys: controller_mode (goal|follow|avoid|wait|cautious), '
    'controller_parameters {max_linear_velocity_mps, max_angular_velocity_rps, goal_weight, '
    'obstacle_weight, people_weight}, goal {kind:pixel_goal, frame_id, stamp_ns, '
    'value{x,y}, units:normalized|pixel, confidence}, '
    'semantic_entities: [] (must be an empty array), '
    'validity {valid:boolean, reason:string}. '
    'Use Source frame_id/stamp_ns verbatim; normalized x,y in [0,1]. '
    'Choose a point on the walkable floor/ground ahead, visible in the 2D LiDAR plane; '
    'do not choose a table top, wall top, or object above the floor. '
    'No host provenance. For goal/follow with no obstacle ahead set max_linear_velocity_mps=0.8.'
)


def build_grounding_text(prompt: str, *, instruction: str, frame_id: str,
                         stamp_ns: int, width: int, height: int) -> str:
    """Host-supplied source context for the grounding prompt.

    The VLM must be told the exact camera frame, image size and source stamp;
    otherwise it guesses ``camera`` / ``480`` and the host correctly rejects
    the proposal.  The host still owns the authoritative bundle.
    """
    return (
        f'{prompt}\n'
        f'Source frame_id: {frame_id}\n'
        f'Source stamp_ns: {int(stamp_ns)}\n'
        f'Image width: {int(width)}, height: {int(height)} (top-left origin)\n'
        f'Instruction: {instruction}'
    )
PROMPT_VERSION = 'lisn-proposal-prompt-' + hashlib.sha256(
    LISN_PROPOSAL_PROMPT.encode('utf-8')
).hexdigest()[:12]


def assess_goal_trust(goal: Any, *, target_frame: str,
                      known_projection: bool = False,
                      semantic_match: bool = False) -> dict[str, Any]:
    """Host-side trust decision for a VLM goal.

    A VLM free-form coordinate is not a live goal unless the host has a known
    camera-to-map projection or a known semantic target match.
    """
    reasons: list[str] = []
    if not isinstance(goal, Mapping):
        return {'trusted': False, 'source_invalid': True, 'reason': 'goal_not_object',
                'kind': None, 'frame_id': None}
    kind = goal.get('kind')
    frame_id = goal.get('frame_id')
    if kind not in GOAL_KINDS:
        reasons.append('unsupported_goal_kind')
    if not isinstance(goal.get('value'), Mapping):
        reasons.append('goal_value_not_object')
    if not isinstance(goal.get('stamp_ns'), int) or isinstance(goal.get('stamp_ns'), bool):
        reasons.append('goal_stamp_not_integer')
    if kind == 'map_pose' and frame_id != target_frame:
        reasons.append('goal_frame_mismatch')
    if not known_projection and not semantic_match:
        reasons.append('no_known_camera_to_map_projection_or_semantic_match')
    return {
        'trusted': not reasons,
        'source_invalid': bool(reasons),
        'reason': ';'.join(reasons) if reasons else 'trusted',
        'kind': kind,
        'frame_id': frame_id,
    }


def validate_model_identity(*, request_model: str, served_model: str,
                            response_model: str) -> dict[str, Any]:
    """Request/served/response model identity check; aliases are rejected."""
    request_model = str(request_model or '')
    served_model = str(served_model or '')
    response_model = str(response_model or '')
    if request_model in MODEL_ALIASES:
        return {'ok': False, 'reason': 'model_alias_forbidden',
                'request_model': request_model, 'served_model': served_model,
                'response_model': response_model}
    if not request_model or request_model != served_model or request_model != response_model:
        return {'ok': False, 'reason': 'model_identity_mismatch',
                'request_model': request_model, 'served_model': served_model,
                'response_model': response_model}
    return {'ok': True, 'reason': 'match', 'request_model': request_model,
            'served_model': served_model, 'response_model': response_model}


def apply_host_provenance(proposal: Mapping[str, Any], *, model: str,
                          model_revision: str, model_image_id: str,
                          prompt_version: str, tool_version: str,
                          latency_ms: float) -> dict[str, Any]:
    """Overwrite VLM provenance with host-owned identity and timing."""
    result = dict(proposal)
    model_version = str(model or '')
    if model_revision:
        model_version = f'{model_version}@{model_revision}'
    if model_image_id:
        model_version = f'{model_version};image={model_image_id}'
    result['provenance'] = {
        'model_version': model_version,
        'prompt_version': str(prompt_version or PROMPT_VERSION),
        'tool_version': str(tool_version or TOOL_VERSION_DEFAULT),
        'reasoning_latency_ms': round(float(latency_ms), 3),
    }
    return result


class HttpVlmProducer:
    """Real LISN VLM producer over JSON HTTP (OpenAI-compatible by default).

    The request carries the instruction and the RGB bytes.  Credentials come
    only from an existing environment/secret reference.  Every request/response
    is retained in redacted form (hashes + bounded metadata, never the key).
    Any transport or schema failure raises; there is no fixture or silent
    fallback.
    """

    REQUIRED_FIELDS = ('controller_mode', 'controller_parameters', 'goal',
                       'semantic_entities')

    def __init__(self, endpoint: str | None = None, timeout_sec: float = 8.0,
                 config: ProducerConfig | None = None):
        if config is None:
            endpoint = str(endpoint or '').strip()
            if not endpoint:
                raise ValueError('producer endpoint is empty')
            config = ProducerConfig(endpoint=endpoint, timeout_sec=float(timeout_sec))
        self.config = config
        self.endpoint = config.endpoint
        self.timeout_sec = float(config.timeout_sec)
        self.trace: list[dict[str, Any]] = []
        self.last_response_meta: dict[str, Any] = {}
        # A 401/403 is a provider/credential pairing failure, not a transient
        # fault.  Latch it after the first response so no later tick can issue
        # another request until an explicit reset or a new safe configuration.
        self._auth_latched = False
        self.request_count = 0

    @property
    def auth_latched(self) -> bool:
        return self._auth_latched

    def _require_credential(self) -> None:
        """Fail closed before any Request when the provider needs a key.

        Named auth providers always require a key.  A generic endpoint may be
        unauthenticated only when the caller explicitly sets
        ``allow_unauthenticated``; there is no implicit fallback.
        """
        if self.config.api_key:
            return
        if self.config.provider in NAMED_AUTH_PROVIDERS:
            raise ProducerUnavailableError('credential_missing')
        if not self.config.allow_unauthenticated:
            raise ProducerUnavailableError('credential_missing')

    def _record_auth_failure(self, status: int, request_meta: dict[str, Any]) -> None:
        self._auth_latched = True
        self.trace.append({
            'request': request_meta,
            'response': {'status': int(status), 'error': 'http_auth_failure'},
        })

    def redacted_config(self) -> dict[str, Any]:
        return self.config.redacted()

    def redacted_trace(self) -> list[dict[str, Any]]:
        return list(self.trace)

    def list_models(self) -> list[str]:
        """Probe the documented model list endpoint when the provider exposes one."""
        import urllib.error
        import urllib.request

        if self._auth_latched:
            raise ProducerUnavailableError('auth_latched')
        if not self.config.models_endpoint:
            raise ProducerUnavailableError('endpoint_missing')
        self._require_credential()
        headers = {'Accept': 'application/json'}
        if self.config.api_key:
            headers[self.config.auth_header] = (
                f'{self.config.auth_scheme} {self.config.api_key}'.strip()
            )
        request = urllib.request.Request(self.config.models_endpoint, headers=headers, method='GET')
        self.request_count += 1
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            if status in (401, 403):
                self._record_auth_failure(
                    status, {'endpoint_label': safe_endpoint_label(self.config.models_endpoint)}
                )
                raise ProducerUnavailableError('auth_denied', status=status) from None
            raise ProducerUnavailableError('http_error', status=status) from None
        except TimeoutError:
            raise ProducerUnavailableError('timeout') from None
        except (urllib.error.URLError, OSError):
            raise ProducerUnavailableError('transport_unavailable') from None
        data = json.loads(body)
        rows = data.get('data') if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ValueError('model-list response lacks data[]')
        return [str(row.get('id')) for row in rows if isinstance(row, dict) and row.get('id')]

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        if not isinstance(text, str):
            return None
        start, end = text.find('{'), text.rfind('}')
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def _parse_response(self, data: Any) -> dict[str, Any]:
        if isinstance(data, dict) and all(name in data for name in self.REQUIRED_FIELDS):
            return data
        if isinstance(data, dict):
            choices = data.get('choices')
            if isinstance(choices, list) and choices:
                message = choices[0].get('message') if isinstance(choices[0], dict) else None
                if isinstance(message, dict):
                    parsed = self._extract_json(message.get('content') or '')
                    if parsed is not None and all(name in parsed for name in self.REQUIRED_FIELDS):
                        return parsed
                    tool_calls = message.get('tool_calls')
                    if isinstance(tool_calls, list):
                        for call in tool_calls:
                            function = call.get('function') if isinstance(call, dict) else None
                            arguments = function.get('arguments') if isinstance(function, dict) else None
                            parsed = self._extract_json(arguments or '')
                            if parsed is not None and all(name in parsed for name in self.REQUIRED_FIELDS):
                                return parsed
        raise ProducerUnavailableError('response_schema_invalid')

    def produce(self, *, instruction: str, image_bytes: bytes, frame_id: str,
                stamp_ns: int, image_encoding: str = 'rgb8', width: int = 0,
                height: int = 0, channels: int = 0,
                image_mime: str = '') -> dict[str, Any]:
        import base64
        import hashlib
        import urllib.error
        import urllib.request

        if self._auth_latched:
            raise ProducerUnavailableError('auth_latched')
        if self.config.enforce_model_identity and self.config.model in MODEL_ALIASES:
            raise ProducerUnavailableError('model_alias_forbidden')
        self._require_credential()
        encoded, mime = _encode_image_for_vlm(
            bytes(image_bytes), image_encoding=image_encoding, width=int(width),
            height=int(height), channels=int(channels), image_mime=image_mime,
        )
        image_b64 = base64.b64encode(encoded).decode('ascii')
        instruction_sha = hashlib.sha256(str(instruction).encode('utf-8')).hexdigest()
        source_sha = hashlib.sha256(bytes(image_bytes)).hexdigest()
        encoded_sha = hashlib.sha256(encoded).hexdigest()
        image_meta = {
            'source_encoding': str(image_encoding),
            'width': int(width),
            'height': int(height),
            'channels': int(channels),
            'encoded_mime': mime,
            'source_bytes': len(bytes(image_bytes)),
            'encoded_bytes': len(encoded),
            'source_sha256': source_sha,
            'encoded_sha256': encoded_sha,
        }
        prompt = (
            LISN_SCAN_GROUNDING_PROMPT if self.config.grounding_scan
            else LISN_GROUNDING_PROMPT if self.config.grounding_live
            else LISN_PROPOSAL_PROMPT
        )
        if self.config.grounding_live or self.config.grounding_scan:
            text = build_grounding_text(
                prompt, instruction=instruction, frame_id=frame_id,
                stamp_ns=stamp_ns, width=width, height=height,
            )
        else:
            text = f'{prompt}\nInstruction: {instruction}'
        if self.config.openai_compatible:
            payload_obj = {
                'model': self.config.model,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': text},
                        {'type': 'image_url', 'image_url': {
                            'url': f'data:{mime};base64,{image_b64}'}},
                    ],
                }],
                'temperature': 0,
                'max_tokens': int(self.config.max_tokens),
            }
            if self.config.json_response_format:
                payload_obj['response_format'] = {'type': 'json_object'}
        else:
            payload_obj = {
                'instruction': text,
                'rgb_base64': image_b64,
                'image': image_meta,
                'observation': {'frame_id': str(frame_id), 'stamp_ns': int(stamp_ns)},
            }
        if self.config.provider == 'none' or not self.config.endpoint:
            raise ProducerUnavailableError(
                'credential_missing' if self.config.provider == 'none' else 'endpoint_missing'
            )
        headers = {'Content-Type': 'application/json'}
        if self.config.api_key:
            value = f'{self.config.auth_scheme} {self.config.api_key}'.strip()
            headers[self.config.auth_header] = value
        elif self.config.endpoint.startswith('https://api.openai.com'):
            raise ProducerUnavailableError('credential_missing')
        payload = json.dumps(payload_obj).encode('utf-8')
        request = urllib.request.Request(self.endpoint, data=payload, headers=headers, method='POST')
        request_meta = {
            'instruction_sha256': instruction_sha,
            'image': image_meta,
            'frame_id': str(frame_id),
            'stamp_ns': int(stamp_ns),
            'config': self.redacted_config(),
        }
        self.request_count += 1
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                status = int(getattr(response, 'status', 200))
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            if status in (401, 403):
                self._record_auth_failure(status, request_meta)
                raise ProducerUnavailableError('auth_denied', status=status) from None
            raise ProducerUnavailableError('http_error', status=status) from None
        except TimeoutError:
            raise ProducerUnavailableError('timeout') from None
        except (urllib.error.URLError, OSError):
            raise ProducerUnavailableError('transport_unavailable') from None
        body_sha = hashlib.sha256(body).hexdigest()
        try:
            data = json.loads(body)
        except ValueError:
            self.trace.append({
                'request': {'instruction_sha256': instruction_sha, 'image': image_meta,
                            'frame_id': str(frame_id), 'stamp_ns': int(stamp_ns),
                            'config': self.redacted_config()},
                'response': {'status': status, 'body_sha256': body_sha,
                             'error': 'response_not_json'},
            })
            raise ProducerUnavailableError('response_not_json') from None
        choice = {}
        choices = data.get('choices') if isinstance(data, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
        elapsed = time.monotonic() - started
        response_model = str(data.get('model') or '') if isinstance(data, dict) else ''
        usage = data.get('usage') if isinstance(data, dict) else None
        self.last_response_meta = {
            'status': status,
            'response_id': str(data.get('id') or '') if isinstance(data, dict) else '',
            'response_model': response_model,
            'request_model': str(self.config.model),
            'finish_reason': str(choice.get('finish_reason') or ''),
            'usage': usage if isinstance(usage, dict) else {},
            'max_tokens': int(self.config.max_tokens),
            'latency_sec': round(elapsed, 3),
            'latency_ms': round(elapsed * 1000.0, 3),
        }
        try:
            proposal = self._parse_response(data)
        finally:
            self.trace.append({
                'request': {'instruction_sha256': instruction_sha, 'image': image_meta,
                            'frame_id': str(frame_id), 'stamp_ns': int(stamp_ns),
                            'config': self.redacted_config()},
                'response': {'status': status, 'body_sha256': body_sha},
            })
        parameters = proposal.get('controller_parameters')
        if not isinstance(parameters, dict) or 'max_linear_velocity_mps' not in parameters:
            raise ProducerUnavailableError('response_schema_invalid')
        proposal = apply_host_provenance(
            proposal,
            model=self.config.model,
            model_revision=self.config.model_revision,
            model_image_id=self.config.model_image_id,
            prompt_version=self.config.prompt_version,
            tool_version=self.config.tool_version,
            latency_ms=elapsed * 1000.0,
        )
        goal_trust = assess_goal_trust(
            proposal.get('goal'), target_frame=self.config.target_frame,
            known_projection=bool(self.config.assume_goal_trusted), semantic_match=False,
        )
        proposal['goal_trust'] = goal_trust
        validity = proposal.get('validity')
        if not isinstance(validity, Mapping) or not bool(validity.get('valid')):
            proposal['validity'] = {
                'valid': False,
                'reason': 'vlm_validity_false_or_missing',
            }
        elif not goal_trust['trusted']:
            proposal['validity'] = {
                'valid': False,
                'reason': f'goal_untrusted:{goal_trust["reason"]}',
            }
        identity = validate_model_identity(
            request_model=self.config.model,
            served_model=self.config.model,
            response_model=response_model,
        )
        proposal['model_identity'] = identity
        self.last_response_meta['model_identity_match'] = bool(identity['ok'])
        if self.config.enforce_model_identity and not identity['ok']:
            raise ProducerUnavailableError(identity['reason'])
        return proposal


def _subprocess_producer_worker(config_fields, request, conn):
    """Owned child process for one bounded HTTP request.

    The parent terminates this process on deadline; it never waits for a
    background socket to finish.  The credential travels only through the
    in-memory pipe, never argv, files or the environment.
    """
    producer = None
    try:
        config = ProducerConfig(**config_fields)
        producer = HttpVlmProducer(config=config)
        proposal = producer.produce(**request)
        payload = {
            'ok': True,
            'proposal': proposal,
            'meta': dict(producer.last_response_meta),
            'request_count': int(producer.request_count),
            'auth_latched': bool(producer.auth_latched),
        }
        conn.send_bytes(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    except BaseException as exc:
        payload = {
            'ok': False,
            'code': safe_error_code(exc),
            'status': getattr(exc, 'status', None),
            'request_count': int(getattr(producer, 'request_count', 0)),
            'auth_latched': bool(getattr(producer, 'auth_latched', False)),
        }
        try:
            conn.send_bytes(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        except BaseException:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class SubprocessHttpVlmProducer:
    """One bounded HTTP request per owned supervisor process group.

    The parent starts a short-lived supervisor with ``start_new_session=True``
    and supervises the whole group against an absolute monotonic deadline.  The
    supervisor owns the HTTP request and the response JSON decode; the parent
    can kill the group even while the supervisor is blocked in bootstrap or
    decode.  The credential travels only through the stdin pipe.
    """

    CLEANUP_BUDGET_SEC = 2.0
    MAX_REQUEST_BYTES = 8 * 1024 * 1024
    MAX_RESULT_BYTES = 4 * 1024 * 1024

    def __init__(self, config: ProducerConfig, *, supervisor_cmd=None):
        self.config = config
        deadline = float(config.deadline_sec)
        if not math.isfinite(deadline) or deadline <= 0:
            raise ProducerUnavailableError('invalid_deadline')
        self.deadline_sec = deadline
        self.request_count = 0
        self.auth_latched = False
        self.last_response_meta: dict[str, Any] = {}
        self.last_worker_alive = False
        self._terminal_latched = False
        self._terminal_code = ''
        self._supervisor_cmd = (
            list(supervisor_cmd) if supervisor_cmd is not None
            else [sys.executable, '-m', 'arena_bringup.lisn_producer_supervisor']
        )

    def _terminal(self, code: str) -> None:
        if not self._terminal_latched:
            self._terminal_latched = True
            self._terminal_code = code
        if code == 'auth_denied':
            self.auth_latched = True

    def _kill_group(self, proc) -> None:
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=self.CLEANUP_BUDGET_SEC / 2)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=self.CLEANUP_BUDGET_SEC / 2)
            except subprocess.TimeoutExpired:
                pass
        self.last_worker_alive = proc.poll() is None
        if self.last_worker_alive:
            self._terminal('abort_failed')
            raise ProducerUnavailableError('abort_failed')

    def _decode_result(self, raw: bytes, deadline_abs: float):
        try:
            result = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            self._terminal('response_not_json')
            raise ProducerUnavailableError('response_not_json') from None
        if time.monotonic() > deadline_abs:
            self._terminal('deadline_exceeded')
            raise ProducerUnavailableError('deadline_exceeded')
        self._validate_result(result)
        return result

    @staticmethod
    def _validate_result(result) -> None:
        if not isinstance(result, dict):
            raise ProducerUnavailableError('response_schema_invalid')
        allowed = {'ok', 'proposal', 'meta', 'request_count', 'auth_latched', 'code', 'status'}
        if set(result) - allowed:
            raise ProducerUnavailableError('response_schema_invalid')
        if not isinstance(result.get('ok'), bool):
            raise ProducerUnavailableError('response_schema_invalid')
        if result['ok']:
            proposal = result.get('proposal')
            if not isinstance(proposal, dict):
                raise ProducerUnavailableError('response_schema_invalid')
            if not all(name in proposal for name in HttpVlmProducer.REQUIRED_FIELDS):
                raise ProducerUnavailableError('response_schema_invalid')
            meta = result.get('meta')
            if not isinstance(meta, dict) or len(meta) > 16:
                raise ProducerUnavailableError('response_schema_invalid')
            for value in meta.values():
                if isinstance(value, str) and len(value) > 512:
                    raise ProducerUnavailableError('response_schema_invalid')
            entities = proposal.get('semantic_entities')
            if isinstance(entities, list) and len(entities) > 64:
                raise ProducerUnavailableError('response_schema_invalid')
        elif not isinstance(result.get('code'), str):
            raise ProducerUnavailableError('response_schema_invalid')

    def produce(self, *, instruction: str, image_bytes: bytes, frame_id: str,
                stamp_ns: int, image_encoding: str = 'rgb8', width: int = 0,
                height: int = 0, channels: int = 0, image_mime: str = '') -> dict[str, Any]:
        if self.auth_latched:
            raise ProducerUnavailableError('auth_latched')
        if self._terminal_latched:
            raise ProducerUnavailableError(self._terminal_code or 'deadline_exceeded')

        deadline_abs = time.monotonic() + self.deadline_sec
        request = {
            'instruction': str(instruction),
            'frame_id': str(frame_id),
            'stamp_ns': int(stamp_ns),
            'image_encoding': str(image_encoding),
            'width': int(width),
            'height': int(height),
            'channels': int(channels),
            'image_mime': str(image_mime),
            'image_bytes': base64.b64encode(bytes(image_bytes)).decode('ascii'),
        }
        payload = {'config': asdict(self.config), 'request': request}
        raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        if len(raw) > self.MAX_REQUEST_BYTES:
            self._terminal('oversize_request')
            raise ProducerUnavailableError('oversize_request')

        proc = None
        out = b''
        try:
            try:
                proc = subprocess.Popen(
                    self._supervisor_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except BaseException:
                self._terminal('transport_unavailable')
                raise ProducerUnavailableError('transport_unavailable') from None
            self.request_count += 1
            self.last_worker_alive = True
            remaining = deadline_abs - time.monotonic()
            if remaining <= 0:
                self._kill_group(proc)
                self._terminal('deadline_exceeded')
                raise ProducerUnavailableError('deadline_exceeded')
            try:
                out, _ = proc.communicate(raw, timeout=remaining)
            except subprocess.TimeoutExpired:
                self._kill_group(proc)
                self._terminal('deadline_exceeded')
                raise ProducerUnavailableError('deadline_exceeded') from None
            if proc.poll() is None:
                self._kill_group(proc)
            if time.monotonic() > deadline_abs:
                self._terminal('deadline_exceeded')
                raise ProducerUnavailableError('deadline_exceeded')
            if len(out) > self.MAX_RESULT_BYTES:
                self._terminal('oversize_result')
                raise ProducerUnavailableError('oversize_result')
            result = self._decode_result(out, deadline_abs)
        finally:
            self._kill_group(proc)
            self.last_worker_alive = False

        if not result.get('ok'):
            code = str(result.get('code') or 'unknown')
            if code not in ERROR_CODES:
                code = 'unknown'
            if code == 'auth_denied':
                self.auth_latched = True
            raise ProducerUnavailableError(code, status=result.get('status'))
        self.last_response_meta = dict(result.get('meta') or {})
        self.auth_latched = bool(result.get('auth_latched', False))
        return result['proposal']


def build_grounded_goal_msg(map_pose: Mapping[str, Any]):
    """Build the single published PoseStamped from the resolved map goal.

    The header frame and stamp are the immutable source bundle's map goal,
    never the wall clock or a re-bound fresh observation.
    """
    from geometry_msgs.msg import PoseStamped

    stamp_ns = int(map_pose['stamp_ns'])
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    msg.pose.position.x = float(map_pose['x'])
    msg.pose.position.y = float(map_pose['y'])
    msg.pose.position.z = 0.0
    msg.pose.orientation.w = 1.0
    return msg


def build_snapshot(*, current_parameters: Mapping[str, float], proposal: Mapping[str, Any],
                   observation_id: str, frame_id: str, stamp_ns: int,
                   target_frame: str, now_ns: int, max_age_ns: int,
                   control_path: str = 'lisn_style_social_mpc') -> dict[str, Any]:
    """Merge one real producer proposal with the observed frozen parameters."""
    goal_trust = proposal.get('goal_trust')
    if isinstance(goal_trust, Mapping) and not bool(goal_trust.get('trusted')):
        raise ProducerUnavailableError('goal_untrusted')
    validity = proposal.get('validity')
    if isinstance(validity, Mapping) and not bool(validity.get('valid')):
        raise ProducerUnavailableError('goal_untrusted')
    parameters = {
        name: float(current_parameters[name]) for name in ENGINEERING_LOGICAL_PARAMETERS
    }
    parameters['max_linear_velocity_mps'] = float(
        proposal['controller_parameters']['max_linear_velocity_mps']
    )
    goal = copy.deepcopy(proposal['goal'])
    if 'goal_trust' in proposal:
        # The VLM must never own the host observation timestamp.
        goal['stamp_ns'] = int(stamp_ns)
    return {
        'schema': 'arena.hierarchical_modulation',
        'schema_version': 1,
        'update_semantics': 'replace_all',
        'update_id': 'pending',
        'control_path': control_path,
        'generated_at_ns': int(now_ns),
        'valid_until_ns': int(now_ns) + int(max_age_ns),
        'source_observation': {
            'id': str(observation_id),
            'frame_id': str(frame_id),
            'stamp_ns': int(stamp_ns),
        },
        'target_frame': str(target_frame),
        'goal': goal,
        'semantic_entities': copy.deepcopy(proposal['semantic_entities']),
        'controller_mode': str(proposal['controller_mode']),
        'controller_parameters': parameters,
        'provenance': copy.deepcopy(proposal['provenance']),
        'reason': 'engineering producer proposal',
    }


@dataclass
class ProducerTick:
    """Live producer -> snapshot -> core transaction, independent of ROS."""

    producer: Any
    adapter: Any
    runner: TickRunner
    target_frame: str
    max_observation_age_ns: int
    control_path: str = 'lisn_style_social_mpc'

    def produce(self, *, instruction: str, image_bytes: bytes, frame_id: str,
                stamp_ns: int, image_encoding: str = 'rgb8', width: int = 0,
                height: int = 0, channels: int = 0, image_mime: str = '') -> dict[str, Any]:
        """One HTTP producer call.  The caller owns the worker/future."""
        return self.producer.produce(
            instruction=instruction,
            image_bytes=image_bytes,
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            image_encoding=image_encoding,
            width=width,
            height=height,
            channels=channels,
            image_mime=image_mime,
        )

    def commit(self, *, proposal: Mapping[str, Any], image_bytes: bytes,
               frame_id: str, stamp_ns: int, observation_id: str,
               received_monotonic_ns: int, now_monotonic_ns: int,
               now_ros_ns: int) -> dict[str, Any]:
        """Apply an already-produced proposal through the contract.

        This is the only place that reads/sets controller parameters.  It must
        run on a worker thread, never inside a ROS timer callback.
        """
        current = self.adapter.read(ENGINEERING_LOGICAL_PARAMETERS)
        snapshot = build_snapshot(
            current_parameters=current,
            proposal=proposal,
            observation_id=observation_id,
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            target_frame=self.target_frame,
            now_ns=now_ros_ns,
            max_age_ns=self.max_observation_age_ns,
            control_path=self.control_path,
        )
        return self.runner.tick(
            snapshot,
            received_monotonic_ns=received_monotonic_ns,
            now_monotonic_ns=now_monotonic_ns,
            now_ros_ns=now_ros_ns,
        )

    def process(self, *, instruction: str, image_bytes: bytes, frame_id: str,
                stamp_ns: int, observation_id: str,
                received_monotonic_ns: int, now_monotonic_ns: int,
                now_ros_ns: int, image_encoding: str = 'rgb8', width: int = 0,
                height: int = 0, channels: int = 0, image_mime: str = '') -> dict[str, Any]:
        """Convenience path: exactly one produce + one commit."""
        proposal = self.produce(
            instruction=instruction,
            image_bytes=image_bytes,
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            image_encoding=image_encoding,
            width=width,
            height=height,
            channels=channels,
            image_mime=image_mime,
        )
        return self.commit(
            proposal=proposal,
            image_bytes=image_bytes,
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            observation_id=observation_id,
            received_monotonic_ns=received_monotonic_ns,
            now_monotonic_ns=now_monotonic_ns,
            now_ros_ns=now_ros_ns,
        )


def run_producer_wiring_smoke(*, policy_path: Path, evidence_path: Path,
                              target_frame: str = 'map',
                              episode_id: str = 'producer-wiring-smoke') -> dict[str, Any]:
    """Exercise the real HTTP producer + engineering adapter with a local mock.

    The endpoint is an in-process mock, not a real LISN agent.  The receipt says
    so explicitly.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    response_payload = {
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
            body = json.dumps(response_payload).encode()
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
        policy = load_slow_loop_policy(Path(policy_path))
        endpoint = f'http://127.0.0.1:{server.server_port}/v1/propose'
        producer = HttpVlmProducer(config=ProducerConfig(endpoint=endpoint, timeout_sec=3.0, allow_unauthenticated=True))
        current = {
            'max_linear_velocity_mps': 0.35,
            'max_angular_velocity_rps': 1.4,
            'goal_weight': 1.0,
            'obstacle_weight': 15.0,
            'people_weight': 8.0,
        }
        adapter = EngineeringParameterAdapter(FixtureParameterAdapter(current))
        runner = TickRunner(
            policy=policy,
            episode_id=episode_id,
            epoch_monotonic_ns=1_000_000_000,
            target_frame=target_frame,
            adapter=adapter,
            evidence_path=Path(evidence_path),
            wiring_smoke=True,
        )
        rgb = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        proposal = producer.produce(
            instruction='walk forward', image_bytes=rgb,
            frame_id='Ai2_Bot2/head_camera', stamp_ns=1_400_000_000,
            image_encoding='rgb8', width=2, height=2, channels=3,
        )
        # The wiring smoke uses an in-process mock, not a real grounded goal.
        proposal['goal_trust'] = {'trusted': True, 'source_invalid': False,
                                  'reason': 'wiring_mock'}
        proposal['validity'] = {'valid': True, 'reason': 'wiring_mock'}
        snapshot = build_snapshot(
            current_parameters=adapter.read(ENGINEERING_LOGICAL_PARAMETERS),
            proposal=proposal,
            observation_id='producer-smoke-rgb-1',
            frame_id='Ai2_Bot2/head_camera',
            stamp_ns=1_400_000_000,
            target_frame=target_frame,
            now_ns=1_500_000_000,
            max_age_ns=policy.max_observation_age_ns,
        )
        scheduled = runner.scheduled_for_ns()
        record = runner.tick(
            snapshot,
            received_monotonic_ns=scheduled + 1 - 100_000_000,
            now_monotonic_ns=scheduled + 1,
            now_ros_ns=1_500_000_000,
        )
        receipt = {
            'schema': EVIDENCE_SCHEMA,
            'kind': 'PRODUCER_WIRING_SMOKE',
            'labels': {
                'wiring_smoke': True,
                'real_agent': False,
                'scientific_result': False,
                'runtime_executed': False,
                'producer_endpoint_kind': 'local_mock_http',
            },
            'policy_path': str(Path(policy_path)),
            'policy_sha256': policy.policy_sha256,
            'policy_runtime_ready': policy.runtime_ready,
            'policy_blockers': list(policy.runtime_blockers),
            'producer_request': {
                'instruction': 'walk forward',
                'frame_id': 'Ai2_Bot2/head_camera',
                'stamp_ns': 1_400_000_000,
                'rgb_bytes': len(rgb),
            },
            'applied_parameter': 'max_linear_velocity_mps',
            'requested_max_linear_velocity_mps': 0.30,
            'decision': record.get('decision'),
            'reason_codes': record.get('reason_codes'),
            'evidence_path': str(Path(evidence_path)),
            'runtime_executed': False,
        }
        Path(evidence_path).with_name('PRODUCER_WIRING_SMOKE.json').write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + '\n'
        )
        return receipt
    finally:
        server.shutdown()
        server.server_close()


class LisnSlowLoopNode:
    """Thin rclpy transport with a real RGB+instruction producer path.

    Production refuses while the policy is not runtime-ready.  ``wiring_live``
    is an explicit wiring-only mode: it requires a real endpoint, never falls
    back to a fixture, and labels every evidence record as wiring.  It never
    claims production or flips ``runtime_ready``.
    """

    def __init__(self, *, policy_path: Path, evidence_path: Path, controller_node: str,
                 proposal_topic: str, target_frame: str, episode_id: str,
                 engineering: bool = False, producer_endpoint: str = '',
                 rgb_topic: str = '/task_generator_node/Ai2_Bot2/head_camera/image',
                 instruction_topic: str = '/task_generator_node/vln_instruction',
                 model_status_topic: str = '',
                 eval_ready_topic: str = '/task_generator_node/eval_ready_status',
                 status_heartbeat_sec: float = 1.0,
                 producer_timeout_sec: float | None = None,
                 producer_deadline_sec: float | None = None,
                 wiring_live: bool = False,
                 grounding_live: bool = False,
                 grounding_scan: bool = False,
                 scan_topic: str = '/task_generator_node/Ai2_Bot2/laser_scan',
                 scan_max_age_sec: float = 0.6,
                 max_scan_range_m: float = 12.0,
                 depth_topic: str = '/task_generator_node/Ai2_Bot2/head_camera/depth',
                 camera_info_topic: str = '/task_generator_node/Ai2_Bot2/head_camera/camera_info',
                 grounded_goal_topic: str = '/task_generator_node/Ai2_Bot2/lisn_slow_loop/grounded_goal',
                 semantic_cost_topic: str = '/task_generator_node/Ai2_Bot2/lisn_semantic_cost',
                 semantic_cost_frame: str = 'Ai2_Bot2/odom',
                 semantic_ttl_sec: float = 12.0,
                 semantic_status_topic: str = '/lisn_semantic_layer_status',
                 min_confidence: float = 0.5,
                 latency_mode: str = 'disabled',
                 latency_ms: float = 0.0,
                 latency_profile: str = ''):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        self.policy = load_policy(Path(policy_path), engineering=engineering)
        self.blockers = production_blockers(self.policy)
        self.wiring_live = bool(wiring_live)
        self.engineering = bool(engineering)
        self._grounding_live = bool(grounding_live) or bool(grounding_scan)
        self._grounding_scan = bool(grounding_scan)
        self._scan_topic = str(scan_topic or '').strip()
        self._scan_max_age_ns = int(max(0.0, float(scan_max_age_sec)) * 1_000_000_000)
        self._max_scan_range_m = float(max_scan_range_m)
        self._depth_topic = str(depth_topic or '').strip()
        self._camera_info_topic = str(camera_info_topic or '').strip()
        self._grounded_goal_topic = str(grounded_goal_topic or '').strip()
        self._semantic_cost_topic = str(semantic_cost_topic or '').strip()
        self._semantic_cost_frame = str(semantic_cost_frame or '').strip()
        self._semantic_ttl_sec = max(0.0, float(semantic_ttl_sec))
        self._semantic_status_topic = str(semantic_status_topic or '').strip()
        self._semantic_pub = None
        self._semantic_status_sub = None
        self._semantic_status: dict[str, object] = {}
        self._semantic_receipt_written = False
        self._min_confidence = float(min_confidence)
        self._depth_buffer: dict[int, tuple[object, float]] = {}
        self._camera_info_buffer: dict[int, tuple[tuple[float, ...], float]] = {}
        self._scan_buffer: dict[int, tuple[dict[str, object], float]] = {}
        self._tf_buffer = None
        self._tf_listener = None
        self._grounded_goal_pub = None
        self._grounding_bundle_receipt_written = False
        self._vlm_trace_index = 0
        self._parameter_observation_written = False
        self._eval_ready_topic = str(eval_ready_topic or '').strip()
        self._status_heartbeat_sec = max(0.1, float(status_heartbeat_sec))
        self._session_id = None
        self._session_episode = None
        self._last_status_ns = 0
        self.producer_endpoint = str(producer_endpoint or '').strip()
        self._producer_config = None
        if not self.producer_endpoint:
            resolved = ProducerConfig.from_env()
            if resolved.provider != 'none':
                self._producer_config = resolved
                self.producer_endpoint = resolved.endpoint
        self.node = Node('lisn_slow_loop_bridge')
        self._latency_mode = str(latency_mode or 'disabled').strip().lower()
        self._latency_ms = float(latency_ms or 0.0)
        if latency_profile:
            payload = json.loads(Path(latency_profile).read_text(encoding='utf-8'))
            self._latency_mode = str(payload.get('mode') or self._latency_mode).strip().lower()
            self._latency_ms = float(payload.get('latency_ms', self._latency_ms))
        self._latency_gate = None
        self._latency_previous_response_sim_ns = None
        if self._latency_mode != 'disabled':
            from arena_bringup.paper_pipeline import LatencyCommitGate
            self._latency_gate = LatencyCommitGate(
                mode=self._latency_mode,
                latency_ms=self._latency_ms,
                now_sim_ns_fn=lambda: self.node.get_clock().now().nanoseconds,
            )
        self.refused = False
        if self.blockers and not self.wiring_live:
            self.refused = True
            for blocker in self.blockers:
                self.node.get_logger().error(f'[lisn_slow_loop] refused: {blocker}')
            return
        if self.wiring_live:
            for blocker in self.blockers:
                self.node.get_logger().warn(
                    f'[lisn_slow_loop] WIRING-LIVE bypassing runtime_ready: {blocker}'
                )
        if self.wiring_live and not self.producer_endpoint:
            self.refused = True
            self.node.get_logger().error(
                '[lisn_slow_loop] wiring-live requires a real producer endpoint or a resolved '
                'credential env; no fixture fallback'
            )
            return
        base_adapter = RosParameterAdapter(self.node, controller_node, self.policy)
        if self._grounding_scan:
            self.adapter = ProxGoalOnlyAdapter(base_adapter)
            self._controller_identity = 'FollowPath/ProxMpcController'
            self._parameter_policy = 'static_pinned_prox_no_dynamic_callback'
        elif engineering:
            self.adapter = EngineeringParameterAdapter(base_adapter)
            self._controller_identity = 'FollowPath/SocialMPCController'
            self._parameter_policy = 'engineering_hot_reload_max_linear_velocity'
        else:
            self.adapter = base_adapter
            self._controller_identity = 'FollowPath/SocialMPCController'
            self._parameter_policy = 'contract_all_parameters'
        self.runner = TickRunner(
            policy=self.policy,
            episode_id=episode_id,
            epoch_monotonic_ns=time.monotonic_ns(),
            target_frame=target_frame,
            adapter=self.adapter,
            evidence_path=Path(evidence_path),
            wiring_smoke=self.wiring_live,
        )
        self._proposal = None
        self._received_ns = 0
        self._image = None
        self._image_received_ns = 0
        self._instruction = ''
        self._pending = None
        self._pending_meta = None
        self._pending_started_ns = 0
        effective_deadline = (
            float(producer_deadline_sec)
            if producer_deadline_sec is not None
            else float(getattr(self._producer_config, 'deadline_sec', 120.0))
        )
        # The child producer enforces the exact budget; this timer is a
        # safety net five seconds later, so it never races a healthy abort.
        self._producer_deadline_ns = int((effective_deadline + 5.0) * 1_000_000_000)
        self._commit_future = None
        self._commit_meta = None
        self._controller_verified = False
        self._auth_gate = ProducerAuthGate()
        self._executor = None
        self._commit_executor = None
        self._status_pub = self.node.create_publisher(
            String, model_status_topic or (proposal_topic + '/status'), 10
        )
        self._status_pub.publish(String(data=json.dumps({
            'status': 'ready', 'source': 'lisn_slow_loop', 'engineering': self.engineering
        })))
        # The task generator publishes the per-episode identity on
        # /eval_ready_status.  Without it no heartbeat can be session-bound and
        # the model_reachable barrier can never pass.
        if self._eval_ready_topic:
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

            # The task generator latches eval_ready with TRANSIENT_LOCAL.  A
            # volatile subscriber that connects after the last stage change
            # never sees the session identity, so match the publisher's
            # durability and depth.
            ready_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._eval_ready_sub = self.node.create_subscription(
                String, self._eval_ready_topic, self._on_eval_ready, ready_qos
            )
        if self.producer_endpoint:
            from concurrent.futures import ThreadPoolExecutor
            from sensor_msgs.msg import Image

            if self._producer_config is not None:
                config = self._producer_config
                if self._grounding_live and not config.grounding_live:
                    config = replace(config, grounding_live=True)
                if self._grounding_scan and not config.grounding_scan:
                    config = replace(config, grounding_scan=True)
                if producer_timeout_sec is not None:
                    config = replace(config, timeout_sec=float(producer_timeout_sec))
                if producer_deadline_sec is not None:
                    config = replace(config, deadline_sec=float(producer_deadline_sec))
                self.producer = SubprocessHttpVlmProducer(config=config)
            else:
                config = ProducerConfig.local(
                    self.producer_endpoint,
                    timeout_sec=float(producer_timeout_sec or 8.0),
                    deadline_sec=float(producer_deadline_sec or 120.0),
                    grounding_live=self._grounding_live,
                    grounding_scan=self._grounding_scan,
                )
                self.producer = SubprocessHttpVlmProducer(config=config)
            self.producer_tick = ProducerTick(
                producer=self.producer,
                adapter=self.adapter,
                runner=self.runner,
                target_frame=target_frame,
                max_observation_age_ns=self.policy.max_observation_age_ns,
                control_path=self.policy.active_control_path,
            )
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._commit_executor = ThreadPoolExecutor(max_workers=1)
            self._image_sub = self.node.create_subscription(
                Image, rgb_topic, self._on_image, 10
            )
            self._instruction_sub = self.node.create_subscription(
                String, instruction_topic, self._on_instruction, 10
            )
            if self._grounding_live:
                import tf2_ros
                from rclpy.duration import Duration
                from geometry_msgs.msg import PoseStamped

                self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=15.0))
                self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)
                self._grounded_goal_pub = self.node.create_publisher(
                    PoseStamped, self._grounded_goal_topic, 10
                )
                if self._grounding_scan:
                    # A-track: RGB bearing + same-source real 2D LiDAR range.
                    # CameraInfo is calibration only; no depth image and no
                    # semantic grid are consumed.
                    from sensor_msgs.msg import CameraInfo, LaserScan

                    self._scan_sub = self.node.create_subscription(
                        LaserScan, self._scan_topic, self._on_scan, 10
                    )
                    self._camera_info_sub = self.node.create_subscription(
                        CameraInfo, self._camera_info_topic, self._on_camera_info, 10
                    )
                else:
                    from sensor_msgs.msg import CameraInfo

                    self._depth_sub = self.node.create_subscription(
                        Image, self._depth_topic, self._on_depth, 10
                    )
                    self._camera_info_sub = self.node.create_subscription(
                        CameraInfo, self._camera_info_topic, self._on_camera_info, 10
                    )
                    from nav_msgs.msg import OccupancyGrid

                    self._semantic_pub = self.node.create_publisher(
                        OccupancyGrid, self._semantic_cost_topic, 10
                    )
                    if self._semantic_status_topic:
                        self._semantic_status_sub = self.node.create_subscription(
                            String, self._semantic_status_topic,
                            self._on_semantic_status, 10,
                        )
            self.node.get_logger().warn(
                f'[lisn_slow_loop] producer mode endpoint={self.producer_endpoint} '
                f'rgb={rgb_topic} instruction={instruction_topic} wiring_live={self.wiring_live}'
            )
        else:
            self.producer = None
            self._subscription = self.node.create_subscription(
                String, proposal_topic, self._on_proposal, 10
            )
        self._timer = self.node.create_timer(0.2, self._tick)
        self._latency_timer = None
        if self._latency_gate is not None:
            self._latency_timer = self.node.create_timer(0.1, self._tick_latency_commits)

    def _on_eval_ready(self, msg):
        session_id, episode = parse_eval_ready_payload(str(getattr(msg, 'data', '') or ''))
        if session_id is None:
            return
        if (session_id, episode) != (self._session_id, self._session_episode):
            self._session_id, self._session_episode = session_id, episode
            self._last_status_ns = 0

    def _publish_heartbeat(self, now_ns):
        from std_msgs.msg import String

        if self._session_id is None:
            return
        period_ns = self._status_heartbeat_sec * 1e9
        if self._last_status_ns and now_ns - self._last_status_ns < period_ns:
            return
        self._last_status_ns = now_ns
        self._status_pub.publish(String(data=json.dumps(session_status_payload(
            session_id=self._session_id,
            episode=self._session_episode,
            status='ready',
            engineering=self.engineering,
            producer_mode=bool(self.producer),
            wiring_live=self.wiring_live,
        ))))

    def _on_proposal(self, msg):
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if isinstance(payload, dict):
            self._proposal = payload
            self._received_ns = time.monotonic_ns()

    def _on_image(self, msg):
        self._image = msg
        self._image_received_ns = time.monotonic_ns()

    def _on_depth(self, msg):
        if not self._grounding_live:
            return
        try:
            depth = self._depth_metric(msg)
        except (TypeError, ValueError, ImportError):
            return
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self._depth_buffer[stamp] = (depth, time.monotonic())
        if len(self._depth_buffer) > 50:
            oldest = sorted(self._depth_buffer)[0]
            self._depth_buffer.pop(oldest, None)

    def _on_camera_info(self, msg):
        if not self._grounding_live:
            return
        try:
            k = tuple(float(value) for value in msg.k)
        except (TypeError, ValueError):
            return
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self._camera_info_buffer[stamp] = (k, time.monotonic())
        if len(self._camera_info_buffer) > 50:
            oldest = sorted(self._camera_info_buffer)[0]
            self._camera_info_buffer.pop(oldest, None)

    def _on_scan(self, msg):
        if not self._grounding_scan:
            return
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self._scan_buffer[stamp] = ({
            'stamp_ns': stamp,
            'frame_id': str(msg.header.frame_id),
            'ranges': tuple(float(value) for value in msg.ranges),
            'angle_min': float(msg.angle_min),
            'angle_increment': float(msg.angle_increment),
            'range_min': float(msg.range_min),
            'range_max': float(msg.range_max),
        }, time.monotonic())
        if len(self._scan_buffer) > 50:
            oldest = sorted(self._scan_buffer)[0]
            self._scan_buffer.pop(oldest, None)

    def _on_semantic_status(self, msg):
        try:
            payload = json.loads(str(getattr(msg, 'data', '') or ''))
        except (TypeError, ValueError):
            payload = {'raw': str(getattr(msg, 'data', '') or '')[:256]}
        if not isinstance(payload, dict):
            payload = {'raw': payload}
        self._semantic_status = payload
        self._semantic_status_count = getattr(self, '_semantic_status_count', 0) + 1
        if self._semantic_status_count % 10 == 1:
            history = Path(self.runner.evidence_path).with_name(
                'SEMANTIC_CONSUMER_HISTORY.jsonl'
            )
            history.parent.mkdir(parents=True, exist_ok=True)
            with history.open('a') as stream:
                stream.write(json.dumps({
                    'monotonic_ns': time.monotonic_ns(),
                    'status': payload,
                }, sort_keys=True) + '\n')
        if self._semantic_status_count % 50 == 1:
            self._write_semantic_consumer_receipt()

    def _write_semantic_consumer_receipt(self):
        path = Path(self.runner.evidence_path).with_name('SEMANTIC_CONSUMER.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'schema': 'arena.lisn.semantic_consumer.v1',
            'consumer': 'arena_semantic_costmap_plugin::SemanticLayer',
            'semantic_cost_topic': self._semantic_cost_topic,
            'semantic_status_topic': self._semantic_status_topic,
            'status': self._semantic_status,
            'observed_monotonic_ns': time.monotonic_ns(),
        }, indent=2, sort_keys=True) + '\n')

    def _map_entities_to_semantic_frame(self, entities, stamp_ns):
        """Transform projected map entities into the costmap global frame."""
        from rclpy.time import Time

        if self._tf_buffer is None or not self._semantic_cost_frame:
            raise ProducerUnavailableError('semantic_tf_missing')
        try:
            transform = self._tf_buffer.lookup_transform(
                self._semantic_cost_frame, 'map', Time(nanoseconds=int(stamp_ns))
            )
        except Exception:
            raise ProducerUnavailableError('semantic_tf_missing') from None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        x, y, z, w = (float(rotation.x), float(rotation.y),
                      float(rotation.z), float(rotation.w))
        cos_yaw = 1 - 2 * (y * y + z * z)
        sin_yaw = 2 * (x * y - z * w)
        moved = []
        for entity in entities:
            mx = float(entity['map_x'])
            my = float(entity['map_y'])
            ox = cos_yaw * mx - sin_yaw * my + float(translation.x)
            oy = sin_yaw * mx + cos_yaw * my + float(translation.y)
            copy = dict(entity)
            copy['map_x'] = ox
            copy['map_y'] = oy
            moved.append(copy)
        return moved

    def _resolve_semantic_entities(self, raw_entities, bundle):
        """Project trusted VLM entities and build the LISN-only cost grid."""
        from arena_bringup.lisn_semantic import (
            SemanticError,
            build_semantic_grid,
            contract_entity,
            project_entity,
            validate_entity,
        )

        receipt = {
            'schema': 'arena.lisn.semantic_entities.v1',
            'source_stamp_ns': int(bundle['stamp_ns']),
            'source_frame_id': str(bundle['frame_id']),
            'semantic_cost_topic': self._semantic_cost_topic,
            'cost_frame': self._semantic_cost_frame,
            'goal_chain_independent': True,
            'raw_count': len(raw_entities) if isinstance(raw_entities, list) else 0,
            'projected_count': 0,
            'rejected': [],
            'raw_entities': raw_entities if isinstance(raw_entities, list) else [],
            'projected': [],
            'grid': None,
        }
        if not isinstance(raw_entities, list) or not raw_entities:
            receipt['noop'] = True
            self._write_semantic_entities_receipt(receipt)
            return [], None, receipt
        projected = []
        for index, entity in enumerate(raw_entities):
            try:
                validated = validate_entity(entity)
                projected.append(project_entity(
                    validated,
                    width=int(bundle['width']),
                    height=int(bundle['height']),
                    depth_image=bundle['depth'],
                    camera_info_k=bundle['k'],
                    tf_map_camera=bundle['tf'],
                    expected_frame_id=str(bundle['frame_id']),
                    source_stamp_ns=int(bundle['stamp_ns']),
                    min_confidence=self._min_confidence,
                ))
            except SemanticError as exc:
                receipt['rejected'].append({
                    'index': index,
                    'reason': str(exc),
                    'class_name': entity.get('class_name') if isinstance(entity, dict) else None,
                })
        receipt['projected_count'] = len(projected)
        receipt['projected'] = [{
            'class_name': entity['class_name'],
            'attribute': entity['attribute'],
            'instance_id': entity['instance_id'],
            'map_x': entity['map_x'],
            'map_y': entity['map_y'],
            'confidence': entity['confidence'],
            'cost_value': entity['cost_value'],
            'inflation_radius_m': entity['inflation_radius_m'],
            'decay_rate': entity['decay_rate'],
            'geometry_ref': entity['geometry_ref'],
        } for entity in projected]
        if not projected:
            receipt['noop'] = True
            self._write_semantic_entities_receipt(receipt)
            return [], None, receipt
        try:
            framed = self._map_entities_to_semantic_frame(projected, int(bundle['stamp_ns']))
            grid = build_semantic_grid(
                framed, frame_id=self._semantic_cost_frame,
                stamp_ns=int(bundle['stamp_ns']),
            )
        except (ProducerUnavailableError, SemanticError) as exc:
            receipt['grid_error'] = str(exc)
            self._write_semantic_entities_receipt(receipt)
            return [contract_entity(entity) for entity in projected], None, receipt
        receipt['grid'] = {
            'frame_id': grid['frame_id'],
            'stamp_ns': grid['stamp_ns'],
            'resolution': grid['resolution'],
            'width': grid['width'],
            'height': grid['height'],
            'origin': grid['origin'],
            'nonzero_cells': grid['nonzero_cells'],
            'max_cost': grid['max_cost'],
            'data': grid['data'],
        }
        self._write_semantic_entities_receipt(receipt)
        return [contract_entity(entity) for entity in projected], grid, receipt

    def _write_semantic_entities_receipt(self, receipt):
        path = Path(self.runner.evidence_path).with_name('SEMANTIC_ENTITIES.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')

    def _publish_semantic_grid(self, grid):
        if self._semantic_pub is None or grid is None:
            return
        from nav_msgs.msg import OccupancyGrid

        message = OccupancyGrid()
        message.header.frame_id = str(grid['frame_id'])
        message.header.stamp.sec = int(grid['stamp_ns']) // 1_000_000_000
        message.header.stamp.nanosec = int(grid['stamp_ns']) % 1_000_000_000
        message.info.resolution = float(grid['resolution'])
        message.info.width = int(grid['width'])
        message.info.height = int(grid['height'])
        message.info.origin.position.x = float(grid['origin']['x'])
        message.info.origin.position.y = float(grid['origin']['y'])
        message.info.origin.orientation.w = 1.0
        message.data = [int(value) for value in grid['data']]
        self._semantic_pub.publish(message)

    @staticmethod
    def _depth_metric(msg):
        import numpy as np

        height, width = int(msg.height), int(msg.width)
        encoding = str(getattr(msg, 'encoding', '32FC1') or '32FC1').lower()
        if encoding == '32fc1':
            dtype = np.float32
        elif encoding == '64fc1':
            dtype = np.float64
        elif encoding in {'16uc1', 'mono16'}:
            dtype = np.uint16
        else:
            raise ValueError(f'unsupported depth encoding {encoding}')
        itemsize = np.dtype(dtype).itemsize
        row_bytes = int(getattr(msg, 'step', 0) or 0) or width * itemsize
        if row_bytes < width * itemsize or row_bytes % itemsize != 0:
            raise ValueError('invalid depth row stride')
        raw = np.frombuffer(msg.data, dtype=dtype)
        if raw.size < height * (row_bytes // itemsize):
            raise ValueError('depth buffer shorter than height*stride')
        rows = raw.reshape(height, row_bytes // itemsize)[:, :width]
        if bool(getattr(msg, 'is_bigendian', 0)):
            rows = rows.byteswap()
        depth = np.array(rows, copy=True)
        if encoding in {'16uc1', 'mono16'}:
            depth = depth.astype(np.float32) / 1000.0
        return depth

    def _lookup_tf_matrix(self, target: str, source: str, stamp_ns: int) -> list[list[float]]:
        from rclpy.time import Time

        transform = self._tf_buffer.lookup_transform(
            target, source, Time(nanoseconds=int(stamp_ns))
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        x, y, z, w = (float(rotation.x), float(rotation.y),
                      float(rotation.z), float(rotation.w))
        return [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(translation.x)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(translation.y)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(translation.z)],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _freeze_scan_bundle(self, image_msg, frame_id, stamp_ns):
        """Freeze RGB K + nearest real LaserScan + source-time TF before the VLM call.

        The scan is chosen once, nearest to the RGB source stamp within a
        bounded age.  After the VLM returns, only this frozen bundle is used;
        no live scan buffer or latest-TF lookup is allowed.
        """
        if not self._scan_topic:
            raise ProducerUnavailableError('grounding_scan_topic_missing')
        if self._tf_buffer is None:
            raise ProducerUnavailableError('grounding_tf_missing')
        if int(stamp_ns) not in self._camera_info_buffer:
            raise ProducerUnavailableError('grounding_scan_camera_info_missing')
        if not self._scan_buffer:
            raise ProducerUnavailableError('grounding_scan_sync_missing')
        source_stamp = int(stamp_ns)
        scan_stamp = min(self._scan_buffer, key=lambda value: abs(value - source_stamp))
        scan_age_ns = abs(int(scan_stamp) - source_stamp)
        if scan_age_ns > self._scan_max_age_ns:
            raise ProducerUnavailableError('grounding_scan_age_exceeded')
        scan = dict(self._scan_buffer[scan_stamp][0])
        scan['ranges'] = tuple(scan['ranges'])
        k = tuple(self._camera_info_buffer[source_stamp][0])
        try:
            tf_camera = self._lookup_tf_matrix('map', str(frame_id), source_stamp)
            tf_scan = self._lookup_tf_matrix('map', str(scan['frame_id']), int(scan_stamp))
        except Exception:
            raise ProducerUnavailableError('grounding_scan_tf_missing') from None
        if not self._grounding_bundle_receipt_written:
            receipt = {
                'schema': 'arena.lisn.grounding_first_bundle.v1',
                'grounding_mode': 'scan',
                'stamp_ns': source_stamp,
                'frame_id': str(frame_id),
                'width': int(image_msg.width),
                'height': int(image_msg.height),
                'image_sha256': hashlib.sha256(bytes(image_msg.data)).hexdigest(),
                'camera_info_k': list(k),
                'tf_map_camera': tf_camera,
                'scan': {
                    'topic': self._scan_topic,
                    'stamp_ns': int(scan_stamp),
                    'age_ns': int(scan_age_ns),
                    'frame_id': scan['frame_id'],
                    'beam_count': len(scan['ranges']),
                    'angle_min_rad': scan['angle_min'],
                    'angle_increment_rad': scan['angle_increment'],
                    'range_min_m': scan['range_min'],
                    'range_max_m': scan['range_max'],
                    'finite_count': sum(1 for value in scan['ranges'] if math.isfinite(value)),
                },
                'tf_map_scan': tf_scan,
            }
            path = Path(self.runner.evidence_path).with_name('GROUNDING_FIRST_BUNDLE.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
            self._grounding_bundle_receipt_written = True
        return {
            'stamp_ns': source_stamp, 'frame_id': str(frame_id),
            'width': int(image_msg.width), 'height': int(image_msg.height),
            'k': k, 'tf': tf_camera,
            'scan': scan, 'tf_map_scan': tf_scan,
            'scan_stamp_ns': int(scan_stamp), 'scan_age_ns': int(scan_age_ns),
            'grounding_mode': 'scan',
        }

    def _freeze_grounding_bundle(self, image_msg, frame_id, stamp_ns):
        """Freeze the exact source-time RGB/depth/K/TF bundle before the VLM call."""
        if not self._grounding_live:
            return None
        if self._grounding_scan:
            return self._freeze_scan_bundle(image_msg, frame_id, stamp_ns)
        if self._tf_buffer is None:
            raise ProducerUnavailableError('grounding_tf_missing')
        if stamp_ns not in self._depth_buffer or stamp_ns not in self._camera_info_buffer:
            raise ProducerUnavailableError('grounding_sync_missing')
        import numpy as np

        depth = np.array(self._depth_buffer[stamp_ns][0], copy=True)
        k = tuple(self._camera_info_buffer[stamp_ns][0])
        try:
            from rclpy.time import Time
            transform = self._tf_buffer.lookup_transform(
                'map', str(frame_id), Time(nanoseconds=int(stamp_ns))
            )
        except Exception:
            raise ProducerUnavailableError('grounding_tf_missing') from None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        x, y, z, w = float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
        tf = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(translation.x)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(translation.y)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(translation.z)],
            [0.0, 0.0, 0.0, 1.0],
        ]
        if not self._grounding_bundle_receipt_written:
            import numpy as np

            finite = np.isfinite(depth)
            finite_values = depth[finite]
            receipt = {
                'schema': 'arena.lisn.grounding_first_bundle.v1',
                'stamp_ns': int(stamp_ns),
                'frame_id': str(frame_id),
                'width': int(image_msg.width),
                'height': int(image_msg.height),
                'image_sha256': hashlib.sha256(bytes(image_msg.data)).hexdigest(),
                'depth_shape': list(depth.shape),
                'depth_finite_count': int(np.count_nonzero(finite)),
                'depth_min': float(finite_values.min()) if finite_values.size else None,
                'depth_max': float(finite_values.max()) if finite_values.size else None,
                'camera_info_k': list(k),
                'tf_map_camera': tf,
            }
            path = Path(self.runner.evidence_path).with_name('GROUNDING_FIRST_BUNDLE.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
            self._grounding_bundle_receipt_written = True
        return {
            'stamp_ns': int(stamp_ns), 'frame_id': str(frame_id),
            'width': int(image_msg.width), 'height': int(image_msg.height),
            'depth': depth, 'k': k, 'tf': tf,
        }

    def _resolve_scan_grounding(self, proposal, bundle):
        """Resolve pixel/bearing with the frozen real LaserScan bundle only."""
        from arena_bringup.lisn_grounding import GroundingError, resolve_scan_grounded_goal

        if bundle is None:
            raise ProducerUnavailableError('grounding_sync_missing')
        scan = bundle['scan']
        try:
            result = resolve_scan_grounded_goal(
                goal=proposal.get('goal'), width=bundle['width'], height=bundle['height'],
                camera_info_k=bundle['k'], tf_map_camera=bundle['tf'],
                scan_ranges=scan['ranges'], scan_angle_min=scan['angle_min'],
                scan_angle_increment=scan['angle_increment'],
                scan_range_min=scan['range_min'], scan_range_max=scan['range_max'],
                tf_map_scan=bundle['tf_map_scan'],
                expected_camera_frame=bundle['frame_id'],
                expected_scan_frame=scan['frame_id'],
                source_stamp_ns=bundle['stamp_ns'],
                scan_stamp_ns=bundle['scan_stamp_ns'],
                scan_age_ns=bundle['scan_age_ns'],
                max_scan_age_ns=self._scan_max_age_ns,
                min_confidence=self._min_confidence,
                max_scan_range_m=self._max_scan_range_m,
            )
        except GroundingError as exc:
            goal = proposal.get('goal') if isinstance(proposal.get('goal'), Mapping) else {}
            self.node.get_logger().error(
                f'[lisn_slow_loop] scan grounding rejected: {exc}; '
                f'goal_kind={goal.get("kind")!r} units={goal.get("units")!r} '
                f'goal_frame={goal.get("frame_id")!r} '
                f'confidence={goal.get("confidence")!r} '
                f'image_frame={bundle.get("frame_id")!r} '
                f'scan_frame={scan.get("frame_id")!r} '
                f'scan_age_ns={bundle.get("scan_age_ns")!r}'
            )
            proposal['goal_trust'] = {
                'trusted': False, 'source_invalid': True, 'reason': str(exc),
            }
            rejection = {
                'schema': 'arena.lisn.grounding_first_rejection.v1',
                'grounding_mode': 'scan',
                'reason': str(exc),
                'goal': {
                    'kind': goal.get('kind'),
                    'units': goal.get('units'),
                    'frame_id': goal.get('frame_id'),
                    'confidence': goal.get('confidence'),
                    'value': goal.get('value'),
                    'stamp_ns': goal.get('stamp_ns'),
                },
                'bundle': {
                    'stamp_ns': bundle.get('stamp_ns'),
                    'frame_id': bundle.get('frame_id'),
                    'width': bundle.get('width'),
                    'height': bundle.get('height'),
                    'scan_stamp_ns': bundle.get('scan_stamp_ns'),
                    'scan_age_ns': bundle.get('scan_age_ns'),
                    'scan_frame_id': scan.get('frame_id'),
                    'scan_beam_count': len(scan.get('ranges', ())),
                    'scan_range_min_m': scan.get('range_min'),
                    'scan_range_max_m': scan.get('range_max'),
                },
            }
            path = Path(self.runner.evidence_path).with_name('GROUNDING_FIRST_REJECTION.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rejection, indent=2, sort_keys=True) + '\n')
            proposal['validity'] = {'valid': False, 'reason': f'grounding_rejected:{exc}'}
            raise ProducerUnavailableError('grounding_rejected') from None
        map_pose = result['map_pose']
        proposal['goal'] = {
            'kind': 'map_pose',
            'frame_id': 'map',
            'stamp_ns': int(bundle['stamp_ns']),
            'value': {'x': map_pose['x'], 'y': map_pose['y'], 'yaw': 0.0},
        }
        proposal['goal_trust'] = {
            'trusted': True, 'source_invalid': False, 'reason': 'scan_grounded',
            'point': result['point'], 'scan': result['scan'],
        }
        proposal['validity'] = {'valid': True, 'reason': 'scan_grounded'}
        proposal['_grounded_map_pose'] = {
            'x': float(map_pose['x']), 'y': float(map_pose['y']),
            'stamp_ns': int(bundle['stamp_ns']),
        }
        # A first version is goal/bearing only; no semantic grid is consumed.
        proposal['semantic_entities'] = []
        return proposal

    def _resolve_grounding(self, proposal, bundle):
        """Project with the frozen bundle only; no live buffer or latest TF lookup."""
        from arena_bringup.lisn_grounding import GroundingError, resolve_grounded_goal

        if not self._grounding_live:
            return proposal
        if self._grounding_scan:
            return self._resolve_scan_grounding(proposal, bundle)
        if bundle is None:
            raise ProducerUnavailableError('grounding_sync_missing')
        try:
            result = resolve_grounded_goal(
                goal=proposal.get('goal'), width=bundle['width'], height=bundle['height'],
                depth_image=bundle['depth'], camera_info_k=bundle['k'],
                tf_map_camera=bundle['tf'], expected_frame_id=bundle['frame_id'],
                source_stamp_ns=bundle['stamp_ns'], min_confidence=self._min_confidence,
            )
        except GroundingError as exc:
            goal = proposal.get('goal') if isinstance(proposal.get('goal'), Mapping) else {}
            self.node.get_logger().error(
                f'[lisn_slow_loop] grounding rejected: {exc}; '
                f'goal_kind={goal.get("kind")!r} units={goal.get("units")!r} '
                f'goal_frame={goal.get("frame_id")!r} '
                f'confidence={goal.get("confidence")!r} '
                f'image_frame={bundle.get("frame_id")!r}'
            )
            proposal['goal_trust'] = {
                'trusted': False, 'source_invalid': True, 'reason': str(exc),
            }
            rejection = {
                'schema': 'arena.lisn.grounding_first_rejection.v1',
                'reason': str(exc),
                'goal': {
                    'kind': goal.get('kind'),
                    'units': goal.get('units'),
                    'frame_id': goal.get('frame_id'),
                    'confidence': goal.get('confidence'),
                    'value': goal.get('value'),
                    'stamp_ns': goal.get('stamp_ns'),
                },
                'bundle': {
                    'stamp_ns': bundle.get('stamp_ns'),
                    'frame_id': bundle.get('frame_id'),
                    'width': bundle.get('width'),
                    'height': bundle.get('height'),
                },
            }
            path = Path(self.runner.evidence_path).with_name('GROUNDING_FIRST_REJECTION.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rejection, indent=2, sort_keys=True) + '\n')
            proposal['validity'] = {'valid': False, 'reason': f'grounding_rejected:{exc}'}
            raise ProducerUnavailableError('grounding_rejected') from None
        map_pose = result['map_pose']
        proposal['goal'] = {
            'kind': 'map_pose',
            'frame_id': 'map',
            'stamp_ns': int(bundle['stamp_ns']),
            'value': {'x': map_pose['x'], 'y': map_pose['y'], 'yaw': 0.0},
        }
        proposal['goal_trust'] = {
            'trusted': True, 'source_invalid': False, 'reason': 'grounded',
            'point': result['point'],
        }
        proposal['validity'] = {'valid': True, 'reason': 'grounded'}
        proposal['_grounded_map_pose'] = {
            'x': float(map_pose['x']), 'y': float(map_pose['y']),
            'stamp_ns': int(bundle['stamp_ns']),
        }
        # Semantic cost is an independent guidance chain: it is published from
        # the same frozen bundle even if the parameter apply later fails, and
        # it never touches the goal identity or the real HuNav people tracks.
        try:
            entities, grid, _receipt = self._resolve_semantic_entities(
                proposal.get('semantic_entities'), bundle
            )
            proposal['semantic_entities'] = entities
            self._publish_semantic_grid(grid)
        except Exception as exc:  # noqa: BLE001 - never break the goal chain
            proposal['semantic_entities'] = []
            self.node.get_logger().error(
                f'[lisn_slow_loop] semantic chain failed closed; '
                f'code={safe_error_code(exc)}; category={safe_error_category(exc)}'
            )
        return proposal

    def _publish_grounded_goal(self, map_pose):
        if self._grounded_goal_pub is None or map_pose is None:
            return
        msg = build_grounded_goal_msg(map_pose)
        self._grounded_goal_pub.publish(msg)

    def _on_instruction(self, msg):
        text = str(getattr(msg, 'data', '') or '').strip()
        if text:
            self._instruction = text

    @staticmethod
    def _image_channels(encoding: str) -> int:
        encoding = str(encoding or '').strip().lower()
        if encoding in {'rgb8', 'bgr8'}:
            return 3
        if encoding in {'mono8', '8uc1'}:
            return 1
        return 0

    @staticmethod
    def _observation_identity(image_msg, instruction: str):
        import hashlib

        rgb_sha = hashlib.sha256(bytes(image_msg.data)).hexdigest()
        content = canonical_sha256({'rgb_sha256': rgb_sha, 'instruction': instruction})
        stamp = image_msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        observation_id = f'{stamp_ns}-{rgb_sha[:16]}'
        return observation_id, str(image_msg.header.frame_id), stamp_ns, content

    def _publish_status(self, record):
        from std_msgs.msg import String

        extra = {
            'decision': record.get('decision'),
            'reason_codes': record.get('reason_codes'),
            'tick_index': record.get('tick_index'),
            'wiring_live': self.wiring_live,
            'producer_mode': bool(self.producer),
        }
        if self._session_id is not None:
            payload = session_status_payload(
                session_id=self._session_id,
                episode=self._session_episode,
                status='running',
                engineering=self.engineering,
                **extra,
            )
        else:
            payload = {
                'status': 'ready',
                'source': 'lisn_slow_loop',
                'engineering': self.engineering,
                **extra,
            }
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _write_vlm_trace(self, *, proposal, image_msg, stamp_ns, received_ns,
                         raw_goal, raw_mode, raw_params, raw_validity,
                         grounding_status, grounding_reason, record=None):
        """Append one allowlisted VLM response/decision trace line.

        Never writes prompt, key, auth header, env, reasoning, or CoT.  The raw
        structured goal is preserved as returned; host provenance stays host-side.
        """
        path = Path(self.runner.evidence_path).with_name('VLM_TRACE.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        self._vlm_trace_index += 1
        now_sim = self.node.get_clock().now().nanoseconds
        source_stamp = int(stamp_ns)
        source_age = int(now_sim) - source_stamp
        latency_ms = (time.monotonic_ns() - int(received_ns)) / 1_000_000.0
        image_sha = hashlib.sha256(bytes(image_msg.data)).hexdigest()
        goal_trust = proposal.get('goal_trust')
        if not isinstance(goal_trust, Mapping):
            goal_trust = {}
        point = goal_trust.get('point') if isinstance(goal_trust.get('point'), Mapping) else {}
        scan = goal_trust.get('scan') if isinstance(goal_trust.get('scan'), Mapping) else {}
        config = self._producer_config
        model = str(getattr(config, 'model', '') or '') if config is not None else ''
        revision = str(getattr(config, 'model_revision', '') or '') if config is not None else ''
        image_id = str(getattr(config, 'model_image_id', '') or '') if config is not None else ''
        line = {
            'event': 'response',
            'request_id': f'{self._session_id or "unknown"}-{self._vlm_trace_index}',
            'session_id': self._session_id,
            'episode': self._session_episode,
            'source_stamp_ns': source_stamp,
            'response_sim_ns': int(now_sim),
            'source_age_ns': int(source_age),
            'outdated': bool(source_age > 500_000_000),
            'latency_ms': latency_ms,
            'image_sha256': image_sha,
            'image_width': int(image_msg.width),
            'image_height': int(image_msg.height),
            'model': model,
            'model_revision': revision,
            'model_image_id': image_id,
            'raw_goal': raw_goal,
            'raw_controller_mode': raw_mode,
            'raw_controller_parameters': raw_params,
            'raw_validity': raw_validity,
            'grounding': {
                'status': grounding_status,
                'reason': grounding_reason,
                'point': dict(point),
                'scan': dict(scan),
                'map_pose': proposal.get('_grounded_map_pose'),
            },
            'goal_trust': {
                'trusted': goal_trust.get('trusted'),
                'source_invalid': goal_trust.get('source_invalid'),
                'reason': goal_trust.get('reason'),
            },
            'decision': record.get('decision') if record else None,
            'reason_codes': record.get('reason_codes') if record else None,
            'readback_parameters': record.get('readback_parameters') if record else None,
            'update_id': record.get('update_id') if record else None,
            'typed_rejection': grounding_reason if grounding_status == 'rejected' else None,
        }
        with path.open('a') as stream:
            stream.write(json.dumps(line, sort_keys=True) + '\n')

    def _prepare_proposal(self, proposal, image_msg, instruction, observation_id,
                          frame_id, stamp_ns, received_ns, grounding_bundle=None):
        """Resolve/validate and freeze one proposal without committing it."""
        raw_goal = copy.deepcopy(proposal.get('goal'))
        raw_mode = proposal.get('controller_mode')
        raw_params = copy.deepcopy(proposal.get('controller_parameters'))
        raw_validity = copy.deepcopy(proposal.get('validity'))
        if self._grounding_live:
            try:
                proposal = self._resolve_grounding(proposal, grounding_bundle)
            except ProducerUnavailableError as exc:
                self._write_vlm_trace(
                    proposal=proposal, image_msg=image_msg, stamp_ns=stamp_ns,
                    received_ns=received_ns, raw_goal=raw_goal, raw_mode=raw_mode,
                    raw_params=raw_params, raw_validity=raw_validity,
                    grounding_status='rejected', grounding_reason=str(exc),
                )
                raise
        if self._grounding_scan:
            # ProxMPC has no runtime parameter callback.  Host pins the
            # controller parameters to the static config; the adapter accepts
            # only an exact match and records no dynamic apply.
            proposal['controller_parameters'] = dict(ProxGoalOnlyAdapter.STATIC)
            proposal['controller_parameter_policy'] = self._parameter_policy
        if not self._controller_verified:
            self.adapter.verify_social_mpc()
            self._controller_verified = True
        return {
            'proposal': copy.deepcopy(proposal),
            'image_msg': image_msg,
            'image_bytes': bytes(image_msg.data),
            'frame_id': frame_id,
            'stamp_ns': stamp_ns,
            'observation_id': observation_id,
            'received_monotonic_ns': received_ns,
            'instruction': instruction,
            'grounding_bundle': grounding_bundle,
            'raw_goal': raw_goal,
            'raw_mode': raw_mode,
            'raw_params': raw_params,
            'raw_validity': raw_validity,
        }

    def _finalize_commit(self, payload):
        """Commit one frozen payload to the existing commit path exactly once."""
        proposal = payload['proposal']
        record = self.producer_tick.commit(
            proposal=proposal,
            image_bytes=payload['image_bytes'],
            frame_id=payload['frame_id'],
            stamp_ns=payload['stamp_ns'],
            observation_id=payload['observation_id'],
            received_monotonic_ns=payload['received_monotonic_ns'],
            now_monotonic_ns=time.monotonic_ns(),
            now_ros_ns=self.node.get_clock().now().nanoseconds,
        )
        self._write_parameter_observation_receipt()
        goal_trust = proposal.get('goal_trust')
        if not isinstance(goal_trust, Mapping):
            goal_trust = {}
        grounding_status = (
            'accepted' if bool(goal_trust.get('trusted'))
            else 'not_applicable' if not self._grounding_live
            else 'rejected'
        )
        self._write_vlm_trace(
            proposal=proposal, image_msg=payload['image_msg'], stamp_ns=payload['stamp_ns'],
            received_ns=payload['received_monotonic_ns'],
            raw_goal=payload['raw_goal'], raw_mode=payload['raw_mode'],
            raw_params=payload['raw_params'], raw_validity=payload['raw_validity'],
            grounding_status=grounding_status,
            grounding_reason=str(goal_trust.get('reason') or ''),
            record=record,
        )
        if record.get('decision') == 'accepted':
            map_pose = proposal.get('_grounded_map_pose')
            proposal.pop('_grounded_map_pose', None)
            self._publish_grounded_goal(map_pose)
        self._publish_status(record)
        if record.get('decision') == 'accepted':
            Path(self.runner.evidence_path).with_name('LIVE_APPLY.json').write_text(json.dumps({
                'schema': 'arena.lisn.live_apply.v1',
                'runtime_executed': True,
                'engineering_not_scientific': True,
                'controller_identity': self._controller_identity,
                'parameter_policy': self._parameter_policy,
                'before_parameters': record.get('before_parameters'),
                'requested_parameters': record.get('requested_parameters'),
                'readback_parameters': record.get('readback_parameters'),
                'decision': record.get('decision'),
                'reason_codes': record.get('reason_codes'),
                'tick_index': record.get('tick_index'),
                'wiring_live': self.wiring_live,
            }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        return record

    def _write_latency_receipt(self, receipt):
        receipt_path = Path(self.runner.evidence_path).with_name('LATENCY_COMMIT.json')
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        with receipt_path.with_suffix('.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(receipt, sort_keys=True) + '\n')

    def _tick_latency_commits(self):
        if self._latency_gate is None:
            return
        now_sim = self.node.get_clock().now().nanoseconds
        results = self._latency_gate.poll_ready(now_sim, self._finalize_commit)
        for _item, _record, receipt in results:
            self._write_latency_receipt(receipt)

    def _commit_proposal(self, proposal, image_msg, instruction, observation_id,
                         frame_id, stamp_ns, received_ns, grounding_bundle=None):
        payload = self._prepare_proposal(
            proposal, image_msg, instruction, observation_id,
            frame_id, stamp_ns, received_ns, grounding_bundle)
        return self._finalize_commit(payload)

    def _write_parameter_observation_receipt(self):
        """Record which frozen parameters were read-only observed.

        Only ``max_linear_velocity_mps`` is hot-applied.  Ambiguous frozen
        bindings are read through their first declared candidate for the
        all-field readback comparison and are never written.
        """
        if self._parameter_observation_written:
            return
        bindings = getattr(self.adapter, 'last_read_bindings', {}) or {}
        policy_bindings = self.policy.parameter_bindings
        observed = {}
        for logical in ENGINEERING_LOGICAL_PARAMETERS:
            binding = policy_bindings.get(logical) or {}
            observed[logical] = {
                'read_name': bindings.get(logical),
                'status': binding.get('status'),
                'applied': (
                    False if self._grounding_scan
                    else logical in ENGINEERING_APPLIED_PARAMETERS
                ),
            }
        if self._grounding_scan:
            applied_parameters = []
            frozen_parameters = list(ProxGoalOnlyAdapter.STATIC)
            apply_rule = (
                'ProxMPC has no runtime parameter callback; host pins the static '
                'parameters and records no dynamic apply. Goal chain is independent.'
            )
        else:
            applied_parameters = list(ENGINEERING_APPLIED_PARAMETERS)
            frozen_parameters = list(ENGINEERING_FROZEN_PARAMETERS)
            apply_rule = (
                'only max_linear_velocity_mps is hot-applied; frozen parameters '
                'are read-only observed and never changed'
            )
        path = Path(self.runner.evidence_path).with_name('PARAMETER_OBSERVATION.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'schema': 'arena.lisn.parameter_observation.v1',
            'applied_parameters': applied_parameters,
            'frozen_parameters': frozen_parameters,
            'observed_bindings': observed,
            'apply_rule': apply_rule,
        }, indent=2, sort_keys=True) + '\n')
        self._parameter_observation_written = True

    def _publish_terminal_error(self, reason: str) -> None:
        from std_msgs.msg import String

        if self._session_id is not None:
            payload = session_status_payload(
                session_id=self._session_id,
                episode=self._session_episode,
                status='error',
                engineering=self.engineering,
                reason=str(reason),
                producer_mode=True,
            )
        else:
            payload = {
                'status': 'error',
                'source': 'lisn_slow_loop',
                'engineering': self.engineering,
                'reason': str(reason),
                'producer_mode': True,
            }
        self._status_pub.publish(String(data=json.dumps(payload)))

    def _handle_producer_failure(self, exc: BaseException) -> None:
        """Log a static code and latch only the first auth failure.

        The raw exception text is never logged: HTTP reasons, response bodies
        and URLs can echo credentials or query secrets.
        """
        code = safe_error_code(exc)
        auth_failure = code == 'auth_denied' or bool(
            getattr(self.producer, 'auth_latched', False)
        )
        terminal = auth_failure or code in {'timeout', 'deadline_exceeded'}
        if terminal:
            if self._auth_gate.record_failure(exc):
                if code == 'deadline_exceeded':
                    reason = 'producer_deadline_exceeded'
                elif code == 'timeout':
                    reason = 'producer_timeout'
                else:
                    reason = 'producer_auth_failure'
                self.node.get_logger().error(
                    f'[lisn_slow_loop] producer terminal latch; code={code}; '
                    f'category={safe_error_category(exc)}; phase=producer; '
                    'no retry, no fallback'
                )
                self._publish_terminal_error(reason)
            return
        self.node.get_logger().error(
            f'[lisn_slow_loop] producer failed closed; code={code}; '
            f'category={safe_error_category(exc)}; phase=producer'
        )

    def _tick_producer(self, now):
        if not self._auth_gate.should_request():
            return
        if (
            self._pending is not None
            and not self._pending.done()
            and self._producer_deadline_ns
            and now - self._pending_started_ns > self._producer_deadline_ns
        ):
            # The request exceeded the global budget.  Latch and never harvest
            # a late result; the worker thread may still finish, but its result
            # is ignored and no next request is submitted.
            self._handle_producer_failure(ProducerUnavailableError('deadline_exceeded'))
            return
        # Harvest a finished commit first; never block the timer callback.
        if self._commit_future is not None and self._commit_future.done():
            future, self._commit_future = self._commit_future, None
            commit_meta, self._commit_meta = self._commit_meta, None
            try:
                result = future.result()
            except Exception as exc:
                proposal = commit_meta[0] if commit_meta else None
                proposal_valid = bool(
                    isinstance(proposal, dict)
                    and all(name in proposal for name in HttpVlmProducer.REQUIRED_FIELDS)
                )
                self.node.get_logger().error(
                    f'[lisn_slow_loop] commit failed closed; '
                    f'code={safe_error_code(exc)}; '
                    f'category={safe_error_category(exc)}; '
                    f'phase=commit; proposal_valid={proposal_valid}'
                )
                return
            if self._latency_gate is not None:
                payload = result
                _item, receipt = self._latency_gate.enqueue(
                    source_stamp_ns=payload['stamp_ns'],
                    source_frame_id=payload['frame_id'],
                    source_identity={
                        'image_sha256': hashlib.sha256(payload['image_bytes']).hexdigest(),
                    },
                    payload=payload,
                    previous_response_sim_ns=self._latency_previous_response_sim_ns,
                )
                self._latency_previous_response_sim_ns = (
                    self.node.get_clock().now().nanoseconds)
                self._write_latency_receipt(receipt)
            return
        if self._commit_future is not None:
            return
        # Harvest exactly one finished HTTP proposal and hand it to a commit
        # worker.  The proposal is reused; produce() is never called again.
        if self._pending is not None and self._pending.done():
            pending, self._pending = self._pending, None
            meta, self._pending_meta = self._pending_meta, None
            try:
                proposal = pending.result()
            except Exception as exc:
                self._handle_producer_failure(exc)
                return
            image_msg, instruction, observation_id, frame_id, stamp_ns, received_ns, grounding_bundle = meta
            self._commit_meta = (proposal, image_msg, instruction, observation_id,
                                 frame_id, stamp_ns, received_ns, grounding_bundle)
            prepare = (self._prepare_proposal if self._latency_gate is not None
                       else self._commit_proposal)
            self._commit_future = self._commit_executor.submit(
                prepare,
                proposal, image_msg, instruction, observation_id,
                frame_id, stamp_ns, received_ns, grounding_bundle,
            )
            return
        if self._pending is not None:
            return
        if self._image is None or not self._instruction:
            return
        if now < self.runner.scheduled_for_ns():
            return
        image_msg, self._image = self._image, None
        observation_id, frame_id, stamp_ns, content = self._observation_identity(
            image_msg, self._instruction
        )
        self._pending_started_ns = now
        try:
            grounding_bundle = self._freeze_grounding_bundle(image_msg, frame_id, stamp_ns)
        except ProducerUnavailableError as exc:
            self.node.get_logger().error(
                f'[lisn_slow_loop] grounding freeze failed closed; code={exc.code}; '
                'no producer request submitted'
            )
            return
        future = self._executor.submit(
            self.producer_tick.produce,
            instruction=self._instruction,
            image_bytes=bytes(image_msg.data),
            frame_id=frame_id,
            stamp_ns=stamp_ns,
            image_encoding=str(image_msg.encoding),
            width=int(image_msg.width),
            height=int(image_msg.height),
            channels=self._image_channels(image_msg.encoding),
        )
        self._pending_meta = (image_msg, self._instruction, observation_id, frame_id,
                              stamp_ns, self._image_received_ns, grounding_bundle)
        self._pending = future

    def _tick(self):
        now = time.monotonic_ns()
        self._publish_heartbeat(self.node.get_clock().now().nanoseconds)
        if self.producer is not None:
            self._tick_producer(now)
            return
        if self._proposal is None:
            return
        if now < self.runner.scheduled_for_ns():
            return
        proposal, self._proposal = self._proposal, None
        record = self.runner.tick(
            proposal,
            received_monotonic_ns=self._received_ns,
            now_monotonic_ns=now,
            now_ros_ns=self.node.get_clock().now().nanoseconds,
        )
        self._publish_status(record)

    def spin(self):
        import rclpy

        try:
            rclpy.spin(self.node)
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            if self._commit_executor is not None:
                self._commit_executor.shutdown(wait=False)
            self.node.destroy_node()


def bridge_cli_argv(*, policy, evidence, controller_node, proposal_topic,
                    target_frame, episode_id, producer_endpoint, rgb_topic,
                    instruction_topic, wiring_live, producer_timeout_sec=None,
                    producer_deadline_sec=None, model_status_topic=None,
                    eval_ready_topic=None, status_heartbeat_sec=None,
                    grounding_live=None, grounding_scan=None, scan_topic=None,
                    scan_max_age_sec=None, max_scan_range_m=None,
                    depth_topic=None, camera_info_topic=None,
                    grounded_goal_topic=None, semantic_cost_topic=None,
                    semantic_cost_frame=None, semantic_ttl_sec=None,
                    semantic_status_topic=None, min_confidence=None) -> list:
    """Single source of truth for the launch argv (values may be substitutions)."""
    argv = [
        '--engineering',
        '--policy', policy,
        '--evidence', evidence,
        '--controller-node', controller_node,
        '--proposal-topic', proposal_topic,
        '--target-frame', target_frame,
        '--episode-id', episode_id,
        '--producer-endpoint', producer_endpoint,
        '--rgb-topic', rgb_topic,
        '--instruction-topic', instruction_topic,
        '--wiring-live', wiring_live,
    ]
    if producer_timeout_sec is not None:
        argv += ['--producer-timeout-sec', producer_timeout_sec]
    if producer_deadline_sec is not None:
        argv += ['--producer-deadline-sec', producer_deadline_sec]
    if model_status_topic is not None:
        argv += ['--model-status-topic', model_status_topic]
    if eval_ready_topic is not None:
        argv += ['--eval-ready-topic', eval_ready_topic]
    if status_heartbeat_sec is not None:
        argv += ['--status-heartbeat-sec', status_heartbeat_sec]
    if grounding_live is not None:
        argv += ['--grounding-live', grounding_live]
    if grounding_scan is not None:
        argv += ['--grounding-scan', grounding_scan]
    if scan_topic is not None:
        argv += ['--scan-topic', scan_topic]
    if scan_max_age_sec is not None:
        argv += ['--scan-max-age-sec', scan_max_age_sec]
    if max_scan_range_m is not None:
        argv += ['--max-scan-range-m', max_scan_range_m]
    if depth_topic is not None:
        argv += ['--depth-topic', depth_topic]
    if camera_info_topic is not None:
        argv += ['--camera-info-topic', camera_info_topic]
    if grounded_goal_topic is not None:
        argv += ['--grounded-goal-topic', grounded_goal_topic]
    if semantic_cost_topic is not None:
        argv += ['--semantic-cost-topic', semantic_cost_topic]
    if semantic_cost_frame is not None:
        argv += ['--semantic-cost-frame', semantic_cost_frame]
    if semantic_ttl_sec is not None:
        argv += ['--semantic-ttl-sec', semantic_ttl_sec]
    if semantic_status_topic is not None:
        argv += ['--semantic-status-topic', semantic_status_topic]
    if min_confidence is not None:
        argv += ['--min-confidence', min_confidence]
    return argv


def build_parser():
    parser = argparse.ArgumentParser(description='LISN slow-loop bridge (wiring only).')
    parser.add_argument('--policy', default=str(DEFAULT_POLICY))
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--fixture', default=str(DEFAULT_FIXTURE))
    parser.add_argument('--controller-node', default='/task_generator_node/Ai2_Bot2/controller_server')
    parser.add_argument('--proposal-topic', default='/task_generator_node/Ai2_Bot2/lisn_slow_loop/proposal')
    parser.add_argument('--target-frame', default='map')
    parser.add_argument('--episode-id', default='lisn-episode')
    parser.add_argument('--engineering', action='store_true',
                        help='Use the engineering profile: apply only the exact max_linear_velocity binding.')
    parser.add_argument('--producer-endpoint', default='',
                        help='Real LISN VLM JSON-HTTP endpoint; empty means proposal-topic mode.')
    parser.add_argument('--rgb-topic',
                        default='/task_generator_node/Ai2_Bot2/head_camera/image')
    parser.add_argument('--instruction-topic',
                        default='/task_generator_node/vln_instruction')
    parser.add_argument('--model-status-topic', default='',
                        help='Topic for model/status; defaults to <proposal-topic>/status.')
    parser.add_argument('--eval-ready-topic', default='/task_generator_node/eval_ready_status',
                        help='Task-generator topic carrying the per-episode (session_id, episode) identity.')
    parser.add_argument('--status-heartbeat-sec', type=float, default=1.0,
                        help='Session-bound model/status heartbeat period; must stay below the task '
                             'generator model_heartbeat_timeout_sec.')
    parser.add_argument('--producer-timeout-sec', type=float, default=None,
                        help='Explicit bounded HTTP timeout for the VLM producer; overrides '
                             'ARENA_LISN_VLM_TIMEOUT_SEC and the ProducerConfig default.')
    parser.add_argument('--producer-deadline-sec', type=float, default=None,
                        help='Global per-tick producer deadline; a pending request past it '
                             'is latched and its late result is never applied.')
    parser.add_argument('--wiring-live', nargs='?', const='true', default='false',
                        choices=['true', 'false', 'True', 'False', '1', '0'],
                        help='Wiring-only live run (true/false); requires a real endpoint, never flips runtime_ready.')
    parser.add_argument('--grounding-live', nargs='?', const='true', default='false',
                        choices=['true', 'false', 'True', 'False', '1', '0'],
                        help='Enable same-stamp RGB/depth/CameraInfo/TF grounding.')
    parser.add_argument('--grounding-scan', nargs='?', const='true', default='false',
                        choices=['true', 'false', 'True', 'False', '1', '0'],
                        help='A-track: RGB bearing + same-source real LaserScan range (no depth).')
    parser.add_argument('--scan-topic', default='/task_generator_node/Ai2_Bot2/laser_scan')
    parser.add_argument('--scan-max-age-sec', type=float, default=0.6,
                        help='Bounded RGB-to-scan age before the frozen bundle is rejected.')
    parser.add_argument('--max-scan-range-m', type=float, default=12.0,
                        help='Reject a scan range beyond this metric bound.')
    parser.add_argument('--depth-topic', default='/task_generator_node/Ai2_Bot2/head_camera/depth')
    parser.add_argument('--camera-info-topic', default='/task_generator_node/Ai2_Bot2/head_camera/camera_info')
    parser.add_argument('--grounded-goal-topic',
                        default='/task_generator_node/Ai2_Bot2/lisn_slow_loop/grounded_goal')
    parser.add_argument('--semantic-cost-topic',
                        default='/task_generator_node/Ai2_Bot2/lisn_semantic_cost')
    parser.add_argument('--semantic-cost-frame', default='Ai2_Bot2/odom')
    parser.add_argument('--semantic-ttl-sec', type=float, default=12.0)
    parser.add_argument('--semantic-status-topic', default='/lisn_semantic_layer_status')
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--latency-mode',
                        choices=('disabled', 'passthrough', 'delay', 'drop', 'out_of_order'),
                        default='disabled',
                        help='Disabled keeps the synchronous producer commit; enabled polls the ROS clock before the single commit.')
    parser.add_argument('--latency-ms', type=float, default=0.0)
    parser.add_argument('--latency-profile', default='',
                        help='Optional JSON with {mode, latency_ms}; overrides --latency-mode/--latency-ms.')
    parser.add_argument('--wiring-smoke', action='store_true',
                        help='Run one deterministic offline tick; not a real agent or science result.')
    parser.add_argument('--producer-wiring-smoke', action='store_true',
                        help='Exercise the real HTTP producer code against a local mock; not a real agent.')
    return parser


def strip_ros_args(argv):
    """Drop ROS launch's --ros-args tail before argparse sees it."""
    argv = list(argv or [])
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    return argv


def main(argv=None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(strip_ros_args(raw))
    args.wiring_live = str(args.wiring_live).strip().lower() in {'true', '1'}
    args.grounding_live = str(args.grounding_live).strip().lower() in {'true', '1'}
    args.grounding_scan = str(args.grounding_scan).strip().lower() in {'true', '1'}

    if args.wiring_smoke:
        receipt = run_wiring_smoke(
            policy_path=Path(args.policy),
            fixture_path=Path(args.fixture),
            evidence_path=Path(args.evidence),
            target_frame=args.target_frame,
            episode_id=args.episode_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    if args.producer_wiring_smoke:
        receipt = run_producer_wiring_smoke(
            policy_path=Path(args.policy),
            evidence_path=Path(args.evidence),
            target_frame=args.target_frame,
            episode_id=args.episode_id,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    policy = load_policy(Path(args.policy), engineering=args.engineering)
    blockers = production_blockers(policy)
    if blockers and not args.wiring_live:
        print(json.dumps({
            'refused': True,
            'runtime_ready': policy.runtime_ready,
            'blockers': blockers,
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    if args.wiring_live and not args.producer_endpoint:
        resolved = ProducerConfig.from_env()
        if resolved.provider == 'none':
            print(json.dumps({
                'refused': True,
                'reason': 'wiring-live requires --producer-endpoint or a resolved credential env; '
                          'no fixture fallback',
            }, indent=2, sort_keys=True), file=sys.stderr)
            return 2

    import rclpy

    rclpy.init()
    try:
        node = LisnSlowLoopNode(
            policy_path=Path(args.policy),
            evidence_path=Path(args.evidence),
            controller_node=args.controller_node,
            proposal_topic=args.proposal_topic,
            target_frame=args.target_frame,
            episode_id=args.episode_id,
            engineering=args.engineering,
            producer_endpoint=args.producer_endpoint,
            rgb_topic=args.rgb_topic,
            instruction_topic=args.instruction_topic,
            model_status_topic=args.model_status_topic,
            eval_ready_topic=args.eval_ready_topic,
            status_heartbeat_sec=args.status_heartbeat_sec,
            producer_timeout_sec=args.producer_timeout_sec,
            producer_deadline_sec=args.producer_deadline_sec,
            wiring_live=args.wiring_live,
            grounding_live=args.grounding_live,
            grounding_scan=args.grounding_scan,
            scan_topic=args.scan_topic,
            scan_max_age_sec=args.scan_max_age_sec,
            max_scan_range_m=args.max_scan_range_m,
            depth_topic=args.depth_topic,
            camera_info_topic=args.camera_info_topic,
            grounded_goal_topic=args.grounded_goal_topic,
            semantic_cost_topic=args.semantic_cost_topic,
            semantic_cost_frame=args.semantic_cost_frame,
            semantic_ttl_sec=args.semantic_ttl_sec,
            semantic_status_topic=args.semantic_status_topic,
            min_confidence=args.min_confidence,
            latency_mode=args.latency_mode,
            latency_ms=args.latency_ms,
            latency_profile=args.latency_profile,
        )
        node.spin()
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
