#!/usr/bin/env python3
"""Paper-pipeline contracts for Arena5 A/B/C tracks.

This module is deliberately ROS-free at import time.  It provides the CPU
contracts that the paper lanes depend on:

* strict RGB-only B fast-guard validation (on/off arms share the same proposal)
* latency injection that preserves frozen source identity/stamp/frame
* static/open-loop/closed-loop scenario + sensor-only settings validation
* dataset split/manifest validation
* a fail-close metrics stage wrapper with independent exit codes
* unique output prefixes and automatic Docker endpoint resolution

The module is a contract/CLI layer; it does not start ROS, Isaac, or a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import yaml
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from arena_bringup import benchmark_profile

DEFAULT_WORKSPACE_ROOT = Path('/home/ubuntu/arena_jazzy_ws')


def _default_repo_root() -> Path:
    """Locate the tracked Arena repo for relative contract paths.

    Source/editable runs resolve to the checkout.  An installed overlay cannot
    infer the checkout, so it falls back to the workspace source tree and
    ``ARENA_REPO_ROOT`` can override either case explicitly.
    """
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / 'arena_bringup').is_dir() and (candidate / '_meta').is_dir():
        return candidate
    fallback = DEFAULT_WORKSPACE_ROOT / 'src' / 'Arena'
    if (fallback / 'arena_bringup').is_dir():
        return fallback
    return candidate


REPO_ROOT = Path(os.environ.get('ARENA_REPO_ROOT') or _default_repo_root())

SETTINGS_SCHEMA = 'arena.paper.pipeline_settings.v1'
SCENARIO_MODES = ('static', 'openloop', 'closedloop')
TRACKS = ('A', 'B', 'C')
SETTINGS_REQUIRED = (
    'schema', 'track', 'scenario_mode', 'world', 'scenario', 'robot',
    'episodes', 'pedestrian_seed', 'sensor_rgb_topic', 'ego_odom_topic',
    'sensor_only', 'latency_ms', 'output_prefix',
)
GT_FORBIDDEN_KEYS = (
    'gt_map', 'gt_people', 'gt_agents', 'eval_goal', 'true_goal',
    'start_goal', 'oracle_trajectory', 'usd_state',
)
B_FORBIDDEN_KEYS = (
    'sensor_scan_topic', 'sensor_depth_topic', 'sensor_pointcloud_topic',
) + GT_FORBIDDEN_KEYS

BFAST_SCHEMA = 'arena.bfast.proposal.v1'
BFAST_PROFILE_SCHEMA = 'arena.bfast.profile.v1'
BFAST_ACTIONS = ('goal', 'bearing', 'stop', 'hold', 'error')
BFAST_UNITS = ('normalized', 'pixel')
BFAST_FORBIDDEN_KEYS = (
    'scan', 'laser_scan', 'depth', 'pointcloud', 'gt_map', 'gt_people',
    'people', 'eval_goal', 'usd_state', 'oracle_trajectory',
)
EXECUTOR_ACTIONS = ('follow_path', 'bearing', 'stop', 'hold', 'error')

DATASET_SCHEMA = 'arena.paper.dataset_manifest.v1'
DATASET_REQUIRED_CASE_FIELDS = (
    'case_id', 'world', 'scenario', 'robot', 'instruction', 'pedestrian_seed',
)

METRICS_SCHEMA = 'arena.paper.metrics_pipeline.v1'
DEFAULT_STAGE_NAMES = (
    'stage_a_metrics', 'stage_b_vln_task_metrics', 'stage_c_social_metrics',
    'validation',
)


class PipelineRejected(ValueError):
    """A contract rejection.  CLI maps this to rc=2."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['payload:not_object']}
    if payload.get('schema') != SETTINGS_SCHEMA:
        errors.append('schema')
    track = payload.get('track')
    if track not in TRACKS:
        errors.append('track')
    mode = payload.get('scenario_mode')
    if mode not in SCENARIO_MODES:
        errors.append('scenario_mode')
    for key in SETTINGS_REQUIRED:
        if payload.get(key) is None or payload.get(key) == '':
            errors.append(f'missing:{key}')
    episodes = payload.get('episodes')
    if not isinstance(episodes, int) or episodes < 1:
        errors.append('episodes')
    seed = payload.get('pedestrian_seed')
    if not isinstance(seed, int) or seed < 0:
        errors.append('pedestrian_seed')
    latency = payload.get('latency_ms')
    if latency is not None and (not _finite(latency) or float(latency) < 0.0):
        errors.append('latency_ms')
    if track == 'A':
        if not payload.get('sensor_scan_topic'):
            errors.append('A:sensor_scan_topic')
        for key in GT_FORBIDDEN_KEYS:
            if payload.get(key) is not None:
                errors.append(f'A:forbidden:{key}')
    if track == 'B':
        if payload.get('sensor_only') is not True:
            errors.append('B:sensor_only_must_be_true')
        for key in B_FORBIDDEN_KEYS:
            if payload.get(key) is not None:
                errors.append(f'B:forbidden:{key}')
    if track == 'C':
        if payload.get('teacher_oracle') is not True:
            errors.append('C:teacher_oracle_must_be_true')
    return {'valid': not errors, 'errors': errors}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Scenario profiles: static / open-loop dynamic / closed-loop HuNav
# ---------------------------------------------------------------------------

SCENARIO_PROFILE_SCHEMA = 'arena.paper.scenario_profile.v1'
SCENARIO_PROFILE_DEFAULTS = {
    'static': {
        'human_source': 'none',
        'dynamic_humans': False,
        'robot_human_feedback': False,
        'description': 'no dynamic humans; static world and objects',
    },
    'openloop': {
        'human_source': 'hunav',
        'dynamic_humans': True,
        'robot_human_feedback': False,
        'description': 'scripted HuNav humans; robot does not affect human behavior',
    },
    'closedloop': {
        'human_source': 'hunav',
        'dynamic_humans': True,
        'robot_human_feedback': True,
        'description': 'HuNav humans react to robot state through the SFM/behavior tree',
    },
}


def validate_scenario_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['payload:not_object'], 'profile': None}
    if payload.get('schema') != SCENARIO_PROFILE_SCHEMA:
        errors.append('schema')
    mode = payload.get('mode')
    if mode not in SCENARIO_PROFILE_DEFAULTS:
        errors.append('mode')
        expected = None
    else:
        expected = SCENARIO_PROFILE_DEFAULTS[mode]
    for key in ('world', 'scenario', 'robot'):
        if payload.get(key) in (None, ''):
            errors.append(f'missing:{key}')
    episodes = payload.get('episodes')
    if not isinstance(episodes, int) or episodes < 1:
        errors.append('episodes')
    seed = payload.get('pedestrian_seed')
    if not isinstance(seed, int) or seed < 0:
        errors.append('pedestrian_seed')
    if expected is not None:
        for key in ('human_source', 'dynamic_humans', 'robot_human_feedback'):
            if key in payload and payload.get(key) != expected[key]:
                errors.append(f'{mode}:{key}_must_be_{expected[key]}')
            if key not in payload:
                errors.append(f'missing:{key}')
        if mode in ('openloop', 'closedloop') and not payload.get('hunav_config'):
            errors.append(f'{mode}:hunav_config')
        if mode == 'static' and payload.get('hunav_config') not in (None, ''):
            errors.append('static:hunav_config_must_be_absent')
    resolved_hunav = None
    if mode in ('openloop', 'closedloop') and payload.get('hunav_config'):
        raw = str(payload['hunav_config'])
        candidate = Path(raw) if os.path.isabs(raw) else REPO_ROOT / raw
        if not candidate.is_file():
            fallback = REPO_ROOT / 'arena_bringup' / 'configs' / 'hunav' / raw
            if fallback.is_file():
                candidate = fallback
        if not candidate.is_file():
            errors.append('hunav_config_path_missing')
        else:
            resolved_hunav = str(candidate.resolve())
            expected_sha = payload.get('hunav_config_sha256')
            if expected_sha:
                actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if actual != expected_sha:
                    errors.append('hunav_config_sha256')
    if mode == 'openloop':
        runtime_status = 'UNSUPPORTED'
        runtime_reason = ('UNSUPPORTED_OPENLOOP_WITHOUT_SCRIPTED_HUMANS: '
                          'HuNav SFM reacts to the robot, so an open-loop arm '
                          'needs a real scripted-human interface')
    elif mode in ('static', 'closedloop'):
        runtime_status = 'SUPPORTED'
        runtime_reason = None
    else:
        runtime_status = 'UNSUPPORTED'
        runtime_reason = 'UNSUPPORTED_SCENARIO_MODE'
    return {
        'valid': not errors,
        'errors': errors,
        'mode': mode,
        'profile': dict(expected) if expected else None,
        'resolved_hunav_config': resolved_hunav,
        'hunav_config_sha256': payload.get('hunav_config_sha256'),
        'runtime_status': runtime_status,
        'runtime_reason': runtime_reason,
    }


def scenario_command(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Copy-able Arena argv + env for one scenario profile.

    The HuNav config is selected with ``ARENA_HUNAV_CONFIG``; the loader reads
    that env var.  ``--social-eval`` is only a metrics flag and is never used
    as proof of robot-human feedback.
    """
    validation = validate_scenario_profile(profile)
    if not validation['valid']:
        raise PipelineRejected(';'.join(validation['errors']))
    mode = str(profile.get('mode'))
    if validation['runtime_status'] != 'SUPPORTED':
        raise PipelineRejected(str(validation['runtime_reason']))
    argv = [
        'ros2', 'run', 'arena_bringup', 'internnav_eval',
        '--world', str(profile.get('world')),
        '--scenario-file', str(profile.get('scenario')),
        '--robot', str(profile.get('robot')),
        '--episodes', str(profile.get('episodes')),
        '--timeout', str(profile.get('timeout', 300)),
        '--pedestrian-seed', str(profile.get('pedestrian_seed')),
    ]
    env: dict[str, str] = {}
    if mode == 'static':
        argv += ['--human', 'dummy', '--expected-human-count', '0']
    else:
        argv += ['--human', 'hunav', '--social-eval']
        env['ARENA_HUNAV_CONFIG'] = str(
            validation.get('resolved_hunav_config') or profile['hunav_config'])
    return {
        'argv': argv,
        'env': env,
        'mode': mode,
        'robot_human_feedback': bool(profile.get('robot_human_feedback')),
        'feedback_source': 'hunav_sfm' if mode == 'closedloop' else 'none',
        'social_eval_is_metrics_only': True,
    }


# ---------------------------------------------------------------------------
# B strict RGB-only fast guard
# ---------------------------------------------------------------------------

def validate_bfast_proposal(proposal: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(proposal, Mapping):
        return ['proposal:not_object']
    if proposal.get('schema') != BFAST_SCHEMA:
        errors.append('schema')
    action = proposal.get('action')
    if action not in BFAST_ACTIONS:
        errors.append('action')
    if action in ('goal', 'bearing'):
        point = proposal.get('point')
        if not isinstance(point, Mapping):
            errors.append('point')
        else:
            units = point.get('units')
            if units not in BFAST_UNITS:
                errors.append('units')
            for key in ('u', 'v'):
                if not _finite(point.get(key)):
                    errors.append(f'point.{key}')
            confidence = point.get('confidence')
            if not _finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
                errors.append('point.confidence')
    if action == 'bearing' and not _finite(proposal.get('bearing_rad')):
        errors.append('bearing_rad')
    for key in BFAST_FORBIDDEN_KEYS:
        if key in proposal:
            errors.append(f'forbidden:{key}')
    return errors


FLOW_RISK_SCHEMA = 'arena.bfast.flow_risk_proxy.v1'
B_TOPIC_DENYLIST = (
    '/scan', '/laser_scan', 'laser_scan', '/depth', 'depth', '/points',
    'pointcloud', '/global_costmap', '/map', '/people', 'people',
    'eval_goal', 'gt_map', 'gt_people', 'oracle',
)


def validate_bfast_topics(topics: Sequence[str]) -> dict[str, Any]:
    """Reject any B subscription that would leak scan/depth/GT/People."""
    bad = []
    for topic in topics:
        text = str(topic).lower()
        if any(token in text for token in B_TOPIC_DENYLIST):
            bad.append(str(topic))
    return {'valid': not bad, 'forbidden_topics': bad}


def _default_flow(prev_gray, curr_gray):
    """Farneback flow backend; imported lazily so CPU contracts need no cv2."""
    import cv2
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def compute_flow_risk_proxy(
    prev_rgb,
    curr_rgb,
    *,
    dt_sec: float,
    risk_threshold: float = 0.5,
    flow_backend: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """RGB-only flow-risk proxy (not a causal TTC and not a depth metric).

    The score is the positive radial expansion in the central ROI normalized
    by image texture.  A pure lateral translation or camera rotation has a
    near-zero mean radial component and therefore does not trip the guard.
    A low-texture ROI is ``UNKNOWN`` and never vetoes.  Invalid frames or flow
    raise ``PipelineRejected``; the runtime node maps that to STOP.
    """
    if not _finite(dt_sec) or float(dt_sec) <= 0.0:
        raise PipelineRejected('dt_sec')
    if not _finite(risk_threshold) or float(risk_threshold) <= 0.0:
        raise PipelineRejected('risk_threshold')
    try:
        import numpy as np
        prev = np.asarray(prev_rgb)
        curr = np.asarray(curr_rgb)
    except Exception as exc:  # noqa: BLE001
        raise PipelineRejected(f'rgb_array:{exc}') from exc
    if prev.dtype.kind not in 'uif' or curr.dtype.kind not in 'uif':
        raise PipelineRejected('rgb_dtype')
    if prev.shape != curr.shape or prev.ndim not in (2, 3):
        raise PipelineRejected('rgb_shape_mismatch')
    if prev.size == 0:
        raise PipelineRejected('rgb_empty')
    prev_gray = prev.mean(axis=2) if prev.ndim == 3 else prev
    curr_gray = curr.mean(axis=2) if curr.ndim == 3 else curr
    if not np.isfinite(prev_gray).all() or not np.isfinite(curr_gray).all():
        raise PipelineRejected('rgb_nonfinite')
    height, width = prev_gray.shape
    x0, x1 = int(width * 0.25), int(width * 0.75)
    y0, y1 = int(height * 0.25), int(height * 0.75)
    texture_energy = float(prev_gray[y0:y1, x0:x1].std())
    texture_state = 'UNKNOWN' if texture_energy < 1e-6 else 'OK'
    backend = flow_backend or _default_flow
    try:
        flow = backend(prev_gray, curr_gray)
    except Exception as exc:  # noqa: BLE001
        raise PipelineRejected(f'optical_flow_failed:{exc}') from exc
    if flow is None:
        raise PipelineRejected('optical_flow_failed')
    flow = np.asarray(flow)
    if flow.shape != (height, width, 2):
        raise PipelineRejected('optical_flow_shape')
    if not np.isfinite(flow).all():
        raise PipelineRejected('optical_flow_nonfinite')
    roi = flow[y0:y1, x0:x1]
    if roi.size == 0:
        raise PipelineRejected('roi_empty')
    ys, xs = np.mgrid[y0:y1, x0:x1]
    nx = (xs - (width - 1) / 2.0)
    ny = (ys - (height - 1) / 2.0)
    norm = np.sqrt(nx * nx + ny * ny)
    norm[norm < 1e-9] = 1.0
    nx, ny = nx / norm, ny / norm
    radial = roi[:, :, 0] * nx + roi[:, :, 1] * ny
    radial_expansion = float(radial.mean())
    magnitude = np.sqrt((roi ** 2).sum(axis=2))
    flow_energy = float(magnitude.mean())
    lateral = roi - np.stack([radial * nx, radial * ny], axis=2)
    lateral_energy = float(np.sqrt((lateral ** 2).sum(axis=2)).mean())
    risk_score = max(0.0, radial_expansion) / max(texture_energy, 1e-6)
    stop = bool(texture_state == 'OK' and risk_score >= float(risk_threshold))
    if texture_state == 'UNKNOWN':
        reason = 'low_texture_unknown'
    elif stop:
        reason = 'flow_risk_above_threshold'
    else:
        reason = 'flow_risk_clear'
    return {
        'schema': FLOW_RISK_SCHEMA,
        'method': 'farneback_central_roi_radial_expansion_risk',
        'dt_sec': float(dt_sec),
        'risk_threshold': float(risk_threshold),
        'texture_energy': texture_energy,
        'texture_state': texture_state,
        'flow_energy': flow_energy,
        'lateral_energy': lateral_energy,
        'radial_expansion': radial_expansion,
        'risk_score': risk_score,
        'stop': stop,
        'reason': reason,
        'uses_scan': False,
        'uses_depth': False,
        'uses_gt': False,
    }


def _validate_executor_input(executor_input: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """Return (validated common action, typed unsupported reason)."""
    if executor_input is None:
        return None, 'UNSUPPORTED_NO_COMMON_ACTION'
    action = executor_input.get('action')
    if action not in EXECUTOR_ACTIONS:
        return None, f'UNSUPPORTED_EXECUTOR_ACTION:{action}'
    if action == 'follow_path':
        poses = executor_input.get('poses')
        if not isinstance(poses, (list, tuple)) or not poses:
            return None, 'UNSUPPORTED_NO_METRIC_PATH'
        for pose in poses:
            if not isinstance(pose, (list, tuple)) or len(pose) not in (2, 3):
                return None, 'UNSUPPORTED_PATH_POSE'
            if not all(_finite(value) for value in pose):
                return None, 'UNSUPPORTED_PATH_POSE_NOT_FINITE'
        return {'action': 'follow_path',
                'poses': [list(pose) for pose in poses]}, None
    if action == 'bearing':
        # The common Arena executor has no bearing action.  A pixel/bearing
        # without a metric path must not be turned into a fabricated Path.
        return None, 'UNSUPPORTED_BEARING_ACTION'
    return {'action': str(action)}, None


def apply_bfast(proposal: Mapping[str, Any], *, enabled: bool,
                executor_input: Mapping[str, Any] | None = None,
                flow_risk: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate one B proposal against the common executor action.

    ``enabled`` is the only difference between the two B arms.  The proposal,
    prompt, and common action are identical.  The guard may only pass through
    the common action or veto to STOP; it never fabricates a metric Path from a
    bare RGB point.
    """
    errors = validate_bfast_proposal(proposal)
    if errors:
        raise PipelineRejected(';'.join(errors))
    action = str(proposal['action'])
    if executor_input is None and action in ('stop', 'hold', 'error'):
        executor_input = {'action': action}
    validated, unsupported = _validate_executor_input(executor_input)
    if unsupported:
        validated = {'action': 'stop'}
    flow_stop = bool(enabled and flow_risk and flow_risk.get('stop'))
    if flow_stop:
        validated = {'action': 'stop'}
    return {
        'schema': 'arena.bfast.decision.v1',
        'fast_enabled': bool(enabled),
        'source': ('rgb_only_fast_flow_veto' if flow_stop
                   else 'rgb_only_fast' if enabled
                   else 'rgb_only_passthrough'),
        'executor_input': validated,
        'unsupported': bool(unsupported),
        'unsupported_reason': unsupported,
        'flow_risk': dict(flow_risk) if flow_risk else None,
        'fast_veto': flow_stop,
        'proposal': dict(proposal),
    }


def validate_bfast_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one explicit B off/on identity profile.

    ``model_adapter=internnav`` is only the runner entry; the model identity is
    the Qwen/LISN proposal recorded in this manifest.  A profile that lacks the
    prompt hash or the topic denylist is rejected.
    """
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['payload:not_object']}
    if payload.get('schema') != BFAST_PROFILE_SCHEMA:
        errors.append('schema')
    arm = payload.get('arm')
    if arm not in ('off', 'on'):
        errors.append('arm')
    for key in ('model', 'model_revision', 'prompt_sha256', 'action_mapping',
                'rgb_topic', 'decision_topic'):
        if payload.get(key) in (None, ''):
            errors.append(f'missing:{key}')
    prompt = str(payload.get('prompt_sha256') or '')
    if prompt and not re.fullmatch(r'[0-9a-f]{64}', prompt):
        errors.append('prompt_sha256')
    allowlist = payload.get('topic_allowlist')
    denylist = payload.get('topic_denylist')
    if not isinstance(allowlist, list) or not allowlist:
        errors.append('topic_allowlist')
    if not isinstance(denylist, list) or not denylist:
        errors.append('topic_denylist')
    if isinstance(denylist, list):
        for token in ('laser_scan', 'depth', 'people', 'global_costmap', 'gt_map'):
            if not any(token in str(item).lower() for item in denylist):
                errors.append(f'topic_denylist:missing:{token}')
    if isinstance(allowlist, list):
        for item in allowlist:
            if any(token in str(item).lower() for token in
                   ('laser_scan', '/depth', 'people', 'global_costmap', 'gt_map')):
                errors.append(f'topic_allowlist:forbidden:{item}')
    guard_enabled = payload.get('guard_enabled')
    if not isinstance(guard_enabled, bool):
        errors.append('guard_enabled')
    if arm == 'on':
        if guard_enabled is not True:
            errors.append('on:guard_enabled_must_be_true')
        if not payload.get('veto_topic'):
            errors.append('on:veto_topic')
        if not payload.get('executor_veto_topic'):
            errors.append('on:executor_veto_topic')
        if not isinstance(payload.get('guard_required'), bool):
            errors.append('on:guard_required')
    if arm == 'off':
        if guard_enabled is not False:
            errors.append('off:guard_enabled_must_be_false')
        if payload.get('executor_veto_topic'):
            errors.append('off:executor_veto_topic_must_be_empty')
    return {
        'valid': not errors,
        'errors': errors,
        'arm': arm,
        'model': payload.get('model'),
        'model_revision': payload.get('model_revision'),
        'prompt_sha256': payload.get('prompt_sha256'),
        'action_mapping': payload.get('action_mapping'),
        'identity_source': 'manifest',
        'runner_entry_only': 'model_adapter=internnav is only the runner entry',
        'topic_allowlist': allowlist,
        'topic_denylist': denylist,
    }


def build_bfast_plan(payload: Mapping[str, Any], *,
                     risk_threshold: float | None = None,
                     max_frame_gap_sec: float | None = None,
                     profile_path: str = '') -> dict[str, Any]:
    """Validate a B profile and produce the plan identity + launch arguments."""
    validation = validate_bfast_profile(payload)
    errors = list(validation['errors'])
    arm = validation.get('arm')
    risk = risk_threshold if risk_threshold is not None else payload.get('risk_threshold')
    gap = max_frame_gap_sec if max_frame_gap_sec is not None else payload.get('max_frame_gap_sec')
    if arm == 'on':
        if risk is None:
            errors.append('on:risk_threshold_required')
        elif not _finite(risk) or float(risk) <= 0.0:
            errors.append('on:risk_threshold')
        if gap is None:
            errors.append('on:max_frame_gap_sec_required')
        elif not _finite(gap) or float(gap) <= 0.0:
            errors.append('on:max_frame_gap_sec')
    launch_arguments = [
        f'fast_guard_enabled:={str(arm == "on").lower()}',
        f'fast_guard_veto_topic:={payload.get("executor_veto_topic") or ""}',
        f'fast_guard_required:={str(bool(payload.get("guard_required"))).lower()}',
    ]
    if arm == 'on':
        if risk is not None:
            launch_arguments.append(f'bfast_risk_threshold:={float(risk)}')
        if gap is not None:
            launch_arguments.append(f'bfast_max_frame_gap_sec:={float(gap)}')
        launch_arguments.append(
            f'fast_guard_veto_timeout_sec:={float(payload.get("executor_veto_timeout_sec") or 1.0)}')
    profile_has_setting = (
        payload.get('risk_threshold') is not None
        and payload.get('max_frame_gap_sec') is not None)
    explicit_override = risk_threshold is not None or max_frame_gap_sec is not None
    if arm == 'on' and explicit_override and not profile_has_setting:
        setting_status = 'SMOKE_ONLY_NOT_PAPER_SETTING'
    elif arm == 'on' and not profile_has_setting:
        setting_status = 'MISSING_FROZEN_SETTING'
    elif arm == 'on':
        setting_status = 'PROFILE_SETTING'
    else:
        setting_status = 'NOT_APPLICABLE'
    identity = {
        'schema': 'arena.bfast.plan.v1',
        'profile_path': profile_path,
        'arm': arm,
        'model': payload.get('model'),
        'model_revision': payload.get('model_revision'),
        'prompt_sha256': payload.get('prompt_sha256'),
        'action_mapping': payload.get('action_mapping'),
        'guard_enabled': bool(payload.get('guard_enabled')),
        'guard_required': bool(payload.get('guard_required')),
        'veto_topic': payload.get('executor_veto_topic') or '',
        'guard_timeout_sec': float(payload.get('executor_veto_timeout_sec') or 1.0),
        'risk_threshold': float(risk) if risk is not None else None,
        'max_frame_gap_sec': float(gap) if gap is not None else None,
        'setting_status': setting_status,
        'identity_source': 'manifest',
        'model_adapter_is_runner_entry_only': True,
        'launch_arguments': launch_arguments,
    }
    return {'valid': not errors, 'errors': errors, 'identity': identity}


# ---------------------------------------------------------------------------
# Latency injection
# ---------------------------------------------------------------------------

LATENCY_MODES = ('passthrough', 'delay', 'drop', 'out_of_order')


def inject_latency(*, source_stamp_ns: int | None, source_frame_id: str | None,
                   source_identity: Mapping[str, Any] | None, latency_ms: float,
                   now_sim_ns: int, mode: str = 'delay',
                   previous_response_sim_ns: int | None = None) -> dict[str, Any]:
    """Inject latency/drop/out-of-order without refreshing the source bundle.

    The frozen source stamp, frame, and image hash are copied unchanged.  The
    source age is recomputed from the frozen stamp; it is never refreshed.
    """
    if source_stamp_ns is None or not source_frame_id or source_identity is None:
        raise PipelineRejected('frozen_source_identity_missing')
    if not source_identity.get('image_sha256'):
        raise PipelineRejected('frozen_source_identity_missing:image_sha256')
    if mode not in LATENCY_MODES:
        raise PipelineRejected('latency_mode')
    if not _finite(latency_ms) or float(latency_ms) < 0.0:
        raise PipelineRejected('latency_ms')
    if mode == 'passthrough' and float(latency_ms) != 0.0:
        raise PipelineRejected('passthrough_latency_must_be_zero')
    latency_ns = int(round(float(latency_ms) * 1_000_000))
    dropped = mode == 'drop'
    out_of_order = False
    applied = True
    release_policy = 'immediate'
    if dropped:
        response_sim_ns = None
        source_age_ns = None
        release_time_sim_ns = None
        queue_wait_ms = None
        applied = False
        release_policy = 'dropped_no_release'
    elif mode == 'out_of_order':
        if previous_response_sim_ns is None:
            raise PipelineRejected('previous_response_sim_ns')
        response_sim_ns = int(previous_response_sim_ns) - max(latency_ns, 1)
        source_age_ns = response_sim_ns - int(source_stamp_ns)
        release_time_sim_ns = response_sim_ns
        queue_wait_ms = (release_time_sim_ns - int(now_sim_ns)) / 1_000_000.0
        out_of_order = True
        applied = False
        release_policy = 'buffered_release_order'
    else:
        response_sim_ns = int(now_sim_ns) + latency_ns
        source_age_ns = response_sim_ns - int(source_stamp_ns)
        release_time_sim_ns = response_sim_ns
        queue_wait_ms = latency_ns / 1_000_000.0
    return {
        'schema': 'arena.latency.injection.v1',
        'mode': mode,
        'source_stamp_ns': int(source_stamp_ns),
        'source_frame_id': str(source_frame_id),
        'source_identity': dict(source_identity),
        'requested_latency_ms': float(latency_ms),
        'response_sim_ns': response_sim_ns,
        'source_age_ns': source_age_ns,
        'source_unchanged': True,
        'refreshed': False,
        'dropped': dropped,
        'out_of_order': out_of_order,
        'applied': applied,
        'release_time_sim_ns': release_time_sim_ns,
        'queue_wait_ms': queue_wait_ms,
        'release_policy': release_policy,
        'dropped_reason': 'injected_drop' if dropped else None,
        'metrics': {
            'latency_ms': float(latency_ms),
            'source_age_ns': source_age_ns,
            'queue_wait_ms': queue_wait_ms,
            'release_time_sim_ns': release_time_sim_ns,
            'dropped': dropped,
            'out_of_order': out_of_order,
            'refreshed': False,
        },
    }


class ReleaseQueue:
    """Pollable release dispatcher for injected latency/drop/out-of-order.

    The queue stores the frozen source stamp/frame/identity produced by
    ``inject_latency`` and releases by ``release_time_sim_ns``.  It never
    refreshes the observation.  This is the real dispatcher; wiring it into the
    runner's producer response path remains IMPLEMENTED_UNVERIFIED.
    """

    def __init__(self, *, now_sim_ns_fn=None, now_monotonic_ns_fn=None) -> None:
        self._items: list[dict[str, Any]] = []
        self._next_id = 0
        self._now_sim = now_sim_ns_fn or (lambda: int(time.time() * 1e9))
        self._now_mono = now_monotonic_ns_fn or time.monotonic_ns

    def submit(self, *, source_stamp_ns, source_frame_id, source_identity,
               mode: str = 'delay', latency_ms: float = 0.0,
               previous_response_sim_ns: int | None = None,
               payload: Any = None) -> dict[str, Any]:
        now_sim = int(self._now_sim())
        item = inject_latency(
            source_stamp_ns=source_stamp_ns,
            source_frame_id=source_frame_id,
            source_identity=source_identity,
            latency_ms=latency_ms,
            now_sim_ns=now_sim,
            mode=mode,
            previous_response_sim_ns=previous_response_sim_ns,
        )
        self._next_id += 1
        item['queue_id'] = self._next_id
        item['enqueued_sim_ns'] = now_sim
        item['enqueued_monotonic_ns'] = int(self._now_mono())
        item['payload'] = payload
        item['release_sequence'] = None
        if item['dropped']:
            item['release_policy'] = 'dropped_no_release'
            return item
        item['release_policy'] = (
            'buffered_release_order' if item['out_of_order'] else 'ordered_release')
        self._items.append(item)
        return item

    def poll(self, now_sim_ns: int | None = None) -> list[dict[str, Any]]:
        now = int(now_sim_ns if now_sim_ns is not None else self._now_sim())
        ready = [item for item in self._items
                 if item.get('release_time_sim_ns') is not None
                 and int(item['release_time_sim_ns']) <= now]
        # Buffered release order: queue insertion order, not a re-sort that
        # would let an out-of-order response jump ahead of an earlier item.
        ready.sort(key=lambda item: int(item['queue_id']))
        self._items = [item for item in self._items if item not in ready]
        for sequence, item in enumerate(ready, 1):
            item['release_sequence'] = sequence
            item['released_sim_ns'] = now
            item['queue_wait_at_release_ms'] = (
                now - int(item['enqueued_sim_ns'])) / 1_000_000.0
        return ready

    def pending(self) -> int:
        return len(self._items)


class LatencyCommitGate:
    """Disabled-by-default gate around the existing synchronous commit.

    When disabled it calls ``commit_fn`` immediately (current behavior).  When
    enabled it freezes the payload/source bundle, drops without commit, or
    polls the ROS/sim clock and calls ``commit_fn`` exactly once after release.
    """

    def __init__(self, *, mode: str = 'disabled', latency_ms: float = 0.0,
                 now_sim_ns_fn=None, now_monotonic_ns_fn=None,
                 sleep_fn=None) -> None:
        if mode not in ('disabled',) + LATENCY_MODES:
            raise PipelineRejected('latency_mode')
        self.mode = mode
        self.latency_ms = float(latency_ms)
        self.queue = ReleaseQueue(
            now_sim_ns_fn=now_sim_ns_fn,
            now_monotonic_ns_fn=now_monotonic_ns_fn,
        )
        self.sleep_fn = sleep_fn or time.sleep
        self.receipts: list[dict[str, Any]] = []

    def enqueue(self, *, source_stamp_ns, source_frame_id, source_identity,
                payload: Any, previous_response_sim_ns=None):
        """Non-blocking enqueue.  Returns (item, submit receipt)."""
        if self.mode == 'disabled':
            raise PipelineRejected('latency_gate_disabled')
        item = self.queue.submit(
            source_stamp_ns=source_stamp_ns,
            source_frame_id=source_frame_id,
            source_identity=source_identity,
            mode=self.mode,
            latency_ms=self.latency_ms,
            previous_response_sim_ns=previous_response_sim_ns,
            payload=payload,
        )
        receipt = {
            'schema': 'arena.latency.commit.v1',
            'phase': 'submit',
            'mode': self.mode,
            'queue_id': item['queue_id'],
            'dropped': bool(item['dropped']),
            'dropped_reason': item.get('dropped_reason'),
            'release_time_sim_ns': item.get('release_time_sim_ns'),
            'out_of_order': bool(item.get('out_of_order')),
            'commit_called': False,
            'source_stamp_ns': int(source_stamp_ns),
            'source_frame_id': str(source_frame_id),
            'source_identity': dict(source_identity),
            'source_unchanged': True,
            'refreshed': False,
        }
        self.receipts.append(receipt)
        return item, receipt

    def poll_ready(self, now_sim_ns, commit_fn):
        """Release ready items and call commit_fn exactly once per item."""
        released = self.queue.poll(now_sim_ns)
        results = []
        for item in released:
            result = commit_fn(item['payload'])
            receipt = {
                'schema': 'arena.latency.commit.v1',
                'phase': 'release',
                'mode': self.mode,
                'queue_id': item['queue_id'],
                'dropped': False,
                'release_time_sim_ns': item.get('release_time_sim_ns'),
                'release_sequence': item.get('release_sequence'),
                'queue_wait_at_release_ms': item.get('queue_wait_at_release_ms'),
                'out_of_order': bool(item.get('out_of_order')),
                'commit_called': True,
                'exact_once': True,
                'source_stamp_ns': int(item['source_stamp_ns']),
                'source_frame_id': str(item['source_frame_id']),
                'source_identity': dict(item['source_identity']),
                'source_unchanged': True,
                'refreshed': False,
            }
            self.receipts.append(receipt)
            results.append((item, result, receipt))
        return results

    def commit(self, *, source_stamp_ns, source_frame_id, source_identity,
               payload: Any, commit_fn, previous_response_sim_ns=None):
        if self.mode == 'disabled':
            result = commit_fn(payload)
            receipt = {
                'schema': 'arena.latency.commit.v1',
                'mode': 'disabled',
                'commit_called': True,
                'exact_once': True,
                'source_stamp_ns': int(source_stamp_ns),
                'source_frame_id': str(source_frame_id),
                'source_identity': dict(source_identity),
                'source_unchanged': True,
                'refreshed': False,
            }
            self.receipts.append(receipt)
            return result, receipt
        item = self.queue.submit(
            source_stamp_ns=source_stamp_ns,
            source_frame_id=source_frame_id,
            source_identity=source_identity,
            mode=self.mode,
            latency_ms=self.latency_ms,
            previous_response_sim_ns=previous_response_sim_ns,
            payload=payload,
        )
        if item['dropped']:
            receipt = {
                'schema': 'arena.latency.commit.v1',
                'mode': self.mode,
                'queue_id': item['queue_id'],
                'dropped': True,
                'dropped_reason': item['dropped_reason'],
                'commit_called': False,
                'exact_once': False,
                'source_stamp_ns': int(source_stamp_ns),
                'source_frame_id': str(source_frame_id),
                'source_identity': dict(source_identity),
                'source_unchanged': True,
                'refreshed': False,
            }
            self.receipts.append(receipt)
            return {'decision': 'dropped', 'reason_codes': ['LATENCY_DROP']}, receipt
        while True:
            released = self.queue.poll()
            if released:
                break
            self.sleep_fn(0.01)
        result = commit_fn(item['payload'])
        receipt = {
            'schema': 'arena.latency.commit.v1',
            'mode': self.mode,
            'queue_id': item['queue_id'],
            'dropped': False,
            'release_time_sim_ns': item.get('release_time_sim_ns'),
            'release_sequence': item.get('release_sequence'),
            'queue_wait_at_release_ms': item.get('queue_wait_at_release_ms'),
            'commit_called': True,
            'exact_once': True,
            'out_of_order': bool(item.get('out_of_order')),
            'source_stamp_ns': int(source_stamp_ns),
            'source_frame_id': str(source_frame_id),
            'source_identity': dict(source_identity),
            'source_unchanged': True,
            'refreshed': False,
        }
        self.receipts.append(receipt)
        return result, receipt


# ---------------------------------------------------------------------------
# Dataset split/manifest validation
# ---------------------------------------------------------------------------

def validate_dataset_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['payload:not_object'], 'case_count': 0}
    if payload.get('schema') != DATASET_SCHEMA:
        errors.append('schema')
    split = payload.get('split')
    if not isinstance(split, str) or not split.strip():
        errors.append('split')
    cases = payload.get('cases')
    if not isinstance(cases, list) or not cases:
        errors.append('cases')
        cases = []
    seen: set[str] = set()
    case_semantics: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            errors.append(f'cases[{index}]:not_object')
            continue
        for key in DATASET_REQUIRED_CASE_FIELDS:
            if case.get(key) in (None, ''):
                errors.append(f'cases[{index}]:missing:{key}')
        case_id = case.get('case_id')
        if case_id in seen:
            errors.append(f'cases[{index}]:duplicate:{case_id}')
        seen.add(str(case_id))
        for key in ('gt_map', 'gt_people', 'gt_agents', 'eval_goal', 'oracle_trajectory'):
            if key in case:
                errors.append(f'cases[{index}]:forbidden:{key}')
        semantics = case.get('semantics') if isinstance(case.get('semantics'), Mapping) else {}
        case_semantics[str(case_id)] = {
            'scene_id': semantics.get('scene_id') or case.get('scene_id'),
            'task_id': semantics.get('task_id') or case.get('task_id'),
            'assets': list(semantics.get('assets') or case.get('assets') or []),
        }
    splits = payload.get('splits')
    leakage: list[str] = []
    if splits is not None:
        if not isinstance(splits, Mapping) or not all(key in splits for key in ('train', 'val', 'test')):
            errors.append('splits')
        else:
            assigned: dict[str, str] = {}
            for split_name, ids in splits.items():
                if not isinstance(ids, list):
                    errors.append(f'splits:{split_name}:not_list')
                    continue
                for case_id in ids:
                    text = str(case_id)
                    if text in assigned and assigned[text] != split_name:
                        leakage.append(f'case:{text}:{assigned[text]}/{split_name}')
                    assigned[text] = split_name
                    semantics = case_semantics.get(text) or {}
                    for key in ('scene_id', 'task_id'):
                        value = semantics.get(key)
                        if value:
                            token = f'{key}:{value}'
                            if token in assigned and assigned[token] != split_name:
                                leakage.append(f'{token}:{assigned[token]}/{split_name}')
                            assigned[token] = split_name
                    for asset in semantics.get('assets') or []:
                        token = f'asset:{asset}'
                        if token in assigned and assigned[token] != split_name:
                            leakage.append(f'{token}:{assigned[token]}/{split_name}')
                        assigned[token] = split_name
    if leakage:
        errors.append('split_leakage')
    return {'valid': not errors, 'errors': errors, 'case_count': len(cases),
            'leakage': leakage}


# ---------------------------------------------------------------------------
# Teacher/oracle export manifest (C track)
# ---------------------------------------------------------------------------

TEACHER_SCHEMA = 'arena.paper.teacher_export.v1'
TEACHER_SOURCE_TRACKS = ('C', 'GT', 'oracle')
TEACHER_GT_INPUT_SOURCES = ('gt_map', 'gt_people', 'gt_agents', 'oracle_trajectory',
                            'eval_goal', 'usd_state')


def validate_teacher_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a repo-owned C teacher/oracle export manifest.

    Old GT runs keep their original label.  A/B manifests must not carry GT
    inputs; a teacher manifest must declare its C/GT/oracle source explicitly.
    Instruction QC must be ``required`` with concrete bounds; ``not_set`` is a
    fail-close state, not a silent default.
    """
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['payload:not_object'], 'qc_not_set': False}
    if payload.get('schema') != TEACHER_SCHEMA:
        errors.append('schema')
    source_track = payload.get('source_track')
    if source_track not in TEACHER_SOURCE_TRACKS:
        errors.append('source_track')
        if source_track in ('A', 'B'):
            errors.append('non_gt_mixing')
    runs = payload.get('source_runs')
    if not isinstance(runs, list) or not runs:
        errors.append('source_runs')
        runs = []
    provenance_status = 'OK'
    for index, run in enumerate(runs):
        if isinstance(run, str):
            errors.append(f'source_runs[{index}]:missing_provenance')
            provenance_status = 'MISSING'
            continue
        if not isinstance(run, Mapping):
            errors.append(f'source_runs[{index}]:not_object')
            provenance_status = 'MISSING'
            continue
        for key in ('run_id', 'run_label', 'manifest_sha256'):
            if run.get(key) in (None, ''):
                errors.append(f'source_runs[{index}]:missing:{key}')
                provenance_status = 'MISSING'
        label = str(run.get('run_label') or '').upper()
        if label in ('A', 'B', 'NON_GT', 'NONGT'):
            errors.append(f'source_runs[{index}]:non_gt_label')
        digest = str(run.get('manifest_sha256') or '')
        if digest and not re.fullmatch(r'[0-9a-f]{64}', digest):
            errors.append(f'source_runs[{index}]:manifest_sha256')
        source_files = run.get('source_files')
        if not isinstance(source_files, list) or not source_files:
            errors.append(f'source_runs[{index}]:source_files')
        else:
            for file_index, item in enumerate(source_files):
                if not isinstance(item, Mapping):
                    errors.append(f'source_runs[{index}].source_files[{file_index}]:not_object')
                    continue
                if not item.get('path'):
                    errors.append(f'source_runs[{index}].source_files[{file_index}]:path')
                file_sha = str(item.get('sha256') or '')
                if not re.fullmatch(r'[0-9a-f]{64}', file_sha):
                    errors.append(f'source_runs[{index}].source_files[{file_index}]:sha256')
    inputs = payload.get('inputs')
    if not isinstance(inputs, list) or not inputs:
        errors.append('inputs')
        inputs = []
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping):
            errors.append(f'inputs[{index}]:not_object')
            continue
        for key in ('name', 'source', 'track'):
            if item.get(key) in (None, ''):
                errors.append(f'inputs[{index}]:missing:{key}')
        if item.get('source') in TEACHER_GT_INPUT_SOURCES and item.get('track') not in ('C', 'GT', 'oracle'):
            errors.append(f'inputs[{index}]:gt_mixed_into_non_gt')
    splits = payload.get('splits')
    leakage: list[str] = []
    if not isinstance(splits, Mapping) or not all(key in splits for key in ('train', 'val', 'test')):
        errors.append('splits')
        splits = {}
    for split_name, ids in splits.items():
        if not isinstance(ids, list):
            errors.append(f'splits:{split_name}:not_list')
            continue
        seen: set[str] = set()
        for item in ids:
            text = str(item)
            if text in seen:
                errors.append(f'splits:{split_name}:duplicate:{text}')
            seen.add(text)
    all_ids: dict[str, str] = {}
    for split_name, ids in splits.items():
        if not isinstance(ids, list):
            continue
        for item in ids:
            text = str(item)
            previous = all_ids.get(text)
            if previous and previous != split_name:
                leakage.append(f'{text}:{previous}/{split_name}')
            all_ids[text] = split_name
    semantic_splits = payload.get('semantic_splits')
    if semantic_splits is not None:
        if not isinstance(semantic_splits, Mapping):
            errors.append('semantic_splits')
        else:
            for kind in ('assets', 'scenes', 'tasks'):
                per_split = semantic_splits.get(kind)
                if not isinstance(per_split, Mapping):
                    continue
                assigned: dict[str, str] = {}
                for split_name, ids in per_split.items():
                    if not isinstance(ids, list):
                        errors.append(f'semantic_splits:{kind}:{split_name}:not_list')
                        continue
                    for item in ids:
                        token = f'{kind}:{item}'
                        previous = assigned.get(token)
                        if previous and previous != split_name:
                            leakage.append(f'{token}:{previous}/{split_name}')
                        assigned[token] = split_name
    if leakage:
        errors.append('split_leakage')
    qc = payload.get('instruction_qc')
    qc_not_set = False
    if not isinstance(qc, Mapping):
        errors.append('instruction_qc')
        qc = {}
    status = qc.get('status')
    if status not in ('required', 'not_set'):
        errors.append('instruction_qc:status')
    elif status == 'not_set':
        qc_not_set = True
        errors.append('instruction_qc:not_set')
    elif status == 'required':
        for key in ('min_chars', 'max_chars', 'language'):
            if qc.get(key) in (None, ''):
                errors.append(f'instruction_qc:missing:{key}')
        if not isinstance(qc.get('min_chars'), int):
            errors.append('instruction_qc:min_chars_type')
        if not isinstance(qc.get('max_chars'), int):
            errors.append('instruction_qc:max_chars_type')
        if not isinstance(qc.get('language'), str) or not qc.get('language').strip():
            errors.append('instruction_qc:language_type')
        if (isinstance(qc.get('min_chars'), int) and isinstance(qc.get('max_chars'), int)
                and qc['min_chars'] > qc['max_chars']):
            errors.append('instruction_qc:bounds')
    return {
        'valid': not errors,
        'errors': errors,
        'source_track': source_track,
        'source_run_count': len(runs),
        'input_count': len(inputs),
        'leakage': leakage,
        'qc_not_set': qc_not_set,
        'provenance_status': provenance_status,
        'missing_source_sample': provenance_status == 'MISSING',
        'scene_count': payload.get('scene_count'),
        'episode_count': payload.get('episode_count'),
        'fail_close': bool(errors),
    }


def audit_teacher_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """C provenance audit with an explicit typed blocker."""
    result = validate_teacher_manifest(payload)
    blockers = []
    if result.get('missing_source_sample'):
        blockers.append('TEACHER_SOURCE_SAMPLE_MISSING')
    if result.get('qc_not_set'):
        blockers.append('INSTRUCTION_QC_NOT_SET')
    if result.get('leakage'):
        blockers.append('SPLIT_LEAKAGE')
    if result.get('source_track') in ('A', 'B'):
        blockers.append('NON_GT_MIXING')
    result['typed_blocker'] = '+'.join(blockers) if blockers else None
    result['old_gt_labels_preserved'] = result.get('source_track') in (
        'C', 'GT', 'oracle')
    result['audit_schema'] = 'arena.paper.c_provenance_audit.v1'
    return result


SOURCE_AUDIT_FILES = (
    'run_manifest.yaml', 'CONTROLLER_PEOPLE.json', 'HUMAN_PEOPLE_BRIDGE.json',
    'human_states.csv', 'pedsim_agents_data.csv', 'odom.csv', 'start_goal.csv',
)


def _launch_arg(launch_command, name: str) -> str:
    prefix = f'{name}:='
    for item in launch_command or []:
        text = str(item)
        if text.startswith(prefix):
            return text[len(prefix):]
    return ''


def audit_source_run(source_run: str | Path, derived_root: str | Path,
                     *, provenance: Mapping[str, Any] | None = None
                     ) -> tuple[dict[str, Any], Path]:
    """Read-only audit of a historical source run; receipt only in derived root."""
    source = Path(source_run).resolve()
    derived = Path(derived_root).resolve() / source.name
    if derived.exists():
        raise PipelineRejected(f'derived_run_exists:{derived}')
    derived.mkdir(parents=True)
    before = {
        name: _file_digest(source / name)
        for name in SOURCE_AUDIT_FILES if (source / name).is_file()
    }
    missing = [name for name in SOURCE_AUDIT_FILES if not (source / name).is_file()]
    controller_people = None
    controller_path = source / 'CONTROLLER_PEOPLE.json'
    if controller_path.is_file():
        try:
            controller_people = int(json.loads(controller_path.read_text()).get('people'))
        except (TypeError, ValueError):
            controller_people = None
    bridge_people = None
    published_count = None
    bridge_path = source / 'HUMAN_PEOPLE_BRIDGE.json'
    if bridge_path.is_file():
        try:
            bridge = json.loads(bridge_path.read_text())
            bridge_people = len(bridge.get('people') or [])
            published_count = bridge.get('published_count')
        except (TypeError, ValueError):
            bridge_people = None
    manifest = {}
    manifest_path = source / 'run_manifest.yaml'
    if manifest_path.is_file():
        try:
            manifest = yaml.safe_load(manifest_path.read_text()) or {}
        except Exception:  # noqa: BLE001
            manifest = {}
    launch_command = manifest.get('launch_command') or []
    model_adapter = _launch_arg(launch_command, 'model_adapter')
    model_server = _launch_arg(launch_command, 'model_server_url')
    model_backend = str(manifest.get('model_backend') or '')
    executor_identity = 'UNKNOWN'
    if model_adapter == 'internnav' or '5801/eval_dual' in model_server:
        executor_identity = 'InternNav'
    elif model_adapter == 'lisn_qwen3vl_fp8' or model_backend == 'lisn_qwen3vl_fp8':
        executor_identity = 'LISN/Qwen'
    blockers = []
    human_present = any((source / name).is_file()
                        for name in ('human_states.csv', 'pedsim_agents_data.csv'))
    required_missing = (
        any(name in missing for name in (
            'run_manifest.yaml', 'CONTROLLER_PEOPLE.json',
            'HUMAN_PEOPLE_BRIDGE.json', 'odom.csv', 'start_goal.csv'))
        or not human_present)
    if required_missing:
        blockers.append('MISSING_SOURCE_FILES')
    if controller_people is None:
        blockers.append('UNKNOWN_CONTROLLER_PEOPLE')
    elif controller_people == 0:
        blockers.append('CONTROLLER_PEOPLE_ZERO')
    if executor_identity == 'UNKNOWN':
        blockers.append('UNKNOWN_EXECUTOR_IDENTITY')
    elif executor_identity == 'InternNav':
        blockers.append('EXECUTOR_IDENTITY_INTERNNAV')
    receipt = {
        'schema': 'arena.paper.source_run_provenance_audit.v1',
        'source_run': str(source),
        'derived_root': str(derived),
        'source_files': before,
        'missing_files': missing,
        'controller_people': controller_people,
        'bridge_people': bridge_people,
        'published_count': published_count,
        'model_adapter': model_adapter,
        'model_server_url': model_server,
        'model_backend': model_backend,
        'executor_identity': executor_identity,
        'blockers': blockers,
        'gt_available_for_evaluation': bool(human_present),
        'gt_consumed_by_controller': bool(controller_people and controller_people > 0),
        'source_eligible': not blockers,
        'teacher_export_created': False,
        'old_gt_labels_preserved': True,
        'note': 'Read-only audit; no teacher trajectory is generated.',
    }
    after = {
        name: _file_digest(source / name)
        for name in SOURCE_AUDIT_FILES if (source / name).is_file()
    }
    if before != after:
        blockers.append('SOURCE_HASH_MISMATCH')
    receipt['blockers'] = blockers
    receipt['source_eligible'] = not blockers
    receipt['source_hashes_before'] = before
    receipt['source_hashes_after'] = after
    receipt['source_unchanged'] = before == after
    if provenance is not None:
        require_provenance_authenticity(provenance, expected_run_dir=source)
        receipt['provenance'] = dict(provenance)
    path = derived / 'SOURCE_RUN_PROVENANCE_AUDIT.json'
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8')
    return receipt, path


AGENTIC_REQUEST_SCHEMA = 'arena.agentic.operator_request.v1'
AGENTIC_REQUEST_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
AGENTIC_DEFAULT_PROFILE = 'arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml'
AGENTIC_DEFAULT_HUNAV = 'arena_bringup/configs/hunav/default_sf60.yaml'
AGENTIC_DEFAULT_HUNAV_SHA256 = (
    'a834b6bb42e6c8f14ea4734aa5ab1fd03791a1897f9445d24c2fb7ccade9eeb1')
AGENTIC_UNSUPPORTED_MODES = ('audit', 'full_eval')
AGENTIC_TRACKS = ('generic_internnav', 'strict_b', 'oracle', 'qwen_prox_a')
AUDIT_IDENTITY_SCHEMA = 'arena.agentic.audit_identity.v1'
REQUEST_PROFILE_VALIDITY_BLOCKERS = (
    'PROFILE_REQUIRED', 'PROFILE_INVALID', 'MODEL_PROFILE_MISMATCH',
    'MODEL_REVISION_PROFILE_MISMATCH', 'AUDIT_IDENTITY_INVALID',
    'AUDIT_IDENTITY_MISMATCH', 'HUNAV_CONFIG_REQUIRED', 'CASE_MISSING',
)


def _load_audit_identity(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except Exception as exc:  # noqa: BLE001
        raise PipelineRejected(f'audit_identity:read:{exc}') from exc
    if not isinstance(payload, Mapping):
        raise PipelineRejected('audit_identity:not_object')
    if payload.get('schema') != AUDIT_IDENTITY_SCHEMA:
        raise PipelineRejected('audit_identity:schema')
    for key in ('track', 'model', 'model_revision', 'model_backend',
                'action_control', 'sensor_profile', 'controller',
                'run_profile_id', 'run_profile_sha256'):
        if payload.get(key) in (None, ''):
            raise PipelineRejected(f'audit_identity:missing:{key}')
    if payload.get('launchable') is not False:
        raise PipelineRejected('audit_identity:must_be_audit_only')
    return dict(payload)


def _request_validity_blockers(request_plan: Mapping[str, Any]) -> list[str]:
    return [
        blocker for blocker in (request_plan.get('blockers') or [])
        if blocker.startswith(REQUEST_PROFILE_VALIDITY_BLOCKERS)
    ]


def _is_int_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _resolve_workspace_root(payload: Mapping[str, Any], errors: list[str]) -> Path | None:
    """Resolve the real host workspace, never the source worktree."""
    raw = (payload.get('workspace_root')
           or os.environ.get('ARENA_WS_DIR')
           or os.environ.get('HOST_ARENA_WS_DIR')
           or str(DEFAULT_WORKSPACE_ROOT))
    if not isinstance(raw, str) or not raw.strip():
        errors.append('workspace_root:missing')
        return None
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        errors.append('workspace_root:not_absolute')
        return None
    path = path.resolve()
    if not path.is_dir():
        errors.append('workspace_root:missing')
        return None
    return path


def _resolve_arena_dir(payload: Mapping[str, Any], workspace_root: Path | None,
                       errors: list[str]) -> Path | None:
    raw = payload.get('arena_dir')
    if raw is not None and not isinstance(raw, str):
        errors.append('arena_dir:not_string')
        return None
    if raw:
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            errors.append('arena_dir:not_absolute')
            return None
        path = path.resolve()
    elif workspace_root is not None:
        path = (workspace_root / 'src' / 'Arena').resolve()
    else:
        return None
    if not path.is_dir():
        errors.append('arena_dir:missing')
        return None
    return path


def _resolve_agentic_path(value: Any, *, roots: Sequence[Path], label: str,
                          errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f'{label}:missing')
        return None
    raw = value.strip()
    if os.path.isabs(raw):
        path = Path(raw).expanduser()
    else:
        if '..' in Path(raw).parts:
            errors.append(f'{label}:path_traversal')
            return None
        path = next((root / raw for root in roots if (root / raw).exists()), None)
        if path is None:
            errors.append(f'{label}:missing')
            return None
    try:
        resolved = path.resolve()
    except OSError:
        errors.append(f'{label}:unresolvable')
        return None
    if not resolved.exists():
        errors.append(f'{label}:missing')
        return None
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            break
        except ValueError:
            continue
    else:
        errors.append(f'{label}:outside_allowed_root')
        return None
    return resolved


def _agentic_profile_identity(profile_path: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    server = profile['server']
    runtime = profile['runtime']
    evaluation = profile['evaluation']
    return {
        'profile_id': profile['id'],
        'profile_path': str(profile_path),
        'profile_sha256': benchmark_profile.profile_sha256(profile_path),
        'model': server['mode'],
        'model_revision': Path(server['model_path']).name,
        'model_path': server['model_path'],
        'adapter_target': profile['internnav']['adapter_target'],
        'server_mode': server['mode'],
        'runtime': {
            'device': runtime['device'],
            'http_port': runtime['http_port'],
            'ros_domain_id': runtime['ros_domain_id'],
            'rmw_implementation': runtime['rmw_implementation'],
            'fastdds_builtin_transports': runtime['fastdds_builtin_transports'],
            'ros_localhost_only': runtime['ros_localhost_only'],
            'ros_automatic_discovery_range': runtime['ros_automatic_discovery_range'],
        },
        'evaluation': {
            'world': evaluation['world'],
            'scenario': evaluation['scenario'],
            'robot': evaluation['robot'],
            'episodes': evaluation['episodes'],
            'human_simulator': evaluation['human_simulator'],
            'local_planner': evaluation['local_planner'],
        },
    }


def validate_agentic_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an operator request and resolve its executable plan."""
    errors: list[str] = []
    blockers: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['payload:not_object']}
    if payload.get('schema') != AGENTIC_REQUEST_SCHEMA:
        errors.append('schema')
    for key in ('request_id', 'mode', 'track', 'case', 'profile',
                'model', 'model_revision', 'action_control', 'expected_outputs'):
        if payload.get(key) in (None, ''):
            errors.append(f'missing:{key}')
    request_id = payload.get('request_id')
    if (not isinstance(request_id, str)
            or not AGENTIC_REQUEST_ID_RE.match(request_id)
            or '..' in request_id):
        errors.append('request_id:unsafe')
    case = payload.get('case')
    case_valid = isinstance(case, Mapping)
    if not case_valid:
        errors.append('case')
    else:
        for key in ('world', 'scenario', 'robot'):
            if case.get(key) in (None, ''):
                errors.append(f'case:missing:{key}')
    mode = payload.get('mode')
    if mode not in ('single_case', 'full_eval', 'audit'):
        errors.append('mode')
    track = payload.get('track')
    if track not in AGENTIC_TRACKS:
        errors.append('track')
    if payload.get('action_control') not in ('common_follow_path', 'none'):
        errors.append('action_control')
    outputs = payload.get('expected_outputs')
    if not isinstance(outputs, list) or not outputs:
        errors.append('expected_outputs')
    elif any(not isinstance(item, str) or '..' in item for item in outputs):
        errors.append('expected_outputs:unsafe')
    episodes = payload.get('episodes')
    if not _is_int_value(episodes) or episodes != 1:
        errors.append('episodes:must_be_one')
    seed = payload.get('seed')
    if not _is_int_value(seed) or seed < 0:
        errors.append('seed')
    timeout = payload.get('timeout')
    if not _finite(timeout) or float(timeout) < 1.0:
        errors.append('timeout')
    for key in ('risk_threshold', 'max_frame_gap_sec'):
        value = payload.get(key)
        if value is not None and (not _finite(value) or float(value) <= 0.0):
            errors.append(key)
    if 'smoke_only' in payload and not isinstance(payload['smoke_only'], bool):
        errors.append('smoke_only')
    workspace_root = _resolve_workspace_root(payload, errors)
    arena_dir = _resolve_arena_dir(payload, workspace_root, errors)
    roots = [REPO_ROOT] + ([workspace_root] if workspace_root else [])
    profile_path = _resolve_agentic_path(
        payload.get('profile') or AGENTIC_DEFAULT_PROFILE,
        roots=roots, label='profile', errors=errors)
    hunav_path = _resolve_agentic_path(
        payload.get('hunav_config') or AGENTIC_DEFAULT_HUNAV,
        roots=roots, label='hunav_config', errors=errors)
    hunav_sha = ''
    expected_sha = (payload.get('hunav_config_sha256')
                    or AGENTIC_DEFAULT_HUNAV_SHA256)
    if hunav_path is not None:
        hunav_sha = hashlib.sha256(hunav_path.read_bytes()).hexdigest()
        if not isinstance(expected_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
            errors.append('hunav_config_sha256:format')
        elif hunav_sha != expected_sha:
            errors.append('hunav_config_sha256')
    source_binding_raw = payload.get('source_binding')
    source_binding = (arena_dir if source_binding_raw is None
                      else _resolve_agentic_path(
                          source_binding_raw, roots=roots,
                          label='source_binding', errors=errors))
    identity = {
        'model': payload.get('model'),
        'model_revision': payload.get('model_revision'),
        'action_control': payload.get('action_control'),
        'profile_path': str(profile_path) if profile_path else None,
        'workspace_root': str(workspace_root) if workspace_root else None,
        'arena_dir': str(arena_dir) if arena_dir else None,
        'source_binding': str(source_binding) if source_binding else None,
        'hunav_config': str(hunav_path) if hunav_path else None,
        'hunav_config_sha256': hunav_sha or None,
    }
    commands: list[list[str]] = []
    setting_status = 'NOT_APPLICABLE'
    if mode in AGENTIC_UNSUPPORTED_MODES:
        blockers.append(f'UNSUPPORTED_MODE:{mode}')
    if track == 'qwen_prox_a':
        if not case_valid:
            blockers.append('CASE_MISSING')
        elif payload.get('action_control') != 'common_follow_path':
            blockers.append('A_REQUIRES_COMMON_FOLLOW_PATH')
        elif profile_path is None or not profile_path.is_file():
            blockers.append('PROFILE_REQUIRED')
        else:
            try:
                contract = _load_audit_identity(profile_path)
            except Exception as exc:  # noqa: BLE001
                blockers.append(f'AUDIT_IDENTITY_INVALID:{exc}')
            else:
                identity['audit_identity'] = contract
                if payload.get('model') != contract['model']:
                    blockers.append('MODEL_PROFILE_MISMATCH')
                if payload.get('model_revision') != contract['model_revision']:
                    blockers.append('MODEL_REVISION_PROFILE_MISMATCH')
                resolved_sensor = str(payload.get('sensor_profile') or 'rgb_lidar')
                for field, request_value in (
                        ('track', track),
                        ('action_control', payload.get('action_control')),
                        ('sensor_profile', resolved_sensor),
                        ('controller', contract.get('controller'))):
                    if request_value != contract.get(field):
                        blockers.append(f'AUDIT_IDENTITY_MISMATCH:{field}')
                blockers.append('AUDIT_ONLY_TRACK_NOT_LAUNCHABLE')
                setting_status = 'AUDIT_ONLY_NOT_PAPER_SETTING'
    elif track == 'generic_internnav':
        if not case_valid:
            blockers.append('CASE_MISSING')
        elif payload.get('action_control') != 'common_follow_path':
            blockers.append('GENERIC_REQUIRES_COMMON_FOLLOW_PATH')
        elif profile_path is None or not profile_path.is_file():
            blockers.append('PROFILE_REQUIRED')
        else:
            try:
                profile = benchmark_profile.load_profile(profile_path)
            except Exception as exc:  # noqa: BLE001
                blockers.append(f'PROFILE_INVALID:{exc}')
            else:
                profile_identity = _agentic_profile_identity(profile_path, profile)
                identity['profile'] = profile_identity
                identity['runtime'] = profile_identity['runtime']
                if payload.get('model') != profile_identity['model']:
                    blockers.append('MODEL_PROFILE_MISMATCH')
                if payload.get('model_revision') != profile_identity['model_revision']:
                    blockers.append('MODEL_REVISION_PROFILE_MISMATCH')
                if (arena_dir is None
                        or not (arena_dir / '_meta' / 'docker' / 'features'
                                / 'benchmark' / 'main').is_file()):
                    errors.append('arena_dir:benchmark_entry_missing')
                if (profile_identity['evaluation']['human_simulator'] == 'hunav'
                        and hunav_path is None):
                    blockers.append('HUNAV_CONFIG_REQUIRED')
                if not errors and not blockers:
                    entry = (arena_dir / '_meta' / 'docker' / 'features'
                             / 'benchmark' / 'main')
                    commands = [[
                        str(entry), 'run',
                        '--profile', str(profile_path),
                        '--case', f'{case["world"]}/{case["scenario"]}',
                        '--robot', str(case['robot']),
                        '--episodes', '1',
                        '--pedestrian-seed', str(seed),
                        '--timeout', str(timeout),
                        '--', '--output-prefix', f'agentic/{request_id}',
                    ]]
                    setting_status = ('GENERIC_INTERNAV_SMOKE_ONLY'
                                      if payload.get('smoke_only')
                                      else 'GENERIC_INTERNAV')
    elif track == 'strict_b':
        if payload.get('action_control') != 'common_follow_path':
            blockers.append('STRICT_B_REQUIRES_COMMON_FOLLOW_PATH')
        endpoint = str(payload.get('producer_endpoint') or '')
        if '/v1/chat/completions' not in endpoint:
            blockers.append('STRICT_B_REQUIRES_QWEN_ENDPOINT')
        if profile_path and profile_path.is_file():
            try:
                profile_payload = json.loads(profile_path.read_text(encoding='utf-8'))
                plan = build_bfast_plan(
                    profile_payload,
                    risk_threshold=payload.get('risk_threshold'),
                    max_frame_gap_sec=payload.get('max_frame_gap_sec'),
                    profile_path=str(profile_path),
                )
                if not plan['valid']:
                    errors.extend(plan['errors'])
                else:
                    plan_identity = plan['identity']
                    identity.update({
                        'model': plan_identity['model'],
                        'model_revision': plan_identity['model_revision'],
                        'prompt_sha256': plan_identity['prompt_sha256'],
                        'model_adapter': 'lisn_qwen3vl_fp8',
                        'runner_entry': 'internnav',
                    })
                    setting_status = plan_identity['setting_status']
                    if payload.get('model') != plan_identity['model']:
                        blockers.append('MODEL_PROFILE_MISMATCH')
                    if payload.get('model_revision') != plan_identity['model_revision']:
                        blockers.append('MODEL_REVISION_PROFILE_MISMATCH')
            except Exception as exc:  # noqa: BLE001
                errors.append(f'profile:read:{exc}')
        blockers.append('QWEN_LISN_COMMON_METRIC_ACTION_BRIDGE_MISSING')
    elif track == 'oracle':
        blockers.append('ORACLE_NOT_OPERATOR_EXECUTABLE')
    executable = not blockers and not errors
    env: dict[str, str] = {}
    if executable and isinstance(identity.get('runtime'), Mapping):
        runtime = identity['runtime']
        env = {
            'ARENA_WS_DIR': str(workspace_root),
            'HOST_ARENA_WS_DIR': str(workspace_root),
            'ARENA_DIR': str(arena_dir),
            'ARENA_PROJECT_NAME': 'arena-arena_jazzy_ws',
            'ARENA_IMAGE': 'arena:dev',
            'ROS_DOMAIN_ID': str(runtime['ros_domain_id']),
            'RMW_IMPLEMENTATION': str(runtime['rmw_implementation']),
            'FASTDDS_BUILTIN_TRANSPORTS': str(runtime['fastdds_builtin_transports']),
            'ROS_LOCALHOST_ONLY': str(runtime['ros_localhost_only']),
            'ROS_AUTOMATIC_DISCOVERY_RANGE': str(runtime['ros_automatic_discovery_range']),
            'ARENA_BENCHMARK_DEVICE': str(runtime['device']),
        }
        if hunav_path is not None:
            env['ARENA_HUNAV_CONFIG'] = str(hunav_path)
            env['ARENA_HUNAV_CONFIG_SHA256'] = hunav_sha
    return {
        'schema': 'arena.agentic.operator_plan.v1',
        'valid': not errors,
        'errors': errors,
        'request_id': request_id,
        'mode': mode,
        'track': track,
        'executable': executable,
        'commands': commands if executable else [],
        'env': env,
        'blockers': blockers,
        'setting_status': setting_status,
        'expected_outputs': outputs if isinstance(outputs, list) else [],
        'identity': identity,
    }


OPERATOR_LIFECYCLE_SCHEMA = 'arena.agentic.operator_lifecycle.v1'
OPERATOR_ACTIONS = ('plan', 'start', 'cancel', 'finalize', 'read', 'audit')
OPERATOR_STATE_TRANSITIONS = {
    'planned': ('start', 'cancel', 'read', 'audit'),
    'started': ('cancel', 'read', 'audit', 'finalize'),
    'cancelled': ('read', 'audit'),
    'read_ready': ('audit',),
    'finalize_ready': ('read', 'audit'),
    'audit_ready': (),
    'blocked': (),
}
OPERATOR_ACTION_FROM_STATES = {
    'plan': ('new', 'planned', 'blocked'),
    'start': ('planned',),
    'cancel': ('planned', 'started'),
    'finalize': ('started',),
    'read': ('planned', 'started', 'cancelled', 'read_ready',
             'finalize_ready', 'audit_ready'),
    'audit': ('planned', 'started', 'cancelled', 'read_ready',
              'finalize_ready', 'audit_ready'),
}
OPERATOR_EXIT_TAXONOMY = {
    'ok': 0,
    'scientific_fail': 1,
    'contract_or_evidence_blocked': 2,
}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def _paper_pipeline_argv(*args: str) -> list[str]:
    return ['python3', '-m', 'arena_bringup.paper_pipeline', *args]


def _lifecycle_workspace_root(payload: Mapping[str, Any]) -> Path | None:
    raw = (payload.get('workspace_root')
           or os.environ.get('ARENA_WS_DIR')
           or os.environ.get('HOST_ARENA_WS_DIR')
           or str(DEFAULT_WORKSPACE_ROOT))
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else None


def _resolve_lifecycle_run_dir(payload: Mapping[str, Any],
                               workspace_root: Path | None,
                               errors: list[str]) -> Path | None:
    raw = payload.get('run_dir')
    if not isinstance(raw, str) or not raw.strip():
        errors.append('run_dir:missing')
        return None
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        if workspace_root is None:
            errors.append('run_dir:unresolvable')
            return None
        path = workspace_root / path
    path = path.resolve()
    if not path.is_dir():
        errors.append('run_dir:missing')
        return None
    return path


def _resolve_lifecycle_output_root(payload: Mapping[str, Any],
                                   workspace_root: Path | None,
                                   run_dir: Path | None,
                                   errors: list[str]) -> Path | None:
    raw = payload.get('output_root')
    if not isinstance(raw, str) or not raw.strip():
        errors.append('output_root:missing')
        return None
    if '..' in Path(raw.strip()).parts:
        errors.append('output_root:path_traversal')
        return None
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        if workspace_root is None:
            errors.append('output_root:unresolvable')
            return None
        path = workspace_root / path
    path = path.resolve()
    if path.exists():
        errors.append('output_root:exists')
        return None
    if run_dir is not None:
        try:
            path.relative_to(run_dir)
        except ValueError:
            pass
        else:
            errors.append('output_root:inside_run')
            return None
    return path


def _lifecycle_env(request_plan: Mapping[str, Any]) -> dict[str, str]:
    env = dict(request_plan.get('env') or {})
    arena_dir = (request_plan.get('identity') or {}).get('arena_dir')
    if arena_dir:
        env['PYTHONPATH'] = str(Path(arena_dir) / 'arena_bringup')
    env['ARENA_REPO_ROOT'] = str(REPO_ROOT)
    return env


def build_operator_lifecycle(payload: Mapping[str, Any], action: str,
                             from_state: str | None = None) -> dict[str, Any]:
    """Build a deterministic operator response for one lifecycle action.

    ``plan``/``cancel`` bind existing benchmark/main verbs.  ``read``,
    ``audit`` and ``finalize`` bind read-only paper_pipeline evidence tools
    into a fresh output root.  ``start`` stays blocked until the runtime entry
    has a real idempotency guard; a hash alone is not claimed as dedupe.
    """
    request_plan = validate_agentic_request(payload)
    identity = dict(request_plan.get('identity') or {})
    idempotency_key = 'sha256:' + _canonical_sha256({
        'request_id': payload.get('request_id'),
        'action': action,
        'from_state': from_state,
        'track': payload.get('track'),
        'mode': payload.get('mode'),
        'case': payload.get('case'),
        'episodes': payload.get('episodes'),
        'seed': payload.get('seed'),
        'timeout': payload.get('timeout'),
        'run_dir': payload.get('run_dir'),
        'output_root': payload.get('output_root'),
        'identity': identity,
    })
    commands: list[list[str]] = []
    blockers: list[str] = []
    errors: list[str] = list(request_plan.get('errors') or [])
    state = 'blocked'
    allowed_next: tuple[str, ...] = ()
    preconditions: list[str] = []
    cleanup: list[str] = []
    expected_outputs: list[str] = []
    output_root: Path | None = None
    run_dir: Path | None = None
    audit_manifest: dict[str, Any] | None = None
    run_identity: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    sensor_profile = str(payload.get('sensor_profile') or (
        'rgb_only' if payload.get('track') == 'strict_b' else 'rgb_lidar'))
    if sensor_profile not in SENSOR_PROFILES:
        errors.append(f'sensor_profile:{sensor_profile}')
    if action not in OPERATOR_ACTIONS:
        blockers.append(f'ACTION_UNKNOWN:{action}')
    elif from_state is not None and from_state not in OPERATOR_ACTION_FROM_STATES[action]:
        blockers.append(f'INVALID_STATE_TRANSITION:{from_state}->{action}')
    elif not request_plan.get('valid'):
        blockers.extend(f'REQUEST_INVALID:{error}' for error in errors or ['unknown'])
    elif action == 'plan':
        if request_plan.get('executable'):
            run_command = list(request_plan['commands'][0])
            commands = [[run_command[0], 'plan', 'run', *run_command[2:]]]
            state = 'planned'
            allowed_next = OPERATOR_STATE_TRANSITIONS['planned']
            expected_outputs = list(request_plan.get('expected_outputs') or [])
        else:
            blockers.extend(request_plan.get('blockers') or ['REQUEST_NOT_EXECUTABLE'])
    elif action == 'start':
        blockers.append('START_RUNTIME_IDEMPOTENCY_GUARD_MISSING')
    elif action == 'cancel':
        arena_dir = identity.get('arena_dir')
        if not arena_dir:
            blockers.append('ARENA_DIR_REQUIRED')
        else:
            entry = Path(str(arena_dir)) / '_meta' / 'docker' / 'features' / 'benchmark' / 'main'
            if not entry.is_file():
                blockers.append('BENCHMARK_ENTRY_MISSING')
            else:
                commands = [[str(entry), 'stop']]
                state = 'cancelled'
                allowed_next = OPERATOR_STATE_TRANSITIONS['cancelled']
                preconditions = ['owned_run_identity']
                cleanup = [
                    'verify LOCK_FREE',
                    'verify no model/Isaac/eval process remains',
                ]
    elif action == 'finalize':
        blockers.append('ACTION_NOT_BOUND:finalize')
    elif action in ('read', 'audit'):
        validity_blockers = _request_validity_blockers(request_plan)
        if validity_blockers:
            blockers.extend(validity_blockers)
        else:
            workspace_root = _lifecycle_workspace_root(payload)
            run_dir = _resolve_lifecycle_run_dir(payload, workspace_root, errors)
            output_root = _resolve_lifecycle_output_root(
                payload, workspace_root, run_dir, errors)
            if errors:
                blockers.extend(f'REQUEST_INVALID:{error}' for error in errors)
            else:
                assert run_dir is not None and output_root is not None
                try:
                    run_identity = extract_run_identity(run_dir)
                except Exception as exc:  # noqa: BLE001
                    blockers.append(f'RUN_IDENTITY_UNRESOLVED:{exc}')
                else:
                    mismatches = bind_request_to_run(
                        payload, request_plan, run_identity)
                    if mismatches:
                        blockers.extend(mismatches)
                    else:
                        provenance = build_run_provenance(
                            payload.get('request_id'), run_identity)
                        provenance_path = output_root / 'provenance.json'
                        index_path = output_root / 'evidence_index.json'
                        verdict_path = output_root / 'evidence_verdict.json'
                        commands = [
                            _paper_pipeline_argv(
                                'evidence-index', '--run-dir', str(run_dir),
                                '--output', str(index_path),
                                '--sensor-profile', sensor_profile,
                                '--provenance-file', str(provenance_path)),
                            _paper_pipeline_argv(
                                'evidence-verdict', '--index', str(index_path),
                                '--output', str(verdict_path),
                                '--provenance-file', str(provenance_path)),
                        ]
                        expected_outputs = [
                            'provenance.json', 'evidence_index.json',
                            'evidence_verdict.json']
                        cleanup = [
                            'write the inline provenance to provenance_path first',
                            'verify output_root is outside the source run',
                            'verify no source file changed',
                            'keep scientific FAIL as exit 1, never rewrite it as success',
                        ]
                        if action == 'read':
                            state = 'read_ready'
                            allowed_next = OPERATOR_STATE_TRANSITIONS['read_ready']
                        else:
                            benchmark_result_path = output_root / 'benchmark_result.json'
                            commands.append([
                                'python3', '-m', 'arena_bringup.benchmark_result',
                                '--dir', str(run_dir),
                                '--output', str(benchmark_result_path),
                                '--require-valid',
                                '--provenance-file', str(provenance_path),
                            ])
                            audit_manifest = {
                                'schema': 'arena.agentic.audit_manifest.v1',
                                'request_id': payload.get('request_id'),
                                'track': payload.get('track'),
                                'run_dir': str(run_dir),
                                'output_root': str(output_root),
                                'provenance': provenance,
                                'request_identity': {
                                    'track': payload.get('track'),
                                    'model': payload.get('model'),
                                    'model_revision': payload.get('model_revision'),
                                    'case': payload.get('case'),
                                    'seed': payload.get('seed'),
                                    'episodes': payload.get('episodes'),
                                    'action_control': payload.get('action_control'),
                                    'sensor_profile': sensor_profile,
                                },
                                'receipts': [
                                    {'name': 'benchmark_result',
                                     'path': str(benchmark_result_path),
                                     'required': True},
                                    {'name': 'evidence_index',
                                     'path': str(index_path),
                                     'required': True},
                                    {'name': 'evidence_verdict',
                                     'path': str(verdict_path),
                                     'required': True},
                                ],
                            }
                            if payload.get('track') == 'oracle':
                                source_root = output_root / 'source_audit'
                                commands.append(_paper_pipeline_argv(
                                    'source-run-audit',
                                    '--source-run', str(run_dir),
                                    '--derived-root', str(source_root),
                                    '--provenance-file', str(provenance_path)))
                                expected_outputs.append('source_run_audit.json')
                                audit_manifest['receipts'].append({
                                    'name': 'source_run_audit',
                                    'path': str(source_root / run_dir.name
                                                 / 'SOURCE_RUN_PROVENANCE_AUDIT.json'),
                                    'required': True,
                                })
                            else:
                                metrics_root = output_root / 'metrics_derived'
                                commands.append(_paper_pipeline_argv(
                                    'metrics',
                                    '--run-dir', str(run_dir),
                                    '--derived-root', str(metrics_root),
                                    '--provenance-file', str(provenance_path)))
                                expected_outputs.append('metrics_receipt.json')
                                audit_manifest['receipts'].append({
                                    'name': 'metrics_receipt',
                                    'path': str(metrics_root / run_dir.name
                                                / 'PAPER_METRICS_RECEIPT.json'),
                                    'required': True,
                                })
                            manifest_path = output_root / 'audit_manifest.json'
                            consolidated_path = output_root / 'audit_consolidated.json'
                            commands.append(_paper_pipeline_argv(
                                'audit-consolidate',
                                '--manifest', str(manifest_path),
                                '--output', str(consolidated_path)))
                            expected_outputs.append('audit_consolidated.json')
                            state = 'audit_ready'
                            allowed_next = OPERATOR_STATE_TRANSITIONS['audit_ready']
    executable = bool(commands) and not blockers and request_plan.get('valid')
    return {
        'schema': OPERATOR_LIFECYCLE_SCHEMA,
        'request_id': payload.get('request_id'),
        'action': action,
        'from_state': from_state,
        'state': state,
        'executable': executable,
        'commands': commands if executable else [],
        'env': _lifecycle_env(request_plan) if executable else {},
        'blockers': blockers,
        'errors': errors,
        'exit_taxonomy': dict(OPERATOR_EXIT_TAXONOMY),
        'expected_outputs': expected_outputs if executable else [],
        'allowed_next_actions': list(allowed_next),
        'preconditions': preconditions,
        'cleanup': cleanup,
        'idempotency_key': idempotency_key,
        'identity': identity,
        'request_valid': bool(request_plan.get('valid')),
        'run_dir': str(run_dir) if run_dir else None,
        'output_root': str(output_root) if output_root else None,
        'sensor_profile': sensor_profile,
        'run_identity': run_identity,
        'provenance': provenance,
        'provenance_path': (
            str(output_root / 'provenance.json')
            if provenance is not None and output_root is not None else None),
        'audit_manifest': audit_manifest if (
            executable and action == 'audit') else None,
    }


EVIDENCE_INDEX_FILES = (
    'benchmark_result.json', 'run_manifest.yaml', 'metrics.csv',
    'vln_task_metrics.json', 'social_metrics.json', 'artifact_validation.json',
    'model_trace.jsonl', 'model_diagnostic_summary.json', 'model_status.json',
    'timing_summary.json', 'cmd_vel.csv', 'cmd_vel_trace.jsonl', 'odom.csv',
    'human_states.csv', 'pedsim_agents_data.csv', 'start_goal.csv',
    'scan.csv', 'video_index.json', 'episode.csv', 'episode_outcome.json',
    'path_execution.jsonl', 'rtf.csv', 'recording_segments.csv',
    'overlay_provenance.json', 'params.yaml',
)
EVIDENCE_INDEX_GLOBS = ('executor_*.jsonl',)
SENSOR_PROFILES = {
    'rgb_lidar': ('rgb', 'camera_info', 'laser_scan'),
    'rgb_only': ('rgb', 'camera_info'),
    'none': (),
}
RUN_IDENTITY_SCHEMA = 'arena.agentic.run_identity.v1'
PROVENANCE_SCHEMA = 'arena.agentic.provenance.v1'
MODEL_BACKEND_IDENTITIES = {
    'lisn_qwen3vl_fp8': {
        'model': 'Qwen/Qwen3-VL-8B-Instruct-FP8',
        'model_revision': '9cdc6310a8cb770ce18efaf4e9935334512aee45',
    },
    'internnav': {
        'model': 'internnav',
        'model_revision': 'InternVLA-N1-DualVLN',
    },
    'lightnav': {'model': 'lightnav', 'model_revision': ''},
    'navila': {'model': 'navila', 'model_revision': ''},
}
RUN_IDENTITY_BINDING_FIELDS = (
    'track', 'model', 'model_revision', 'profile_id', 'profile_sha256',
    'case', 'seed', 'episodes', 'action_control', 'sensor_profile',
)


def _read_run_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / 'run_manifest.yaml'
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_run_result(run_dir: Path) -> dict[str, Any]:
    path = run_dir / 'benchmark_result.json'
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_run_identity(run_dir: str | Path) -> dict[str, Any]:
    """Resolve the actual model/case identity of an existing run.

    The actual model comes from ``model_backend``/``model_provenance``, never
    from a legacy launch/profile ``model_adapter`` field.
    """
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise PipelineRejected(f'run_dir:missing:{run}')
    manifest = _read_run_manifest(run)
    result = _read_run_result(run)
    parameters = manifest.get('parameters') if isinstance(
        manifest.get('parameters'), Mapping) else {}
    model_provenance = manifest.get('model_provenance') if isinstance(
        manifest.get('model_provenance'), Mapping) else {}
    result_execution = result.get('execution') if isinstance(
        result.get('execution'), Mapping) else {}
    result_run = result.get('run') if isinstance(result.get('run'), Mapping) else {}
    model_backend = str(
        manifest.get('model_backend')
        or model_provenance.get('model')
        or parameters.get('model_backend')
        or result_execution.get('model_backend')
        or '')
    actual = MODEL_BACKEND_IDENTITIES.get(model_backend)
    model = str(actual['model']) if actual else model_backend
    model_revision = str(actual['model_revision']) if actual else ''
    profile = manifest.get('benchmark_profile') if isinstance(
        manifest.get('benchmark_profile'), Mapping) else {}
    profile_id = str(profile.get('id') or '')
    profile_sha256 = str(profile.get('sha256') or '')
    snapshot = profile.get('snapshot_path')
    if profile_sha256 and snapshot and Path(str(snapshot)).is_file():
        pass
    control_path = str(
        parameters.get('control_path')
        or result_execution.get('control_path')
        or '')
    controller = str(parameters.get('controller') or '')
    if 'strict_b' in model_backend or 'bfast' in model_backend:
        track = 'strict_b'
    elif model_backend == 'lisn_qwen3vl_fp8' and (
            'prox_mpc' in control_path or controller == 'prox_mpc'):
        track = 'qwen_prox_a'
    elif ('nav2_follow_path' in control_path or 'prox_mpc' in control_path
          or 'dual_vln' in control_path):
        track = 'generic_internnav'
    else:
        track = 'oracle'
    sensor_profile = 'rgb_only' if track == 'strict_b' else 'rgb_lidar'
    world = str(parameters.get('world') or result_run.get('world') or '')
    scenario = str(
        parameters.get('scenario_file') or result_run.get('scenario') or '')
    robot = str(parameters.get('robot') or result_run.get('robot') or '')
    seed = parameters.get('pedestrian_seed', result_run.get('pedestrian_seed'))
    episodes = parameters.get('episodes', result_run.get('episodes_requested'))
    timeout = parameters.get('timeout')
    scan_topic = str(parameters.get('eval_video_scan_topic') or '')
    rgb_topic = str(parameters.get('vln_rgb_topic') or '')
    identity = {
        'schema': RUN_IDENTITY_SCHEMA,
        'run_dir': str(run),
        'run_name': run.name,
        'track': track,
        'model': model,
        'model_revision': model_revision,
        'model_backend': model_backend,
        'model_identity_source': 'model_backend',
        'model_identity_known': bool(actual and model_revision),
        'profile_id': profile_id,
        'profile_sha256': profile_sha256,
        'case': {'world': world, 'scenario': scenario, 'robot': robot},
        'seed': seed,
        'episodes': episodes,
        'timeout': timeout,
        'control_path': control_path,
        'controller': controller,
        'scan_topic': scan_topic,
        'rgb_topic': rgb_topic,
        'action_control': (
            'common_follow_path' if 'nav2_follow_path' in control_path
            else 'none'),
        'sensor_profile': sensor_profile,
    }
    identity['run_identity_digest'] = 'sha256:' + _canonical_sha256(identity)
    tree = source_hash_manifest(run)
    tree_view = tree.get('tree') or {}
    identity['source_tree_digest'] = 'sha256:' + _canonical_sha256(tree_view)
    file_set = sorted(
        (name, record.get('size')) for name, record in tree_view.items())
    identity['source_file_set_digest'] = 'sha256:' + _canonical_sha256(file_set)
    return identity


def build_run_provenance(request_id: Any,
                         identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'schema': PROVENANCE_SCHEMA,
        'request_id': request_id,
        'run_dir': identity['run_dir'],
        'run_name': identity['run_name'],
        'run_identity': {
            key: value for key, value in identity.items()
            if key not in ('run_identity_digest', 'source_tree_digest',
                           'source_file_set_digest')
        },
        'run_identity_digest': identity['run_identity_digest'],
        'source_tree_digest': identity['source_tree_digest'],
        'source_file_set_digest': identity['source_file_set_digest'],
        'immutable': True,
    }


def validate_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['provenance:not_object']}
    if payload.get('schema') != PROVENANCE_SCHEMA:
        errors.append('provenance:schema')
    for key in ('request_id', 'run_dir', 'run_identity', 'run_identity_digest',
                'source_tree_digest', 'source_file_set_digest'):
        if payload.get(key) in (None, ''):
            errors.append(f'provenance:missing:{key}')
    if payload.get('immutable') is not True:
        errors.append('provenance:immutable')
    return {'valid': not errors, 'errors': errors}


def recompute_provenance(request_id: Any, run_dir: str | Path) -> dict[str, Any]:
    return build_run_provenance(request_id, extract_run_identity(run_dir))


PROVENANCE_AUTHENTICITY_FIELDS = (
    'request_id', 'run_dir', 'run_identity', 'run_identity_digest',
    'source_tree_digest', 'source_file_set_digest', 'immutable',
)


def verify_provenance_authenticity(
        provenance: Mapping[str, Any], *,
        expected_request_id: Any = None,
        expected_run_dir: str | Path | None = None) -> dict[str, Any]:
    """Recompute the envelope from request_id + canonical run_dir.

    Receipts that merely agree with each other are not enough: the envelope
    must match a fresh recomputation over the immutable source run.
    """
    validation = validate_provenance(provenance)
    errors = list(validation['errors'])
    if errors:
        return {'valid': False, 'errors': errors}
    run_dir = Path(str(provenance['run_dir'])).expanduser().resolve()
    if expected_run_dir is not None:
        expected = Path(str(expected_run_dir)).expanduser().resolve()
        if run_dir != expected:
            errors.append('PROVENANCE_AUTHENTICITY_MISMATCH:run_dir')
    if expected_request_id is not None and provenance.get('request_id') != expected_request_id:
        errors.append('PROVENANCE_AUTHENTICITY_MISMATCH:request_id')
    try:
        recomputed = recompute_provenance(provenance.get('request_id'), run_dir)
    except Exception as exc:  # noqa: BLE001
        return {
            'valid': False,
            'errors': [f'PROVENANCE_AUTHENTICITY_MISMATCH:run_dir:{exc}'],
        }
    for field in ('run_identity', 'run_identity_digest',
                  'source_tree_digest', 'source_file_set_digest'):
        if provenance.get(field) != recomputed.get(field):
            errors.append(f'PROVENANCE_AUTHENTICITY_MISMATCH:{field}')
    return {'valid': not errors, 'errors': errors}


def require_provenance_authenticity(
        provenance: Mapping[str, Any], *,
        expected_request_id: Any = None,
        expected_run_dir: str | Path | None = None) -> dict[str, Any]:
    result = verify_provenance_authenticity(
        provenance, expected_request_id=expected_request_id,
        expected_run_dir=expected_run_dir)
    if not result['valid']:
        raise PipelineRejected(','.join(result['errors']))
    return result


def _request_profile_sha256(request_plan: Mapping[str, Any]) -> str:
    path_value = (request_plan.get('identity') or {}).get('profile_path')
    if not path_value:
        return ''
    path = Path(str(path_value))
    if not path.is_file():
        return ''
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_profile_id(request_plan: Mapping[str, Any]) -> str:
    path_value = (request_plan.get('identity') or {}).get('profile_path')
    if not path_value:
        return ''
    path = Path(str(path_value))
    if not path.is_file():
        return ''
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return ''
    if isinstance(payload, Mapping):
        return str(payload.get('id') or '')
    return ''


def bind_request_to_run(payload: Mapping[str, Any],
                        request_plan: Mapping[str, Any],
                        identity: Mapping[str, Any]) -> list[str]:
    """Return typed ``RUN_IDENTITY_MISMATCH:<field>`` entries."""
    mismatches: list[str] = []
    if not identity.get('model_identity_known'):
        mismatches.append('RUN_IDENTITY_UNKNOWN:model_identity')
    if payload.get('track') != identity.get('track'):
        mismatches.append('RUN_IDENTITY_MISMATCH:track')
    if payload.get('model') != identity.get('model'):
        mismatches.append('RUN_IDENTITY_MISMATCH:model')
    if payload.get('model_revision') != identity.get('model_revision'):
        mismatches.append('RUN_IDENTITY_MISMATCH:model_revision')
    if payload.get('track') == 'qwen_prox_a':
        contract = (request_plan.get('identity') or {}).get('audit_identity') or {}
        for field in ('track', 'model', 'model_revision', 'action_control',
                      'sensor_profile', 'controller'):
            if contract.get(field) != identity.get(field):
                mismatches.append(f'RUN_IDENTITY_MISMATCH:{field}')
        if contract.get('run_profile_id') != identity.get('profile_id'):
            mismatches.append('RUN_IDENTITY_MISMATCH:run_profile_id')
        if contract.get('run_profile_sha256') != identity.get('profile_sha256'):
            mismatches.append('RUN_IDENTITY_MISMATCH:run_profile_sha256')
    else:
        profile_sha = _request_profile_sha256(request_plan)
        if not profile_sha or profile_sha != identity.get('profile_sha256'):
            mismatches.append('RUN_IDENTITY_MISMATCH:profile_sha256')
        profile_id = _request_profile_id(request_plan)
        if not profile_id or profile_id != identity.get('profile_id'):
            mismatches.append('RUN_IDENTITY_MISMATCH:profile_id')
    case = payload.get('case') if isinstance(payload.get('case'), Mapping) else {}
    expected_case = identity.get('case') or {}
    for key in ('world', 'scenario', 'robot'):
        if case.get(key) != expected_case.get(key):
            mismatches.append(f'RUN_IDENTITY_MISMATCH:case.{key}')
    if payload.get('seed') != identity.get('seed'):
        mismatches.append('RUN_IDENTITY_MISMATCH:seed')
    if payload.get('episodes') != identity.get('episodes'):
        mismatches.append('RUN_IDENTITY_MISMATCH:episodes')
    if payload.get('action_control') != identity.get('action_control'):
        mismatches.append('RUN_IDENTITY_MISMATCH:action_control')
    sensor_profile = str(payload.get('sensor_profile') or (
        'rgb_only' if payload.get('track') == 'strict_b' else 'rgb_lidar'))
    if sensor_profile != identity.get('sensor_profile'):
        mismatches.append('RUN_IDENTITY_MISMATCH:sensor_profile')
    return mismatches


METHOD_BUNDLE_SCHEMA = 'arena.agentic.method_bundle.v1'
COMMON_CONTRACT_SCHEMA = 'arena.agentic.common_contract.v1'
METHOD_PLAN_SCHEMA = 'arena.agentic.method_plan.v1'
METHOD_PLAN_COMPARISON_SCHEMA = 'arena.agentic.method_plan_comparison.v1'
METHOD_IDS = (
    'internnav', 'navila', 'lightnav', 'qwen_agent_proxmpc',
    'strict_b_on', 'strict_b_off', 'socialmpc_oracle',
)
METHOD_COMMON_KEYS = (
    'case', 'seed', 'episodes', 'instruction', 'hunav', 'timeout',
    'metrics', 'video', 'recording',
)
METHOD_PRIVATE_KEYS = (
    'profile', 'model', 'sensors', 'adapter', 'controller', 'ros',
    'output', 'lock', 'cleanup', 'source_binding',
)


def validate_common_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['common_contract:not_object']}
    if payload.get('schema') != COMMON_CONTRACT_SCHEMA:
        errors.append('common_contract:schema')
    case = payload.get('case')
    if not isinstance(case, Mapping):
        errors.append('common_contract:case')
    else:
        for key in ('world', 'scenario', 'robot'):
            if case.get(key) in (None, ''):
                errors.append(f'common_contract:case:{key}')
    if not _is_int_value(payload.get('seed')) or payload.get('seed') < 0:
        errors.append('common_contract:seed')
    if not _is_int_value(payload.get('episodes')) or payload.get('episodes') != 1:
        errors.append('common_contract:episodes')
    timeout = payload.get('timeout')
    if not isinstance(timeout, Mapping) or not _finite(timeout.get('sim_sec')):
        errors.append('common_contract:timeout')
    elif float(timeout['sim_sec']) <= 0:
        errors.append('common_contract:timeout')
    hunav = payload.get('hunav')
    if not isinstance(hunav, Mapping):
        errors.append('common_contract:hunav')
    else:
        if hunav.get('backend') != 'hunav':
            errors.append('common_contract:hunav:backend')
        if not isinstance(hunav.get('config'), str) or not hunav['config']:
            errors.append('common_contract:hunav:config')
        if not re.fullmatch(r'[0-9a-f]{64}', str(hunav.get('sha256') or '')):
            errors.append('common_contract:hunav:sha256')
    for key in ('instruction', 'metrics', 'video', 'recording'):
        if not isinstance(payload.get(key), Mapping):
            errors.append(f'common_contract:{key}')
    contract_hash = 'sha256:' + _canonical_sha256(payload)
    return {
        'valid': not errors,
        'errors': errors,
        'contract_hash': contract_hash,
        'contract': dict(payload) if isinstance(payload, Mapping) else {},
    }


def validate_method_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['method_bundle:not_object']}
    if payload.get('schema') != METHOD_BUNDLE_SCHEMA:
        errors.append('method_bundle:schema')
    method_id = payload.get('method_id')
    if method_id not in METHOD_IDS:
        errors.append('method_bundle:method_id')
    if payload.get('status') not in ('READY', 'BLOCKED'):
        errors.append('method_bundle:status')
    if not isinstance(payload.get('launchable'), bool):
        errors.append('method_bundle:launchable')
    blockers = payload.get('blockers')
    if not isinstance(blockers, list):
        errors.append('method_bundle:blockers')
    elif payload.get('status') == 'BLOCKED' and not blockers:
        errors.append('method_bundle:blocked_without_blocker')
    elif payload.get('status') == 'READY' and blockers:
        errors.append('method_bundle:ready_with_blocker')
    overrides = sorted(set(payload) & set(METHOD_COMMON_KEYS))
    for key in overrides:
        errors.append(f'METHOD_OVERRIDES_COMMON:{key}')
    for key in ('profile', 'model', 'sensors', 'adapter', 'controller',
                'ros', 'output', 'lock'):
        if not isinstance(payload.get(key), Mapping):
            errors.append(f'method_bundle:{key}')
    profile = payload.get('profile') if isinstance(payload.get('profile'), Mapping) else {}
    required_profile_fields = ('kind', 'id')
    if profile.get('kind') != 'oracle':
        required_profile_fields = ('kind', 'path', 'id', 'sha256')
    for key in required_profile_fields:
        if profile.get(key) in (None, ''):
            errors.append(f'method_bundle:profile:{key}')
    model = payload.get('model') if isinstance(payload.get('model'), Mapping) else {}
    for key in ('image', 'service', 'container', 'entrypoint', 'revision',
                'checkpoint_source'):
        if key not in model:
            errors.append(f'method_bundle:model:{key}')
    sensors = payload.get('sensors') if isinstance(payload.get('sensors'), Mapping) else {}
    if sensors.get('sensor_profile') not in SENSOR_PROFILES:
        errors.append('method_bundle:sensors:sensor_profile')
    if not isinstance(sensors.get('recording_contract'), str):
        errors.append('method_bundle:sensors:recording_contract')
    if not isinstance(sensors.get('allowed_topics'), list):
        errors.append('method_bundle:sensors:allowed_topics')
    adapter = payload.get('adapter') if isinstance(payload.get('adapter'), Mapping) else {}
    for key in ('name', 'action_mapping', 'common_executor'):
        if adapter.get(key) in (None, ''):
            errors.append(f'method_bundle:adapter:{key}')
    controller = payload.get('controller') if isinstance(payload.get('controller'), Mapping) else {}
    for key in ('local_planner', 'control_path'):
        if controller.get(key) in (None, ''):
            errors.append(f'method_bundle:controller:{key}')
    ros = payload.get('ros') if isinstance(payload.get('ros'), Mapping) else {}
    if not _is_int_value(ros.get('domain_id')):
        errors.append('method_bundle:ros:domain_id')
    if not isinstance(ros.get('namespace'), str) or not ros['namespace']:
        errors.append('method_bundle:ros:namespace')
    if not isinstance(ros.get('topics'), Mapping):
        errors.append('method_bundle:ros:topics')
    ports = ros.get('ports')
    if not isinstance(ports, list) or any(not _is_int_value(p) for p in ports):
        errors.append('method_bundle:ros:ports')
    output = payload.get('output') if isinstance(payload.get('output'), Mapping) else {}
    for key in ('prefix', 'run_root'):
        if output.get(key) in (None, ''):
            errors.append(f'method_bundle:output:{key}')
    lock = payload.get('lock') if isinstance(payload.get('lock'), Mapping) else {}
    for key in ('path', 'scope'):
        if lock.get(key) in (None, ''):
            errors.append(f'method_bundle:lock:{key}')
    if not isinstance(payload.get('cleanup'), list):
        errors.append('method_bundle:cleanup')
    private_fields = payload.get('private_fields')
    if not isinstance(private_fields, list):
        errors.append('method_bundle:private_fields')
    elif not set(METHOD_PRIVATE_KEYS) <= set(private_fields):
        errors.append('method_bundle:private_fields:incomplete')
    if not isinstance(payload.get('source_binding'), str) or not payload['source_binding']:
        errors.append('method_bundle:source_binding')
    if not re.fullmatch(r'[0-9a-f]{7,40}', str(payload.get('source_commit') or '')):
        errors.append('method_bundle:source_commit')
    return {
        'valid': not errors,
        'errors': errors,
        'bundle': dict(payload) if isinstance(payload, Mapping) else {},
    }


def _method_private_hash(bundle: Mapping[str, Any]) -> str:
    view = {
        key: bundle.get(key) for key in bundle.get('private_fields') or []
        if key in METHOD_PRIVATE_KEYS
    }
    return 'sha256:' + _canonical_sha256(view)


def _commit_file_sha256(commit: str, path_relative: str) -> str:
    """Hash a file at a pinned commit, so unrelated working-tree dirt is ignored."""
    if not commit or not path_relative:
        return ''
    try:
        result = subprocess.run(
            ['git', '-C', str(REPO_ROOT), 'show',
             f'{commit}:{path_relative}'],
            capture_output=True)
    except Exception:  # noqa: BLE001
        return ''
    if result.returncode != 0:
        return ''
    return hashlib.sha256(result.stdout).hexdigest()


def _method_profile_sha_error(bundle: Mapping[str, Any]) -> str:
    profile = bundle.get('profile') if isinstance(bundle.get('profile'), Mapping) else {}
    if profile.get('kind') == 'oracle':
        return ''
    path_relative = str(profile.get('path') or '')
    expected = str(profile.get('sha256') or '')
    commit = str(bundle.get('source_commit') or '')
    pinned_sha = _commit_file_sha256(commit, path_relative)
    if pinned_sha:
        return '' if pinned_sha == expected else 'METHOD_BUNDLE_PROFILE_SHA_MISMATCH'
    working = (REPO_ROOT / path_relative).resolve() if path_relative else None
    if working is None or not working.is_file():
        return 'METHOD_BUNDLE_PROFILE_MISSING'
    actual = hashlib.sha256(working.read_bytes()).hexdigest()
    return '' if actual == expected else 'METHOD_BUNDLE_PROFILE_SHA_MISMATCH'


def _current_profile_drift_error(bundle: Mapping[str, Any]) -> str:
    """The runner reads the current file; it must match the pinned profile SHA."""
    profile = bundle.get('profile') if isinstance(bundle.get('profile'), Mapping) else {}
    if profile.get('kind') == 'oracle':
        return ''
    path_relative = str(profile.get('path') or '')
    if not path_relative:
        return 'METHOD_BUNDLE_CURRENT_PROFILE_MISSING'
    current = (REPO_ROOT / path_relative).resolve()
    if not current.is_file():
        return 'METHOD_BUNDLE_CURRENT_PROFILE_MISSING'
    current_sha = hashlib.sha256(current.read_bytes()).hexdigest()
    expected = str(profile.get('sha256') or '')
    return '' if current_sha == expected else 'METHOD_BUNDLE_CURRENT_PROFILE_DRIFT'


def _current_profile_sha(bundle: Mapping[str, Any]) -> str:
    profile = bundle.get('profile') if isinstance(bundle.get('profile'), Mapping) else {}
    path_relative = str(profile.get('path') or '')
    if not path_relative:
        return ''
    current = (REPO_ROOT / path_relative).resolve()
    if not current.is_file():
        return ''
    return hashlib.sha256(current.read_bytes()).hexdigest()


def _common_case_source_errors(common: Mapping[str, Any],
                               bundle: Mapping[str, Any]) -> list[str]:
    """Verify the common case exists and matches the current benchmark profile."""
    errors: list[str] = []
    case = common.get('case') if isinstance(common.get('case'), Mapping) else {}
    world = str(case.get('world') or '')
    scenario = str(case.get('scenario') or '')
    robot = str(case.get('robot') or '')
    world_dir = REPO_ROOT / 'arena_simulation_setup' / 'worlds' / world
    if not world_dir.is_dir():
        errors.append(f'COMMON_CASE_SOURCE_MISSING:world:{world}')
    scenario_dir = world_dir / 'scenarios' / scenario
    if not scenario_dir.is_dir():
        errors.append(f'COMMON_CASE_SOURCE_MISSING:scenario:{scenario}')
    robot_dir = REPO_ROOT / 'arena_robots' / 'arena_robots' / 'robots' / robot
    if not robot_dir.is_dir():
        errors.append(f'COMMON_CASE_SOURCE_MISSING:robot:{robot}')
    profile = bundle.get('profile') if isinstance(bundle.get('profile'), Mapping) else {}
    if profile.get('kind') == 'benchmark_profile':
        profile_path = (REPO_ROOT / str(profile.get('path') or '')).resolve()
        if not profile_path.is_file():
            errors.append('COMMON_CASE_PROFILE_MISSING')
        else:
            try:
                profile_payload = yaml.safe_load(
                    profile_path.read_text(encoding='utf-8')) or {}
            except Exception:  # noqa: BLE001
                profile_payload = {}
            evaluation = (profile_payload.get('evaluation')
                          if isinstance(profile_payload, Mapping) else {})
            evaluation = evaluation if isinstance(evaluation, Mapping) else {}
            for key, expected in (('world', world), ('scenario', scenario),
                                  ('robot', robot)):
                actual = evaluation.get(key)
                if actual != expected:
                    errors.append(
                        f'COMMON_CASE_PROFILE_MISMATCH:{key}:{actual}!={expected}')
    return errors


def build_method_plan(method_id: str, common: Mapping[str, Any],
                      bundle: Mapping[str, Any]) -> dict[str, Any]:
    common_check = validate_common_contract(common)
    bundle_check = validate_method_bundle(bundle)
    errors = list(common_check['errors']) + list(bundle_check['errors'])
    blockers: list[str] = []
    if method_id not in METHOD_IDS:
        blockers.append(f'METHOD_UNKNOWN:{method_id}')
    if bundle.get('method_id') != method_id:
        blockers.append('METHOD_ID_MISMATCH')
    profile_sha_error = _method_profile_sha_error(bundle)
    if profile_sha_error:
        blockers.append(profile_sha_error)
    drift_error = _current_profile_drift_error(bundle)
    if drift_error:
        blockers.append(drift_error)
    common_case_blockers = _common_case_source_errors(common, bundle)
    blockers.extend(common_case_blockers)
    if bundle.get('status') == 'BLOCKED':
        blockers.extend(str(item) for item in bundle.get('blockers') or [])
    if bundle.get('launchable') is not True:
        blockers.append('METHOD_NOT_LAUNCHABLE')
    conflicts: list[str] = []
    warnings: list[str] = []
    ports = list((bundle.get('ros') or {}).get('ports') or [])
    if len(ports) != len(set(ports)):
        conflicts.append('METHOD_CONFLICT:duplicate_port')
    topics = list(((bundle.get('ros') or {}).get('topics') or {}).values())
    if len(topics) != len(set(topics)):
        conflicts.append('METHOD_CONFLICT:duplicate_topic')
    if conflicts:
        blockers.extend(conflicts)
    executable = (
        not errors and not blockers
        and bundle.get('status') == 'READY'
        and bundle.get('launchable') is True
    )
    bundle_source_digest = 'sha256:' + _canonical_sha256(bundle)
    return {
        'schema': METHOD_PLAN_SCHEMA,
        'method_id': method_id,
        'role': bundle.get('role'),
        'status': bundle.get('status'),
        'runtime_status': bundle.get('runtime_status'),
        'executable': executable,
        'plan_only': True,
        'commands': [],
        'future_launch_command': [],
        'common_contract_hash': common_check['contract_hash'],
        'common_contract': common_check['contract'],
        'method_private_hash': _method_private_hash(bundle),
        'bundle_source_digest': bundle_source_digest,
        'canonical_bundle_path': str(
            REPO_ROOT / 'configs' / 'agentic_eval' / 'methods'
            / f'{method_id}.v1.json'),
        'pinned_profile_sha256': (
            (bundle.get('profile') or {}).get('sha256')
            if isinstance(bundle.get('profile'), Mapping) else None),
        'current_profile_sha256': _current_profile_sha(bundle) or None,
        'current_profile_drift': bool(drift_error),
        'common_case_blockers': sorted(set(common_case_blockers)),
        'method_bundle': bundle_check['bundle'],
        'errors': errors,
        'blockers': sorted(set(blockers)),
        'conflicts': sorted(set(conflicts)),
        'warnings': warnings,
        'launchable': bool(bundle.get('launchable')),
        'container': (bundle.get('model') or {}).get('container'),
        'ports': ports,
        'topics': (bundle.get('ros') or {}).get('topics'),
        'namespace': (bundle.get('ros') or {}).get('namespace'),
        'output_prefix': (bundle.get('output') or {}).get('prefix'),
        'source_binding': bundle.get('source_binding'),
        'lock': bundle.get('lock'),
        'cleanup': bundle.get('cleanup') or [],
        'note': 'Plan only; no service, container, Isaac, or episode is started.',
    }


def build_method_plan_comparison(plans: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    common_hashes = {plan.get('common_contract_hash') for plan in plans}
    private_hashes = [plan.get('method_private_hash') for plan in plans]
    common_hash_identical = len(common_hashes) == 1
    private_hashes_distinct = len(set(private_hashes)) == len(private_hashes)
    conflict_blockers: list[str] = []
    conflict_warnings: list[str] = []
    ready = [plan for plan in plans if plan.get('status') == 'READY']
    for field, label in (
            ('ports', 'port'), ('output_prefix', 'output_prefix')):
        seen: dict[Any, str] = {}
        for plan in ready:
            values = plan.get(field) or []
            if field == 'ports':
                values = list(values)
            else:
                values = [values]
            for value in values:
                if value in seen and seen[value] != plan['method_id']:
                    conflict_blockers.append(
                        f'METHOD_CONFLICT:{label}:{value}:{seen[value]}:{plan["method_id"]}')
                else:
                    seen[value] = plan['method_id']
    topic_seen: dict[Any, str] = {}
    for plan in ready:
        for value in (plan.get('topics') or {}).values():
            if value in topic_seen and topic_seen[value] != plan['method_id']:
                conflict_blockers.append(
                    f'METHOD_CONFLICT:topic:{value}:{topic_seen[value]}:{plan["method_id"]}')
            else:
                topic_seen[value] = plan['method_id']
    for field, label in (
            ('namespace', 'namespace'), ('container', 'container'),
            ('source_binding', 'source_binding')):
        seen: dict[Any, str] = {}
        for plan in ready:
            value = plan.get(field)
            if value in seen and seen[value] != plan['method_id']:
                conflict_warnings.append(
                    f'METHOD_SHARED:{label}:{value}:{seen[value]}:{plan["method_id"]}')
            else:
                seen[value] = plan['method_id']
    lock_seen: dict[Any, str] = {}
    for plan in ready:
        value = (plan.get('lock') or {}).get('path')
        if value in lock_seen and lock_seen[value] != plan['method_id']:
            conflict_warnings.append(
                f'METHOD_SHARED:lock:{value}:{lock_seen[value]}:{plan["method_id"]}')
        else:
            lock_seen[value] = plan['method_id']
    ready_blocked = [
        plan['method_id'] for plan in plans
        if plan.get('status') == 'READY' and not plan.get('executable')
    ]
    ok = (common_hash_identical and private_hashes_distinct
          and not conflict_blockers and not ready_blocked)
    return {
        'schema': METHOD_PLAN_COMPARISON_SCHEMA,
        'common_contract_hash': next(iter(common_hashes)) if common_hash_identical else None,
        'common_hash_identical': common_hash_identical,
        'private_hashes_distinct': private_hashes_distinct,
        'ready_methods_blocked': ready_blocked,
        'conflict_blockers': sorted(set(conflict_blockers)),
        'conflict_warnings': sorted(set(conflict_warnings)),
        'methods': {
            plan['method_id']: {
                'status': plan.get('status'),
                'runtime_status': plan.get('runtime_status'),
                'executable': plan.get('executable'),
                'common_contract_hash': plan.get('common_contract_hash'),
                'method_private_hash': plan.get('method_private_hash'),
                'bundle_source_digest': plan.get('bundle_source_digest'),
                'current_profile_drift': plan.get('current_profile_drift'),
                'common_case_blockers': plan.get('common_case_blockers'),
                'blockers': plan.get('blockers'),
            }
            for plan in plans
        },
        'ok': ok,
        'deterministic': True,
    }


def _bag_index(path: Path, *, sensor_profile: str = 'rgb_lidar') -> dict[str, Any]:
    """Read-only bag index; sensors are classified by ROS message type."""
    if sensor_profile not in SENSOR_PROFILES:
        raise PipelineRejected(f'sensor_profile:{sensor_profile}')
    if not path.is_file():
        return {'present': False, 'path': str(path), 'topics': {},
                'topic_counts': {}, 'playback_status': 'BLOCKED',
                'sensor_profile': sensor_profile,
                'blockers': ['BAG_MISSING']}
    import sqlite3
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        topics = connection.execute(
            'SELECT id, name, type FROM topics').fetchall()
        counts = dict(connection.execute(
            'SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id').fetchall())
    finally:
        connection.close()
    topic_rows = {
        int(topic_id): {'name': str(name), 'type': str(type_name),
                        'messages': int(counts.get(topic_id, 0))}
        for topic_id, name, type_name in topics
    }
    rows = list(topic_rows.values())

    def _typed(message_type: str, name_hint: str | None = None) -> list[dict[str, Any]]:
        found = [row for row in rows if row['type'] == message_type]
        if name_hint is not None:
            found = [row for row in found if name_hint in row['name']]
        return found

    rgb = _typed('sensor_msgs/msg/Image', 'image')
    if not rgb:
        rgb = [row for row in _typed('sensor_msgs/msg/Image')
               if 'camera' in row['name']]
    camera_info = _typed('sensor_msgs/msg/CameraInfo')
    scans = _typed('sensor_msgs/msg/LaserScan')
    required = SENSOR_PROFILES[sensor_profile]
    blockers: list[str] = []
    if 'rgb' in required and (not rgb or sum(row['messages'] for row in rgb) == 0):
        blockers.append('RGB_NOT_IN_BAG')
    if 'camera_info' in required and (
            not camera_info or sum(row['messages'] for row in camera_info) == 0):
        blockers.append('CAMERA_INFO_NOT_IN_BAG')
    if 'laser_scan' in required and (
            not scans or sum(row['messages'] for row in scans) == 0):
        blockers.append('LASER_SCAN_ZERO_MESSAGES')
    return {
        'present': True,
        'path': str(path),
        'size': path.stat().st_size,
        'sha256': _file_digest(path)['sha256'],
        'sensor_profile': sensor_profile,
        'topics': topic_rows,
        'rgb_topics': [row['name'] for row in rgb],
        'camera_info_topics': [row['name'] for row in camera_info],
        'laser_scan_topics': [row['name'] for row in scans],
        'rgb_messages': sum(row['messages'] for row in rgb),
        'camera_info_messages': sum(row['messages'] for row in camera_info),
        'laser_scan_messages': sum(row['messages'] for row in scans),
        'playback_status': 'READY' if not blockers else 'BLOCKED',
        'blockers': blockers,
    }


def _is_inside(path_value: Any, root: Path) -> bool:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        return False
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _resolve_run_relative(raw: Any, base: Path, run: Path) -> tuple[Path | None, str]:
    if not raw:
        return None, 'UNRECORDED'
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve()
    except OSError:
        return None, 'UNRESOLVABLE'
    try:
        resolved.relative_to(run)
    except ValueError:
        return resolved, 'EXTERNAL'
    return resolved, 'RUN_RELATIVE'


def _index_path(path: Path | None, *, ownership: str) -> dict[str, Any]:
    if path is None:
        return {'present': False, 'ownership': ownership}
    if not path.is_file():
        return {'present': False, 'path': str(path), 'ownership': ownership}
    digest = _file_digest(path)
    return {
        'present': True,
        'path': str(path),
        'ownership': ownership,
        'size': digest['size'],
        'sha256': digest['sha256'],
        'mtime': digest['mtime'],
    }


def _video_index_entries(video_index: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not isinstance(video_index, Mapping):
        return entries
    for episode in video_index.get('episodes') or []:
        if isinstance(episode, Mapping):
            entries.append(dict(episode))
    legacy = video_index.get('videos')
    if isinstance(legacy, Mapping):
        entries.append({'episode': 0, 'directory': None, **legacy})
    return entries


def build_evidence_index(run_dir: str | Path, output: str | Path,
                         *, sensor_profile: str = 'rgb_lidar',
                         provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    out = Path(output).resolve()
    if sensor_profile not in SENSOR_PROFILES:
        raise PipelineRejected(f'sensor_profile:{sensor_profile}')
    if out.exists():
        raise PipelineRejected(f'output_exists:{out}')
    try:
        out.relative_to(run)
    except ValueError:
        pass
    else:
        raise PipelineRejected(f'output_inside_source_run:{out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in EVIDENCE_INDEX_FILES:
        files[name] = _index_path(run / name, ownership='RUN_RELATIVE')
    for pattern in EVIDENCE_INDEX_GLOBS:
        for path in sorted(run.glob(pattern)):
            files[path.name] = _index_path(path, ownership='RUN_RELATIVE')
    videos: dict[str, Any] = {}
    aliases_by_path: dict[str, list[str]] = {}
    pts_sidecars: dict[str, Any] = {}
    video_blockers: list[str] = []
    index_path = run / 'video_index.json'
    video_index: Any = {}
    if index_path.is_file():
        try:
            video_index = json.loads(index_path.read_text(encoding='utf-8'))
        except ValueError:
            video_blockers.append('VIDEO_INDEX_INVALID')
            video_index = {}
    else:
        video_blockers.append('VIDEO_INDEX_MISSING')
    video_keys = ('ego_video', 'top_down_video', 'map_top_down_video',
                  'debug_overlay_video', 'sim_top_down_video')
    for entry in _video_index_entries(video_index):
        episode = entry.get('episode', 0)
        directory_raw = entry.get('directory')
        if directory_raw:
            ep_dir, _ = _resolve_run_relative(directory_raw, run, run)
            ep_dir = ep_dir or run
        else:
            ep_dir = run / 'videos' / f'episode_{int(episode):04d}'
        pts_raw = entry.get('video_frame_pts') or 'video_frame_pts.json'
        pts_path, pts_ownership = _resolve_run_relative(pts_raw, ep_dir, run)
        pts_record = _index_path(pts_path, ownership=pts_ownership)
        if pts_record.get('present'):
            try:
                pts_payload = json.loads(pts_path.read_text(encoding='utf-8'))
            except ValueError:
                pts_record['schema'] = 'UNPARSEABLE'
            else:
                pts_record['schema'] = pts_payload.get('schema')
                pts_record['views'] = sorted((pts_payload.get('views') or {}).keys())
        else:
            pts_record['schema'] = 'UNRECORDED'
        pts_key = f'episode_{int(episode)}'
        pts_sidecars[pts_key] = pts_record
        for key in video_keys:
            raw = entry.get(key)
            if not raw:
                continue
            path, ownership = _resolve_run_relative(raw, ep_dir, run)
            name = f'{pts_key}/{key}'
            if path is not None and str(path) in aliases_by_path:
                aliases_by_path[str(path)].append(name)
                continue
            record = _index_path(path, ownership=ownership)
            record['aliases'] = []
            record['video_frame_pts'] = {
                'present': pts_record.get('present', False),
                'path': pts_record.get('path'),
                'sha256': pts_record.get('sha256'),
                'schema': pts_record.get('schema'),
            }
            if ownership == 'EXTERNAL':
                video_blockers.append(f'VIDEO_PATH_OUTSIDE_RUN:{name}')
            videos[name] = record
            if path is not None:
                aliases_by_path[str(path)] = [name]
    for name, record in videos.items():
        record['aliases'] = [
            alias for alias in aliases_by_path.get(str(record.get('path')), [])
            if alias != name
        ]
    if not videos:
        video_blockers.append('VIDEO_INDEX_EMPTY')
    decode_records: dict[str, Any] = {}
    timebase_records: dict[str, Any] = {}

    def _collect_mp4_paths(value: Any, found: list[str]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key == 'path' and isinstance(item, str) and item.endswith('.mp4'):
                    found.append(item)
                else:
                    _collect_mp4_paths(item, found)
        elif isinstance(value, list):
            for item in value:
                _collect_mp4_paths(item, found)

    for base, scope in ((run, 'RUN_RELATIVE'), (run.parent, 'PARENT')):
        for path in sorted(base.glob('*.json')):
            upper = path.name.upper()
            if 'DECODE' not in upper and 'TIMEBASE' not in upper and 'PTS_AUDIT' not in upper:
                continue
            record = _index_path(path, ownership='RUN_RELATIVE')
            record['scope'] = scope
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except ValueError:
                record['schema'] = 'UNPARSEABLE'
                record['references_run'] = None
            else:
                record['schema'] = payload.get('schema')
                found_paths: list[str] = []
                _collect_mp4_paths(payload, found_paths)
                if found_paths:
                    record['references_run'] = any(
                        _is_inside(path_str, run) for path_str in found_paths)
                else:
                    record['references_run'] = None
            if 'DECODE' in upper:
                decode_records[f'{scope}:{path.name}'] = record
            else:
                timebase_records[f'{scope}:{path.name}'] = record

    def _audit_status(records: Mapping[str, Any]) -> str:
        if not records:
            return 'UNKNOWN'
        if any(record.get('references_run') is True for record in records.values()):
            return 'AVAILABLE'
        if all(record.get('references_run') is None for record in records.values()):
            return 'AVAILABLE_UNVERIFIED'
        return 'AVAILABLE_NOT_RUN_SCOPED'

    bag = _bag_index(run / 'recording' / 'recording_0.db3',
                     sensor_profile=sensor_profile)
    result = {
        'schema': 'arena.paper.evidence_index.v1',
        'run_dir': str(run),
        'sensor_profile': sensor_profile,
        'files': files,
        'videos': videos,
        'video_frame_pts': pts_sidecars,
        'decode': {
            'status': _audit_status(decode_records),
            'records': decode_records,
        },
        'timebase': {
            'status': _audit_status(timebase_records),
            'records': timebase_records,
        },
        'bag': bag,
        'playback_status': bag['playback_status'],
        'blockers': list(bag['blockers']),
        'artifact_blockers': video_blockers,
        'all_blockers': list(bag['blockers']) + video_blockers,
    }
    result['verdict'] = build_evidence_verdict(result)
    if provenance is not None:
        require_provenance_authenticity(provenance, expected_run_dir=run)
        result['provenance'] = dict(provenance)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n',
                   encoding='utf-8')
    return result


EVIDENCE_VERDICT_SCHEMA = 'arena.paper.evidence_verdict.v1'
EVIDENCE_VERDICT_REQUIRED_FILES = (
    'benchmark_result.json', 'run_manifest.yaml', 'metrics.csv',
    'vln_task_metrics.json', 'social_metrics.json',
    'artifact_validation.json', 'video_index.json',
)
EVIDENCE_VERDICT_POLICY = {
    'version': 1,
    'precedence': [
        'contract_blocked', 'playback_blocked', 'scientific_fail',
        'artifact_complete',
    ],
    'unknown_never_pass': True,
}


def _evidence_verdict_view(index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: index.get(key)
        for key in ('schema', 'run_dir', 'sensor_profile', 'files', 'videos',
                    'video_frame_pts', 'decode', 'timebase', 'bag',
                    'playback_status', 'blockers', 'artifact_blockers')
    }


def _read_index_json(index: Mapping[str, Any], name: str) -> tuple[Any, str]:
    record = (index.get('files') or {}).get(name) or {}
    if not record.get('present'):
        return None, 'MISSING'
    try:
        return json.loads(Path(record['path']).read_text(encoding='utf-8')), 'OK'
    except (OSError, ValueError):
        return None, 'UNPARSEABLE'


def build_evidence_verdict(index: Mapping[str, Any]) -> dict[str, Any]:
    """Classify an evidence index deterministically.

    Precedence: contract_blocked > playback_blocked > scientific_fail >
    artifact_complete.  Missing or UNKNOWN evidence never becomes PASS.
    """
    contract_reasons: list[str] = []
    unknowns: list[str] = []
    if not isinstance(index, Mapping) or index.get('schema') != 'arena.paper.evidence_index.v1':
        contract_reasons.append('EVIDENCE_INDEX_SCHEMA')
    bag = index.get('bag') or {}
    if not bag.get('present'):
        contract_reasons.append('BAG_MISSING')
    artifact_blockers = list(index.get('artifact_blockers') or [])
    for blocker in artifact_blockers:
        if blocker.startswith('VIDEO_INDEX'):
            contract_reasons.append(blocker)
    files = index.get('files') or {}
    missing_artifacts = [
        name for name in EVIDENCE_VERDICT_REQUIRED_FILES
        if not (files.get(name) or {}).get('present')
    ]
    videos = index.get('videos') or {}
    missing_videos = [
        name for name, record in videos.items() if not record.get('present')
    ]
    playback_blocked = index.get('playback_status') != 'READY'
    task, task_status = _read_index_json(index, 'vln_task_metrics.json')
    social, social_status = _read_index_json(index, 'social_metrics.json')
    artifact, artifact_status = _read_index_json(index, 'artifact_validation.json')
    task_failed: bool | None = None
    social_failed: bool | None = None
    if task_status == 'OK' and isinstance(task, Mapping):
        goal = task.get('goal') or {}
        timing = task.get('episode_timing') or {}
        if isinstance(goal.get('goal_reached'), bool):
            task_failed = not goal['goal_reached']
        if isinstance(timing.get('timed_out'), bool):
            task_failed = bool(task_failed) or timing['timed_out']
        if task_failed is None and isinstance(timing.get('end_reason'), str):
            task_failed = 'timeout' in timing['end_reason']
    if task_failed is None:
        unknowns.append(f'TASK_EVIDENCE_{task_status}')
    if social_status == 'OK' and isinstance(social, Mapping):
        if isinstance(social.get('social_success'), bool):
            social_failed = not social['social_success']
    if social_failed is None:
        unknowns.append(f'SOCIAL_EVIDENCE_{social_status}')
    if artifact_status != 'OK':
        unknowns.append(f'ARTIFACT_VALIDATION_{artifact_status}')
    metrics_known = task_failed is not None and social_failed is not None
    if not metrics_known:
        contract_reasons.append('METRICS_EVIDENCE_UNKNOWN')
    scientific_fail = bool(task_failed or social_failed) if metrics_known else None
    artifact_complete = (
        not contract_reasons
        and not missing_artifacts
        and not missing_videos
        and bool(videos)
        and not playback_blocked
    )
    flags = {
        'contract_blocked': bool(contract_reasons),
        'playback_blocked': bool(playback_blocked),
        'scientific_fail': scientific_fail,
        'artifact_complete': bool(artifact_complete),
    }
    if flags['contract_blocked']:
        primary = 'contract_blocked'
    elif flags['playback_blocked']:
        primary = 'playback_blocked'
    elif scientific_fail:
        primary = 'scientific_fail'
    elif artifact_complete:
        primary = 'artifact_complete'
    else:
        primary = 'contract_blocked'
        contract_reasons.append('VERDICT_UNKNOWN')
    for name in missing_artifacts:
        unknowns.append(f'ARTIFACT_MISSING:{name}')
    for name in missing_videos:
        unknowns.append(f'VIDEO_MISSING:{name}')
    for blocker in (index.get('blockers') or []):
        unknowns.append(f'PLAYBACK_BLOCKER:{blocker}')
    decode_status = (index.get('decode') or {}).get('status', 'UNKNOWN')
    timebase_status = (index.get('timebase') or {}).get('status', 'UNKNOWN')
    if decode_status != 'AVAILABLE':
        unknowns.append(f'DECODE_{decode_status}')
    if timebase_status != 'AVAILABLE':
        unknowns.append(f'TIMEBASE_{timebase_status}')
    secondary = [
        name for name in ('playback_blocked', 'scientific_fail', 'artifact_complete')
        if flags.get(name) is True and name != primary
    ]
    return {
        'schema': EVIDENCE_VERDICT_SCHEMA,
        'verdict': primary,
        'secondary_verdicts': secondary,
        'flags': flags,
        'contract_reasons': sorted(set(contract_reasons)),
        'unknowns': sorted(set(unknowns)),
        'metrics_evidence': {
            'task_status': task_status,
            'social_status': social_status,
            'artifact_status': artifact_status,
            'task_failed': task_failed,
            'social_failed': social_failed,
            'metrics_known': metrics_known,
        },
        'input_sha256': _canonical_sha256(_evidence_verdict_view(index)),
        'policy': dict(EVIDENCE_VERDICT_POLICY),
        'deterministic': True,
    }


def write_evidence_verdict(index_path: str | Path, output: str | Path,
                           *, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    index = load_json(index_path)
    out = Path(output).resolve()
    if out.exists():
        raise PipelineRejected(f'output_exists:{out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    verdict = build_evidence_verdict(index)
    if provenance is None and isinstance(index, Mapping):
        provenance = index.get('provenance')
    if provenance is not None:
        require_provenance_authenticity(
            provenance,
            expected_run_dir=(index.get('run_dir') if isinstance(index, Mapping)
                              else None))
        verdict['provenance'] = dict(provenance)
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + '\n',
                   encoding='utf-8')
    return verdict


def evidence_verdict_exit_code(verdict: Mapping[str, Any]) -> int:
    primary = verdict.get('verdict')
    if primary == 'artifact_complete':
        return 0
    if primary == 'scientific_fail':
        return 1
    return 2


AUDIT_MANIFEST_SCHEMA = 'arena.agentic.audit_manifest.v1'
AUDIT_CONSOLIDATED_SCHEMA = 'arena.agentic.audit_consolidated.v1'
AUDIT_PRECEDENCE = (
    'contract_blocked', 'playback_blocked', 'scientific_fail',
    'artifact_complete',
)
AUDIT_PROVENANCE_FIELDS = (
    'request_id', 'run_dir', 'run_identity_digest', 'source_tree_digest',
    'source_file_set_digest',
)


def _read_audit_receipt(path_value: Any) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value.strip():
        return {'present': False, 'status': 'MISSING', 'path': None}
    path = Path(path_value)
    if not path.is_file():
        return {'present': False, 'status': 'MISSING', 'path': str(path)}
    digest = _file_digest(path)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except ValueError:
        return {
            'present': True, 'status': 'UNPARSEABLE', 'path': str(path),
            'sha256': digest['sha256'], 'size': digest['size'],
        }
    return {
        'present': True, 'status': 'OK', 'path': str(path),
        'sha256': digest['sha256'], 'size': digest['size'],
        'payload': payload,
    }


def build_audit_consolidated(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Combine audit receipts into one deterministic typed verdict."""
    blockers: list[str] = []
    reasons: list[str] = []
    if not isinstance(manifest, Mapping) or manifest.get('schema') != AUDIT_MANIFEST_SCHEMA:
        blockers.append('AUDIT_MANIFEST_SCHEMA')
    expected_provenance = manifest.get('provenance') if isinstance(
        manifest.get('provenance'), Mapping) else None
    run_dir = Path(str(manifest.get('run_dir') or '')).resolve() if manifest.get('run_dir') else None
    output_root = Path(str(manifest.get('output_root') or '')).resolve() if manifest.get('output_root') else None
    if run_dir is None:
        blockers.append('AUDIT_MANIFEST_RUN_DIR_MISSING')
    if expected_provenance is None:
        blockers.append('AUDIT_MANIFEST_PROVENANCE_MISSING')
    elif run_dir is not None:
        authenticity = verify_provenance_authenticity(
            expected_provenance,
            expected_request_id=manifest.get('request_id'),
            expected_run_dir=run_dir)
        blockers.extend(f'AUDIT_MANIFEST:{error}'
                        for error in authenticity['errors'])
    request_identity = manifest.get('request_identity')
    if not isinstance(request_identity, Mapping):
        blockers.append('AUDIT_MANIFEST_REQUEST_IDENTITY_MISSING')
    elif run_dir is not None:
        try:
            recomputed_identity = extract_run_identity(run_dir)
        except Exception as exc:  # noqa: BLE001
            blockers.append(f'PROVENANCE_AUTHENTICITY_MISMATCH:run_dir:{exc}')
        else:
            for field in ('track', 'model', 'model_revision', 'case',
                          'seed', 'episodes', 'action_control',
                          'sensor_profile'):
                if request_identity.get(field) != recomputed_identity.get(field):
                    blockers.append(
                        f'PROVENANCE_AUTHENTICITY_MISMATCH:request_identity:{field}')
    receipts: dict[str, Any] = {}
    for entry in (manifest.get('receipts') or []):
        if not isinstance(entry, Mapping):
            blockers.append('AUDIT_RECEIPT_ENTRY_INVALID')
            continue
        name = str(entry.get('name') or 'unknown')
        record = _read_audit_receipt(entry.get('path'))
        if record['status'] != 'OK' and entry.get('required', True):
            blockers.append(f'RECEIPT_MISSING:{name}')
        if record['status'] == 'OK':
            path = Path(str(record['path'])).resolve()
            if run_dir is not None:
                try:
                    path.relative_to(run_dir)
                except ValueError:
                    pass
                else:
                    blockers.append(f'RECEIPT_PATH_INSIDE_SOURCE:{name}')
            if output_root is not None:
                try:
                    path.relative_to(output_root)
                except ValueError:
                    blockers.append(f'RECEIPT_PATH_OUTSIDE_OUTPUT_ROOT:{name}')
            receipt_provenance = (record.get('payload') or {}).get('provenance')
            if not isinstance(receipt_provenance, Mapping):
                blockers.append(f'RECEIPT_PROVENANCE_MISSING:{name}')
            elif run_dir is not None:
                authenticity = verify_provenance_authenticity(
                    receipt_provenance,
                    expected_request_id=manifest.get('request_id'),
                    expected_run_dir=run_dir)
                blockers.extend(
                    f'RECEIPT_{name}:{error}'
                    for error in authenticity['errors'])
        receipts[name] = record
    for required_name in ('benchmark_result', 'evidence_verdict'):
        if receipts.get(required_name, {}).get('status') != 'OK':
            blockers.append(f'RECEIPT_MISSING:{required_name}')
    verdict_payload = (receipts.get('evidence_verdict') or {}).get('payload') or {}
    primary = verdict_payload.get('verdict') or 'contract_blocked'
    if primary not in AUDIT_PRECEDENCE:
        primary = 'contract_blocked'
        reasons.append('EVIDENCE_VERDICT_UNKNOWN')
    benchmark = (receipts.get('benchmark_result') or {}).get('payload') or {}
    benchmark_verdict = benchmark.get('verdict') or {}
    if receipts.get('benchmark_result', {}).get('status') == 'OK':
        if not benchmark_verdict.get('benchmark_ready'):
            if primary == 'artifact_complete':
                primary = 'contract_blocked'
            reasons.append('BENCHMARK_NOT_READY')
        elif not benchmark_verdict.get('benchmark_success'):
            if primary == 'artifact_complete':
                primary = 'scientific_fail'
            reasons.append('BENCHMARK_SCIENTIFIC_FAIL')
    metrics = (receipts.get('metrics_receipt') or {}).get('payload') or {}
    if receipts.get('metrics_receipt', {}).get('status') == 'OK':
        metrics_status = metrics.get('status')
        if metrics_status == 'fault':
            primary = 'contract_blocked'
            reasons.append('METRICS_STAGE_FAULT')
        elif metrics_status == 'engineering_closed_scientific_fail':
            if primary == 'artifact_complete':
                primary = 'scientific_fail'
            reasons.append('METRICS_SCIENTIFIC_FAIL')
    source_audit = (receipts.get('source_run_audit') or {}).get('payload') or {}
    if receipts.get('source_run_audit', {}).get('status') == 'OK':
        if not source_audit.get('source_eligible'):
            primary = 'contract_blocked'
            reasons.append('SOURCE_RUN_NOT_ELIGIBLE')
    if blockers:
        primary = 'contract_blocked'
    exit_code = (0 if primary == 'artifact_complete'
                 else 1 if primary == 'scientific_fail' else 2)
    return {
        'schema': AUDIT_CONSOLIDATED_SCHEMA,
        'request_id': manifest.get('request_id'),
        'track': manifest.get('track'),
        'verdict': primary,
        'exit_code': exit_code,
        'blockers': sorted(set(blockers)),
        'reasons': sorted(set(reasons)),
        'receipts': {
            name: {key: value for key, value in record.items()
                   if key != 'payload'}
            for name, record in receipts.items()
        },
        'policy': {
            'version': 1,
            'precedence': list(AUDIT_PRECEDENCE),
            'unknown_never_pass': True,
        },
        'deterministic': True,
    }


def write_audit_consolidated(manifest_path: str | Path,
                             output: str | Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    out = Path(output).resolve()
    if out.exists():
        raise PipelineRejected(f'output_exists:{out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    result = build_audit_consolidated(manifest)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n',
                   encoding='utf-8')
    return result


RECORDING_CONTRACT_SCHEMA = 'arena.agentic.recording_contract.v1'
RECORDING_SENSOR_TYPES = {
    'rgb_lidar': {
        'rgb': ('sensor_msgs/msg/Image',),
        'camera_info': ('sensor_msgs/msg/CameraInfo',),
        'laser_scan': ('sensor_msgs/msg/LaserScan',),
    },
    'rgb_only': {
        'rgb': ('sensor_msgs/msg/Image',),
        'camera_info': ('sensor_msgs/msg/CameraInfo',),
    },
}
RECORDING_TRACK_SENSORS = {'A': 'rgb_lidar', 'B': 'rgb_only'}
RECORDING_FORBIDDEN_TOPIC_HINTS = (
    'scan', 'lidar', 'pointcloud', 'depth', 'people', 'gt_map',
    'global_costmap', 'tracked_obstacles', 'eval_goal',
)


def validate_recording_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return {'valid': False, 'errors': ['payload:not_object']}
    if payload.get('schema') != RECORDING_CONTRACT_SCHEMA:
        errors.append('schema')
    track = payload.get('track')
    if track not in RECORDING_TRACK_SENSORS:
        errors.append('track')
    sensor_profile = payload.get('sensor_profile')
    if sensor_profile not in RECORDING_SENSOR_TYPES:
        errors.append('sensor_profile')
    elif track in RECORDING_TRACK_SENSORS and (
            sensor_profile != RECORDING_TRACK_SENSORS[track]):
        errors.append(f'track_sensor_mismatch:{track}:{sensor_profile}')
    bag_path = payload.get('bag_path')
    if not isinstance(bag_path, str) or not bag_path.strip():
        errors.append('bag_path')
    elif os.path.isabs(bag_path) or '..' in Path(bag_path).parts:
        errors.append('bag_path:unsafe')
    command = payload.get('recording_command')
    if (not isinstance(command, list) or not command
            or any(not isinstance(item, str) or not item for item in command)):
        errors.append('recording_command')
    if payload.get('open_mode') != 'ro':
        errors.append('open_mode:must_be_ro')
    playback_policy = payload.get('playback_policy')
    if not isinstance(playback_policy, str) or not playback_policy:
        errors.append('playback_policy')
    required_types = payload.get('required_types')
    required_topics = payload.get('required_topics')
    min_messages = payload.get('min_messages')
    allowed_types = RECORDING_SENSOR_TYPES.get(str(sensor_profile), {})
    if not isinstance(required_types, Mapping) or not isinstance(required_topics, Mapping):
        errors.append('required_sensors')
    else:
        for key, allowed in allowed_types.items():
            types = required_types.get(key)
            if not isinstance(types, list) or not types:
                errors.append(f'required_types:{key}')
            elif any(item not in allowed for item in types):
                errors.append(f'required_types:unsupported:{key}')
            topics = required_topics.get(key)
            if (not isinstance(topics, list) or not topics
                    or any(not isinstance(item, str) or not item for item in topics)):
                errors.append(f'required_topics:{key}')
        if sensor_profile == 'rgb_lidar' and 'laser_scan' not in required_types:
            errors.append('required_types:laser_scan')
        if sensor_profile == 'rgb_only':
            for topics in required_topics.values():
                for topic in topics if isinstance(topics, list) else []:
                    if any(hint in str(topic).lower()
                           for hint in RECORDING_FORBIDDEN_TOPIC_HINTS):
                        errors.append(f'strict_b_forbidden_topic:{topic}')
    if not isinstance(min_messages, Mapping):
        errors.append('min_messages')
    else:
        for key in allowed_types:
            value = min_messages.get(key)
            if not _is_int_value(value) or value < 1:
                errors.append(f'min_messages:{key}')
    return {
        'schema': 'arena.agentic.recording_contract_check.v1',
        'valid': not errors,
        'errors': errors,
        'track': track,
        'sensor_profile': sensor_profile,
        'identity': {
            'bag_path': bag_path,
            'recording_command': command,
            'open_mode': payload.get('open_mode'),
            'required_types': required_types,
            'required_topics': required_topics,
            'min_messages': min_messages,
            'playback_policy': playback_policy,
        },
    }


def recording_preflight(contract: Mapping[str, Any], run_dir: str | Path,
                        output: str | Path) -> dict[str, Any]:
    validation = validate_recording_contract(contract)
    if not validation['valid']:
        raise PipelineRejected(
            'recording_contract_invalid:' + ','.join(validation['errors']))
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise PipelineRejected(f'run_dir:missing:{run}')
    out = Path(output).resolve()
    if out.exists():
        raise PipelineRejected(f'output_exists:{out}')
    try:
        out.relative_to(run)
    except ValueError:
        pass
    else:
        raise PipelineRejected(f'output_inside_source_run:{out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    bag_path = run / str(contract['bag_path'])
    bag = _bag_index(bag_path, sensor_profile=str(contract['sensor_profile']))
    rows = list((bag.get('topics') or {}).values())
    by_name = {str(row['name']).lstrip('/'): row for row in rows}
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for key, topics in (contract.get('required_topics') or {}).items():
        allowed_types = (contract.get('required_types') or {}).get(key) or []
        minimum = (contract.get('min_messages') or {}).get(key, 1)
        for topic in topics:
            normalized = str(topic).lstrip('/')
            row = by_name.get(normalized)
            record = {
                'sensor': key,
                'topic': str(topic),
                'required_types': list(allowed_types),
                'min_messages': minimum,
                'present': row is not None,
                'type': (row or {}).get('type'),
                'messages': int((row or {}).get('messages', 0)),
                'status': 'PASS',
            }
            if row is None:
                record['status'] = 'MISSING_TOPIC'
                blockers.append(f'RECORDING_TOPIC_MISSING:{topic}')
            elif row['type'] not in allowed_types:
                record['status'] = 'TYPE_MISMATCH'
                blockers.append(f'RECORDING_TYPE_MISMATCH:{topic}')
            elif int(row['messages']) < int(minimum):
                record['status'] = 'MESSAGES_BELOW_MIN'
                blockers.append(f'RECORDING_MESSAGES_BELOW_MIN:{topic}')
            checks.append(record)
    playback_status = 'READY' if not blockers else 'BLOCKED'
    result = {
        'schema': 'arena.agentic.recording_preflight.v1',
        'contract_schema': contract.get('schema'),
        'track': contract.get('track'),
        'sensor_profile': contract.get('sensor_profile'),
        'run_dir': str(run),
        'bag_path': str(bag_path),
        'bag_present': bool(bag.get('present')),
        'bag_sha256': bag.get('sha256'),
        'open_mode': 'ro',
        'checks': checks,
        'playback_status': playback_status,
        'blockers': sorted(set(blockers)),
        'contract_status': 'CONTRACT_ONLY',
        'current_run_status': playback_status,
        'note': 'Preflight only; it never creates missing recording evidence.',
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n',
                   encoding='utf-8')
    return result


# ---------------------------------------------------------------------------
# Metrics history protection
# ---------------------------------------------------------------------------

METRICS_OUTPUT_NAMES = (
    'metrics.csv', 'vln_task_metrics.json', 'social_metrics.json',
    'artifact_validation.json', 'PAPER_METRICS_RECEIPT.json',
)
METRICS_GUARD_NAMES = (
    'odom.csv', 'cmd_vel.csv', 'human_states.csv', 'pedsim_agents_data.csv',
    'start_goal.csv', 'scan.csv', 'run_manifest.yaml', 'params.yaml',
    'video_index.json',
) + METRICS_OUTPUT_NAMES


def _file_digest(path: Path) -> dict[str, Any]:
    return {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'mtime': path.stat().st_mtime,
        'size': path.stat().st_size,
    }


def source_hash_manifest(source_run: str | Path) -> dict[str, Any]:
    """Guard the files metrics reads plus every path under the source run."""
    root = Path(source_run).resolve()
    guard = {
        name: _file_digest(root / name)
        for name in METRICS_GUARD_NAMES
        if (root / name).is_file()
    }
    tree: dict[str, Any] = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = str(path.relative_to(root))
        record = {
            'size': path.stat().st_size,
            'mtime': path.stat().st_mtime,
        }
        if path.stat().st_size <= 8 * 1024 * 1024:
            record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            record['sha256'] = None
            record['sha_skipped'] = 'size>8MiB'
        tree[relative] = record
    return {'guard_files': guard, 'tree': tree}


def _tree_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    before_tree = before.get('tree') or {}
    after_tree = after.get('tree') or {}
    changes = []
    for name in sorted(set(before_tree) | set(after_tree)):
        if before_tree.get(name) != after_tree.get(name):
            changes.append(name)
    return changes


def prepare_derived_run(source_run: str | Path, derived_root: str | Path) -> Path:
    """Copy read-only inputs to a fresh derived run; never write the source."""
    source = Path(source_run).resolve()
    root = Path(derived_root).resolve()
    if root == source:
        raise PipelineRejected('derived_root_equals_source')
    try:
        root.relative_to(source)
    except ValueError:
        pass
    else:
        raise PipelineRejected(f'derived_root_inside_source:{root}')
    derived = root / source.name
    if derived.exists():
        raise PipelineRejected(f'derived_run_exists:{derived}')
    derived.mkdir(parents=True)
    for item in source.iterdir():
        if item.name in METRICS_OUTPUT_NAMES:
            continue
        target = derived / item.name
        if item.is_symlink():
            raise PipelineRejected(f'source_symlink_not_copyable:{item}')
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return derived


def write_metrics_overwrite_incident(source_run: str | Path, command: str) -> Path:
    """Record the prior in-place overwrite; never claim exact restore."""
    source = Path(source_run)
    overwritten = {
        name: _file_digest(source / name)
        for name in METRICS_OUTPUT_NAMES
        if (source / name).is_file()
    }
    incident = {
        'schema': 'arena.paper.metrics_overwrite_incident.v1',
        'source_run': str(source.resolve()),
        'overwritten_files': overwritten,
        'command': command,
        'exact_backup_available': False,
        'restore_claimed': False,
        'note': (
            'These source files were rewritten by an earlier in-place metrics '
            'run. No exact pre-run backup exists; do not claim exact restore.'
        ),
    }
    path = source / 'METRICS_OVERWRITE_INCIDENT.json'
    path.write_text(json.dumps(incident, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# Metrics stage wrapper
# ---------------------------------------------------------------------------

def default_stage_commands(run_dir: str) -> list[tuple[str, list[str]]]:
    return [
        ('stage_a_metrics',
         ['ros2', 'run', 'arena_evaluation', 'metrics', '--dir', run_dir]),
        ('stage_b_vln_task_metrics',
         ['ros2', 'run', 'arena_evaluation', 'vln_task_metrics', '--dir', run_dir]),
        ('stage_c_social_metrics',
         ['ros2', 'run', 'arena_evaluation', 'social_metrics', '--dir', run_dir]),
        ('validation',
         ['ros2', 'run', 'arena_bringup', 'social_nav_validation', '--dir', run_dir]),
    ]


def _subprocess_stage_runner(command: Sequence[str]) -> int:
    result = subprocess.run(list(command), capture_output=True, text=True)
    return int(result.returncode)


def run_metrics_pipeline(
    *,
    run_dir: str,
    stage_runner: Callable[[Sequence[str]], int] | None = None,
    commands: Sequence[tuple[str, Sequence[str]]] | None = None,
    validation_scientific_rc: int = 1,
) -> dict[str, Any]:
    """Run ordered stages with fail-close semantics.

    Contract:
    - any non-validation stage rc != 0 -> fault, ``failed_stage`` set, stop
    - validation not reached -> fault, ``validation_executed=False``
    - validation rc == ``validation_scientific_rc`` (default 1) -> engineering
      closure with ``scientific_fail=True``
    - validation rc == 2 (or any other non-zero) -> fault
    - all stages rc 0 -> pass
    """
    runner = stage_runner or _subprocess_stage_runner
    plan = list(commands or default_stage_commands(run_dir))
    results: list[dict[str, Any]] = []
    for name, command in plan:
        rc = int(runner(command))
        results.append({'stage': name, 'rc': rc, 'command': list(command)})
        if rc == 0:
            continue
        if name == 'validation' and rc == validation_scientific_rc:
            return {
                'schema': METRICS_SCHEMA,
                'status': 'engineering_closed_scientific_fail',
                'failed_stage': None,
                'validation_executed': True,
                'validation_rc': rc,
                'scientific_fail': True,
                'fault': False,
                'stages': results,
            }
        return {
            'schema': METRICS_SCHEMA,
            'status': 'fault',
            'failed_stage': name,
            'validation_executed': name == 'validation',
            'validation_rc': rc if name == 'validation' else None,
            'scientific_fail': False,
            'fault': True,
            'stages': results,
        }
    validation = next((row for row in results if row['stage'] == 'validation'), None)
    if validation is None:
        return {
            'schema': METRICS_SCHEMA,
            'status': 'fault',
            'failed_stage': 'validation',
            'validation_executed': False,
            'validation_rc': None,
            'scientific_fail': False,
            'fault': True,
            'reason': 'validation_not_executed',
            'stages': results,
        }
    return {
        'schema': METRICS_SCHEMA,
        'status': 'pass',
        'failed_stage': None,
        'validation_executed': True,
        'validation_rc': 0,
        'scientific_fail': False,
        'fault': False,
        'stages': results,
    }


def metrics_exit_code(result: Mapping[str, Any]) -> int:
    if result.get('status') == 'pass':
        return 0
    if result.get('status') == 'engineering_closed_scientific_fail':
        return 1
    return 2


# ---------------------------------------------------------------------------
# Unique output prefix and Docker endpoint
# ---------------------------------------------------------------------------

def unique_output_prefix(prefix: str, *, now: float | None = None) -> str:
    path = Path(prefix)
    if not path.exists():
        return str(path)
    stamp = int(now if now is not None else time.time())
    candidate = f'{path}_{stamp}'
    index = 1
    while Path(candidate).exists():
        candidate = f'{path}_{stamp}_{index}'
        index += 1
    return candidate


def _docker_ip(container: str) -> str:
    result = subprocess.run(
        ['docker', 'inspect', container, '--format',
         '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ''
    return result.stdout.strip()


def resolve_container_endpoint(
    container: str,
    *,
    port: int = 8000,
    path: str = '/v1/chat/completions',
    inspect: Callable[[str], str] | None = None,
) -> str:
    inspector = inspect or _docker_ip
    ip = inspector(container)
    if not ip:
        raise PipelineRejected(f'container_endpoint_unresolved:{container}')
    return f'http://{ip}:{port}{path}'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _cmd_settings(path: str) -> int:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [f'read:{exc}']})
        return 2
    result = validate_settings(payload)
    _emit(result)
    return 0 if result['valid'] else 2


def _cmd_bfast(path: str, enabled: bool, executor_path: str = '',
               flow_risk_path: str = '') -> int:
    try:
        proposal = load_json(path)
        executor_input = load_json(executor_path) if executor_path else None
        flow_risk = load_json(flow_risk_path) if flow_risk_path else None
        decision = apply_bfast(proposal, enabled=enabled,
                               executor_input=executor_input, flow_risk=flow_risk)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [str(exc)]})
        return 2
    _emit(decision)
    return 2 if decision.get('unsupported') else 0


def _cmd_bfast_profile(path: str, risk_threshold=None,
                       max_frame_gap_sec=None) -> int:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [f'read:{exc}']})
        return 2
    result = build_bfast_plan(payload, risk_threshold=risk_threshold,
                              max_frame_gap_sec=max_frame_gap_sec,
                              profile_path=path)
    _emit(result)
    return 0 if result['valid'] else 2


def _cmd_latency(args: argparse.Namespace) -> int:
    try:
        result = inject_latency(
            source_stamp_ns=args.source_stamp_ns,
            source_frame_id=args.source_frame_id,
            source_identity={'image_sha256': args.source_identity_sha256},
            latency_ms=args.latency_ms,
            now_sim_ns=args.now_sim_ns,
            mode=args.mode,
            previous_response_sim_ns=args.previous_response_sim_ns,
        )
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [str(exc)]})
        return 2
    _emit(result)
    return 0


def _cmd_scenario(path: str) -> int:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [f'read:{exc}']})
        return 2
    result = validate_scenario_profile(payload)
    if result['valid'] and result['runtime_status'] == 'SUPPORTED':
        result['command'] = scenario_command(payload)
    _emit(result)
    return 0 if result['valid'] and result['runtime_status'] == 'SUPPORTED' else 2


def _cmd_teacher(path: str, audit: bool = False) -> int:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [f'read:{exc}']})
        return 2
    result = audit_teacher_manifest(payload) if audit else validate_teacher_manifest(payload)
    _emit(result)
    return 0 if result['valid'] else 2


def _cmd_dataset(path: str) -> int:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [f'read:{exc}'], 'case_count': 0})
        return 2
    result = validate_dataset_manifest(payload)
    _emit(result)
    return 0 if result['valid'] else 2


def _cmd_evidence_index(run_dir: str, output: str,
                        sensor_profile: str = 'rgb_lidar',
                        provenance_file: str = '') -> int:
    try:
        provenance = load_json(provenance_file) if provenance_file else None
        result = build_evidence_index(
            run_dir, output, sensor_profile=sensor_profile,
            provenance=provenance)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [str(exc)]})
        return 2
    _emit(result)
    return 0


def _cmd_evidence_verdict(index_path: str, output: str,
                          provenance_file: str = '') -> int:
    try:
        provenance = load_json(provenance_file) if provenance_file else None
        verdict = write_evidence_verdict(
            index_path, output, provenance=provenance)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [str(exc)]})
        return 2
    _emit(verdict)
    return evidence_verdict_exit_code(verdict)


def _cmd_agentic_request(path: str) -> int:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [f'read:{exc}']})
        return 2
    result = validate_agentic_request(payload)
    _emit(result)
    return 0 if result['valid'] else 2


def _cmd_agentic_lifecycle(path: str, action: str,
                           from_state: str | None = None) -> int:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [f'read:{exc}']})
        return 2
    result = build_operator_lifecycle(payload, action, from_state=from_state)
    _emit(result)
    return 0 if result['executable'] else 2


def _cmd_method_plan(method: str, common_path: str, bundle_path: str = '',
                     output: str = '') -> int:
    try:
        common = load_json(common_path)
        if method not in METHOD_IDS:
            plan = build_method_plan(method, common, {})
        else:
            canonical_bundle_file = (
                REPO_ROOT / 'configs' / 'agentic_eval' / 'methods'
                / f'{method}.v1.json')
            if bundle_path:
                override = Path(bundle_path).resolve()
                canonical = canonical_bundle_file.resolve()
                if (not canonical.is_file()
                        or hashlib.sha256(override.read_bytes()).hexdigest()
                        != hashlib.sha256(canonical.read_bytes()).hexdigest()):
                    _emit({
                        'schema': METHOD_PLAN_SCHEMA,
                        'method_id': method,
                        'status': 'BLOCKED',
                        'executable': False,
                        'plan_only': True,
                        'commands': [],
                        'future_launch_command': [],
                        'blockers': ['METHOD_BUNDLE_SOURCE_MISMATCH'],
                        'errors': [],
                    })
                    return 2
            bundle = load_json(canonical_bundle_file)
            plan = build_method_plan(method, common, bundle)
        if output:
            out = Path(output).resolve()
            if out.exists():
                raise PipelineRejected(f'output_exists:{out}')
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n',
                           encoding='utf-8')
    except Exception as exc:  # noqa: BLE001
        _emit({'schema': METHOD_PLAN_SCHEMA, 'valid': False,
               'errors': [str(exc)], 'commands': [],
               'future_launch_command': []})
        return 2
    _emit(plan)
    return 0 if plan['executable'] else 2


def _cmd_method_compare(common_path: str, output: str,
                        methods: str = '') -> int:
    try:
        common = load_json(common_path)
        method_ids = [item.strip() for item in (
            methods.split(',') if methods else METHOD_IDS) if item.strip()]
        plans = []
        for method_id in method_ids:
            if method_id not in METHOD_IDS:
                plans.append(build_method_plan(method_id, common, {}))
                continue
            bundle_file = (REPO_ROOT / 'configs' / 'agentic_eval' / 'methods'
                           / f'{method_id}.v1.json')
            plans.append(build_method_plan(
                method_id, common, load_json(bundle_file)))
        comparison = build_method_plan_comparison(plans)
        out = Path(output).resolve()
        if out.exists():
            raise PipelineRejected(f'output_exists:{out}')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + '\n',
                       encoding='utf-8')
    except Exception as exc:  # noqa: BLE001
        _emit({'schema': METHOD_PLAN_COMPARISON_SCHEMA, 'valid': False,
               'errors': [str(exc)]})
        return 2
    _emit(comparison)
    return 0 if comparison['ok'] else 2


def _cmd_audit_consolidate(manifest: str, output: str) -> int:
    try:
        result = write_audit_consolidated(manifest, output)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [str(exc)]})
        return 2
    _emit(result)
    return int(result['exit_code'])


def _cmd_recording_contract(path: str) -> int:
    try:
        payload = load_json(path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [f'read:{exc}']})
        return 2
    result = validate_recording_contract(payload)
    _emit(result)
    return 0 if result['valid'] else 2


def _cmd_recording_preflight(contract: str, run_dir: str, output: str) -> int:
    try:
        payload = load_json(contract)
        result = recording_preflight(payload, run_dir, output)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [str(exc)]})
        return 2
    _emit(result)
    return 0 if result['playback_status'] == 'READY' else 2


def _cmd_source_run_audit(source_run: str, derived_root: str,
                          provenance_file: str = '') -> int:
    try:
        provenance = load_json(provenance_file) if provenance_file else None
        receipt, path = audit_source_run(
            source_run, derived_root, provenance=provenance)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [str(exc)]})
        return 2
    receipt['receipt_path'] = str(path)
    _emit(receipt)
    return 0 if receipt['source_eligible'] else 2


def _cmd_metrics(run_dir: str, derived_root: str = '',
                 provenance_file: str = '') -> int:
    source = Path(run_dir).resolve()
    provenance = None
    if provenance_file:
        try:
            provenance = load_json(provenance_file)
            require_provenance_authenticity(
                provenance, expected_run_dir=source)
        except Exception as exc:  # noqa: BLE001
            _emit({
                'schema': METRICS_SCHEMA,
                'status': 'fault',
                'fault': True,
                'failed_stage': 'provenance',
                'reason': str(exc),
            })
            return 2
    if not derived_root:
        _emit({
            'schema': METRICS_SCHEMA,
            'status': 'fault',
            'fault': True,
            'failed_stage': 'source_immutable_guard',
            'reason': 'SOURCE_IMMUTABLE_GUARD: --derived-root is required',
        })
        return 2
    try:
        derived = prepare_derived_run(source, derived_root)
    except Exception as exc:  # noqa: BLE001
        _emit({
            'schema': METRICS_SCHEMA,
            'status': 'fault',
            'fault': True,
            'failed_stage': 'derived_run',
            'reason': str(exc),
        })
        return 2
    before = source_hash_manifest(source)
    historical_incident = None
    incident_path = source / 'METRICS_OVERWRITE_INCIDENT.json'
    if incident_path.is_file():
        historical_incident = {
            'path': str(incident_path),
            'sha256': _file_digest(incident_path)['sha256'],
            'read_only': True,
            'note': 'Pre-existing historical incident; not rewritten by this run.',
        }
    result = run_metrics_pipeline(run_dir=str(derived))
    after = source_hash_manifest(source)
    changes = _tree_changes(before, after)
    result.update({
        'source_run': str(source),
        'derived_run': str(derived),
        'derived_root': str(Path(derived_root).resolve()),
        'source_hash_before': before,
        'source_hash_after': after,
        'source_changed_paths': changes,
        'source_unchanged': not changes,
        'historical_metrics_incident': historical_incident,
        'derived_receipt': str(derived / 'PAPER_METRICS_RECEIPT.json'),
    })
    if provenance is not None:
        result['provenance'] = dict(provenance)
    if changes:
        result['status'] = 'fault'
        result['fault'] = True
        result['failed_stage'] = 'source_immutable_guard'
        result['reason'] = 'source_modified_during_derived_run'
    (derived / 'PAPER_METRICS_RECEIPT.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    _emit(result)
    return metrics_exit_code(result)


def _cmd_endpoint(container: str, port: int, path: str) -> int:
    try:
        url = resolve_container_endpoint(container, port=port, path=path)
    except Exception as exc:  # noqa: BLE001
        _emit({'valid': False, 'errors': [str(exc)]})
        return 2
    _emit({'valid': True, 'endpoint': url})
    return 0


def _cmd_prefix(prefix: str) -> int:
    _emit({'valid': True, 'prefix': unique_output_prefix(prefix)})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Arena5 paper-pipeline contracts')
    sub = parser.add_subparsers(dest='cmd', required=True)

    settings = sub.add_parser('settings', help='Validate A/B/C settings and sensor-only contract')
    settings.add_argument('--file', required=True)
    settings.set_defaults(func=lambda a: _cmd_settings(a.file))

    bfast = sub.add_parser('bfast', help='Validate one strict RGB-only B proposal')
    bfast.add_argument('--file', required=True)
    bfast.add_argument('--executor-file', default='',
                       help='JSON common executor action (follow_path poses or stop/hold/error)')
    bfast.add_argument('--flow-risk-file', default='',
                       help='JSON output of compute_flow_risk_proxy')
    group = bfast.add_mutually_exclusive_group(required=True)
    group.add_argument('--enabled', action='store_true')
    group.add_argument('--disabled', action='store_true')
    bfast.set_defaults(func=lambda a: _cmd_bfast(
        a.file, bool(a.enabled), a.executor_file, a.flow_risk_file))

    bfast_profile = sub.add_parser('bfast-profile',
                                   help='Validate one explicit B off/on identity profile')
    bfast_profile.add_argument('--file', required=True)
    bfast_profile.add_argument('--risk-threshold', type=float, default=None)
    bfast_profile.add_argument('--max-frame-gap-sec', type=float, default=None)
    bfast_profile.set_defaults(func=lambda a: _cmd_bfast_profile(
        a.file, a.risk_threshold, a.max_frame_gap_sec))

    latency = sub.add_parser('latency', help='Inject latency without refreshing the source')
    latency.add_argument('--source-stamp-ns', type=int, required=True)
    latency.add_argument('--source-frame-id', required=True)
    latency.add_argument('--source-identity-sha256', required=True)
    latency.add_argument('--latency-ms', type=float, required=True)
    latency.add_argument('--now-sim-ns', type=int, required=True)
    latency.add_argument('--mode', choices=LATENCY_MODES, default='delay')
    latency.add_argument('--previous-response-sim-ns', type=int, default=None)
    latency.set_defaults(func=_cmd_latency)

    scenario = sub.add_parser('scenario', help='Validate static/openloop/closedloop profile')
    scenario.add_argument('--file', required=True)
    scenario.set_defaults(func=lambda a: _cmd_scenario(a.file))

    dataset = sub.add_parser('dataset', help='Validate a dataset split/manifest')
    dataset.add_argument('--manifest', required=True)
    dataset.set_defaults(func=lambda a: _cmd_dataset(a.manifest))

    teacher = sub.add_parser('teacher', help='Validate a C teacher/oracle export manifest')
    teacher.add_argument('--manifest', required=True)
    teacher.set_defaults(func=lambda a: _cmd_teacher(a.manifest))

    teacher_audit = sub.add_parser(
        'teacher-audit', help='Audit C provenance and emit a typed blocker')
    teacher_audit.add_argument('--manifest', required=True)
    teacher_audit.set_defaults(func=lambda a: _cmd_teacher(a.manifest, audit=True))

    agentic = sub.add_parser(
        'agentic-request', help='Validate an operator request and resolve its plan')
    agentic.add_argument('--file', required=True)
    agentic.set_defaults(func=lambda a: _cmd_agentic_request(a.file))

    lifecycle = sub.add_parser(
        'agentic-lifecycle',
        help='Build a deterministic operator lifecycle response')
    lifecycle.add_argument('--file', required=True)
    lifecycle.add_argument('--action', choices=OPERATOR_ACTIONS, required=True)
    lifecycle.add_argument('--from-state', default=None,
                           choices=('new', 'planned', 'started', 'cancelled',
                                    'read_ready', 'finalize_ready',
                                    'audit_ready', 'blocked'))
    lifecycle.set_defaults(func=lambda a: _cmd_agentic_lifecycle(
        a.file, a.action, a.from_state))

    method_plan = sub.add_parser(
        'method-plan',
        help='Build a machine-readable plan for one method bundle')
    method_plan.add_argument('--method', required=True)
    method_plan.add_argument('--common', required=True)
    method_plan.add_argument('--bundle', default='')
    method_plan.add_argument('--output', default='')
    method_plan.set_defaults(func=lambda a: _cmd_method_plan(
        a.method, a.common, a.bundle, a.output))

    method_compare = sub.add_parser(
        'method-compare',
        help='Compare method plans and verify the common contract hash')
    method_compare.add_argument('--common', required=True)
    method_compare.add_argument('--output', required=True)
    method_compare.add_argument('--methods', default='')
    method_compare.set_defaults(func=lambda a: _cmd_method_compare(
        a.common, a.output, a.methods))

    audit = sub.add_parser(
        'audit-consolidate',
        help='Combine audit receipts into one typed consolidated verdict')
    audit.add_argument('--manifest', required=True)
    audit.add_argument('--output', required=True)
    audit.set_defaults(func=lambda a: _cmd_audit_consolidate(
        a.manifest, a.output))

    recording_contract = sub.add_parser(
        'recording-contract',
        help='Validate a versioned future-bag recording contract')
    recording_contract.add_argument('--file', required=True)
    recording_contract.set_defaults(func=lambda a: _cmd_recording_contract(a.file))

    recording_preflight = sub.add_parser(
        'recording-preflight',
        help='Check an existing bag against a recording contract (mode=ro)')
    recording_preflight.add_argument('--contract', required=True)
    recording_preflight.add_argument('--run-dir', required=True)
    recording_preflight.add_argument('--output', required=True)
    recording_preflight.set_defaults(func=lambda a: _cmd_recording_preflight(
        a.contract, a.run_dir, a.output))

    evidence = sub.add_parser(
        'evidence-index', help='Summarize run artifacts and bag playback readiness')
    evidence.add_argument('--run-dir', required=True)
    evidence.add_argument('--output', required=True)
    evidence.add_argument('--sensor-profile', choices=tuple(SENSOR_PROFILES),
                          default='rgb_lidar',
                          help='Required sensors; strict RGB-only B uses rgb_only')
    evidence.add_argument('--provenance-file', default='',
                          help='Optional provenance envelope JSON to embed')
    evidence.set_defaults(func=lambda a: _cmd_evidence_index(
        a.run_dir, a.output, a.sensor_profile, a.provenance_file))

    verdict = sub.add_parser(
        'evidence-verdict',
        help='Classify a saved evidence index deterministically')
    verdict.add_argument('--index', required=True)
    verdict.add_argument('--output', required=True)
    verdict.add_argument('--provenance-file', default='',
                         help='Optional provenance envelope JSON to embed')
    verdict.set_defaults(func=lambda a: _cmd_evidence_verdict(
        a.index, a.output, a.provenance_file))

    source_audit = sub.add_parser(
        'source-run-audit',
        help='Read-only audit of a historical source run; receipt only in derived root')
    source_audit.add_argument('--source-run', required=True)
    source_audit.add_argument('--derived-root', required=True)
    source_audit.add_argument('--provenance-file', default='')
    source_audit.set_defaults(func=lambda a: _cmd_source_run_audit(
        a.source_run, a.derived_root, a.provenance_file))

    metrics = sub.add_parser('metrics', help='Run metrics stages with fail-close exit codes')
    metrics.add_argument('--run-dir', required=True)
    metrics.add_argument('--derived-root', required=True,
                         help='Fresh derived root; the source run is never written.')
    metrics.add_argument('--provenance-file', default='',
                         help='Optional provenance envelope JSON to embed')
    metrics.set_defaults(func=lambda a: _cmd_metrics(
        a.run_dir, a.derived_root, a.provenance_file))

    endpoint = sub.add_parser('endpoint', help='Resolve a Docker container HTTP endpoint')
    endpoint.add_argument('--container', required=True)
    endpoint.add_argument('--port', type=int, default=8000)
    endpoint.add_argument('--path', default='/v1/chat/completions')
    endpoint.set_defaults(func=lambda a: _cmd_endpoint(a.container, a.port, a.path))

    prefix = sub.add_parser('prefix', help='Return a unique output prefix')
    prefix.add_argument('--prefix', required=True)
    prefix.set_defaults(func=lambda a: _cmd_prefix(a.prefix))

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
