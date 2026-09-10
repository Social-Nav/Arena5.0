"""Versioned configuration for the current Arena benchmark entry point.

This is intentionally not a ROS parameter file.  The benchmark driver is a
process orchestrator which eventually creates ROS nodes and launch arguments,
so its profile is loaded before argparse and translated into driver defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
DEFAULT_PROFILE_RELATIVE = 'configs/benchmark/profiles/internnav_grscenes.yaml'


class BenchmarkProfileError(ValueError):
    """Raised when a benchmark profile is missing, malformed, or unsupported."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int | float) and not isinstance(value, bool))


FIELD_TYPES: dict[str, dict[str, Any]] = {
    'runtime': {
        'ros_domain_id': _is_int,
        'rmw_implementation': str,
        'fastdds_builtin_transports': str,
        'ros_localhost_only': _is_int,
        'ros_automatic_discovery_range': str,
        'device': str,
        'http_port': _is_int,
    },
    'server': {
        'mode': str,
        'model_path': str,
        'rgb_topic': str,
        'depth_topic': str,
        'camera_info_topic': str,
        'odom_topic': str,
        'raw_cmd_vel_topic': str,
        'status_topic': str,
        'visualization_topic': str,
        'action_visualization_topic': str,
        'model_output_topic': str,
        'model_output_policy': str,
        'visualization_enabled': bool,
        'visualization_rate_hz': _is_number,
        'planning_rate_hz': _is_number,
        'inference_timeout_sec': _is_number,
    },
    'evaluation': {
        'simulator': str,
        'human_simulator': str,
        'world': str,
        'scenario': str,
        'robot': str,
        'local_planner': str,
        'inter_planner': str,
        'global_planner': str,
        'episodes': _is_int,
        'timeout_sec': _is_int,
        'timeout_wall_factor': _is_number,
        'timeout_wall_sec': _is_number,
        'task_generator_robots': str,
        'task_generator_obstacles': str,
        'headless': str,
        'log_level': str,
        'vln_instruction': str,
        'social_eval': bool,
        'launch_timeout_sec': _is_number,
        'shutdown_grace_period_sec': _is_number,
        'output_prefix': str,
    },
    'internnav': {
        'adapter_target': str,
        'external_server': bool,
        'direct_cmd_vel': bool,
        'timing_mode': str,
        'model_latency_sec': _is_number,
        'latency_policy': str,
    },
    'video': {
        'enabled': bool,
        'fps': _is_number,
        'top_down_size_px': _is_int,
        'top_down_window_m': _is_number,
        'sim_top_down_topic': str,
        'debug_overlay_topic': str,
    },
}

MACHINE_SPECIFIC_PROFILE_FIELDS = frozenset({('runtime', 'device')})

CHOICES: dict[tuple[str, str], frozenset[str]] = {
    ('runtime', 'rmw_implementation'): frozenset({'rmw_fastrtps_cpp'}),
    ('runtime', 'fastdds_builtin_transports'): frozenset({'UDPv4'}),
    ('runtime', 'ros_automatic_discovery_range'): frozenset({'SUBNET'}),
    ('server', 'mode'): frozenset({'internnav'}),
    ('server', 'model_output_policy'): frozenset({'trajectory', 'discrete', 'raw'}),
    ('evaluation', 'simulator'): frozenset({'isaac_eval'}),
    ('evaluation', 'human_simulator'): frozenset({'hunav'}),
    ('evaluation', 'headless'): frozenset({'-1', '0', '1', '2'}),
    ('evaluation', 'log_level'): frozenset({'debug', 'info', 'warn', 'error', 'fatal'}),
    ('internnav', 'timing_mode'): frozenset({'wall', 'sim_time_realworld'}),
    ('internnav', 'latency_policy'): frozenset({'fixed', 'measured'}),
}

REQUIRED_FIELDS = {section: frozenset(fields) for section, fields in FIELD_TYPES.items()}
TOP_LEVEL_FIELDS = frozenset({'schema_version', 'id', 'description', *FIELD_TYPES})

EVALUATION_FIELD_MAP = {
    'simulator': 'sim',
    'human_simulator': 'human',
    'world': 'world',
    'scenario': 'scenario_file',
    'robot': 'robot',
    'local_planner': 'local_planner',
    'inter_planner': 'inter_planner',
    'global_planner': 'global_planner',
    'episodes': 'episodes',
    'timeout_sec': 'timeout',
    'timeout_wall_factor': 'timeout_wall_factor',
    'timeout_wall_sec': 'timeout_wall_sec',
    'task_generator_robots': 'tm_robots',
    'task_generator_obstacles': 'tm_obstacles',
    'headless': 'headless',
    'log_level': 'log_level',
    'vln_instruction': 'vln_instruction',
    'social_eval': 'social_eval',
    'launch_timeout_sec': 'launch_timeout_sec',
    'shutdown_grace_period_sec': 'shutdown_grace_period_sec',
    'output_prefix': 'output_prefix',
}

SERVER_EVAL_FIELD_MAP = {
    'model_path': 'dual_vln_model_path',
    'rgb_topic': 'dual_vln_rgb_topic',
    'depth_topic': 'dual_vln_depth_topic',
    'camera_info_topic': 'dual_vln_camera_info_topic',
    'raw_cmd_vel_topic': 'dual_vln_raw_cmd_vel_topic',
    'status_topic': 'dual_vln_status_topic',
    'visualization_topic': 'dual_vln_visualization_topic',
    'action_visualization_topic': 'dual_vln_action_visualization_topic',
    'model_output_topic': 'dual_vln_model_output_topic',
    'model_output_policy': 'dual_vln_model_output_policy',
    'visualization_enabled': 'dual_vln_enable_visualization',
    'visualization_rate_hz': 'dual_vln_visualization_rate_hz',
    'planning_rate_hz': 'dual_vln_inference_rate_hz',
    'inference_timeout_sec': 'dual_vln_inference_timeout_sec',
}

INTERNNAV_EVAL_FIELD_MAP = {
    'adapter_target': 'dual_vln_adapter_target',
    'external_server': 'internnav_external_server',
    'direct_cmd_vel': 'internnav_direct_cmd_vel',
    'timing_mode': 'dual_vln_timing_mode',
    'model_latency_sec': 'dual_vln_model_latency_sec',
    'latency_policy': 'dual_vln_latency_policy',
}

VIDEO_EVAL_FIELD_MAP = {
    'enabled': 'save_eval_video',
    'fps': 'eval_video_fps',
    'top_down_size_px': 'eval_video_top_down_size_px',
    'top_down_window_m': 'eval_video_top_down_window_m',
    'sim_top_down_topic': 'eval_video_sim_top_down_topic',
    'debug_overlay_topic': 'eval_video_debug_overlay_topic',
}


def default_profile_path(package_share: str | Path) -> Path:
    return Path(package_share) / DEFAULT_PROFILE_RELATIVE


def _type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, type):
        return isinstance(value, expected)
    return bool(expected(value))


def validate_profile(data: Any, *, source: str = '<profile>') -> dict[str, Any]:
    if not isinstance(data, dict):
        raise BenchmarkProfileError(f'{source}: profile root must be a mapping')
    unknown_top = sorted(set(data) - TOP_LEVEL_FIELDS)
    if unknown_top:
        raise BenchmarkProfileError(f'{source}: unknown top-level fields: {unknown_top}')
    if data.get('schema_version') != SCHEMA_VERSION:
        raise BenchmarkProfileError(
            f'{source}: schema_version must be {SCHEMA_VERSION}, got {data.get("schema_version")!r}'
        )
    profile_id = data.get('id')
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise BenchmarkProfileError(f'{source}: id must be a non-empty string')

    for section, field_types in FIELD_TYPES.items():
        value = data.get(section)
        if not isinstance(value, dict):
            raise BenchmarkProfileError(f'{source}: {section} must be a mapping')
        unknown = sorted(set(value) - set(field_types))
        if unknown:
            raise BenchmarkProfileError(f'{source}: unknown {section} fields: {unknown}')
        missing = sorted(REQUIRED_FIELDS[section] - set(value))
        if missing:
            raise BenchmarkProfileError(f'{source}: missing {section} fields: {missing}')
        for field, expected in field_types.items():
            if not _type_matches(value[field], expected):
                raise BenchmarkProfileError(
                    f'{source}: {section}.{field} has invalid type/value {value[field]!r}'
                )
            choices = CHOICES.get((section, field))
            if choices is not None and value[field] not in choices:
                raise BenchmarkProfileError(
                    f'{source}: {section}.{field} must be one of {sorted(choices)}, got {value[field]!r}'
                )

    runtime = data['runtime']
    server = data['server']
    evaluation = data['evaluation']
    internnav = data['internnav']
    video = data['video']
    if runtime['ros_domain_id'] != 1:
        raise BenchmarkProfileError(
            f'{source}: runtime.ros_domain_id must be 1 for the current compose topology'
        )
    if runtime['ros_localhost_only'] != 0:
        raise BenchmarkProfileError(
            f'{source}: runtime.ros_localhost_only must be 0 for cross-container discovery'
        )
    if runtime['http_port'] <= 0 or runtime['http_port'] > 65535:
        raise BenchmarkProfileError(f'{source}: runtime.http_port must be a valid TCP port')
    if evaluation['episodes'] != 1:
        raise BenchmarkProfileError(f'{source}: evaluation.episodes must be 1')
    if evaluation['timeout_sec'] <= 0:
        raise BenchmarkProfileError(f'{source}: evaluation.timeout_sec must be positive')
    if not internnav['external_server'] or not internnav['direct_cmd_vel']:
        raise BenchmarkProfileError(
            f'{source}: current benchmark requires external_server=true and direct_cmd_vel=true'
        )
    if evaluation['local_planner'] != 'dual_vln':
        raise BenchmarkProfileError(f'{source}: evaluation.local_planner must be dual_vln')
    if not evaluation['social_eval']:
        raise BenchmarkProfileError(f'{source}: evaluation.social_eval must be true')
    if not video['enabled']:
        raise BenchmarkProfileError(f'{source}: video.enabled must be true')
    if video['fps'] <= 0 or video['top_down_size_px'] <= 0 or video['top_down_window_m'] <= 0:
        raise BenchmarkProfileError(f'{source}: video numeric fields must be positive')
    return data


def load_profile(path: str | os.PathLike[str]) -> dict[str, Any]:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise BenchmarkProfileError(f'benchmark profile does not exist: {profile_path}')
    try:
        data = yaml.safe_load(profile_path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise BenchmarkProfileError(f'{profile_path}: invalid YAML: {exc}') from exc
    return validate_profile(data, source=str(profile_path))


def profile_sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def profile_metadata(path: str | os.PathLike[str], data: dict[str, Any] | None = None) -> dict[str, Any]:
    profile_path = Path(path).expanduser().resolve()
    profile = data if data is not None else load_profile(profile_path)
    return {
        'id': profile['id'],
        'schema_version': profile['schema_version'],
        'path': str(profile_path),
        'sha256': profile_sha256(profile_path),
    }


def manifest_record(
    metadata: dict[str, Any],
    resolved_parameters: dict[str, Any],
    *,
    snapshot_path: str | None,
) -> dict[str, Any]:
    return {
        **metadata,
        'snapshot_path': snapshot_path,
        'precedence': 'cli_or_case > machine_local_paths_or_device > benchmark_profile > legacy_code_defaults',
        'resolved_parameters': dict(resolved_parameters),
    }


def evaluation_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        dest: profile['evaluation'][field]
        for field, dest in EVALUATION_FIELD_MAP.items()
    }
    defaults.update({
        dest: profile['server'][field]
        for field, dest in SERVER_EVAL_FIELD_MAP.items()
    })
    defaults.update({
        dest: profile['internnav'][field]
        for field, dest in INTERNNAV_EVAL_FIELD_MAP.items()
    })
    defaults.update({
        dest: profile['video'][field]
        for field, dest in VIDEO_EVAL_FIELD_MAP.items()
    })
    defaults['dual_vln_mode'] = profile['server']['mode']
    defaults['dual_vln_device'] = profile['runtime']['device']
    return defaults


def shell_assignments(path: str | os.PathLike[str], profile: dict[str, Any]) -> str:
    metadata = profile_metadata(path, profile)
    values: dict[str, Any] = {
        'PROFILE_ID': metadata['id'],
        'PROFILE_SCHEMA_VERSION': metadata['schema_version'],
        'PROFILE_PATH': metadata['path'],
        'PROFILE_SHA256': metadata['sha256'],
    }
    for section in FIELD_TYPES:
        for field, value in profile[section].items():
            values[f'PROFILE_{section}_{field}'.upper()] = value
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = 'true' if value else 'false'
        lines.append(f'{key}={shlex.quote(str(value))}')
    return '\n'.join(lines)


def validate_machine_environment(
    profile: dict[str, Any],
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return allowed machine-local overrides and reject semantic drift."""
    env = os.environ if environment is None else environment
    runtime = profile['runtime']
    required = {
        'ROS_DOMAIN_ID': str(runtime['ros_domain_id']),
        'RMW_IMPLEMENTATION': runtime['rmw_implementation'],
        'FASTDDS_BUILTIN_TRANSPORTS': runtime['fastdds_builtin_transports'],
        'ROS_LOCALHOST_ONLY': str(runtime['ros_localhost_only']),
        'ROS_AUTOMATIC_DISCOVERY_RANGE': runtime['ros_automatic_discovery_range'],
    }
    conflicts = {
        name: {'environment': str(env[name]), 'profile': expected}
        for name, expected in required.items()
        if name in env and str(env[name]).strip() and str(env[name]).strip() != expected
    }
    if conflicts:
        details = ', '.join(
            f"{name}={values['environment']!r} (profile {values['profile']!r})"
            for name, values in sorted(conflicts.items())
        )
        raise BenchmarkProfileError(f'ambient ROS environment conflicts with benchmark profile: {details}')
    return {
        'device': str(env.get('ARENA_BENCHMARK_DEVICE') or runtime['device']),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validate and inspect an Arena benchmark profile.')
    parser.add_argument('command', choices=('validate', 'json', 'shell'))
    parser.add_argument('--profile', required=True)
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
        metadata = profile_metadata(args.profile, profile)
    except BenchmarkProfileError as exc:
        parser.error(str(exc))

    if args.command == 'validate':
        print(f"OK {metadata['id']} schema={metadata['schema_version']} sha256={metadata['sha256']} path={metadata['path']}")
    elif args.command == 'json':
        print(json.dumps({'metadata': metadata, 'profile': profile}, indent=2, sort_keys=True))
    else:
        print(shell_assignments(args.profile, profile))
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
