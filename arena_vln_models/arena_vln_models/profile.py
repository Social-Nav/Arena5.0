"""Version 3: model-independent native FollowPath benchmark configuration."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any
import yaml

SCHEMA_VERSION = 3
DEFAULT_PROFILE_RELATIVE = 'configs/benchmark/profiles/internnav_grscenes.yaml'

class BenchmarkProfileError(ValueError):
    """Raised when a benchmark profile is missing, malformed, or unsupported."""

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)

def _is_number(value: Any) -> bool:
    return (isinstance(value, int | float) and not isinstance(value, bool))

FIELD_TYPES = {'runtime': {'ros_domain_id': _is_int,
             'rmw_implementation': str,
             'fastdds_builtin_transports': str,
             'ros_localhost_only': _is_int,
             'ros_automatic_discovery_range': str},
 'evaluation': {'simulator': str,
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
                'output_prefix': str},
 'video': {'enabled': bool,
           'fps': _is_number,
           'top_down_size_px': _is_int,
           'top_down_window_m': _is_number,
           'sim_top_down_topic': str,
           'debug_overlay_topic': str}}
FIELD_TYPES['execution'] = {'mode': str, 'rgb_topic': str, 'heartbeat_timeout_sec': _is_number}
FIELD_TYPES['external'] = {'command': list}

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


def validate_profile(data, *, source='<profile>'):
    if not isinstance(data, dict) or data.get('schema_version') != SCHEMA_VERSION:
        raise BenchmarkProfileError(f'{source}: migrate to schema_version=3 / nav2_follow_path; '
                                    'model adapters, direct velocity, MPC and timed actions were removed')
    allowed = {'schema_version', 'id', 'description', 'model', *FIELD_TYPES}
    if set(data) - allowed or any(not isinstance(data.get(k), str) or not data[k].strip() for k in ('id', 'model')):
        raise BenchmarkProfileError(f'{source}: invalid profile fields, id or model label')
    for section, fields in FIELD_TYPES.items():
        actual = data.get(section)
        if not isinstance(actual, dict) or set(actual) != set(fields):
            raise BenchmarkProfileError(f'{source}: {section} must contain exactly {sorted(fields)}')
        for key, kind in fields.items():
            value = actual[key]
            good = isinstance(value, kind) if isinstance(kind, type) else kind(value)
            if not good or (isinstance(value, float) and not math.isfinite(value)):
                raise BenchmarkProfileError(f'{source}: invalid {section}.{key}')
    if data['execution']['mode'] != 'nav2_follow_path':
        raise BenchmarkProfileError(f'{source}: execution.mode must be nav2_follow_path')
    ev = data['evaluation']
    if (ev['local_planner'] != 'regulated_pure_pursuit' or ev['inter_planner'] != 'none'
            or ev['global_planner'] != 'none'):
        raise BenchmarkProfileError(f'{source}: use RPP tracking with inter_planner/global_planner=none')
    for section, keys in {'evaluation': ('episodes', 'timeout_sec', 'timeout_wall_factor',
                                          'launch_timeout_sec', 'shutdown_grace_period_sec'),
                          'video': ('fps', 'top_down_size_px', 'top_down_window_m'),
                          'execution': ('heartbeat_timeout_sec',)}.items():
        if any(data[section][key] <= 0 for key in keys):
            raise BenchmarkProfileError(f'{source}: {section} durations/counts must be positive')
    if ev['episodes'] != 1:
        raise BenchmarkProfileError(f'{source}: each benchmark result must contain one episode; use full-eval for scene loops')
    command = data['external']['command']
    if any(not isinstance(arg, str) or not arg or '\0' in arg for arg in command):
        raise BenchmarkProfileError(f'{source}: external.command must be an argv list of nonempty strings')
    if not 0 <= data['runtime']['ros_domain_id'] <= 232:
        raise BenchmarkProfileError(f'{source}: invalid ROS domain')
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
        'precedence': 'cli_or_case > benchmark_profile > code_defaults',
        'resolved_parameters': dict(resolved_parameters),
    }

def shell_assignments(path: str | os.PathLike[str], profile: dict[str, Any]) -> str:
    metadata = profile_metadata(path, profile)
    values: dict[str, Any] = {
        'PROFILE_ID': metadata['id'],
        'PROFILE_SCHEMA_VERSION': metadata['schema_version'],
        'PROFILE_PATH': metadata['path'],
        'PROFILE_SHA256': metadata['sha256'],
    }
    for section, fields in profile.items():
        if not isinstance(fields, dict):
            continue
        for field, value in fields.items():
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
    return required


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

def evaluation_defaults(profile):
    values = {dest: profile['evaluation'][field] for field, dest in EVALUATION_FIELD_MAP.items()}
    values.update({dest: profile['video'][field] for field, dest in VIDEO_EVAL_FIELD_MAP.items()})
    values.update(model_backend=profile['model'], execution_mode='nav2_follow_path',
                  vln_rgb_topic=profile['execution']['rgb_topic'], model_status_topic='model/status',
                  model_heartbeat_timeout_sec=profile['execution']['heartbeat_timeout_sec'])
    return values

if __name__ == '__main__':
    raise SystemExit(main())
