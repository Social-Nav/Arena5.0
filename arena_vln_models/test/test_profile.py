"""Version 3 profile validation, reproducibility and evaluation defaults."""
from pathlib import Path

import pytest

from arena_vln_models.profile import (
    BenchmarkProfileError,
    default_profile_path,
    evaluation_defaults,
    load_profile,
    manifest_record,
    profile_metadata,
    validate_machine_environment,
    validate_profile,
)


@pytest.fixture
def profile_path():
    return (Path(__file__).resolve().parents[2]
            / 'arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml')


@pytest.fixture
def profile(profile_path):
    return load_profile(profile_path)


@pytest.mark.parametrize('section', [None, 'evaluation', 'execution', 'external'])
def test_unknown_profile_fields_are_rejected(profile, section):
    target = profile if section is None else profile[section]
    target['legacy_control_option'] = True
    with pytest.raises(BenchmarkProfileError, match='invalid profile fields|must contain exactly'):
        validate_profile(profile)


@pytest.mark.parametrize('section, field, value', [
    ('runtime', 'ros_domain_id', True),
    ('evaluation', 'episodes', '1'),
    ('evaluation', 'social_eval', 'true'),
    ('execution', 'heartbeat_timeout_sec', True),
    ('execution', 'heartbeat_timeout_sec', float('nan')),
    ('video', 'fps', float('inf')),
    ('external', 'command', 'python client.py'),
    ('external', 'command', ['python', '']),
])
def test_invalid_profile_value_types_are_rejected(profile, section, field, value):
    profile[section][field] = value
    with pytest.raises(BenchmarkProfileError, match=rf'{section}\.{field}'):
        validate_profile(profile)


@pytest.mark.parametrize('section, field, value, message', [
    ('execution', 'mode', 'direct_velocity', 'nav2_follow_path'),
    ('execution', 'heartbeat_timeout_sec', 0, 'positive'),
    ('evaluation', 'local_planner', 'dwb', 'RPP tracking'),
    ('evaluation', 'episodes', 2, 'one episode'),
    ('runtime', 'ros_domain_id', 233, 'ROS domain'),
])
def test_profile_rejects_unsupported_execution_settings(profile, section, field, value, message):
    profile[section][field] = value
    with pytest.raises(BenchmarkProfileError, match=message):
        validate_profile(profile)


@pytest.mark.parametrize('name, conflicting_value', [
    ('ROS_DOMAIN_ID', '97'),
    ('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
    ('FASTDDS_BUILTIN_TRANSPORTS', 'SHM'),
    ('ROS_LOCALHOST_ONLY', '1'),
    ('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST'),
])
def test_machine_environment_rejects_each_conflicting_ros_setting(profile, name, conflicting_value):
    with pytest.raises(BenchmarkProfileError, match=name):
        validate_machine_environment(profile, {name: conflicting_value})


def test_machine_environment_ignores_unrelated_values_and_accepts_matching_ros_settings(profile):
    expected = {
        'ROS_DOMAIN_ID': '1',
        'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
        'FASTDDS_BUILTIN_TRANSPORTS': 'UDPv4',
        'ROS_LOCALHOST_ONLY': '0',
        'ROS_AUTOMATIC_DISCOVERY_RANGE': 'SUBNET',
    }
    assert validate_machine_environment(profile, {}) == expected
    environment = {**expected, 'HOME': '/machine/local/path', 'ARENA_IMAGE': 'arena:custom'}
    before = environment.copy()
    assert validate_machine_environment(profile, environment) == expected
    assert environment == before
    assert validate_machine_environment(profile, {'ROS_DOMAIN_ID': ' '}) == expected


def test_metadata_identifies_exact_profile_bytes_and_manifest_snapshot(profile_path, tmp_path):
    copied = tmp_path / 'profile.yaml'
    copied.write_bytes(profile_path.read_bytes())
    first = profile_metadata(copied)
    assert first['id'] == 'internnav_grscenes'
    assert first['schema_version'] == 3
    assert first['path'] == str(copied.resolve())
    assert len(first['sha256']) == 64
    assert profile_metadata(copied)['sha256'] == first['sha256']

    # Even a comment changes the published input fingerprint, while keeping
    # the effective profile valid and its experiment identity unchanged.
    copied.write_bytes(copied.read_bytes() + b'\n# machine-specific profile copy\n')
    second = profile_metadata(copied)
    assert second['id'] == first['id']
    assert second['sha256'] != first['sha256']
    resolved = {'world': 'grscenes_14_v1', 'vln_instruction': 'Turn left.'}
    record = manifest_record(second, resolved, snapshot_path='benchmark_profile.yaml')
    resolved['world'] = 'later_change'
    assert record['sha256'] == second['sha256']
    assert record['snapshot_path'] == 'benchmark_profile.yaml'
    assert record['precedence'] == 'cli_or_case > benchmark_profile > code_defaults'
    assert record['resolved_parameters']['world'] == 'grscenes_14_v1'


def test_evaluation_defaults_preserve_profile_inputs_and_use_generic_interface(profile):
    profile['model'] = 'another_model'
    profile['evaluation'].update(
        simulator='isaac_eval', world='grscenes_14_v1', scenario='default_1',
        task_generator_obstacles='scenario', timeout_sec=123, vln_instruction='Turn left.',
    )
    profile['video'].update(enabled=False, debug_overlay_topic='/optional/debug')
    profile['execution'].update(rgb_topic='rgb/image', heartbeat_timeout_sec=4.0)
    defaults = evaluation_defaults(validate_profile(profile))
    expected = {
        'sim': 'isaac_eval', 'world': 'grscenes_14_v1', 'scenario_file': 'default_1',
        'tm_obstacles': 'scenario', 'timeout': 123, 'vln_instruction': 'Turn left.',
        'save_eval_video': False, 'eval_video_debug_overlay_topic': '/optional/debug',
        'model_backend': 'another_model', 'execution_mode': 'nav2_follow_path',
        'vln_rgb_topic': 'rgb/image', 'model_status_topic': 'model/status',
        'model_heartbeat_timeout_sec': 4.0,
    }
    assert {name: defaults[name] for name in expected} == expected


def test_default_profile_path_is_relative_to_bringup_share(profile_path):
    share = profile_path.parents[3]
    assert default_profile_path(share) == profile_path
