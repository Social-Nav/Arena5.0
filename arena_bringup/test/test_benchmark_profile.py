import hashlib
import sys
from pathlib import Path

import pytest
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from arena_bringup.benchmark_profile import (
    BenchmarkProfileError,
    evaluation_defaults,
    load_profile,
    manifest_record,
    profile_metadata,
    validate_machine_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPO_ROOT / 'arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml'


def test_tracked_default_profile_is_valid_and_isaac_only():
    profile = load_profile(PROFILE)
    assert profile['id'] == 'internnav_grscenes'
    assert profile['evaluation']['simulator'] == 'isaac_eval'
    assert profile['evaluation']['episodes'] == 1
    assert profile['internnav']['external_server'] is True
    assert profile['internnav']['direct_cmd_vel'] is True
    assert 'gazebo' not in PROFILE.read_text(encoding='utf-8').lower()


def test_profile_metadata_contains_content_hash():
    metadata = profile_metadata(PROFILE)
    assert metadata['sha256'] == hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    assert metadata['path'] == str(PROFILE.resolve())


def test_profile_values_become_eval_defaults():
    defaults = evaluation_defaults(load_profile(PROFILE))
    assert defaults['sim'] == 'isaac_eval'
    assert defaults['world'] == 'grscenes_20_v1'
    assert defaults['scenario_file'] == 'default_2'
    assert defaults['dual_vln_device'] == 'cuda:0'
    assert defaults['internnav_external_server'] is True
    assert defaults['internnav_direct_cmd_vel'] is True
    assert defaults['save_eval_video'] is True


def test_manifest_record_keeps_identity_snapshot_and_resolved_values():
    metadata = profile_metadata(PROFILE)
    record = manifest_record(
        metadata,
        {'world': 'grscenes_15_v1', 'scenario': 'default_1'},
        snapshot_path='/run/snapshots/internnav_grscenes.yaml',
    )
    assert record['id'] == 'internnav_grscenes'
    assert record['sha256'] == metadata['sha256']
    assert record['snapshot_path'].endswith('internnav_grscenes.yaml')
    assert record['resolved_parameters']['world'] == 'grscenes_15_v1'


def test_machine_environment_allows_device_but_rejects_dds_drift():
    profile = load_profile(PROFILE)
    machine = validate_machine_environment(profile, {'ARENA_BENCHMARK_DEVICE': 'cuda:1'})
    assert machine['device'] == 'cuda:1'
    with pytest.raises(BenchmarkProfileError, match='ROS_DOMAIN_ID'):
        validate_machine_environment(profile, {'ROS_DOMAIN_ID': '7'})


def test_unknown_profile_field_fails_closed(tmp_path):
    data = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
    data['evaluation']['typo_timeout'] = 1
    invalid = tmp_path / 'invalid.yaml'
    invalid.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(BenchmarkProfileError, match='unknown evaluation fields'):
        load_profile(invalid)


def test_non_isaac_profile_is_rejected(tmp_path):
    data = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
    data['evaluation']['simulator'] = 'dummy'
    invalid = tmp_path / 'invalid.yaml'
    invalid.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(BenchmarkProfileError, match='evaluation.simulator must be one of'):
        load_profile(invalid)


def test_cross_container_domain_is_fixed(tmp_path):
    data = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
    data['runtime']['ros_domain_id'] = 7
    invalid = tmp_path / 'invalid.yaml'
    invalid.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(BenchmarkProfileError, match='ros_domain_id must be 1'):
        load_profile(invalid)


@pytest.mark.parametrize(
    ('section', 'field', 'value', 'message'),
    [
        ('evaluation', 'local_planner', 'dwb', 'local_planner must be dual_vln'),
        ('evaluation', 'social_eval', False, 'social_eval must be true'),
        ('internnav', 'external_server', False, 'requires external_server=true'),
        ('video', 'enabled', False, 'video.enabled must be true'),
    ],
)
def test_strict_contract_fields_fail_closed(tmp_path, section, field, value, message):
    data = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
    data[section][field] = value
    invalid = tmp_path / 'invalid.yaml'
    invalid.write_text(yaml.safe_dump(data), encoding='utf-8')
    with pytest.raises(BenchmarkProfileError, match=message):
        load_profile(invalid)
