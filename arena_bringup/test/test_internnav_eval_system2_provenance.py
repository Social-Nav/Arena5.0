import hashlib
import json

import arena_bringup.internnav_eval as internnav_eval
from arena_bringup.internnav_eval import (
    _checkpoint_provenance,
    _runtime_container_provenance,
    _system2_checkpoint_provenance,
)


def test_system2_checkpoint_provenance_records_index_and_revision(tmp_path):
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(
        '{"metadata": {"total_size": 3}, "weight_map": {"model.w": "shard.safetensors"}}',
        encoding='utf-8',
    )
    (tmp_path / 'shard.safetensors').write_bytes(b'abc')
    revision = {"repo_id": "org/socialgen", "revision": "abc123"}
    (tmp_path / '.arena_system2_revision.json').write_text(
        json.dumps(revision), encoding='utf-8'
    )

    result = _system2_checkpoint_provenance(str(tmp_path))

    assert result['path'] == str(tmp_path.resolve())
    assert result['revision'] == revision
    assert result['index_sha256'] == hashlib.sha256(index.read_bytes()).hexdigest()
    assert result['tensor_count'] == 1
    assert result['weight_shard_count'] == 1
    assert result['declared_total_size_bytes'] == 3
    assert result['weight_shards'] == [{'name': 'shard.safetensors', 'size_bytes': 3}]


def test_empty_system2_checkpoint_has_no_provenance():
    assert _system2_checkpoint_provenance('') is None


def test_base_checkpoint_provenance_uses_base_revision_file(tmp_path):
    (tmp_path / 'config.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'vocab.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'merges.txt').write_text('#version: 0.2\n', encoding='utf-8')
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    (tmp_path / 'model.safetensors.index.json').write_text(
        '{"weight_map": {"model.w": "model.safetensors"}}', encoding='utf-8'
    )
    revision = {'repo_id': 'org/base', 'revision': 'base123'}
    (tmp_path / '.arena_checkpoint_revision.json').write_text(
        json.dumps(revision), encoding='utf-8'
    )

    result = _checkpoint_provenance(str(tmp_path))

    assert result['revision'] == revision
    assert result['fingerprint_complete'] is True
    assert result['fingerprint_errors'] == []
    assert len(result['fingerprint_manifest_sha256']) == 64
    assert {
        item['name'] for item in result['fingerprinted_files']
    } >= {
        'model.safetensors.index.json',
        'model.safetensors',
        'config.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt',
    }
    assert all(
        item['size_bytes'] >= 0 and len(item['sha256']) == 64
        for item in result['fingerprinted_files']
    )


def test_base_checkpoint_fingerprint_fails_closed_when_runtime_assets_are_missing(tmp_path):
    (tmp_path / 'model.safetensors.index.json').write_text(
        '{"weight_map": {"model.w": "missing.safetensors"}}', encoding='utf-8'
    )

    result = _checkpoint_provenance(str(tmp_path))

    assert result['fingerprint_complete'] is False
    assert 'weight_shard_fingerprint_incomplete' in result['fingerprint_errors']
    assert 'model_config_not_fingerprinted' in result['fingerprint_errors']
    assert 'tokenizer_config_not_fingerprinted' in result['fingerprint_errors']


def test_runtime_container_provenance_is_explicit_and_service_scoped(monkeypatch):
    monkeypatch.setenv('ARENA_EVAL_ISAAC_CONTAINER_ID', 'container-isaac')
    monkeypatch.setenv('ARENA_EVAL_ISAAC_CONTAINER_IMAGE_ID', 'sha256:isaac')
    monkeypatch.setenv('ARENA_EVAL_ISAAC_CONTAINER_IMAGE_REF', 'arena:dev')

    result = _runtime_container_provenance()

    assert result == {
        'isaac': {
            'container_id': 'container-isaac',
            'image_id': 'sha256:isaac',
            'image_ref': 'arena:dev',
        }
    }


def test_container_system2_path_resolves_to_workspace_for_provenance(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'deps' / 'models' / 'socialgen'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'model.safetensors.index.json').write_text(
        '{"weight_map": {}}', encoding='utf-8'
    )
    monkeypatch.setattr(internnav_eval, '_workspace_root_from_runtime', lambda: str(tmp_path))

    result = _system2_checkpoint_provenance('/opt/arena_ws/deps/models/socialgen')

    assert result['path'] == '/opt/arena_ws/deps/models/socialgen'
    assert result['host_path'] == str(checkpoint)
    assert result['index_sha256'] == hashlib.sha256(
        (checkpoint / 'model.safetensors.index.json').read_bytes()
    ).hexdigest()
