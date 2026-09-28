import hashlib
import json

import arena_bringup.internnav_eval as internnav_eval
from arena_bringup.internnav_eval import _system2_checkpoint_provenance


def test_system2_checkpoint_provenance_records_index_and_revision(tmp_path):
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text('{"weight_map": {"model.w": "shard.safetensors"}}', encoding='utf-8')
    revision = {"repo_id": "org/socialgen", "revision": "abc123"}
    (tmp_path / '.arena_system2_revision.json').write_text(
        json.dumps(revision), encoding='utf-8'
    )

    result = _system2_checkpoint_provenance(str(tmp_path))

    assert result['path'] == str(tmp_path.resolve())
    assert result['revision'] == revision
    assert result['index_sha256'] == hashlib.sha256(index.read_bytes()).hexdigest()


def test_empty_system2_checkpoint_has_no_provenance():
    assert _system2_checkpoint_provenance('') is None


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
