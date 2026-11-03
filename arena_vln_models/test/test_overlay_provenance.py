"""Overlay provenance identity and install inventory."""
import json

from arena_vln_models.provenance import (
    collect_overlay_provenance,
    install_inventory,
    summary,
)


def test_install_inventory_is_deterministic_and_tracks_content(tmp_path):
    root = tmp_path / 'install'
    (root / 'lib').mkdir(parents=True)
    (root / 'setup.bash').write_text('source x\n', encoding='utf-8')
    (root / 'lib' / 'pkg').symlink_to('/src/pkg')

    first = install_inventory(root)
    second = install_inventory(root)
    assert first['install_inventory_sha256'] == second['install_inventory_sha256']
    assert first['install_file_count'] == 1
    assert first['install_symlink_count'] == 1
    assert first['install_entry_count'] == 3

    (root / 'setup.bash').write_text('source y\n', encoding='utf-8')
    assert install_inventory(root)['install_inventory_sha256'] != first['install_inventory_sha256']


def test_collect_overlay_provenance_records_both_identities(tmp_path):
    repo = tmp_path / 'worktree'
    repo.mkdir()
    install = tmp_path / 'install'
    install.mkdir()
    (install / 'setup.bash').write_text('source x\n', encoding='utf-8')

    record = collect_overlay_provenance(
        repo, install, runtime_commit='b6500c8', validator_commit='1e58825'
    )
    assert record['source_worktree'] == str(repo.resolve())
    assert record['source_commit'] == 'b6500c8'
    assert record['validator_source_commit'] == '1e58825'
    assert record['install_root'] == str(install.resolve())
    assert record['install_inventory_sha256']
    assert record['entries']

    compact = summary(record)
    assert 'entries' not in compact
    assert compact['source_commit'] == 'b6500c8'
    assert compact['install_inventory_sha256'] == record['install_inventory_sha256']


def test_missing_install_root_is_not_claimed_present(tmp_path):
    record = collect_overlay_provenance(
        tmp_path, tmp_path / 'missing', runtime_commit='abc123'
    )
    assert record['install_present'] is False
    assert record['install_inventory_sha256'] is None
    assert json.loads(json.dumps(record))['source_commit'] == 'abc123'
