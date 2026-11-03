"""Auditable overlay provenance for the unified VLN runner (no ROS imports).

The unified runner may execute code from a source worktree that is different
from the long-lived base runtime checkout.  This module records both identities
side by side: the base runtime keeps its own ``provenance`` block, while the
overlay is described here with an exact source commit and a content hash of the
install tree that was actually sourced.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

OVERLAY_PROVENANCE_SCHEMA = 'arena.vln.overlay_provenance.v1'
OVERLAY_PROVENANCE_FILE = 'overlay_provenance.json'


def _git(repo, *args):
    try:
        result = subprocess.run(
            ['git', '-C', str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def install_root_from_environment(default='/opt/arena_ws/install'):
    """Return the overlay install root that is actually on the environment."""
    explicit = str(os.environ.get('ARENA_VLN_OVERLAY_INSTALL_ROOT', '') or '').strip()
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    for entry in str(os.environ.get('AMENT_PREFIX_PATH', '') or '').split(':'):
        candidate = entry.strip()
        if candidate and 'install' in Path(candidate).name:
            return str(Path(candidate).resolve())
    return str(Path(default).resolve())


def install_inventory(root):
    """Deterministic inventory of an install tree.

    Symlink installs keep the payload in the source worktree, so recording the
    link targets plus the hashes of the small generated files is enough to pin
    the tree without copying or re-hashing model weights.
    """
    root = Path(root)
    entries = []
    if not root.is_dir():
        return {
            'install_root': str(root),
            'install_present': False,
            'install_entry_count': 0,
            'install_file_count': 0,
            'install_symlink_count': 0,
            'install_inventory_sha256': None,
            'entries': [],
        }
    for path in sorted(root.rglob('*')):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            entries.append({'path': relative, 'type': 'symlink', 'target': os.readlink(path)})
        elif path.is_dir():
            entries.append({'path': relative, 'type': 'dir'})
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            entries.append({
                'path': relative,
                'type': 'file',
                'size': path.stat().st_size,
                'sha256': digest.hexdigest(),
            })
        else:
            entries.append({'path': relative, 'type': 'other'})
    canonical = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {
        'install_root': str(root),
        'install_present': True,
        'install_entry_count': len(entries),
        'install_file_count': sum(1 for entry in entries if entry['type'] == 'file'),
        'install_symlink_count': sum(1 for entry in entries if entry['type'] == 'symlink'),
        'install_inventory_sha256': hashlib.sha256(canonical).hexdigest(),
        'entries': entries,
    }


def collect_overlay_provenance(
    source_worktree,
    install_root=None,
    *,
    runtime_commit=None,
    validator_commit=None,
    extra=None,
):
    """Record overlay code identity without touching the base runtime record."""
    worktree = str(Path(source_worktree).expanduser().resolve())
    inventory = install_inventory(install_root) if install_root else install_inventory('')
    return {
        'schema': OVERLAY_PROVENANCE_SCHEMA,
        'source_worktree': worktree,
        'source_branch': _git(worktree, 'branch', '--show-current'),
        'source_commit': runtime_commit or _git(worktree, 'rev-parse', 'HEAD'),
        'source_dirty': _git(worktree, 'status', '--porcelain', '--untracked-files=no') not in (None, ''),
        'validator_source_commit': validator_commit or _git(worktree, 'rev-parse', 'HEAD'),
        'install_root': inventory.get('install_root'),
        'install_present': inventory.get('install_present'),
        'install_entry_count': inventory.get('install_entry_count'),
        'install_file_count': inventory.get('install_file_count'),
        'install_symlink_count': inventory.get('install_symlink_count'),
        'install_inventory_sha256': inventory.get('install_inventory_sha256'),
        'entries': inventory.get('entries', []),
        'collected_at': datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }


def summary(record):
    """Manifest-sized view: identity and hashes, no per-file entry list."""
    if not isinstance(record, dict):
        return {}
    keys = (
        'schema', 'source_worktree', 'source_branch', 'source_commit', 'source_dirty',
        'validator_source_commit', 'install_root', 'install_present', 'install_entry_count',
        'install_file_count', 'install_symlink_count', 'install_inventory_sha256',
        'collected_at',
    )
    return {key: record.get(key) for key in keys if key in record}
