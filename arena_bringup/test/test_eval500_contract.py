"""Additive eval500 contract/profile-version checks for the clean integration branch.

This test is intentionally additive: it does not overwrite the old frozen
common contract or the old pinned method bundles.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from arena_bringup import paper_pipeline as pp
from arena_bringup import test_split


ROOT = Path(__file__).parents[2]
METHODS = ROOT / 'configs' / 'agentic_eval' / 'methods'


def _load(path):
    return json.loads(path.read_text())


def test_eval500_common_contract_is_additive_and_valid():
    old = _load(METHODS / 'common_contract.v1.json')
    new = _load(METHODS / 'common_contract_eval500_v1.json')
    assert old['case'] == {
        'world': 'grscenes_20_v1', 'scenario': 'default_2',
        'robot': 'Ai2_Bot2'}
    assert new['case'] == {
        'world': 'grscenes_1', 'scenario': 'eval_1', 'robot': 'Ai2_Bot2'}
    validation = pp.validate_common_contract(new)
    assert validation['valid'] is True
    assert validation['contract_hash'].startswith('sha256:')
    assert new['contract_id'] != old['contract_id']


def test_eval500_profile_version_matches_current_and_pinned_source():
    record = _load(METHODS / 'eval500_profile_version.v1.json')
    schema = _load(METHODS / 'profile_version.v1.schema.json')
    jsonschema = pytest.importorskip('jsonschema')
    jsonschema.validate(record, schema)

    profile = ROOT / record['profile_path']
    current_sha = hashlib.sha256(profile.read_bytes()).hexdigest()
    assert current_sha == record['profile_sha256']
    pinned = subprocess.run(
        ['git', '-C', str(ROOT), 'show',
         f"{record['source_commit']}:{record['profile_path']}"],
        capture_output=True, check=True)
    assert hashlib.sha256(pinned.stdout).hexdigest() == record['profile_sha256']

    evaluation = yaml.safe_load(profile.read_text(encoding='utf-8'))['evaluation']
    assert evaluation['world'] == record['case']['world']
    assert evaluation['scenario'] == record['case']['scenario']
    assert evaluation['robot'] == record['case']['robot']


def test_eval500_split_selected_and_discovered_are_500():
    report = test_split.selection_report(
        ROOT / 'arena_simulation_setup' / 'test_split.json',
        ROOT / 'arena_simulation_setup')
    assert report['selected_count'] == 500
    assert report['discovered_count'] == 500
    assert report['skipped_count'] == 0


def test_old_bundles_do_not_become_ready_under_eval500_contract():
    old_common = _load(METHODS / 'common_contract.v1.json')
    new_common = _load(METHODS / 'common_contract_eval500_v1.json')
    for method_id in ('internnav', 'navila', 'lightnav'):
        bundle = _load(METHODS / f'{method_id}.v1.json')
        old_plan = pp.build_method_plan(method_id, old_common, bundle)
        new_plan = pp.build_method_plan(method_id, new_common, bundle)
        assert old_plan['executable'] is False
        assert new_plan['executable'] is False
        assert new_plan['commands'] == []
    assert _load(METHODS / 'internnav.v1.json')['profile']['sha256'] == (
        '81b0303057728f00b402a3f202853db713bd23eb5c340da334ff4e184777cbb5')
