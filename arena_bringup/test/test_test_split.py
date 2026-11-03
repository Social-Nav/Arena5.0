import json
from pathlib import Path

import pytest

from arena_bringup.test_split import (
    TestSplitError as SplitError,
    discover_grscenes_cases,
    load_test_split,
    selection_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_SETUP_ROOT = REPO_ROOT / 'arena_simulation_setup'
CHECKED_IN_SPLIT = SIMULATION_SETUP_ROOT / 'test_split.json'


def _add_case(root: Path, world: str, scenario: str) -> str:
    scenario_dir = root / 'worlds' / world / 'scenarios' / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / 'scenario.yaml').write_text('robots: []\n', encoding='utf-8')
    (scenario_dir / 'episode_metadata.json').write_text('{}\n', encoding='utf-8')
    return f'worlds/{world}/scenarios/{scenario}/episode_metadata.json'


def _write_split(path: Path, payload) -> None:
    path.write_text(json.dumps(payload) + '\n', encoding='utf-8')


def test_checked_in_split_selects_all_500_eval_episodes_without_skips():
    payload = json.loads(CHECKED_IN_SPLIT.read_text(encoding='utf-8'))
    report = selection_report(CHECKED_IN_SPLIT, SIMULATION_SETUP_ROOT)

    assert isinstance(payload, list)
    assert len(payload) == 500
    assert all(isinstance(path, str) and not Path(path).is_absolute() for path in payload)
    assert report['selected_count'] == 500
    assert report['discovered_count'] == 500
    assert report['skipped_count'] == 0

    identities = {(case['world'], case['scenario']) for case in report['selected']}
    # Frozen eval500 naming: no _v1 suffix on worlds, eval_N scenarios.
    assert all(world.startswith('grscenes_') and not world.endswith('_v1')
               for world, _ in identities)
    assert all(scenario.startswith('eval_') for _, scenario in identities)
    # Spot-check a representative episode and the documented missing world.
    assert ('grscenes_9', 'eval_4') in identities
    assert not any(world == 'grscenes_26' for world, _ in identities)
    assert len({world for world, _ in identities}) == 29


def test_split_preserves_file_order_and_derives_case_identity(tmp_path):
    first = _add_case(tmp_path, 'grscenes_2', 'eval_2')
    second = _add_case(tmp_path, 'grscenes_1', 'eval_1')
    split = tmp_path / 'test_split.json'
    _write_split(split, [first, second])

    cases = load_test_split(split, tmp_path)

    assert [(case.world, case.scenario) for case in cases] == [
        ('grscenes_2', 'eval_2'),
        ('grscenes_1', 'eval_1'),
    ]


@pytest.mark.parametrize(
    'world,scenario',
    [
        ('grscenes_1', 'eval_1'),
        ('grscenes_30', 'eval_18'),
        # Legacy versioned/default naming remains parseable for archived datasets.
        ('grscenes_1_v1', 'default'),
        ('grscenes_20_v1', 'default_2'),
    ],
)
def test_split_accepts_eval500_and_legacy_naming_shapes(tmp_path, world, scenario):
    relative = _add_case(tmp_path, world, scenario)
    split = tmp_path / 'test_split.json'
    _write_split(split, [relative])

    cases = load_test_split(split, tmp_path)

    assert [(c.world, c.scenario) for c in cases] == [(world, scenario)]


def test_discovery_orders_eval_episodes_by_world_then_scenario(tmp_path):
    for world, scenario in [
        ('grscenes_1', 'eval_2'),
        ('grscenes_1', 'eval_1'),
        ('grscenes_10', 'eval_1'),
        ('grscenes_2', 'eval_1'),
    ]:
        _add_case(tmp_path, world, scenario)
    # Non-matching directories must be ignored by discovery.
    junk = tmp_path / 'worlds' / 'grscenes_1' / 'scenarios' / 'notes'
    junk.mkdir(parents=True)
    (junk / 'scenario.yaml').write_text('robots: []\n', encoding='utf-8')

    discovered = [(case.world, case.scenario) for case in discover_grscenes_cases(tmp_path)]

    assert discovered == [
        ('grscenes_1', 'eval_1'),
        ('grscenes_1', 'eval_2'),
        ('grscenes_2', 'eval_1'),
        ('grscenes_10', 'eval_1'),
    ]


@pytest.mark.parametrize(
    'payload,error',
    [
        ({'cases': []}, 'JSON array'),
        ([], 'at least one'),
        ([3], 'non-empty strings'),
        ([' worlds/grscenes_1/scenarios/eval_1/episode_metadata.json'], 'normalized and relative'),
        (['/worlds/grscenes_1/scenarios/eval_1/episode_metadata.json'], 'normalized and relative'),
        (['worlds/grscenes_1/scenarios/../eval_1/episode_metadata.json'], 'traversal'),
        (['worlds/not_grscenes/scenarios/eval_1/episode_metadata.json'], 'unsupported GRScenes world'),
        (['worlds/grscenes_0/scenarios/eval_1/episode_metadata.json'], 'unsupported GRScenes world'),
        (['worlds/grscenes_1/scenarios/case_1/episode_metadata.json'], 'unsupported GRScenes scenario'),
        (['worlds/grscenes_1/scenarios/eval_0/episode_metadata.json'], 'unsupported GRScenes scenario'),
    ],
)
def test_split_rejects_malformed_payloads(tmp_path, payload, error):
    split = tmp_path / 'test_split.json'
    _write_split(split, payload)

    with pytest.raises(SplitError, match=error):
        load_test_split(split, tmp_path, require_files=False)


def test_split_rejects_duplicate_paths(tmp_path):
    relative = _add_case(tmp_path, 'grscenes_1', 'eval_1')
    split = tmp_path / 'test_split.json'
    _write_split(split, [relative, relative])

    with pytest.raises(SplitError, match='duplicate split path'):
        load_test_split(split, tmp_path)


@pytest.mark.parametrize(
    'missing_name,error',
    [
        ('episode_metadata.json', 'episode metadata does not exist'),
        ('scenario.yaml', 'split scenario does not exist'),
    ],
)
def test_split_requires_both_metadata_and_scenario_files(tmp_path, missing_name, error):
    relative = _add_case(tmp_path, 'grscenes_1', 'eval_1')
    (tmp_path / 'worlds/grscenes_1/scenarios/eval_1' / missing_name).unlink()
    split = tmp_path / 'test_split.json'
    _write_split(split, [relative])

    with pytest.raises(SplitError, match=error):
        load_test_split(split, tmp_path)
