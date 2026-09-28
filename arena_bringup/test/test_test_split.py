import json
from pathlib import Path

import pytest

from arena_bringup.test_split import (
    TestSplitError as SplitError,
    load_test_split,
    selection_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_SETUP_ROOT = REPO_ROOT / 'arena_simulation_setup'
CHECKED_IN_SPLIT = SIMULATION_SETUP_ROOT / 'test_split.json'
DENSITY_SPLIT = SIMULATION_SETUP_ROOT / 'density_eval_split_10.json'


def _add_case(root: Path, world: str, scenario: str) -> str:
    scenario_dir = root / 'worlds' / world / 'scenarios' / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / 'scenario.yaml').write_text('robots: []\n', encoding='utf-8')
    (scenario_dir / 'episode_metadata.json').write_text('{}\n', encoding='utf-8')
    return f'worlds/{world}/scenarios/{scenario}/episode_metadata.json'


def _write_split(path: Path, payload) -> None:
    path.write_text(json.dumps(payload) + '\n', encoding='utf-8')


def test_checked_in_split_selects_133_and_skips_the_known_17():
    payload = json.loads(CHECKED_IN_SPLIT.read_text(encoding='utf-8'))
    report = selection_report(CHECKED_IN_SPLIT, SIMULATION_SETUP_ROOT)

    assert isinstance(payload, list)
    assert len(payload) == 133
    assert all(isinstance(path, str) and not Path(path).is_absolute() for path in payload)
    assert report['selected_count'] == 133
    assert report['discovered_count'] == 150
    assert report['skipped_count'] == 17
    assert {f"{case['world']}/{case['scenario']}" for case in report['skipped']} == {
        *(f'grscenes_19_v1/{scenario}' for scenario in ('default', 'default_1', 'default_2', 'default_3', 'default_4')),
        *(f'grscenes_23_v1/{scenario}' for scenario in ('default', 'default_1', 'default_2', 'default_3', 'default_4')),
        'grscenes_26_v1/default_2',
        'grscenes_26_v1/default_3',
        'grscenes_26_v1/default_4',
        'grscenes_27_v1/default_4',
        'grscenes_29_v1/default_4',
        'grscenes_30_v1/default_4',
        'grscenes_4_v1/default_4',
    }


def test_density_split_has_ten_short_cross_world_cases():
    cases = load_test_split(DENSITY_SPLIT, SIMULATION_SETUP_ROOT)
    metadata = [
        json.loads((SIMULATION_SETUP_ROOT / case.relative_path).read_text())
        for case in cases
    ]

    assert len(cases) == 10
    assert len({case.world for case in cases}) == 10
    assert all(
        item['reference_trajectory']['length_m'] <= 5.3
        for item in metadata
    )
    assert all(
        (
            SIMULATION_SETUP_ROOT
            / 'worlds'
            / case.world
            / 'scenarios'
            / case.scenario
            / 'instruction'
            / 'instruction.json'
        ).is_file()
        for case in cases
    )


def test_split_preserves_file_order_and_derives_case_identity(tmp_path):
    first = _add_case(tmp_path, 'grscenes_2_v1', 'default_1')
    second = _add_case(tmp_path, 'grscenes_1_v1', 'default')
    split = tmp_path / 'test_split.json'
    _write_split(split, [first, second])

    cases = load_test_split(split, tmp_path)

    assert [(case.world, case.scenario) for case in cases] == [
        ('grscenes_2_v1', 'default_1'),
        ('grscenes_1_v1', 'default'),
    ]


@pytest.mark.parametrize(
    'payload,error',
    [
        ({'cases': []}, 'JSON array'),
        ([], 'at least one'),
        ([3], 'non-empty strings'),
        ([' worlds/grscenes_1_v1/scenarios/default/episode_metadata.json'], 'normalized and relative'),
        (['/worlds/grscenes_1_v1/scenarios/default/episode_metadata.json'], 'normalized and relative'),
        (['worlds/grscenes_1_v1/scenarios/../default/episode_metadata.json'], 'traversal'),
        (['worlds/not_grscenes/scenarios/default/episode_metadata.json'], 'unsupported GRScenes world'),
        (['worlds/grscenes_1_v1/scenarios/case_1/episode_metadata.json'], 'unsupported GRScenes scenario'),
    ],
)
def test_split_rejects_malformed_payloads(tmp_path, payload, error):
    split = tmp_path / 'test_split.json'
    _write_split(split, payload)

    with pytest.raises(SplitError, match=error):
        load_test_split(split, tmp_path, require_files=False)


def test_split_rejects_duplicate_paths(tmp_path):
    relative = _add_case(tmp_path, 'grscenes_1_v1', 'default')
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
    relative = _add_case(tmp_path, 'grscenes_1_v1', 'default')
    (tmp_path / 'worlds/grscenes_1_v1/scenarios/default' / missing_name).unlink()
    split = tmp_path / 'test_split.json'
    _write_split(split, [relative])

    with pytest.raises(SplitError, match=error):
        load_test_split(split, tmp_path)
