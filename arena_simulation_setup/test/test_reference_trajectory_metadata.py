import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from arena_simulation_setup.reference_trajectory_metadata import (
    EPISODE_METADATA_SCHEMA,
    import_reference_trajectory_metadata,
    read_reference_table,
)


def _write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    strings = [value for row in rows for value in row]
    shared = ''.join(f'<si><t>{value}</t></si>' for value in strings)
    offset = 0
    xml_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column, _ in enumerate(row):
            letter = chr(ord('A') + column)
            cells.append(f'<c r="{letter}{row_index}" t="s"><v>{offset}</v></c>')
            offset += 1
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        archive.writestr(
            'xl/sharedStrings.xml',
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + shared + '</sst>',
        )
        archive.writestr(
            'xl/worksheets/sheet1.xml',
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>',
        )


def test_import_reference_trajectory_metadata(tmp_path):
    workbook = tmp_path / 'reference.xlsx'
    header = [
        'world', 'episode_index', 'trajectory_file', 'n_frames',
        'duration_s', 'length_m', 'pose_column', 'robot_scale',
    ]
    _write_minimal_xlsx(workbook, [
        header,
        ['grscenes_1', '0', 'grscenes_1/trajectories/episode_000000.npz', '31', '1.0', '4.5', 'pose', '0.8'],
        ['grscenes_1', '2', 'grscenes_1/trajectories/episode_000002.npz', '61', '2.0', '8.0', 'pose', '0.8'],
    ])
    worlds = tmp_path / 'worlds'
    for scenario in ('default', 'default_2'):
        path = worlds / 'grscenes_1_v1' / 'scenarios' / scenario
        path.mkdir(parents=True)
        (path / 'scenario.yaml').write_text('robots: []\n', encoding='utf-8')

    report = import_reference_trajectory_metadata(workbook, worlds)

    assert report['source_rows'] == 2
    assert report['written_count'] == 2
    assert report['test_split_count'] == 2
    assert report['expected_scenario_count'] == 2
    assert report['unannotated_scenario_count'] == 0
    metadata = json.loads((worlds / 'grscenes_1_v1/scenarios/default_2/episode_metadata.json').read_text())
    assert metadata['schema'] == EPISODE_METADATA_SCHEMA
    assert metadata['episode']['source_episode_index'] == 2
    assert metadata['reference_trajectory']['length_m'] == pytest.approx(8.0)
    assert metadata['provenance']['source_table_row'] == 3
    assert len(metadata['provenance']['source_file_sha256']) == 64
    split = json.loads((worlds.parent / 'test_split.json').read_text())
    assert split == [
        'worlds/grscenes_1_v1/scenarios/default/episode_metadata.json',
        'worlds/grscenes_1_v1/scenarios/default_2/episode_metadata.json',
    ]


def test_reference_table_requires_expected_columns(tmp_path):
    workbook = tmp_path / 'reference.xlsx'
    _write_minimal_xlsx(workbook, [['world'], ['grscenes_1']])

    with pytest.raises(ValueError, match='missing columns'):
        read_reference_table(workbook)
