"""Import per-episode reference trajectory metadata from an XLSX table.

The benchmark consumes the generated ``episode_metadata.json`` files directly
from each Arena scenario directory.  Keeping the metric input beside the
scenario avoids a runtime join against an operator-local spreadsheet and makes
the exact source row and source-file digest auditable.

Only the Python standard library is used to read XLSX files.  This keeps the
runtime package independent of openpyxl while still supporting the simple
single-sheet table used by the GRScenes annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile


EPISODE_METADATA_SCHEMA = "arena.grscenes_episode_metadata"
EPISODE_METADATA_SCHEMA_VERSION = 1
EPISODE_METADATA_FILENAME = "episode_metadata.json"
TEST_SPLIT_FILENAME = "test_split.json"
REQUIRED_COLUMNS = (
    "world",
    "episode_index",
    "trajectory_file",
    "n_frames",
    "duration_s",
    "length_m",
    "pose_column",
    "robot_scale",
)
_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.iter(f"{_SHEET_NS}t"))
        for item in root.findall(f"{_SHEET_NS}si")
    ]


def _cell_column(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference.upper())
    if not match:
        raise ValueError(f"invalid XLSX cell reference: {reference!r}")
    result = 0
    for char in match.group(1):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def read_reference_table(path: str | Path) -> list[dict[str, str]]:
    """Read the first worksheet and return non-empty rows as dictionaries."""
    workbook_path = Path(path).expanduser().resolve()
    with ZipFile(workbook_path) as archive:
        shared_strings = _shared_strings(archive)
        worksheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        raw_rows: list[list[str]] = []
        for row in worksheet.findall(f".//{_SHEET_NS}sheetData/{_SHEET_NS}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{_SHEET_NS}c"):
                column = _cell_column(str(cell.attrib.get("r") or ""))
                value_node = cell.find(f"{_SHEET_NS}v")
                value = "" if value_node is None else str(value_node.text or "")
                if cell.attrib.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                values[column] = value
            if values:
                width = max(values) + 1
                raw_rows.append([values.get(index, "") for index in range(width)])

    if not raw_rows:
        raise ValueError(f"reference trajectory table is empty: {workbook_path}")
    header = [value.strip() for value in raw_rows[0]]
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"reference trajectory table is missing columns: {missing}")

    rows = []
    for source_row, values in enumerate(raw_rows[1:], start=2):
        padded = values + [""] * max(0, len(header) - len(values))
        record = {key: padded[index].strip() for index, key in enumerate(header)}
        if any(record.values()):
            record["_source_row"] = str(source_row)
            rows.append(record)
    return rows


def scenario_identity(row: dict[str, str]) -> tuple[str, str, int]:
    source_world = str(row["world"]).strip()
    match = re.fullmatch(r"grscenes_(\d+)", source_world)
    if not match:
        raise ValueError(f"unsupported world value: {source_world!r}")
    episode_index = int(row["episode_index"] or "-1")
    if episode_index < 0:
        raise ValueError(f"invalid episode_index: {row['episode_index']!r}")
    world = f"{source_world}_v1"
    scenario = "default" if episode_index == 0 else f"default_{episode_index}"
    return world, scenario, episode_index


def metadata_from_row(
    row: dict[str, str],
    *,
    source_name: str,
    source_sha256: str,
    source_row_count: int,
) -> dict[str, Any]:
    world, scenario, episode_index = scenario_identity(row)
    length_m = float(row["length_m"])
    duration_s = float(row["duration_s"])
    robot_scale = float(row["robot_scale"])
    n_frames = int(row["n_frames"])
    if not math.isfinite(length_m) or length_m <= 0.0:
        raise ValueError(f"reference length must be finite and positive: {length_m}")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"duration must be finite and positive: {duration_s}")
    if not math.isfinite(robot_scale) or robot_scale <= 0.0:
        raise ValueError(f"robot_scale must be finite and positive: {robot_scale}")
    if n_frames < 2:
        raise ValueError(f"reference trajectory needs at least two frames: {n_frames}")
    return {
        "schema": EPISODE_METADATA_SCHEMA,
        "schema_version": EPISODE_METADATA_SCHEMA_VERSION,
        "episode": {
            "world": world,
            "scenario": scenario,
            "source_world": row["world"],
            "source_episode_index": episode_index,
        },
        "reference_trajectory": {
            "length_m": length_m,
            "n_frames": n_frames,
            "duration_s": duration_s,
            "trajectory_file": row["trajectory_file"],
            "pose_column": row["pose_column"],
            "robot_scale": robot_scale,
        },
        "provenance": {
            "source_type": "provided_reference_trajectory_length_table",
            "source_file": source_name,
            "source_file_sha256": source_sha256,
            "source_table_row": int(row["_source_row"]),
            "source_table_data_rows": source_row_count,
        },
    }


def import_reference_trajectory_metadata(
    workbook: str | Path,
    worlds_dir: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    workbook_path = Path(workbook).expanduser().resolve()
    worlds_path = Path(worlds_dir).expanduser().resolve()
    rows = read_reference_table(workbook_path)
    source_sha256 = hashlib.sha256(workbook_path.read_bytes()).hexdigest()
    written: list[str] = []
    missing_scenarios: list[str] = []
    identities: set[tuple[str, str]] = set()

    for row in rows:
        metadata = metadata_from_row(
            row,
            source_name=workbook_path.name,
            source_sha256=source_sha256,
            source_row_count=len(rows),
        )
        episode = metadata["episode"]
        identity = (episode["world"], episode["scenario"])
        if identity in identities:
            raise ValueError(f"duplicate episode annotation: {identity[0]}/{identity[1]}")
        identities.add(identity)
        scenario_dir = worlds_path / identity[0] / "scenarios" / identity[1]
        if not (scenario_dir / "scenario.yaml").is_file():
            missing_scenarios.append(f"{identity[0]}/{identity[1]}")
            continue
        output_path = scenario_dir / EPISODE_METADATA_FILENAME
        if not dry_run:
            output_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        written.append(str(output_path))

    if missing_scenarios:
        raise ValueError(f"rows did not map to Arena scenarios: {missing_scenarios}")
    split_paths = [
        Path(path).resolve().relative_to(worlds_path.parent).as_posix()
        for path in written
    ]
    split_path = worlds_path.parent / TEST_SPLIT_FILENAME
    if not dry_run:
        split_path.write_text(
            json.dumps(split_paths, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    expected_episode_count = 0
    annotated_identities = set(identities)
    unannotated_scenarios: list[str] = []
    for world_dir in sorted(worlds_path.glob("grscenes_*_v1")):
        for episode_index in range(5):
            scenario = "default" if episode_index == 0 else f"default_{episode_index}"
            scenario_path = world_dir / "scenarios" / scenario / "scenario.yaml"
            if not scenario_path.is_file():
                continue
            expected_episode_count += 1
            if (world_dir.name, scenario) not in annotated_identities:
                unannotated_scenarios.append(f"{world_dir.name}/{scenario}")
    return {
        "source_file": str(workbook_path),
        "source_sha256": source_sha256,
        "source_rows": len(rows),
        "written_count": len(written),
        "test_split_path": str(split_path),
        "test_split_count": len(split_paths),
        "expected_scenario_count": expected_episode_count,
        "unannotated_scenario_count": len(unannotated_scenarios),
        "unannotated_scenarios": unannotated_scenarios,
        "dry_run": dry_run,
        "outputs": written,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import GRScenes reference trajectory metadata from XLSX")
    parser.add_argument("workbook", help="Input XLSX table")
    parser.add_argument("--worlds-dir", required=True, help="arena_simulation_setup/worlds directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = import_reference_trajectory_metadata(args.workbook, args.worlds_dir, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
