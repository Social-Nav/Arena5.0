"""Validate and enumerate the versioned GRScenes full-eval test split.

The split intentionally contains only POSIX relative paths.  Each path points
to the per-episode metadata beside an Arena ``scenario.yaml``; the world and
scenario identities are therefore derived from the path rather than duplicated
inside the split file.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

# The module name is intentionally ``test_split`` because it is the public
# benchmark concept, not a pytest module.
__test__ = False


WORLD_PATTERN = re.compile(r"grscenes_[1-9][0-9]*_v1")
SCENARIO_PATTERN = re.compile(r"default(?:_[1-9][0-9]*)?")
SPLIT_BASENAME = "test_split.json"
EPISODE_METADATA_BASENAME = "episode_metadata.json"


class TestSplitError(ValueError):
    """Raised when a split is malformed or points outside the package."""


@dataclass(frozen=True)
class TestSplitCase:
    relative_path: str
    world: str
    scenario: str


def _parse_relative_path(value: Any) -> TestSplitCase:
    if not isinstance(value, str) or not value.strip():
        raise TestSplitError(f"split entries must be non-empty strings, got {value!r}")
    text = value
    if text != text.strip():
        raise TestSplitError(f"split path must be normalized and relative: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or text != path.as_posix():
        raise TestSplitError(f"split path must be normalized and relative: {text!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise TestSplitError(f"split path may not contain empty/dot/traversal segments: {text!r}")
    if len(path.parts) != 5:
        raise TestSplitError(
            "split path must be worlds/<world>/scenarios/<scenario>/episode_metadata.json: "
            f"{text!r}"
        )
    worlds_key, world, scenarios_key, scenario, filename = path.parts
    if worlds_key != "worlds" or scenarios_key != "scenarios" or filename != EPISODE_METADATA_BASENAME:
        raise TestSplitError(
            "split path must be worlds/<world>/scenarios/<scenario>/episode_metadata.json: "
            f"{text!r}"
        )
    if not WORLD_PATTERN.fullmatch(world):
        raise TestSplitError(f"unsupported GRScenes world in split: {world!r}")
    if not SCENARIO_PATTERN.fullmatch(scenario):
        raise TestSplitError(f"unsupported GRScenes scenario in split: {scenario!r}")
    return TestSplitCase(relative_path=text, world=world, scenario=scenario)


def load_test_split(
    split_path: str | Path,
    package_root: str | Path,
    *,
    require_files: bool = True,
) -> list[TestSplitCase]:
    split = Path(split_path).expanduser().resolve()
    root = Path(package_root).expanduser().resolve()
    try:
        payload = json.loads(split.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TestSplitError(f"test split not found: {split}") from exc
    except json.JSONDecodeError as exc:
        raise TestSplitError(f"test split is not valid JSON: {split}: {exc}") from exc
    if not isinstance(payload, list):
        raise TestSplitError("test_split.json must be a JSON array of relative paths")

    cases: list[TestSplitCase] = []
    seen_paths: set[str] = set()
    seen_cases: set[tuple[str, str]] = set()
    for raw in payload:
        case = _parse_relative_path(raw)
        if case.relative_path in seen_paths:
            raise TestSplitError(f"duplicate split path: {case.relative_path}")
        identity = (case.world, case.scenario)
        if identity in seen_cases:
            raise TestSplitError(f"duplicate world/scenario in split: {case.world}/{case.scenario}")
        seen_paths.add(case.relative_path)
        seen_cases.add(identity)
        if require_files:
            metadata_path = root.joinpath(*PurePosixPath(case.relative_path).parts)
            scenario_path = metadata_path.with_name("scenario.yaml")
            if not metadata_path.is_file():
                raise TestSplitError(f"split episode metadata does not exist: {metadata_path}")
            if not scenario_path.is_file():
                raise TestSplitError(f"split scenario does not exist: {scenario_path}")
        cases.append(case)
    if not cases:
        raise TestSplitError("test split must contain at least one episode path")
    return cases


def discover_grscenes_cases(package_root: str | Path) -> list[TestSplitCase]:
    root = Path(package_root).expanduser().resolve()
    cases = []
    for scenario_path in root.glob("worlds/grscenes_*_v1/scenarios/*/scenario.yaml"):
        world = scenario_path.parents[2].name
        scenario = scenario_path.parent.name
        if not WORLD_PATTERN.fullmatch(world) or not SCENARIO_PATTERN.fullmatch(scenario):
            continue
        relative = scenario_path.with_name(EPISODE_METADATA_BASENAME).relative_to(root).as_posix()
        cases.append(TestSplitCase(relative_path=relative, world=world, scenario=scenario))
    def key(case: TestSplitCase) -> tuple[int, int]:
        world_index = int(case.world.removeprefix("grscenes_").removesuffix("_v1"))
        scenario_index = 0 if case.scenario == "default" else int(case.scenario.removeprefix("default_"))
        return world_index, scenario_index

    return sorted(cases, key=key)


def selection_report(split_path: str | Path, package_root: str | Path) -> dict[str, Any]:
    selected = load_test_split(split_path, package_root)
    discovered = discover_grscenes_cases(package_root)
    selected_identities = {(case.world, case.scenario) for case in selected}
    discovered_identities = {(case.world, case.scenario) for case in discovered}
    unknown = selected_identities - discovered_identities
    if unknown:
        raise TestSplitError(f"split contains undiscoverable cases: {sorted(unknown)}")
    skipped = [case for case in discovered if (case.world, case.scenario) not in selected_identities]
    return {
        "split_path": str(Path(split_path).expanduser().resolve()),
        "package_root": str(Path(package_root).expanduser().resolve()),
        "selected_count": len(selected),
        "discovered_count": len(discovered),
        "skipped_count": len(skipped),
        "selected": [asdict(case) for case in selected],
        "skipped": [asdict(case) for case in skipped],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate/list the Arena GRScenes full-eval test split")
    parser.add_argument("--split", required=True, help="Path to test_split.json")
    parser.add_argument("--package-root", required=True, help="arena_simulation_setup package root")
    parser.add_argument("--format", choices=("json", "tsv", "summary"), default="json")
    args = parser.parse_args(argv)
    try:
        report = selection_report(args.split, args.package_root)
    except TestSplitError as exc:
        parser.error(str(exc))

    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.format == "summary":
        print(
            f"selected={report['selected_count']} "
            f"discovered={report['discovered_count']} skipped={report['skipped_count']}"
        )
    else:
        for case in report["selected"]:
            print(f"{case['world']}\t{case['scenario']}\t{case['relative_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
