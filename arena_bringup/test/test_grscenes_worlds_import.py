import binascii
import importlib.util
import json
import math
import shutil
import struct
import sys
import zlib
from pathlib import Path, PurePosixPath

import pytest
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "arena_bringup" / "grscenes_worlds_import.py"
SPEC = importlib.util.spec_from_file_location("grscenes_worlds_import_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)

IMPORT_ID = "worlds_aaaaaaaa"
ARCHIVE_SHA256 = "a" * 64


def _png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def _png(width=10, height=10):
    rows = b"".join(b"\x00" + b"\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(rows)) + _png_chunk(b"IEND", b"")


def _scenario(world_index, scenario_name):
    return {
        "dynamic": [
            {
                "name": "hunav_1",
                "model": "female_adult_business_02",
                "pose": [3.0, 3.0, 2.0],
                "behavior_tree": "BTRegularNav.xml",
                "velocity": 0.8,
                "desired_velocity": 1.0,
                "waypoints": [[4.0, 4.0, 45.0]],
            }
        ],
        "robot": {
            "name": "robot",
            "pose": [1.0, 1.0, 1.0],
            "waypoints": [[2.0, 2.0, -2.0]],
            "behavior": f"navigate world {world_index} case {scenario_name}",
        },
    }


def _make_input(tmp_path, mutate=None):
    worlds = tmp_path / "input_worlds"
    for world_index in range(1, 31):
        world_name = f"grscenes_{world_index}"
        world_dir = worlds / world_name
        (world_dir / "map").mkdir(parents=True)
        (world_dir / "world.yaml").write_text(
            yaml.safe_dump(
                {
                    "world_type": "usd",
                    "usd_scene": {
                        "path": f"/data/scenes/{world_name}/start_result_navigation.usd",
                        "scale": 0.01,
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                    },
                    "metadata": {"source": "fixture"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (world_dir / "map" / "map.yaml").write_text(
            yaml.safe_dump(
                {
                    "image": "map.png",
                    "resolution": 1.0,
                    "origin": [0.0, 0.0, 0.0],
                    "occupied_thresh": 0.65,
                    "free_thresh": 0.196,
                    "negate": 0,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (world_dir / "map" / "map.png").write_bytes(_png())
        for scenario_name in IMPORTER.SCENARIO_NAMES:
            scenario_dir = world_dir / "scenarios" / scenario_name
            scenario_dir.mkdir(parents=True)
            scenario = _scenario(world_index, scenario_name)
            if mutate is not None:
                mutate(world_index, scenario_name, scenario)
            (scenario_dir / "scenario.yaml").write_text(
                yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    return worlds


def _make_bt(tmp_path):
    path = tmp_path / "BTRegularNav.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<root main_tree_to_execute="RegularNavTree">\n'
        '  <BehaviorTree ID="RegularNavTree"/>\n'
        '</root>\n',
        encoding="utf-8",
    )
    return path


def _run(worlds, output_root, bt, **overrides):
    arguments = {
        "input_worlds": worlds,
        "output_root": output_root,
        "output_worlds": "worlds",
        "output_catalog": "catalog/cases.json",
        "output_report": "reports/import_report.json",
        "output_manifest": "import_manifest.json",
        "import_id": IMPORT_ID,
        "source_archive_sha256": ARCHIVE_SHA256,
        "bt_template": bt,
    }
    arguments.update(overrides)
    return IMPORTER.materialize(**arguments)


def _tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _catalog(root):
    return json.loads((root / "catalog" / "cases.json").read_text(encoding="utf-8"))


def _case(catalog, case_id):
    return next(case for case in catalog["cases"] if case["identity"]["id"] == case_id)


def test_deterministic_fresh_materializations_and_no_overwrite(tmp_path):
    worlds = _make_input(tmp_path)
    bt = _make_bt(tmp_path)
    output_a = tmp_path / "preview_a"
    output_b = tmp_path / "preview_b"

    result_a = _run(worlds, output_a, bt)
    result_b = _run(worlds, output_b, bt)
    tree_a = _tree_bytes(output_a)
    tree_b = _tree_bytes(output_b)

    assert tree_a == tree_b
    assert result_a["manifest_tree_sha256"] == result_b["manifest_tree_sha256"]
    assert result_a["file_count"] == len(tree_a)
    manifest = json.loads(tree_a["import_manifest.json"])
    assert len(manifest["files"]) == len(tree_a)
    manifest_entry = next(entry for entry in manifest["files"] if entry["path"] == "import_manifest.json")
    assert manifest_entry["sha256"] is None
    assert manifest_entry["size_bytes"] == len(tree_a["import_manifest.json"])

    before = dict(tree_a)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(worlds, output_a, bt)
    assert _tree_bytes(output_a) == before


def test_normalizes_robot_humans_waypoints_behavior_and_bt(tmp_path):
    raw_behavior = "Carry café  supplies exactly"

    def mutate(world_index, scenario_name, scenario):
        if (world_index, scenario_name) == (1, "default"):
            scenario["robot"]["pose"] = [1.0, 1.0, 1.0]
            scenario["robot"]["waypoints"] = [[1.5, 1.5, 0.5], [2.0, 2.0, -2.0]]
            scenario["robot"]["behavior"] = raw_behavior
            scenario["dynamic"][0]["pose"] = [3.0, 3.0, 2.0]
            scenario["dynamic"][0]["waypoints"] = [[4.0, 4.0, 45.0]]
            scenario["dynamic"][0]["is_cyclic"] = False

    worlds = _make_input(tmp_path, mutate)
    bt = _make_bt(tmp_path)
    output = tmp_path / "preview"
    _run(worlds, output, bt)

    scenario_dir = output / "worlds" / "grscenes_1_v1" / "scenarios" / "default"
    native = yaml.safe_load((scenario_dir / "scenario.yaml").read_text(encoding="utf-8"))
    assert "robot" not in native
    assert native["robots"] == [
        {
            "start": [1.0, 1.0, round(math.radians(1.0), 12)],
            "goal": [2.0, 2.0, round(math.radians(-2.0), 12)],
        }
    ]
    human = native["dynamic"][0]
    assert human["pose"] == [3.0, 3.0, round(math.radians(2.0), 12)]
    assert human["waypoints"] == [[4.0, 4.0, 0.0]]
    assert human["behavior_tree"] == "./BTRegularNav.xml"
    assert human["model"] == "female_adult_business_02"
    assert human["velocity"] == 0.8
    assert human["desired_velocity"] == 1.0
    assert human["is_cyclic"] is False
    assert (scenario_dir / "BTRegularNav.xml").read_bytes() == bt.read_bytes()

    case = _case(_catalog(output), f"{IMPORT_ID}:grscenes_1:default")
    assert case["route"]["raw_waypoints_xy_yaw_deg"] == [[1.5, 1.5, 0.5], [2.0, 2.0, -2.0]]
    assert case["humans"]["agents"][0]["raw_waypoints_xy_heading_deg"] == [[4.0, 4.0, 45.0]]
    assert case["humans"]["agents"][0]["emitted_waypoints_xyz_m"] == [[4.0, 4.0, 0.0]]
    assert case["language"]["raw_behavior_text"] == raw_behavior
    assert case["language"]["derived_instruction"] == {
        "derived": True,
        "equivalence_claimed": False,
        "review_status": "pending",
        "source_field": "robot.behavior",
        "text": raw_behavior,
    }


def test_blocks_no_human_and_out_of_map_without_rewriting_route(tmp_path):
    def mutate(world_index, scenario_name, scenario):
        if (world_index, scenario_name) == (1, "default"):
            scenario.pop("dynamic")
        if (world_index, scenario_name) == (2, "default"):
            scenario["robot"]["pose"][0] = 11.0

    worlds = _make_input(tmp_path, mutate)
    bt = _make_bt(tmp_path)
    output = tmp_path / "preview"
    result = _run(worlds, output, bt)
    catalog = _catalog(output)

    no_human = _case(catalog, f"{IMPORT_ID}:grscenes_1:default")
    outside = _case(catalog, f"{IMPORT_ID}:grscenes_2:default")
    assert no_human["admission"]["block_reasons"] == ["no_active_humans_for_social_case"]
    assert outside["admission"]["block_reasons"] == ["robot_start_out_of_map_bounds"]
    assert outside["route"]["normalized_start_xy_yaw_rad"][0] == 11.0
    assert result["blocked_case_count"] == 2
    no_human_dir = output / "worlds" / "grscenes_1_v1" / "scenarios" / "default"
    assert not (no_human_dir / "BTRegularNav.xml").exists()


@pytest.mark.parametrize("failure_kind", ["malformed_pose", "nonfinite", "missing_image", "malformed_world"])
def test_fails_closed_on_malformed_nonfinite_and_missing_input(tmp_path, failure_kind):
    def mutate(world_index, scenario_name, scenario):
        if (world_index, scenario_name) != (1, "default"):
            return
        if failure_kind == "malformed_pose":
            scenario["robot"]["pose"] = [1.0, 2.0]
        elif failure_kind == "nonfinite":
            scenario["dynamic"][0]["velocity"] = float("nan")

    worlds = _make_input(tmp_path, mutate)
    if failure_kind == "missing_image":
        (worlds / "grscenes_1" / "map" / "map.png").unlink()
    elif failure_kind == "malformed_world":
        (worlds / "grscenes_1" / "world.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    bt = _make_bt(tmp_path)
    output = tmp_path / "preview"

    with pytest.raises(IMPORTER.ImportValidationError):
        _run(worlds, output, bt)
    assert not output.exists()


def test_catalog_identity_hash_provenance_and_no_fabricated_record_metadata(tmp_path):
    worlds = _make_input(tmp_path)
    bt = _make_bt(tmp_path)
    output = tmp_path / "preview"
    _run(worlds, output, bt)
    catalog = _catalog(output)

    assert catalog["runtime_dependency"] is False
    assert catalog["statistics"]["world_count"] == 30
    assert catalog["statistics"]["case_count"] == 150
    assert len({case["identity"]["id"] for case in catalog["cases"]}) == 150
    assert all(case["identity"]["kind"] == "scenario_case" for case in catalog["cases"])
    assert all(case["identity"]["episode"] is None for case in catalog["cases"])
    assert all(case["identity"]["timestamp"] is None for case in catalog["cases"])
    case = _case(catalog, f"{IMPORT_ID}:grscenes_1:default")
    source_bytes = (worlds / "grscenes_1" / "scenarios" / "default" / "scenario.yaml").read_bytes()
    generated_path = output / Path(case["native_scenario"]["generated_path"])
    assert case["source"]["scenario_sha256"] == IMPORTER.sha256_bytes(source_bytes)
    assert case["native_scenario"]["generated_sha256"] == IMPORTER.sha256_bytes(generated_path.read_bytes())
    assert case["source"]["archive_sha256"] == ARCHIVE_SHA256

    asset = catalog["assets"][0]
    assert asset["source"]["world_yaml"]["sha256"] == asset["generated"]["world_yaml"]["sha256"]
    allowlist = catalog["future_promotion_allowlist"]
    assert allowlist["status"] == "generated_artifact_backed_not_promoted"
    assert allowlist["path_count"] == 61
    assert len(allowlist["source_world_roots"]) == 30
    assert len(allowlist["installed_share_world_roots"]) == 30
    assert len(allowlist["offline_catalog_destinations"]) == 1


def test_promotion_allowlist_is_manifest_derived_and_excludes_deferred_configs(tmp_path):
    worlds = _make_input(tmp_path)
    bt = _make_bt(tmp_path)
    output = tmp_path / "preview"
    _run(worlds, output, bt)
    catalog = _catalog(output)
    manifest = json.loads((output / "import_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["files"]}

    allowlist = catalog["future_promotion_allowlist"]
    source_roots = allowlist["source_world_roots"]
    installed_roots = allowlist["installed_share_world_roots"]
    catalog_destinations = allowlist["offline_catalog_destinations"]
    all_allowed_paths = source_roots + installed_roots + catalog_destinations

    assert allowlist["path_count"] == 61 == len(all_allowed_paths) == len(set(all_allowed_paths))
    for source_root in source_roots:
        variant_id = PurePosixPath(source_root).name
        assert any(path.startswith(f"worlds/{variant_id}/") for path in manifest_paths)
    assert installed_roots == [f"share/{source_root}" for source_root in source_roots]
    assert catalog_destinations == [f"arena_bringup/configs/grscenes_benchmark/{IMPORT_ID}_catalog.json"]

    evidence = catalog["promotion_artifact_evidence"]
    assert evidence["generated_world_roots"] == [
        f"worlds/{PurePosixPath(source_root).name}" for source_root in source_roots
    ]
    assert evidence["generated_offline_catalog"] == "catalog/cases.json"
    assert evidence["generated_offline_catalog"] in manifest_paths

    deferred = catalog["deferred_generation_classes"]["benchmark_case_configs"]
    assert deferred == {
        "allowlisted_path_count": 0,
        "generated_artifact_count": 0,
        "status": "deferred_pending_exactly_three_selection_and_human_instruction_review",
    }
    governance_text = json.dumps(
        {
            "future_promotion_allowlist": allowlist,
            "deferred_generation_classes": catalog["deferred_generation_classes"],
        },
        sort_keys=True,
    )
    assert "benchmark_case_config_files" not in governance_text
    assert "grscenes_benchmark/cases/" not in governance_text


def test_rejects_world_name_and_path_traversal(tmp_path):
    worlds = _make_input(tmp_path)
    bt = _make_bt(tmp_path)
    extra = worlds / "grscenes_evil"
    extra.mkdir()

    with pytest.raises(IMPORTER.ImportValidationError, match="exactly grscenes_1..30"):
        _run(worlds, tmp_path / "preview", bt)
    extra.rmdir()

    map_yaml_path = worlds / "grscenes_1" / "map" / "map.yaml"
    map_yaml = yaml.safe_load(map_yaml_path.read_text(encoding="utf-8"))
    map_yaml["image"] = "../escape.png"
    map_yaml_path.write_text(yaml.safe_dump(map_yaml, sort_keys=False), encoding="utf-8")
    with pytest.raises(IMPORTER.ImportValidationError, match="unsafe path"):
        _run(worlds, tmp_path / "preview", bt)

    with pytest.raises(IMPORTER.ImportValidationError, match="traversal"):
        _run(worlds, tmp_path / "preview", bt, output_worlds="../worlds")
    assert not (tmp_path / "preview").exists()


def test_source_mutation_changes_source_generated_and_tree_hashes(tmp_path):
    worlds_a = _make_input(tmp_path / "a")
    worlds_b = tmp_path / "b" / "input_worlds"
    worlds_b.parent.mkdir(parents=True)
    shutil.copytree(worlds_a, worlds_b)
    target_b = worlds_b / "grscenes_1" / "scenarios" / "default" / "scenario.yaml"
    mutated = yaml.safe_load(target_b.read_text(encoding="utf-8"))
    mutated["robot"]["waypoints"][-1][0] = 2.25
    target_b.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")
    bt = _make_bt(tmp_path)

    result_a = _run(worlds_a, tmp_path / "preview_a", bt)
    result_b = _run(worlds_b, tmp_path / "preview_b", bt)
    case_a = _case(_catalog(tmp_path / "preview_a"), f"{IMPORT_ID}:grscenes_1:default")
    case_b = _case(_catalog(tmp_path / "preview_b"), f"{IMPORT_ID}:grscenes_1:default")

    assert case_a["source"]["scenario_sha256"] != case_b["source"]["scenario_sha256"]
    assert case_a["native_scenario"]["generated_sha256"] != case_b["native_scenario"]["generated_sha256"]
    assert result_a["manifest_tree_sha256"] != result_b["manifest_tree_sha256"]
    assert case_b["route"]["normalized_goal_xy_yaw_rad"][0] == 2.25


def test_cli_requires_explicit_paths_and_generates_catalog(tmp_path, capsys):
    worlds = _make_input(tmp_path)
    bt = _make_bt(tmp_path)
    output = tmp_path / "preview"

    assert (
        IMPORTER.main(
            [
                "--input-worlds",
                str(worlds),
                "--output-root",
                str(output),
                "--output-worlds",
                "worlds",
                "--output-config-catalog",
                "catalog/cases.json",
                "--output-report",
                "reports/import_report.json",
                "--output-manifest",
                "import_manifest.json",
                "--import-id",
                IMPORT_ID,
                "--source-archive-sha256",
                ARCHIVE_SHA256,
                "--bt-template",
                str(bt),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["world_count"] == 30
    assert (output / "catalog" / "cases.json").is_file()
