"""Deterministically normalize the immutable GRScenes worlds archive.

This module is an offline import tool.  Runtime code consumes only the generated
native world/scenario files; it does not read the generated catalog.
"""

from __future__ import annotations

import argparse
import binascii
import copy
import hashlib
import json
import math
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


CONVERSION_VERSION = 2
WORLD_NAMES = tuple(f"grscenes_{index}" for index in range(1, 31))
SCENARIO_NAMES = ("default", "default_1", "default_2", "default_3", "default_4")
PEDESTRIAN_WAYPOINT_Z_METERS = 0.0
IMPORT_ID_PATTERN = re.compile(r"worlds_([0-9a-f]{8})\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ImportValidationError(ValueError):
    """Raised when raw or requested output data violates the import contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _yaml_bytes(value: Any) -> bytes:
    return yaml.safe_dump(
        value,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    ).encode("utf-8")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_output_relative(value: str, *, label: str, allow_directory: bool) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ImportValidationError(f"{label} must be a nonempty portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ImportValidationError(f"{label} must not be absolute or contain traversal: {value!r}")
    if not allow_directory and path.name in ("", ".", ".."):
        raise ImportValidationError(f"{label} must name a file: {value!r}")
    return path


def _safe_source_relative(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ImportValidationError(f"{label} must be a nonempty portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ImportValidationError(f"{label} contains an unsafe path: {value!r}")
    return path


def _validate_import_identity(import_id: str, source_archive_sha256: str) -> str:
    match = IMPORT_ID_PATTERN.fullmatch(import_id)
    if match is None:
        raise ImportValidationError("import_id must match worlds_<eight lowercase hex digits>")
    if SHA256_PATTERN.fullmatch(source_archive_sha256) is None:
        raise ImportValidationError("source_archive_sha256 must be 64 lowercase hexadecimal digits")
    suffix = match.group(1)
    if not source_archive_sha256.startswith(suffix):
        raise ImportValidationError("import_id suffix must equal the source archive SHA-256 prefix")
    return suffix


def _require_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ImportValidationError(f"{label} must be a real directory, not a symlink: {path}")


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ImportValidationError(f"missing or non-regular {label}: {path}")
    return path.read_bytes()


def _load_yaml_bytes(data: bytes, *, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImportValidationError(f"{label} is not valid UTF-8") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ImportValidationError(f"{label} is not valid YAML: {exc}") from exc


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImportValidationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ImportValidationError(f"{label} must be finite")
    return result


def _finite_vector(value: Any, length: int, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ImportValidationError(f"{label} must be a {length}-element list")
    return [_finite_number(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ImportValidationError(f"{label} must be a nonempty string")
    return value


def _validate_json_value(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite_number(value, label=label)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ImportValidationError(f"{label} contains a non-string key")
            _validate_json_value(item, label=f"{label}.{key}")
        return
    raise ImportValidationError(f"{label} contains unsupported YAML value {type(value).__name__}")


def _radians(degrees: float) -> float:
    value = round(math.radians(degrees), 12)
    return 0.0 if value == 0.0 else value


def _parse_png_dimensions(data: bytes, *, label: str) -> tuple[int, int]:
    if not data.startswith(PNG_SIGNATURE):
        raise ImportValidationError(f"{label} is not a PNG image")
    offset = len(PNG_SIGNATURE)
    dimensions: tuple[int, int] | None = None
    saw_iend = False
    chunk_index = 0
    while offset < len(data):
        if offset + 12 > len(data):
            raise ImportValidationError(f"{label} has a truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise ImportValidationError(f"{label} has a truncated PNG payload")
        chunk_data = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ImportValidationError(f"{label} has an invalid PNG chunk checksum")
        if chunk_index == 0:
            if chunk_type != b"IHDR" or length != 13:
                raise ImportValidationError(f"{label} does not start with a valid PNG IHDR")
            width, height = struct.unpack(">II", chunk_data[:8])
            if width <= 0 or height <= 0:
                raise ImportValidationError(f"{label} has invalid PNG dimensions")
            dimensions = (width, height)
        if chunk_type == b"IEND":
            if length != 0 or end != len(data):
                raise ImportValidationError(f"{label} has an invalid PNG IEND")
            saw_iend = True
            break
        offset = end
        chunk_index += 1
    if dimensions is None or not saw_iend:
        raise ImportValidationError(f"{label} is missing required PNG chunks")
    return dimensions


def _validate_world_yaml(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ImportValidationError(f"{label} must be a mapping")
    _validate_json_value(value, label=label)
    _nonempty_string(value.get("world_type"), label=f"{label}.world_type")
    scene = value.get("usd_scene")
    if not isinstance(scene, dict):
        raise ImportValidationError(f"{label}.usd_scene must be a mapping")
    usd_path = _nonempty_string(scene.get("path"), label=f"{label}.usd_scene.path")
    usd_parts = PurePosixPath(usd_path)
    if not usd_parts.is_absolute() or ".." in usd_parts.parts:
        raise ImportValidationError(f"{label}.usd_scene.path must be an absolute traversal-free asset path")
    if _finite_number(scene.get("scale"), label=f"{label}.usd_scene.scale") <= 0.0:
        raise ImportValidationError(f"{label}.usd_scene.scale must be positive")
    _finite_vector(scene.get("position"), 3, label=f"{label}.usd_scene.position")
    _finite_vector(scene.get("orientation"), 4, label=f"{label}.usd_scene.orientation")
    metadata = value.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ImportValidationError(f"{label}.metadata must be a mapping")
    return value


def _validate_map_yaml(value: Any, *, label: str) -> tuple[dict[str, Any], PurePosixPath, float, list[float]]:
    if not isinstance(value, dict):
        raise ImportValidationError(f"{label} must be a mapping")
    _validate_json_value(value, label=label)
    image_relative = _safe_source_relative(value.get("image"), label=f"{label}.image")
    resolution = _finite_number(value.get("resolution"), label=f"{label}.resolution")
    if resolution <= 0.0:
        raise ImportValidationError(f"{label}.resolution must be positive")
    origin = _finite_vector(value.get("origin"), 3, label=f"{label}.origin")
    occupied = _finite_number(value.get("occupied_thresh"), label=f"{label}.occupied_thresh")
    free = _finite_number(value.get("free_thresh"), label=f"{label}.free_thresh")
    if not 0.0 <= occupied <= 1.0 or not 0.0 <= free <= 1.0:
        raise ImportValidationError(f"{label} thresholds must be in [0, 1]")
    negate = _finite_number(value.get("negate"), label=f"{label}.negate")
    if negate not in (0.0, 1.0):
        raise ImportValidationError(f"{label}.negate must be 0 or 1")
    return value, image_relative, resolution, origin


def _validate_scenario(value: Any, *, label: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise ImportValidationError(f"{label} must be a mapping")
    _validate_json_value(value, label=label)
    if "robots" in value:
        raise ImportValidationError(f"{label} must contain the raw singular robot block only")
    robot = value.get("robot")
    if not isinstance(robot, dict):
        raise ImportValidationError(f"{label}.robot must be a mapping")
    _nonempty_string(robot.get("name"), label=f"{label}.robot.name")
    _finite_vector(robot.get("pose"), 3, label=f"{label}.robot.pose")
    waypoints = robot.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ImportValidationError(f"{label}.robot.waypoints must be a nonempty list")
    for index, waypoint in enumerate(waypoints):
        _finite_vector(waypoint, 3, label=f"{label}.robot.waypoints[{index}]")
    _nonempty_string(robot.get("behavior"), label=f"{label}.robot.behavior")

    dynamic_value = value.get("dynamic", [])
    if dynamic_value is None:
        dynamic_value = []
    if not isinstance(dynamic_value, list):
        raise ImportValidationError(f"{label}.dynamic must be a list when present")
    dynamic: list[dict[str, Any]] = []
    for agent_index, agent in enumerate(dynamic_value):
        agent_label = f"{label}.dynamic[{agent_index}]"
        if not isinstance(agent, dict):
            raise ImportValidationError(f"{agent_label} must be a mapping")
        _nonempty_string(agent.get("name"), label=f"{agent_label}.name")
        _nonempty_string(agent.get("model"), label=f"{agent_label}.model")
        _finite_vector(agent.get("pose"), 3, label=f"{agent_label}.pose")
        behavior_tree = _nonempty_string(agent.get("behavior_tree"), label=f"{agent_label}.behavior_tree")
        if PurePosixPath(behavior_tree).name != "BTRegularNav.xml" or len(PurePosixPath(behavior_tree).parts) != 1:
            raise ImportValidationError(f"{agent_label}.behavior_tree must name BTRegularNav.xml")
        _finite_number(agent.get("velocity"), label=f"{agent_label}.velocity")
        _finite_number(agent.get("desired_velocity"), label=f"{agent_label}.desired_velocity")
        agent_waypoints = agent.get("waypoints")
        if not isinstance(agent_waypoints, list) or not agent_waypoints:
            raise ImportValidationError(f"{agent_label}.waypoints must be a nonempty list")
        for waypoint_index, waypoint in enumerate(agent_waypoints):
            _finite_vector(waypoint, 3, label=f"{agent_label}.waypoints[{waypoint_index}]")
        dynamic.append(agent)
    return value, robot, dynamic


def _point_inside_bounds(point: list[float], bounds: dict[str, float]) -> bool:
    return bounds["min_x_m"] <= point[0] <= bounds["max_x_m"] and bounds["min_y_m"] <= point[1] <= bounds["max_y_m"]


def _promotion_governance(
    files: dict[str, bytes],
    *,
    output_worlds_relative: PurePosixPath,
    output_catalog_relative: PurePosixPath,
    variant_ids: list[str],
    import_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    generated_world_roots: list[str] = []
    for variant_id in variant_ids:
        generated_root = (output_worlds_relative / variant_id).as_posix()
        generated_prefix = f"{generated_root}/"
        if not any(path.startswith(generated_prefix) for path in files):
            raise ImportValidationError(f"promotion root has no generated members: {generated_root}")
        generated_world_roots.append(generated_root)

    source_world_roots = [
        f"arena_simulation_setup/worlds/{PurePosixPath(root).name}" for root in generated_world_roots
    ]
    installed_share_world_roots = [
        f"share/arena_simulation_setup/worlds/{PurePosixPath(root).name}" for root in generated_world_roots
    ]
    offline_catalog_destinations = [
        f"arena_bringup/configs/grscenes_benchmark/{import_id}_catalog.json"
    ]
    allowlist = {
        "status": "generated_artifact_backed_not_promoted",
        "path_count": len(source_world_roots) + len(installed_share_world_roots) + len(offline_catalog_destinations),
        "source_world_roots": source_world_roots,
        "installed_share_world_roots": installed_share_world_roots,
        "offline_catalog_destinations": offline_catalog_destinations,
    }
    evidence = {
        "generated_world_roots": generated_world_roots,
        "generated_offline_catalog": output_catalog_relative.as_posix(),
        "verification": "every allowlisted class must map to a generated manifest member",
    }
    deferred = {
        "benchmark_case_configs": {
            "status": "deferred_pending_exactly_three_selection_and_human_instruction_review",
            "generated_artifact_count": 0,
            "allowlisted_path_count": 0,
        }
    }
    return allowlist, evidence, deferred


def _add_file(files: dict[str, bytes], relative: PurePosixPath, data: bytes) -> None:
    key = relative.as_posix()
    if key in files:
        raise ImportValidationError(f"generated output collision: {key}")
    files[key] = data


def _build_manifest(
    files: dict[str, bytes],
    *,
    manifest_relative: PurePosixPath,
    import_id: str,
    source_archive_sha256: str,
    bt_template_sha256: str,
) -> bytes:
    manifest_key = manifest_relative.as_posix()
    entries = [
        {"path": path, "sha256": sha256_bytes(data), "size_bytes": len(data)}
        for path, data in sorted(files.items())
    ]
    tree_input = "".join(
        f"{entry['path']}\0{entry['size_bytes']}\0{entry['sha256']}\n" for entry in entries
    ).encode("utf-8")
    self_entry = {
        "digest_exclusion": "self_referential_manifest",
        "path": manifest_key,
        "sha256": None,
        "size_bytes": 0,
    }
    manifest = {
        "schema_version": 1,
        "import_id": import_id,
        "source_archive_sha256": source_archive_sha256,
        "bt_template_sha256": bt_template_sha256,
        "tree_sha256": sha256_bytes(tree_input),
        "tree_sha256_scope": "all generated files except this manifest",
        "files": entries + [self_entry],
    }
    manifest["files"].sort(key=lambda entry: entry["path"])
    previous_size = -1
    while self_entry["size_bytes"] != previous_size:
        previous_size = self_entry["size_bytes"]
        encoded = _json_bytes(manifest)
        self_entry["size_bytes"] = len(encoded)
    encoded = _json_bytes(manifest)
    if len(encoded) != self_entry["size_bytes"]:
        raise RuntimeError("manifest self-size did not converge")
    return encoded


def materialize(
    *,
    input_worlds: Path,
    output_root: Path,
    output_worlds: str,
    output_catalog: str,
    output_report: str,
    output_manifest: str,
    import_id: str,
    source_archive_sha256: str,
    bt_template: Path,
) -> dict[str, Any]:
    """Validate and write one complete normalized tree into a fresh output root."""

    _validate_import_identity(import_id, source_archive_sha256)
    input_worlds = input_worlds.expanduser().resolve(strict=True)
    _require_directory(input_worlds, label="input_worlds")
    output_root = output_root.expanduser().resolve(strict=False)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output_root already exists; refusing to overwrite: {output_root}")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise ImportValidationError("output_root parent must already be a real directory")
    if _is_within(output_root, input_worlds):
        raise ImportValidationError("output_root must not be inside the immutable input tree")

    output_worlds_relative = _safe_output_relative(output_worlds, label="output_worlds", allow_directory=True)
    output_catalog_relative = _safe_output_relative(output_catalog, label="output_catalog", allow_directory=False)
    output_report_relative = _safe_output_relative(output_report, label="output_report", allow_directory=False)
    output_manifest_relative = _safe_output_relative(output_manifest, label="output_manifest", allow_directory=False)
    output_file_relatives = {output_catalog_relative, output_report_relative, output_manifest_relative}
    if len(output_file_relatives) != 3:
        raise ImportValidationError("catalog, report, and manifest paths must be distinct")
    for relative in output_file_relatives:
        try:
            relative.relative_to(output_worlds_relative)
        except ValueError:
            continue
        raise ImportValidationError("catalog, report, and manifest must be outside output_worlds")

    bt_template = bt_template.expanduser().resolve(strict=True)
    bt_template_bytes = _read_regular_file(bt_template, label="BT template")
    if not bt_template_bytes.strip():
        raise ImportValidationError("BT template must not be empty")
    try:
        ET.fromstring(bt_template_bytes)
    except ET.ParseError as exc:
        raise ImportValidationError(f"BT template is not valid XML: {exc}") from exc
    bt_template_sha256 = sha256_bytes(bt_template_bytes)

    expected_worlds = set(WORLD_NAMES)
    discovered_worlds = {entry.name for entry in input_worlds.iterdir() if entry.name.startswith("grscenes_")}
    if discovered_worlds != expected_worlds:
        missing = sorted(expected_worlds - discovered_worlds)
        unexpected = sorted(discovered_worlds - expected_worlds)
        raise ImportValidationError(f"input must contain exactly grscenes_1..30; missing={missing}, unexpected={unexpected}")

    files: dict[str, bytes] = {}
    assets: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    block_reason_counts: Counter[str] = Counter()
    human_count = 0
    pedestrian_waypoint_count = 0
    robot_yaw_conversion_count = 0
    dynamic_pose_yaw_conversion_count = 0
    variant_ids = [f"{world_name}_v1" for world_name in WORLD_NAMES]

    for world_name, variant_id in zip(WORLD_NAMES, variant_ids):
        world_dir = input_worlds / world_name
        _require_directory(world_dir, label=f"source world {world_name}")
        for descendant in world_dir.rglob("*"):
            if descendant.is_symlink():
                raise ImportValidationError(f"source world contains a symlink: {descendant}")

        world_path = world_dir / "world.yaml"
        world_bytes = _read_regular_file(world_path, label=f"{world_name}/world.yaml")
        world_data = _validate_world_yaml(
            _load_yaml_bytes(world_bytes, label=f"{world_name}/world.yaml"),
            label=f"{world_name}/world.yaml",
        )
        map_yaml_path = world_dir / "map" / "map.yaml"
        map_yaml_bytes = _read_regular_file(map_yaml_path, label=f"{world_name}/map/map.yaml")
        map_data, map_image_relative, map_resolution, map_origin = _validate_map_yaml(
            _load_yaml_bytes(map_yaml_bytes, label=f"{world_name}/map/map.yaml"),
            label=f"{world_name}/map/map.yaml",
        )
        map_image_path = world_dir / "map" / Path(*map_image_relative.parts)
        if not _is_within(map_image_path.resolve(strict=False), (world_dir / "map").resolve(strict=True)):
            raise ImportValidationError(f"{world_name} map image escapes its map directory")
        map_image_bytes = _read_regular_file(map_image_path, label=f"{world_name} map image")
        width_px, height_px = _parse_png_dimensions(map_image_bytes, label=f"{world_name} map image")
        bounds = {
            "min_x_m": map_origin[0],
            "min_y_m": map_origin[1],
            "max_x_m": map_origin[0] + width_px * map_resolution,
            "max_y_m": map_origin[1] + height_px * map_resolution,
        }

        generated_world_path = output_worlds_relative / variant_id / "world.yaml"
        generated_map_yaml_path = output_worlds_relative / variant_id / "map" / "map.yaml"
        generated_map_image_path = output_worlds_relative / variant_id / "map" / map_image_relative
        _add_file(files, generated_world_path, world_bytes)
        _add_file(files, generated_map_yaml_path, map_yaml_bytes)
        _add_file(files, generated_map_image_path, map_image_bytes)

        asset = {
            "variant_id": variant_id,
            "source_world": world_name,
            "generated_relative_path": (output_worlds_relative / variant_id).as_posix(),
            "source": {
                "world_yaml": {
                    "path": f"{world_name}/world.yaml",
                    "sha256": sha256_bytes(world_bytes),
                },
                "map_yaml": {
                    "path": f"{world_name}/map/map.yaml",
                    "sha256": sha256_bytes(map_yaml_bytes),
                },
                "map_image": {
                    "path": f"{world_name}/map/{map_image_relative.as_posix()}",
                    "sha256": sha256_bytes(map_image_bytes),
                },
            },
            "generated": {
                "world_yaml": {
                    "path": generated_world_path.as_posix(),
                    "sha256": sha256_bytes(world_bytes),
                },
                "map_yaml": {
                    "path": generated_map_yaml_path.as_posix(),
                    "sha256": sha256_bytes(map_yaml_bytes),
                },
                "map_image": {
                    "path": generated_map_image_path.as_posix(),
                    "sha256": sha256_bytes(map_image_bytes),
                },
            },
            "world": {
                "world_type": world_data["world_type"],
                "usd_scene": copy.deepcopy(world_data["usd_scene"]),
                "availability_status": "external_asset_not_embedded_unvalidated",
            },
            "map": {
                "image_width_px": width_px,
                "image_height_px": height_px,
                "resolution_m_per_px": map_resolution,
                "origin_xy_yaw": map_origin,
                "origin_yaw_unit": "radians",
                "bounds_axis_aligned_m": bounds,
                "raw_yaml": copy.deepcopy(map_data),
            },
        }
        assets.append(asset)

        scenarios_dir = world_dir / "scenarios"
        _require_directory(scenarios_dir, label=f"{world_name}/scenarios")
        scenario_directories = {entry.name for entry in scenarios_dir.iterdir() if entry.is_dir()}
        if scenario_directories != set(SCENARIO_NAMES):
            missing = sorted(set(SCENARIO_NAMES) - scenario_directories)
            unexpected = sorted(scenario_directories - set(SCENARIO_NAMES))
            raise ImportValidationError(
                f"{world_name} must contain exactly default..default_4 scenario directories; "
                f"missing={missing}, unexpected={unexpected}"
            )

        for scenario_name in SCENARIO_NAMES:
            source_scenario_path = scenarios_dir / scenario_name / "scenario.yaml"
            source_scenario_bytes = _read_regular_file(
                source_scenario_path,
                label=f"{world_name}/scenarios/{scenario_name}/scenario.yaml",
            )
            raw_scenario, raw_robot, raw_dynamic = _validate_scenario(
                _load_yaml_bytes(
                    source_scenario_bytes,
                    label=f"{world_name}/scenarios/{scenario_name}/scenario.yaml",
                ),
                label=f"{world_name}/scenarios/{scenario_name}/scenario.yaml",
            )

            raw_start = _finite_vector(raw_robot["pose"], 3, label="robot.pose")
            raw_robot_waypoints = [
                _finite_vector(waypoint, 3, label=f"robot.waypoints[{index}]")
                for index, waypoint in enumerate(raw_robot["waypoints"])
            ]
            raw_goal = raw_robot_waypoints[-1]
            normalized_start = [raw_start[0], raw_start[1], _radians(raw_start[2])]
            normalized_goal = [raw_goal[0], raw_goal[1], _radians(raw_goal[2])]
            robot_yaw_conversion_count += 2

            normalized_dynamic: list[dict[str, Any]] = []
            human_catalog: list[dict[str, Any]] = []
            for agent_index, raw_agent in enumerate(raw_dynamic):
                raw_pose = _finite_vector(raw_agent["pose"], 3, label=f"dynamic[{agent_index}].pose")
                raw_waypoints = [
                    _finite_vector(waypoint, 3, label=f"dynamic[{agent_index}].waypoints[{waypoint_index}]")
                    for waypoint_index, waypoint in enumerate(raw_agent["waypoints"])
                ]
                normalized_pose = [raw_pose[0], raw_pose[1], _radians(raw_pose[2])]
                emitted_waypoints = [
                    [waypoint[0], waypoint[1], PEDESTRIAN_WAYPOINT_Z_METERS] for waypoint in raw_waypoints
                ]
                normalized_agent = copy.deepcopy(raw_agent)
                normalized_agent["pose"] = normalized_pose
                normalized_agent["behavior_tree"] = "./BTRegularNav.xml"
                normalized_agent["waypoints"] = emitted_waypoints
                normalized_dynamic.append(normalized_agent)
                human_catalog.append(
                    {
                        "index": agent_index,
                        "raw_agent": copy.deepcopy(raw_agent),
                        "normalized_pose_xy_yaw_rad": normalized_pose,
                        "raw_waypoints_xy_heading_deg": raw_waypoints,
                        "emitted_waypoints_xyz_m": emitted_waypoints,
                        "emitted_waypoint_z_m": PEDESTRIAN_WAYPOINT_Z_METERS,
                    }
                )
                human_count += 1
                pedestrian_waypoint_count += len(raw_waypoints)
                dynamic_pose_yaw_conversion_count += 1

            native_scenario = {
                key: copy.deepcopy(value)
                for key, value in raw_scenario.items()
                if key not in ("robot", "robots", "dynamic")
            }
            if normalized_dynamic:
                native_scenario["dynamic"] = normalized_dynamic
            native_scenario["robots"] = [{"start": normalized_start, "goal": normalized_goal}]
            generated_scenario_bytes = _yaml_bytes(native_scenario)
            generated_scenario_path = (
                output_worlds_relative / variant_id / "scenarios" / scenario_name / "scenario.yaml"
            )
            _add_file(files, generated_scenario_path, generated_scenario_bytes)

            generated_bt_path: PurePosixPath | None = None
            if normalized_dynamic:
                generated_bt_path = (
                    output_worlds_relative / variant_id / "scenarios" / scenario_name / "BTRegularNav.xml"
                )
                _add_file(files, generated_bt_path, bt_template_bytes)

            block_reasons: list[str] = []
            start_inside = _point_inside_bounds(normalized_start, bounds)
            goal_inside = _point_inside_bounds(normalized_goal, bounds)
            if not start_inside:
                block_reasons.append("robot_start_out_of_map_bounds")
            if not goal_inside:
                block_reasons.append("robot_goal_out_of_map_bounds")
            if not normalized_dynamic:
                block_reasons.append("no_active_humans_for_social_case")
            block_reason_counts.update(block_reasons)

            behavior_text = _nonempty_string(raw_robot["behavior"], label="robot.behavior")
            case_id = f"{import_id}:{world_name}:{scenario_name}"
            case = {
                "identity": {
                    "id": case_id,
                    "kind": "scenario_case",
                    "episode": None,
                    "timestamp": None,
                },
                "source": {
                    "import_id": import_id,
                    "archive_sha256": source_archive_sha256,
                    "world": world_name,
                    "scenario": scenario_name,
                    "scenario_path": f"{world_name}/scenarios/{scenario_name}/scenario.yaml",
                    "scenario_sha256": sha256_bytes(source_scenario_bytes),
                },
                "asset": {
                    "variant_id": variant_id,
                    "generated_relative_path": (output_worlds_relative / variant_id).as_posix(),
                    "world_yaml_sha256": sha256_bytes(world_bytes),
                    "map_yaml_sha256": sha256_bytes(map_yaml_bytes),
                    "map_image_sha256": sha256_bytes(map_image_bytes),
                    "usd_path": world_data["usd_scene"]["path"],
                },
                "native_scenario": {
                    "generated_path": generated_scenario_path.as_posix(),
                    "generated_sha256": sha256_bytes(generated_scenario_bytes),
                    "behavior_tree_path": generated_bt_path.as_posix() if generated_bt_path else None,
                    "behavior_tree_sha256": bt_template_sha256 if generated_bt_path else None,
                },
                "units_and_frames": {
                    "route_frame": "map",
                    "robot_source_xy_unit": "meters",
                    "robot_source_yaw_unit": "degrees",
                    "robot_generated_yaw_unit": "radians",
                    "dynamic_source_pose_yaw_unit": "degrees",
                    "dynamic_generated_pose_yaw_unit": "radians",
                    "pedestrian_source_waypoint_semantics": ["x_m", "y_m", "heading_degrees"],
                    "pedestrian_generated_waypoint_semantics": ["x_m", "y_m", "z_m"],
                    "pedestrian_generated_waypoint_z_m": PEDESTRIAN_WAYPOINT_Z_METERS,
                },
                "route": {
                    "raw_start_xy_yaw_deg": raw_start,
                    "raw_waypoints_xy_yaw_deg": raw_robot_waypoints,
                    "normalized_start_xy_yaw_rad": normalized_start,
                    "normalized_goal_xy_yaw_rad": normalized_goal,
                    "map_bounds_axis_aligned_m": bounds,
                    "start_inside_map_bounds": start_inside,
                    "goal_inside_map_bounds": goal_inside,
                },
                "humans": {
                    "count": len(normalized_dynamic),
                    "agents": human_catalog,
                },
                "language": {
                    "raw_behavior_text": behavior_text,
                    "raw_behavior_utf8_sha256": sha256_bytes(behavior_text.encode("utf-8")),
                    "derived_instruction": {
                        "text": behavior_text,
                        "derived": True,
                        "review_status": "pending",
                        "source_field": "robot.behavior",
                        "equivalence_claimed": False,
                    },
                },
                "admission": {
                    "blocked": bool(block_reasons),
                    "block_reasons": block_reasons,
                    "review_flags": [
                        "external_usd_asset_not_embedded_unvalidated",
                        "derived_instruction_pending_review",
                    ],
                },
            }
            cases.append(case)

    future_allowlist, promotion_evidence, deferred_generation = _promotion_governance(
        files,
        output_worlds_relative=output_worlds_relative,
        output_catalog_relative=output_catalog_relative,
        variant_ids=variant_ids,
        import_id=import_id,
    )
    blocked_cases = [case["identity"]["id"] for case in cases if case["admission"]["blocked"]]
    catalog = {
        "schema_version": 1,
        "catalog_kind": "offline_grscenes_scenario_case_catalog",
        "runtime_dependency": False,
        "import": {
            "id": import_id,
            "source_archive_sha256": source_archive_sha256,
            "conversion_implementation": "arena_bringup.grscenes_worlds_import",
            "conversion_version": CONVERSION_VERSION,
            "bt_template_sha256": bt_template_sha256,
        },
        "semantics": {
            "identity_kind": "scenario_case",
            "source_episode_metadata_available": False,
            "source_timestamp_metadata_available": False,
            "robot_yaw_conversion": "degrees_to_radians_for_all_values",
            "dynamic_pose_yaw_conversion": "degrees_to_radians_for_all_values",
            "pedestrian_waypoint_policy": (
                "preserve raw [x,y,heading_degrees] in catalog; emit native [x,y,z] with z=0.0 meters"
            ),
            "derived_instruction_policy": "verbatim robot.behavior candidate pending review; no equivalence claimed",
        },
        "statistics": {
            "world_count": len(assets),
            "case_count": len(cases),
            "human_count": human_count,
            "pedestrian_waypoint_count": pedestrian_waypoint_count,
            "blocked_case_count": len(blocked_cases),
            "block_reason_counts": dict(sorted(block_reason_counts.items())),
        },
        "future_promotion_allowlist": future_allowlist,
        "promotion_artifact_evidence": promotion_evidence,
        "deferred_generation_classes": deferred_generation,
        "assets": assets,
        "cases": cases,
    }
    catalog_bytes = _json_bytes(catalog)
    _add_file(files, output_catalog_relative, catalog_bytes)

    report = {
        "schema_version": 1,
        "import_id": import_id,
        "source_archive_sha256": source_archive_sha256,
        "validation": "passed",
        "runtime_catalog_dependency": False,
        "statistics": {
            "world_count": len(assets),
            "case_count": len(cases),
            "scenario_case_identity_count": len({case["identity"]["id"] for case in cases}),
            "human_count": human_count,
            "pedestrian_waypoint_count": pedestrian_waypoint_count,
            "robot_yaw_conversion_count": robot_yaw_conversion_count,
            "dynamic_pose_yaw_conversion_count": dynamic_pose_yaw_conversion_count,
            "blocked_case_count": len(blocked_cases),
            "block_reason_counts": dict(sorted(block_reason_counts.items())),
        },
        "blocked_case_ids": blocked_cases,
        "future_promotion_allowlist": future_allowlist,
        "promotion_artifact_evidence": promotion_evidence,
        "deferred_generation_classes": deferred_generation,
        "notes": [
            "No source episode or timestamp was available or fabricated.",
            "No benchmark case configuration was generated.",
            "Out-of-bounds routes were recorded and blocked without clamping or shifting.",
            "The external USD payload remains unvalidated and is not embedded in the source archive.",
        ],
    }
    report_bytes = _json_bytes(report)
    _add_file(files, output_report_relative, report_bytes)

    manifest_bytes = _build_manifest(
        files,
        manifest_relative=output_manifest_relative,
        import_id=import_id,
        source_archive_sha256=source_archive_sha256,
        bt_template_sha256=bt_template_sha256,
    )
    _add_file(files, output_manifest_relative, manifest_bytes)

    try:
        output_root.mkdir()
        for relative, data in sorted(files.items()):
            destination = output_root / Path(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise

    manifest = json.loads(manifest_bytes)
    return {
        "output_root": str(output_root),
        "world_count": len(assets),
        "case_count": len(cases),
        "blocked_case_count": len(blocked_cases),
        "file_count": len(files),
        "manifest_tree_sha256": manifest["tree_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and deterministically normalize the immutable GRScenes worlds archive."
    )
    parser.add_argument("--input-worlds", required=True, type=Path, help="Immutable extracted worlds root.")
    parser.add_argument("--output-root", required=True, type=Path, help="Fresh root; it must not already exist.")
    parser.add_argument(
        "--output-worlds",
        required=True,
        help="Portable path below output-root for package-like generated worlds.",
    )
    parser.add_argument(
        "--output-catalog",
        "--output-config-catalog",
        dest="output_catalog",
        required=True,
        help="Portable path below output-root for the offline config/catalog JSON.",
    )
    parser.add_argument("--output-report", required=True, help="Portable path below output-root for the report JSON.")
    parser.add_argument("--output-manifest", required=True, help="Portable path below output-root for the manifest JSON.")
    parser.add_argument("--import-id", required=True, help="Import ID in worlds_<archive SHA prefix> form.")
    parser.add_argument("--source-archive-sha256", required=True, help="Pinned lowercase SHA-256 of the raw archive.")
    parser.add_argument("--bt-template", required=True, type=Path, help="Explicit BTRegularNav.xml template.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize(
        input_worlds=args.input_worlds,
        output_root=args.output_root,
        output_worlds=args.output_worlds,
        output_catalog=args.output_catalog,
        output_report=args.output_report,
        output_manifest=args.output_manifest,
        import_id=args.import_id,
        source_archive_sha256=args.source_archive_sha256,
        bt_template=args.bt_template,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
