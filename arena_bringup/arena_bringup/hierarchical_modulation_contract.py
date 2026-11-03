"""Offline validator for atomic hierarchical-agent modulation snapshots.

This module deliberately has no ROS imports, publishers, subscribers, launch
entry points, network clients, or model dependencies.  It is a P0/P3 contract
fixture only; production consumers must be added behind a later phase gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CONTRACT_SCHEMA = "arena.hierarchical_modulation"
CONTRACT_VERSION = 1
CONTROL_PATHS = frozenset({
    "vln_direct_cmd_vel",
    "vln_plus_social_mpc",
    "lisn_style_social_mpc",
    "lisn_style_prox_mpc",
    "lisn_sfm",
})
CONTROLLER_MODES = frozenset({"goal", "follow", "avoid", "wait", "cautious"})
GOAL_KINDS = frozenset({"map_pose", "pixel_goal", "target_identity"})

CONTROLLER_PARAMETER_RANGES = {
    "max_linear_velocity_mps": (0.0, 10.0),
    "max_angular_velocity_rps": (0.0, 20.0),
    "goal_weight": (0.0, 100.0),
    "obstacle_weight": (0.0, 100.0),
    "people_weight": (0.0, 100.0),
}

TOP_LEVEL_FIELDS = frozenset({
    "schema",
    "schema_version",
    "update_semantics",
    "update_id",
    "control_path",
    "generated_at_ns",
    "valid_until_ns",
    "source_observation",
    "target_frame",
    "goal",
    "semantic_entities",
    "controller_mode",
    "controller_parameters",
    "provenance",
    "reason",
})
SOURCE_FIELDS = frozenset({"id", "frame_id", "stamp_ns"})
GOAL_FIELDS = frozenset({"kind", "frame_id", "stamp_ns", "value"})
ENTITY_FIELDS = frozenset({
    "class_name",
    "attribute",
    "instance_id",
    "frame_id",
    "stamp_ns",
    "confidence",
    "geometry_ref",
    "cost_value",
    "inflation_radius_m",
    "decay_rate",
})
# ``attribute`` is optional for backward compatibility with older fixtures and
# snapshots; the LISN host always fills it.
ENTITY_REQUIRED_FIELDS = ENTITY_FIELDS - {"attribute"}
PROVENANCE_FIELDS = frozenset({
    "model_version",
    "prompt_version",
    "tool_version",
    "reasoning_latency_ms",
})


@dataclass(frozen=True)
class ContractValidationError(ValueError):
    """All fail-closed validation errors for one snapshot."""

    errors: tuple[str, ...]

    def __str__(self) -> str:
        return "; ".join(self.errors)


def _unknown_fields(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> list[str]:
    return [f"{path}: unknown field {name!r}" for name in sorted(set(value) - allowed)]


def _missing_fields(value: Mapping[str, Any], required: frozenset[str], path: str) -> list[str]:
    return [f"{path}: missing required field {name!r}" for name in sorted(required - set(value))]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_modulation_snapshot(
    payload: Mapping[str, Any],
    *,
    now_ns: int,
    expected_target_frame: str,
    max_observation_age_ns: int = 5_000_000_000,
) -> Mapping[str, Any]:
    """Validate one complete replacement snapshot or reject it atomically.

    The function returns the original mapping only when every check passes.  It
    never applies a subset of fields.  Callers must catch
    :class:`ContractValidationError` and retain their previous accepted state.
    """

    errors: list[str] = []
    if not isinstance(payload, Mapping):
        raise ContractValidationError(("root: expected object",))

    errors.extend(_unknown_fields(payload, TOP_LEVEL_FIELDS, "root"))
    errors.extend(_missing_fields(payload, TOP_LEVEL_FIELDS, "root"))

    if payload.get("schema") != CONTRACT_SCHEMA:
        errors.append(f"schema: expected {CONTRACT_SCHEMA!r}")
    version = payload.get("schema_version")
    if version != CONTRACT_VERSION:
        errors.append(f"schema_version: unsupported version {version!r}")
    if payload.get("update_semantics") != "replace_all":
        errors.append("update_semantics: only 'replace_all' is accepted")
    if not _nonempty_string(payload.get("update_id")):
        errors.append("update_id: expected non-empty string")
    if payload.get("control_path") not in CONTROL_PATHS:
        errors.append(f"control_path: unsupported value {payload.get('control_path')!r}")
    if payload.get("controller_mode") not in CONTROLLER_MODES:
        errors.append(f"controller_mode: unsupported value {payload.get('controller_mode')!r}")

    generated_at_ns = payload.get("generated_at_ns")
    valid_until_ns = payload.get("valid_until_ns")
    if not isinstance(generated_at_ns, int) or isinstance(generated_at_ns, bool):
        errors.append("generated_at_ns: expected integer nanoseconds")
    if not isinstance(valid_until_ns, int) or isinstance(valid_until_ns, bool):
        errors.append("valid_until_ns: expected integer nanoseconds")
    if isinstance(generated_at_ns, int) and isinstance(valid_until_ns, int):
        if generated_at_ns >= valid_until_ns:
            errors.append("validity: generated_at_ns must be earlier than valid_until_ns")
        if now_ns > valid_until_ns:
            errors.append("validity: snapshot is stale")
        if generated_at_ns > now_ns:
            errors.append("validity: generated_at_ns is in the future")

    target_frame = payload.get("target_frame")
    if target_frame != expected_target_frame:
        errors.append(
            f"target_frame: expected {expected_target_frame!r}, got {target_frame!r}"
        )

    source = payload.get("source_observation")
    if not isinstance(source, Mapping):
        errors.append("source_observation: expected object")
    else:
        errors.extend(_unknown_fields(source, SOURCE_FIELDS, "source_observation"))
        errors.extend(_missing_fields(source, SOURCE_FIELDS, "source_observation"))
        if not _nonempty_string(source.get("id")):
            errors.append("source_observation.id: expected non-empty string")
        if not _nonempty_string(source.get("frame_id")):
            errors.append("source_observation.frame_id: expected non-empty string")
        stamp_ns = source.get("stamp_ns")
        if not isinstance(stamp_ns, int) or isinstance(stamp_ns, bool):
            errors.append("source_observation.stamp_ns: expected integer nanoseconds")
        elif isinstance(generated_at_ns, int):
            if stamp_ns > generated_at_ns:
                errors.append("source_observation: stamp is newer than generation time")
            elif generated_at_ns - stamp_ns > max_observation_age_ns:
                errors.append("source_observation: observation is too old")

    goal = payload.get("goal")
    if goal is not None:
        if not isinstance(goal, Mapping):
            errors.append("goal: expected object or null")
        else:
            errors.extend(_unknown_fields(goal, GOAL_FIELDS, "goal"))
            errors.extend(_missing_fields(goal, GOAL_FIELDS, "goal"))
            if goal.get("kind") not in GOAL_KINDS:
                errors.append(f"goal.kind: unsupported value {goal.get('kind')!r}")
            if not _nonempty_string(goal.get("frame_id")):
                errors.append("goal.frame_id: expected non-empty string")
            if goal.get("kind") == "map_pose" and goal.get("frame_id") != target_frame:
                errors.append("goal.frame_id: map_pose must use target_frame")
            if not isinstance(goal.get("stamp_ns"), int) or isinstance(goal.get("stamp_ns"), bool):
                errors.append("goal.stamp_ns: expected integer nanoseconds")
            if not isinstance(goal.get("value"), Mapping):
                errors.append("goal.value: expected object")

    entities = payload.get("semantic_entities")
    if not isinstance(entities, list):
        errors.append("semantic_entities: expected array")
    else:
        for index, entity in enumerate(entities):
            path = f"semantic_entities[{index}]"
            if not isinstance(entity, Mapping):
                errors.append(f"{path}: expected object")
                continue
            errors.extend(_unknown_fields(entity, ENTITY_FIELDS, path))
            errors.extend(_missing_fields(entity, ENTITY_REQUIRED_FIELDS, path))
            if not _nonempty_string(entity.get("class_name")):
                errors.append(f"{path}.class_name: expected non-empty string")
            if "attribute" in entity and not _nonempty_string(entity.get("attribute")):
                errors.append(f"{path}.attribute: expected non-empty string")
            if not _nonempty_string(entity.get("instance_id")):
                errors.append(f"{path}.instance_id: expected non-empty string")
            if entity.get("frame_id") != target_frame:
                errors.append(f"{path}.frame_id: must match target_frame")
            if not isinstance(entity.get("stamp_ns"), int) or isinstance(entity.get("stamp_ns"), bool):
                errors.append(f"{path}.stamp_ns: expected integer nanoseconds")
            confidence = entity.get("confidence")
            if not _number(confidence) or not 0.0 <= float(confidence) <= 1.0:
                errors.append(f"{path}.confidence: expected range [0, 1]")
            cost = entity.get("cost_value")
            if not isinstance(cost, int) or isinstance(cost, bool) or not 0 <= cost <= 254:
                errors.append(f"{path}.cost_value: expected integer range [0, 254]")
            radius = entity.get("inflation_radius_m")
            if not _number(radius) or not 0.0 <= float(radius) <= 5.0:
                errors.append(f"{path}.inflation_radius_m: expected range [0, 5]")
            decay = entity.get("decay_rate")
            if not _number(decay) or not 0.0 <= float(decay) <= 10.0:
                errors.append(f"{path}.decay_rate: expected range [0, 10]")
            if not _nonempty_string(entity.get("geometry_ref")):
                errors.append(f"{path}.geometry_ref: expected non-empty string")

    parameters = payload.get("controller_parameters")
    required_parameters = frozenset(CONTROLLER_PARAMETER_RANGES)
    if not isinstance(parameters, Mapping):
        errors.append("controller_parameters: expected object")
    else:
        errors.extend(_unknown_fields(parameters, required_parameters, "controller_parameters"))
        errors.extend(_missing_fields(parameters, required_parameters, "controller_parameters"))
        for name, (minimum, maximum) in CONTROLLER_PARAMETER_RANGES.items():
            value = parameters.get(name)
            if not _number(value) or not minimum <= float(value) <= maximum:
                errors.append(
                    f"controller_parameters.{name}: expected range [{minimum}, {maximum}]"
                )

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance: expected object")
    else:
        errors.extend(_unknown_fields(provenance, PROVENANCE_FIELDS, "provenance"))
        errors.extend(_missing_fields(provenance, PROVENANCE_FIELDS, "provenance"))
        for name in ("model_version", "prompt_version", "tool_version"):
            if not _nonempty_string(provenance.get(name)):
                errors.append(f"provenance.{name}: expected non-empty string")
        latency = provenance.get("reasoning_latency_ms")
        if not _number(latency) or float(latency) < 0.0:
            errors.append("provenance.reasoning_latency_ms: expected non-negative number")

    if not _nonempty_string(payload.get("reason")):
        errors.append("reason: expected non-empty string")

    if errors:
        raise ContractValidationError(tuple(dict.fromkeys(errors)))
    return payload
