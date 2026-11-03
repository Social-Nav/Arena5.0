"""Fail-closed contract between density configs and obstacle consumers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "arena.density_consumer_contract.v1"
DENSITY_CONFIG = re.compile(r"^task_generator_n(?P<count>[0-9]+)\.yaml$")


def _runtime_path(path: str | Path) -> Path:
    raw = str(path)
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    container_prefix = "/opt/arena_ws/src/Arena/"
    if raw.startswith(container_prefix):
        repository = Path(__file__).resolve().parents[2]
        mapped = repository / raw[len(container_prefix):]
        if mapped.is_file():
            return mapped.resolve()
    return candidate.resolve()


def _density_count(config: dict[str, Any]) -> tuple[int | None, list[str]]:
    failures = []
    try:
        values = config["/**"]["ros__parameters"]["task"]["random"]["dynamic"]["n"]
    except (KeyError, TypeError):
        return None, ["random_dynamic_count_missing"]
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or any(value < 0 for value in values)
    ):
        return None, ["random_dynamic_count_invalid"]
    if values[0] != values[1]:
        return None, ["random_dynamic_count_not_fixed"]
    return int(values[0]), failures


def resolve_density_contract(
    task_generator_parameter_file: str | Path | None,
    *,
    requested_consumer: str,
    resolved_consumer: str,
    expected_human_count: int,
) -> dict[str, Any]:
    """Resolve and validate a fixed-density random-human contract."""
    raw_path = str(task_generator_parameter_file or "").strip()
    basename = Path(raw_path).name if raw_path else ""
    match = DENSITY_CONFIG.fullmatch(basename)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "applicable": bool(match),
        "valid": True,
        "config_path_requested": raw_path or None,
        "config_path_resolved": None,
        "config_sha256": None,
        "configured_human_count": None,
        "filename_human_count": int(match.group("count")) if match else None,
        "expected_human_count": int(expected_human_count),
        "consumer": {
            "requested": str(requested_consumer),
            "resolved": str(resolved_consumer),
            "required": "random" if match else None,
        },
        "failure_reasons": [],
    }
    if not match:
        return result

    failures: list[str] = []
    config_path = _runtime_path(raw_path)
    result["config_path_resolved"] = str(config_path)
    if not config_path.is_file():
        failures.append("density_config_missing")
    else:
        data = config_path.read_bytes()
        result["config_sha256"] = hashlib.sha256(data).hexdigest()
        try:
            config = yaml.safe_load(data) or {}
        except yaml.YAMLError:
            config = {}
            failures.append("density_config_yaml_invalid")
        configured_count, count_failures = _density_count(config)
        failures.extend(count_failures)
        result["configured_human_count"] = configured_count
        if configured_count is not None:
            if configured_count != result["filename_human_count"]:
                failures.append("filename_count_mismatch")
            if configured_count != int(expected_human_count):
                failures.append("expected_count_mismatch")

    if str(resolved_consumer).strip().lower() != "random":
        failures.append("density_consumer_must_be_random")
    result["failure_reasons"] = sorted(set(failures))
    result["valid"] = not result["failure_reasons"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--requested-consumer", required=True)
    parser.add_argument("--resolved-consumer", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    args = parser.parse_args(argv)
    result = resolve_density_contract(
        args.config,
        requested_consumer=args.requested_consumer,
        resolved_consumer=args.resolved_consumer,
        expected_human_count=args.expected_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
