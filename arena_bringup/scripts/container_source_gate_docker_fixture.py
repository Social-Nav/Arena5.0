#!/usr/bin/env python3
"""Run the production source collector in two network-none, zero-GPU containers."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from pathlib import Path

from arena_bringup.container_source_gate import (
    _atomic_new_json,
    collect_live_evidence,
    validate_container_source_binding,
)


def run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {argv!r}; "
            f"stderr={completed.stderr.strip()!r}"
        )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", default="arena:dev")
    parser.add_argument(
        "--inject-untracked-relative",
        help="create one fixture-only untracked file under the mounted source",
    )
    parser.add_argument(
        "--expect-gate-pass",
        choices=("true", "false"),
        default="true",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if source_root != Path(contract["host_source_root"]).resolve():
        raise ValueError("fixture source root does not match frozen contract")
    if not (source_root / "arena_isaac/.git").exists():
        raise ValueError("fixture requires a populated arena_isaac submodule")

    token = uuid.uuid4().hex[:12]
    names = {
        "arena": f"arena-source-gate-fixture-arena-{token}",
        "isaac": f"arena-source-gate-fixture-isaac-{token}",
    }
    started: list[str] = []
    injected_path = None
    try:
        if args.inject_untracked_relative:
            relative = Path(args.inject_untracked_relative)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("fixture untracked path must stay under source root")
            injected_path = source_root / relative
            if injected_path.exists():
                raise FileExistsError(
                    f"fixture refuses to overwrite existing path: {injected_path}"
                )
            injected_path.parent.mkdir(parents=True, exist_ok=True)
            injected_path.write_text("source-gate-untracked-fixture\n", encoding="utf-8")
        for name in names.values():
            run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    name,
                    "--network",
                    "none",
                    "--mount",
                    (
                        f"type=bind,src={source_root},"
                        "dst=/opt/arena_ws/src/Arena,readonly"
                    ),
                    args.image,
                    "sleep",
                    "300",
                ]
            )
            started.append(name)

        live_contract = dict(contract)
        live_contract["containers"] = names
        evidence = collect_live_evidence(live_contract)
        result = validate_container_source_binding(live_contract, evidence)
        container_constraints = {}
        for role, name in names.items():
            inspect = json.loads(run(["docker", "inspect", name]).stdout)[0]
            device_requests = inspect.get("HostConfig", {}).get("DeviceRequests")
            network_mode = inspect.get("HostConfig", {}).get("NetworkMode")
            container_constraints[role] = {
                "container": name,
                "image": inspect.get("Image"),
                "network_mode": network_mode,
                "device_requests": device_requests,
                "zero_gpu": not device_requests,
            }
        constraints_pass = all(
            item["network_mode"] == "none" and item["zero_gpu"]
            for item in container_constraints.values()
        )
        result["fixture"] = {
            "schema": "arena.container_source_binding_real_docker_fixture.v1",
            "image_ref": args.image,
            "source_root": str(source_root),
            "container_constraints": container_constraints,
            "network_none_and_zero_gpu": constraints_pass,
            "expected_gate_pass": args.expect_gate_pass == "true",
            "injected_untracked_relative": args.inject_untracked_relative,
            "model_started": False,
            "isaac_started": False,
            "ros_started": False,
        }
        expected_gate_pass = args.expect_gate_pass == "true"
        expectation_met = result["pass"] is expected_gate_pass
        result["fixture"]["expectation_met"] = expectation_met
        result["fixture"]["fixture_pass"] = bool(
            constraints_pass and expectation_met
        )
        _atomic_new_json(args.output.resolve(), result)
        return 0 if result["fixture"]["fixture_pass"] else 1
    finally:
        cleanup_errors = []
        for name in reversed(started):
            completed = run(["docker", "rm", "-f", name], check=False)
            if completed.returncode:
                cleanup_errors.append(
                    f"{name}:rc={completed.returncode}:{completed.stderr.strip()}"
                )
        if injected_path is not None:
            try:
                injected_path.unlink()
            except Exception as exc:
                cleanup_errors.append(
                    f"{injected_path}:unlink:{type(exc).__name__}:{exc}"
                )
        if cleanup_errors:
            raise RuntimeError(f"fixture cleanup failed: {cleanup_errors!r}")


if __name__ == "__main__":
    raise SystemExit(main())
