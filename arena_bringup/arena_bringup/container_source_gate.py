"""Fail-closed Gate0 for the source tree actually mounted in runtime containers.

The benchmark may be launched from a reviewed Git worktree while Docker still
bind-mounts a different checkout at ``/opt/arena_ws/src/Arena``.  This module
proves one chain without assuming that a linked-worktree ``.git`` absolute path
is visible inside the container: Docker supplies the exact bind source, host
Git supplies the commit/gitlink for that source, and the container supplies the
bytes it will import.  All three must match one frozen contract before Isaac,
ROS or a model is started.

The validator is intentionally pure; fixtures can exercise every failure mode
without Docker.  ``collect_live_evidence`` is the small read-only Docker
adapter used by a future Gate0 package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "arena.container_source_binding_gate.v1"
DEFAULT_CONTAINER_PATH = "/opt/arena_ws/src/Arena"
REQUIRED_ROLES = ("arena", "isaac")

class SourceBindingError(RuntimeError):
    """Raised when live source identity cannot be collected exactly."""


def _normalized_path(value: str) -> str:
    path = os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))
    if not os.path.isabs(path):
        raise ValueError(f"source path must be absolute: {value!r}")
    return path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a new evidence file atomically; never overwrite prior evidence."""

    path = Path(path)
    if not path.is_absolute():
        raise ValueError("source-binding evidence path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_value = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _expected_file_map(contract: Mapping[str, Any]) -> dict[str, str]:
    files = contract.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("source-binding contract requires a non-empty files map")
    result: dict[str, str] = {}
    for relative, digest in files.items():
        relative_value = str(relative).strip().lstrip("/")
        digest_value = str(digest).strip().lower()
        if not relative_value or ".." in Path(relative_value).parts:
            raise ValueError(f"invalid relative source path: {relative!r}")
        if len(digest_value) != 64 or any(c not in "0123456789abcdef" for c in digest_value):
            raise ValueError(f"invalid SHA256 for {relative_value}: {digest!r}")
        result[relative_value] = digest_value
    return result


def validate_container_source_binding(
    contract: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare collected container identity with a frozen source contract."""

    expected_root = _normalized_path(str(contract["host_source_root"]))
    expected_container_path = str(
        contract.get("container_path", DEFAULT_CONTAINER_PATH)
    ).rstrip("/")
    expected_head = str(contract["superproject_head"]).strip()
    expected_gitlinks = {
        str(key): str(value).strip()
        for key, value in dict(contract.get("gitlinks", {})).items()
    }
    expected_files = _expected_file_map(contract)
    required_roles = tuple(contract.get("required_roles", REQUIRED_ROLES))
    failures: list[dict[str, str]] = []
    checks: dict[str, bool] = {}

    containers = evidence.get("containers")
    if not isinstance(containers, Mapping):
        containers = {}

    def check(key: str, condition: bool, detail: str) -> None:
        checks[key] = bool(condition)
        if not condition:
            failures.append({"check": key, "detail": detail})

    top_collection_errors = evidence.get("collection_errors")
    if not isinstance(top_collection_errors, list):
        top_collection_errors = []
    check(
        "collection_errors_empty",
        not top_collection_errors,
        f"source evidence collection failed: {top_collection_errors!r}",
    )

    for role in required_roles:
        role_value = str(role)
        record = containers.get(role_value)
        check(
            f"container:{role_value}:present",
            isinstance(record, Mapping),
            f"required container evidence is missing for role {role_value}",
        )
        if not isinstance(record, Mapping):
            continue

        collection_errors = record.get("collection_errors")
        if not isinstance(collection_errors, list):
            collection_errors = []
        check(
            f"container:{role_value}:collection_errors_empty",
            not collection_errors,
            f"{role_value} collection errors: {collection_errors!r}",
        )

        check(
            f"container:{role_value}:path_exists",
            record.get("path_exists") is True,
            f"{role_value}:{expected_container_path} is absent or not a directory",
        )
        mount = record.get("mount")
        check(
            f"container:{role_value}:exact_mount",
            isinstance(mount, Mapping),
            f"{role_value} has no exact bind mount at {expected_container_path}",
        )
        if isinstance(mount, Mapping):
            actual_source = _normalized_path(str(mount.get("source", "")))
            check(
                f"container:{role_value}:mount_source",
                actual_source == expected_root,
                f"{role_value} mount source {actual_source!r} != {expected_root!r}",
            )
            check(
                f"container:{role_value}:mount_destination",
                str(mount.get("destination", "")).rstrip("/")
                == expected_container_path,
                f"{role_value} mount destination is not {expected_container_path}",
            )
            check(
                f"container:{role_value}:mount_read_only",
                mount.get("rw") is False,
                f"{role_value} source bind must be read-only",
            )

        host_identity = record.get("host_git_identity")
        check(
            f"container:{role_value}:host_git_identity",
            isinstance(host_identity, Mapping),
            f"{role_value} has no host Git identity for its exact mount source",
        )
        if not isinstance(host_identity, Mapping):
            host_identity = {}
        check(
            f"container:{role_value}:superproject_head",
            str(host_identity.get("superproject_head", "")) == expected_head,
            f"{role_value} superproject HEAD drifted",
        )
        actual_host_root = str(host_identity.get("source_root", ""))
        check(
            f"container:{role_value}:host_source_root",
            bool(actual_host_root)
            and _normalized_path(actual_host_root) == expected_root,
            f"{role_value} host Git identity is not for the exact expected source",
        )
        check(
            f"container:{role_value}:host_worktree_clean",
            str(host_identity.get("status_porcelain", "")) == "",
            f"{role_value} exact mounted worktree is dirty",
        )
        actual_gitlinks = host_identity.get("gitlinks")
        if not isinstance(actual_gitlinks, Mapping):
            actual_gitlinks = {}
        actual_submodule_heads = host_identity.get("submodule_heads")
        if not isinstance(actual_submodule_heads, Mapping):
            actual_submodule_heads = {}
        actual_submodule_status = host_identity.get("submodule_status_porcelain")
        if not isinstance(actual_submodule_status, Mapping):
            actual_submodule_status = {}
        for name, expected_value in expected_gitlinks.items():
            check(
                f"container:{role_value}:gitlink:{name}",
                str(actual_gitlinks.get(name, "")) == expected_value,
                f"{role_value} gitlink {name} drifted or is missing",
            )
            check(
                f"container:{role_value}:populated_submodule:{name}",
                str(actual_submodule_heads.get(name, "")) == expected_value,
                f"{role_value} submodule {name} is absent or not at its gitlink",
            )
            check(
                f"container:{role_value}:submodule_clean:{name}",
                str(actual_submodule_status.get(name, "")) == "",
                f"{role_value} submodule {name} has tracked or untracked changes",
            )

        actual_files = record.get("files")
        if not isinstance(actual_files, Mapping):
            actual_files = {}
        for relative, expected_digest in expected_files.items():
            actual_digest = actual_files.get(relative)
            check(
                f"container:{role_value}:file:{relative}",
                str(actual_digest or "").lower() == expected_digest,
                f"{role_value} file {relative} is absent or SHA256 drifted",
            )

    # Both runtime containers must expose byte-identical source, not merely
    # match separate host-side expectations.
    if all(isinstance(containers.get(role), Mapping) for role in required_roles):
        first = containers[str(required_roles[0])]
        for role in required_roles[1:]:
            current = containers[str(role)]
            check(
                f"cross_container:{required_roles[0]}:{role}:identity",
                (
                    first.get("mount") == current.get("mount")
                    and first.get("host_git_identity")
                    == current.get("host_git_identity")
                    and first.get("files") == current.get("files")
                ),
                f"{required_roles[0]} and {role} do not expose identical source identity",
            )

    return {
        "schema": SCHEMA,
        "identity_scheme": "docker_exact_mount+host_git_identity+container_file_sha256",
        "pass": not failures,
        "checks": checks,
        "failures": failures,
        "expected": {
            "host_source_root": expected_root,
            "container_path": expected_container_path,
            "superproject_head": expected_head,
            "gitlinks": expected_gitlinks,
            "files": expected_files,
            "required_roles": list(required_roles),
        },
        "evidence": dict(evidence),
    }


def _run_checked(argv: Sequence[str]) -> str:
    completed = subprocess.run(
        list(argv),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise SourceBindingError(
            f"command failed rc={completed.returncode}: {list(argv)!r}; "
            f"stderr={completed.stderr.strip()!r}"
        )
    return completed.stdout


def _collect_host_git_identity(
    *,
    source_root: str,
    gitlinks: Sequence[str],
    optional: Callable[[Sequence[str], str], str],
) -> dict[str, Any]:
    """Bind host Git identity to the source returned by Docker inspect."""

    root = _normalized_path(source_root)
    head = optional(
        ["git", "-C", root, "rev-parse", "HEAD"], "host_superproject_head"
    ).strip()
    status = optional(
        [
            "git",
            "-C",
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        "host_worktree_status",
    ).strip()
    actual_gitlinks: dict[str, str] = {}
    submodule_heads: dict[str, str] = {}
    submodule_status: dict[str, str] = {}
    for name in gitlinks:
        line = optional(
            ["git", "-C", root, "ls-tree", "HEAD", str(name)],
            f"host_gitlink:{name}",
        ).strip()
        fields = line.split()
        actual_gitlinks[str(name)] = (
            fields[2] if len(fields) >= 4 and fields[1] == "commit" else ""
        )
        submodule_heads[str(name)] = optional(
            ["git", "-C", f"{root}/{name}", "rev-parse", "HEAD"],
            f"host_submodule_head:{name}",
        ).strip()
        submodule_status[str(name)] = optional(
            [
                "git",
                "-C",
                f"{root}/{name}",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            f"host_submodule_status:{name}",
        ).strip()
    return {
        "source_root": root,
        "superproject_head": head,
        "gitlinks": actual_gitlinks,
        "submodule_heads": submodule_heads,
        "submodule_status_porcelain": submodule_status,
        "status_porcelain": status,
    }


def collect_live_container_record(
    *,
    container: str,
    container_path: str,
    file_paths: Sequence[str],
    gitlinks: Sequence[str],
    runner: Callable[[Sequence[str]], str] = _run_checked,
) -> dict[str, Any]:
    """Collect one container's exact source identity using read-only commands."""

    inspect_payload = json.loads(runner(["docker", "inspect", container]))
    if not isinstance(inspect_payload, list) or len(inspect_payload) != 1:
        raise SourceBindingError(f"docker inspect returned unexpected data for {container}")
    mounts = inspect_payload[0].get("Mounts") or []
    exact_mounts = [
        item
        for item in mounts
        if str(item.get("Destination", "")).rstrip("/")
        == container_path.rstrip("/")
    ]
    mount = None
    if len(exact_mounts) == 1:
        item = exact_mounts[0]
        mount = {
            "source": str(item.get("Source", "")),
            "destination": str(item.get("Destination", "")),
            "rw": bool(item.get("RW", False)),
            "type": str(item.get("Type", "")),
        }
    elif len(exact_mounts) > 1:
        raise SourceBindingError(
            f"multiple exact mounts target {container}:{container_path}"
        )

    collection_errors: list[str] = []

    def optional(argv: Sequence[str], label: str) -> str:
        try:
            return runner(argv)
        except Exception as exc:
            collection_errors.append(f"{label}:{type(exc).__name__}:{exc}")
            return ""

    optional(
        ["docker", "exec", container, "test", "-d", container_path],
        "path_exists",
    )
    # ``test`` emits no stdout on success.  A failed command is converted to a
    # collection error by ``optional``; use that exact label to distinguish it.
    path_exists = not any(item.startswith("path_exists:") for item in collection_errors)
    host_identity = None
    if mount is not None and mount.get("source"):
        host_identity = _collect_host_git_identity(
            source_root=str(mount["source"]),
            gitlinks=gitlinks,
            optional=optional,
        )

    actual_files: dict[str, str] = {}
    for relative in file_paths:
        relative_value = str(relative).strip().lstrip("/")
        output = optional(
            [
                "docker",
                "exec",
                container,
                "sha256sum",
                f"{container_path.rstrip('/')}/{relative_value}",
            ],
            f"file:{relative_value}",
        ).strip()
        digest = output.split(maxsplit=1)[0] if output else ""
        actual_files[relative_value] = digest.lower() if len(digest) == 64 else ""

    return {
        "container": container,
        "path_exists": path_exists,
        "mount": mount,
        "host_git_identity": host_identity,
        "files": actual_files,
        "collection_errors": collection_errors,
    }


def collect_live_evidence(
    contract: Mapping[str, Any], *, runner: Callable[[Sequence[str]], str] = _run_checked
) -> dict[str, Any]:
    roles = contract.get("containers")
    if not isinstance(roles, Mapping):
        raise ValueError("live source-binding contract requires containers mapping")
    container_path = str(contract.get("container_path", DEFAULT_CONTAINER_PATH))
    files = list(_expected_file_map(contract))
    gitlinks = list(dict(contract.get("gitlinks", {})))
    return {
        "containers": {
            str(role): collect_live_container_record(
                container=str(container),
                container_path=container_path,
                file_paths=files,
                gitlinks=gitlinks,
                runner=runner,
            )
            for role, container in roles.items()
        }
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the source actually mounted inside Arena runtime containers"
    )
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if args.evidence is not None:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    else:
        try:
            evidence = collect_live_evidence(contract)
        except Exception as exc:
            evidence = {
                "containers": {},
                "collection_errors": [f"{type(exc).__name__}:{exc}"],
            }
    result = validate_container_source_binding(contract, evidence)
    _atomic_new_json(args.output, result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
