import json
import subprocess
from pathlib import Path

import pytest
import yaml

from arena_bringup.container_source_gate import (
    DEFAULT_CONTAINER_PATH,
    _collect_host_git_identity,
    _atomic_new_json,
    _run_checked,
    collect_live_container_record,
    validate_container_source_binding,
)


TARGET = "/reviewed/worktree"
HEAD = "1" * 40
ISAAC = "2" * 40
FILES = {
    "arena_isaac/arena_isaac/arena_isaac/run_isaacsim.py": "a" * 64,
    "arena_isaac/arena_isaac/arena_isaac/services/SpawnUsdRobot.py": "b" * 64,
    "arena_isaac/arena_isaac/isaac_utils/camera_publish_backend.py": "c" * 64,
    "arena_isaac/arena_isaac/isaac_utils/camera_dynamic_tf.py": "d" * 64,
    "arena_isaac/arena_isaac/isaac_utils/camera_runtime_registry.py": "e" * 64,
}


def _contract():
    return {
        "host_source_root": TARGET,
        "container_path": DEFAULT_CONTAINER_PATH,
        "superproject_head": HEAD,
        "gitlinks": {"arena_isaac": ISAAC},
        "files": FILES,
        "required_roles": ["arena", "isaac"],
    }


def _record():
    return {
        "path_exists": True,
        "mount": {
            "source": TARGET,
            "destination": DEFAULT_CONTAINER_PATH,
            "rw": False,
            "type": "bind",
        },
        "host_git_identity": {
            "source_root": TARGET,
            "superproject_head": HEAD,
            "gitlinks": {"arena_isaac": ISAAC},
            "submodule_heads": {"arena_isaac": ISAAC},
            "submodule_status_porcelain": {"arena_isaac": ""},
            "status_porcelain": "",
        },
        "files": dict(FILES),
        "collection_errors": [],
    }


def _evidence():
    return {"containers": {"arena": _record(), "isaac": _record()}}


def _failure_keys(result):
    return {item["check"] for item in result["failures"]}


def test_target_worktree_exact_read_only_mount_passes():
    result = validate_container_source_binding(_contract(), _evidence())
    assert result["pass"] is True
    assert result["failures"] == []


def test_old_main_checkout_mount_fails_closed():
    evidence = _evidence()
    evidence["containers"]["isaac"]["mount"]["source"] = "/workspace/src/Arena"
    result = validate_container_source_binding(_contract(), evidence)
    assert result["pass"] is False
    assert "container:isaac:mount_source" in _failure_keys(result)


def test_container_path_missing_fails_before_runtime():
    evidence = _evidence()
    evidence["containers"]["arena"]["path_exists"] = False
    result = validate_container_source_binding(_contract(), evidence)
    assert result["pass"] is False
    assert "container:arena:path_exists" in _failure_keys(result)


def test_sha_drift_fails_closed():
    evidence = _evidence()
    path = next(iter(FILES))
    evidence["containers"]["arena"]["files"][path] = "f" * 64
    result = validate_container_source_binding(_contract(), evidence)
    assert result["pass"] is False
    assert f"container:arena:file:{path}" in _failure_keys(result)
    assert "cross_container:arena:isaac:identity" in _failure_keys(result)


def test_gitlink_drift_fails_closed():
    evidence = _evidence()
    evidence["containers"]["isaac"]["host_git_identity"]["gitlinks"][
        "arena_isaac"
    ] = "3" * 40
    result = validate_container_source_binding(_contract(), evidence)
    assert result["pass"] is False
    assert "container:isaac:gitlink:arena_isaac" in _failure_keys(result)


def test_unpopulated_submodule_fails_closed():
    evidence = _evidence()
    evidence["containers"]["isaac"]["host_git_identity"]["submodule_heads"][
        "arena_isaac"
    ] = ""
    result = validate_container_source_binding(_contract(), evidence)
    assert result["pass"] is False
    assert "container:isaac:populated_submodule:arena_isaac" in _failure_keys(result)


def test_any_collection_error_fails_even_when_other_fields_match():
    evidence = _evidence()
    evidence["containers"]["arena"]["collection_errors"] = [
        "file:RuntimeError:unreadable"
    ]
    result = validate_container_source_binding(_contract(), evidence)
    assert result["pass"] is False
    assert "container:arena:collection_errors_empty" in _failure_keys(result)


def _git(args, cwd):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _real_populated_git_tree(tmp_path):
    root = tmp_path / "super"
    root.mkdir()
    _git(["init", "-q"], root)
    _git(["config", "user.name", "fixture"], root)
    _git(["config", "user.email", "fixture@example.invalid"], root)
    (root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "runtime.py"], root)
    _git(["commit", "-qm", "root"], root)

    submodule = root / "arena_isaac"
    submodule.mkdir()
    _git(["init", "-q"], submodule)
    _git(["config", "user.name", "fixture"], submodule)
    _git(["config", "user.email", "fixture@example.invalid"], submodule)
    (submodule / "runtime_module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(["add", "runtime_module.py"], submodule)
    _git(["commit", "-qm", "submodule"], submodule)
    submodule_head = _git(["rev-parse", "HEAD"], submodule)
    _git(
        ["update-index", "--add", "--cacheinfo", f"160000,{submodule_head},arena_isaac"],
        root,
    )
    _git(["commit", "-qm", "gitlink"], root)
    return root, submodule_head


def _collect_real_identity(root):
    errors = []

    def optional(argv, label):
        try:
            return _run_checked(argv)
        except Exception as exc:
            errors.append(f"{label}:{type(exc).__name__}:{exc}")
            return ""

    value = _collect_host_git_identity(
        source_root=str(root), gitlinks=["arena_isaac"], optional=optional
    )
    assert errors == []
    return value


def test_real_git_super_untracked_runtime_source_is_not_clean(tmp_path):
    root, _head = _real_populated_git_tree(tmp_path)
    (root / "untracked_runtime_module.py").write_text("VALUE = 3\n", encoding="utf-8")
    identity = _collect_real_identity(root)
    assert "?? untracked_runtime_module.py" in identity["status_porcelain"]
    evidence = _evidence()
    for record in evidence["containers"].values():
        record["mount"]["source"] = str(root)
        record["host_git_identity"] = identity
    contract = _contract()
    contract["host_source_root"] = str(root)
    contract["superproject_head"] = identity["superproject_head"]
    contract["gitlinks"] = identity["gitlinks"]
    result = validate_container_source_binding(contract, evidence)
    assert result["pass"] is False
    assert "container:arena:host_worktree_clean" in _failure_keys(result)


def test_real_git_submodule_untracked_config_is_not_clean(tmp_path):
    root, _head = _real_populated_git_tree(tmp_path)
    (root / "arena_isaac/untracked_runtime_config.yaml").write_text(
        "enabled: true\n", encoding="utf-8"
    )
    identity = _collect_real_identity(root)
    assert "?? untracked_runtime_config.yaml" in identity[
        "submodule_status_porcelain"
    ]["arena_isaac"]
    evidence = _evidence()
    for record in evidence["containers"].values():
        record["mount"]["source"] = str(root)
        record["host_git_identity"] = identity
    contract = _contract()
    contract["host_source_root"] = str(root)
    contract["superproject_head"] = identity["superproject_head"]
    contract["gitlinks"] = identity["gitlinks"]
    result = validate_container_source_binding(contract, evidence)
    assert result["pass"] is False
    assert "container:arena:submodule_clean:arena_isaac" in _failure_keys(result)


@pytest.mark.parametrize(
    "missing",
    [
        "arena_isaac/arena_isaac/isaac_utils/camera_publish_backend.py",
        "arena_isaac/arena_isaac/isaac_utils/camera_dynamic_tf.py",
        "arena_isaac/arena_isaac/isaac_utils/camera_runtime_registry.py",
    ],
)
def test_actual_camera_backend_or_manifest_module_missing_fails_closed(missing):
    evidence = _evidence()
    evidence["containers"]["isaac"]["files"].pop(missing)
    result = validate_container_source_binding(_contract(), evidence)
    assert result["pass"] is False
    assert f"container:isaac:file:{missing}" in _failure_keys(result)


def test_writable_mount_is_not_accepted_as_reviewed_source():
    evidence = _evidence()
    evidence["containers"]["arena"]["mount"]["rw"] = True
    result = validate_container_source_binding(_contract(), evidence)
    assert result["pass"] is False
    assert "container:arena:mount_read_only" in _failure_keys(result)


def test_atomic_evidence_is_new_and_cannot_overwrite(tmp_path):
    output = (tmp_path / "gate.json").resolve()
    _atomic_new_json(output, {"pass": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"pass": True}
    with pytest.raises(FileExistsError):
        _atomic_new_json(output, {"pass": False})
    assert json.loads(output.read_text(encoding="utf-8")) == {"pass": True}


def test_live_collector_uses_exact_mount_and_container_paths():
    calls = []

    def runner(argv):
        calls.append(list(argv))
        if argv[:2] == ["docker", "inspect"]:
            return json.dumps(
                [
                    {
                        "Mounts": [
                            {
                                "Source": TARGET,
                                "Destination": DEFAULT_CONTAINER_PATH,
                                "RW": False,
                                "Type": "bind",
                            }
                        ]
                    }
                ]
            )
        if argv == ["git", "-C", TARGET, "rev-parse", "HEAD"]:
            return HEAD + "\n"
        if argv == [
            "git",
            "-C",
            TARGET,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ]:
            return ""
        if argv == ["git", "-C", TARGET, "ls-tree", "HEAD", "arena_isaac"]:
            return f"160000 commit {ISAAC}\tarena_isaac\n"
        if argv == ["git", "-C", f"{TARGET}/arena_isaac", "rev-parse", "HEAD"]:
            return ISAAC + "\n"
        if argv == [
            "git",
            "-C",
            f"{TARGET}/arena_isaac",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ]:
            return ""
        if "sha256sum" in argv:
            relative = argv[-1].split(DEFAULT_CONTAINER_PATH + "/", 1)[1]
            return f"{FILES[relative]}  {argv[-1]}\n"
        if argv[-3:] == ["test", "-d", DEFAULT_CONTAINER_PATH]:
            return ""
        raise AssertionError(argv)

    record = collect_live_container_record(
        container="runtime-isaac",
        container_path=DEFAULT_CONTAINER_PATH,
        file_paths=list(FILES),
        gitlinks=["arena_isaac"],
        runner=runner,
    )
    assert record["mount"]["source"] == TARGET
    assert record["mount"]["rw"] is False
    assert record["host_git_identity"] == {
        "source_root": TARGET,
        "superproject_head": HEAD,
        "gitlinks": {"arena_isaac": ISAAC},
        "submodule_heads": {"arena_isaac": ISAAC},
        "submodule_status_porcelain": {"arena_isaac": ""},
        "status_porcelain": "",
    }
    assert record["files"] == FILES
    assert ["docker", "inspect", "runtime-isaac"] in calls


def test_tracked_compose_override_binds_both_runtime_roles_read_only():
    root = Path(__file__).resolve().parents[2]
    compose = yaml.safe_load(
        (root / "_meta/docker/features/benchmark/docker-compose.source-binding.yaml")
        .read_text(encoding="utf-8")
    )
    assert set(compose["services"]) == {"arena", "isaac"}
    for role in ("arena", "isaac"):
        volumes = compose["services"][role]["volumes"]
        assert volumes == [
            {
                "type": "bind",
                "source": "${ARENA_BENCHMARK_SOURCE_ROOT:?set ARENA_BENCHMARK_SOURCE_ROOT to the reviewed superproject worktree}",
                "target": DEFAULT_CONTAINER_PATH,
                "read_only": True,
            }
        ]


def test_real_docker_fixture_uses_production_collector_network_none_and_no_gpu():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/container_source_gate_docker_fixture.py").read_text(
        encoding="utf-8"
    )
    assert "collect_live_evidence(live_contract)" in source
    assert "validate_container_source_binding(live_contract, evidence)" in source
    assert '"--network",\n                    "none"' in source
    assert '"--gpus"' not in source
    assert '"device_requests": device_requests' in source
    assert '"network_none_and_zero_gpu": constraints_pass' in source
    assert '"--inject-untracked-relative"' in source
    assert '"--expect-gate-pass"' in source
    assert 'injected_path.unlink()' in source
