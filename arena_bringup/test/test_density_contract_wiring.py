from pathlib import Path


ARENA_BRINGUP = Path(__file__).resolve().parents[1]
REPO_ROOT = ARENA_BRINGUP.parent
INTERNNAV_EVAL = ARENA_BRINGUP / "arena_bringup" / "internnav_eval.py"
BENCHMARK_RESULT = ARENA_BRINGUP / "arena_bringup" / "benchmark_result.py"
BENCHMARK_ENTRY = REPO_ROOT / "_meta/docker/features/benchmark/main"
SETUP = ARENA_BRINGUP / "setup.py"


def test_internnav_rejects_density_conflicts_before_creating_output_or_launching():
    source = INTERNNAV_EVAL.read_text(encoding="utf-8")

    contract = source.index("density_contract = resolve_density_contract(")
    reject = source.index("density/task-generator contract rejected before launch")
    output = source.index("os.makedirs(snapshots_dir, exist_ok=True)")
    launch = source.index("launch_proc = subprocess.Popen(")
    assert contract < reject < output < launch


def test_source_provenance_can_bind_to_the_isolated_worktree():
    source = INTERNNAV_EVAL.read_text(encoding="utf-8")
    assert "ARENA_EVAL_SOURCE_ROOT" in source
    assert "explicit_source_root" in source


def test_manifest_and_canonical_record_resolved_density_contract():
    eval_source = INTERNNAV_EVAL.read_text(encoding="utf-8")
    canonical_source = BENCHMARK_RESULT.read_text(encoding="utf-8")

    assert eval_source.count("'density_contract': density_contract") >= 2
    assert "manifest.get('density_contract')" in canonical_source
    assert "parameters.get('density_contract')" in canonical_source


def test_host_wrapper_checks_contract_before_model_server_paths():
    source = BENCHMARK_ENTRY.read_text(encoding="utf-8")

    dispatch_check = source.index('if [ -n "$TASK_GENERATOR_PARAMETER_FILE" ]')
    model_dispatch = source.index('case "$VERB" in')
    assert dispatch_check < model_dispatch
    assert "rejected before containers, model, or Isaac" in source


def test_density_contract_cli_is_packaged():
    setup = SETUP.read_text(encoding="utf-8")
    assert "density_contract = arena_bringup.density_contract:main" in setup
