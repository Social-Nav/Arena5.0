"""Tests for the `benchmark` feature entry point.

`_meta/docker/features/benchmark/main` is the single top-level command that
brings up the three containers and runs an evaluation.  These tests exercise it
without Docker, without a GPU and without starting anything, by driving its
`plan` / `--dry-run` contract and its refusal paths.

The load-bearing property is that `plan` is *pure*: it must describe what it
would do without touching a container, an image or the daemon.  That is asserted
directly, by running `plan` with a PATH from which `docker` has been removed --
if any code path shelled out to docker, the command would fail.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]          # src/Arena
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]     # the umbrella workspace
SCRIPT = REPO_ROOT / '_meta' / 'docker' / 'features' / 'benchmark' / 'main'

# Verbs that `plan` must be able to describe.
PLANNABLE_VERBS = ['run', 'build', 'runtime', 'up', 'serve', 'eval', 'stop', 'down']

# The declared feature set, which must be visible in the script rather than
# inherited from the untracked src/Arena/.installed.
DECLARED_FEATURES = ['isaac', 'internnav']


def _run(args, env=None, cwd=None):
    """Invoke the script and capture everything."""
    base = dict(os.environ)
    # Pin identity so assertions do not depend on the developer's machine.
    base.setdefault('HOST_ARENA_WS_DIR', str(WORKSPACE_ROOT))
    base.setdefault('ARENA_PROJECT_NAME', 'arena-testproj')
    if env:
        base.update(env)
    return subprocess.run(
        ['bash', str(SCRIPT), *args],
        env=base,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _docker_free_env(tmp_path):
    """A PATH with the usual tools but no `docker` executable.

    Used to prove that plan mode never reaches the daemon.  Anything the script
    legitimately needs in plan mode (bash builtins, python3, coreutils) is
    symlinked in; `docker` deliberately is not.
    """
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    for tool in ('bash', 'python3', 'basename', 'dirname', 'cd', 'stat', 'id',
                 'sort', 'paste', 'grep', 'seq', 'printf', 'env', 'mkdir',
                 'setsid', 'sed', 'cat', 'head', 'tail'):
        found = shutil.which(tool)
        if found:
            target = bindir / tool
            if not target.exists():
                target.symlink_to(found)
    # A `docker` that fails loudly if anything calls it.
    sabotage = bindir / 'docker'
    sabotage.write_text('#!/bin/sh\necho "PLAN MODE CALLED DOCKER" >&2\nexit 99\n')
    sabotage.chmod(sabotage.stat().st_mode | stat.S_IEXEC)
    return {'PATH': str(bindir)}


# --------------------------------------------------------------------------
# Packaging / shape
# --------------------------------------------------------------------------

def test_script_exists_and_is_executable():
    assert SCRIPT.is_file(), f'missing entry point: {SCRIPT}'
    assert os.access(SCRIPT, os.X_OK), f'{SCRIPT} is not executable'


def test_script_is_syntactically_valid():
    result = subprocess.run(['bash', '-n', str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shebang_is_not_interactive():
    """An interactive shebang takes SIGTTIN when backgrounded.

    `run` backgrounds the model server, so this script must not use
    `#!/bin/bash -i` the way features/internnav/main does.
    """
    first_line = SCRIPT.read_text().splitlines()[0]
    assert first_line.startswith('#!'), first_line
    assert ' -i' not in first_line, (
        'interactive shebang would stop on SIGTTIN when backgrounded'
    )


def test_follows_upstream_feature_dispatch_shape():
    """Same structure as the upstream-owned features/docker/main."""
    text = SCRIPT.read_text()
    assert 'help()' in text
    assert 'case "$1" in' in text or 'case "$VERB" in' in text
    upstream = REPO_ROOT / '_meta' / 'docker' / 'features' / 'docker' / 'main'
    assert upstream.is_file(), 'reference feature script vanished'
    assert '*)' in text, 'dispatcher must have a catch-all that shows help'


# --------------------------------------------------------------------------
# Help text (the owner's actual goal: a newcomer can get started from it)
# --------------------------------------------------------------------------

def test_no_arguments_prints_help_and_fails():
    result = _run([])
    assert result.returncode == 1
    assert 'Usage:' in result.stdout


def test_help_exits_zero():
    result = _run(['--help'])
    assert result.returncode == 0
    assert 'Usage:' in result.stdout


@pytest.mark.parametrize('verb', PLANNABLE_VERBS + ['doctor', 'plan', 'status'])
def test_help_documents_every_verb(verb):
    out = _run(['help']).stdout
    assert verb in out, f'help text does not mention the {verb!r} verb'


def test_help_names_the_three_containers():
    out = _run(['help']).stdout
    for container in ('arena', 'isaac', 'internnav'):
        assert container in out


def test_help_states_the_cost_of_the_slow_paths():
    """Requirement: say what each verb costs before someone runs it."""
    out = _run(['help']).stdout
    assert 'hours' in out, 'help must warn that build can take hours'
    assert 'minutes' in out


def test_help_tells_a_newcomer_where_results_land():
    out = _run(['help']).stdout
    assert 'outputs/' in out


# --------------------------------------------------------------------------
# plan purity -- the property that makes this reviewable without a GPU
# --------------------------------------------------------------------------

@pytest.mark.parametrize('verb', PLANNABLE_VERBS)
def test_plan_never_invokes_docker(verb, tmp_path):
    result = _run(['plan', verb], env=_docker_free_env(tmp_path))
    assert 'PLAN MODE CALLED DOCKER' not in result.stderr, (
        f'plan {verb} reached the docker daemon'
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('verb', PLANNABLE_VERBS)
def test_plan_emits_numbered_steps(verb):
    out = _run(['plan', verb]).stdout
    assert '1.' in out, f'plan {verb} produced no steps:\n{out}'
    assert '$ ' in out, f'plan {verb} showed no command:\n{out}'


def test_plan_says_nothing_was_executed():
    out = _run(['plan', 'up']).stdout
    assert 'nothing above was executed' in out


def test_dry_run_flag_is_equivalent_to_plan():
    a = _run(['plan', 'up']).stdout
    b = _run(['up', '--dry-run']).stdout
    assert a == b


def test_plan_rejects_an_unknown_verb():
    result = _run(['plan', 'frobnicate'])
    assert result.returncode == 1
    assert 'frobnicate' in result.stderr


# --------------------------------------------------------------------------
# Ordering: the readiness dependency documented in running-benchmarks.md
# --------------------------------------------------------------------------

def test_run_plan_orders_containers_then_model_then_eval():
    out = _run(['plan', 'run']).stdout
    i_up = out.find('start containers')
    i_serve = out.find('start the model server')
    i_ready = out.find('report ready')
    i_eval = out.find('run the evaluation')
    assert -1 not in (i_up, i_serve, i_ready, i_eval), out
    assert i_up < i_serve < i_ready < i_eval, (
        'the model must be serving and ready before the evaluation starts'
    )


def test_run_plan_checks_preconditions_first():
    out = _run(['plan', 'run']).stdout
    assert 'precondition' in out
    assert out.find('precondition') < out.find('start containers')


def test_run_plan_stops_the_model_server_afterwards():
    out = _run(['plan', 'run']).stdout
    assert out.find('stop the model server') > out.find('run the evaluation')


def test_server_cleanup_is_a_trap_not_a_trailing_statement():
    """A failed evaluation must still stop the GPU model server.

    run_step calls die(), which exits, so cleanup placed after the eval step
    would be skipped on failure and leave a model server holding GPU memory.
    """
    text = SCRIPT.read_text()
    assert 'trap benchmark_stop_server EXIT INT TERM' in text, (
        'model server cleanup must be trap-registered'
    )
    out = _run(['plan', 'run']).stdout
    assert 'trap' in out, 'the plan should disclose that cleanup is trapped'


def test_run_plan_backgrounds_the_server_without_a_controlling_terminal():
    """features/internnav/main is interactive; backgrounding it needs setsid."""
    out = _run(['plan', 'run']).stdout
    assert 'setsid' in out
    assert '</dev/null' in out


# --------------------------------------------------------------------------
# Destructive verbs must be opt-in
# --------------------------------------------------------------------------

@pytest.mark.parametrize('verb', ['build', 'down'])
def test_destructive_verb_refuses_without_yes(verb, tmp_path):
    result = _run([verb], env=_docker_free_env(tmp_path))
    assert result.returncode == 1, f'{verb} ran without confirmation'
    assert 'PLAN MODE CALLED DOCKER' not in result.stderr, (
        f'{verb} touched docker before refusing'
    )
    assert '--yes' in result.stderr


def test_build_is_not_reachable_from_up():
    """`up` must never build.

    Checked behaviourally, on the commands the plan would run, rather than by
    grepping for the word 'build' -- the plan legitimately contains the phrases
    'without building' and '/.built marker', and a substring test matches those.
    """
    commands = [
        line.strip()[2:] for line in _run(['plan', 'up']).stdout.splitlines()
        if line.strip().startswith('$ ')
    ]
    assert commands, 'plan up emitted no commands'
    for cmd in commands:
        assert 'compose build' not in cmd, f'up would build: {cmd}'
        assert 'main install' not in cmd, f'up would provision: {cmd}'


def test_up_passes_no_build_explicitly():
    out = _run(['plan', 'up']).stdout
    assert '--no-build' in out


def test_run_does_not_build():
    out = _run(['plan', 'run']).stdout
    assert 'compose build' not in out


# --------------------------------------------------------------------------
# The DDS contract, asserted rather than hoped for
# --------------------------------------------------------------------------

def test_eval_forwards_the_udpv4_transport():
    """Each container has a private /dev/shm; the default SHM locator fails."""
    out = _run(['plan', 'eval']).stdout
    assert 'FASTDDS_BUILTIN_TRANSPORTS=UDPv4' in out


def test_eval_forwards_the_rmw_and_domain():
    out = _run(['plan', 'eval']).stdout
    assert 'RMW_IMPLEMENTATION=rmw_fastrtps_cpp' in out
    assert 'ROS_DOMAIN_ID=1' in out


def test_doctor_fails_when_the_transport_is_wrong(tmp_path):
    result = _run(['doctor'], env={'FASTDDS_BUILTIN_TRANSPORTS': 'SHM'})
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert 'FASTDDS_BUILTIN_TRANSPORTS' in combined
    assert 'FAIL' in combined


# --------------------------------------------------------------------------
# The evaluation contract itself
# --------------------------------------------------------------------------

def test_eval_selects_the_async_direct_cmdvel_launch_path():
    """--internnav-direct-cmd-vel is what makes the generated launch command
    use internnav_async_eval.launch.py. Losing it silently changes the run."""
    out = _run(['plan', 'eval']).stdout
    assert '--internnav-direct-cmd-vel' in out


def test_eval_calls_the_existing_driver_rather_than_reimplementing_it():
    out = _run(['plan', 'eval']).stdout
    assert 'ros2 run arena_bringup internnav_eval' in out


def test_eval_requests_video_artifacts():
    out = _run(['plan', 'eval']).stdout
    assert '--save-eval-video' in out


def test_eval_runs_in_the_arena_container():
    out = _run(['plan', 'eval'], env={'ARENA_PROJECT_NAME': 'arena-testproj'}).stdout
    assert 'arena-testproj-arena-1' in out


def test_extra_arguments_are_appended_to_the_driver():
    out = _run(['plan', 'eval', '--', '--scenario-file', 'foo.yaml']).stdout
    assert '--scenario-file foo.yaml' in out


@pytest.mark.parametrize(
    'flag,value,expected',
    [
        ('--robot', 'Jackal', 'Jackal'),
        ('--world', 'grscenes_20_v1', 'grscenes_20_v1'),
        ('--episodes', '7', '--episodes 7'),
        ('--timeout', '900', '--timeout 900'),
        ('--device', 'cuda:1', 'cuda:1'),
    ],
)
def test_options_reach_the_eval_command(flag, value, expected):
    out = _run(['plan', 'eval', flag, value]).stdout
    assert expected in out


def test_unknown_option_is_rejected_loudly():
    result = _run(['up', '--nonsense'])
    assert result.returncode == 1
    assert '--nonsense' in result.stderr


# --------------------------------------------------------------------------
# Declared topology, not inherited machine state
# --------------------------------------------------------------------------

def test_feature_set_is_declared_in_the_script():
    text = SCRIPT.read_text()
    assert 'BENCHMARK_FEATURES' in text
    for feature in DECLARED_FEATURES:
        assert feature in text


def test_plan_does_not_depend_on_the_untracked_installed_file(tmp_path):
    """src/Arena/.installed is untracked, so plan must not need it."""
    fake_ws = tmp_path / 'ws'
    (fake_ws / 'src' / 'Arena' / '_meta' / 'docker').mkdir(parents=True)
    result = _run(['plan', 'up'], env={'HOST_ARENA_WS_DIR': str(fake_ws)})
    assert result.returncode == 0, result.stderr
    assert 'nothing above was executed' in result.stdout


def test_doctor_fails_loudly_when_installed_is_missing(tmp_path):
    fake_ws = tmp_path / 'ws'
    (fake_ws / 'src' / 'Arena' / '_meta' / 'docker').mkdir(parents=True)
    result = _run(['doctor'], env={'HOST_ARENA_WS_DIR': str(fake_ws)})
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert '.installed' in combined
    # A failure has to say how to fix itself.
    assert 'fix:' in combined


def test_doctor_fails_loudly_on_a_feature_set_mismatch(tmp_path):
    fake_ws = tmp_path / 'ws'
    arena = fake_ws / 'src' / 'Arena'
    (arena / '_meta' / 'docker' / 'features' / 'isaac').mkdir(parents=True)
    (arena / '_meta' / 'docker' / 'features' / 'internnav').mkdir(parents=True)
    (arena / '_meta' / 'docker' / 'docker-compose.yaml').write_text('services: {}\n')
    (arena / '.installed').write_text('isaac\n')  # internnav missing
    result = _run(['doctor'], env={'HOST_ARENA_WS_DIR': str(fake_ws)})
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert 'does not match' in combined
    assert 'internnav' in combined


def test_doctor_changes_nothing(tmp_path):
    """doctor is a diagnosis, not a repair."""
    fake_ws = tmp_path / 'ws'
    arena = fake_ws / 'src' / 'Arena'
    (arena / '_meta' / 'docker').mkdir(parents=True)
    before = sorted(p.relative_to(fake_ws).as_posix() for p in fake_ws.rglob('*'))
    _run(['doctor'], env={'HOST_ARENA_WS_DIR': str(fake_ws)})
    after = sorted(p.relative_to(fake_ws).as_posix() for p in fake_ws.rglob('*'))
    assert before == after, 'doctor created or removed files'


def test_doctor_reports_every_check_it_ran(tmp_path):
    """No silent skipping: each check prints PASS or FAIL."""
    result = _run(['doctor'], env={'HOST_ARENA_WS_DIR': str(tmp_path)})
    combined = result.stdout + result.stderr
    n_reported = combined.count('[PASS]') + combined.count('[FAIL]')
    assert n_reported >= 10, f'only {n_reported} checks reported:\n{combined}'


def test_doctor_summarises_the_failure_count(tmp_path):
    result = _run(['doctor'], env={'HOST_ARENA_WS_DIR': str(tmp_path)})
    assert 'precondition(s) failed' in result.stderr


# --------------------------------------------------------------------------
# Wrapping, not replacing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    'verb,delegate',
    [
        ('build', 'features/internnav/main'),
        ('runtime', 'features/internnav/main'),
        ('serve', 'features/internnav/main'),
    ],
)
def test_verbs_delegate_to_the_existing_feature_script(verb, delegate):
    out = _run(['plan', verb]).stdout
    assert delegate in out, f'{verb} should call the existing {delegate}'


def test_the_delegate_scripts_exist():
    for rel in ['_meta/docker/features/internnav/main',
                '_meta/docker/features/docker/main',
                '_meta/docker/source']:
        assert (REPO_ROOT / rel).is_file(), f'missing dependency: {rel}'


def test_entry_point_adds_no_lines_to_existing_files():
    """This lane wraps; it must not have edited the machinery it calls."""
    result = subprocess.run(
        ['git', 'diff', '--name-only', 'HEAD', '--',
         '_meta/docker/features/internnav/main',
         '_meta/docker/features/docker/main',
         '_meta/docker/source'],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.stdout.strip() == '', (
        f'existing machinery was modified: {result.stdout}'
    )
