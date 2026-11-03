# Selective FollowPath integration handoff

This checklist describes how to integrate
`integration/vln-follow-path-selective-20261013` toward
`feat/internnav-eval-progress`. It is an integration plan, not authorization to
merge or push. Never merge or cherry-pick `origin/skz/vln-integration` itself.

## Captured refs and scope

- Fixed target and current local/remote target at capture time:
  `c275eb429dcce7a2220cc670add4e1b27dcbbe15`.
- Selective source inspected, but intentionally not made an ancestor:
  `origin/skz/vln-integration@b750120ad05ae2b7f9397bbf62ca184be90bb498`.
- Pre-handoff Lane M implementation tip:
  `8b2eb174044f7114364f2ea67bc8c43eb3d137e8`.
- This checklist and the ament-python discovery fix are in the later Lane M
  commit containing this file. Resolve that commit with:

  ```bash
  git log -1 --format=%H -- docs/vln_follow_path_selective_integration_handoff.md
  ```

The implementation is selective at file and hunk level. It preserves the
existing direct-control implementation and adds a separate native Nav2
`FollowPath` path. The two canonical result labels are
`vln_direct_cmd_vel` and `nav2_follow_path_rpp`; aggregate inputs containing
both labels are rejected.

## Commit chain

The first five commits are approved E0 changes cherry-picked onto the fixed
target. The next two commits are the Lane M implementation.

| Commit | Parent | Purpose |
| --- | --- | --- |
| `a853b4b8aad4a8dc26d88fe7f50165f130b0b0a2` | `c275eb429dcce7a2220cc670add4e1b27dcbbe15` | E0 qualification evidence |
| `f32b08a2518e79ff45654d57145cd6aa4333c9e7` | `a853b4b8aad4a8dc26d88fe7f50165f130b0b0a2` | Density consumer contract |
| `5f430e93e40ec861c5ca29f7b1b92fb271783136` | `f32b08a2518e79ff45654d57145cd6aa4333c9e7` | Isolated overlay provenance |
| `385369952f74430ac53f10f9c6254496b649f655` | `5f430e93e40ec861c5ca29f7b1b92fb271783136` | Artifact validity semantics |
| `e4266bfd9d66c62415539279404e72817069bcc2` | `385369952f74430ac53f10f9c6254496b649f655` | Versioned parser provenance |
| `42e51efe02c0e8b291504d028efeba4b1d13c7b0` | `e4266bfd9d66c62415539279404e72817069bcc2` | Parallel FollowPath runtime and profiles |
| `8b2eb174044f7114364f2ea67bc8c43eb3d137e8` | `42e51efe02c0e8b291504d028efeba4b1d13c7b0` | Lifecycle, task-manager, result-path integration |

The original E0 chain ending at
`52e32b1693414fdb76da06a2315e3f3aa09cfef9` is patch-equivalent to the five
local E0 commits. `git cherry 52e32b1 HEAD` must show `-` for those five E0
commits and `+` for Lane M commits.

## Protected files and invariants

The following files must remain present. Except for the explicitly additive
`follow-path` subcommand in `internnav/main`, they were byte-identical to the E0
tip when Lane M was committed.

```text
arena_vln_models/arena_vln_models/backends.py
arena_vln_models/arena_vln_models/core.py
arena_vln_models/arena_vln_models/internnav.py
arena_vln_models/arena_vln_models/visualization.py
_meta/docker/features/internnav/Dockerfile
_meta/docker/features/internnav/docker-compose.yaml
_meta/docker/features/internnav/entrypoint.sh
arena_simulation_setup/launch/internnav_async_eval.launch.py
arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml
```

Also preserve:

- The direct profile's `schema_version: 1`, `direct_cmd_vel: true`, and external
  InternNav server semantics.
- All E0 density, timing, qualification, artifact-validity, and provenance
  tests.
- `task_generator/latched_stage_topic.py` and its stronger existing depth-1
  stage-isolation contract.
- The case-specific `internnav_async_eval.launch.py`; do not move its behavior
  into a shared robot launch.
- The existing InternNav Docker runtime. The new `follow-path` entry is only a
  runner shim and must not replace `launch`, `install`, or model-server actions.

## Explicitly deferred paths

Do not acquire these changes while resolving conflicts:

- `arena_isaac` gitlink changes from `b750120`.
- All 156 files under
  `arena_simulation_setup/worlds/*/scenarios/*/scenario.yaml` changed by the
  source branch.
- Source-branch deletion or replacement of:

  ```text
  arena_vln_models/arena_vln_models/backends.py
  arena_vln_models/arena_vln_models/core.py
  arena_vln_models/arena_vln_models/internnav.py
  arena_vln_models/arena_vln_models/visualization.py
  arena_vln_models/test/offline_inference_replay.py
  arena_vln_models/test/test_action_yaw_sign.py
  arena_vln_models/test/test_arena_vln_models.py
  _meta/docker/features/internnav/Dockerfile
  _meta/docker/features/internnav/docker-compose.yaml
  _meta/docker/features/internnav/entrypoint.sh
  arena_simulation_setup/launch/internnav_async_eval.launch.py
  arena_simulation_setup/launch/robot.launch.py
  ```

- Source changes to `arena_simulation_setup/launch/nav2.launch.py`, the
  `arena_isaac` gitlink, generic benchmark Docker entry point, installation
  scripts, MkDocs navigation, and broad documentation rewrites.
- Source-only model examples, live RPP integration test, LightNav/NaVILA model
  compose files, model downloads, and model-runtime implementation.

## Expected conflicts and resolution rules

At capture time, `feat/internnav-eval-progress` is exactly the fixed target, so
a direct integration is a fast-forward with no textual conflicts. If another
lane advances the target first, expect conflicts in these shared surfaces:

```text
arena_bringup/arena_bringup/internnav_eval.py
arena_bringup/arena_bringup/benchmark_result.py
arena_bringup/arena_bringup/social_nav_metrics_aggregate.py
arena_bringup/arena_bringup/social_nav_validation.py
arena_bringup/launch/arena.launch.py
arena_bringup/test/test_benchmark_result.py
arena_vln_models/package.xml
arena_vln_models/setup.py
task_generator/launch/task_generator.launch.py
task_generator/package.xml
task_generator/task_generator/manager/robot_manager/robot_manager.py
task_generator/task_generator/node.py
task_generator/task_generator/tasks/robots/__init__.py
```

Resolve those conflicts with these rules:

1. Preserve existing direct InternNav flags, adapter imports, timing/provenance
   artifacts, and direct profile defaults.
2. Keep v3 FollowPath profile loading as a separate schema path; never replace
   `arena_bringup.benchmark_profile` wholesale.
3. Keep all FollowPath robot/task behavior behind
   `execution_mode == "nav2_follow_path"`.
4. Keep the velocity smoother as the only final `cmd_vel` publisher in
   FollowPath mode, while leaving direct/wrapper publishers unchanged.
5. Preserve E0 qualification and artifact-validity semantics in result and
   validation code, then add `control_path` fields around them.
6. Keep `extras_require["test"] == ["pytest"]`; do not restore deprecated
   `tests_require`, because that makes colcon choose `unittest` and discover
   zero tests with current setuptools.
7. Reject any resolution that changes `arena_isaac`, a scenario YAML, the
   direct profile, or deletes a protected path.

## Proposed integration procedures

Run every Git procedure from a clean integration worktree. These commands are
proposals only; this handoff does not execute them or push anything.

### Case A: target is still the fixed target

This is the current captured state and should be a strict fast-forward:

```bash
git fetch origin
git status --short
test "$(git rev-parse feat/internnav-eval-progress)" = \
  c275eb429dcce7a2220cc670add4e1b27dcbbe15
git merge-base --is-ancestor feat/internnav-eval-progress \
  integration/vln-follow-path-selective-20261013
git switch feat/internnav-eval-progress
git merge --ff-only integration/vln-follow-path-selective-20261013
```

### Case B: target already contains the original E0 chain

First confirm patch equivalence. The five local E0 commits should be reported
with `-`; only Lane M commits should be reported with `+`:

```bash
git cherry feat/internnav-eval-progress \
  integration/vln-follow-path-selective-20261013
```

Then cherry-pick only the Lane M commits in first-parent order:

```bash
handoff_commit="$(git log -1 --format=%H \
  integration/vln-follow-path-selective-20261013 -- \
  docs/vln_follow_path_selective_integration_handoff.md)"
git switch feat/internnav-eval-progress
git cherry-pick \
  42e51efe02c0e8b291504d028efeba4b1d13c7b0 \
  8b2eb174044f7114364f2ea67bc8c43eb3d137e8 \
  "$handoff_commit"
```

If the final hash is already included because a merge rather than cherry-pick
is being tested, omit it rather than duplicating the commit.

### Case C: target advanced without equivalent E0 commits

Prefer a normal merge in a disposable integration branch so both histories and
the approved E0 baseline remain visible:

```bash
git switch -c integration/vln-follow-path-merge-check \
  feat/internnav-eval-progress
git merge --no-commit --no-ff \
  integration/vln-follow-path-selective-20261013
```

Resolve only the expected shared surfaces above. Abort if the merge introduces
deferred paths or protected-file deletion. After verification, either commit
the merge or abort it; do not rebase Lane M and do not cherry-pick `b750120`.

## Conflict and preservation audit

Before committing an integration result:

```bash
git diff --check
git diff --diff-filter=D --name-only \
  feat/internnav-eval-progress...HEAD
git diff --name-only feat/internnav-eval-progress...HEAD -- \
  arena_isaac 'arena_simulation_setup/worlds/**/scenarios/**'
git merge-base --is-ancestor \
  b750120ad05ae2b7f9397bbf62ca184be90bb498 HEAD && exit 1 || true
```

The deletion and deferred-path commands must print nothing. Inspect every
protected file explicitly, and verify the direct profile still resolves to
`vln_direct_cmd_vel` while all three FollowPath profiles resolve to
`nav2_follow_path_rpp`.

## Containerized verification checklist

All ROS, pytest, and colcon commands belong in
`arena-arena_jazzy_ws-arena-1`. Use fresh build/install/log directories for the
merge candidate.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash

python3 -m pytest -q arena_vln_models/test

colcon --log-base "$TEST_ROOT/log" build \
  --base-paths "$ARENA_TREE" \
  --build-base "$TEST_ROOT/build" \
  --install-base "$TEST_ROOT/install" \
  --merge-install --symlink-install \
  --packages-select arena_bringup arena_simulation_setup \
    arena_vln_models task_generator

source "$TEST_ROOT/install/setup.bash"
colcon --log-base "$TEST_ROOT/test-log" test \
  --base-paths "$ARENA_TREE" \
  --build-base "$TEST_ROOT/build" \
  --install-base "$TEST_ROOT/install" \
  --test-result-base "$TEST_ROOT/test-results" \
  --merge-install --packages-select arena_vln_models \
  --return-code-on-test-failure

colcon test-result --test-result-base "$TEST_ROOT/test-results" --verbose
```

The `colcon test` command must auto-select pytest without
`--python-testing pytest`, discover a nonzero test count, and pass. Finally run
the E0 regression selection, MkDocs strict, launch `--show-args`, and the three
profile plan/dry-run commands recorded in the Lane M handoff.

## Lane M follow-up verification evidence

The test-discovery follow-up was verified on 2026-10-13 inside
`arena-arena_jazzy_ws-arena-1`:

- Direct `python3 -m pytest -q arena_vln_models/test`: 104 passed.
- Fresh isolated colcon build: four selected packages passed.
- Default `colcon test` for `arena_vln_models`, without a
  `--python-testing` override: 104 tests, zero failures, zero errors.
- The recorded colcon command contains `python3 -m pytest`, proving metadata
  selected the pytest extension rather than the previous empty unittest path.
- XUnit verification: one XML file, 104 tests, zero failures, zero errors.
- MkDocs strict build passed after adding this checklist.

The local container log for the isolated build/test run is
`/tmp/lane_m_followup_colcon.log`; the direct pytest log is
`/tmp/lane_m_followup_direct_pytest.log`.
