# Unified VLN ModelAdapter Audit

Date: 2026-10-22
Author: offline subworker (no GPU / no Isaac / no model runtime)

## 1. OpenSpec rules and authoritative baseline

### OpenSpec format

The repository's OpenSpec is a plain-directory, spec-driven convention. There is
no `openspec` CLI on this host (`command not found`), so validation is structural
and test-based, not CLI-based.

Authoritative change format observed in
`/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/openspec/changes`:

```text
<change-name>/
  .openspec.yaml                 # schema: spec-driven; created: YYYY-MM-DD
  proposal.md                    # ## Why / ## What Changes / ## Impact
  design.md                      # ## Context / ## Goals / ## Decisions / ## Risks / ## Migration Plan
  tasks.md                       # numbered checklist, verification notes inline
  specs/<capability>/spec.md     # ## ADDED Requirements / ### Requirement: ... / #### Scenario: ...
```

Reference changes used for the exact style:

- `openspec/changes/stabilize-external-internnav-eval/`
- `openspec/changes/add-social-nav-scenario-runner/`
- `openspec/changes/diagnose-internnav-rotation/specs/internnav-inference-diagnostics/spec.md`

The workspace-level mirror `/home/ubuntu/arena_jazzy_ws/openspec` is not a git
repository and is stale relative to the Arena repo; the committed change belongs
under the Arena worktree's `openspec/changes/`. The same directory is mirrored to
the workspace-level OpenSpec for discoverability.

### Authoritative branch / worktree

| Item | Value |
| --- | --- |
| Branch | `feat/internnav-eval-progress` |
| HEAD | `6d255dea7d147f054be1defd4c79b2abaa951bfd` |
| Write worktree | `/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013` |
| Worktree before inventory | clean (`git status --porcelain` empty), no untracked files |
| Ancestor integration | `integration/vln-follow-path-selective` @ `29f4f50` is an ancestor of `6d255de` |
| Candidate | `candidate/lisn-v5-target-integration` @ `3cd247e` is a child of `6d255de` and touches no `arena_vln_models` files |
| Active LightNav session | `arena-lightnav-real-20261022-session008-*`; its read-only tree `lisn-v5-target-integration-20261021` was not modified |

All edits are confined to the write worktree. No new worktree, push, reset,
amend, NFS write, container restart, Isaac launch, model launch, install, or
download was performed.

## 2. Current architecture in one paragraph

`arena_vln_models` currently contains two different integration generations.
The legacy generation (`core.py`, `backends.py`, `internnav.py`,
`visualization.py`) is a direct `cmd_vel` wrapper: an adapter returns a mapping,
`PythonAdapterBackend` converts the mapping to `ModelSimDecision(linear_x,
angular_z)`, and `task_generator` republishes that status command on
`<robot>/cmd_vel`. The native generation (`runner.py`, `profile.py`,
`lifecycle.py`, `follow_path.launch.py`, `tracking_config.py`) is model-agnostic
Nav2 `FollowPath`: an external client in each model repository sends
`nav2_msgs/action/FollowPath` goals, Arena's unmodified RPP controller and
velocity smoother publish the only final `cmd_vel`, and `WaypointLifecycle`
enforces the model heartbeat/STOP contract. The refactor must unify the first
generation's raw model output decoding with the second generation's path
execution without keeping two controllers or two model-specific launches.

## 3. Raw inference -> adapter -> controller chains (file:line)

### 3.1 InternNav

Raw inference return:

- `/home/ubuntu/arena_jazzy_ws/deps/InternNav/internnav/agent/internvla_n1_agent_realworld.py:366-403`
  `InternVLAN1AsyncAgent.step(rgb, depth, pose, instruction, intrinsic,
  look_down)` returns `S2Output`.
- `.../internvla_n1_agent_realworld.py:399-401` calls
  `traj_to_actions(trajectories, use_discrate_action=False)` and assigns
  `dual_sys_output.output_trajectory`.
- `/home/ubuntu/arena_jazzy_ws/deps/InternNav/internnav/model/utils/vln_utils.py:63-144`
  `traj_to_actions` unnormalises deltas (`/4.0`), cumulatively reconstructs
  `[T+1, 2]` local `(x, y)` waypoints, averages over the batch, and returns the
  trajectory. There is no yaw column in the native return.
- `/home/ubuntu/arena_jazzy_ws/deps/InternNav/internnav/model/utils/vln_utils.py:158-167`
  `S2Output` fields: `output_action`, `output_trajectory`, `output_pixel`,
  `output_latent`.

Arena wrapper / decode:

- `/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_vln_models/arena_vln_models/internnav.py:736-757`
  the subprocess worker calls `agent.step(...)`, then `_normalize_output(...)`,
  and emits JSON. `agent.step` is the original inference return.
- `arena_vln_models/internnav.py:616-656` `_normalize_output` preserves
  `output_trajectory`, `output_pixel`, `discrete_action`, and synthesises a
  one-point trajectory when a symbolic action arrives under the trajectory policy
  (`trajectory_synthesized_from_discrete_action=True`, line 654).
- `arena_vln_models/internnav.py:1183-1221` `InternNavAdapter._real_response`
  calls `self._adapter.step(...)` at 1200-1207 and `normalize_backend_output(...)`
  at 1209.
- `arena_vln_models/internnav.py:383-450` `InternVLARealworldHttpAdapter.compute`
  posts to `/eval_dual` and normalises the HTTP response at 436.
- `arena_vln_models/core.py:185-230` `normalize_backend_output` maps
  `output_action`/`action` -> `discrete_action`, `trajectory` ->
  `output_trajectory`, `target_pixel`/`pixel_goal` -> `output_pixel`.

Premature command conversion (the defect):

- `arena_vln_models/backends.py:744-975` `PythonAdapterBackend`.
- `arena_vln_models/backends.py:814-841` `_trajectory_decision` selects the last
  waypoint via `_trajectory_control_step` (`backends.py:79-107`), computes
  `linear_x = hypot(x, y)` and `angular_z = yaw`, and returns
  `ModelSimDecision(v, w)`. This discards all intermediate geometry and treats a
  path as a velocity.
- `arena_vln_models/backends.py:843-871` `_discrete_decision` converts one
  action id to a bounded `cmd_vel` (`_action_to_command`, lines 160-214) or an
  official primitive (`_official_discrete_primitive`, lines 217-268).
- `arena_vln_models/backends.py:873-905` policy precedence
  (`trajectory` default, then `discrete`, then `raw`).

Controller:

- Direct path: `task_generator/task_generator/manager/robot_manager/robot_manager.py:1216-1252`
  `_update_direct_dual_vln_status_twist` and
  `_publish_direct_dual_vln_status_command` mirror `linear_x`/`angular_z` from
  `model/status` onto `<robot>/cmd_vel`. The publisher is created only when
  `_uses_waypoints()` is false (`robot_manager.py:220-231`).
- FollowPath path: `arena_vln_models/launch/follow_path.launch.py:24-35`
  starts `controller_server` (RPP `FollowPath`) and `velocity_smoother`
  (smoother is the final `cmd_vel` publisher). `arena_vln_models/lifecycle.py:23-202`
  `WaypointLifecycle` owns controller activate/deactivate, heartbeat, STOP, and
  final-zero observation. In FollowPath mode `robot_manager.py:1634-1639`
  deliberately does not send a Nav2 `navigate_to_pose` goal; the external client
  owns `FollowPath` goals.
- There is currently **no** InternNav producer for FollowPath inside this repo.
  The three profiles only label the experiment (`model: internnav`), and
  `external.command` is empty.

### 3.2 NaVILA

Raw inference return (external model repo, read-only audit):

- `/mnt/bn/robot-mllm-data-hl/arena_vln_baselines/repos/NaVILA/integrations/arena/core.py:125-183`
  `TCPModel.generate(images, query)` sends 8-byte big-endian length + JSON and
  returns a JSON **string** (the raw model text).
- `.../NaVILA/integrations/arena/core.py:17-46` `parse_action(text)` is the
  strict text decoder: forward 25/50/75 cm, left/right 15/30/45 degrees, or an
  affirmative STOP. Unknown/conflicting text raises `ValueError`.
- `.../NaVILA/integrations/arena/core.py:49-72` `action_poses(action,
  observation)` converts one primitive to fixed-odom poses (5 cm sampling for
  forward; one pure-rotation pose for turns; empty for STOP).

Controller:

- `.../NaVILA/integrations/arena/client.py:99` creates the `FollowPath`
  ActionClient; `client.py:101` publishes `model/status`.
- `.../NaVILA/integrations/arena/client.py:438` calls `parse_action(text)`;
  `client.py:358-376` builds the `nav_msgs/Path` and calls
  `send_goal_async`.
- Arena side: no adapter. `arena_bringup/configs/benchmark/profiles/navila_grscenes.yaml`
  only selects `model: navila`, `execution.mode: nav2_follow_path`,
  `external.command: []`. `arena_vln_models/runner.py:156-175` starts the
  evaluator; `follow_path.launch.py` owns RPP and the smoother.

### 3.3 LightNav

Raw inference return (external model repo, read-only audit):

- `/mnt/bn/robot-mllm-data-hl/arena_vln_baselines/repos/LightNav-0/integrations/arena/transport.py:146-177`
  `_connect` performs `login` then `reset`.
- `.../LightNav-0/integrations/arena/transport.py:179-198` `_generate` sends
  `{"action": "next", "data": {"seq": ..., "image": ..., "instruction": ...}}`
  and returns the unmodified JSON response.
- `.../LightNav-0/integrations/arena/core.py:49-82` `parse_prediction(response,
  expected_seq)` validates `action == "next"`, `data.rc == 0`, `data.seq`,
  `data.stop` boolean, and `data.actions.actions` as finite 3-column
  `[forward_m, lateral_m, yaw_rad]` cumulative ego poses. Non-STOP with no
  waypoints is an error.
- `.../LightNav-0/integrations/arena/core.py:85-119` `path_poses` rigidly
  projects the cumulative ego path into the fixed odom frame using the
  observation-time TF, samples at 5 cm, and retains the final orientation.

Controller:

- `.../LightNav-0/integrations/arena/client.py:102` creates the `FollowPath`
  ActionClient; `client.py:104` publishes `model/status`.
- `.../LightNav-0/integrations/arena/client.py:353-374` builds the path and
  calls `send_goal_async`.
- Arena side: no adapter. `arena_bringup/configs/benchmark/profiles/lightnav_grscenes.yaml`
  only labels the experiment; `external.command` is empty.

## 4. Proposed changed files

New unified implementation (inside `arena_vln_models`):

| File | Purpose |
| --- | --- |
| `arena_vln_models/arena_vln_models/model_adapter.py` | `ModelAdapter` protocol, capability/registry/factory, canonical `ModelObservation`/`ModelRawOutput`/`ModelPrediction`, geometry and validation errors |
| `arena_vln_models/arena_vln_models/adapters/base.py` | shared adapter base; no ROS imports, no `cmd_vel` |
| `arena_vln_models/arena_vln_models/adapters/internnav.py` | HTTP `/eval_dual` raw-output adapter; native trajectory / synthetic / primitive provenance |
| `arena_vln_models/arena_vln_models/adapters/navila.py` | TCP text adapter; strict primitive expansion |
| `arena_vln_models/arena_vln_models/adapters/lightnav.py` | WebSocket JSON adapter; full cumulative waypoint path |
| `arena_vln_models/arena_vln_models/execution.py` | pure `ExecutionCore`: session, timeout, STOP, path validation, continuity, error mapping |
| `arena_vln_models/arena_vln_models/executor.py` | `ArenaModelExecutor` ROS node; sole `FollowPath` ActionClient/sender; `model/status` publisher |
| `arena_vln_models/config/model_executor.yaml` | one shared executor config; model selected by parameter |
| `arena_vln_models/test/test_model_adapter_contract.py` | per-model positive/negative fixtures |
| `arena_vln_models/test/test_model_adapter_registry.py` | registry/CLI/capability/source invariants |
| `arena_vln_models/test/test_executor_contract.py` | pure executor contract and three-model plan diff |
| `arena_vln_models/test/fixtures/*.json` | raw model output fixtures |

Modified:

| File | Change |
| --- | --- |
| `arena_vln_models/arena_vln_models/core.py` | evolve `ArenaVLNAdapter`; add canonical `ModelPrediction`; keep `ArenaVLNOutput` |
| `arena_vln_models/arena_vln_models/backends.py` | remove automatic trajectory/discrete -> velocity unless explicit `legacy_native_cmd_vel=true` |
| `arena_vln_models/arena_vln_models/internnav.py` | legacy classes become deprecated thin wrappers over the new adapters |
| `arena_vln_models/arena_vln_models/runner.py` | add `--model`, `--control-mode`, `--model-adapter`; base+overlay profiles; model-consistency check |
| `arena_vln_models/arena_vln_models/profile.py` | optional `adapter` section and base+overlay merge |
| `arena_vln_models/launch/follow_path.launch.py` | optional `model_adapter` arg includes `ArenaModelExecutor`; existing external-client mode unchanged |
| `arena_vln_models/setup.py`, `package.xml` | install config, `model_executor` console script |
| `arena_bringup/configs/benchmark/profiles/base_follow_path_grscenes.yaml` | one shared base profile |
| `arena_bringup/configs/benchmark/profiles/models/{internnav,navila,lightnav}.yaml` | model overlays only |
| `_meta/docker/features/{lightnav,navila}/main` | thin wrappers over `arena_vln_models.runner --model ...` |
| `docs/benchmark/running-benchmarks.md` | unified CLI examples |
| `openspec/changes/unify-vln-model-adapters/**` | proposal/design/tasks/spec deltas |

## 5. Real inconsistencies and risks

1. **Two control contracts coexist.** `PythonAdapterBackend` returns
   `(linear_x, angular_z)`; the native FollowPath path consumes paths. There is
   no common raw-output type. The refactor introduces `ModelPrediction` and makes
   the executor the only path sender.
2. **InternNav raw output has no yaw.** `traj_to_actions(...,
   use_discrate_action=False)` returns `[T+1, 2]`. The adapter must derive a
   heading per segment and mark the source. It must not treat the synthetic
   one-point trajectory from a symbolic action as native.
3. **Symbolic-action synthesis is already inside the wrapper.**
   `internnav.py:648-655` synthesises `[[0.25, 0, ±15°]]` when the model answers
   with an arrow. That is useful and auditable, but it is not a raw model
   trajectory. It must carry `source=synthetic` and a manifest/debug marker.
4. **InternNav native trajectory is local and cumulative.** NaVILA/LightNav
   paths are absolute odom poses (NaVILA primitives) or cumulative ego poses
   (LightNav). The unified adapter must normalise all three to fixed-odom
   `[x, y, yaw]` before the executor sends FollowPath.
5. **Unit drift.** InternNav trajectory is metres after `/4.0`; NaVILA text is
   centimetres/degrees; LightNav JSON is metres/radians. The adapters must
   convert at the boundary and reject non-finite values.
6. **Frame and stamp drift.** The external clients use observation-time TF and
   a fixed odom frame. The executor must reject empty `frame_id`, zero stamp, or
   a pose that does not match the RGB/odom capture time.
7. **STOP semantics differ.** InternNav action `0`, NaVILA affirmative text,
   LightNav top-level `data.stop`. A pointing/visibility hint is not STOP.
   `Session` already rejects STOP before release; the executor must preserve
   that rule.
8. **Timeout/error semantics.** NaVILA and LightNav clients distinguish timeout,
   protocol error, and service error; the legacy backend collapses many to
   `safe_stop`. The executor must map timeout to `error`/hold and never silently
   continue.
9. **Duplicate profiles.** The three v3 profiles differ only in
   `id/description/model/output_prefix`; `external.command` is empty in all
   three. A base+overlay removes the duplication while keeping the old files
   readable.
10. **Launch ownership.** `follow_path.launch.py` already owns RPP and the
    smoother. Adding a second executor launch would create two controller
    stacks; the unified launch must extend the existing file.
11. **No ROS/model runtime in this task.** All new tests use fixtures and
    injected transports; no model server, GPU, Isaac, or real ROS graph is used.
12. **Legacy compatibility.** `PythonAdapterBackend` conversion must remain
    available only through an explicit `legacy_native_cmd_vel=true` flag so the
    old direct-cmd_vel eval can still run while the new default refuses to turn
    paths into velocities.
