## 1. Planning and Audit

- [x] 1.1 Inventory the authoritative worktree, OpenSpec format and all model
  adapter call sites.
- [x] 1.2 Write `UNIFIED_MODEL_ADAPTER_AUDIT.md` with raw inference -> adapter ->
  controller file:line chains and risks.
- [x] 1.3 Record the three real raw contracts: InternNav `S2Output` trajectory /
  `output_pixel`, NaVILA text primitive, LightNav cumulative waypoint JSON.

## 2. Unified Adapter Core

- [ ] 2.1 Add `ModelAdapter` protocol, `ModelCapability`, `ModelRawOutput`,
  `ModelPrediction`, registry and factory.
- [ ] 2.2 Add shared geometry: local-to-odom transform, 5 cm sampling, yaw wrap,
  finite/continuity/frame/stamp validation, adjacent duplicate-point removal,
  pure-rotation preservation, empty-path and STOP handling.
- [ ] 2.3 Add deterministic InternNav pixel-goal projection using same-frame
  depth and intrinsics; fail-close when either is invalid.
- [ ] 2.4 Add typed errors: unknown adapter, capability mismatch, protocol error,
  timeout, empty output, non-finite output.
- [ ] 2.5 Add raw-response hash and decoded `modality` to every raw/prediction
  manifest fragment.

## 3. Model Adapters

- [ ] 3.1 InternNav adapter: HTTP `/eval_dual`; prefer native
  `output_trajectory` (`source=native`, `modality=trajectory`); label
  `trajectory_synthesized_from_discrete_action` as `source=synthetic`; expand
  discrete actions as `source=primitive`, `modality=primitive`; project
  pixel-only output as `source=native`, `modality=pixel_projection`.
- [ ] 3.2 NaVILA adapter: TCP 8-byte length protocol; strict text parsing;
  forward 25/50/75 cm and turns 15/30/45 degrees; affirmative STOP only. A text
  primitive produces `source=primitive`, `modality=primitive`.
- [ ] 3.3 LightNav adapter: WebSocket login/reset/next; validate rc/seq/stop;
  retain every finite cumulative waypoint (`source=native`,
  `modality=waypoints`).
- [ ] 3.4 Adapters contain no ROS imports, no `cmd_vel` publisher, no `Twist`,
  and no `send_goal_async`.

## 4. Executor and Launch

- [ ] 4.1 Add pure `ExecutionCore` for session, freshness, timeout, STOP and
  path validation.
- [ ] 4.2 Add `ArenaModelExecutor` ROS node with `eval_ready`, `task_reset`,
  `vln_instruction`, RGB, odom, `model/status`, and the only `FollowPath`
  ActionClient in `arena_vln_models`.
- [ ] 4.3 Add one `config/model_executor.yaml` and extend
  `follow_path.launch.py` with an optional `model_adapter` argument.
- [ ] 4.4 Default and only unified `control_mode=arena_follow_path`. Reject
  `native` in the executor; route `native` to the old isolated legacy entry
  without instantiating `ArenaModelExecutor`. The executor SHALL NOT create a
  `cmd_vel` publisher or `Twist`.

## 5. Legacy Compatibility

- [ ] 5.1 Gate `PythonAdapterBackend` trajectory/discrete conversion behind
  `legacy_native_cmd_vel=true`; default returns a degraded
  `path_requires_executor` decision.
- [ ] 5.2 Turn `InternNavAdapter` and `InternVLARealworldHttpAdapter` into
  deprecated wrappers over the new adapter.
- [ ] 5.3 Keep `create_model_backend`, `create_backend`, `DualVLN*` aliases
  importable.

## 6. Runner and Profile Convergence

- [ ] 6.1 Add `base_follow_path_grscenes.yaml` and
  `profiles/models/{internnav,navila,lightnav}.yaml`.
- [ ] 6.2 Add `runner --model` and `--control-mode`; preserve `--profile` and
  fail before launch when `--profile` and `--model` disagree.
- [ ] 6.3 Record base path/hash, overlay path/hash, canonical resolved hash and
  resolved snapshot in the plan/configuration.
- [ ] 6.4 Convert `_meta/docker/features/{lightnav,navila}/main` to thin
  wrappers over the unified runner.
- [ ] 6.5 Prove three-model dry-run plans differ only in the model whitelist,
  using the real grammar `python3 -m arena_vln_models.runner plan run --model
  <model>`.

## 7. Offline Verification (0 GPU / 0 Isaac / 0 model runtime)

- [ ] 7.1 Add per-model positive/negative fixtures covering coordinate, unit,
  yaw, frame, stamp, continuity, adjacent duplicate points, NaN, empty, STOP,
  pixel projection, timeout and error.
- [ ] 7.2 Add source invariants: no adapter publishes `cmd_vel`; only the
  executor contains `FollowPath`/`send_goal_async`; executor has no `Twist` or
  `cmd_vel` publisher.
- [ ] 7.3 Add registry/CLI/capability mismatch tests.
- [ ] 7.4 Add OpenSpec structure checks: required files exist, every spec has at
  least one `Requirement` and one `Scenario`, and the change names are stable.
- [ ] 7.5 Run `python3 -m pytest` for `arena_vln_models/test` inside
  `arena-arena_jazzy_ws-arena-1`; the historical 104+ tests must not decrease.
- [ ] 7.6 Run an isolated colcon build/test for `arena_vln_models` in the Arena
  container.
- [ ] 7.7 Run three-model `runner plan run --model ...` dry-runs and diff the
  outputs.
- [ ] 7.8 Commit locally on `feat/internnav-eval-progress`; do not push.

## 8. Real Three-Docker Validation (after shared lock is available)

- [ ] 8.1 Confirm the shared GPU/Isaac/model runtime lock is free and use new
  output prefixes; do not rerun an old canary.
- [ ] 8.2 InternNav: one minimal episode through the unified entry with
  `--model internnav`; record artifacts, model trace, FollowPath goals and
  terminal reason.
- [ ] 8.3 LightNav: the unified in-process executor is blocked by missing
  websockets/aiohttp in arena:dev. Use the verified external LightNav client
  closure in the model container for the real episode and record the unified
  LightNav adapter as `fixture/plan qualified, runtime not qualified`; do not
  install or rebuild the image.
- [ ] 8.4 NaVILA: fixture + plan/import only under legal hold; record explicitly
  as `not runtime-qualified`.
- [ ] 8.5 If a shared Isaac/clock harness blocker prevents the episodes, record
  the blocker at the harness layer and do not mark the adapter runtime PASS.
- [ ] 8.6 If a real episode exposes an adapter defect, make an incremental local
  fix commit; no push.
