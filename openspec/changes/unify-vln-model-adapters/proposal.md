## Why

InternNav, NaVILA and LightNav currently reach Arena through three different
contracts. InternNav's legacy wrapper decodes raw model output into
`linear_x`/`angular_z` inside `PythonAdapterBackend`, while NaVILA and LightNav
each ship an external ROS client that sends `nav2_msgs/action/FollowPath` goals.
The three v3 benchmark profiles are near-duplicates, there is no common raw
model-output type, and the same trajectory can be converted to velocity in one
path and to a path in another. This makes model comparison ambiguous and keeps
the control-path contract split between `vln_direct_cmd_vel` and
`nav2_follow_path_rpp`.

The audit is in `UNIFIED_MODEL_ADAPTER_AUDIT.md` at the Arena worktree root.

## What Changes

- Add one `ModelAdapter` protocol with canonical `ModelObservation`,
  `ModelRawOutput` and `ModelPrediction` types, plus a shared registry/factory.
- Add three adapters that decode each model's original inference output:
  InternNav HTTP `/eval_dual` (native trajectory, synthetic symbolic trajectory,
  primitive action, evidenced pixel projection), NaVILA TCP text primitives, and
  LightNav WebSocket cumulative waypoints.
- Add one `ArenaModelExecutor` that owns episode identity, observation freshness,
  model status heartbeats, timeout/error handling and the only `FollowPath`
  action client/goal sender in `arena_vln_models`. The unified executor runs
  `control_mode=arena_follow_path` only; it never publishes `cmd_vel` and never
  instantiates a native velocity controller.
- Remove automatic trajectory/discrete-action to `cmd_vel` conversion from
  `PythonAdapterBackend`; keep it only behind an explicit
  `legacy_native_cmd_vel=true` compatibility flag in the existing legacy entry.
- Default `control_mode=arena_follow_path` with the existing Arena RPP
  controller. `control_mode=native` is routed to the old isolated legacy entry
  and must not instantiate `ArenaModelExecutor`.
- Converge the three v3 profiles onto one shared base profile plus model
  overlays. `arena_vln_models.runner` accepts `--model internnav|lightnav|navila`
  while preserving explicit `--profile` compatibility and validating that the
  profile's `model` matches. The resolved profile gets a deterministic canonical
  hash and snapshot in addition to base/overlay hashes.
- Convert `_meta/docker/features/{lightnav,navila}/main` into thin wrappers over
  the unified runner. Keep the existing InternNav direct-cmd_vel path as a
  deprecated compatibility entry.
- Add fixture-driven positive/negative tests for coordinates, units, yaw,
  frames, stamps, continuity, duplicate points, NaN, empty output, STOP, pixel
  projection, timeout, error, registry selection, source scanning and
  three-model dry-run equivalence.
- Run offline tests first, then a minimal real three-Docker validation on
  authorized models (InternNav and LightNav one episode each). NaVILA remains
  legal-hold and is only fixture/plan/import-qualified, not runtime-qualified.

## Capabilities

### New Capabilities

- `unified-vln-model-adapter`: one protocol, registry, factory and three
  raw-output adapters that produce fixed-odom paths and never publish `cmd_vel`.
- `unified-vln-eval-runner`: one model-independent FollowPath executor and one
  base profile with model overlays, selected by `--model`.

### Modified Capabilities

- `dual-vln-eval-runner`: `arena_vln_models.runner` gains model selection,
  model-consistency validation and a canonical resolved-profile identity.
- `dual-vln-external-model-adapter`: legacy `adapter_target` classes become
  deprecated wrappers; path-to-velocity conversion requires explicit opt-in in
  the legacy entry.
- `official-internnav-client-lifecycle`: InternNav raw output provenance
  (`native`, `synthetic`, `primitive`, `pixel_projection`) is recorded and the
  executor owns the FollowPath goal instead of the legacy status-to-cmd_vel
  bridge.
- `dual-vln-debug-observability`: executor traces record raw response hashes,
  decoded modality, adapter source, validation results, goal UUIDs and terminal
  reasons.

## Impact

- `arena_vln_models/arena_vln_models/model_adapter.py`,
  `adapters/{base,internnav,navila,lightnav}.py`, `execution.py`,
  `executor.py`.
- `arena_vln_models/arena_vln_models/{core,backends,internnav,runner,profile}.py`.
- `arena_vln_models/launch/follow_path.launch.py`,
  `arena_vln_models/config/model_executor.yaml`,
  `arena_vln_models/setup.py`, `arena_vln_models/package.xml`.
- `arena_bringup/configs/benchmark/profiles/base_follow_path_grscenes.yaml`
  and `.../profiles/models/*.yaml`.
- `_meta/docker/features/{lightnav,navila}/main`.
- Tests and fixtures under `arena_vln_models/test`.
- No new third-party dependency is required for the default path. LightNav's
  default WebSocket transport is optional and injectable; NaVILA uses stdlib
  sockets and InternNav uses the existing `requests`/urllib fallback.
- No model checkpoint, NFS path, GPU, Isaac or model runtime is touched by the
  offline phase. The real validation phase reuses already-authorized model
  services and does not download or migrate checkpoints.
