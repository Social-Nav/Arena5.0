## Context

The audit (`UNIFIED_MODEL_ADAPTER_AUDIT.md`) traced three real inference
contracts:

- InternNav: `InternVLAN1AsyncAgent.step()` returns `S2Output`; the native
  trajectory is `[T+1, 2]` local metres after `/4.0`. The wrapper may also
  synthesise a one-point trajectory from a symbolic arrow.
- NaVILA: TCP returns a JSON text string; strict parsing yields one primitive
  (25/50/75 cm forward or 15/30/45 degree turn or STOP).
- LightNav: WebSocket returns `data.actions.actions` as cumulative
  `[forward_m, lateral_m, yaw_rad]` ego poses plus top-level `data.stop`.

The existing code converts InternNav paths to velocity inside
`PythonAdapterBackend` (`backends.py:814-905`), while NaVILA/LightNav clients
send `FollowPath` goals. The refactor must preserve each raw contract and make
one component own path execution.

## Goals / Non-Goals

**Goals:**

- One `ModelAdapter` protocol and registry/factory shared by all three models.
- One canonical `ModelPrediction` in fixed odom with explicit provenance.
- One `ArenaModelExecutor` that is the only `FollowPath` sender in
  `arena_vln_models` and never lets an adapter publish `cmd_vel`.
- Default `arena_follow_path` + existing Arena RPP; `native` only when explicitly
  configured.
- One base profile plus model overlays; `runner --model` selects the model.
- Fixture-driven tests for raw output, negative cases and plan equivalence.

**Non-Goals:**

- Do not change model weights, prompts, checkpoints, NFS paths or model servers.
- Do not download, migrate or delete a checkpoint.
- Do not delete the legacy direct-cmd_vel path; gate it explicitly and keep it
  in a separate legacy entry.
- Do not introduce a second Nav2 controller stack or a second launch.
- Do not claim `openspec validate`; the CLI is not installed.

**Validation scope (not a non-goal):**

- Offline first: fixture tests, import tests, isolated colcon and three-model
  dry-run plans must pass with 0 GPU / 0 Isaac / 0 model runtime.
- Then, after the shared GPU/Isaac lock is available, run a minimal real
  three-Docker validation: one episode each for InternNav and LightNav on the
  already-authorized/available model services, using new output prefixes and the
  unified entry with only `--model` changed.
- NaVILA is under legal hold; it is qualified by raw fixtures, plan/import and
  adapter contract only. It SHALL NOT be reported as runtime-qualified.
- If a shared Isaac/clock harness blocker prevents the real episode, record the
  blocker at the harness layer and do not rewrite it as an adapter PASS.
- A follow-up fix commit is allowed if the real episode exposes an adapter bug;
  no push.

### LightNav runtime dependency boundary (P1-3)

The approved `arena:dev` runtime container does not provide `websockets` or
`aiohttp`; the running `arena-1` container confirms both are absent.  The
unified in-process `LightNavAdapter` is therefore fixture/plan/import qualified
only.  Real LightNav evaluation continues through the existing external LightNav
client closure in the model container (the same aiohttp client verified in
session_008), not through `ArenaModelExecutor`.  This is recorded as a runtime
qualification blocker: no image rebuild, package install or host mutation is
performed.  InternNav real validation is independent of this boundary.

## Decisions

### Decision 1: Evolve `core.py`, do not replace it

`core.py` already defines `ArenaVLNAdapter` and `ArenaVLNOutput`. Add
`ModelPrediction` and `ModelObservation` there (or re-export from
`model_adapter.py`) and keep `ArenaVLNOutput` for deprecated adapters.
`ModelAdapter` is the new protocol; `ArenaVLNAdapter` remains the legacy
protocol so existing `adapter_target` imports continue to work.

### Decision 2: Raw output stays raw until the adapter boundary

`ModelRawOutput` retains the original response object and a typed decoded form.
The adapter is responsible for protocol validation (rc/seq/stop/finite numbers)
and for expanding primitives. It must not emit velocity. `ModelPrediction`
contains only fixed-odom `poses`, `stop`, `status`, `source` and diagnostics.

### Decision 3: One geometry implementation for all three adapters

All adapters normalise to cumulative local `(x, y, yaw)` metres/radians and call
one shared `local_to_odom` helper. The helper:

- rejects non-finite or malformed points;
- enforces a maximum segment length (continuity);
- samples segments at most 5 cm apart (RPP lookahead behaviour);
- transforms with the observation pose and wraps yaw;
- preserves the final orientation for pure rotations;
- returns an empty path for STOP.

InternNav `[x, y]` trajectories derive per-segment heading
`atan2(dy, dx)`. NaVILA primitives expand to a forward line or one rotation.
LightNav keeps every provided waypoint (no decimation).

### Decision 4: Provenance is explicit and modality is separate

`ModelPrediction.source` is one of:

- `native`: the model returned direct geometric output, i.e. InternNav
  `output_trajectory`, LightNav cumulative waypoints, or an InternNav pixel goal
  projected with same-frame depth/intrinsics (`modality=pixel_projection`).
- `synthetic`: the InternNav wrapper first synthesised a trajectory from a
  symbolic action (`trajectory_synthesized_from_discrete_action=true`).
- `primitive`: the adapter expanded a symbolic action/text primitive into
  geometry (InternNav discrete actions, NaVILA forward/turn text).

Text being model-native does not make the generated path `native`.  The trace
stores both `source` and `modality` (`trajectory`, `waypoints`, `primitive`,
`pixel_projection`, `stop`) plus a hash of the raw response and a short preview.
Synthetic output is never labelled native.

### Decision 5: Executor owns episode and goal lifecycle; no native velocity

`ArenaModelExecutor` mirrors the proven external-client contract:

- `eval_ready` (transient-local, depth 1) binds `session_id` and `episode`;
- `task_reset` can revoke execution but cannot create identity;
- `vln_instruction` is volatile and only accepted after release;
- `model/status` is reliable/volatile JSON with `ready/running/hold/stop/error`;
- the executor waits for fresh RGB/odom, calls the adapter, validates the
  prediction, cancels an in-flight goal before replacing it, and records goal
  UUIDs and results.

The unified executor supports `control_mode=arena_follow_path` only. It does
not create a `cmd_vel` publisher, does not construct `Twist`, and does not
instantiate a native velocity controller. If `control_mode=native` is requested,
the runner routes to the old isolated legacy direct entry; `ArenaModelExecutor`
is not instantiated. This keeps the unified result path exclusively
`nav2_follow_path_rpp`.

### Decision 6: Runner profile resolution and canonical identity

`runner.py` keeps `--profile` as the explicit override. New `--model` selects a
model overlay merged onto `base_follow_path_grscenes.yaml`. If both are given,
the profile's `model` must equal `--model`; otherwise the runner exits before
building a command. The three old v3 profiles remain valid and readable.

The resolved profile is serialized deterministically (sorted keys, no volatile
paths), hashed, and written as a snapshot next to the source profile. The plan
records:

```text
base_profile_path, base_profile_sha256,
overlay_profile_path, overlay_profile_sha256,
resolved_profile_sha256, resolved_profile_snapshot,
profile_id, profile_path, profile_sha256
```

`profile_sha256` is the canonical resolved hash, not merely one source YAML.
The dry-run `plan` output differs between models only in the whitelist:

```text
model, model_adapter, profile_id, profile_path, profile_sha256,
base_profile_path, base_profile_sha256, overlay_profile_path,
overlay_profile_sha256, resolved_profile_sha256, resolved_profile_snapshot,
output_prefix, external_command, model_server, checkpoint, adapter
```

All runtime, evaluation, controller, action, topic and video fields are shared.
The real CLI grammar is `python3 -m arena_vln_models.runner plan run --model
navila` (`verb=plan`, `target=run`).

### Decision 7: Legacy compatibility is a separate entry

`PythonAdapterBackend` keeps `_action_to_command` and
`_trajectory_control_step` only for `legacy_native_cmd_vel=true`. Without the
flag, an output that contains only a trajectory/discrete action returns a
degraded `path_requires_executor` decision with zero velocity and a diagnostic.
This prevents silent path-to-velocity conversion while keeping the old direct
eval runnable when explicitly configured.

The legacy direct entry (`_meta/docker/features/internnav/main` and the
`internnav_grscenes.yaml` schema-1 profile) remains the only place that may
publish direct velocity. The unified runner never selects it implicitly.

### Decision 8: Duplicate points are legal

InternNav native trajectories commonly include the origin and can repeat a
pose. The shared geometry removes adjacent identical poses, preserves a pure
rotation as one pose, and rejects the path only when no executable geometry
remains (non-STOP) or a genuine discontinuity exceeds the segment limit.

`InternNavAdapter`, `InternVLARealworldHttpAdapter`,
`load_internnav_adapter`, `load_internvla_realworld_http_adapter`,
`PythonAdapterBackend` and `create_model_backend` remain importable but are
documented as deprecated thin wrappers.

## Risks / Trade-offs

- [Risk] Removing default path-to-velocity conversion can break an unconfigured
  direct eval. Mitigation: explicit `legacy_native_cmd_vel=true` and a clear
  `path_requires_executor` diagnostic.
- [Risk] InternNav `[x, y]` has no yaw. Mitigation: derive segment heading,
  reject zero-length segments, and mark the derivation in debug.
- [Risk] LightNav's default WebSocket client needs `aiohttp` or `websockets`.
  Mitigation: transport is injectable; the default raises a named error if no
  client library is present, and no new dependency is mandatory.
- [Risk] NaVILA requires 8 frames. Mitigation: maintain a per-session history and
  pad deterministically with black frames, matching the external client.
- [Risk] The executor could race a legacy external client for FollowPath goals.
  Mitigation: a single `model_adapter` is selected per launch; the old
  external-client profiles keep `external.command: []` and `model_adapter: ''`.
- [Risk] Base+overlay profiles may drift from the old full profiles.
  Mitigation: tests load both and assert shared evaluation/runtime/controller
  fields and only whitelisted model differences.

## Migration Plan

1. Add the audit and OpenSpec change.
2. Add canonical types, registry and geometry helpers (including duplicate-point
   de-duplication and pixel projection).
3. Add the three adapters and fixture tests.
4. Add the pure execution core and executor; extend `follow_path.launch.py`.
5. Gate legacy conversion in `backends.py`; turn legacy classes into wrappers.
6. Add runner `--model`, base/overlay profiles and canonical resolved identity;
   convert shell entries.
7. Run fixture tests, import tests, isolated colcon and dry-run plan equivalence
   inside the Arena container with 0 GPU / 0 model runtime.
8. Commit locally; do not push.
9. After the shared GPU/Isaac lock is available, run one InternNav and one
   LightNav episode through the unified entry with only `--model` changed and
   new output prefixes. Record NaVILA as fixture/plan/import-only under legal
   hold. If a harness blocker appears, record it separately; do not mark the
   adapter as runtime PASS. A follow-up fix commit is allowed if needed.

Rollback is a normal revert of the change commit. The existing v3 profiles and
external-client flow remain usable throughout.
