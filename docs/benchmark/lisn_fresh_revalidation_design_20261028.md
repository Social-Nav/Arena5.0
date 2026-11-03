# LISN fresh-revalidation design candidate (docs only)

Status: **design candidate, not implemented**. No runtime, no model API, no source
default or threshold change is made by this document. The stable index row remains
`BLOCKED_CONTRACT_5S_SOURCE_AGE` / `NOT_RUNTIME_OBSERVED`.

Verified against worktree HEAD `fe8549477acad5c3d8fdcea0ee002aa926f0e43b`.

## 1. Deterministic conflict (existing code facts)

### 1.1 Producer wait cap

- `arena_vln_models/arena_vln_models/runner.py:392`
  `--lisn-producer-timeout-sec` default **120.0**.
- `arena_vln_models/arena_vln_models/runner.py:394`
  `--lisn-producer-deadline-sec` default **120.0**.
- `arena_vln_models/arena_vln_models/runner.py:296-298` forwards
  `lisn_slow_loop_producer_timeout_sec` / `lisn_slow_loop_producer_deadline_sec`
  to the launch.
- `arena_bringup/arena_bringup/lisn_slow_loop_ros.py:611-612`
  `ProducerConfig.timeout_sec` default **8.0**, `deadline_sec` default **120.0**.
- `arena_bringup/arena_bringup/lisn_slow_loop_ros.py:1226-1244`
  `build_snapshot` builds `valid_until_ns = now_ns + max_age_ns` from the
  **commit-time** `now_ns`. This is a snapshot validity window, not a
  measurement of how long the request waited.

The **20s wait cap is an explicit CLI/env run parameter only** (for example
`--lisn-producer-timeout-sec 20 --lisn-producer-deadline-sec 20`). It is not a
profile value and must not be written into a profile file.

### 1.2 Contract gates

- `arena_bringup/configs/lisn/slow_loop_engineering_v1.json:10`
  `max_observation_age_ns = 5000000000` (**5s**). The same value appears in
  `slow_loop_gemini_engineering_v1.json:12` and
  `slow_loop_gemini_engineering_v2.json:94`.
- `arena_bringup/arena_bringup/lisn_slow_loop_contract.py:103`
  policy field `max_observation_age_ns`; `:309` parses it from the policy JSON.
- `arena_bringup/arena_bringup/lisn_slow_loop_contract.py:537-541`
  monotonic received-age (`now_monotonic_ns - observation.received_monotonic_ns`)
  > `max_observation_age_ns` -> reject `OBSERVATION_STALE`. This is a monotonic
  age check, not a simulation-stamp difference check.
- `arena_bringup/arena_bringup/lisn_slow_loop_contract.py:559`
  `source.stamp_ns != observation.ros_stamp_ns` -> reject
  `SOURCE_OBSERVATION_IDENTITY_MISMATCH`.
- `arena_bringup/arena_bringup/lisn_slow_loop_ros.py:1268` policy field
  `max_observation_age_ns`; `:1304` / `:1417` pass `max_age_ns` into the
  contract.

### 1.3 Consequence

A producer wait of 20s is a **maximum request wait cap**, not a fixed
per-request delay. Whether a returned proposal is rejected depends on the actual
monotonic received-age measured at commit time: only when
`now_monotonic_ns - observation.received_monotonic_ns > max_observation_age_ns`
(5s) does the contract reject with `OBSERVATION_STALE`
(`lisn_slow_loop_contract.py:537-541`). The single measured independent sample
was 10.508s (>5s), so that sample would be stale on the modulation path, but one
sample does not imply that every future request necessarily exceeds 5s.

Separately, `SOURCE_OBSERVATION_IDENTITY_MISMATCH`
(`lisn_slow_loop_contract.py:559`) requires the proposal's
`source_observation.id/frame_id/stamp_ns` to equal the current observation. If a
returned proposal's source does not match the current observation, it must be
dropped, and `valid_until_ns` generated at commit time
(`lisn_slow_loop_ros.py:1226-1244`) is not evidence that the proposal was
already expired merely because the request waited.

The correct behaviour for any rejected proposal is a typed safe drop
(`OBSERVATION_STALE` or `SOURCE_OBSERVATION_IDENTITY_MISMATCH`), with
`apply=0` and `readback=0`.

Fresh TF/geometry does **not** make the old proposal valid. Rebinding the old
proposal's source identity to a fresh observation, or rewriting the proposal
stamp, is forbidden.

## 2. Minimal design candidate (not implemented)

The candidate is an **independent revalidation evidence object**. It records the
relationship between a proposal and a later observation; it does not grant the
proposal validity by itself.

### a) Immutable proposal envelope

- `proposal_message`, `generated_at_ns`, and the original
  `source_observation` (episode, observation id, frame id, `ros_stamp_ns`,
  `received_monotonic_ns`, `content_sha256`) are preserved verbatim.
- No field in this envelope may be overwritten by a fresh observation.

### b) Independent current-observation record

- A separate record with its own `stamp_ns`, `frame_id`, TF snapshot and
  geometry snapshot.
- It is stored alongside, never merged into, the proposal envelope.

### c) Fail-closed validation table

| Check | Inputs | Parameter slots | Failure |
|---|---|---|---|
| Semantic consistency | proposal `goal`/`controller_mode`, current instruction | semantic rule (TBD) | `REVALIDATION_SEMANTIC_MISMATCH` |
| Geometry position | proposal goal frame vs current TF/pose | distance threshold (TBD) | `REVALIDATION_GEOMETRY_MISMATCH` |
| Geometry orientation | proposal goal yaw vs current TF/pose | orientation threshold (TBD) | `REVALIDATION_ORIENTATION_MISMATCH` |
| Observation identity | proposal source stamp vs current observation | exact identity rule | `REVALIDATION_IDENTITY_MISMATCH` |

All threshold values are placeholders. They must be supplied as explicit new
parameters only after Main approval.

### d) Decision semantics

- The evidence object proves only “the current observation is related to this
  proposal”.
- It never authorises the old proposal to be applied.
- Final apply still requires a validity rule approved by Main. Until then the
  current behaviour (safe drop) is unchanged.

## 3. Focused unit-test naming list (not written, not passing)

Proposed file: `arena_bringup/test/test_lisn_contract_revalidation_design.py`

- `test_proposal_envelope_is_immutable_under_fresh_observation`
- `test_current_observation_has_independent_stamp_and_frame`
- `test_semantic_mismatch_fails_closed`
- `test_geometry_mismatch_fails_closed`
- `test_orientation_mismatch_fails_closed`
- `test_evidence_object_does_not_authorize_apply`
- `test_stale_source_identity_is_not_rebound`
- `test_valid_until_expiry_is_still_enforced`

These names describe intended tests only; no implementation is added here.

## 4. Decision points for Main (worker does not adopt)

1. **Option A**: keep the 5s source-age gate and safe drop (zero code change,
   current behaviour).
2. **Option B**: add an independent TTL/validity semantic for the evidence
   object, timed separately from the proposal source age. The exact rule and
   clock are TBD.
3. **Geometry parameters**: any distance/orientation/overlap threshold must be a
   new explicit parameter. The previously rejected 0.05 m distance, 0.1 rad
   orientation and hash-prefix safe-hold thresholds must not be revived.
4. **Scope**: P1A frozen results, existing test assertions and the source-age
   contract are unchanged.

## 5. Boundaries

- No LISN episode was run for this document.
- No model API was called.
- No source default or threshold was changed.
- NaVILA, LightNav and InternNav artifacts and sealed rows are untouched.
- The stable index row stays `BLOCKED_CONTRACT_5S_SOURCE_AGE` /
  `NOT_RUNTIME_OBSERVED`.
