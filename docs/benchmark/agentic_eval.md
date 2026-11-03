# Agentic Evaluation Contract

AI agents in this repository are **operators**. They may preflight, plan,
launch, cancel, read, and audit. They are not navigation policies and must not
produce robot actions.

## Hard boundaries

An operator agent must not:

- publish `Twist`, `FollowPath`, or any robot control command
- act as a model/policy or replace the common Arena executor
- guess a metric distance, depth, or Path from a pixel/bearing
- change an unfrozen case, taxonomy, split, threshold, seed, or scientific
  setting
- consume GT global map, HuNav People/Agents truth, eval true goal, USD state,
  or simulator depth in a non-GT arm
- create a teacher trajectory from an ineligible source run

The agent's output is limited to commands, plans, launch/stop actions, and
evidence receipts.

## Stable commands

```bash
src/Arena/_meta/docker/features/benchmark/main config
src/Arena/_meta/docker/features/benchmark/main doctor
src/Arena/_meta/docker/features/benchmark/main plan run
src/Arena/_meta/docker/features/benchmark/main run
src/Arena/_meta/docker/features/benchmark/main full-eval
src/Arena/_meta/docker/features/benchmark/main stop
```

All ROS/launch/eval/build commands run in `arena-arena_jazzy_ws-arena-1`.
Isaac Sim runs only in `arena-arena_jazzy_ws-isaac-1`; Qwen only in
`arena-qwen3vl-fp8`. The shared lock is
`/tmp/arena_jazzy_ws_isaac_gpu_runtime.lock`.

### Validators

```bash
ros2 run arena_bringup paper_pipeline settings --file <json>
ros2 run arena_bringup paper_pipeline bfast --file <json> --enabled|--disabled
ros2 run arena_bringup paper_pipeline bfast-profile --file <json>
ros2 run arena_bringup paper_pipeline latency --mode <mode> ...
ros2 run arena_bringup paper_pipeline scenario --file <json>
ros2 run arena_bringup paper_pipeline dataset --manifest <json>
ros2 run arena_bringup paper_pipeline teacher --manifest <json>
ros2 run arena_bringup paper_pipeline teacher-audit --manifest <json>
ros2 run arena_bringup paper_pipeline source-run-audit \
  --source-run <run> --derived-root <fresh-root>
ros2 run arena_bringup paper_pipeline agentic-request --file <json>
ros2 run arena_bringup paper_pipeline agentic-lifecycle \
  --file <json> --action plan|start|cancel|finalize|read|audit \
  [--from-state <state>]
ros2 run arena_bringup paper_pipeline audit-consolidate \
  --manifest <json> --output <fresh-json>
ros2 run arena_bringup paper_pipeline evidence-index \
  --run-dir <run> --output <fresh-json>
ros2 run arena_bringup paper_pipeline evidence-verdict \
  --index <evidence-index.json> --output <fresh-json>
ros2 run arena_bringup paper_pipeline recording-contract --file <json>
ros2 run arena_bringup paper_pipeline recording-preflight \
  --contract <json> --run-dir <run> --output <fresh-json>
ros2 run arena_bringup paper_pipeline metrics \
  --run-dir <run> --derived-root <fresh-root>
ros2 run arena_bringup benchmark_result --dir <run> --output <fresh-json> \
  --require-valid
```

The installed console aliases require the `arena_bringup` overlay rebuild;
the source form `python3 -m arena_bringup.paper_pipeline ...` is the
fallback.

## Method routing

The only stable method-plan entry is:

```bash
src/Arena/_meta/docker/features/benchmark/main plan --method <id> \
  --common configs/agentic_eval/methods/common_contract.v1.json \
  --output <fresh-json>
```

Equivalent source form:

```bash
python3 -m arena_bringup.paper_pipeline method-plan \
  --method <id> --common <common-json> --output <fresh-json>
```

It returns a machine-readable plan and never starts a container, model,
Isaac, or episode. Each method bundle binds method-private fields:
profile/audit identity, model image/service/container/entrypoint, revision/
checkpoint source, allowed sensors, adapter/action mapping, controller/
local planner, ports, ROS namespace/topics/domain, common executor, output
prefix, shared lock, cleanup, and source binding.

The common contract is provided once:
`configs/agentic_eval/methods/common_contract.v1.json`. It owns case/world/
scenario/robot, seed, episodes, instruction, HuNav, timeout, metrics, video,
and recording. A bundle with a common field is
`METHOD_OVERRIDES_COMMON:<field>`. Switching method must keep
`common_contract_hash` identical and change only `method_private_hash`.

| Method ID | Role | Status |
| --- | --- | --- |
| `internnav` | `generic_internnav` | plan-only; current tree BLOCKED (`METHOD_BUNDLE_CURRENT_PROFILE_DRIFT`) |
| `navila` | `navila` | plan-only; current tree BLOCKED (`COMMON_CASE_SOURCE_MISSING`) |
| `lightnav` | `lightnav` | plan-only; current tree BLOCKED (`COMMON_CASE_SOURCE_MISSING`) |
| `qwen_agent_proxmpc` | `qwen_prox_a` | BLOCKED: `QWEN_AGENT_PROXMPC_RUNTIME_PROFILE_MISSING`; audit-only identity is still readable |
| `strict_b_on` / `strict_b_off` | `strict_b` | BLOCKED: `STRICT_B_EPISODE_NOT_FROZEN` |
| `socialmpc_oracle` | `oracle` | BLOCKED: `ORACLE_NOT_OPERATOR_EXECUTABLE` |

`method-plan` also checks the **current** profile/audit identity file against
the bundle's pinned profile SHA. A current-file drift is
`METHOD_BUNDLE_CURRENT_PROFILE_DRIFT`/BLOCKED. The common case world/scenario/
robot must exist and match the current benchmark profile evaluation; otherwise
`COMMON_CASE_SOURCE_MISSING` / `COMMON_CASE_PROFILE_MISMATCH`. Historical
pinned digests remain auditable, but they do not make the current tree READY.
The external `8bb1896` eval500 swap means the current InternNav profile is
`402704…` while the pinned bundle profile is `81b030…`; the current common
case `grscenes_20_v1/default_2` is absent, so internnav/navila/lightnav are
correctly BLOCKED until the profile/common contract is re-frozen.

`method-compare --common <json> --output <fresh-json>` builds all bundles and
returns `common_hash_identical`, `private_hashes_distinct`,
`conflict_blockers`, and `conflict_warnings`. Duplicate ports/topics/output
prefixes are blockers; shared namespace/container/source-binding/lock are
warnings because methods are mutually exclusive and serialized by the shared
lock. Unknown or unready methods return `commands=[]` with typed blockers.
No future launch command is generated.

The public entry has no `--bundle` override; a `--bundle` argument returns
`BUNDLE_OVERRIDE_NOT_ALLOWED`/rc2. Each plan carries
`bundle_source_digest` and `canonical_bundle_path`, so model/container/
entrypoint/revision/sensors/controller/profile all come from the same
canonical repo bundle. A forged profile SHA is
`METHOD_BUNDLE_PROFILE_SHA_MISMATCH`; an internal override bundle is
`METHOD_BUNDLE_SOURCE_MISMATCH`/rc2. The profile SHA is checked against the
bundle's pinned `source_commit`, so unrelated working-tree dirt does not
change the canonical identity.

## Operator request schema

Schema: `configs/agentic_eval/operator_request.v1.schema.json`.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `schema` | `arena.agentic.operator_request.v1` |
| `request_id` | safe stable identity (`[A-Za-z0-9._-]`, no `..`) |
| `mode` | `single_case` / `full_eval` / `audit` |
| `track` | `qwen_prox_a` (A, audit-only) / `generic_internnav` / `strict_b` / `oracle` |
| `case` | `{world, scenario, robot}` |
| `episodes`, `seed`, `timeout` | frozen case fields; `episodes` must be exactly 1 |
| `profile` | versioned benchmark profile path |
| `model`, `model_revision` | must match the loaded profile identity |
| `action_control` | `common_follow_path` / `none` |
| `expected_outputs` | requested artifact list |

Optional binding fields:

| Field | Meaning |
| --- | --- |
| `workspace_root` | real host workspace (default `/home/ubuntu/arena_jazzy_ws`) |
| `arena_dir` | explicit Arena source root (default `<workspace>/src/Arena`) |
| `source_binding` | explicit source binding (default `arena_dir`) |
| `hunav_config`, `hunav_config_sha256` | pinned HuNav config path and SHA256 |
| `risk_threshold`, `max_frame_gap_sec` | explicit strict-B smoke overrides |

`paper_pipeline agentic-request --file <json>` validates the request and
returns:

- `executable`
- `commands` (empty when blocked)
- `env` for the resolved operator command
- `blockers`
- `setting_status`
- `expected_outputs`
- resolved profile/HuNav SHA/model/action-control identity

It never starts a run.

### Binding rules

- The benchmark profile is loaded with the same
  `arena_bringup.benchmark_profile` loader the real entry point uses.
- `model` must equal the profile server mode; `model_revision` must equal the
  profile checkpoint basename. A mismatch yields
  `MODEL_PROFILE_MISMATCH` / `MODEL_REVISION_PROFILE_MISMATCH` and
  `commands=[]`.
- The generated command carries `--profile`, `--case`, `--robot`,
  `--episodes 1`, `--pedestrian-seed`, and `--timeout`, so robot and seed
  changes are visible in the real effective config.
- `workspace_root` is the host workspace, never the source worktree.
  `arena_dir`/`source_binding` are explicit and must exist.
- `audit` and `full_eval` are currently `UNSUPPORTED_MODE:<mode>`; they are
  never mapped to `run`.
- `episodes != 1`, unsafe ids, path traversal, and non-finite numeric values
  fail closed (`valid=false`, rc=2).
- HuNav config path is resolved to an absolute path and its SHA256 is
  forwarded as `ARENA_HUNAV_CONFIG` / `ARENA_HUNAV_CONFIG_SHA256`.

### Examples

| Example | Status |
| --- | --- |
| `configs/agentic_eval/examples/generic_internnav_single_case.json` | executable operator plan; profile `internnav_grscenes`, model `internnav`, revision `InternVLA-N1-DualVLN` |
| `configs/agentic_eval/examples/strict_b_single_case.json` | blocked: `QWEN_LISN_COMMON_METRIC_ACTION_BRIDGE_MISSING` |

The strict-B request is non-executable because the Qwen/LISN sidecar emits
pixel/bearing/STOP and has no common metric FollowPath action bridge. It is
`SMOKE_ONLY_NOT_PAPER_SETTING`; no Path is fabricated.

## Operator lifecycle response

Schema: `configs/agentic_eval/operator_lifecycle.v1.schema.json`.
Examples: `operator_lifecycle_plan.example.json`,
`operator_lifecycle_read.example.json`,
`operator_lifecycle_audit.example.json`.

`paper_pipeline agentic-lifecycle --file <request> --action <action>
[--from-state <state>]` returns a deterministic response.

| Action | Bound command | State | Allowed next |
| --- | --- | --- | --- |
| `plan` | `benchmark/main plan run ...` (dry-run) | `planned` | `start`, `cancel`, `read`, `audit` |
| `start` | none: `START_RUNTIME_IDEMPOTENCY_GUARD_MISSING` | `blocked` | none |
| `cancel` | `benchmark/main stop` | `cancelled` | `read`, `audit` |
| `read` | `evidence-index` + `evidence-verdict` in fresh `output_root` | `read_ready` | `audit` |
| `finalize` | none: `ACTION_NOT_BOUND:finalize` | `blocked` | none |
| `audit` | `benchmark_result --require-valid` + evidence tools + `metrics` or `source-run-audit` + `audit-consolidate` | `audit_ready` | none |

`read`/`audit` require `run_dir` and a non-existing `output_root` outside the
run. They bind the request to the run's actual manifest identity:
`track/model/model_revision/profile/case/robot/seed/episodes/action_control/
sensor_profile`. The actual model comes from `model_backend`/
`model_provenance`, never from a legacy launch/profile adapter field, so an
InternNav request cannot read a Qwen run. A mismatch is
`RUN_IDENTITY_MISMATCH:<field>` with `commands=[]`.

`qwen_prox_a` is the A role: Qwen FP8 + RGB + real RTX LiDAR + common
FollowPath + Prox. Its contract is
`configs/agentic_eval/profiles/qwen_prox_a_audit_identity.v1.json`
(`AUDIT_ONLY`, `launchable=false`). The correct Qwen A request
(`qwen_a_single_case.json`) is the positive read/audit example; its
plan/start are `AUDIT_ONLY_TRACK_NOT_LAUNCHABLE`. The generic InternNav
request against the Qwen run is the negative example. A request that pairs an
InternNav profile with a Qwen model is `MODEL_PROFILE_MISMATCH` and is blocked
for read/audit too. `PROFILE_INVALID` and `MODEL_PROFILE_MISMATCH` are
validity blockers, not launch-only blockers.

`finalize` is `ACTION_NOT_BOUND:finalize` because no safe flush/finalize
entry exists; it is not an alias for evidence read. `read`/`audit` never
write the source or create missing evidence.

Every response carries:

- `request_id`, `action`, `from_state`, `state`, `executable`
- `idempotency_key` = `sha256:<canonical request_id + action + run/output +
  identity>`
- `exit_taxonomy` (`ok=0`, `scientific_fail=1`,
  `contract_or_evidence_blocked=2`)
- `expected_outputs`, `preconditions`, `cleanup`
- `allowed_next_actions` from the frozen state machine
- `blockers` / `errors`; unbound or invalid actions always return
  `commands=[]`
- `run_dir`, `output_root`, `sensor_profile`, and `audit_manifest` when
  applicable

Invalid `--from-state` transitions return
`INVALID_STATE_TRANSITION:<from>-><action>`. Repeated same request+action
responses are byte-identical; `read`/`audit`/`finalize` also fail closed if
the output root already exists. Repeated `start` is blocked, so it cannot
create a second run/output.

Strict-B and other non-executable requests still support read/audit/finalize
because those actions only consume an existing run; plan/start inherit the
model-execution blockers.

## Recording contract and preflight

Schemas: `recording_contract.v1.schema.json`,
`recording_preflight.v1.schema.json`.
Profiles: `configs/agentic_eval/recording/a_rgb_lidar.json`,
`configs/agentic_eval/recording/strict_b_rgb_only.json`.

```bash
python3 -m arena_bringup.paper_pipeline recording-contract \
  --file configs/agentic_eval/recording/a_rgb_lidar.json
python3 -m arena_bringup.paper_pipeline recording-preflight \
  --contract configs/agentic_eval/recording/a_rgb_lidar.json \
  --run-dir <run> --output <fresh-json>
```

The contract declares the future bag path, `mode=ro`, recording command,
required ROS types/topics, minimum messages, and playback policy. A requires
`Image` + `CameraInfo` + `LaserScan`; strict-B requires `Image` +
`CameraInfo` only and rejects scan/depth/PointCloud/People/GT topics.
Preflight only reads an existing bag; it never creates missing evidence.
The current A run is `BLOCKED` (RGB/CameraInfo/LaserScan absent); the
contract does not backfill history.

## Audit consolidation

`paper_pipeline audit-consolidate --manifest <json> --output <fresh-json>`
combines `benchmark_result`, `evidence_index`, `evidence_verdict`, and the
track-applicable `metrics_receipt` or `source_run_audit` into one typed
receipt. Precedence is `contract_blocked > playback_blocked >
scientific_fail > artifact_complete`.

The manifest must carry the same provenance envelope as every receipt:

```json
{
  "schema": "arena.agentic.provenance.v1",
  "request_id": "...",
  "run_dir": "/canonical/run",
  "run_identity_digest": "sha256:...",
  "source_tree_digest": "sha256:...",
  "source_file_set_digest": "sha256:...",
  "immutable": true
}
```

`audit-consolidate` rejects with rc2 when:

- the manifest has no provenance: `AUDIT_MANIFEST_PROVENANCE_MISSING`
- a receipt has no envelope: `RECEIPT_PROVENANCE_MISSING:<name>`
- a receipt envelope differs on request/run/digest:
  `RECEIPT_PROVENANCE_MISMATCH:<name>:<field>`
- a receipt path is inside the source run or outside the fresh output root
- a required receipt is missing: `RECEIPT_MISSING:<name>`

Minimal JSONs without request/run/source identity can no longer become
`artifact_complete`. Cross-run receipt mixing is a typed mismatch. It never
rewrites a scientific FAIL into success.

## Provenance envelope

Every audit receipt (`benchmark_result`, `evidence_index`,
`evidence_verdict`, `metrics_receipt`, `source_run_audit`) carries the same
`provenance` object. The lifecycle response includes it inline and a
`provenance_path`; write it to that path before running the audit commands.
The envelope binds the request, the canonical run directory, the full actual
run identity (from `model_backend`/`model_provenance`), and the immutable
source tree/file-set digests.

Producers that accept `--provenance-file` recompute the envelope from
`request_id` + canonical `run_dir` before writing. `audit-consolidate`
recomputes from `manifest.request_id` + `manifest.run_dir` and compares the
manifest, `request_identity`, and every receipt. Merely agreeing with each
other is not enough: a consistent forged envelope or a receipt from another
run is `PROVENANCE_AUTHENTICITY_MISMATCH:<field>`/rc2. Missing or mismatched
provenance is a contract block, never a PASS.

## Evidence index

`paper_pipeline evidence-index --run-dir <run> --output <fresh-json>
[--sensor-profile rgb_lidar|rgb_only|none]` writes a fresh JSON and never
writes the run directory. Output inside the run directory is rejected.

It records:

- each expected artifact: path, size, SHA256, presence
- `video_index.episodes[]` entries: ego, map/top-down, sim top-down, and
  optional debug-overlay videos, with run-relative ownership, size, SHA256,
  aliases, and the per-episode `video_frame_pts.json`
- existing `DECODE*` / `TIMEBASE*` audit JSON files (run or parent scope) with
  their schema and whether they reference this run
- bag topics/types/message counts read with `mode=ro`

Sensor requirements come from `--sensor-profile` and are classified by ROS
message type, not topic-name substrings:

| Profile | Required |
| --- | --- |
| `rgb_lidar` (default) | `sensor_msgs/msg/Image`, `CameraInfo`, `LaserScan` |
| `rgb_only` (strict B) | `Image`, `CameraInfo`; **no** LaserScan requirement |
| `none` | no sensor requirement |

`playback_status` reflects bag replay only. Missing video index or external
video paths are reported in `artifact_blockers`, never as playback success.
Missing PTS/decode evidence is reported as `UNKNOWN`; the index does not claim
decode/timebase PASS without an audit that references the run.

Current A run fact: `cmd_vel`, `odom`, and human messages exist, but the
LaserScan count is zero and RGB/CameraInfo are not in the bag. The index
therefore reports `playback_status=BLOCKED`; it does not claim complete
replay.

Actual evidence index for the saved 2035 A run (2092 increment):

```text
outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2/evidence_index_2092/evidence_index.json
sha256=adc23d78417135b9b4672b18d282d902ca4749a502134970d28574d3645588a6
playback_status=BLOCKED
blockers=[RGB_NOT_IN_BAG, CAMERA_INFO_NOT_IN_BAG, LASER_SCAN_ZERO_MESSAGES]
videos: episode_0/ego_video, episode_0/top_down_video (alias map_top_down_video), episode_0/sim_top_down_video; all present, RUN_RELATIVE
video_frame_pts: episode_0 present
decode=AVAILABLE_NOT_RUN_SCOPED (PARENT:DECODE_v2.json)
timebase=AVAILABLE_UNVERIFIED (PARENT:TIMEBASE_AUDIT_v2.json)
bag counts: odom=337, cmd_vel=127, human_states=337; typed scan topics=0
```

## Evidence verdict

Schema: `configs/agentic_eval/evidence_verdict.v1.schema.json`.
Example: `configs/agentic_eval/examples/evidence_verdict_playback_blocked.example.json`.

`paper_pipeline evidence-verdict --index <evidence-index.json> --output
<fresh-json>` is deterministic: the same index yields the same
`input_sha256` and verdict. `evidence-index` also embeds this verdict.

| Verdict | Meaning | Exit |
| --- | --- | --- |
| `artifact_complete` | required artifacts and videos present, bag READY, metrics known with no task/social fail | 0 |
| `scientific_fail` | evidence playable and complete enough, metrics show task or social fail | 1 |
| `playback_blocked` | bag is not READY (RGB/CameraInfo/required LaserScan missing) | 2 |
| `contract_blocked` | invalid index, missing bag/video index, or metrics evidence UNKNOWN | 2 |

Precedence is `contract_blocked > playback_blocked > scientific_fail >
artifact_complete`. Missing PTS/decode/timebase evidence stays in `unknowns`
and never becomes PASS. `scientific_fail` is a valid scored result, not a
pipeline failure.

### Actual request receipts

| Request | rc | Result |
| --- | --- | --- |
| `generic_internnav_single_case.json` | 0 | `executable=true`, 1 profile-bound command, no blockers, `GENERIC_INTERNAV_SMOKE_ONLY`; `benchmark/main plan run` parse rc=0 |
| `strict_b_single_case.json` | 0 | `executable=false`, `commands=[]`, blocker `QWEN_LISN_COMMON_METRIC_ACTION_BRIDGE_MISSING`, `SMOKE_ONLY_NOT_PAPER_SETTING` |
| generic lifecycle `plan` | 0 | state `planned`, dry-run command, idempotency key |
| generic lifecycle `start` | 2 | state `blocked`, `START_RUNTIME_IDEMPOTENCY_GUARD_MISSING`, `commands=[]` |
| generic lifecycle `cancel` | 0 | state `cancelled`, `benchmark/main stop` |
| generic lifecycle `read` | 0 | state `read_ready`, evidence-index + evidence-verdict in fresh `output_root` |
| generic lifecycle `finalize` | 0 | state `finalize_ready`, read-only evidence tools |
| generic lifecycle `audit` | 0 | state `audit_ready`, 5 steps incl. `audit-consolidate`, inline manifest |
| invalid transition `started->plan` | 2 | `INVALID_STATE_TRANSITION:started->plan`, `commands=[]` |
| current A recording preflight | 2 | `BLOCKED`, RGB/CameraInfo/LaserScan topics missing |
| saved 2035 evidence verdict | 2 | `playback_blocked`; secondary `scientific_fail`; `artifact_complete=false`; `ARTIFACT_MISSING:benchmark_result.json` in `unknowns` |

An empty source directory with `source-run-audit` returns rc=2 with
`MISSING_SOURCE_FILES`, `UNKNOWN_CONTROLLER_PEOPLE`, and
`UNKNOWN_EXECUTOR_IDENTITY`; `source_eligible=false`. A hash change during the
audit adds `SOURCE_HASH_MISMATCH`.

## Metrics history protection

`paper_pipeline metrics --run-dir <run> --derived-root <fresh-root>`:

- requires `--derived-root`; a derived root inside the source run is rejected
- copies the source into the derived run (directories are copied, never
  symlinked) and runs every stage there
- never writes `METRICS_OVERWRITE_INCIDENT.json` into the source; an existing
  historical incident is read-only and hashed into the receipt
- guards the source with a full file-tree snapshot (size, mtime, SHA256 for
  files up to 8 MiB) before and after the run
- reports `source_unchanged` and `source_changed_paths`; any change is a
  `source_immutable_guard` fault
- writes `PAPER_METRICS_RECEIPT.json` only into the fresh derived run

2092 recompute receipt:

```text
outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2/metrics_derived_2092/20261102_182650_grscenes_20_v1_Ai2_Bot2_lisn_qwen3vl_fp8/PAPER_METRICS_RECEIPT.json
sha256=090f2525e40977b62ebe1491d2568508c5870f7030793b51e9b7c4a35a0d2694
status=pass
source_unchanged=true
source_changed_paths=[]
historical_metrics_incident.read_only=true
```

The source run's four metrics files and the historical incident have the same
SHA256 before and after; the earlier `metrics_derived_2075` receipt is
preserved.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | artifact/metric pipeline completed |
| 1 | scientific task/social FAIL scored (valid result, not a crash) |
| 2 | contract/evidence/config failure (invalid or blocked) |

`artifact pipeline complete` is not `task/social success`. A TIMEOUT, missed
goal, collision, or social-threshold failure is a valid scored result. Missing
evidence, an invalid profile, or a blocked identity is `invalid/blocked`.

## Resources, idempotency, resume, cleanup

- Qwen lane <= 40 GiB; ordinary Arena lane <= 10 GiB; do not borrow.
- Model cache is read-only on NFS; no cloud API and no new weights.
- Output root: `outputs/<prefix>/<timestamp>_<world>_<robot>_<mode>/`.
- One run per output directory; historical runs are immutable.
- A busy shared lock means `LOCK_BUSY`: do not start a second run.
- Stop with `benchmark/main stop` and the lane cleanup; then verify the lock
  is free and no model/Isaac/eval process remains.
- Resume only from a clean stop; do not reuse a dirty run directory.
