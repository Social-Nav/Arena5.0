# Arena5 Agentic Eval — Operator Contract

This repository uses AI agents as **operators**, not as navigation policies.

## Allowed role

An agent may:

- run preflight, plan, launch, cancel, and cleanup
- read and audit existing run artifacts
- validate profiles, requests, and evidence
- report exact commands, return codes, and artifact paths

An agent must not:

- act as a navigation policy or produce robot actions
- publish `Twist`, `FollowPath`, or any control command directly
- guess a metric distance or fabricate a Path from pixels/bearings
- change an unfrozen case, taxonomy, split, threshold, or scientific setting
- bypass the common Arena executor or the A/B/C input contracts
- consume GT map/People, eval goals, USD state, or simulator depth in non-GT arms

## Stable operator commands

```bash
src/Arena/_meta/docker/features/benchmark/main config
src/Arena/_meta/docker/features/benchmark/main doctor
src/Arena/_meta/docker/features/benchmark/main plan run
src/Arena/_meta/docker/features/benchmark/main run
src/Arena/_meta/docker/features/benchmark/main full-eval
src/Arena/_meta/docker/features/benchmark/main stop

ros2 run arena_bringup paper_pipeline <validator>
ros2 run arena_bringup benchmark_result --dir <run> --require-valid
```

When running the installed overlay outside the repo root, export
`ARENA_REPO_ROOT=<checkout>` so relative contract paths resolve.
`evidence-index` takes `--sensor-profile rgb_lidar|rgb_only|none`; strict-B
uses `rgb_only`, which does not require a LaserScan.

The `paper_pipeline` validators are `settings`, `bfast`, `bfast-profile`,
`latency`, `scenario`, `dataset`, `teacher`, `teacher-audit`,
`source-run-audit`, `metrics`, `agentic-request`, `agentic-lifecycle`,
`audit-consolidate`, `evidence-index`, `evidence-verdict`,
`recording-contract`, `recording-preflight`, `method-plan`, `method-compare`,
`endpoint`, and `prefix`.

## Method routing

The only stable method-plan entry is:

```bash
src/Arena/_meta/docker/features/benchmark/main plan --method <id> \
  --common configs/agentic_eval/methods/common_contract.v1.json \
  --output <fresh-json>
```

It returns a machine-readable plan and never starts a container, model,
Isaac, or episode. The common contract is hashed once; switching method must
keep `common_contract_hash` identical and change only `method_private_hash`.

| Method ID | Role | Status |
| --- | --- | --- |
| `internnav` | `generic_internnav` | plan-only; current tree BLOCKED (`METHOD_BUNDLE_CURRENT_PROFILE_DRIFT`) |
| `navila` | `navila` | plan-only; current tree BLOCKED (`COMMON_CASE_SOURCE_MISSING`) |
| `lightnav` | `lightnav` | plan-only; current tree BLOCKED (`COMMON_CASE_SOURCE_MISSING`) |
| `qwen_agent_proxmpc` | `qwen_prox_a` | BLOCKED: `QWEN_AGENT_PROXMPC_RUNTIME_PROFILE_MISSING` |
| `strict_b_on` / `strict_b_off` | `strict_b` | BLOCKED: `STRICT_B_EPISODE_NOT_FROZEN` |
| `socialmpc_oracle` | `oracle` | BLOCKED: `ORACLE_NOT_OPERATOR_EXECUTABLE` |

`method-plan` also checks the **current** profile/audit identity file against
the bundle's pinned profile SHA. A current-file drift is
`METHOD_BUNDLE_CURRENT_PROFILE_DRIFT`/BLOCKED. The common case world/scenario/
robot must exist and match the current benchmark profile evaluation; otherwise
`COMMON_CASE_SOURCE_MISSING` / `COMMON_CASE_PROFILE_MISMATCH`. Historical
pinned digests remain auditable, but they do not make the current tree READY.

`method-compare --common <json> --output <fresh-json>` builds all bundles,
checks the common hash, private hashes, and port/topic/output-prefix/container/
namespace/source-binding conflicts. Unknown or unready methods return
`commands=[]` with typed blockers. The public entry has no `--bundle` override;
a `--bundle` argument is `BUNDLE_OVERRIDE_NOT_ALLOWED`/rc2. Each plan carries
`bundle_source_digest` and `canonical_bundle_path`, so model/container/
entrypoint/revision/sensors/controller/profile all come from the same
canonical repo bundle. A forged profile SHA is
`METHOD_BUNDLE_PROFILE_SHA_MISMATCH`; an override bundle is
`METHOD_BUNDLE_SOURCE_MISMATCH`.

## Operator lifecycle

`agentic-lifecycle --file <request> --action <action> [--from-state <state>]`
is the machine-readable operator response. Bound actions:

| Action | Bound command | State |
| --- | --- | --- |
| `plan` | `benchmark/main plan run ...` (dry-run) | `planned` |
| `start` | none: `START_RUNTIME_IDEMPOTENCY_GUARD_MISSING` | `blocked` |
| `cancel` | `benchmark/main stop` | `cancelled` |
| `read` | `evidence-index` + `evidence-verdict` in a fresh `output_root` | `read_ready` |
| `finalize` | none: `ACTION_NOT_BOUND:finalize` | `blocked` |
| `audit` | `benchmark_result --require-valid`, evidence tools, `metrics` or `source-run-audit`, then `audit-consolidate` | `audit_ready` |

`read`/`audit` require `run_dir` and a non-existing `output_root` outside the
run, and bind the request to the run's actual manifest identity
(`track/model/model_revision/profile/case/robot/seed/episodes/action_control/
sensor_profile`). A mismatch is `RUN_IDENTITY_MISMATCH:<field>` with
`commands=[]`. The actual model comes from `model_backend`/`model_provenance`,
never from a legacy launch/profile adapter field. `qwen_prox_a` is the
audit-only A role (Qwen FP8 + RGB + RTX LiDAR + common FollowPath + Prox);
its plan/start are `AUDIT_ONLY_TRACK_NOT_LAUNCHABLE`, while read/audit are
allowed after identity binding. `PROFILE_INVALID` /
`MODEL_PROFILE_MISMATCH` block read/audit too.

Every audit receipt carries the same provenance envelope (`request_id`,
canonical `run_dir`, full `run_identity`, `run_identity_digest`,
`source_tree_digest`, `source_file_set_digest`). Producers recompute it from
the immutable run before writing; `audit-consolidate` recomputes from
`manifest.request_id` + `manifest.run_dir`. Consistent forged or cross-run
envelopes are `PROVENANCE_AUTHENTICITY_MISMATCH:<field>`/rc2.
`finalize` stays `ACTION_NOT_BOUND` until a safe flush/finalize entry exists.
They never write the source or create missing evidence.
`start` stays blocked because the current entry has no runtime idempotency
guard; a hash alone is not claimed as dedupe. Every response carries a
deterministic `idempotency_key`, `exit_taxonomy`, `expected_outputs`,
`allowed_next_actions`, preconditions, and cleanup steps. Invalid
`--from-state` transitions are typed `INVALID_STATE_TRANSITION:<from>-><action>`.

`recording-contract --file <json>` validates a versioned future-bag contract;
`recording-preflight --contract <json> --run-dir <run> --output <fresh-json>`
checks the existing bag in `mode=ro`. A requires RGB+CameraInfo+LaserScan;
strict-B requires RGB+CameraInfo only. The current A bag is `BLOCKED`; the
contract never backfills history.

`audit-consolidate --manifest <json> --output <fresh-json>` combines
`benchmark_result`, `evidence_index`, `evidence_verdict`, and track-applicable
metrics/source-run receipts into one typed verdict.

`evidence-verdict --index <evidence-index.json> --output <fresh-json>`
classifies evidence as `artifact_complete`, `scientific_fail`,
`playback_blocked`, or `contract_blocked`. Precedence is contract, playback,
scientific, artifact. Missing/UNKNOWN evidence is never upgraded to PASS.

## Runtime topology and resources

- Three containers: `arena-arena_jazzy_ws-isaac-1` (sim/render),
  `arena-arena_jazzy_ws-arena-1` (eval/runner/recorder/renderer),
  `arena-qwen3vl-fp8` (Qwen model service).
- Shared GPU/runtime lock: `/tmp/arena_jazzy_ws_isaac_gpu_runtime.lock`.
- Qwen lane <= 40 GiB; ordinary Arena lane <= 10 GiB; no borrowing.
- Model cache is read-only on NFS; no cloud API, no new weights.
- Output root is `outputs/<prefix>/<timestamp>_<world>_<robot>_<mode>/`.
- Use one run per output directory; never overwrite a historical run.
- Resume only after a clean stop/cleanup; do not start a second run on a busy lock.

## Exit-code semantics

| Code | Meaning |
| --- | --- |
| 0 | the artifact/metric pipeline completed |
| 1 | a scientific task/social FAIL was scored (a valid result, not a crash) |
| 2 | a contract/evidence/config failure (invalid or blocked) |

`artifact pipeline complete` is not `task/social success`. A TIMEOUT, missed
goal, collision, or social-threshold failure is a valid scored result. Missing
evidence, an invalid profile, or a blocked identity is `invalid/blocked`.

See `docs/benchmark/agentic_eval.md` for the full request schema, examples,
evidence-index contract, and copy-able commands.
