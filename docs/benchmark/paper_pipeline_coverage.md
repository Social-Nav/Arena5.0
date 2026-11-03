# Paper Pipeline Coverage (A/B/C Tracks)

This page maps the current Arena5 engineering entry points to the planned paper
experiments. It is a coverage ledger, not a results table. It does not freeze
scene/episode counts, taxonomy/splits, or instruction QC; those fields are
explicitly `not set`.

Source: user-provided paper outline (manual group read):
<https://my.feishu.cn/docx/P7OCdJemRoByaQxo134ckQj4nYg>. The docx was not
downloaded or parsed in this turn; no numeric claims are taken from it.

## Track contracts

### A: LISN-style RGB + real LiDAR + ProxMPC (strong baseline)

- Allowed input: real Isaac RGB, same-source real 2D LiDAR, camera calibration,
  public ego odometry and sensor extrinsics.
- Forbidden input: simulator depth, global GT map, HuNav Agents/People truth,
  benchmark true goal/start-goal, USD object poses/semantics, oracle trajectories.
- Slow loop: Qwen3-VL-8B-FP8 structured goal/bearing/STOP. A pixel/bearing may
  become a metric map goal only through a same-source, bounded-time real LiDAR
  range; no range means fail-closed.
- Control: one common `FollowPath` sender through ProxMPC, smoother, and the
  independent CollisionMonitor backstop. No second sender.
- The 2035 diagnostic run is A-track engineering evidence. It is not a
  scientific PASS.

### B: strict RGB-only System-1 fast-reactivity ablation

- Allowed input: RGB, instruction, public ego odometry, and the same Qwen/prompt/
  high-level action mapping as A.
- Forbidden input: scan/PointCloud, simulator GT depth, GT map, HuNav tracks,
  GT semantics, eval goal, untrusted metric depth.
- Guard: RGB-only optical-flow risk proxy (`compute_flow_risk_proxy`), not a
  learned policy and not a causal metric-depth TTC. It may only veto to STOP.
- Status: CPU contract + real guard ROS smoke + real ArenaModelExecutor ROS
  smoke are `READY`. A full B episode is `BLOCKED` until the B case, risk
  threshold, and max frame gap are frozen; the runner profile selection is
  `BLOCKED` for full eval, though `--bfast-profile` plan/dry-run is verified.
- B must not inherit A's LiDAR benefit; A and B are separate manifests.

### C: GT teacher / oracle

- GT global map, HuNav People/Agents truth, and eval endpoint may be used only
  for teacher trajectory synthesis, platform validation, or a separately
  reported oracle upper bound.
- Old GT runs remain GT runs; they are not relabeled non-GT.
- Status: export-manifest contract is `READY` (CPU); a real teacher/oracle
  sample is `BLOCKED` (`missing_source_sample`); collection is `PLANNED`.
  No teacher synthesis or oracle episode is implemented in this lane.

## Coverage ledger

Status legend:

- `READY`: code entry exists and has an actual executed receipt.
- `IMPLEMENTED_UNVERIFIED`: code/CLI exists but was not executed in the current
  receipt.
- `BLOCKED`: required dependency or decision is missing; no fabricated PASS.
- `SEALED`: intentionally not active in this lane.

| Paper item | Code entry / config | Actual verification command + receipt | Status | Missing decision / dependency | Minimal next step |
| --- | --- | --- | --- | --- | --- |
| A: model consume | `arena_vln_models/runner.py`, `arena_bringup/arena_bringup/lisn_slow_loop_ros.py` | `MODE=diag bash run_prox_qwen_lane.sh`; `VLM_TRACE.jsonl` 17 responses, 5 accepted / 12 rejected | READY | none | keep the same frozen case |
| A: pixel/bearing → real LiDAR range | `lisn_slow_loop_ros.py` scan grounding; profile `slow_loop_grounding_scan_prox_v1.json` | same run; `grounding` typed status and `scan` fields in `VLM_TRACE.jsonl` | READY | none | add more range-valid entities only with new evidence |
| A: source-time TF map→odom goal | `lisn_slow_loop_ros.build_grounded_goal_msg`, `prox_diag_recorder.transform_pose_stamped_to_odom` | 2035 `PROX_DIAG.jsonl` goal `tf_status=OK`, `result_type=PoseStamped`, stamp `27716668112` | READY | none | keep source stamp/frame immutable |
| A: unique FollowPath sender | `arena_bringup/launch/arena.launch.py`, common runner, `nav2_follow_path_prox_mpc` | `artifact_validation.json` model_control pass, `goal_accepted_count=1`, `follow_path_goal_count=1` | READY | none | no second sender |
| A: motion / STOP / timeout | ProxMPC controller, smoother, CollisionMonitor | `episode_outcome.json` `sim_timeout`; `cmd_vel.csv`; `RUN_TIMING.json` episode sim 60.2 s | READY (engineering) | scientific success is not expected from this case | preserve TIMEOUT verdict |
| A: metrics | `arena_evaluation` metrics / vln / social | `vln_task_metrics.json` strict task false; `social_metrics.json` strict social false; `artifact_validation.json` overall_pass true | READY | manual recompute not re-run this turn | run the postprocess command in `prox_mpc_a_track_runbook.md` |
| A: playable video | `render_prox_diag_video.py --name-suffix v2` | `ego_vlm_debug_v2.mp4` 274 frames; `local_costmap_prox_debug_v2.mp4` 386 frames; `DECODE_v2.json` | READY | none | keep v1 files untouched |
| A: v2 timebase audit | renderer `TIMEBASE_AUDIT_v2.json` | `xN` first frame `N/A`, later `1.0`; zero huge/non-finite; held counts 258/233 | READY | none | no `1e-9` denominators |
| A: exact source RGB lookup | renderer `_exact_frame_lookup` | `SOURCE_PATH_MATCH_v2.json` 17/17 `MATCH`; negative test `test_exact_source_lookup_never_falls_back_to_nearest` | READY | none | no nearest/guess match |
| A: panel layout | renderer `_draw_panel` | `TIMEBASE_AUDIT_v2.json` `clipped_frames=0`, `max_line_width_px<=228`; panel PNGs | READY | none | keep request/update/units/typed reason visible |
| A: semantic costmap in Prox solver | none for A | old SocialMPC D evidence is sealed; no A-track semantic consumer | BLOCKED | A-track semantic grid and Prox solver consumer are not implemented | implement a separate LISN-only semantic layer, prove numeric optimizer consumption |
| A: scenario intervention | `paper_pipeline scenario`; `arena_simulation_setup` | `pytest arena_bringup/test/test_paper_pipeline.py` 37 passed rc0; CLI smoke rc0/rc2 | READY (CPU contract) | episode arm not run | freeze perturbation manifest fields |
| A: static / open-loop / closed-loop controls | `paper_pipeline scenario` profiles | static/closedloop SUPPORTED; openloop typed UNSUPPORTED | READY (CPU contract) | no scripted-human interface for openloop | freeze arm definitions and scripted-human source |
| A: perception on/off | A vs B manifests | A run exists; B CPU contract exists | IMPLEMENTED_UNVERIFIED | B episode arm not run | run B on/off pair after freeze |
| A: latency injection | `paper_pipeline latency` + `ReleaseQueue` + `LatencyCommitGate` | CPU two-response + real LisnSlowLoopNode/ROS-clock smoke rc=0: B then A release, exact-once, drop no commit, source unchanged | READY (CPU + node smoke) | real episode not exercised | freeze latency values |
| A: dataset manifest validation | `paper_pipeline dataset` | CPU: missing/duplicate/GT + scene/task/asset split leakage | READY (CPU contract) | scene/episode counts not frozen | freeze manifest counts |
| A: metrics fail-close wrapper | `paper_pipeline metrics` | EXECUTED on a fresh derived root: all stages rc0; full source-tree `source_unchanged=true`; existing source incident read-only; scientific TIMEOUT/social false unchanged | READY | none | use as the stable recompute entry |
| B: off/on launch + profile | `bfast_off.launch.py`, `bfast_on.launch.py`, `configs/lisn/bfast_manifest_v1.json`, `bfast_off_v1.json`, `bfast_on_v1.json` | installed overlay `--show-args` rc=0 for both; `bfast-profile` rc=0; both use same RGB/model/prompt/action | READY (installed launch/profile) | full B episode not run | run B on/off pair after freeze |
| B: runner/eval parameter chain | `runner --bfast-profile` -> `internnav_eval` -> `arena.launch.py` -> `task_generator` -> `robot_manager` -> `follow_path.launch.py` | full-eval dry-run rc=0 but `executable=false`, `commands=[]`; top-level model=Qwen, model_backend/adapter=`lisn_qwen3vl_fp8`, runner_entry=internnav only; no InternNav 5801/eval_dual or InternVLA checkpoint | READY (blocked plan) | Qwen sidecar has no common metric action/Path bridge; full B eval typed BLOCKED | implement the Qwen→common FollowPath bridge after freeze |
| B: static/closed-loop smoke profiles | `scenario_static_smoke_v1.json`, `scenario_closedloop_smoke_v1.json` | `paper_pipeline scenario` rc=0; generated internnav_eval argv parse-only rc=0; closed-loop pins repo path + SHA256 `a834b6bb...`; resolved absolute `ARENA_HUNAV_CONFIG`; `SMOKE_ONLY_NOT_PAPER_SETTING` | READY (smoke-only) | no paper setting frozen | freeze scenario settings |
| B: strict RGB-only on/off | `paper_pipeline bfast` + `compute_flow_risk_proxy` + `bfast_guard_node` | CPU: on/off same common action, flow-risk veto, bare point/bearing UNSUPPORTED STOP, unknown units fail-close | READY (CPU contract) | B episode arm not run | run B on/off pair after freeze |
| B: guard → executor real ROS path | `ros_bfast_guard_smoke.py` (ROS_DOMAIN_ID=97) | EXECUTED rc=0: warmup/clear allow follow_path; invalid/direct veto STOP; stale required STOP, stale optional allow | READY (ROS smoke) | full B episode not run | run B on/off pair after freeze |
| B: actual executor guard path | `ros_bfast_executor_smoke.py` (ROS_DOMAIN_ID=98) | EXECUTED rc=0: real ArenaModelExecutor + real guard + FollowPath server; clear sends path; direct/invalid/required stale STOP; trace evidence | READY (ROS executor smoke) | full B episode not run | run B on/off pair after freeze |
| B: action → Twist, follow_path/stop/error/hold | `paper_pipeline bfast` + executor `fast_guard_veto_topic` | CPU: follow_path/stop/hold/error mapping; bearing UNSUPPORTED; executor veto hook present | READY (CPU contract) | full executor Twist integration not run | run B on/off pair after freeze |
| B: no scan/depth/GT input | `paper_pipeline bfast` denylist + `bfast_guard_node` static check | CPU: scan/depth/people/GT topics rejected; node subscribes RGB only | READY (CPU contract) | none | keep denylist in the B launch |
| C: GT teacher/oracle | `paper_pipeline teacher-audit`; `source-run-audit`; old GT SocialMPC/HuNav runs | template audit rc=2 `TEACHER_SOURCE_SAMPLE_MISSING`; candidate rc=2 `CONTROLLER_PEOPLE_ZERO`+`EXECUTOR_IDENTITY_INTERNNAV`; empty/incomplete input rc=2 `MISSING_SOURCE_FILES`/`UNKNOWN_CONTROLLER_PEOPLE`/`UNKNOWN_EXECUTOR_IDENTITY`/`SOURCE_HASH_MISMATCH`; `gt_available=true`, `gt_consumed_by_controller=false`, `source_eligible=false`, `teacher_export_created=false`, `old_gt_labels_preserved=true` | BLOCKED (typed) | no eligible teacher sample; controller people=0 and InternNav executor in candidate | fix controller people/executor identity or provide a clean Qwen/LISN teacher source |
| C: teacher trajectory synthesis | `paper_pipeline teacher` manifest | manifest contract only | PLANNED | synthesis contract not frozen | define teacher dataset and oracle upper bound |
| Operator: agentic request | `configs/agentic_eval/operator_request.v1.schema.json`, `paper_pipeline agentic-request` | generic example rc=0 `executable=true`, profile-loader-bound `--profile/--case/--robot/--episodes 1/--pedestrian-seed`, profile-matched model/revision, resolved HuNav SHA, workspace vs repo root split; `benchmark/main plan run` parse rc=0; robot/seed changes visible in effective config; model mismatch and `audit`/`full_eval` fail closed; strict-B rc=0 `executable=false` + bridge blocker | READY (plan) | full B not executable | implement the Qwen→common FollowPath bridge after freeze |
| Operator: evidence index | `paper_pipeline evidence-index` | 2035 run rc=0: `video_index.episodes[]` 3 videos indexed with run-relative ownership/aliases/PTS; parent DECODE/TIMEBASE indexed as `AVAILABLE_NOT_RUN_SCOPED`/`AVAILABLE_UNVERIFIED`; bag typed (mode=ro); `playback_status=BLOCKED` (RGB/CameraInfo absent, scan zero); `--sensor-profile rgb_only` does not require LiDAR; odom=337/cmd=127/human=337; receipt saved | READY | A bag lacks RGB/scan | add RGB/CameraInfo/LaserScan to the bag for full replay |
| Operator: lifecycle response | `configs/agentic_eval/operator_lifecycle.v1.schema.json`, `paper_pipeline agentic-lifecycle` | plan/cancel bound; start typed `START_RUNTIME_IDEMPOTENCY_GUARD_MISSING`; read/audit bind to actual run identity (`qwen_prox_a` A role; `RUN_IDENTITY_MISMATCH:<field>`); `PROFILE_INVALID`/`MODEL_PROFILE_MISMATCH` block read/audit; finalize typed `ACTION_NOT_BOUND:finalize`; invalid transitions typed | READY (plan/read/audit) | A plan/start audit-only; finalize/start blocked | add a real A run profile and a safe finalize entry |
| Operator: audit consolidation | `audit_consolidated.v1.schema.json`, `paper_pipeline audit-consolidate` | recomputes provenance from manifest request_id/run_dir and compares manifest + request_identity + each receipt; consistent forged/cross-run envelopes `PROVENANCE_AUTHENTICITY_MISMATCH`/rc2; minimal fixtures `contract_blocked`; never rewrites scientific FAIL | READY (CPU + fixture receipts) | audit commands not executed this turn | run once on a fresh derived root |
| Operator: recording contract/preflight | `recording_contract.v1.schema.json`, `recording_preflight.v1.schema.json`, A/B profiles | A requires RGB+CameraInfo+LaserScan; strict-B requires RGB+CameraInfo only and rejects scan/depth/People/GT; `mode=ro`; current A run `BLOCKED` with three missing topics | READY (contract/preflight) | current A bag lacks RGB/CameraInfo/LaserScan | record a new A bag with the contract |
| Operator: method routing | `configs/agentic_eval/methods/*.v1.json`, `benchmark/main plan --method`, `paper_pipeline method-plan/method-compare` | 7 bundles; common hash identical; private hashes distinct; current-source drift check (`METHOD_BUNDLE_CURRENT_PROFILE_DRIFT`) and common case source check (`COMMON_CASE_SOURCE_MISSING`/`COMMON_CASE_PROFILE_MISMATCH`); under current eval500 drift internnav/navila/lightnav BLOCKED; qwen/strictB/oracle typed BLOCKED; no public `--bundle` | READY (gate implemented) | current profile/common case drifted by external `8bb1896`; no runtime | re-freeze profile/common contract and bundle source_commit before any real run |
| Operator: evidence verdict | `configs/agentic_eval/evidence_verdict.v1.schema.json`, `paper_pipeline evidence-verdict` | deterministic `input_sha256`; precedence contract > playback > scientific > artifact; saved 2035 verdict `playback_blocked` + secondary `scientific_fail`, `artifact_complete=false`, `ARTIFACT_MISSING:benchmark_result.json`; exit 2 | READY | no PASS upgrade for UNKNOWN/BLOCKED | keep verdict policy versioned |

## Frozen vs not set

| Field | State |
| --- | --- |
| A case | `grscenes_20_v1/default_2`, `Ai2_Bot2`, 2 humans, seed 0 |
| A episode cap | configured 59 s, hard target 60 s |
| Model pin | `Qwen/Qwen3-VL-8B-Instruct-FP8`, revision `9cdc6310a8cb770ce18efaf4e9935334512aee45` |
| Scene count | `not set` |
| Episode count / repeats | `not set` |
| Taxonomy / splits | `not set` |
| Instruction QC | `not set` (teacher manifest fails closed until required bounds are frozen) |
| B guard implementation | CPU + real guard ROS + real executor ROS smoke READY; full B episode BLOCKED; openloop arm UNSUPPORTED |
| B runner/eval chain | dry-run rc=0 with `executable=false`, `commands=[]`; top-level model=Qwen, backend/adapter=`lisn_qwen3vl_fp8`, runner_entry=internnav only |
| B static/closed-loop smoke | `scenario_*_smoke_v1.json`; scenario command rc=0; generated internnav_eval argv parse rc=0 |
| C provenance audit | `teacher-audit` rc=2 typed `TEACHER_SOURCE_SAMPLE_MISSING`; `source-run-audit` candidate rc=2 `CONTROLLER_PEOPLE_ZERO`+`EXECUTOR_IDENTITY_INTERNNAV`; empty input rc=2 missing/unknown/hash blockers; old GT labels preserved; no fabricated provenance |
| Operator contract | root `AGENTS.md`, `docs/benchmark/agentic_eval.md`, schema + 2 examples; generic request executable, strict-B blocked |
| Evidence index | 2035 A run `playback_status=BLOCKED` (RGB/CameraInfo absent, LaserScan zero); receipt saved |
| B profile setting | canonical profile has no risk/gap; smoke overrides marked `SMOKE_ONLY_NOT_PAPER_SETTING`; default_2 is smoke case |
| B installed launch | `outputs/arena5_paper_overlay_20261103/install`; off/on `--show-args` rc=0 |
| Latency dispatcher | CPU + real LisnSlowLoopNode/ROS-clock smoke READY; real episode IMPLEMENTED_UNVERIFIED |
| Stable wrapper | stage exit-code fault injection READY; in-repo full lane run IMPLEMENTED_UNVERIFIED |
| C teacher synthesis | export manifest contract READY; provenance sample missing (typed BLOCKED); collection PLANNED |
| Metrics wrapper | EXECUTED on a fresh derived root; source_unchanged=true; incident recorded; all stages rc0 |

When a required field is `not set`, the manifest/command must fail closed and
explain the missing field. Do not infer counts or thresholds from the paper
outline.

## What this page does not claim

- It does not claim a scientific PASS for A, B, or C.
- It does not claim ProxMPC dynamic prediction success from
  `num_active_obstacles=4`; the static/dynamic split is `UNKNOWN`.
- It does not claim `planner_path` or CollisionMonitor success from zero
  messages.
- It does not claim B inherits A's LiDAR benefit.
- It does not claim the B flow-risk proxy is a learned policy or a causal
  metric-depth time-to-contact; it is an RGB-only optical-flow risk score and
  may only veto to STOP.
- It does not claim the C teacher manifest is a completed dataset; collection
  and counts remain not frozen.
- It does not relabel the sealed SF60 three-VLN runs or the 2028 GT runs.
