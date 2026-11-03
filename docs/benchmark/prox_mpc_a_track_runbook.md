# ProxMPC A-Track Diagnostic Runbook

This page documents the real, copy-able entry points for the Arena5
LISN/Qwen + ProxMPC A-track diagnostic lane. It is a runbook, not a
scientific result page.

## Scope and status

| Item | Status | Evidence |
| --- | --- | --- |
| A-track diagnostic lane (`MODE=diag`) | EXECUTED | `qwen_diag_v2/20261102_182650_grscenes_20_v1_Ai2_Bot2_lisn_qwen3vl_fp8` |
| v2 offline re-render of the 2028 diag data | EXECUTED | `outputs/arena5_prox_isaac_a_20261102/diag/*_v2.*` |
| Source-time goal TF projection | EXECUTED | goal event `tf_status=OK`, `result_type=PoseStamped`, stamp `27716668112` |
| ProxMPC dynamic obstacle prediction | IMPLEMENTED_UNVERIFIED | native markers recorded; static/dynamic split not exposed |
| `planner_path` / CollisionMonitor intervention | UNKNOWN | zero messages in the run; not a pass |
| B-track strict RGB-only fast reactivity | CPU/ROS smoke READY; full B episode BLOCKED | guard + actual executor smoke; no B episode yet |
| B runner profile selection | BLOCKED | profiles/launch exist; runner `--bfast-profile` plan is verified, full eval wiring not |
| C-track GT teacher/oracle | manifest contract READY; sample BLOCKED; collection PLANNED | old GT runs are not relabeled |

The 2028 raw 300-sim episode under
`outputs/arena5_prox_isaac_a_20261102/qwen_episode/20261102_122056_grscenes_20_v1_Ai2_Bot2_lisn_qwen3vl_fp8`
is preserved and is not overwritten by this lane. The 2035 diagnostic run is
reported separately; it is not a scientific PASS.

## Three-container topology

| Container | Role | Must not do |
| --- | --- | --- |
| `arena-arena_jazzy_ws-isaac-1` | Isaac Sim render/physics and the RTX LiDAR publisher | Do not install ProxSuite/Nav2 Prox or run ROS eval nodes here |
| `arena-arena_jazzy_ws-arena-1` | evaluator, common runner, recorder, renderer, ProxMPC controller/tracker | Do not start a real model server here |
| `arena-qwen3vl-fp8` | Qwen3-VL-8B-FP8 vLLM service only | Do not install ROS or the Arena stack here |

The shared GPU/runtime lock is:

```bash
/tmp/arena_jazzy_ws_isaac_gpu_runtime.lock
```

Every lane must hold this lock from model startup through simulator/eval,
render/flush, and cleanup. Check it read-only first:

```bash
flock -n /tmp/arena_jazzy_ws_isaac_gpu_runtime.lock -c 'echo LOCK_FREE' || echo LOCK_BUSY
```

## A-track input contract

Allowed:

| Input | Topic / frame | Notes |
| --- | --- | --- |
| RGB | `/task_generator_node/Ai2_Bot2/head_camera/image` | real Isaac camera; exact source stamp is frozen per VLM request |
| 2D LiDAR | `/task_generator_node/Ai2_Bot2/laser_scan` | real RTX scan; source frame `Ai2_Bot2/lidar_frame` |
| CameraInfo | `/task_generator_node/Ai2_Bot2/head_camera/camera_info` | calibration only; A-track does not consume sim depth |
| Ego odometry | `/task_generator_node/Ai2_Bot2/odom` | public robot state |
| Extrinsics/TF | `Ai2_Bot2/base_footprint`, `Ai2_Bot2/lidar_frame`, `Ai2_Bot2/base_footprint/head_camera`, `Ai2_Bot2/odom`, `map` | TF only for public robot state and sensor extrinsics |

Forbidden as agent input: simulator depth images, global GT map, HuNav
Agents/People truth, benchmark true goal/start-goal, USD object poses or
semantics, and any oracle trajectory. The true start/goal may only be used by
the evaluator for scoring.

## Frozen case and model pin

| Field | Value |
| --- | --- |
| Case | `grscenes_20_v1/default_2`, `Ai2_Bot2` |
| Humans | HuNav `default_sf60.yaml`, 2 humans, seed `0` |
| Episode cap | configured `59` s; hard target `60` s from the episode-start barrier |
| Cruise | `--lisn-cruise-vel 0.8` (smoother/actuator limits may reduce it) |
| Scan gate | `--lisn-scan-max-age-sec 0.6`, `--lisn-max-scan-range-m 12.0` |
| Model | `Qwen/Qwen3-VL-8B-Instruct-FP8` |
| Model revision | `9cdc6310a8cb770ce18efaf4e9935334512aee45` |
| Model image | `sha256:c1975562dbe35cf5201d55ba84b8e6093043e415ce0728c0c0bcf6d57ee3ff82` |
| Model staging | `/mnt/bn/navigation-hl/mlx/users/chenjunting/arena_models/_staging/qwen3vl_fp8_20261030` |
| Budget | Qwen lane <= 40 GiB, Arena lane <= 10 GiB; model cache on NFS |

The model is served only by `arena-qwen3vl-fp8`. No cloud API and no new
weights are used.

## Exact lane entry (EXECUTED via the outputs copy; in-repo copy IMPLEMENTED_UNVERIFIED)

The 2035 run used the temporary outputs copy:

```bash
cd /home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102
MODE=diag bash run_prox_qwen_lane.sh
```

The repo-owned stable wrapper is now:

```bash
MODE=diag bash arena_bringup/scripts/run_prox_qwen_lane.sh
```

The in-repo copy is syntax-checked (`bash -n`) but was not executed in this
documentation turn, so its runtime status is `IMPLEMENTED_UNVERIFIED`. It
accepts `LANE_ROOT`, `PREFIX`, `RUN_TAG`, `TIMEOUT`, `WORKTREE`,
`TARGET_INSTALL_ROOT`, `MODEL_STAGE`, and `QWEN_CONTAINER` overrides.

`MODE=diag` selects:

- output prefix `arena5_prox_isaac_a_20261102/qwen_diag_v2`
- run tag / raw diag root `qwen_diag_v2`
- `TIMEOUT=59` (the task-generator `--timeout` is an integer)
- recorder started before the controller, so the pinned Prox marker publisher
  sees a subscriber
- v2 renderer after eval with `--name-suffix v2`

## Internal evaluator argv (EXECUTED)

The lane runs this inside `arena-arena_jazzy_ws-arena-1`:

```bash
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
source /home/ubuntu/arena_jazzy_ws/outputs/arena_seed21_four_agent_batch_20261026/target_install/setup.bash
source /opt/arena_ws/install_prox/setup.bash
export ARENA_WS_DIR=/home/ubuntu/arena_jazzy_ws
export HOST_ARENA_WS_DIR=/home/ubuntu/arena_jazzy_ws
export PYTHONPATH=/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_vln_models:$PYTHONPATH
cd /home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013
/opt/venv/bin/python3 -m arena_vln_models.runner eval --local \
  --base-profile base_follow_path_grscenes_task34.yaml \
  --model internnav --robot Ai2_Bot2 --case grscenes_20_v1/default_2 \
  --episodes 1 --pedestrian-seed 0 --timeout 59 \
  --lisn-slow-loop --lisn-wiring-live --prox-mpc \
  --lisn-producer-endpoint http://<qwen-container-ip>:8000/v1/chat/completions \
  --lisn-policy /home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_bringup/configs/lisn/slow_loop_grounding_scan_prox_v1.json \
  --lisn-grounding-scan \
  --lisn-scan-topic /task_generator_node/Ai2_Bot2/laser_scan \
  --lisn-scan-max-age-sec 0.6 \
  --lisn-max-scan-range-m 12.0 \
  --lisn-evidence /home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2/evidence.jsonl \
  --lisn-cruise-vel 0.8 \
  --lisn-grounded-goal-topic /task_generator_node/Ai2_Bot2/lisn_slow_loop/grounded_goal \
  -- --output-prefix arena5_prox_isaac_a_20261102/qwen_diag_v2
```

The `--model internnav` flag only borrows the runner entry point. The
manifest, panel, and control path are LISN/Qwen + `nav2_follow_path_prox_mpc`,
not InternNav/RPP. In the 2035 run the panel shows
`VLM_model=Qwen/Qwen3-VL-8B-Instruct-FP8`. The preserved 2028 offline v2
re-render shows `model=UNKNOWN` because that old trace did not record model
identity; it is not a different model and is not relabeled.

## Recorder and renderer entries (EXECUTED)

Recorder, before controller activation:

```bash
docker exec -d arena-arena_jazzy_ws-arena-1 bash -lc '
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
source /home/ubuntu/arena_jazzy_ws/outputs/arena_seed21_four_agent_batch_20261026/target_install/setup.bash
source /opt/arena_ws/install_prox/setup.bash
export ROS_DOMAIN_ID=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export PYTHONPATH=/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_bringup:$PYTHONPATH
exec /opt/venv/bin/python3 /home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_bringup/arena_bringup/prox_diag_recorder.py --ros-args \
  -p use_sim_time:=true \
  -p robot_namespace:=/task_generator_node/Ai2_Bot2 \
  -p record_dir:=/home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2 \
  -p costmap_hz:=5.0 -p ego_hz:=5.0
'
```

v2 renderer, after eval and recorder flush:

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
source /home/ubuntu/arena_jazzy_ws/outputs/arena_seed21_four_agent_batch_20261026/target_install/setup.bash
source /opt/arena_ws/install_prox/setup.bash
export PYTHONPATH=/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_bringup:$PYTHONPATH
exec /opt/venv/bin/python3 /home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_bringup/arena_bringup/render_prox_diag_video.py \
  --diag-dir /home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2 \
  --out-dir  /home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2 \
  --fps 5.0 --name-suffix v2
'
```

The renderer writes new v2 names and never overwrites the v1 MP4s. The source
window uses an exact request/source stamp; a missing capture is
`SOURCE_FRAME_UNRECORDED`, not a nearest-frame guess.

## Run roots and v2 artifacts

| Artifact | Path / identity |
| --- | --- |
| Raw diag root | `/home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2` |
| Eval run root | `.../qwen_diag_v2/20261102_182650_grscenes_20_v1_Ai2_Bot2_lisn_qwen3vl_fp8` |
| `ego_vlm_debug_v2.mp4` | `.../qwen_diag_v2/ego_vlm_debug_v2.mp4`, 274 frames, 54.8 s, SHA256 `0583d4c93d08ecfc3ab982234a1cd6b72b957a449258d47bf3b028b520effdbe` |
| `local_costmap_prox_debug_v2.mp4` | `.../qwen_diag_v2/local_costmap_prox_debug_v2.mp4`, 386 frames, 77.2 s, SHA256 `b0ca07fb4c3c6f98d0a515de3cba0464f3f1dd87e5dffa4a16dff7e61d8f34e4` |
| PTS ego | `.../video_frame_pts_ego_v2.json`, SHA256 `b3c1ee95f885bf22c135e8205b6cc7c917e7cbf7996c88b623f10ad56d505420` |
| PTS costmap | `.../video_frame_pts_costmap_v2.json`, SHA256 `452278e26dfd11ba906c0e656061dccdba66804931424c72412722831e017801` |
| Timebase audit | `.../TIMEBASE_AUDIT_v2.json`, SHA256 `c5843fe36278a93b307bed848e23af8522933799a817f950836b0ab93cbab309` |
| Source path match | `.../SOURCE_PATH_MATCH_v2.json`, 17/17 `MATCH`, SHA256 `faa65c7e0181111de4dcc3aadebcf196dbfb9dad6d588a7dc4a9c73b2850997e` |
| Render receipt | `.../PROX_DIAG_RENDER_RECEIPT_v2.json`, SHA256 `bc0289a98c25ca30bd99981135c16fdab3c96b12a679512727c128492dd7c573` |
| Run timing | `.../RUN_TIMING.json`, SHA256 `94614f38a174dc0a3433a391f13e3b48be0fe6870b36a14e2cc632e6bb6a1356` |
| Panel checks | `.../PANEL_CHECK_EGO_v2.png`, `.../PANEL_CHECK_COSTMAP_v2.png` |

Offline v2 re-render of the preserved 2028 diag data:

| Artifact | Path / identity |
| --- | --- |
| Ego | `.../diag/ego_vlm_debug_v2.mp4`, 274 frames, 54.8 s, SHA256 `e3e0e86bb2c66ef38f2ede40b47ef7a386362b68c65b674a9ccbd6bfee0e0771` |
| Costmap | `.../diag/local_costmap_prox_debug_v2.mp4`, 389 frames, 77.8 s, SHA256 `bd05a340ef632b7b3bf252bf6d93f4209e564398b006490e667c84dcd48c04aa` |
| Timebase audit | `.../diag/TIMEBASE_AUDIT_v2.json` |
| Source match | `.../diag/SOURCE_PATH_MATCH_v2.json`, 17/17 `MATCH` |
| TF availability | `TF_UNRECORDED` (the old recorder used the broken `do_transform_pose` call) |

## Timebase and layout checks

The v2 audit reports, for both runs:

- first video PTS `0.0` and `xN` `N/A` on the first frame; `xN=1.0` on all
  later presentation frames (sim elapsed / video PTS elapsed)
- zero `huge` or non-finite `xN` frames
- `ego_captured_frames=69` versus `rendered_frames=274`; `unique_main_frames=16`
  and `held_main_frames=258` are held presentation, not unique captures
- 2035 run: `unique_costmap_stamps=141`, `held_costmap_frames=233`
- 2028 offline run: `unique_costmap_stamps=142`, `held_costmap_frames=235`
- panel layout `clipped_frames=0`, `max_line_width_px <= 228`,
  `panel_width_px=240`
- wall latency and sim source age are separate fields; negative ages are
  labelled `CLOCK_AHEAD` rather than hidden

The 2035 run's source-time TF is real:

```text
frame_id=map  stamp_ns=27716668112
odom frame=Ai2_Bot2/odom  result_type=PoseStamped  tf_status=OK
goal request=bef88274-5684-401a-978b-8ca4e1e26260-5
goal update=lisn-000038-a333e9da16da1bafe444
```

The 2035 costmap panel shows the goal marker at the end of the episode
(`Qwen goal 0.9m`). While the goal is outside the 12 m presentation window, the
renderer draws a real edge indicator with the measured goal distance; it never
invents an in-window costmap cell. The panel also exposes
`VLM_model=Qwen/Qwen3-VL-8B-Instruct-FP8`, `VLM_typed`, and `schema=v2`.

The 2028 saved data has no source-time transform, so its v2 goal overlay is
`TF_UNRECORDED`; no fresh TF is substituted for the old run.

## Scan publisher ownership

`PROX_DIAG_META.json` reports `scan publisher_count=2` for
`/task_generator_node/Ai2_Bot2/laser_scan`. The saved metadata records only the
count, not node/GID/owner/reader attribution, so ownership is `UNKNOWN` in this
runbook. No wide kill or runtime change is performed from that count alone.

## Metrics and verdict (2035 run)

| Field | Value |
| --- | --- |
| Episode reason | `sim_timeout` |
| Sim time at end | `74.183337202` s |
| Episode-start sim | `13.983334062` s |
| Episode sim elapsed | `60.20000314` s |
| Hard-cap overrun | `0.20000314` s (configured 59 s, 0.5 s poll granularity) |
| Strict task success | `false` (`episode_timeout`, `goal_not_reached`) |
| Strict social success | `false` (1 footprint near miss; 0 human collisions; min human distance 0.720958 m) |
| Artifact validation | `overall_pass=true` |
| Model control | `nav2_follow_path_prox_mpc`; grounded goal 1, decision 1, accepted 1 |
| VLM responses | 17; 5 accepted, 12 rejected; 1 accepted, 1 rejected, 3 missed tick; all `OUTDATED` |
| Counts | scan 141, costmap 140, tracker 141, local_plan 1752, marker 1752, cmd 1751, diagnostics 1752, ego 66, grounded_goal 1 |
| `planner_path` / CollisionMonitor | 0 / 0; `UNKNOWN`, not success |

This is an engineering-complete diagnostic run with a scientific TIMEOUT, not a
benchmark PASS.

## Metrics recompute and video decode (EXECUTED on the 2035 run)

Use the fail-close wrapper instead of a shell `|| validation_rc` chain. The
wrapper runs each stage independently, stops at the first non-zero stage, and
records `failed_stage`; only an actually executed validation stage with the
contract rc1 (scientific FAIL) is allowed to close as engineering-complete:

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
export PYTHONPATH=/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_bringup:$PYTHONPATH
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline metrics \
  --run-dir /home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2/20261102_182650_grscenes_20_v1_Ai2_Bot2_lisn_qwen3vl_fp8
'
```

The `ros2 run arena_bringup paper_pipeline` alias requires an `arena_bringup`
rebuild; it is `IMPLEMENTED_UNVERIFIED` in this turn.

Wrapper exit codes:

| Code | Meaning |
| --- | --- |
| 0 | all stages 0, validation 0 |
| 1 | validation executed and returned the contract rc1 (scientific FAIL; engineering closure allowed) |
| 2 | a non-validation stage returned non-zero, validation returned rc2, or validation was not executed |

The JSON receipt (`arena.paper.metrics_pipeline.v1`) carries
`status`, `failed_stage`, `validation_executed`, `validation_rc`,
`scientific_fail`, `fault`, and per-stage `rc`/`command`. It never swallows a
preceding metrics error.

The wrapper now requires `--derived-root` and never writes the source run:

```bash
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline metrics \
  --run-dir /home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2/20261102_182650_grscenes_20_v1_Ai2_Bot2_lisn_qwen3vl_fp8 \
  --derived-root /home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/qwen_diag_v2/metrics_derived_2092
```

An earlier in-place run already rewrote the 2035 source
`metrics.csv`, `vln_task_metrics.json`, `social_metrics.json`, and
`artifact_validation.json`. The existing
`METRICS_OVERWRITE_INCIDENT.json` in the source run records their current
hashes/mtimes, the command, and
`exact_backup_available=false`, `restore_claimed=false`; no exact restore is
claimed. The normal wrapper path no longer writes that incident: it is read
only and hashed into the new receipt. Directories are copied, not symlinked,
so derived stages cannot write back through a source directory.

The fresh derived recompute rc=0; stages `stage_a_metrics`,
`stage_b_vln_task_metrics`, `stage_c_social_metrics`, and `validation` all
rc=0; `source_unchanged=true` over the full source file tree. The derived
receipt is
`.../metrics_derived_2092/20261102_182650.../PAPER_METRICS_RECEIPT.json`
(SHA256 `090f2525e40977b62ebe1491d2568508c5870f7030793b51e9b7c4a35a0d2694`).
The earlier `metrics_derived_2075` receipt is preserved.
The derived scientific verdict is unchanged: strict task false
(`episode_timeout`, `goal_not_reached`), strict social false
(`footprint_near_miss`, `personal_space_violation`), `social_success=false`,
min human distance `0.720958 m`, artifact `overall_pass=true`. Wrapper rc0
means the artifact/metric pipeline completed; it is not a scientific PASS.

Video decode is verified by `DECODE_v2.json` (ffprobe) in the raw diag root.
The renderer is the stable entry point.

## Stop, flush, and cleanup (EXECUTED by the lane trap)

The lane trap performs, in order:

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc "pkill -TERM -f 'prox_tracker_age_monitor|prox_diag_recorder' || true"
docker exec -u root arena-arena_jazzy_ws-arena-1 bash -lc "python3 /home/ubuntu/arena_jazzy_ws/outputs/arena5_prox_isaac_a_20261102/kill_lane_ros.py"
docker exec -u root arena-arena_jazzy_ws-isaac-1 bash -lc "pkill -TERM -f 'run_isaacsim' || true; sleep 2; pkill -KILL -f 'run_isaacsim' || true"
bash /mnt/bn/navigation-hl/mlx/users/chenjunting/arena_models/_staging/qwen3vl_fp8_20261030/stop_server.sh
```

Post-run checks: Qwen container absent, Isaac `run_isaacsim` absent, Arena eval
nodes absent, GPU back to the desktop baseline, shared lock free.

## Paper pipeline contracts (CPU-tested; episode status separate)

The in-repo `paper_pipeline` CLI is ROS-free and runs in the eval Docker:

Executed in this turn from the source tree:

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
export PYTHONPATH=/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013/arena_bringup:$PYTHONPATH
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline --help
'
```

The installed console entry `ros2 run arena_bringup paper_pipeline` is
`IMPLEMENTED_UNVERIFIED` until `arena_bringup` is rebuilt/installed with the
new module. The frozen target install used by the 2035 lane was not rebuilt in
this turn.

| Contract | Command | Status |
| --- | --- | --- |
| A/B/C settings + sensor-only | `paper_pipeline settings --file <json>` | CPU READY |
| B strict RGB-only fast guard | `paper_pipeline bfast --file <json> --enabled|--disabled` | CPU READY; episode arm not run |
| Optical-flow/TTC veto | `compute_ttc_guard(...)` inside the B contract | CPU READY; proxy labelled `ttc_proxy_sec` |
| Latency 0/delay/drop/out-of-order | `paper_pipeline latency --source-stamp-ns ... --mode ...` | CPU READY |
| Static/open-loop/closed-loop | `paper_pipeline scenario --file <json>` | CPU READY; episode arms not run |
| Dataset split/manifest | `paper_pipeline dataset --manifest <json>` | CPU READY |
| C teacher/oracle export | `paper_pipeline teacher --manifest <json>` | CPU READY; teacher collection not run |
| Metrics fail-close wrapper | `paper_pipeline metrics --run-dir <run>` | CPU READY; not re-executed on the 2035 run this turn |
| Docker endpoint | `paper_pipeline endpoint --container <name>` | CPU READY |
| Unique output prefix | `paper_pipeline prefix --prefix <path>` | CPU READY |

CPU CLI smoke in `arena-1` (actual rc):

```text
settings A valid                       rc=0
settings B with scan+GT                rc=2
bfast with common follow_path          rc=0
bfast bare point (no metric path)      rc=2 UNSUPPORTED_NO_COMMON_ACTION
bfast unknown units                    rc=2
latency passthrough (0 ms)             rc=0
latency drop mode                      rc=0
scenario closedloop                    rc=0
scenario openloop                      rc=2 UNSUPPORTED_OPENLOOP_WITHOUT_SCRIPTED_HUMANS
dataset manifest                       rc=0
teacher missing provenance             rc=2 (typed missing_source_sample)
endpoint missing container             rc=2
```

The B guard uses only RGB observables. It may veto to STOP; it never rewrites
the Qwen goal. Both B arms use the same proposal, prompt, and action mapping;
only `enabled` changes. Forbidden scan/depth/GT/People topics fail closed. The
flow-risk value is an optical-flow proxy, not a causal TTC or a depth
measurement.

### B off/on launch and profile

| Artifact | Path |
| --- | --- |
| Manifest | `arena_bringup/configs/lisn/bfast_manifest_v1.json` |
| Off profile | `arena_bringup/configs/lisn/bfast_off_v1.json` |
| On profile | `arena_bringup/configs/lisn/bfast_on_v1.json` |
| Off launch | `arena_bringup/launch/bfast_off.launch.py` |
| On launch | `arena_bringup/launch/bfast_on.launch.py` |
| Guard node | `arena_bringup/arena_bringup/bfast_guard_node.py` |
| Executor hook | `arena_vln_models/arena_vln_models/executor.py` (`fast_guard_veto_topic`) |

Copy-able control launches (they do not start Isaac or a model server):

```bash
ros2 launch arena_bringup bfast_off.launch.py \
  robot:=Ai2_Bot2 namespace:=/task_generator_node/Ai2_Bot2

ros2 launch arena_bringup bfast_on.launch.py \
  robot:=Ai2_Bot2 namespace:=/task_generator_node/Ai2_Bot2 \
  fast_guard_veto_timeout_sec:=1.0
```

The off arm starts the common FollowPath controller/executor with no guard and
no veto topic. The on arm starts the same controller/executor plus the guard and
passes `/task_generator_node/Ai2_Bot2/bfast/veto` to the executor. Both use the
same RGB topic, model, prompt hash, and action mapping. Full B episodes are
`IMPLEMENTED_UNVERIFIED`; only the control launch parse and ROS smoke are
verified.

The in-repo source launch files parse, and the installed overlay launch works
by package name:

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
cd /home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013
colcon build --merge-install --symlink-install \
  --packages-select arena_bringup arena_vln_models \
  --build-base /home/ubuntu/arena_jazzy_ws/outputs/arena5_paper_overlay_20261103/build \
  --install-base /home/ubuntu/arena_jazzy_ws/outputs/arena5_paper_overlay_20261103/install
source /home/ubuntu/arena_jazzy_ws/outputs/arena5_paper_overlay_20261103/install/setup.bash
ros2 launch arena_bringup bfast_off.launch.py --show-args
ros2 launch arena_bringup bfast_on.launch.py --show-args
'
```

Actual rc: build `0`; installed `bfast_off` `--show-args` `0`; installed
`bfast_on` `--show-args` `0`. The overlay is
`outputs/arena5_paper_overlay_20261103/install`; the user's main
`/opt/arena_ws/install` is not overwritten.

B profile identity validation:

```bash
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline bfast-profile \
  --file arena_bringup/configs/lisn/bfast_on_v1.json   # rc=0
```

The profile records `model=Qwen/Qwen3-VL-8B-Instruct-FP8`, its revision, the
LISN prompt SHA-256, and the common action mapping. `model_adapter=internnav`
is only the runner entry; it is not the model identity. The B runner profile
selection is still a typed blocker for full eval, so a full B episode is
`IMPLEMENTED_UNVERIFIED`.

Runner plan with the explicit B profile (no episode):

```bash
cd /home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013
PYTHONPATH=$PWD/arena_vln_models:$PWD/arena_bringup:$PYTHONPATH \
  /opt/venv/bin/python3 -m arena_vln_models.runner plan --local \
    --base-profile base_follow_path_grscenes_task34.yaml \
    --model internnav --robot Ai2_Bot2 --case grscenes_20_v1/default_2 \
    --lisn-slow-loop \
    --lisn-producer-endpoint http://127.0.0.1:8000/v1/chat/completions \
    --bfast-profile arena_bringup/configs/lisn/bfast_on_v1.json \
    --bfast-risk-threshold 0.5 --bfast-max-frame-gap-sec 1.0 \
    -- --output-prefix arena5_paper_bfast_plan
```

`--bfast-profile` requires the Qwen/LISN producer path (`--lisn-slow-loop` plus
a `/v1/chat/completions` endpoint). The generated command uses
`--model-adapter lisn_qwen3vl_fp8` and `--model-backend lisn_qwen3vl_fp8`;
it does not contain the InternNav `5801/eval_dual` server or an InternVLA
checkpoint. `model_adapter=internnav` remains only the runner entry label.

The Qwen/LISN sidecar currently emits pixel/bearing/STOP and has no common
metric FollowPath action bridge. The `lisn_qwen3vl_fp8` adapter is therefore a
typed blocker (`qwen_lisn_common_metric_action_bridge_missing`), not a Path
fabricator. Full B eval remains typed `BLOCKED`.

The canonical `bfast_on_v1.json` does not freeze `risk_threshold` or
`max_frame_gap_sec`; the values above are explicit smoke overrides and the
plan marks `setting_status=SMOKE_ONLY_NOT_PAPER_SETTING`. The A case
`grscenes_20_v1/default_2` is likewise a smoke case, not a frozen B scientific
setting. The on profile fails closed if either threshold is absent and no
explicit override is supplied.

The same profile is attached to the actual `internnav_eval` argv:

```text
runner --bfast-profile
  -> internnav_eval --bfast-profile/--bfast-guard-enabled/--bfast-veto-topic/
     --bfast-guard-required/--bfast-guard-timeout-sec/
     --bfast-risk-threshold/--bfast-max-frame-gap-sec
  -> internnav_eval launch_cmd fast_guard_*/bfast_* arguments
  -> arena.launch.py -> task_generator.launch.py -> robot_manager.py
  -> follow_path.launch.py (bfast_guard node + executor veto topic)
```

Actual full-eval dry-run with the on profile:

```bash
/opt/venv/bin/python3 -m arena_vln_models.runner full-eval --dry-run --local \
  --base-profile base_follow_path_grscenes_task34.yaml \
  --model internnav --robot Ai2_Bot2 \
  --lisn-slow-loop \
  --lisn-producer-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --bfast-profile arena_bringup/configs/lisn/bfast_on_v1.json \
  --bfast-risk-threshold 0.5 --bfast-max-frame-gap-sec 1.0
```

Actual rc=0, but the B plan is **non-executable**: `executable=false`,
`commands=[]`, `blocked_reason=qwen_lisn_common_metric_action_bridge_missing`.
The top-level identity is `model=Qwen/Qwen3-VL-8B-Instruct-FP8`,
`model_backend=lisn_qwen3vl_fp8`, `model_adapter=lisn_qwen3vl_fp8`,
`runner_entry=internnav` (subordinate label only); there is no InternNav
top-level identity, no `5801/eval_dual`, and no InternVLA checkpoint.

`internnav_eval --print-effective-config` with on/off profiles proves the two
arms differ only in the guard fields; all other resolved parameters are
identical. The actual off/on `internnav_eval` argv construction is still
tested parse-only; the off command sends no `None` threshold arguments. A full
B episode is still not run and remains typed `BLOCKED`.

### Real ROS guard smoke (EXECUTED)

```bash
ROS_DOMAIN_ID=97 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  /opt/venv/bin/python3 arena_bringup/test/ros_bfast_guard_smoke.py
```

Actual output (rc=0):

```text
warmup          veto=false  decision=WARMUP  executor=follow_path
clear           veto=false  decision=PASS    executor=follow_path
invalid_32fc1   veto=true   decision=INVALID executor=stop
direct_veto     veto=true   reason=bfast_veto          executor=stop
stale_required  veto=true   reason=bfast_guard_stale   executor=stop
stale_optional  veto=true   reason=(none)              executor=follow_path
```

This uses real ROS topics/QoS and a controlled executor-compatible veto
consumer; it does not use Isaac, GPU, or a model.

### Actual ArenaModelExecutor smoke (EXECUTED)

```bash
ROS_DOMAIN_ID=98 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  PYTHONPATH=$WORKTREE/arena_vln_models:$WORKTREE/arena_bringup:$PYTHONPATH \
  /opt/venv/bin/python3 arena_bringup/test/ros_bfast_executor_smoke.py
```

This starts the real `ArenaModelExecutor`, the real `bfast_guard`, and a real
FollowPath action server. A deterministic fake adapter supplies a fixed
forward path; it is not a controlled veto consumer. Actual rc=0 and trace
evidence:

```text
clear_follow_path   trace_follow_path_goals >= 1
direct_veto         trace bfast_veto reason=bfast_veto; goal count unchanged
invalid_input       guard decision INVALID; trace bfast_veto; goal count unchanged
required_stale      trace bfast_veto reason=bfast_guard_stale; goal count unchanged
```

The smoke fails if any of the direct/invalid/required-stale scenarios increases
the FollowPath action-server goal count; trace veto lines alone are not
sufficient.

The executor trace is `/tmp/bfast_executor_smoke/*.jsonl`. The fake adapter is
registered only by the smoke; no model weights or GPU are used.

### Latency release dispatcher

`paper_pipeline.ReleaseQueue` and `LatencyCommitGate` implement a non-blocking
producer path. Disabled is the default and keeps the synchronous commit.
Enabled (`--latency-mode`, `--latency-ms`, `--latency-profile`) freezes the
proposal/image/source stamp/frame/hash, enqueues immediately, clears the
producer pending, and lets later responses enqueue. A 0.1 s node timer polls
ready items and commits each exactly once; drop records immediately and never
commits. Typed receipts are written to `LATENCY_COMMIT.json/jsonl`.

Deterministic fake-clock tests cover disabled, delay, drop, source
non-refresh, and two responses whose arrival order differs from release order
with no duplicate commit. The code is wired into
`arena_bringup/arena_bringup/lisn_slow_loop_ros.py` (`_prepare_proposal` /
`_tick_latency_commits` / `_finalize_commit`).

Real node smoke (no GPU/Isaac/model):

```bash
ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  /opt/venv/bin/python3 arena_bringup/test/ros_lisn_latency_node_smoke.py
```

Actual rc=0: a real `LisnSlowLoopNode` with its real latency timer and a fake
producer enqueues response A (500 ms) then B (0 ms); release order is B then A;
each is committed exactly once; drop does not commit; source stamps/hashes are
unchanged; shutdown is bounded. The commit callback is a deterministic test
double, so controller apply is not claimed. A real episode has not exercised
the latency path, so the full runtime layer remains
`IMPLEMENTED_UNVERIFIED`.

### Stable wrapper stage exit codes and fault injection

The stable wrapper exits non-zero for model-start, readiness, eval, or render
failures. `paper_pipeline metrics` is the only place that maps validation rc1 to
scientific FAIL.

```bash
DRY_RUN=1 FAULT_INJECT=model_start bash arena_bringup/scripts/run_prox_qwen_lane.sh  # rc=22
DRY_RUN=1 FAULT_INJECT=readiness   bash arena_bringup/scripts/run_prox_qwen_lane.sh  # rc=21
DRY_RUN=1 FAULT_INJECT=eval        bash arena_bringup/scripts/run_prox_qwen_lane.sh  # rc=3
DRY_RUN=1 FAULT_INJECT=render      bash arena_bringup/scripts/run_prox_qwen_lane.sh  # rc=2
DRY_RUN=1 FAULT_INJECT=none        bash arena_bringup/scripts/run_prox_qwen_lane.sh  # rc=0
```

Cleanup kills only recorded owned PIDs after verifying the PID command still
matches the expected recorder/eval/Isaac pattern; a mismatch is logged as
`pid_identity_mismatch`, not force-killed. The exact Qwen container is stopped
by its own script. Cleanup errors are written to `cleanup_errors.log`.

The scenario profiles explicitly declare `robot_human_feedback`:

| Mode | Humans | Feedback | Runtime status |
| --- | --- | --- | --- |
| static | none (`--human dummy`, count 0) | false | SUPPORTED |
| openloop | HuNav | false | UNSUPPORTED: no scripted-human interface; HuNav SFM reacts to the robot |
| closedloop | HuNav (`ARENA_HUNAV_CONFIG`) | true (SFM/behavior tree) | SUPPORTED |

`--social-eval` is a metrics flag only. The closed-loop feedback comes from
HuNav's SFM, not from the metrics flag. The open-loop arm is typed
`UNSUPPORTED_OPENLOOP_WITHOUT_SCRIPTED_HUMANS` until a real scripted-human
interface exists.

### Static/closed-loop smoke-only profiles

| Profile | Path |
| --- | --- |
| static smoke | `arena_bringup/configs/lisn/scenario_static_smoke_v1.json` |
| closed-loop smoke | `arena_bringup/configs/lisn/scenario_closedloop_smoke_v1.json` |

Both carry `setting_status=SMOKE_ONLY_NOT_PAPER_SETTING`. `paper_pipeline
scenario --file <path>` returns rc=0 for both, and the generated
`internnav_eval` argv is executed parse-only with `--print-effective-config`
(rc=0). The static command uses `--human dummy --expected-human-count 0`; the
closed-loop command uses `--human hunav --social-eval` and
`ARENA_HUNAV_CONFIG`.

The closed-loop profile pins the repo config path
`arena_bringup/configs/hunav/default_sf60.yaml` and verifies SHA256
`a834b6bb42e6c8f14ea4734aa5ab1fd03791a1897f9445d24c2fb7ccade9eeb1`.
The generated `ARENA_HUNAV_CONFIG` value is the resolved absolute path, not a
bare filename. Open-loop remains
`UNSUPPORTED_OPENLOOP_WITHOUT_SCRIPTED_HUMANS`.

### C provenance audit

```bash
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline teacher-audit \
  --manifest <teacher-export.json>
```

The audit reports `provenance_status`, `missing_source_sample`, `leakage`,
`qc_not_set`, `old_gt_labels_preserved`, and `typed_blocker`. The repository
has no saved teacher/oracle source sample, so the current smoke manifest
returns rc=2 with `TEACHER_SOURCE_SAMPLE_MISSING`; old GT labels are preserved
and no provenance is fabricated. Split leakage, non-GT relabeling, and
`instruction_qc:not_set` also fail closed.

### C source-run provenance audit (historical candidate)

```bash
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline source-run-audit \
  --source-run /home/ubuntu/arena_jazzy_ws/outputs/arena_hunav_sf60_20261031/lisn_semantic30_fix/20261031_075457_grscenes_20_v1_Ai2_Bot2_lisn_qwen3vl_fp8 \
  --derived-root /home/ubuntu/arena_jazzy_ws/outputs/arena_hunav_sf60_20261031/lisn_semantic30_fix/source_run_audit_2088
```

The audit is read-only over `run_manifest.yaml`, `CONTROLLER_PEOPLE.json`,
`HUMAN_PEOPLE_BRIDGE.json`, `human_states.csv`/`pedsim_agents_data.csv`,
`odom.csv`, and `start_goal.csv`; the receipt is written only to the fresh
derived root. It fails closed with `MISSING_SOURCE_FILES`,
`UNKNOWN_CONTROLLER_PEOPLE`, `UNKNOWN_EXECUTOR_IDENTITY`, or
`SOURCE_HASH_MISMATCH`; an empty source directory returns rc=2 and
`source_eligible=false`. Actual candidate rc=2:

```text
blockers=[CONTROLLER_PEOPLE_ZERO, EXECUTOR_IDENTITY_INTERNNAV]
controller_people=0
bridge_people=2; published_count=8331
executor_identity=InternNav
gt_available_for_evaluation=true
gt_consumed_by_controller=false
source_eligible=false
teacher_export_created=false
old_gt_labels_preserved=true
source_unchanged=true
```

Receipt (latest fail-close rerun):
`.../source_run_audit_2088/20261031_075457_grscenes_20_v1_Ai2_Bot2_lisn_qwen3vl_fp8/SOURCE_RUN_PROVENANCE_AUDIT.json`
SHA256 `3126f0801c7faf9210b674a0dd8b1abc4051ba3b4fafb784fe0194040965effe`.
The earlier `source_run_audit_2084` receipt
(`e6f3e85800156d61b96950b8a4b4780fc352a09cb008081fbaf435839a8711ad`) is
preserved; the source run itself is unchanged.
No teacher trajectory is generated; the empty-manifest
`TEACHER_SOURCE_SAMPLE_MISSING` check remains a separate template check.

### Agentic eval and evidence index

The operator contract is [Agentic Evaluation Contract](agentic_eval.md).
Machine-readable requests live in
`configs/agentic_eval/operator_request.v1.schema.json` with examples
`generic_internnav_single_case.json` and `strict_b_single_case.json`.

```bash
# source form from the repo root
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline agentic-request \
  --file configs/agentic_eval/examples/generic_internnav_single_case.json
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline agentic-request \
  --file configs/agentic_eval/examples/strict_b_single_case.json
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline evidence-index \
  --run-dir <run> --output <fresh-json> \
  --sensor-profile rgb_lidar   # strict B uses rgb_only
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline agentic-lifecycle \
  --file configs/agentic_eval/examples/generic_internnav_single_case.json \
  --action plan|start|cancel|finalize|read|audit [--from-state <state>]
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline evidence-verdict \
  --index <evidence-index.json> --output <fresh-json>
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline recording-contract \
  --file configs/agentic_eval/recording/a_rgb_lidar.json
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline recording-preflight \
  --contract configs/agentic_eval/recording/a_rgb_lidar.json \
  --run-dir <run> --output <fresh-json>
/opt/venv/bin/python3 -m arena_bringup.paper_pipeline audit-consolidate \
  --manifest <fresh-manifest.json> --output <fresh-json>

# installed overlay form needs the repo root for relative contract paths
export ARENA_REPO_ROOT=/home/ubuntu/arena_jazzy_ws/.trae/worktrees/vln-target-merged-20261013
ros2 run arena_bringup paper_pipeline agentic-request --file ...
```

Lifecycle results: `plan`/`cancel` rc=0; `start` rc=2 with
`START_RUNTIME_IDEMPOTENCY_GUARD_MISSING` and `commands=[]` because the
current entry has no runtime dedupe; `finalize` rc=2 with
`ACTION_NOT_BOUND:finalize` and `commands=[]` because no safe flush/finalize
entry exists. `read`/`audit` rc=0 only after the request is bound to the run's
actual manifest identity.

A is the audit-only `qwen_prox_a` role (Qwen FP8 + RGB + real RTX LiDAR +
common FollowPath + Prox), with contract
`configs/agentic_eval/profiles/qwen_prox_a_audit_identity.v1.json`
(`launchable=false`). Its plan/start are `AUDIT_ONLY_TRACK_NOT_LAUNCHABLE`;
read/audit are positive with `qwen_a_single_case.json`. The generic InternNav
request against the Qwen run returns
`RUN_IDENTITY_MISMATCH:track/model/model_revision/profile_sha256/profile_id`
with `commands=[]`. An InternNav profile paired with a Qwen model is
`MODEL_PROFILE_MISMATCH` and is blocked for read/audit too.

Audit requires the operator to write the inline provenance and
`audit_manifest` to their paths before the final `audit-consolidate` step.
Producers recompute provenance from the immutable run before writing;
`audit-consolidate` recomputes from `manifest.request_id` + `manifest.run_dir`.
Consistent forged envelopes or another run's receipt are
`PROVENANCE_AUTHENTICITY_MISMATCH:<field>`/rc2; minimal fixtures are
`contract_blocked`. Invalid `--from-state` transitions return
`INVALID_STATE_TRANSITION:<from>-><action>`. Repeated `read`/`audit` with the
same output root fail closed on `output_root:exists`; repeated `start` cannot
create a second run/output.

Recording preflight on the current A run: `BLOCKED`, with
`RECORDING_TOPIC_MISSING` for RGB, CameraInfo, and LaserScan; strict-B
requires RGB+CameraInfo only. The contract never backfills the existing bag.

### Method routing plan

One stable entry, no service start:

```bash
src/Arena/_meta/docker/features/benchmark/main plan --method <id> \
  --common configs/agentic_eval/methods/common_contract.v1.json \
  --output <fresh-json>
```

| Method ID | Status |
| --- | --- |
| `internnav` | plan-only; current tree BLOCKED (`METHOD_BUNDLE_CURRENT_PROFILE_DRIFT`) |
| `navila` / `lightnav` | plan-only; current tree BLOCKED (`COMMON_CASE_SOURCE_MISSING`) |
| `qwen_agent_proxmpc` | BLOCKED `QWEN_AGENT_PROXMPC_RUNTIME_PROFILE_MISSING`; `qwen_prox_a` audit identity remains readable |
| `strict_b_on` / `strict_b_off` | BLOCKED `STRICT_B_EPISODE_NOT_FROZEN` |
| `socialmpc_oracle` | BLOCKED `ORACLE_NOT_OPERATOR_EXECUTABLE` |

`method-compare --common <json> --output <fresh-json>` proves the common
contract hash is identical across methods, private hashes differ, and no
port/topic/output-prefix conflict exists. Shared namespace/container/source-
binding/lock are warnings because methods are mutually exclusive and serialized
by the shared lock. The plan never contains a launch command. The public entry
rejects `--bundle` with `BUNDLE_OVERRIDE_NOT_ALLOWED`; plans carry
`bundle_source_digest` and `canonical_bundle_path`, and a forged profile SHA is
`METHOD_BUNDLE_PROFILE_SHA_MISMATCH`.

Parse-only validation of the generated command (no containers, model, or
episode):

```bash
<arena_dir>/_meta/docker/features/benchmark/main plan run \
  --profile <profile> --case <world>/<scenario> --robot <robot> \
  --episodes 1 --pedestrian-seed <seed> --timeout <sec> \
  -- --output-prefix agentic/<request_id>
```

`plan` prints the effective config and `nothing above was executed`; it does
not start Docker, Isaac, or the model.

The generic plan is bound to the real `internnav_grscenes` benchmark profile:
`--profile`, `--case`, `--robot`, `--episodes 1`, `--pedestrian-seed`,
`--timeout`, and the resolved `ARENA_HUNAV_CONFIG`/SHA. `workspace_root`
resolves to the host workspace `/home/ubuntu/arena_jazzy_ws`; `arena_dir` and
`source_binding` are explicit and must exist. `model`/`model_revision` must
match the profile; `audit` and `full_eval` are typed
`UNSUPPORTED_MODE:<mode>`.

Actual results: generic rc=0 `executable=true`, one parse-verified command
(`benchmark/main plan run` rc=0, `nothing above was executed`); strict-B
rc=0 `executable=false`, `commands=[]`, blocker
`QWEN_LISN_COMMON_METRIC_ACTION_BRIDGE_MISSING`.

The evidence index reads `video_index.episodes[]`, resolves every video
run-relative, deduplicates aliases, and indexes each episode's
`video_frame_pts.json`. `DECODE*`/`TIMEBASE*` audit files are indexed with
their scope and whether they reference this run; absent evidence is
`UNKNOWN`. Bag topics are classified by ROS message type, and
`--sensor-profile rgb_only` does not require a LaserScan. The saved 2035
evidence index is
`.../qwen_diag_v2/evidence_index_2092/evidence_index.json`
(SHA256 `adc23d78417135b9b4672b18d282d902ca4749a502134970d28574d3645588a6`),
with `playback_status=BLOCKED` because RGB/CameraInfo are absent and LaserScan
count is zero; odom=337, cmd_vel=127, human_states=337.

Evidence verdict for that index: `playback_blocked` (primary),
`scientific_fail` (secondary), `artifact_complete=false`,
`contract_blocked=false`; `unknowns` include
`ARTIFACT_MISSING:benchmark_result.json`, `DECODE_AVAILABLE_NOT_RUN_SCOPED`,
`TIMEBASE_AVAILABLE_UNVERIFIED`, and the three playback blockers; exit 2.
The verdict is deterministic (`input_sha256` over the index view) and never
upgrades UNKNOWN/BLOCKED to PASS.

## Known gaps

- `planner_path=0` (no global planner in FollowPath mode) and
  `collision_monitor=0` (no state messages) are `UNKNOWN`, not passes.
- `num_active_obstacles=4` is not `dynamic=4`; the controller's static/dynamic
  split is not exposed.
- Scan publisher ownership is `UNKNOWN`; only the count is saved.
- B-track strict RGB-only fast reactivity and C-track teacher/oracle synthesis
  are not implemented in this lane.
- The episode hard-cap overrun is 0.2 s due to scheduler/poll granularity; it
  is recorded, not hidden.
