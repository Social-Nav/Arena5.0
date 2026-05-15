## 1. Baseline Analysis

- [x] 1.1 Parse the latest controlfix6 output and summarize action distribution, cmd_vel distribution, odom progress, and goal-distance trend.
- [x] 1.2 Inspect `ego_observation.mp4`, `sim_top_down.mp4`, `ego_debug_overlay.mp4`, and `internnav_status.json` for evidence of stale observations, yaw sign mismatch, or repeated turn actions.
- [x] 1.3 Identify the first likely failure class and document the evidence before changing behavior.

## 2. Per-Inference Tracing

- [x] 2.1 Add JSONL trace output in the InternNav server or wrapper for every real model inference.
- [x] 2.2 Include observation freshness, RGB/depth/camera-info metadata, goal pose/distance/yaw error, raw output, normalized action, converted command, action history, and timing fields.
- [x] 2.3 Add eval manifest/index references to the trace file when available.

## 3. Decision Visualization

- [x] 3.1 Extend the InternNav debug image overlay to render selected action, command, goal distance/yaw error, action history, and freshness status.
- [x] 3.2 Add an action glyph or concise visual cue for forward/left/right/stop decisions without obscuring the ego frame.
- [x] 3.3 Verify `ego_debug_overlay.mp4` contains readable decision diagnostics during an eval run.

## 4. Diagnostic Summary

- [x] 4.1 Add a post-run summary generator for action distribution, rotate/forward/stop ratios, and trace-derived goal-distance progress.
- [x] 4.2 Add candidate root-cause flags for yaw/action sign mismatch, stale observations, and missing depth/camera info.
- [x] 4.3 Write the summary into the eval output directory and include it in `run_manifest.yaml` if feasible.
- [x] 4.4 Persist turn-sign correction evidence in the trace, diagnostic summary, and run manifest.

## 5. Fix and Validation Loop

- [x] 5.1 Apply the minimal fix indicated by diagnostics, such as action sign correction, observation orientation correction, goal/yaw conversion fix, or instruction/goal conditioning fix.
- [x] 5.2 Re-run hospital_1 + Ai2_Bot2 + InternNav eval and compare against controlfix6 baseline.
- [x] 5.3 Confirm the robot shows non-trivial forward progress and the debug overlay explains model decisions frame-by-frame.

## Validation Notes

- 2026-05-15: OpenSpec CLI does not list this change even though the change directory exists, so status/tasks are maintained directly in this file.
- 2026-05-15: Verified the latest available controlfix6 `ego_debug_overlay.mp4` is H.264, 640x480, 1468 frames, and contains readable decision diagnostics (`cmd_vel`, goal distance/yaw error, modality status, adapter target, selected action).
- 2026-05-15: Current diagnostic implementation tests pass in `arena-jazzy-eval`: `13 passed in 0.41s`; a synthetic trace smoke test confirms event counts, invert-turn values, and goal-distance progress in `internnav_diagnostic_summary.json`.
- 2026-05-15: Full real-model validation for 5.2/5.3 remains blocked in this host/dev context because the configured InternNav conda Python (`/home/ubuntu/miniconda3/bin/python`) is unavailable and mock eval attempts hit pre-existing Nav2 lifecycle/configuration issues.
- 2026-05-15: Created an isolated Arena-side InternNav venv at `/home/ubuntu/arena_jazzy_ws/.venvs/internnav` and verified real checkpoint load with `/home/ubuntu/arena_jazzy_ws/.venvs/internnav/bin/python` plus `INTERNNAV_DEPTH_ANYTHING_CKPT=/home/ubuntu/arena_jazzy_ws/deps/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth`.
- 2026-05-15: Fixed the local NextDiT FFN checkpoint compatibility bug in `deps/InternNav/internnav/model/basemodel/internvla_n1/nextdit_traj.py` so the released DualVLN checkpoint now loads in the isolated Arena venv instead of failing with `Linear` weight shape mismatches.
- 2026-05-15: Re-ran `hospital_1 + Ai2_Bot2 + InternNav` from both `arena-jazzy-eval` and the main Arena container using `--internnav-python-executable /home/ubuntu/arena_jazzy_ws/.venvs/internnav/bin/python`; both runs ended with `end_reason: adapter_failure` because InternNav never received initial camera messages (`camera_timeout`, missing `rgb/depth/camera_info`).
- 2026-05-15: During the failed re-run, ROS topic discovery still showed `/task_generator_node/Ai2_Bot2/head_camera/image`, `/task_generator_node/Ai2_Bot2/head_camera/depth`, and `/task_generator_node/Ai2_Bot2/head_camera/camera_info`, but `ros2 topic info -v` reported `Publisher count: 0` for the camera topics. This isolates the remaining blocker to the Isaac side, not the Arena-side InternNav Python isolation.
- 2026-05-15: The current Isaac container on this host is also running without a usable GPU (`No physical device is found`, `no CUDA-capable device is detected` in `/tmp/isaac_sim.log`), which likely explains the missing camera publishers and prevents closing 5.2/5.3 in this environment.
- 2026-05-15: Re-validated from the active `arena-jazzy-eval` container and confirmed the camera diagnosis above was stale/transient rather than the steady-state blocker: `/task_generator_node/Ai2_Bot2/head_camera/image`, `/depth`, `/camera_info`, and `/top_down_camera/image` all had a live publisher and delivered frames, while `internnav_status.json` showed fresh sensor ages below 0.2 s.
- 2026-05-15: A fresh isolated-venv retest at `outputs/internnav_hospital1_ai2_isolatedvenv_retest/20260515_064643_hospital_1_Ai2_Bot2_internnav` finished end-to-end with `end_reason: finished`, `metrics_returncode: 0`, all four MP4 artifacts closed cleanly, and `internnav_trace.jsonl` / `internnav_diagnostic_summary.json` present.
- 2026-05-15: That retest still does not satisfy 5.3. The diagnostic summary reports `rotate_heavy_low_progress`, goal distance regressed slightly from `16.07917290061255` to `16.08194572023998`, the trace contains only turn actions (`2`/`3`) plus cached rotate fallbacks, and Nav2 logged `Failed to make progress`.
- 2026-05-15: The remaining blocker is motion/progress, not camera delivery or the isolated InternNav Python path. In the finished retest, `cmd_vel.csv` contains many non-zero commands, but `odom.csv` reports zero linear/angular velocity throughout the episode after reset, so the robot is effectively not executing commanded motion even though the eval/video/trace pipeline now works.
- 2026-05-15: Patched the Isaac-side control/odometry graph helpers so Ai2_Bot2 now targets the articulation via `robotPath` in `isaac_utils/graphs/control/differential.py` instead of a runtime `OgnGetPrimAtPath -> targetPrim` connection, and so `isaac_utils/graphs/odom.py` republishes real linear/angular velocity via `IsaacComputeOdometry`.
- 2026-05-15: Motion-fix retest `outputs/internnav_hospital1_ai2_motionfix/20260515_080247_hospital_1_Ai2_Bot2_internnav` confirmed the robot now physically moves again (`max_abs_vx≈0.316`, `max_abs_wz≈1.457`, net displacement ≈`1.30 m`, path length ≈`8.59 m`), but with `--internnav-invert-discrete-turns auto`/`true` the run still regressed in goal distance from `2.0651719380670412` to `3.280617073167211` and kept flagging `rotate_heavy_low_progress`.
- 2026-05-15: An A/B rerun with `--internnav-invert-discrete-turns false` at `outputs/internnav_hospital1_ai2_motionfix_noinvert/20260515_081056_hospital_1_Ai2_Bot2_internnav` performed materially better: `end_reason: finished`, `metrics_returncode: 0`, goal distance improved from `13.88780154754676` to `9.738124785104633` (progress `4.149676762442127`), the diagnostic summary cleared `rotate_heavy_low_progress`, `odom.csv` shows non-trivial motion (`net displacement ≈4.95 m`, `path length ≈12.51 m`, `max_abs_vx≈0.348`, `max_abs_wz≈1.469`), and the recorded debug overlay/video artifacts closed normally.
- 2026-05-15: Based on that A/B validation, `internnav_eval.py` now resolves `--internnav-invert-discrete-turns auto` to `false` for the current Isaac + Ai2_Bot2 path (`auto_isaac_ai2_bot2_noinvert`).
