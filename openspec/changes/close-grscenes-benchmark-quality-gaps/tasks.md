## 1. Failure Evidence Review

- [x] 1.1 Add or identify an offline command that summarizes strict failure evidence for one run directory without requiring ROS graph access.
- [x] 1.2 Generate a review report for the latest `grscenes_5` strict failure run, including static occupancy, commanded stuck, goal distance, and footprint social samples.
- [x] 1.3 Classify the `grscenes_5` strict failures as dataset/map registration, robot footprint/config, model behavior, instrumentation, or mixed.

## 2. Instrumentation Readiness

- [x] 2.1 Trace why `ego_debug_overlay.mp4` uses fallback imagery in the latest run and identify the expected model debug image topic/source.
- [x] 2.2 Implement the minimal fix or validation rule so debug overlay fallback is either eliminated or treated as non-ready instrumentation in aggregate summaries.
- [x] 2.3 Add tests or artifact fixtures for debug overlay source classification.

## 3. Language And BDDL Contract

- [x] 3.1 Inventory GRScenes episode instruction metadata, native scenario start/goal, and available BDDL/task predicate fields.
- [x] 3.2 Document which predicates strict task metrics currently evaluate and which are explicitly unsupported.
- [x] 3.3 Add artifact fields or warnings for unsupported BDDL/semantic predicates when present.

## 4. Representative GRScenes Verification

- [x] 4.1 Select an initial representative gate episode with valid map, scenario, instruction, and HuNav trajectory: `grscenes_5/default`.
- [x] 4.2 Run the selected gate episode through Docker + Isaac + external InternNav with social eval and video recording.
- [x] 4.3 Aggregate strict task/social/video/debug-overlay results and write a concise benchmark readiness summary.
- [x] 4.4 Inspect `sim_top_down` and debug videos for the gate run and attach manual review notes/frame-analysis outputs.
- [ ] 4.5 Expand the representative set beyond the gate run before claiming benchmark readiness, e.g. at least one valid episode each from `grscenes_1`, `grscenes_3`, and `grscenes_5`.

### 4.x Evidence Snapshot

- Gate run: `/home/ubuntu/arena_jazzy_ws/outputs/grscenes_benchmark_quality_eval_20260623_rerun/20260623_095109_grscenes_5_Ai2_Bot2_internnav`
- Full Isaac eval completed with `launch_returncode=0`; strict post-processing completed and intentionally returned benchmark failure.
- Videos are complete: `ego_observation.mp4`, `ego_debug_overlay.mp4`, `sim_top_down.mp4`, and `map_top_down_follow.mp4`.
- Frame analysis is attached at `frame_analysis/video_frame_analysis.json`; `sim_top_down` is visually valid, and `ego_debug_overlay` is `PASS_WITH_READINESS_WARNING`.
- Aggregate summary: `strict_task_success_rate=0.0`, `strict_social_success_rate=0.0`, `benchmark_ready_rate=0.0`; legacy task/social success rates are both `1.0`, proving the old success fields were false positives for this episode.
- Classification: HuNav dynamic scene is valid; current blockers are `goal_not_reached`, `episode_timeout`, `static_occupancy_collision`, `footprint_human_collision`, `footprint_near_miss`, and debug-overlay startup fallback.

## 5. Documentation And Handoff

- [x] 5.1 Update local benchmark docs with the new failure classification and readiness criteria.
- [x] 5.2 Update the Lark benchmark plan with representative-run results and remaining blockers.
- [ ] 5.3 Commit the OpenSpec artifacts, docs, and any implementation changes in focused commits.
