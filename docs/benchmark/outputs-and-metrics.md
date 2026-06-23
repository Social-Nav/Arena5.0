# Outputs and Metrics

## Legacy evaluation outputs

The legacy evaluation pipeline writes topic-wise CSV files such as:

- `odom.csv`
- `scan.csv`
- `episode.csv`
- `start_goal.csv`
- `metrics.csv`

These are consumed by `arena_evaluation` scripts for offline metrics and plotting.

## InternNav evaluation outputs

The newer InternNav path adds structured run artifacts on top of metrics:

- `run_manifest.yaml`
- `internnav_status.json`
- `internnav_trace.jsonl`
- `internnav_diagnostic_summary.json`
- `video_index.json`
- `video_recording_error.txt` (when needed)
- `videos/episode_xxxx/*.mp4`
- snapshot config files used for the run

Social-navigation acceptance runs additionally write:

- `human_states.csv`
- `social_metrics.json`
- `artifact_validation.json`
- `frame_analysis/video_frame_analysis.json` when extracted-frame review has been performed
- `frame_analysis/contact_sheet.jpg` for manual/agent video review

`internnav_trace.jsonl` is the per-decision diagnostic stream.  Each line is a
standalone JSON record that can be parsed after ROS and Isaac have exited.  The
most useful fields are:

- `event_type`: `model_result`, `fallback_command`, or `camera_gate`
- `action.selected`, `action.native_label`, and `action.effective_label`
- `action.invert_discrete_turns`, which records the effective turn-sign mapping used for that decision
- `command.linear_x` and `command.angular_z`
- `goal.goal_distance` and `goal.yaw_error`
- `observation.rgb_available`, `observation.depth_available`, and `observation.sensor_ages_sec`
- `timing.infer_time_sec` and `timing.subprocess_compute_sec`

`internnav_diagnostic_summary.json` aggregates the trace into action counts,
rotation/forward ratios, goal-distance progress, and candidate integration-fault
flags such as `rotate_heavy_low_progress`,
`possible_action_or_yaw_sign_mismatch`, `possible_stale_observations`, and
`missing_camera_inputs`.  It also records `invert_discrete_turns_values` so A/B
runs can be compared without reopening the trace.  The same summary is embedded
into `run_manifest.yaml` under `result.internnav_diagnostic_summary` when it is
available.

`run_manifest.yaml` stores the discrete turn-sign control surface under:

- `parameters.internnav_invert_discrete_turns`: the requested CLI value (`auto`, `true`, or `false`)
- `parameters.internnav_invert_discrete_turns_resolved`: the boolean value passed to the backend
- `parameters.internnav_invert_discrete_turns_source`: `auto_isaac_ai2_bot2` or `cli`

Runs created before tracing was added may still contain videos and
`internnav_status.json` but no `internnav_trace.jsonl` or
`internnav_diagnostic_summary.json`; rerun with the current eval runner when the
post-run summary is required.

## Output root behavior

The current InternNav runner defaults user-facing outputs to a workspace-level `outputs/` directory instead of mixing them into package `install/` content.

This is important because benchmark outputs are generated artifacts, not installed resources.

## Video artifacts

Video artifacts are only produced when the eval is launched with
`--save-eval-video`.  Depending on topic availability, a video-enabled run may
produce:

- `ego_observation.mp4`
- `ego_debug_overlay.mp4`
- `map_top_down_follow.mp4`
- `sim_top_down.mp4`

For InternNav runs, `ego_debug_overlay.mp4` overlays the latest selected action,
native/effective turn label, converted `cmd_vel`, action history, goal distance,
yaw error, and RGB/depth freshness on top of the ego camera frame.  Use it to
correlate what the robot saw with what the model requested.

If `run_manifest.yaml` contains `parameters.save_eval_video: false`, then a
successful run should not be expected to contain `video_index.json` or a
`videos/` directory.  This is normal for quick E2E or metrics-only validation
runs.

### Recent three-container validation output

The validated `hospital_1 + Ai2_Bot2 + InternNav` three-container run wrote its
metrics-only artifacts to:

```text
/home/ubuntu/arena_jazzy_ws/outputs/internnav_three_container_final/20260515_162934_hospital_1_Ai2_Bot2_internnav
```

That run had `finished_observed: true`, `launch_returncode: 0`,
`metrics_returncode: 0`, and `end_reason: finished`.  It intentionally did not
produce videos because it was launched without `--save-eval-video`.

### Recent social-navigation acceptance output

The validated `hospital_1 + Ai2_Bot2 + HuNav + InternNav` social-navigation run wrote artifacts to:

```text
/home/ubuntu/arena_jazzy_ws/outputs/social_nav_acceptance_smoke7/20260516_081500_hospital_1_Ai2_Bot2_internnav
```

Acceptance highlights:

- `artifact_validation.json`: `overall_pass=true`, `social_nav_ready=true`, `warnings=[]`
- `social_metrics.json`: `humans_present=true`, `social_success=true`, `human_sample_count=54`, `max_humans_observed=14`, `odom_sample_count=98`, `large_teleports=[]`
- InternNav diagnostics: `internnav_trace.jsonl` has 1019 records, including 142 `model_result` records; final status is `internnav_command`, `degraded=false`
- Required videos: `ego_observation.mp4` 295 frames, `ego_debug_overlay.mp4` 242 frames, `sim_top_down.mp4` 294 frames, `map_top_down_follow.mp4` 295 frames
- Frame review: `frame_analysis/video_frame_analysis.json` reports `overall_visual_pass=true` and `contact_sheet.jpg` was reviewed

## Codec policy

The current recorder prefers H.264 MP4 output:

1. try `imageio` with `libx264`
2. fallback to OpenCV MP4 writer
3. if the detected codec is not H.264, try `ffmpeg` transcoding
4. verify the final file with `ffprobe`

This makes the benchmark outputs easier to archive, inspect, and share.

## Metrics compatibility

`arena_evaluation/get_metrics.py` includes a directory resolver so that metrics can still be generated when the recorder output layout and the newer manifest-based layout coexist.

## arena_evaluation metric calculation

The legacy `metrics` console script reads per-topic CSV files in a run directory
and writes `metrics.csv`.  The core inputs are:

- `odom.csv`: robot pose `[x, y, yaw]` and velocity samples
- `scan.csv`: laser ranges for base collision checks
- `episode.csv`: episode index per sample
- `start_goal.csv`: start and goal pose samples
- `params.yaml`: robot model metadata, including the collision radius

The main calculated fields are:

| Field | Calculation / source |
| --- | --- |
| `path_length` | Sum of Euclidean distances between consecutive odom positions. |
| `path_length_values` | Per-step odom displacement values. |
| `velocity` | Norm of recorded odom velocity per sample. |
| `acceleration` / `jerk` | First and second differences of speed. |
| `curvature`, `normalized_curvature`, `roughness` | Geometry derived from rolling triples of odom positions. |
| `collision_amount` | Rising-edge count of laser scans with ranges below the robot radius. |
| `result` | `TIMEOUT`, `COLLISION`, or `GOAL_REACHED` based on timeout, collision count, and final distance to goal. |

`metrics.csv` is kept for backward compatibility, but it is not the benchmark
success gate for GRScenes social-navigation runs.  In recorded GRScenes data the
legacy `start_goal.csv` can be stale or incomplete, so a row with
`result=GOAL_REACHED` is only a base-metrics signal.  Benchmark success must be
read from `vln_task_metrics.json`, `social_metrics.json`, and
`artifact_validation.json`.

Generate the base metrics for a run directory with:

```bash
ros2 run arena_evaluation metrics --dir /path/to/run_dir
```

InternNav social-eval runs normally invoke this during post-processing unless
`--skip-metrics` is set.

The Arena evaluator environment must include `pandas`; it is declared by the
`arena_evaluation` package and the top-level Arena Python environment.

## Strict VLN task metrics

Social-VLN benchmark runs write `vln_task_metrics.json` after the base metrics.
This file uses the native GRScenes/Arena scenario metadata as the authoritative
start and goal contract, then audits the executed odom and command streams.

Important output fields:

- `strict_task_success`: true only when the episode does not time out, reaches
  the scenario goal within tolerance, avoids commanded-stuck intervals, and does
  not occupy static map obstacles.
- `strict_task_failure_reasons`: failure taxonomy such as `episode_timeout`,
  `goal_not_reached`, `commanded_stuck`, and `static_occupancy_collision`.
- `goal_metrics.navigation_error_m`: final distance to the scenario goal.
- `goal_metrics.oracle_error_m`: best distance to the scenario goal at any point
  on the executed trajectory.
- `goal_metrics.goal_tolerance_m`: tolerance used for the strict goal check.
- `start_goal_consistency`: comparison between native scenario metadata and
  recorder `start_goal.csv`; mismatches are warnings against trusting legacy
  success fields.
- `commanded_stuck.commanded_stuck_time_sec` and
  `commanded_stuck.commanded_stuck_intervals`: intervals where commands request
  motion but odom shows insufficient progress.
- `static_occupancy.collision_sample_count`,
  `static_occupancy.first_collision_sample`, and `static_occupancy.intervals`:
  map-occupancy evidence for the robot footprint intersecting static obstacles.

Strict task success is the VLN instruction-completion gate currently available
in Arena.  It verifies that the robot reached the annotated target implied by
the recorded instruction and scenario contract.  It does not yet evaluate richer
BDDL predicates beyond the goal contract unless those predicates are projected
into the scenario metadata and strict metrics config.

## Social-navigation metrics

HuNav social metrics are generated by `arena_evaluation.social_metrics` and saved
as `social_metrics.json`.  The script aligns each odom sample with the nearest
`human_states.csv` sample, computes robot-human distances, and applies these
default thresholds:

| Threshold | Default |
| --- | --- |
| `personal_space_radius_m` | `1.0` |
| `near_miss_radius_m` | `0.5` |
| `human_collision_radius_m` | `0.25` |
| `crowd_radius_m` | `1.5` |
| `crowd_freezing_speed_mps` | `0.05` |

Important output fields:

- `humans_present`: true when non-empty human states were observed and aligned with odom
- `max_humans_observed` and `observed_human_ids`: sanity checks for HuNav actor coverage
- `path_length_m`: odom path length computed independently from `metrics.csv`
- `min_human_distance_m`: minimum robot-human distance over aligned samples
- `personal_space_violation_time_sec`: time spent within the personal-space radius
- `near_miss_count`: rising-edge count for entering the near-miss radius
- `human_collision_count`: rising-edge count for entering the human-collision radius
- `crowd_freezing_time_sec`: time inside personal space while robot speed is below the freezing threshold
- `large_teleports`: odom jumps above the teleport threshold, used to reject invalid motion traces
- `social_success`: true only when humans are present, there are no human collisions, no near misses, and no large teleports
- `strict_social_success`: benchmark social gate; this preserves the legacy
  social checks and also fails when the dynamic scene is invalid, footprint
  clearance violates human safety thresholds, or strict task diagnostics expose
  static-obstacle/stuck behavior.
- `strict_social_failure_reasons`: failure taxonomy such as
  `dynamic_scene_failed`, `footprint_human_collision`,
  `footprint_near_miss`, `point_near_miss`, `point_human_collision`,
  `large_teleport`, `static_occupancy_collision`, and `commanded_stuck`.
- `min_distance_sample`: robot/human sample at the minimum point-distance.
- `min_footprint_clearance_m`: minimum clearance after subtracting robot and
  human radii.  Negative values mean the footprints overlap.
- `min_footprint_clearance_sample`: auditable sample containing simulation time,
  robot pose, human id, human position, point distance, and footprint clearance.
- `point_near_miss_events`, `point_human_collision_events`,
  `footprint_near_miss_events`, and `footprint_human_collision_events`:
  event-level evidence for social safety failures.
- `review_intervals`: simulation-time intervals to inspect in the videos for
  commanded-stuck and static-occupancy failures imported from strict task
  metrics.

Run it manually with:

```bash
ros2 run arena_evaluation social_metrics --dir /path/to/run_dir
```

`artifact_validation.json` is the acceptance gate on top of base and social
metrics.  For `dual_vln` social-eval runs, it requires artifacts, human samples,
and robot movement; social acceptance is therefore not inferred from a launch
return code alone.

The validation gate treats strict task and strict social success as required for
`social_nav_ready=true`.  If legacy `metrics.csv` reports `GOAL_REACHED` while
`strict_task_success=false`, validation emits a false-positive warning.  If a
legacy or older social field reports success while `strict_social_success=false`,
validation emits the corresponding social false-positive warning.

## Aggregating Dynamic Social VLN runs

Use the aggregate CLI to summarize one or more run directories into a CSV plus
optional JSON and failure-taxonomy outputs:

```bash
ros2 run arena_bringup social_nav_metrics_aggregate \
  --root /home/ubuntu/arena_jazzy_ws/outputs/hospital_1_demo_001 \
  --output-csv /tmp/social_nav_runs.csv \
  --summary-json /tmp/social_nav_summary.json \
  --failure-csv /tmp/social_nav_failures.csv
```

`--root` recursively discovers directories containing `social_metrics.json` or
`artifact_validation.json`; `--run-dir` can be used repeatedly for explicit run
directories.

The aggregate row includes task success, social success, artifact validation,
path length, goal progress, minimum human distance, social violation counts,
human coverage, InternNav command counts, stale-camera diagnostics, and automatic
failure tags.  Current failure tags include:

- `artifact_failure`
- `missing_humans`
- `no_motion`
- `collision`
- `near_miss`
- `personal_space_violation`
- `legacy_task_false_positive`
- `legacy_social_false_positive`
- `footprint_collision`
- `footprint_near_miss`
- `static_occupancy_collision`
- `commanded_stuck`
- `timeout`
- `debug_overlay_fallback`
- `task_failure`
- `stale_observation_candidate`

`stale_observation_candidate` is treated as a failure tag only for unsuccessful
runs; successful runs may still report stale-record diagnostics for triage.

Aggregate rows expose both legacy and strict rates:

- `legacy_task_success` / `legacy_task_success_rate`: compatibility view based
  on `metrics.csv`.
- `strict_task_success` / `strict_task_success_rate`: benchmark task gate based
  on `vln_task_metrics.json`.
- `legacy_social_success` / `legacy_social_success_rate`: compatibility view
  for older social reports.
- `strict_social_success` / `strict_social_success_rate`: benchmark social gate.
- `benchmark_ready`: true only when strict task, strict social, artifact
  validation, and `social_nav_ready` are all true.

The aggregate also carries audit fields for reviewing failures without opening
every JSON file: `goal_progress_m`, `diagnostic_goal_progress_m`,
`diagnostic_goal_distance_min_m`, `navigation_error_m`,
`min_footprint_clearance_m`, `min_footprint_clearance_time_sec`,
`min_footprint_clearance_human_id`, `static_occupancy_collision_samples`,
`commanded_stuck_time_sec`, and `debug_overlay_fallback`.

`debug_overlay_fallback=true` means `ego_debug_overlay.mp4` exists but was built
from the ego camera fallback path because the model debug image stream was not
available.  The video artifact can still be usable for visual review, but the
run should be tagged so model-side debug stream coverage is not overstated.
Use `debug_overlay_source_status`, `debug_overlay_model_frames`,
`debug_overlay_fallback_frames`, and `debug_overlay_received_count` to
distinguish a fully missing debug stream from a short startup fallback.  For
example, `debug_overlay_source_status=model_debug_image` with non-zero model
frames means the model overlay was recorded after startup, while any non-zero
fallback frames should still be preserved as an instrumentation-readiness tag.

### Recent GRScenes strict benchmark output

The 2026-06-23 `grscenes_5 + Ai2_Bot2 + InternNav` Isaac rerun with debug
overlay source diagnostics wrote complete strict benchmark artifacts to:

```text
/home/ubuntu/arena_jazzy_ws/outputs/grscenes_benchmark_quality_eval_20260623_rerun/20260623_095109_grscenes_5_Ai2_Bot2_internnav
```

This run verifies that the pipeline can automatically produce videos,
`metrics.csv`, `vln_task_metrics.json`, `social_metrics.json`,
`artifact_validation.json`, and aggregate rows.  It is intentionally not a
benchmark success:

- `strict_task_success=false` with `episode_timeout`, `goal_not_reached`,
  and `static_occupancy_collision`.
- `strict_social_success=false` with `footprint_human_collision`,
  `footprint_near_miss`, and `static_occupancy_collision`.
- `dynamic_scene_success=true`, `moving_human_count=2`, and manual review of
  `sim_top_down.mp4` confirms animated HuNav pedestrians are visible and walking.
- Legacy success fields can look optimistic for this run, so aggregate failure
  tags include `legacy_task_false_positive` and `legacy_social_false_positive`.
- `debug_overlay_fallback=true`, but `debug_overlay_source.status` is
  `model_debug_image` with 1066 model frames and 3 startup fallback frames.
  The run remains tagged for readiness review, but the overlay is not empty.
- `frame_analysis/video_frame_analysis.json` records a manual keyframe review:
  `sim_top_down` is visual PASS, `ego_debug_overlay` is
  `PASS_WITH_READINESS_WARNING`, and strict task/social failures remain the
  reason the run is not benchmark-ready.

For the 2026-05-17 video rerun, the aggregate-ready acceptance outputs were:

- `artifact_validation.json`: `overall_pass=true`, `social_nav_ready=true`
- `social_metrics.json`: `humans_present=true`, `social_success=true`, `max_humans_observed=14`, `human_collision_count=0`, `near_miss_count=0`
- videos: 244 frames each for ego, debug overlay, sim top-down, and map top-down
- output directory: `/home/ubuntu/arena_jazzy_ws/outputs/social_nav_video_rerun_20260517/20260517_081450_hospital_1_Ai2_Bot2_internnav`

## What to inspect after a run

After a benchmark or InternNav eval run, check these first:

- did the launch finish cleanly?
- is `run_manifest.yaml` present?
- is `end_reason` equal to `finished`?
- is `video_recorder_returncode` equal to `0` when video recording is enabled?
- do the generated videos have frames and the expected codec?
- does `ego_debug_overlay.mp4` show readable action and command diagnostics?
- does `internnav_trace.jsonl` contain records for model, fallback, or camera-gate events?
- does `internnav_diagnostic_summary.json` report forward progress and avoid rotate-heavy flags?
- did `metrics.csv` get generated when metrics were requested?
- for social-navigation runs, does `artifact_validation.json` report `overall_pass=true` and `social_nav_ready=true`?
- for social-navigation runs, does `social_metrics.json` report non-empty human/odom samples and no large teleports?
- if frame analysis was requested, does `frame_analysis/video_frame_analysis.json` report `overall_visual_pass=true`?
