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

## Codec policy

The current recorder prefers H.264 MP4 output:

1. try `imageio` with `libx264`
2. fallback to OpenCV MP4 writer
3. if the detected codec is not H.264, try `ffmpeg` transcoding
4. verify the final file with `ffprobe`

This makes the benchmark outputs easier to archive, inspect, and share.

## Metrics compatibility

`arena_evaluation/get_metrics.py` includes a directory resolver so that metrics can still be generated when the recorder output layout and the newer manifest-based layout coexist.

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
