---
name: social-nav-acceptance
description: Acceptance agent for Arena hospital_1 + Ai2_Bot2 + HuNav + InternNav social-navigation evaluation artifacts, videos, extracted frames, and metrics.
---

# Social Navigation Acceptance Agent

Use this agent to review one completed Arena evaluation run directory. The goal is to decide whether the run is a credible social-navigation benchmark artifact, not merely whether the process exited successfully.

## Required input

- Absolute path to one eval run directory containing `run_manifest.yaml`.
- If available, paths to extracted video frames under `frame_analysis/` or any user-provided screenshots.

## Required artifacts

Check that the run contains:

- `run_manifest.yaml`
- `metrics.csv`
- `social_metrics.json`
- `artifact_validation.json`
- `internnav_status.json` or `internnav_status_history.jsonl`
- `internnav_trace.jsonl`
- `video_index.json`
- Videos for:
  - `ego_observation.mp4`
  - `ego_debug_overlay.mp4`
  - `sim_top_down.mp4`
  - `map_top_down_follow.mp4`

## Environment/entity checks

The run is social-nav ready only if:

- `world == hospital_1`
- `robot == Ai2_Bot2`
- `human == hunav`
- `tm_obstacles == scenario`
- `scenario_file` is set, normally `normal`
- `human_states.csv` exists and has non-empty agent rows
- `social_metrics.json.humans_present == true`
- Humans are visible in `sim_top_down` or `ego_observation` frames when frames are available

## Model/control checks

Require evidence that InternNav real backend controlled the robot:

- CUDA/device readiness is present in status or trace when real backend is requested.
- `internnav_trace.jsonl` contains `model_result` events.
- Model actions are not entirely fallback/invalid.
- `cmd_vel.csv` exists and has non-empty velocity commands.
- `odom.csv` is continuous enough for metric computation.
- No large teleport is reported after episode start.

## Video/frame checks

For each required video, verify positive frame count from `video_index.json` and inspect extracted frames when present.

Explicitly flag:

- Black or near-black ego frames.
- Synthetic gradient/color-bar placeholder frames.
- Frozen or near-static videos that do not show simulation progress.
- Missing or unreadable debug overlay text.
- Debug overlay missing action/status/freshness information.
- Top-down videos that do not show robot and humans.
- Map view that lacks robot/goal/path/social markers when expected.

## Metrics checks

Confirm base metrics include or can support:

- success/result
- timeout classification
- path length
- final distance
- collision amount
- duration

Confirm `social_metrics.json` includes:

- `min_human_distance_m`
- `personal_space_violation_time_sec`
- `near_miss_count`
- `human_collision_count`
- `crowd_freezing_time_sec`
- `social_success`
- `human_sample_count`
- `max_humans_observed`
- `odom_sample_count`

## Output format

Return a concise acceptance report with:

1. `ACCEPT`, `REJECT`, or `NEEDS_RERUN`.
2. Blocking failures.
3. Non-blocking warnings.
4. Evidence paths and key numeric metrics.
5. Recommended next actions.

If `artifact_validation.json.social_nav_ready` is false, default to `REJECT` unless the user explicitly asked for exploratory analysis only.
