## 1. Scenario Config and Validation

- [x] 1.1 Add sample Dynamic Social VLN scenario overlay for `hospital_1_demo_001`.
- [x] 1.2 Implement `SocialNavScenario` loading, defaults, path resolution, and validation.
- [x] 1.3 Warn rather than default to SPL-like metrics for dynamic social navigation.

## 2. Runner and Manifest Integration

- [x] 2.1 Add `social_nav_scenario_validate` console script.
- [x] 2.2 Add `social_nav_scenario_eval` console script that translates scenario YAML into `internnav_eval` args.
- [x] 2.3 Record `scenario_config_id` and `scenario_config_path` in `run_manifest.yaml`.
- [x] 2.4 Default scenario-wrapper runs to `--internnav-mode internnav` so video/topic defaults resolve correctly.

## 3. Aggregation

- [x] 3.1 Add `social_nav_metrics_aggregate` console script.
- [x] 3.2 Summarize task/social/artifact metrics, human coverage, command counts, and failure tags.
- [x] 3.3 Treat stale-observation diagnostics as a failure tag only for unsuccessful runs.

## 4. Documentation

- [x] 4.1 Update MkDocs overview with Dynamic Social VLN overlay model and SPL rationale.
- [x] 4.2 Update running docs with scenario validation/eval commands and Jazzy container guidance.
- [x] 4.3 Update metrics docs with base metrics, social metrics, and aggregation usage.

## 5. Verification

- [x] 5.1 Run Python compile checks for modified modules.
- [x] 5.2 Build `arena_bringup` inside the Arena Jazzy container.
- [x] 5.3 Validate installed CLI entrypoints in the Jazzy container.
- [x] 5.4 Build MkDocs in strict mode.
- [x] 5.5 Run a video-enabled social-nav scenario rerun and verify `overall_pass=true`.

Latest video rerun:

```text
/home/ubuntu/arena_jazzy_ws/outputs/social_nav_video_rerun_20260517/20260517_081450_hospital_1_Ai2_Bot2_internnav
```

Summary:

- `artifact_validation.json`: `overall_pass=true`, `social_nav_ready=true`
- `social_metrics.json`: `humans_present=true`, `social_success=true`, `max_humans_observed=14`, `near_miss_count=0`, `human_collision_count=0`
- videos: ego/debug/sim-top-down/map-top-down MP4s, 244 frames each
