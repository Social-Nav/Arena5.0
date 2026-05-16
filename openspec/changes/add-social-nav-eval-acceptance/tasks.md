## 1. Eval CLI and Scenario Contract

- [x] 1.1 Add `--tm-obstacles` and scenario-file related options to `internnav_eval.py` and forward them to `arena.launch.py`.
- [x] 1.2 Record `tm_obstacles`, scenario file, and social eval expectations in `run_manifest.yaml`.
- [x] 1.3 Ensure a recommended social eval command can request `tm_robots=scenario`, `tm_obstacles=scenario`, and `task.scenario.file=normal`.

## 2. HuNav Human State Publishing and Recording

- [x] 2.1 Publish `hunav_msgs/Agents` on the expected `human_states` topic from `HunavHumanSimulator` update loop.
- [x] 2.2 Add structured `human_states` CSV serialization to `arena_evaluation` data recorder.
- [x] 2.3 Add compatibility for hospital_1 scenario `waypoint` singular dynamic-agent fields.

## 3. Social Metrics

- [x] 3.1 Implement a reusable social metrics module that reads `odom.csv`, `human_states.csv`, optional `metrics.csv`, and writes `social_metrics.json`.
- [x] 3.2 Compute minimum human distance, personal-space violation time, near-miss count, human collision count, crowd freezing time, and social success.
- [x] 3.3 Invoke social metrics generation from the InternNav eval post-processing path when social eval mode is requested.

## 4. Artifact Validation

- [x] 4.1 Implement `artifact_validation.json` generation for environment/entity, model/control, video, and metrics checks.
- [x] 4.2 Validate required videos and frame counts: `ego_observation`, `ego_debug_overlay`, `sim_top_down`, and `map_top_down_follow`.
- [x] 4.3 Validate InternNav trace/status/model-result availability and basic odom continuity/no-large-teleport checks.
- [x] 4.4 Include failed checks, warnings, and social-nav readiness booleans in the validation report.

## 5. Acceptance Agent

- [x] 5.1 Add a repository-local acceptance-agent instruction file for reviewing videos, extracted frames, metrics, and validation outputs.
- [x] 5.2 Include explicit checks for black ego frames, synthetic gradient frames, human visibility, debug overlay readability, and social metrics completeness.

## 6. Verification

- [x] 6.1 Run Python syntax checks for modified modules.
- [x] 6.2 Run social metrics and artifact validation against an existing incomplete run to confirm missing humans/overlay/metrics are flagged.
- [x] 6.3 Run available targeted tests or lightweight import checks without requiring a full Isaac simulation.
- [x] 6.4 Run final `hospital_1 + Ai2_Bot2 + HuNav + InternNav` smoke validation with videos and social metrics.
- [x] 6.5 Extract first/middle/last frames from all required videos, generate `frame_analysis/video_frame_analysis.json`, and rerun artifact validation.
- [x] 6.6 Run the social-nav acceptance review checklist against the final output directory.

Final accepted output directory:

```text
/home/ubuntu/arena_jazzy_ws/outputs/social_nav_acceptance_smoke7/20260516_081500_hospital_1_Ai2_Bot2_internnav
```

Final acceptance summary:

- `artifact_validation.json`: `overall_pass=true`, `social_nav_ready=true`, `warnings=[]`
- `social_metrics.json`: `humans_present=true`, `social_success=true`, `human_sample_count=54`, `max_humans_observed=14`, `odom_sample_count=98`, `large_teleports=[]`
- InternNav trace: 1019 records, including 142 `model_result` records
- Required videos: ego 295 frames, debug overlay 242 frames, sim top-down 294 frames, map top-down 295 frames
- Frame analysis: `overall_visual_pass=true`
