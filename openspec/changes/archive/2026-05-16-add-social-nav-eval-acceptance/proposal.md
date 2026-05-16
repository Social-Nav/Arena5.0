## Why

The current hospital_1 + Ai2_Bot2 + InternNav pipeline can generate basic Isaac videos, but a run can still appear successful while lacking dynamic humans, social-navigation metrics, debug overlays, model traces, or reliable command/sensor recordings. We need an explicit social-navigation acceptance contract so benchmark runs fail visibly unless they contain the entities, videos, diagnostics, and metrics required for a credible social navigation evaluation.

## What Changes

- Add a first-class social navigation eval mode for InternNav runs that can request `tm_obstacles=scenario` and a scenario file from the eval CLI.
- Record HuNav `human_states` in a structured CSV format and keep rosbag recording compatible.
- Generate social navigation metrics from HuNav human states, including human distance, personal-space violations, near misses, crowd freezing, and social success.
- Generate an `artifact_validation.json` acceptance report that validates environment/entity presence, model/control diagnostics, video artifacts, and metrics.
- Add an acceptance-oriented agent instruction file for checking social navigation outputs and extracted video frames.
- Preserve existing legacy metrics and video outputs while making missing social-navigation artifacts explicit failures or warnings.

## Capabilities

### New Capabilities
- `social-nav-eval-acceptance`: Defines the acceptance contract for hospital_1 + Ai2_Bot2 + HuNav + InternNav social navigation eval outputs, including entities, control diagnostics, videos, metrics, and validation reporting.
- `hunav-social-metrics`: Defines structured HuNav human-state recording and social navigation metric generation from robot and human trajectories.

### Modified Capabilities
- None.

## Impact

- Affected packages:
  - `arena_bringup`: InternNav eval CLI, output manifest, artifact validation post-processing, external-server diagnostics contract.
  - `arena_evaluation`: data recorder structured HuNav recording and social metrics generation.
  - `task_generator`: HuNav human-state publishing and scenario compatibility fixes needed for reliable hospital_1 dynamic agents.
  - `.claude/agents` or repository agent instructions: social navigation acceptance checker instructions.
- Affected outputs:
  - `human_states.csv`
  - `metrics.csv`
  - `social_metrics.json`
  - `artifact_validation.json`
  - `internnav_status.json`, `internnav_status_history.jsonl`, `internnav_trace.jsonl`, `internnav_diagnostic_summary.json`
  - `videos/episode_*/ego_observation.mp4`, `ego_debug_overlay.mp4`, `sim_top_down.mp4`, `map_top_down_follow.mp4`
- No breaking changes are intended; existing eval flows should remain valid, while social navigation runs gain stricter validation artifacts.
