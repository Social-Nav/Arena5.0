## Why

The hospital_1 + Ai2_Bot2 + InternNav eval pipeline now produces synchronized videos, odom, and H.264 artifacts, but the latest successful run shows the robot spending the episode mostly rotating in place. We need a model-inference diagnostic layer to distinguish navigation-policy failure from adapter/action-conversion, observation, goal, or camera-alignment issues.

## What Changes

- Add structured per-inference and degraded-decision diagnostics that capture ego observation freshness, goal geometry, selected discrete action, converted velocity command, yaw error, action history, turn-sign correction state, and backend timing.
- Add an action/decision visualization overlay so reviewers can inspect what InternNav output for the current ego frame directly in eval videos.
- Add post-run analysis utilities or recorder output that summarize action distribution, rotation ratio, progress-to-goal, and candidate integration faults.
- Use the diagnostics to identify and fix likely causes of persistent rotate actions, with implemented automatic flags for action/yaw sign mismatch, stale observations, and missing camera inputs, plus trace/overlay evidence for broader camera orientation, goal-frame, or instruction-context investigation.
- Add a scoped, recorded discrete turn-sign correction control for Isaac + Ai2_Bot2 validation and reproducible A/B runs.
- Preserve existing successful eval artifacts and make diagnostics opt-in or safe for normal eval runs.

## Capabilities

### New Capabilities
- `internnav-inference-diagnostics`: Captures and visualizes InternNav model decisions, action conversion, observation metadata, and navigation progress for eval debugging.

### Modified Capabilities
- None.

## Impact

- Affects `arena_vln_models` InternNav adapter/server debug payloads and visualization image generation.
- Affects `arena_bringup` eval video recording if additional diagnostic channels or overlays are recorded.
- May affect task/eval output manifests with new diagnostic summary files.
- No expected breaking API changes; existing eval flags should continue to work.
