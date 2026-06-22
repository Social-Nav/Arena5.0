## Why

Current GRScenes + InternNav runs can report `task_success=100%` and `social_success=100%` even when video review shows the robot stuck, intersecting static obstacles, or visually passing through humans. The benchmark needs stricter, explicit task and social-navigation metrics before aggregate scores can be treated as meaningful.

## What Changes

- Add strict VLN task metrics that evaluate whether the robot followed the episode instruction using scenario-bound start/goal metadata, final-goal tolerance, timeout, robot progress, stuck behavior, and command/control evidence.
- Add social-navigation safety metrics that go beyond point-distance HuNav checks by including static-map occupancy collisions, robot-footprint-vs-human proximity, commanded-stuck intervals, large teleports, and dynamic-scene interaction requirements.
- Update artifact validation and aggregate reporting to distinguish legacy metrics from strict benchmark metrics.
- Add diagnostics that make failures auditable from JSON/CSV artifacts and link back to videos for manual review.
- Preserve existing `metrics.csv` and `social_metrics.json` fields for compatibility, but do not use legacy `GOAL_REACHED` alone as benchmark task success.

## Capabilities

### New Capabilities
- `vln-task-metrics`: Defines strict task-success metrics for instruction-following navigation in recorded GRScenes episodes.
- `social-navigation-safety-metrics`: Defines strict social-navigation and safety metrics for HuNav dynamic humans, static obstacles, stuck behavior, and aggregate social success.

### Modified Capabilities
- None.

## Impact

- Affected packages: `arena_evaluation`, `arena_bringup`, and GRScenes eval postprocessing.
- Affected artifacts: `metrics.csv`, `social_metrics.json`, `artifact_validation.json`, aggregate benchmark summaries, and run manifests.
- Affected workflows: Docker + Isaac eval postprocessing, social-nav validation, Lark benchmark status reporting, and manual video review triage.
