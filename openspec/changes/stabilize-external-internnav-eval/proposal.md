## Why

The three-container `arena + isaac + internnav` path can currently regress from a healthy run like iter8 to a non-moving robot like iter11 when DDS environment, external server discovery, or debug/status streams drift between containers.  We need to make external InternNav eval readiness explicit and fail-fast so `hospital_1 + Ai2_Bot2 + HuNav + InternNav` runs distinguish infrastructure failures from model behavior failures.

## What Changes

- Add deterministic ROS DDS environment handling for external InternNav evals, including recorded `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, and discovery-related environment in manifests.
- Add a preflight phase for `--internnav-external-server` that verifies the namespace-local `get_command` service and InternNav status topic are discoverable before the full episode consumes time.
- Strengthen artifact diagnostics so runs with no model-control loop, missing debug overlay, or zero motion are reported with actionable reasons instead of only generic validation failures.
- Preserve the existing iter8-successful social-navigation acceptance criteria: social pipeline readiness remains separate from `GOAL_REACHED` task success.
- Keep in-process/single-container InternNav behavior compatible; external-server checks only apply when external mode is requested.

## Capabilities

### New Capabilities

- `external-internnav-eval-readiness`: Defines deterministic readiness, preflight, and diagnostics for external InternNav server eval runs.

### Modified Capabilities

- `dual-vln-eval-runner`: External-server eval runs must record DDS environment and preflight outcomes before episode execution.
- `dual-vln-debug-observability`: Eval artifacts must diagnose missing status/trace/overlay and no-motion symptoms as infrastructure issues when applicable.

## Impact

- Affects `arena_bringup/arena_bringup/internnav_eval.py` for DDS env normalization, preflight execution, manifest fields, and post-run diagnostics.
- Affects `arena_bringup/arena_bringup/social_nav_validation.py` for model-control/video diagnostics and optional no-motion warnings.
- May affect task-generator launch/runtime only if readiness gates require additional status/service barriers; no breaking launch argument changes are planned.
- No new third-party dependencies; uses ROS 2 CLI/rclpy already available in the Arena container.
