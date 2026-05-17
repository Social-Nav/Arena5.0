## Context

The phase-1 Dynamic Social VLN workflow is an overlay benchmark: Arena world
geometry, native scenario files, HuNav dynamic agents, and InternNav execution
remain in their existing packages.  The new layer must validate that those pieces
are present and then launch the already-tested eval runner with reproducible
arguments.

## Decisions

### Decision 1: Scenario YAML is an overlay, not a replacement

The scenario config references `world.yaml`, `map.yaml`, and a native Arena
scenario.  It adds language, task predicates, social constraints, metrics, pass
criteria, and artifact requirements.  This avoids duplicating world geometry or
HuNav actor definitions.

### Decision 2: Runner wrapper derives `internnav_eval` arguments

`social_nav_scenario_eval` validates the YAML and translates it into
`internnav_eval` args.  The wrapper sets `--social-eval`, scenario task modes,
the native scenario name, language instruction, scenario metadata, and
`--internnav-mode internnav`.  Extra args after `--` are appended so runtime
choices such as external server mode, headless mode, device, model path, or
camera topics remain user-controlled.

### Decision 3: Aggregation is manifest/artifact based

`social_nav_metrics_aggregate` discovers run directories by markers such as
`social_metrics.json` and `artifact_validation.json`, then combines manifest,
social metrics, artifact validation, and InternNav diagnostic summary fields.
Failure tags are automatic but successful task/social runs are not downgraded
solely because stale-observation diagnostics exist.

### Decision 4: Validate in the Arena Jazzy container

The Docker image `arena:dev` is built from the Jazzy base and contains
`/opt/ros/jazzy/setup.bash`.  The host may have Humble installed for unrelated
work; build and ROS CLI validation for this workflow should run in the Arena
container.

## Validation

- `colcon build --paths src/Arena/arena_bringup` passed inside `arena:dev` with `/opt/ros/jazzy/setup.bash`.
- `social_nav_scenario_validate` passed for `hospital_1_demo_001.yaml`.
- `social_nav_scenario_eval --dry-run` produced the expected `internnav_eval` command with `--internnav-mode internnav`.
- A video rerun produced four MP4 artifacts and passed social/artifact validation:
  `/home/ubuntu/arena_jazzy_ws/outputs/social_nav_video_rerun_20260517/20260517_081450_hospital_1_Ai2_Bot2_internnav`.
