## ADDED Requirements

### Requirement: Scenario overlay validation
The system SHALL validate a benchmark-level Dynamic Social VLN scenario overlay before launching a simulation.

#### Scenario: Valid overlay references native Arena assets
- **WHEN** a scenario YAML references an existing world config, map YAML, native scenario file, robot start/goal, language instruction, HuNav expected count, evaluation metrics, and required artifacts
- **THEN** validation SHALL pass and return a normalized scenario dictionary.

#### Scenario: Invalid overlay is incomplete
- **WHEN** required sections, poses, files, human counts, predicates, metric lists, or artifact lists are missing or inconsistent
- **THEN** validation SHALL report path-specific errors and the eval wrapper SHALL refuse to launch.

### Requirement: Scenario eval wrapper
The system SHALL provide a `social_nav_scenario_eval` CLI that derives an `internnav_eval` command from a validated scenario overlay.

#### Scenario: Dry run requested
- **WHEN** a user runs `social_nav_scenario_eval --dry-run --scenario-config <yaml>`
- **THEN** the wrapper SHALL print the derived `ros2 run arena_bringup internnav_eval ...` command without launching the simulation.

#### Scenario: Full run requested
- **WHEN** a user runs `social_nav_scenario_eval --scenario-config <yaml>` without dry-run
- **THEN** the wrapper SHALL invoke `internnav_eval` with social eval mode, scenario task modes, native scenario name, language instruction, `--internnav-mode internnav`, and scenario metadata arguments.

#### Scenario: Extra runtime args provided
- **WHEN** arguments are provided after `--`
- **THEN** the wrapper SHALL append them to the derived `internnav_eval` argv so runtime controls remain overrideable.

### Requirement: Scenario metadata in manifest
The eval runner SHALL record scenario config identity in the run manifest.

#### Scenario: Scenario wrapper provides metadata
- **WHEN** `internnav_eval` receives `--scenario-config-id` and `--scenario-config-path`
- **THEN** `run_manifest.yaml` SHALL include those values under `parameters`.

### Requirement: Social navigation aggregation
The system SHALL provide a CLI to aggregate social-navigation eval outputs across run directories.

#### Scenario: Runs discovered under root
- **WHEN** `social_nav_metrics_aggregate --root <dir>` is executed
- **THEN** the CLI SHALL recursively discover run directories containing social metrics or artifact validation outputs and write a row per run.

#### Scenario: Summary requested
- **WHEN** `--summary-json` or `--failure-csv` is provided
- **THEN** the CLI SHALL write aggregate success rates and failure-tag counts.

#### Scenario: Successful run has stale diagnostics
- **WHEN** a run has both task success and social success but reports stale-observation diagnostics
- **THEN** the aggregation SHALL keep the run successful and SHALL NOT emit `stale_observation_candidate` as a failure tag.

### Requirement: Jazzy container execution guidance
The documentation SHALL distinguish host ROS installations from the Arena Docker runtime used for Jazzy validation.

#### Scenario: Host lacks `/opt/ros/jazzy`
- **WHEN** the host shell only has `/opt/ros/humble`
- **THEN** docs SHALL instruct users to run build/eval commands inside the Arena container where `/opt/ros/jazzy/setup.bash` exists.
