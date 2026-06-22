## ADDED Requirements

### Requirement: Strict VLN task metric artifact
The system SHALL generate strict VLN task metrics for each InternNav eval run and store them in a machine-readable artifact that can be consumed by validation and aggregate reporting.

#### Scenario: Required run artifacts exist
- **WHEN** an eval run directory contains odometry, command velocity, manifest, scenario start/goal, and base metrics artifacts
- **THEN** the strict VLN task metric artifact SHALL include `strict_task_success`, final goal distance, goal tolerance, timeout status, path length, robot progress, stuck intervals, static obstacle collision summary, and references to source artifacts.

#### Scenario: Required run artifacts are missing
- **WHEN** one or more required inputs for strict task evaluation are unavailable
- **THEN** the strict VLN task metric artifact SHALL be written with `strict_task_success=false` and a list of missing inputs.

### Requirement: Scenario-bound instruction success
The system SHALL treat recorded GRScenes instruction success as reaching the scenario-defined navigation goal within a configured tolerance while satisfying strict task validity checks.

#### Scenario: Robot reaches the scenario goal
- **WHEN** the final robot pose is within the configured tolerance of the native scenario goal and no strict task validity check fails
- **THEN** `strict_task_success` SHALL be true.

#### Scenario: Legacy result reports success but final goal is not reached
- **WHEN** `metrics.csv` reports `GOAL_REACHED` but the final robot pose is outside the configured scenario goal tolerance
- **THEN** `strict_task_success` SHALL be false and the report SHALL include a `goal_tolerance_failed` reason.

### Requirement: Start-goal consistency validation
The system SHALL verify that recorder start/goal artifacts agree with native scenario start/goal metadata before using them for benchmark scoring.

#### Scenario: Recorder and scenario start-goal agree
- **WHEN** `start_goal.csv` and the native scenario `robots` start/goal differ by no more than configured position and yaw tolerances
- **THEN** the task metrics SHALL mark start-goal consistency as passing.

#### Scenario: Recorder and scenario start-goal disagree
- **WHEN** `start_goal.csv` is missing, contains zeros for a nonzero scenario, or differs from native scenario metadata beyond tolerance
- **THEN** strict task validation SHALL fail or mark the recorder start-goal as unusable and use the native scenario metadata as the scoring source.

### Requirement: Commanded-stuck detection
The system SHALL detect intervals where the robot receives sustained motion commands but makes insufficient physical progress.

#### Scenario: Sustained command produces negligible movement
- **WHEN** command velocity magnitude exceeds the configured threshold for at least the configured duration and odometry displacement remains below the configured minimum
- **THEN** the strict task metrics SHALL report a stuck interval with start time, end time, displacement, command summary, and `strict_task_success=false` when total stuck time exceeds the allowed threshold.

#### Scenario: Short startup pause occurs
- **WHEN** a stuck-like interval is shorter than the configured grace duration
- **THEN** the interval MAY be reported as diagnostic data but SHALL NOT by itself fail strict task success.

### Requirement: Static obstacle occupancy check
The system SHALL detect robot footprint overlap with occupied map cells for GRScenes runs with map metadata.

#### Scenario: Robot footprint overlaps occupied cells
- **WHEN** odometry transformed through `map.yaml` places the robot footprint over occupied pixels for one or more samples
- **THEN** the strict task metrics SHALL report occupied samples and intervals and SHALL fail strict task success when the configured collision duration or sample threshold is exceeded.

#### Scenario: Map metadata is unavailable
- **WHEN** the run world does not provide a usable `map.yaml` and map image
- **THEN** strict task metrics SHALL report `static_obstacle_check_available=false` and validation SHALL treat benchmark readiness as incomplete unless another approved collision source is present.

### Requirement: Strict task aggregate reporting
The system SHALL report strict task success separately from legacy task success in aggregate benchmark summaries.

#### Scenario: Aggregating multiple runs
- **WHEN** benchmark aggregation processes multiple run directories
- **THEN** the aggregate output SHALL include both `legacy_task_success_rate` and `strict_task_success_rate`.

#### Scenario: Legacy and strict success disagree
- **WHEN** a run has legacy `GOAL_REACHED` but `strict_task_success=false`
- **THEN** the aggregate output SHALL include that run in a false-positive diagnostics list.
