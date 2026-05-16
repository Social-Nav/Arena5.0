## ADDED Requirements

### Requirement: Structured HuNav human-state recording
The system SHALL record HuNav `human_states` into CSV rows that can be parsed into per-timestep agent dictionaries containing at least agent id, name, position, orientation, and velocity when available.

#### Scenario: Human-state message is received
- **WHEN** the data recorder receives a `hunav_msgs/Agents` message on the configured `human_states` topic
- **THEN** it SHALL write a structured, parseable representation of all agents to `human_states.csv`.

#### Scenario: No human-state message is received
- **WHEN** no `human_states` messages are received during a run
- **THEN** `human_states.csv` SHALL remain empty or contain empty data rows and downstream validation SHALL treat humans as absent.

### Requirement: HuNav social metrics output
The system SHALL compute social navigation metrics from robot odometry and structured HuNav human states and write them to `social_metrics.json`.

#### Scenario: Robot and human states are available
- **WHEN** `odom.csv` and `human_states.csv` contain overlapping episode samples
- **THEN** the metrics generator SHALL write `social_metrics.json` with per-episode and aggregate social metrics.

#### Scenario: Human states are unavailable
- **WHEN** `human_states.csv` is missing or contains no agents
- **THEN** `social_metrics.json` SHALL still be written with `humans_present=false` and failed social readiness indicators.

### Requirement: Required social metrics
The system SHALL compute the requested social metrics: minimum human distance, personal-space violation time, near-miss count, human collision count, crowd freezing time, and social success.

#### Scenario: Distances cross social thresholds
- **WHEN** robot-human distances fall below configured personal-space, near-miss, or collision thresholds
- **THEN** the corresponding duration or count metrics SHALL increase for that episode.

#### Scenario: No social violations occur
- **WHEN** all robot-human distances remain above configured social thresholds and the base navigation result is successful
- **THEN** social success SHALL be true for that episode.

### Requirement: Basic navigation metric availability
The system SHALL keep producing base navigation metrics required for social eval acceptance, including success/result, timeout classification, path length, final distance, collision amount, and duration.

#### Scenario: Base metrics command succeeds
- **WHEN** the base metrics command completes successfully
- **THEN** `metrics.csv` SHALL exist and validation SHALL read required base fields from it or from compatible generated metric artifacts.

#### Scenario: Base metrics command is skipped or fails
- **WHEN** metrics are skipped or `metrics.csv` is missing
- **THEN** social eval acceptance SHALL fail the metrics check.

### Requirement: HuNav scenario compatibility for dynamic humans
The system SHALL support existing hospital_1 HuNav scenario definitions sufficiently to spawn dynamic humans for social eval runs.

#### Scenario: Scenario uses waypoint field
- **WHEN** a HuNav dynamic obstacle uses a singular `waypoint` field in scenario YAML
- **THEN** the conversion to HuNav goals SHALL treat it as an equivalent waypoint list.

#### Scenario: Scenario includes dynamic agents
- **WHEN** `tm_obstacles=scenario` loads a hospital_1 scenario containing dynamic agents
- **THEN** HuNav SHALL receive one or more agents for registration and publishing.
