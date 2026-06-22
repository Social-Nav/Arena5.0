## ADDED Requirements

### Requirement: Strict social safety metric artifact
The system SHALL generate strict social-navigation safety metrics for each HuNav eval run and store them in a machine-readable artifact that can be consumed by validation and aggregate reporting.

#### Scenario: Robot and human state artifacts exist
- **WHEN** odometry and HuNav human-state CSV artifacts contain overlapping samples
- **THEN** the strict social safety metrics SHALL include footprint-aware human distance, near-miss count, human collision count, personal-space violation time, crowd-freezing time, human motion summary, dynamic-scene overlap, and strict social success.

#### Scenario: Human state artifacts are missing
- **WHEN** both `human_states.csv` and `pedsim_agents_data.csv` are missing or contain no agents
- **THEN** strict social safety metrics SHALL be written with `strict_social_success=false` and `humans_present=false`.

### Requirement: Footprint-aware human collision detection
The system SHALL evaluate robot-human safety using robot footprint radius and configurable human radius rather than only point-to-point distance thresholds.

#### Scenario: Robot footprint intersects human safety radius
- **WHEN** the robot footprint radius plus human radius overlaps a HuNav agent position
- **THEN** the strict social safety metrics SHALL increment human collision or overlap counts and SHALL set `strict_social_success=false`.

#### Scenario: Robot enters personal space but does not collide
- **WHEN** robot-human clearance is below the personal-space threshold but above the strict collision threshold
- **THEN** the strict social safety metrics SHALL accumulate personal-space violation time and SHALL report whether the violation exceeds benchmark limits.

### Requirement: Dynamic-scene validity
The system SHALL require dynamic humans and robot motion to overlap in simulation time for a run to be considered a valid social-navigation benchmark episode.

#### Scenario: Humans and robot move together
- **WHEN** at least the configured number of humans move for the configured duration and robot-human motion overlap exceeds configured thresholds
- **THEN** dynamic-scene validity SHALL pass.

#### Scenario: Humans move only before or after robot activity
- **WHEN** HuNav trajectories are present but robot motion does not overlap human motion long enough
- **THEN** dynamic-scene validity SHALL fail and strict benchmark readiness SHALL be false.

### Requirement: Static obstacle safety integration
The system SHALL include static obstacle occupancy results in strict social-navigation safety because safe social navigation requires avoiding both people and scene geometry.

#### Scenario: Static obstacle collision is detected
- **WHEN** strict task metrics report occupied robot-footprint samples or intervals above threshold
- **THEN** strict social safety metrics or validation SHALL set `strict_social_success=false` and include a static obstacle collision reason.

#### Scenario: No static obstacle collision source exists
- **WHEN** neither map occupancy, LaserScan collision, nor approved Isaac contact data is available
- **THEN** validation SHALL mark social benchmark readiness as incomplete rather than assuming obstacle safety passed.

### Requirement: Commanded-stuck safety integration
The system SHALL treat long commanded-stuck intervals as social-navigation benchmark failures because the robot is not safely completing the interaction.

#### Scenario: Long commanded-stuck interval occurs
- **WHEN** strict task metrics report total commanded-stuck time above the configured threshold
- **THEN** strict social success SHALL be false or validation SHALL fail benchmark readiness with a `commanded_stuck` reason.

#### Scenario: No long commanded-stuck interval occurs
- **WHEN** commanded-stuck intervals are absent or below configured tolerance
- **THEN** strict social safety SHALL not fail on stuck behavior.

### Requirement: Strict social aggregate reporting
The system SHALL report strict social success separately from legacy social success in aggregate benchmark summaries.

#### Scenario: Aggregating multiple social-navigation runs
- **WHEN** benchmark aggregation processes multiple run directories
- **THEN** the aggregate output SHALL include `legacy_social_success_rate`, `strict_social_success_rate`, dynamic-scene pass rate, obstacle collision rate, human collision rate, near-miss rate, and stuck-failure rate.

#### Scenario: Legacy and strict social success disagree
- **WHEN** a run has legacy `social_success=true` but `strict_social_success=false`
- **THEN** the aggregate output SHALL include that run in a false-positive diagnostics list with the strict failure reasons.

### Requirement: Video review traceability
The system SHALL preserve links from strict metric failures to video artifacts and frame-analysis hints for manual review.

#### Scenario: Strict failure is reported
- **WHEN** strict task or social metrics fail for a run that has video artifacts
- **THEN** validation SHALL include the relevant video paths and approximate simulation-time intervals for review.

#### Scenario: Video artifacts are unavailable
- **WHEN** strict metric failures occur and required videos are missing
- **THEN** validation SHALL fail video readiness and still report metric-derived failure intervals.
