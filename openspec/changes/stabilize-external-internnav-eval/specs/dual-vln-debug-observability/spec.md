## ADDED Requirements

### Requirement: Artifact validation MUST identify missing model-control loop
The validation report SHALL distinguish missing model-control artifacts from social-metric failures.

#### Scenario: Trace and status are absent
- **WHEN** a run has no InternNav trace records and no status snapshot
- **THEN** `artifact_validation.json` MUST fail the model-control check and include diagnostic fields that indicate trace/status absence

### Requirement: Artifact validation MUST surface no-motion diagnostics
The validation report SHALL expose low robot path length as a diagnostic warning when social metrics are present.

#### Scenario: Robot path is near zero
- **WHEN** social metrics report a very small path length for a dual-VLN eval
- **THEN** `artifact_validation.json` MUST include a warning that the robot appears stationary

#### Scenario: Robot path is non-zero
- **WHEN** social metrics report meaningful path length
- **THEN** `artifact_validation.json` MUST NOT add the stationary-robot warning

### Requirement: Debug overlay absence MUST be reported directly
The validation report SHALL identify missing `ego_debug_overlay` video as an overlay-specific problem.

#### Scenario: Debug overlay is missing
- **WHEN** `video_index.json` lacks frames for `ego_debug_overlay`
- **THEN** `artifact_validation.json` MUST report that video check failure with the missing overlay path and zero frame count
