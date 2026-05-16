## ADDED Requirements

### Requirement: External InternNav evals MUST normalize DDS environment
The system SHALL ensure external InternNav eval subprocesses have explicit ROS DDS environment values for domain and RMW implementation before launch or preflight checks run.

#### Scenario: DDS env is absent
- **WHEN** an external InternNav eval starts without `ROS_DOMAIN_ID` or `RMW_IMPLEMENTATION` in the Arena process environment
- **THEN** the eval runner MUST default them to the validated external-server values and record the resolved values in the run manifest

#### Scenario: DDS env is explicitly provided
- **WHEN** an external InternNav eval starts with explicit DDS environment values
- **THEN** the eval runner MUST preserve those values and record them in the run manifest

### Requirement: External InternNav server MUST pass preflight discovery
The system SHALL verify the namespace-local InternNav command service and status topic are discoverable before running a full external-server eval episode.

#### Scenario: External server is discoverable
- **WHEN** `--internnav-external-server` is requested and the expected `get_command` service and status topic are visible
- **THEN** the eval runner MUST record a passing preflight result and continue to launch the eval

#### Scenario: External server is not discoverable
- **WHEN** `--internnav-external-server` is requested and the expected `get_command` service or status topic cannot be discovered within the timeout
- **THEN** the eval runner MUST record the failed preflight result and stop before consuming a full episode

### Requirement: External readiness artifacts MUST be actionable
The system SHALL write enough preflight detail to explain why an external InternNav run did not reach the model-control loop.

#### Scenario: Preflight fails
- **WHEN** external-server preflight fails
- **THEN** the manifest MUST include the expected service, expected status topic, resolved DDS environment, command outputs, timeout, and missing checks

#### Scenario: Preflight passes
- **WHEN** external-server preflight passes
- **THEN** the manifest MUST include the same fields with `pass=true` so later artifact validation can distinguish discovery success from model behavior failures
