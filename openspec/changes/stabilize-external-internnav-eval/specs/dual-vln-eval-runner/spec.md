## ADDED Requirements

### Requirement: Eval manifest MUST record external-server runtime context
The eval runner SHALL record whether external InternNav server mode was requested and the DDS/runtime context used to run discovery and launch commands.

#### Scenario: External mode is enabled
- **WHEN** a dual-VLN eval runs with `--internnav-external-server`
- **THEN** `run_manifest.yaml` MUST include external server mode, resolved DDS environment, and preflight result fields

#### Scenario: External mode is disabled
- **WHEN** a dual-VLN eval runs without `--internnav-external-server`
- **THEN** the eval runner MUST remain compatible with in-process server startup and MUST NOT require external preflight success

### Requirement: Eval runner MUST fail fast on missing external control service
The eval runner SHALL avoid long-running episodes when the external server is requested but not visible from the Arena container.

#### Scenario: Command service missing
- **WHEN** the expected namespace-local `get_command` service is absent after the preflight timeout
- **THEN** the eval runner MUST return a non-zero code and record `end_reason=external_preflight_failed`

#### Scenario: Command service present
- **WHEN** the expected namespace-local `get_command` service is visible during preflight
- **THEN** the eval runner MUST continue to normal launch and post-processing behavior
