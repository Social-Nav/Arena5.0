## ADDED Requirements

### Requirement: Persist decision traces
The system SHALL write structured trace records for InternNav model decisions during an eval run. Records SHALL cover completed model inferences and degraded decision events such as fallback commands or camera-gate stops. Each record SHALL include timing, observation metadata, goal geometry, instruction preview, normalized model action when available, converted command, action history tail, turn-sign correction state, and backend timing/error fields.

#### Scenario: Trace record written for a model inference
- **WHEN** the InternNav server completes a model inference during an eval episode
- **THEN** a JSONL trace record is appended containing the selected action, converted `linear_x`/`angular_z`, goal distance/yaw error, RGB/depth availability, and inference duration

#### Scenario: Trace records survive episode completion
- **WHEN** an eval episode finishes
- **THEN** the output directory contains the trace file and it can be parsed without relying on ROS topics still being active

#### Scenario: Fallback and gate decisions are traceable
- **WHEN** a model inference is still running or required camera inputs are gated
- **THEN** the trace includes a `fallback_command` or `camera_gate` record with the returned command and enough observation metadata to explain why the command was degraded

### Requirement: Visualize model decisions on ego observations
The system SHALL render InternNav decision diagnostics onto the existing debug visualization stream or an equivalent recorded diagnostic video. The visualization SHALL show the current selected action, converted velocity command, goal distance/yaw error, action history, and basic freshness indicators without obscuring the ego image.

#### Scenario: Debug overlay shows the selected action
- **WHEN** `ego_debug_overlay.mp4` is generated for an eval episode
- **THEN** the overlay includes a readable selected action label and corresponding converted command for the current or most recent inference

#### Scenario: Overlay indicates stale or missing inputs
- **WHEN** RGB, depth, camera info, goal, or instruction input is missing or stale at inference time
- **THEN** the overlay visibly marks the affected input as unavailable or stale

### Requirement: Summarize rotate-in-place and progress behavior
The system SHALL produce a diagnostic summary for each eval episode that reports trace record counts, event counts, action distribution, turn-vs-forward command ratio, stop ratio, trace-derived goal-distance trend, and rotate-heavy/low-progress conditions.

#### Scenario: Rotate-heavy episode is flagged
- **WHEN** an episode has sustained turn commands with low net progress toward the goal
- **THEN** the summary flags the episode as rotate-heavy and includes the supporting action/command/progress statistics

#### Scenario: Turn-sign correction state is summarized
- **WHEN** a run uses the discrete turn-sign correction control
- **THEN** the trace records, diagnostic summary, and run manifest expose whether inversion was requested and whether it resolved to enabled or disabled

### Requirement: Diagnose likely integration faults
The system SHALL report candidate root causes when observed traces match implemented integration-fault patterns, including action/yaw sign mismatch, stale observation reuse, and missing camera inputs. The trace and overlay diagnostics SHALL retain enough goal geometry and observation metadata to support manual investigation of broader camera orientation, goal-frame, or instruction-conditioning issues.

#### Scenario: Action sign mismatch candidate is reported
- **WHEN** selected turn actions consistently increase yaw error or move away from the goal heading
- **THEN** the diagnostic summary reports a possible left/right action mapping or yaw-sign mismatch

#### Scenario: Observation mismatch candidate is reported
- **WHEN** trace metadata shows repeated identical frames, missing depth/camera info, or inconsistent camera frame identifiers
- **THEN** the diagnostic summary reports a possible observation freshness or camera preprocessing issue

### Requirement: Provide scoped turn-sign correction control
The system SHALL expose a reproducible control for InternNav discrete turn sign mapping. The default `auto` mode SHALL enable the correction only for Isaac + Ai2_Bot2, while explicit `true` and `false` values SHALL force the behavior for A/B comparison.

#### Scenario: Isaac Ai2_Bot2 auto correction is enabled
- **WHEN** the eval runner is invoked with `--internnav-invert-discrete-turns auto`, `--sim isaac`, and `--robot Ai2_Bot2`
- **THEN** the runner enables discrete turn inversion for the backend and the manifest records the requested value, resolved boolean, and resolution source

#### Scenario: Forced A/B correction is recorded
- **WHEN** the eval runner is invoked with `--internnav-invert-discrete-turns true` or `false`
- **THEN** the forced value is passed to the backend and persisted in trace/action diagnostics for later comparison
