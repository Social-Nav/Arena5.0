## ADDED Requirements

### Requirement: Unified model adapter protocol
The system SHALL provide one `ModelAdapter` protocol shared by InternNav, NaVILA and LightNav. The protocol SHALL expose `name`, `capability`, `reset(session_id, episode)`, `infer(observation) -> ModelRawOutput`, and `to_prediction(raw, observation) -> ModelPrediction`.

#### Scenario: Adapter selection is deterministic
- **WHEN** a caller requests `internnav`, `navila` or `lightnav` from the registry
- **THEN** the factory SHALL return the corresponding adapter class and capability without importing ROS.

#### Scenario: Unknown adapter fails before launch
- **WHEN** a caller requests an unregistered model name
- **THEN** the factory SHALL raise `UnknownModelAdapterError` and the runner SHALL exit non-zero before producing a launch command.

#### Scenario: Capability mismatch fails before inference
- **WHEN** the observation lacks a required input for the selected model
- **THEN** the executor SHALL report `capability_mismatch` and SHALL NOT call the model transport or send a FollowPath goal.

### Requirement: Raw model output is preserved
The adapter SHALL retain the original model response and SHALL record a deterministic raw-response hash and a decoded `modality`. The adapter SHALL NOT convert a trajectory or path into `linear_x`/`angular_z`.

#### Scenario: Raw response identity
- **WHEN** any adapter decodes a raw response
- **THEN** the raw-output manifest SHALL contain a stable hash of the raw response, a short preview, the decoded `modality`, and the provenance `source`.

### Requirement: Provenance and modality are explicit
The adapter SHALL label the generated geometry as `source=native`, `source=synthetic` or `source=primitive`, and separately record `modality`.

- `native`: direct model geometry (InternNav `output_trajectory`, LightNav waypoints).
- `synthetic`: the InternNav wrapper first synthesised a trajectory from a symbolic action.
- `primitive`: the adapter expanded a symbolic action or text primitive into geometry.

#### Scenario: InternNav native trajectory
- **WHEN** InternNav returns `output_trajectory` without a synthesis marker
- **THEN** the prediction SHALL be `source=native`, `modality=trajectory`, and SHALL retain all finite non-duplicate local waypoints.

#### Scenario: InternNav synthetic symbolic trajectory
- **WHEN** the wrapper synthesises `output_trajectory` from a symbolic action and sets `trajectory_synthesized_from_discrete_action=true`
- **THEN** the prediction SHALL be `source=synthetic` and the trace SHALL record the synthesis reason.

#### Scenario: InternNav discrete primitive
- **WHEN** InternNav returns only `output_action`/`discrete_action`
- **THEN** the adapter SHALL expand the action ids to a local primitive path and SHALL label it `source=primitive`, `modality=primitive`.

#### Scenario: InternNav pixel-only output is unsupported
- **WHEN** InternNav returns only `output_pixel` and no trajectory or action
- **THEN** the adapter SHALL raise `CapabilityMismatchError` and the runner SHALL reject the pixel-only output mode before launch; it SHALL NOT guess depth or camera axes.

#### Scenario: NaVILA primitive expansion
- **WHEN** NaVILA returns `move forward 75 cm`, `turn left 15 degrees` or an affirmative STOP
- **THEN** the adapter SHALL produce the corresponding 5 cm-sampled forward path, one pure-rotation pose, or an empty STOP prediction, and a non-STOP text primitive SHALL be `source=primitive`, `modality=primitive`.

#### Scenario: LightNav full waypoint path
- **WHEN** LightNav returns a non-STOP `data.actions.actions` array
- **THEN** the adapter SHALL retain every finite three-column waypoint, project the cumulative ego path into fixed odom, and label it `source=native`, `modality=waypoints`.

### Requirement: Canonical path validation
Every adapter SHALL normalise to fixed-odom `(x, y, yaw)` metres/radians and validate coordinates, units, yaw, frame, stamp, continuity and finiteness before the executor sends a goal.

#### Scenario: Non-finite or malformed output
- **WHEN** a raw response contains NaN, infinity, a non-numeric value, a malformed point, or fewer than two coordinates
- **THEN** the adapter SHALL raise `ModelProtocolError` and the executor SHALL report `error` without sending a goal.

#### Scenario: Duplicate adjacent points are legal
- **WHEN** an InternNav native trajectory contains its origin and repeated adjacent poses
- **THEN** the adapter SHALL remove adjacent duplicates, preserve any pure rotation, and SHALL NOT reject the path solely because a zero-length segment exists.

#### Scenario: Non-STOP output has no executable geometry
- **WHEN** a non-STOP response has no waypoint, action or pixel goal after decoding
- **THEN** the adapter SHALL raise `EmptyModelOutputError`.

#### Scenario: Discontinuous path
- **WHEN** consecutive distinct waypoints exceed the configured maximum segment length
- **THEN** the adapter SHALL reject the path with a continuity error.

#### Scenario: Missing frame or stamp
- **WHEN** the observation has an empty frame id or a non-positive timestamp
- **THEN** the adapter SHALL reject the observation before inference.

#### Scenario: STOP
- **WHEN** the model declares STOP (InternNav action 0, NaVILA affirmative text, LightNav `data.stop=true`)
- **THEN** the prediction SHALL contain no path and SHALL be terminal.

### Requirement: Adapters never publish velocity
No adapter module SHALL import ROS, create a publisher, publish `cmd_vel`, construct a `Twist`, or call `send_goal_async`.

#### Scenario: Source invariant
- **WHEN** the adapter package source is scanned
- **THEN** it SHALL contain no `create_publisher`, `Twist`, `cmd_vel` publisher, `send_goal_async` or ROS import.
