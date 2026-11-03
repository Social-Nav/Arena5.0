## ADDED Requirements

### Requirement: One FollowPath executor
The system SHALL provide `ArenaModelExecutor` as the only component in `arena_vln_models` that creates a `nav2_msgs/action/FollowPath` action client or sends FollowPath goals.

#### Scenario: Valid prediction sends a path
- **WHEN** the adapter returns a validated fixed-odom path during a released episode
- **THEN** the executor SHALL send exactly one FollowPath goal and SHALL publish `model/status` `running`.

#### Scenario: STOP sends no goal
- **WHEN** the adapter returns a STOP prediction
- **THEN** the executor SHALL cancel any in-flight goal, send no new goal, and publish `model/status` `stop`.

#### Scenario: Adapter error
- **WHEN** the adapter raises timeout, protocol, capability or empty-output errors
- **THEN** the executor SHALL cancel any in-flight goal and publish `model/status` `error` with a reason.

#### Scenario: Goal replacement
- **WHEN** a new valid path arrives while a previous FollowPath goal is in flight
- **THEN** the executor SHALL cancel the previous goal before sending the new one.

#### Scenario: No velocity output
- **WHEN** `ArenaModelExecutor` is instantiated in any mode
- **THEN** it SHALL NOT create a `cmd_vel` publisher, SHALL NOT construct a `Twist`, and SHALL NOT instantiate a native velocity controller.

### Requirement: Episode identity and heartbeat
The executor SHALL bind `session_id` and `episode` from `eval_ready`, accept instructions only after release, and publish a periodic `model/status` heartbeat.

#### Scenario: New episode
- **WHEN** `eval_ready` reports a new `session_id` or episode
- **THEN** the executor SHALL reset the adapter, clear observations and cancel previous goals.

#### Scenario: STOP before release
- **WHEN** a model STOP arrives before the episode is released
- **THEN** the executor SHALL NOT terminate the new episode.

#### Scenario: Stale observation
- **WHEN** the RGB or odom observation is older than the configured maximum age
- **THEN** the executor SHALL publish `hold` and SHALL NOT call the model.

### Requirement: One model-independent runner
The runner SHALL select a model with `--model internnav|lightnav|navila`, merge a model overlay onto one base FollowPath profile, and preserve explicit `--profile` compatibility.

#### Scenario: Model selection
- **WHEN** `python3 -m arena_vln_models.runner plan run --model navila` is executed
- **THEN** the plan SHALL use the shared runtime/evaluation/controller/action fields and the NaVILA adapter/server fields.

#### Scenario: Profile and model disagree
- **WHEN** `--profile` and `--model` are both provided and the profile `model` differs from `--model`
- **THEN** the runner SHALL exit non-zero before producing a command.

#### Scenario: Unknown model
- **WHEN** `--model unknown` is provided
- **THEN** the runner SHALL exit non-zero before producing a command.

#### Scenario: Three-model plan equivalence
- **WHEN** plans are generated for all three models
- **THEN** only the whitelisted model-specific keys (`model`, `model_adapter`, profile identity metadata, output prefix, external command, model server, checkpoint, adapter) MAY differ; runtime, evaluation, controller, action, topic and video fields SHALL be identical.

### Requirement: Canonical resolved-profile identity
The runner SHALL serialize the merged base+overlay profile deterministically, hash it, and write a resolved snapshot. The plan SHALL record base path/hash, overlay path/hash, resolved hash/snapshot, and the resolved profile metadata.

#### Scenario: Resolved identity is reproducible
- **WHEN** the same base, overlay and model are resolved twice
- **THEN** the canonical resolved hash and snapshot bytes SHALL be identical.

#### Scenario: Source hashes are distinct
- **WHEN** a plan is generated from a base+overlay selection
- **THEN** `profile_sha256` SHALL be the canonical resolved hash, not merely one source YAML hash, and base/overlay hashes SHALL also be present.

### Requirement: Default Arena RPP control only
The default and only unified `control_mode` SHALL be `arena_follow_path`, using the existing Arena `regulated_pure_pursuit` controller and velocity smoother. Native direct velocity SHALL remain in the old isolated legacy entry and SHALL NOT instantiate `ArenaModelExecutor`.

#### Scenario: Default mode
- **WHEN** no control mode is configured
- **THEN** the executor SHALL send FollowPath goals and the velocity smoother SHALL be the final `cmd_vel` publisher.

#### Scenario: Native request is routed away
- **WHEN** `control_mode=native` is requested through the unified runner
- **THEN** the runner SHALL route to the legacy direct entry and SHALL NOT instantiate `ArenaModelExecutor`.

#### Scenario: Executor rejects native
- **WHEN** `ArenaModelExecutor` receives `control_mode=native`
- **THEN** it SHALL fail closed without creating a velocity publisher.

### Requirement: Legacy path-to-velocity removal
`PythonAdapterBackend` SHALL NOT convert trajectory or discrete-action output to velocity unless `legacy_native_cmd_vel=true` is explicitly configured in the legacy entry.

#### Scenario: Default backend receives a path
- **WHEN** the legacy backend receives `output_trajectory` without `legacy_native_cmd_vel=true`
- **THEN** it SHALL return a degraded `path_requires_executor` decision with zero velocity and a diagnostic.

#### Scenario: Explicit legacy compatibility
- **WHEN** `legacy_native_cmd_vel=true` is configured in the legacy entry
- **THEN** the legacy conversion SHALL remain available and its diagnostics SHALL be preserved.

### Requirement: Deprecated thin wrappers
The legacy `InternNavAdapter`, `InternVLARealworldHttpAdapter`, `load_internnav_adapter`, `load_internvla_realworld_http_adapter`, `PythonAdapterBackend` and `create_model_backend` entry points SHALL remain importable and SHALL delegate to the unified implementation where practical.

#### Scenario: Existing import remains valid
- **WHEN** existing code imports a legacy adapter or factory
- **THEN** the import SHALL succeed and the object SHALL expose the documented legacy method.

### Requirement: Model-specific shell entries converge
The `_meta/docker/features/{lightnav,navila}/main` entries SHALL become thin wrappers over `arena_vln_models.runner --model ...`.

#### Scenario: Shell entry selects the model
- **WHEN** a user runs the LightNav or NaVILA feature entry with runner arguments
- **THEN** it SHALL invoke the unified runner with the corresponding `--model` and SHALL preserve explicit `--profile` arguments.

### Requirement: Layered validation evidence
The change SHALL record offline validation and real-episode validation separately. NaVILA SHALL remain `not runtime-qualified` under legal hold.

#### Scenario: Offline gate
- **WHEN** fixture tests, import tests, isolated colcon and three-model dry-run plans pass
- **THEN** the change MAY be committed locally without a model runtime.

#### Scenario: Real three-Docker gate
- **WHEN** the shared GPU/Isaac lock is available
- **THEN** InternNav and LightNav SHALL each run one minimal episode through the unified entry with only `--model` changed and a new output prefix; artifacts and model traces SHALL be recorded.

#### Scenario: Shared harness blocker
- **WHEN** a shared Isaac/clock harness blocker prevents a real episode
- **THEN** the blocker SHALL be recorded at the harness layer and SHALL NOT be rewritten as an adapter PASS.

#### Scenario: NaVILA legal hold
- **WHEN** NaVILA is under legal hold
- **THEN** it SHALL be qualified by raw fixtures, plan/import and adapter contract only, and SHALL be reported as `not runtime-qualified`.
