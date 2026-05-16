## ADDED Requirements

### Requirement: Social eval environment contract
The system SHALL provide an acceptance contract for InternNav social navigation runs that verifies the evaluated run used `world=hospital_1`, `robot=Ai2_Bot2`, `human=hunav`, and `tm_obstacles=scenario`.

#### Scenario: Required environment is present
- **WHEN** a validation report is generated for a run manifest with `world=hospital_1`, `robot=Ai2_Bot2`, `human=hunav`, and `tm_obstacles=scenario`
- **THEN** the environment contract check SHALL pass.

#### Scenario: Required environment is missing
- **WHEN** a validation report is generated for a run manifest that does not match one or more required social eval fields
- **THEN** the environment contract check SHALL fail and identify the mismatched fields.

### Requirement: Human entity presence validation
The system SHALL validate that social navigation runs spawned one or more humans and recorded continuous HuNav human-state messages.

#### Scenario: Humans are recorded
- **WHEN** `human_states.csv` contains non-empty human-state rows for at least one episode
- **THEN** the validation report SHALL record `humans_spawned=true` and include the observed human count and sample count.

#### Scenario: Humans are absent
- **WHEN** `human_states.csv` is missing, empty, or contains no agents
- **THEN** the validation report SHALL fail the human entity check.

### Requirement: Model and control diagnostics validation
The system SHALL validate that InternNav social eval runs include model/control diagnostics sufficient to determine whether the real backend produced actions and commands.

#### Scenario: Model diagnostics are present
- **WHEN** `internnav_trace.jsonl` contains at least one `model_result` event and `internnav_status.json` or `internnav_status_history.jsonl` confirms backend readiness
- **THEN** the validation report SHALL pass the model diagnostics check.

#### Scenario: Model diagnostics are missing
- **WHEN** trace, status, or model-result events are missing
- **THEN** the validation report SHALL fail the model diagnostics check and list the missing artifacts.

### Requirement: Video artifact validation
The system SHALL validate that social navigation runs produce non-empty H.264 MP4 videos for `ego_observation`, `ego_debug_overlay`, `sim_top_down`, and `map_top_down_follow`.

#### Scenario: Required videos are valid
- **WHEN** `video_index.json` lists all required videos and each has a positive frame count and expected codec metadata
- **THEN** the validation report SHALL pass the video artifact check.

#### Scenario: Debug overlay is absent
- **WHEN** `ego_debug_overlay.mp4` is missing or `debug_overlay_frames` is zero
- **THEN** the validation report SHALL fail the video artifact check.

### Requirement: Visual quality validation hints
The system SHALL provide machine-readable hints for manual or agent-based review of extracted video frames, including black-screen, synthetic-gradient, top-down visibility, and debug-overlay readability checks.

#### Scenario: Frame analysis exists
- **WHEN** frame analysis artifacts are present for required videos
- **THEN** the validation report SHALL include black-frame and near-static-frame indicators for each analyzed video.

#### Scenario: Frame analysis is missing
- **WHEN** no frame analysis artifacts are present
- **THEN** the validation report SHALL not fail solely for missing frame analysis but SHALL include a warning requiring visual review.

### Requirement: Acceptance report output
The system SHALL write `artifact_validation.json` to the run directory and include an overall social navigation readiness result.

#### Scenario: All acceptance checks pass
- **WHEN** all required environment, entity, model/control, video, and metrics checks pass
- **THEN** `artifact_validation.json` SHALL contain `overall_pass=true` and `social_nav_ready=true`.

#### Scenario: Any required check fails
- **WHEN** one or more required checks fail
- **THEN** `artifact_validation.json` SHALL contain `overall_pass=false`, `social_nav_ready=false`, and a list of failed checks.
