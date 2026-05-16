## Context

Arena already has a working InternNav eval runner, a CSV/rosbag data recorder, legacy navigation metrics, HuNav scenario data for `hospital_1`, and video recording with `video_index.json`. Recent validation showed that an eval can still return success and generate ego/top-down videos while recording zero HuNav agents, zero `human_states` messages, no debug overlay, no InternNav trace, no metrics, and incomplete command/sensor streams. This is not sufficient for a social navigation benchmark.

The implementation spans four subsystems:

- `task_generator`: spawns and updates HuNav dynamic obstacles.
- `arena_evaluation`: records runtime topics and computes offline metrics.
- `arena_bringup`: runs InternNav eval, starts video/status helpers, runs post-processing, and writes manifests.
- agent instructions: provide a repeatable output review checklist for humans or subagents.

Key constraints:

- Existing non-social eval flows and legacy metrics must keep working.
- Three-container external InternNav mode must remain supported.
- Validation must be machine-readable and explicit: missing social-navigation prerequisites should be reported as failed checks rather than hidden in logs.
- The first implementation should avoid adding heavy dependencies and should use existing CSV/JSON/video artifacts.

## Goals / Non-Goals

**Goals:**

- Make `hospital_1 + Ai2_Bot2 + hunav + InternNav` social eval runs explicitly request scenario humans.
- Publish and record HuNav human states continuously enough for metrics.
- Produce social metrics from robot odom and HuNav human states.
- Produce `artifact_validation.json` with pass/fail checks for environment/entities, model/control, videos, and metrics.
- Keep validation usable both after a full eval run and as a standalone checker over an existing output directory.
- Add an acceptance-agent instruction file that standardizes video-frame and metrics review.

**Non-Goals:**

- Do not redesign the whole task-generator scenario model.
- Do not replace legacy `metrics.csv` output.
- Do not require real-time perception of humans from RGB; metrics may use simulator state as ground truth.
- Do not require full social-force quality scoring in the first pass beyond the requested social metrics.

## Decisions

### Decision 1: Use HuNav `human_states.csv` as the social metrics source

Rationale: `arena_evaluation` already subscribes to `{namespace}/human_states` as `hunav_msgs/Agents`, and HuNav can provide simulator-ground-truth human state. Recording this as structured CSV keeps metrics independent from rosbag deserialization and avoids new dependencies.

Alternatives considered:

- Read rosbag directly: more complete, but heavier and harder to run outside ROS environments.
- Use video perception to infer humans: not reliable for benchmark ground truth and out of scope.

### Decision 2: Add HuNav social metrics as a JSON artifact and optionally mirror selected fields into metrics flow

Rationale: Existing `metrics.csv` has a legacy schema with list-like values and navigation-only assumptions. A separate `social_metrics.json` can include structured thresholds, counts, and acceptance fields without breaking existing consumers.

Alternatives considered:

- Extend `metrics.csv` only: difficult for nested per-episode social metrics and likely to break downstream parsing.
- Replace metrics pipeline: too invasive.

### Decision 3: Treat acceptance validation as post-processing in `arena_bringup`

Rationale: `internnav_eval.py` already knows output paths, video topics, trace/status paths, and metrics command return status. Adding a standalone validator module callable from the eval runner gives one authoritative report while remaining reusable for existing runs.

Alternatives considered:

- Put all validation in `metrics.py`: metrics should compute metrics, not validate video/trace artifacts.
- Rely on docs/manual checks: not machine-readable and caused false confidence previously.

### Decision 4: Make scenario obstacle mode explicit in eval CLI

Rationale: Social eval acceptance requires `tm_obstacles=scenario`. Relying on extra launch args makes this easy to forget and hard to record consistently in the manifest.

Alternatives considered:

- Change defaults globally to scenario: could break non-social smoke tests and random-obstacle workflows.
- Only document the extra launch arg: insufficient for reproducibility.

### Decision 5: Publish HuNav human states from the HuNav simulator update loop

Rationale: `HunavHumanSimulator` already holds `response.updated_agents` at 10 Hz. Publishing it to `human_states` aligns with existing evaluation subscriptions and RViz expectations.

Alternatives considered:

- Add a separate bridge node: additional process and lifecycle complexity.
- Record `arena_peds` instead: loses HuNav-specific fields and diverges from existing `human_states` expectation.

## Risks / Trade-offs

- [Risk] Scenario humans may still fail to spawn due to scenario file issues or HuNav service failures. → Mitigation: `artifact_validation.json` checks `humans_spawned`, `human_states_message_count`, and non-empty CSV rows.
- [Risk] Existing hospital_1 scenario uses `waypoint` singular while code expects `waypoints`. → Mitigation: add compatibility in HuNav conversion and prefer scenario files containing both `robots` and `dynamic` such as `normal`.
- [Risk] External InternNav server cannot write trace into eval output directory because of container path mismatch. → Mitigation: validation reports trace missing explicitly; generated launch contract documents expected trace/debug/status settings.
- [Risk] Debug overlay may be absent if external server was started without visualization. → Mitigation: validation fails `debug_overlay_frames` when visualization/video acceptance is required.
- [Risk] Social metrics computed from sparse recording can undercount short interactions. → Mitigation: include sample counts and duration in `social_metrics.json`, and validate continuity rather than treating sparse data as success.
