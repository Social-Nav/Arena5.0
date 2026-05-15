## Context

The current hospital_1 + Ai2_Bot2 + InternNav eval has validated the data path: Isaac camera topics, depth/camera info, InternNav subprocess inference, Nav2 `GetCommand`, odom, and four H.264 videos are produced. However, the latest run shows the robot primarily rotating in place. Existing `internnav/status` debug data captures only the latest model output and basic goal geometry; it is insufficient to determine whether the behavior comes from the model policy, action conversion, camera orientation/intrinsics, stale observations, goal-frame mismatch, or controller/kinematics effects.

## Goals / Non-Goals

**Goals:**
- Persist per-inference decision traces for every eval episode.
- Visualize the model decision on top of the current ego observation so video review can correlate observation and action.
- Produce a post-run diagnostic summary that highlights rotate-in-place, no-progress, stale-observation, and action-mapping anomalies.
- Use these diagnostics to guide targeted fixes for InternNav action conversion, goal/instruction conditioning, or observation preprocessing.

**Non-Goals:**
- Retrain or fine-tune InternNav.
- Replace Nav2 or the Isaac diff-drive controller.
- Declare policy quality from a single episode; the first target is observability and obvious integration bugs.

## Decisions

1. **Record per-inference JSONL traces in the InternNav server.**
   - Rationale: The server is the only component that sees both ROS observations and normalized model outputs before conversion to velocity commands.
   - Alternative: Infer traces from `cmd_vel.csv` and final `internnav_status.json`; rejected because it loses action sequences, raw outputs, observation freshness, and model timing.

2. **Overlay action diagnostics into the existing debug image stream.**
   - Rationale: `ego_debug_overlay.mp4` is already recorded and synchronized with eval videos. Annotating it avoids adding another video stream for the first iteration.
   - Alternative: Add a separate `action_debug.mp4`; possible later, but higher plumbing cost.

3. **Summarize behavior post-run from the decision trace.**
   - Rationale: A compact trace-only summary can flag action distribution, command ratio, goal-distance trend, camera freshness, and sign-mismatch patterns without depending on legacy recorder CSV availability.
   - Alternative: Summarize from trace + odom + cmd_vel immediately; deferred because the first iteration needs robust diagnostics even when CSV metrics are missing or generated after the runner exits.

4. **Classify possible integration failures before changing policy behavior.**
   - Rationale: Persistent rotate can be caused by multiple integration bugs: left/right action sign inversion, camera optical frame mismatch, stale or post-reset pre-goal observations, heading-only goal heuristic overriding model output, or invalid instruction/goal conditioning.

5. **Scope the discrete turn-sign correction to a reproducible eval control.**
   - Rationale: The observed Isaac + Ai2_Bot2 behavior is consistent with a left/right action sign mismatch, but the correction should be explicit and reversible for A/B validation.
   - Decision: `--internnav-invert-discrete-turns auto` resolves to enabled only for Isaac + Ai2_Bot2; `true` and `false` force the value and all modes record the state in the manifest and trace diagnostics.

## Risks / Trade-offs

- **Risk: Logging overhead slows inference.** → Keep JSONL compact and write one line per inference, not per control tick.
- **Risk: Overlay obscures ego observation.** → Draw text and small action glyphs on a semi-transparent header/footer only.
- **Risk: Single episode looks like model failure but is a valid search/turn behavior.** → Report action distribution and progress metrics rather than hard pass/fail from one run.
- **Risk: Debug fields diverge from real command conversion.** → Generate trace fields from the same normalized action and conversion path used to return `GetCommand`.
