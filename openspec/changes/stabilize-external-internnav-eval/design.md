## Context

Iter8 showed that the hospital social-navigation pipeline can work end-to-end: HuNav humans are present, InternNav produces commands, videos are written, metrics/social metrics validate, and the robot moves.  Iter11 showed a different failure mode: the robot did not move, InternNav trace/status artifacts were absent, and debug overlay video was missing even though the rest of the eval framework continued to completion.  Manual checks indicated that the Arena container did not always discover the external InternNav service until the DDS environment was made explicit.

The stable baseline already separates social pipeline health from task success.  This change focuses on making external-server infrastructure health deterministic and diagnosable before model behavior is evaluated.

## Goals / Non-Goals

**Goals:**

- Normalize ROS DDS environment for external InternNav eval subprocesses so Arena and InternNav containers use the same `ROS_DOMAIN_ID` and RMW implementation.
- Add preflight checks for the external `get_command` service and status topic before launching a long eval episode.
- Record preflight, DDS, and no-motion diagnostics into `run_manifest.yaml` and `artifact_validation.json`.
- Fail fast when external server mode is requested but the server is not discoverable.
- Preserve iter8 behavior and social validation semantics.

**Non-Goals:**

- Do not tune InternNav model policy, prompts, or action mapping in this change.
- Do not require `GOAL_REACHED` for social-nav readiness.
- Do not replace ROS 2 discovery with a new transport or dependency.
- Do not remove in-process/single-container InternNav execution.

## Decisions

1. **Normalize DDS env in `internnav_eval.py` rather than relying on container defaults.**
   - Rationale: iter11 reproduced a case where the Arena container shell had empty `ROS_DOMAIN_ID`/`RMW_IMPLEMENTATION`, while the InternNav container had `ROS_DOMAIN_ID=1` and `rmw_fastrtps_cpp`.
   - Alternative considered: update only Docker compose env.  Rejected because eval subprocesses should be self-describing and robust to interactive shell/container drift.

2. **Use a short ROS CLI preflight for external-server discovery.**
   - Rationale: `ros2 service list` and `ros2 topic list` are already available in the Arena container and directly verify what the eval process can discover.
   - Alternative considered: add a new rclpy preflight node.  Deferred because CLI preflight is simpler, easier to inspect in manifests, and sufficient for service/topic discovery.

3. **Make preflight strict only for `--internnav-external-server`.**
   - Rationale: in-process InternNav starts inside the launch graph, so external preflight would fail before the server exists.
   - Alternative considered: always preflight.  Rejected to preserve existing non-external eval behavior.

4. **Report no-motion as a diagnostic warning, not as a hard social failure.**
   - Rationale: social eval readiness validates infrastructure and human safety; no-motion can be caused by model behavior, server discovery, or controller issues.  It should be visible but not conflate with personal-space/collision metrics.

## Risks / Trade-offs

- [Risk] ROS CLI preflight can be slow or flaky during discovery startup. → Mitigation: use bounded retry timeouts and record raw outputs in the manifest.
- [Risk] Defaulting `ROS_DOMAIN_ID=1` may surprise users who intentionally run another domain. → Mitigation: only set defaults when env is absent; explicit env still wins.
- [Risk] Fail-fast can stop runs that previously limped to completion. → Mitigation: apply strict failure only when external server mode is explicitly requested.
- [Risk] No-motion warning can be noisy in intentionally stationary tests. → Mitigation: gate the warning on social eval / dual_vln / available odom metrics and keep it non-fatal.

## Migration Plan

1. Add OpenSpec docs and tasks.
2. Implement DDS env defaulting and manifest recording.
3. Implement external server preflight and wire it before launch.
4. Extend artifact validation diagnostics.
5. Run syntax tests and rerun validation against iter8 and iter11 artifacts.
6. If full containers are available, rerun one short external-server eval and verify preflight fails fast or produces model-control artifacts.

Rollback is a normal git revert of this change; previous iter8 stable semantics remain in the prior commit.
