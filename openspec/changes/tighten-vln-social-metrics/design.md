## Context

The current GRScenes + InternNav pipeline writes legacy `metrics.csv`, social metrics, videos, and validation artifacts, but the success semantics are incomplete. `metrics.csv` reports `GOAL_REACHED` when the run is not timed out and LaserScan collision count is unavailable or below threshold. `social_metrics.json` currently checks HuNav point-distance thresholds and large odometry teleports, but it does not evaluate static obstacle occupancy, command-vs-motion stuck behavior, robot footprint overlap, or visual clipping evidence.

The benchmark needs two explicit metric families:

- VLN task metrics: Did the robot satisfy the episode instruction and navigation goal under the recorded scenario contract?
- Social/safety metrics: Did the robot behave safely around dynamic humans and static scene geometry while the dynamic scene was active?

All implementation and verification must stay on the Docker + Isaac path. Gazebo and host ROS are out of scope.

## Goals / Non-Goals

**Goals:**

- Preserve legacy artifacts for compatibility while introducing strict benchmark fields with clear pass/fail semantics.
- Bind task success to scenario goal tolerance, final distance, timeout, progress, control evidence, and stuck/obstacle failures.
- Bind social success to HuNav presence/motion, robot-human safety distances, dynamic-scene overlap, static-map collisions, and commanded-stuck intervals.
- Make failures auditable from JSON/CSV artifacts and easy to aggregate across GRScenes episodes.
- Keep the first implementation pure postprocessing where possible, using existing `odom.csv`, `cmd_vel.csv`, HuNav CSV, `start_goal.csv`, maps, manifests, and video indices.

**Non-Goals:**

- Do not implement language semantic grounding beyond the recorded episode's instruction-to-goal contract in the first pass.
- Do not require Isaac physics contact sensors before map/footprint checks are available.
- Do not use Gazebo or host ROS for validation.
- Do not make manual video review the primary pass/fail source, though video review hints should remain available.

## Decisions

1. Add strict fields instead of changing legacy fields in place.

   Legacy consumers may still expect `metrics.csv.result` and `social_metrics.social_success`. New aggregate reports should use `strict_task_success`, `strict_social_success`, and `benchmark_ready` fields. This avoids a breaking schema change while making the benchmark result less ambiguous.

2. Use the recorded scenario goal contract as the first VLN instruction-success proxy.

   GRScenes recorded episodes provide an instruction and a corresponding start/goal. The first strict metric should assert that the robot reaches the scenario goal within a configurable tolerance and does not fail safety constraints. Full natural-language predicate evaluation can be added later, but it is not required to fix the current false-positive success reports.

3. Use map occupancy plus robot footprint for static-obstacle checks.

   GRScenes maps already exist under `worlds/grscenes_<id>/map`. A postprocessor can transform odometry positions into map pixels using `map.yaml`, inflate occupied cells by the robot radius, and report occupied-footprint samples and intervals. This directly catches cases where the robot appears stuck in or on geometry, even when LaserScan is absent.

4. Add commanded-stuck checks using `cmd_vel.csv` and `odom.csv`.

   A robot with sustained command magnitude and negligible displacement should fail strict task readiness. The check should report intervals and total stuck time rather than only a boolean, because short startup pauses are acceptable while long blocked intervals are benchmark failures.

5. Keep human safety checks data-driven but footprint-aware.

   Current human checks use point-to-point distances and a `0.25m` collision radius. The strict version should incorporate robot radius and configurable human radius so that pass-through or overlap is not hidden by a too-small threshold. If visual mesh evidence disagrees with CSV positions, the validation report should flag the run for review.

6. Make aggregate reporting show separate success rates.

   Reports should distinguish `legacy_task_success_rate`, `strict_task_success_rate`, `strict_social_success_rate`, and `artifact_validation_pass_rate`. This prevents a run set from being reported as successful when only legacy fields passed.

## Risks / Trade-offs

- Map occupancy can be conservative or misaligned if `map.yaml` origin/resolution disagrees with the USD scene. Mitigation: report sample positions, map indices, and per-scene map metadata; validate against `sim_top_down.mp4` for initial rollout.
- Start and goal metadata may be stale. Mitigation: prefer native scenario `robots` start/goal, compare against `start_goal.csv`, and fail validation if they disagree beyond tolerance.
- Human mesh geometry can differ from HuNav point states. Mitigation: strict metrics should use configurable human radius and keep video review hints for suspected clipping.
- Short intentional pauses could be counted as stuck. Mitigation: require command magnitude, minimum duration, and low displacement over a sliding window; expose thresholds in output.
- Existing dashboards may still read legacy fields. Mitigation: keep legacy fields, add explicit deprecation notes in aggregate summaries, and update the Lark plan with the new interpretation.

## Migration Plan

1. Add pure-Python strict metric helpers and tests for goal distance, occupancy collision, stuck intervals, and footprint-aware human proximity.
2. Extend social metrics generation to write strict safety fields while preserving existing fields.
3. Extend artifact validation and aggregate reporting to use strict fields for benchmark readiness.
4. Re-run one GRScenes episode, compare strict failures against manual video observations, then re-run the five-scene set.
5. Update the Lark benchmark plan with the new metric definitions and the current evidence that legacy 100% was not benchmark-ready.
