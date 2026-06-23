## Context

The current strict-metrics pipeline correctly rejects a recent `grscenes_5`
full Isaac run even though legacy outputs can look successful.  That is the
right direction, but benchmark quality now depends on closing the gap between
"strict failure detected" and "failure is actionable and reproducible."  The
latest run shows four concrete classes of remaining work: task failure near the
goal, footprint-level pedestrian safety failures, debug overlay fallback, and a
still-loose connection between language/BDDL annotations and the strict goal
contract.

All runtime validation must stay inside the Docker + Isaac + external InternNav
topology.  Host-side ROS, Gazebo, local model servers in `arena-1`, and direct
pedestrian pose syncing are not acceptable validation or fallback paths.

## Goals / Non-Goals

**Goals:**

- Make strict failure evidence reviewable from run artifacts without manually
  reconstructing event times from raw CSVs.
- Determine whether the latest `grscenes_5` static occupancy and stuck failures
  are caused by real robot behavior, bad map/asset registration, or scenario
  metadata.
- Make debug overlay fallback visible in validation and either eliminate it or
  define it as a non-ready benchmark state.
- Clarify how native GRScenes language instructions and BDDL predicates feed
  strict task success.
- Re-run a small representative GRScenes set and aggregate strict metrics with
  video-backed evidence.

**Non-Goals:**

- Do not change the InternNav model policy to chase a passing score.
- Do not bypass HuNav or Isaac pedestrian animation with direct USD pose sync.
- Do not use Gazebo, host ROS, or metrics-only smoke tests as benchmark proof.
- Do not treat a single successful episode as aggregate benchmark quality.

## Decisions

1. **Classify before patching scenario failures.**

   The next implementation step is a deterministic offline review tool/report
   for strict failure samples: map cell coordinates, world XY, nearest static
   obstacle, commanded-stuck intervals, and video paths.  Only after that report
   points to metadata error should map/scenario assets be patched.

2. **Keep strict metrics as gates, not advisory fields.**

   `artifact_validation.json` and aggregate output will continue to require
   `strict_task_success` and `strict_social_success` for benchmark readiness.
   Legacy metrics remain compatibility fields and false-positive diagnostics.

3. **Treat debug overlay fallback as readiness degradation.**

   A fallback overlay is useful for manual review, but benchmark-quality
   instrumentation requires knowing whether the model debug stream was present.
   Runs with fallback remain valid failure evidence but should not be counted as
   full instrumentation coverage.

4. **Use scenario/native contract first, BDDL as an extension point.**

   The current minimal task contract is: language instruction maps to recorded
   scenario goal and tolerance.  BDDL predicates should be parsed and surfaced
   when available, but the first implementation can document unsupported
   predicates and fail/flag them rather than silently claiming semantic success.

5. **Verify by rerunning representative episodes, not only reprocessing.**

   Offline reprocessing is sufficient for unit fixes, but benchmark readiness
   requires at least one full Docker + Isaac + external InternNav run after
   changes, with videos and aggregate rows.

## Risks / Trade-offs

- **Risk: strict failures are real model failures, not framework bugs.** →
  Mitigation: preserve failures in aggregate output; only patch dataset/map/code
  when evidence shows metadata or instrumentation error.
- **Risk: BDDL integration expands scope.** → Mitigation: start with a predicate
  inventory and explicit unsupported-predicate reporting before implementing a
  full evaluator.
- **Risk: repeated Isaac runs are expensive and can leave GPU processes.** →
  Mitigation: use focused offline tools first, then run bounded representative
  evals with the existing Docker cleanup procedure.
- **Risk: debug overlay source topics vary by model mode.** → Mitigation: record
  source/fallback status per video and aggregate it before making it a hard fail
  for every run type.
