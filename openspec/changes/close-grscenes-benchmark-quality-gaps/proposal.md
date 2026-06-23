## Why

The strict metric work made GRScenes benchmark failures visible, but the latest
full Isaac run still fails strict task/social gates and exposes remaining
instrumentation gaps.  The benchmark is not quality-complete until each strict
failure can be traced to map/scenario/model behavior, debug video coverage is
not silently degraded, and multiple GRScenes episodes can be run and aggregated
with auditable evidence.

## What Changes

- Diagnose and either fix or explicitly classify the latest `grscenes_5` strict
  failures: goal timeout, commanded stuck, static occupancy, footprint near miss,
  and footprint collision.
- Add benchmark-quality review outputs that link strict failure reasons to
  simulation-time samples, video paths, and JSON evidence.
- Close the `ego_debug_overlay` fallback gap or make fallback/non-fallback
  coverage a first-class readiness criterion.
- Tighten the language-instruction contract by documenting and, where feasible,
  wiring BDDL/native scenario predicates into strict task metrics.
- Re-run a representative GRScenes set through the Docker + Isaac + external
  InternNav pipeline and aggregate strict success, social success, video quality,
  and failure taxonomy.
- Keep all evaluation and verification Isaac-only; no Gazebo fallback.

## Capabilities

### New Capabilities

- `benchmark-quality-evidence`: benchmark runs expose enough structured evidence
  for every strict failure to be audited from JSON, aggregate CSV, and video.
- `grscenes-eval-readiness`: GRScenes episodes have a repeatable readiness gate
  covering native scenario contracts, HuNav motion, Isaac video artifacts,
  InternNav control, strict metrics, and aggregate reporting.

### Modified Capabilities

None.

## Impact

- Affected packages: `arena_bringup`, `arena_evaluation`, GRScenes dataset docs,
  and benchmark documentation.
- Affected outputs: `vln_task_metrics.json`, `social_metrics.json`,
  `artifact_validation.json`, aggregate CSV/JSON, `video_index.json`, and
  optional frame-analysis outputs.
- Runtime constraints: full verification uses the existing three-container
  Docker topology with Isaac Sim and an external InternNav server only.
