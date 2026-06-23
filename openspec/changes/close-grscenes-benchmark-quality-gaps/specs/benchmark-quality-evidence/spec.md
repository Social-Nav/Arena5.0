## ADDED Requirements

### Requirement: Strict failures include review evidence

Every strict task or social failure produced by a benchmark run MUST include
enough evidence to locate the event in JSON artifacts and videos.

#### Scenario: Static occupancy failure is reported
- **WHEN** `vln_task_metrics.json` reports `static_occupancy_collision`
- **THEN** the run artifacts MUST include the first collision sample, collision
  interval, robot world position, and a video path suitable for manual review

#### Scenario: Footprint social failure is reported
- **WHEN** `social_metrics.json` reports `footprint_human_collision` or
  `footprint_near_miss`
- **THEN** the run artifacts MUST include event samples with simulation time,
  robot position, human id, human position, point distance, and footprint
  clearance

### Requirement: Aggregate rows preserve failure evidence

The aggregate CLI MUST expose strict success, legacy success, readiness status,
failure taxonomy, and key evidence fields in each row.

#### Scenario: Legacy success disagrees with strict success
- **WHEN** a run has legacy task or social success but strict success is false
- **THEN** aggregate output MUST include the corresponding false-positive failure
  tag and preserve the strict failure reasons

#### Scenario: Debug overlay uses fallback
- **WHEN** a run's debug overlay is generated from fallback imagery
- **THEN** validation and aggregate output MUST report `debug_overlay_fallback`
  so instrumentation coverage is not overstated

### Requirement: Failure reports are video-reviewable

Benchmark tooling MUST provide a concise path from strict failure reason to
video review.

#### Scenario: Reviewer inspects a failed run
- **WHEN** a reviewer opens the generated report or aggregate row for a failed
  run
- **THEN** the reviewer MUST be able to identify the relevant run directory,
  video file, simulation time, and JSON evidence sample without parsing raw CSVs
