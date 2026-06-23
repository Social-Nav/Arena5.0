## ADDED Requirements

### Requirement: GRScenes run readiness uses strict gates

A GRScenes benchmark run MUST be considered ready only when strict task metrics,
strict social metrics, artifact validation, video artifacts, HuNav motion, and
InternNav control diagnostics are all present and consistent.

#### Scenario: Complete successful run
- **WHEN** a GRScenes run finishes and all strict gates pass
- **THEN** `artifact_validation.json` and aggregate output MUST report the run as
  benchmark-ready

#### Scenario: Dynamic scene is valid but task fails
- **WHEN** HuNav pedestrians move and videos are valid, but strict task success
  is false
- **THEN** the run MUST remain a benchmark failure with task failure reasons
  preserved

### Requirement: GRScenes failures are classified before dataset patches

The system MUST distinguish framework/instrumentation failures from real model
or scenario failures before changing dataset assets or maps.

#### Scenario: Static occupancy is detected
- **WHEN** a GRScenes run reports static occupancy collision
- **THEN** the next review step MUST classify whether the collision comes from
  map/asset registration, robot footprint configuration, or actual robot
  behavior near a static obstacle

### Requirement: Language instruction contract is explicit

The benchmark MUST make clear which part of the language instruction is being
evaluated by strict task metrics.

#### Scenario: Native scenario goal is the task contract
- **WHEN** a GRScenes episode has a recorded instruction and native scenario
  start/goal
- **THEN** strict task metrics MUST state that goal reaching is the evaluated
  instruction predicate and MUST record goal tolerance and source metadata

#### Scenario: BDDL predicates are present but unsupported
- **WHEN** a GRScenes episode includes BDDL or richer semantic predicates that
  are not evaluated
- **THEN** the run artifacts or docs MUST mark those predicates as unsupported or
  not-yet-scored rather than claiming full semantic instruction success

### Requirement: Representative reruns are aggregate-ready

Benchmark-quality validation MUST include a representative set of GRScenes
episodes, not only a single run or offline reprocessing.

#### Scenario: Representative eval set completes
- **WHEN** selected GRScenes episodes are rerun through Docker + Isaac + external
  InternNav
- **THEN** the outputs MUST include videos, strict metrics, validation artifacts,
  and aggregate summaries for each run
