## 1. Strict Task Metrics

- [ ] 1.1 Add a pure-Python strict task metrics module that reads run manifest, native scenario start/goal, `start_goal.csv`, `odom.csv`, `cmd_vel.csv`, and base metrics.
- [ ] 1.2 Implement final-goal distance and tolerance checks using native scenario metadata as the authoritative source.
- [ ] 1.3 Implement start-goal consistency checks between native scenario metadata and recorder `start_goal.csv`.
- [ ] 1.4 Implement commanded-stuck interval detection from `cmd_vel.csv` and `odom.csv` with configurable thresholds.
- [ ] 1.5 Implement static obstacle occupancy checks from `map.yaml`, map image, odometry, and robot radius.
- [ ] 1.6 Write strict task metrics artifacts with `strict_task_success`, failure reasons, thresholds, source paths, and review intervals.

## 2. Strict Social Safety Metrics

- [ ] 2.1 Extend social metrics generation to include footprint-aware robot-human clearance using robot radius and configurable human radius.
- [ ] 2.2 Integrate dynamic-scene validity into strict social success rather than reporting it only as a side field.
- [ ] 2.3 Integrate static obstacle and commanded-stuck failures into strict social safety or validation readiness.
- [ ] 2.4 Preserve legacy `social_success` while adding `strict_social_success` and strict failure reasons.
- [ ] 2.5 Add video path and simulation-time review interval references for strict failures.

## 3. Validation And Aggregation

- [ ] 3.1 Update artifact validation to require strict task and strict social metrics for benchmark readiness.
- [ ] 3.2 Update aggregate reporting to show legacy and strict success rates separately.
- [ ] 3.3 Add false-positive diagnostics for runs where legacy success is true but strict success is false.
- [ ] 3.4 Update run manifests or postprocess command records to list strict metric artifact paths.

## 4. Tests

- [ ] 4.1 Add unit tests for goal-distance success and stale `start_goal.csv` detection.
- [ ] 4.2 Add unit tests for map occupancy collisions using tiny map fixtures.
- [ ] 4.3 Add unit tests for commanded-stuck interval detection.
- [ ] 4.4 Add unit tests for footprint-aware human collision and personal-space thresholds.
- [ ] 4.5 Add validation tests proving strict failures override legacy `GOAL_REACHED` and legacy `social_success=true`.

## 5. Verification

- [ ] 5.1 Run focused Python tests inside `arena-arena_jazzy_ws-arena-1`.
- [ ] 5.2 Reprocess existing GRScenes run directories and confirm strict metrics flag the known stuck/obstacle failures.
- [ ] 5.3 Run one full Docker + Isaac GRScenes eval and verify strict artifacts are produced automatically.
- [ ] 5.4 Update the Lark benchmark plan with definitions, current limitations, and the next validation run results.
