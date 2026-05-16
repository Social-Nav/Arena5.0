## 1. OpenSpec Planning

- [x] 1.1 Create proposal describing external InternNav DDS/preflight stabilization.
- [x] 1.2 Create design covering DDS normalization, preflight, diagnostics, risks, and rollback.
- [x] 1.3 Create specs for external readiness, eval runner context, and debug observability.

## 2. External Runtime Preflight

- [x] 2.1 Normalize default DDS environment in `internnav_eval.py` while preserving explicit user env.
- [x] 2.2 Record resolved DDS and external-server runtime context in `run_manifest.yaml`.
- [x] 2.3 Implement bounded external-server preflight for expected `get_command` service and status topic.
- [x] 2.4 Fail fast with `end_reason=external_preflight_failed` when strict external preflight fails.

## 3. Diagnostics and Validation

- [x] 3.1 Add manifest fields for preflight command outputs, missing checks, and timeout.
- [x] 3.2 Add artifact validation diagnostics for missing trace/status model-control loop.
- [x] 3.3 Add non-fatal stationary-robot warning based on social metrics path length.
- [x] 3.4 Preserve current social-navigation pass criteria independent of `GOAL_REACHED`.

## 4. Verification

- [x] 4.1 Run Python syntax checks for modified modules.
  - 2026-05-17: `python3 -m py_compile` passed for `internnav_eval.py` and `social_nav_validation.py`.
- [x] 4.2 Rerun artifact validation on iter8 and confirm `overall_pass=true`.
  - 2026-05-17: iter8 validation remains `overall_pass=true`.
- [x] 4.3 Rerun artifact validation on iter11 and confirm no-motion/model-control diagnostics are explicit.
  - 2026-05-17: iter11 validation reports stationary robot, missing model-control loop, and missing debug overlay warnings.
- [x] 4.4 Run or simulate external-server preflight success/failure paths without requiring a full Isaac episode.
  - 2026-05-17: helper tests cover success/failure discovery, and a no-server preflight run exits `2` with `end_reason=external_preflight_failed`.

## 5. Finalization

- [x] 5.1 Update task checklist with completed verification notes.
- [ ] 5.2 Commit implementation and OpenSpec artifacts.
