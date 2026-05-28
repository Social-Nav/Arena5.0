# Checkpoint: 2026-05-25 internnav + HuNav + Isaac fixes

This document records the fix set currently in the workspace so the same file-level rollback can be applied later if a regression appears.

## Workspace baseline

- Arena repo HEAD: `d735a8fe99225af091760671b8c10827de99e3cc`
- `arena_isaac` submodule HEAD: `caf20efa6d091cf66a52f2288cd3765899b50476`

These fixes are currently uncommitted local edits on top of the SHAs above.

## Fixes included in this checkpoint

### Arena repo

1. `arena_bringup/arena_bringup/internnav_eval.py`
   - `require_human_states_ready` now depends on `args.human == "hunav"` instead of also requiring `social_eval`.
   - Purpose: prevent eval from starting before HuNav pedestrians are published.

2. `task_generator/task_generator/manager/robot_manager/robot_manager.py`
3. `task_generator/task_generator/tasks/task.py`
4. `utils/arena_rclpy_mixins/arena_rclpy_mixins/Time.py`
   - `/clock` subscriptions now use BEST_EFFORT + VOLATILE QoS.
   - Purpose: match Isaac Sim `/clock` QoS and avoid silent no-message stalls.

5. `task_generator/task_generator/simulators/sim/isaac_eval_simulator.py`
   - `_pause()` and `_unpause()` now call the superclass implementations.
   - Purpose: restore real sim pause/unpause behavior during eval.

6. `arena_vln_models/arena_vln_models/visualization.py`
   - Debug overlay text is simplified to only the current-frame action text.
   - Action visualization glyph/arrow remains; extra diagnostics/goal/metrics text is removed.

### `arena_isaac` submodule

1. `arena_isaac/isaac_utils/utils/path.py`
   - `world_path()` is now idempotent for already-normalized `/World/...` paths.
   - Purpose: avoid invalid `/World/World/...` prim paths.

2. `arena_isaac/arena_isaac/services/EditPrims.py`
   - `move_prim()` now uses the direct `geom.move()` signature without `physics_teleport`.
   - Top-down camera movement now first checks that the camera prim exists.
   - Purpose: prevent hangs on invalid paths and avoid camera move failures when the camera is absent.

3. `arena_isaac/arena_isaac/services/SpawnUrdf.py`
4. `arena_isaac/isaac_utils/utils/geom.py`
   - Removed the deferred articulation teleport queue path and reverted to direct `Articulation.set_world_poses()` handling.
   - `geom.move()` no longer accepts `physics_teleport`.

5. `arena_isaac/arena_isaac/services/SpawnUsdRobot.py`
   - Reverted articulation-root selection to the first valid root.
   - Removed duplicate articulation-root cleanup for Ai2_Bot2.
   - Isaac odom graph is enabled by default unless explicitly disabled with `ARENA_SPAWN_USD_ROBOT_ENABLE_ISAAC_ODOM_GRAPH=0`.

6. `arena_isaac/arena_isaac/run_isaacsim.py`
   - `self._running` is initialized to `False` again.

## User-visible regressions addressed

- Isaac eval hang where robot reset targeted `/World/World/...` and never moved.
- Isaac `/clock` subscriptions silently receiving no messages due to QoS mismatch.
- Eval readiness releasing before HuNav pedestrians existed.
- IsaacEvalSimulator pause/unpause hooks not actually pausing the simulator.
- Overly noisy debug overlay obscuring the current action.

## Verified result already observed before this checkpoint

- `hospital_1 + hunav` completed successfully with generated videos and a `GOAL_REACHED` episode.
- Example diagnostic artifact:
  - `/home/ubuntu/arena_jazzy_ws/outputs/hospital1_hunav_video_verify/20260525_hospital1_hunav_video_verify_v5_hospital_1_Ai2_Bot2_heuristic/internnav_diagnostic_summary.json`

## File-level rollback commands

### Revert Arena repo edits in this checkpoint

```bash
git -C /home/ubuntu/arena_jazzy_ws/src/Arena checkout d735a8fe99225af091760671b8c10827de99e3cc -- \
  arena_bringup/arena_bringup/internnav_eval.py \
  arena_vln_models/arena_vln_models/visualization.py \
  task_generator/task_generator/manager/robot_manager/robot_manager.py \
  task_generator/task_generator/simulators/sim/isaac_eval_simulator.py \
  task_generator/task_generator/tasks/task.py \
  utils/arena_rclpy_mixins/arena_rclpy_mixins/Time.py
```

### Revert `arena_isaac` submodule edits in this checkpoint

```bash
git -C /home/ubuntu/arena_jazzy_ws/src/Arena/arena_isaac checkout caf20efa6d091cf66a52f2288cd3765899b50476 -- \
  arena_isaac/arena_isaac/run_isaacsim.py \
  arena_isaac/arena_isaac/services/EditPrims.py \
  arena_isaac/arena_isaac/services/SpawnUrdf.py \
  arena_isaac/arena_isaac/services/SpawnUsdRobot.py \
  arena_isaac/isaac_utils/utils/geom.py \
  arena_isaac/isaac_utils/utils/path.py
```

## Notes

- Local-only generated paths currently visible in status are not part of this checkpoint intent:
  - `/home/ubuntu/arena_jazzy_ws/src/Arena/.trae/`
  - `/home/ubuntu/arena_jazzy_ws/src/Arena/arena_isaac/arena_isaac/arena_isaac.egg-info/`
