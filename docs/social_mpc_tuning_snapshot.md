# social_mpc + control-chain tuning snapshot (2026-07-16)

Snapshot of the current social_mpc / nav2 control-chain tuning, the full change log
from this debugging session, and the upstream-original values for comparison / revert.

Robot: **Ai2_Bot2** (diff-drive, USD scale 0.8) · sim: Isaac · planner: smac_2d · controller: social_mpc wrapped in RotationShim.

---

## 1. Current social_mpc parameters (as running now)

File: `arena_simulation_setup/configs/nav2/controllers/social_mpc/controller_config.yaml`

```yaml
FollowPath:
  plugin: nav2_rotation_shim_controller::RotationShimController   # wraps social_mpc
  primary_controller: nav2_social_mpc_controller::SocialMPCController
  angular_dist_threshold: 1.75          # ~100°: sole in-place-turn trigger
  forward_sampling_distance: 0.5
  rotate_to_heading_angular_vel: 1.5
  max_angular_accel: 3.2
  simulate_ahead_time: 1.0
  rotate_to_goal_heading: false
  closed_loop: true
  transform_tolerance: 1.0
  trajectorizer:
    omnidirectional: false
    desired_linear_vel: 0.25
    lookahead_dist: 0.2
    max_angular_vel: 1.0
    rotate_to_heading_min_angle: 1.047   # 60°
    base_frame: ${frame:-}${robot_base_frame:-base_link}
    time_step: 0.05
    max_time: 1.5
  optimizer:
    linear_solver_type: DENSE_SCHUR
    param_tol: 1.0e-9 ; fn_tol: 1.0e-5 ; gradient_tol: 1.0e-8
    max_iterations: 50
    min_linear_velocity: 0.0             # reverse forbidden
    control_horizon: 24
    parameter_block_length: 6
    discretization: 1
    current_path_weight: 1.0
    current_cmds_weight: 0.3
    weights:
      distance_weight: 150.0             # "beeline to trajectory ENDPOINT" term
      angle_weight: 800.0                # MISNAMED: per-step path-tracking term (position, not heading)
      social_weight: 80.0
      velocity_weight: 20.0
      agent_angle_weight: 5.0
      velocity_feasibility_weight: 5.0
      goal_align_weight: 20.0
      obstacle_weight: 8.0
      social_clear_distance: 2.0 ; social_safety_distance: 0.95
      social_mid_gain: 1.5 ; social_near_gain: 12.0
      social_retreat_gain: 0.0 ; social_retreat_distance: 0.0
```

Related control-chain values (other files):

| where | param | value |
|---|---|---|
| nav2.yaml goal_checker | xy_goal_tolerance / yaw_goal_tolerance | 0.3 / 3.14 (ignore heading), stateful |
| nav2.yaml local+global costmap | robot_radius / footprint_padding | 0.25 / 0.05 (eff. inscribed 0.30) |
| nav2.yaml inflation (both) | cost_scaling_factor / inflation_radius | 3.0 / 0.7 |
| nav2.yaml behavior_server | local_frame/global_frame/robot_base_frame | `${frame}odom` / `${frame}odom` / `${frame}base_link` |
| model_params (Ai2_Bot2) | max_vel_x / max_vel_theta / scale | 0.4 / 1.5 / 0.8 |
| model_params velocity_smoother | max_velocity / min_velocity | [0.4,0,1.5] / [0.0,0,-1.5] (no reverse) |
| model_params velocity_smoother | max_accel / max_decel | [0.5,0,2.0] / [-0.5,0,-2.0] |
| defaults/model_params collision_monitor | SlowPolygon radius / ratio / enabled | 0.4 / 0.75 / **False (disabled)** |
| arena.launch.py | default global_planner | smac_2d |
| planners/smac_2d smoother | w_smooth / w_data | 0.3 / 0.2 |

---

## 2. Change log (this session)

### 2a. C++ code changes — need `arena build --packages-select <pkg>` + fresh launch

**`src/deps/nav2_social_mpc_controller`** (rebuilt):
- `path_trajectorizer.cpp` — **rewrote the look-ahead selection to monotonic forward pursuit.**
  Old logic scanned the path from the END backwards and grabbed the first pose within
  `lookahead_dist` → on U-turns / self-approaching paths it latched onto a LATER leg that
  merely passed close by → local_plan diverged from global_plan / robot cut across / stuck.
  New logic advances a progress index forward-only (closest point ahead → then lookahead).
- `social_mpc_controller.cpp` — **removed the hand-written in-place-turn override**
  (the `[STUCK-DEBUG] IN-PLACE TURN`, hardcoded 0.35 rad). In-place rotation is now owned
  solely by the RotationShim plugin. (Note: the MPC cost is position-only, so it cannot
  rotate in place by itself — see "known open issue" below.)

**`src/Arena/arena_isaac`** (submodule; graph rebuilt at robot spawn, fresh Isaac needed):
- `isaac_utils/graphs/odom.py` — added `IsaacComputeOdometry` and wired
  `linearVelocity`/`angularVelocity` into `ROS2PublishOdometry`. Previously odom.twist was
  always 0 (pose-only), starving the MPC of velocity feedback. **Verified sign consistent
  with cmd_vel.** (Frame body-vs-world not exhaustively verified — see open issues.)

### 2b. Control-chain config changes — live via symlink, no rebuild

`controller_config.yaml` (social_mpc):
| param | from | to | why |
|---|---|---|---|
| FollowPath.plugin | SocialMPCController | RotationShim wrapping it | dedicated in-place rotate for big turns |
| angular_dist_threshold | (new) | 1.75 (~100°) | only near-U-turns rotate in place |
| desired_linear_vel | 0.35 | 0.25 | slower → less corner-cutting |
| lookahead_dist | 0.3 | 0.2 | tighter pursuit reference |
| max_angular_vel | 1.5 | 1.0 | gentler normal turning |
| rotate_to_heading_min_angle | 0.5236 (30°) | 1.047 (60°) | consistency w/ shim |
| min_linear_velocity | -0.35 | 0.0 | forbid reverse (kill reverse+spin) |
| distance_weight | 600 | 150 | reduce endpoint-beeline (cutting) |
| angle_weight | 800 | 800 (net) | round-tripped 800→200→800; it is the per-step path-tracking term, keep high |

`nav2.yaml`:
- `recoveries_server` → **renamed `behavior_server`** (matches the Jazzy node) + added `local_frame`
  (was defaulting to unprefixed `odom` → recovery behaviors couldn't find the pose → robot froze).
- global costmap `robot_radius` 0.28 → 0.25 + added `footprint_padding: 0.05` (unified with local).
- inflation (both): `cost_scaling_factor` 5.0(local)/1.5(global) → 3.0; `inflation_radius`
  0.50(local)/0.6(global) → 0.55 → **0.7** (unified).

`defaults/model_params.yaml` — collision_monitor SlowPolygon: radius 0.8 → 0.4, slowdown_ratio
0.5 → 0.75, `enabled: True → False` (**collision_monitor gating disabled; now pass-through**).

`Ai2_Bot2/model_params.yaml` — velocity_smoother: max angular 1.3 → 1.5; angular accel 2.0 → 3.0 → **2.0**;
min linear -0.4 → **0.0** (no reverse).

`arena.launch.py` — default `global_planner`: navfn → **smac_2d**.

`planners/smac_2d/planner_config.yaml` — smoother `w_smooth` 0.1 → 0.3, `w_data` 0.4 → 0.2 (smoother paths).

**New debug tool**: `_meta/tools/motion_layers.py` — prints every velocity-layer command + global/local
plan divergence, to localize which layer causes spin/stuck/sway.

> Not mine: the diff also shows tiny edits in `planners/smac_hybrid`, `smac_state_lattice`, `theta_star`
> planner configs — those were changed outside this session (user/linter), noted for completeness.

### 2c. Container / environment fixes (image state, not repo code)

- **Fast-CDR ABI crash fix** (apt): upgraded `ros-jazzy-fastcdr` (2.2.5→2.2.7), `ros-jazzy-fastrtps`,
  and the `rmw-fastrtps` / `rosidl-typesupport-fastrtps` stack (Jan→Jun snapshot) so nav2_msgs
  typesupport loads (`undefined symbol …Cdr9serializeEPc`) and node creation stops throwing
  `BadParamException`. ⚠️ **Persist with `arena feature docker commit`** or it is lost on container recreate.

---

## 3. Upstream original social_mpc parameters (for comparison / revert)

### 3a. Code defaults (`declare_parameter_if_not_declared`, used when unset)

```
trajectorizer: desired_linear_vel 0.4 (traj) / 0.5 (ctrl)  lookahead_dist 0.4  max_angular_vel 1.0
               rotate_to_heading_min_angle π/4 (0.785)  time_step 0.05  max_time 3.0  transform_tolerance 0.1
optimizer:     control_horizon 5  parameter_block_length 5  max_iterations 100
               current_path_weight 1.0  current_cmds_weight 1.0  min_linear_velocity -0.25
weights:       distance 3.0  angle 0.0  social 1.0  velocity 0.5  agent_angle 0.5  goal_align 0.0
               obstacle 0.0  proxemics 90.0  social_mid_gain 1.0  social_near_gain 6.0
               social_safety_distance 0.9  social_retreat_gain 8.0  social_retreat_distance 2.0
misc:          fov_angle π/4  omnidirectional false
```

### 3b. Upstream example (`src/deps/nav2_social_mpc_controller/params/params.yaml`)

```yaml
trajectorizer: {desired_linear_vel: 0.6, lookahead_dist: 1.0, max_angular_vel: 1.4,
                transform_tolerance: 0.3, time_step: 0.05, max_time: 2.0}
optimizer: {max_iterations: 40, control_horizon: 20, parameter_block_length: 4, discretization: 2,
            current_path_weight: 1.0, current_cmds_weight: 0.5}
weights: {distance: 50.0, social: 700.0, velocity: 8.0, angle: 180.0, agent_angle: 0.0,
          velocity_feasibility: 5.0, goal_align: 8.0, obstacle: 0.2}
goal_checker: {xy_goal_tolerance: 0.25, yaw_goal_tolerance: 0.25}
```
(Other upstream reference files: `params/obst_only_parameters_in_benchmark.yaml`,
`params/soc_work_obst_parameters_in_benchmark.yaml`.)

### 3c. Key deviations: current (arena) vs upstream

| param | code default | params.yaml | current |
|---|---|---|---|
| desired_linear_vel | 0.4 | 0.6 | **0.25** |
| lookahead_dist | 0.4 | 1.0 | **0.2** |
| max_angular_vel | 1.0 | 1.4 | 1.0 |
| max_time | 3.0 | 2.0 | 1.5 |
| control_horizon | 5 | 20 | **24** |
| min_linear_velocity | -0.25 | (−0.25) | **0.0** |
| distance_weight | 3.0 | 50 | **150** |
| angle_weight | 0.0 | 180 | **800** |
| social_weight | 1.0 | 700 | **80** |
| velocity_weight | 0.5 | 8 | 20 |
| goal_align_weight | 0.0 | 8 | 20 |
| obstacle_weight | 0.0 | 0.2 | **8** |
| current_cmds_weight | 1.0 | 0.5 | 0.3 |

Takeaway: arena's weights are far from both upstream reference points (esp. distance/angle much
higher, social much lower than the upstream example). If tuning gets stuck, the upstream
`params.yaml` column is a sane reset point to A/B against.

---

## 4. Known open issue (root cause not yet fixed)

The MPC's path-follow cost is **position-only** (4th-power distance to reference points). When the
trajectorizer emits an in-place-rotate reference (`vx=0`, `wz≠0`) the position cost has no gradient
to drive `wz` → the optimizer outputs `wz≈0` → the robot dithers/stalls on turns that fall below
`angular_dist_threshold`. Candidate real fixes: (a) lower `angular_dist_threshold` so the RotationShim
handles those rotations; (b) add a heading-tracking cost term to the optimizer (C++). Use
`_meta/tools/motion_layers.py` to confirm per-layer before changing more.

---

## 5. Session 2 update (2026-07-17) — root causes + fixes

### 5a. RotationShim / turning (controller_config.yaml, config only, no rebuild)
| param | value | why |
|---|---|---|
| angular_dist_threshold | 1.5708 (90°) | ENGAGE: sole "rotate-first" threshold |
| angular_disengage_threshold | 0.35 (~20°) | DISENGAGE: hand back to MPC within 20° (was 0.15/9°) |
| rotate_to_heading_angular_vel | 0.8 | lowered from 1.5: overshoot≈ω²/(2·decel), cuts "over-rotate + hard brake" |
| trajectorizer.rotate_to_heading_min_angle | 3.14 | **DISABLED** trajectorizer's in-place-rotate branch → MPC always gets a trackable arc reference (kills dithering); RotationShim owns all in-place rotation |
| optimizer weights | = robotics-upo master (distance 50 / social 700 / velocity 8 / angle 180 / agent_angle 0 / vel_feas 5 / goal_align 8 / obstacle 0.2) | reverted to upstream |
| optimizer.min_linear_velocity | 0.0 | no reverse |

### 5b. Costmap (nav2.yaml, config only)
- local_costmap: `update_frequency` 20→10, window `width/height` 10→**5 m**, `resolution` 0.1→**0.05** (cell count unchanged 100×100), `inflation_radius` 0.7→**0.55**.
- global+local `inflation_radius` 0.55; robot_radius 0.25 + padding 0.05 (inscribed 0.30).
- controller_server progress_checker `required_movement_radius` 0.5→0.25.

### 5c. C++ fixes (need rebuild in container)
- **retreat cost REMOVED** from social_mpc optimizer + SocialWorkCost (`social_retreat_gain/distance` deleted, `computeRetreatCost` gone). `arena build --packages-select nav2_social_mpc_controller`.
- **social_layer empty-people fix** — see memory [[arena-social-layer-empty-people]]; `arena build --packages-select nav2_social_costmap_plugin`.
- Leftover `[STUCK-DEBUG]` RCLCPP prints still in social_mpc_controller.cpp:170/283 + optimizer.cpp:403 (NOT yet removed).

### 5d. Reset loop + task_generator (see memory [[arena-reset-loop-rviz-gui]])
- RViz `task_generator_gui` panel self-triggered reset via setParams→reset→getParams→Qt signal loop. Fixed (reset now button-only via `doResetTask()`); `arena build --packages-select task_generator_gui`.
- task_generator B1 (auto-reset deadlock) + B2 (timeout baseline captured while paused) fixed in node.py / tasks/robots/__init__.py (Python, no rebuild). Only matter when `auto_reset=true` (default False).

### 5e. Open (not fixed) — TF/clock lag
- "Extrapolation into the future (map→base_link)", loop-rate inf/30Hz, "Simulation time did not advance". Root = Isaac `/tf` relay lags `/clock` + clock freeze/jump on pause/unpause. See memory [[arena-tf-clock-lag]]. Proposed: forward-bias re-stamp in `arena_isaac/isaac_utils/graphs/tf.py`.

### 5f. New debug tool
- `_meta/tools/motion_layers.py` — per-layer cmd_vel + G/L plan divergence + `/rosout` nav-event timeline (catches costmap-timeout→abort→spin).
