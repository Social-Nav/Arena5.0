# Scenario and Social Profile Configuration

Applies to `worlds/<world>/scenarios/<name>/scenario.yaml` and
`configs/nav2/profiles/<name>/profile.yaml`.

Chinese version: [场景与社交配置说明.md](场景与社交配置说明.md)

---

## 1. File structure

```yaml
dynamic:                  # pedestrian list
- name: hunav_1
  model: female_adult_business_02
  pose: [x, y, yaw_deg]
  behavior_tree: BTRegularNav.xml
  behavior:
    type: 1
    social_force_factor: 10.0
  desired_velocity: 1.05
  waypoints:
  - [x, y, heading_deg]
robot:                    # also accepted as robots: (list) or the start/goal form
  name: robot
  pose: [x, y, yaw_deg]
  social_attributes: neutral
  social_yielding: false
  waypoints:
  - [x, y, yaw_deg]
```

Angles and coordinates:

- The 3rd element of `pose` is a yaw in **degrees**, for pedestrians and the robot alike.
- The 3rd element of a **waypoint** is also an angle, but it is discarded. Under SFM a
  pedestrian's heading emerges from its velocity direction and is recomputed every tick, so a
  per-waypoint heading is meaningless.
- A pedestrian's `z` is ignored and fixed at 1.25 m.

Two further forms parse but are unused: `static:` (static obstacles, fields `name` / `model` /
`pose` / `scale`) and the legacy `obstacles:` container (which splits into `static` /
`interactive` / `dynamic`, the first two being merged). None of the 174 shipped scenarios uses
either. Note their `pose` 3rd element is still read as **radians**, inconsistent with
pedestrians and the robot; unify this before enabling static obstacles.

The `interactive` key in `configs/task_generator.yaml` and the benchmark suites belongs to the
random-task mode and does not read scenario files.

---

## 2. Pedestrian fields

| Field | Description |
|---|---|
| `name` | Unique string, conventionally `hunav_N`. |
| `model` | Appearance only. Valid values below. |
| `pose` | Spawn pose `[x, y, yaw_deg]`. |
| `behavior_tree` | Determines behavior, see §3. |
| `behavior` | Force gains and behavior-tree ports, see §4. |
| `desired_velocity` | Speed (m/s). |
| `velocity` | **No effect** — never read by the code. Can be omitted. |
| `waypoints` | Goal list, visited in order. |
| `radius` | Body radius, default 0.3. Written at the same level as `behavior:`. |
| `cyclic_goals` | Whether to loop over the waypoints, default false. |
| `goal_radius` | Arrival threshold, default 0.2. |

Available models (any other value is replaced at random): `female_adult_business_02`,
`female_adult_medical_01`, `female_adult_police_01..03`,
`male_adult_construction_01/02/03/05`, `male_adult_medical_01`, `male_adult_police_04`.

Fields left unset fall back to `arena_bringup/configs/hunav/default.yaml`.

---

## 3. Behavior

Behavior is determined by `behavior_tree`. `type` does not determine behavior, but **the two
must agree**, otherwise you get a pedestrian running one tree while its force logic belongs to
another.

| `type` | Name | Behavior tree | Behavior |
|---|---|---|---|
| 1 | REGULAR | `BTRegularNav.xml` | Walks its route, treats the robot as another pedestrian |
| 2 | IMPASSIVE | — | Treats the robot as a static obstacle, no social avoidance |
| 3 | SURPRISED | `BTSurprisedNav.xml` | Stops and turns to look once it notices the robot |
| 4 | SCARED | `BTScaredNav.xml` | Slows down and moves away from the robot |
| 5 | CURIOUS | `BTCuriousNav.xml` | Approaches the robot, stops at a set distance |
| 6 | THREATENING | `BTThreateningNav.xml` | Cuts in front of the robot to block it |

How the robot enters a pedestrian's force computation depends on whether the current tick is in
a special state (set by the behavior tree when it triggers):

| type | Special state active | Not active |
|---|---|---|
| 1 REGULAR | As a pedestrian in the social-force sum | As a pedestrian |
| 2 IMPASSIVE | As an **obstacle** | As a pedestrian |
| 3–6 | **Excluded from forces** | As a pedestrian |

So type 3–6 pedestrians still feel the robot's social force most of the time; the robot is only
excluded on the tick where the special behavior fires. Nor does impassive always treat the robot
as an obstacle. For these types the dominant mechanism is goal or speed modification, not forces.

Selection guidance: use regular to verify the robot can pass a cooperative pedestrian; impassive
for an unresponsive one; surprised for approach behavior against a stationary target;
threatening for worst-case or adversarial pedestrians.

Regular is also the most direct check of the robot-pose data path — the robot participates in
every tick, so a pedestrian walking a perfectly straight line means the path is broken.

---

## 4. Force gains and behavior-tree ports

| Field | Active when | Description |
|---|---|---|
| `social_force_factor` | type 1; scared | How strongly the pedestrian yields to the robot, default 10.0. At 10.0 the detour is clearly visible. |
| `goal_force_factor` | Always | Pull toward the next waypoint, default 5.0. |
| `obstacle_force_factor` | Always | Repulsion from obstacles, default 2.0. **Adjust this when pedestrians hug walls**, not `social_force_factor`. |
| `other_force_factor` | Always | Pedestrian-to-pedestrian force, default 20.0. |
| `dist` | Trees with a visibility check | Meaning differs per tree, see below. |
| `duration` | Same | Seconds spent in the special state per trigger. |
| `once` | Same | `true` fires once only; `false` fires on every re-entry. |
| `vel` | Same | Speed while in the special state (0 = stationary). |
| `configuration` | No effect | Keep at 0. |

`dist` / `duration` / `once` / `vel` have **no effect under regular** — that tree contains
neither a visibility nor a timing check, so writing them is only misleading.

### Meaning of `dist`

| Behavior tree | Meaning | Detection radius |
|---|---|---|
| Regular | Unused | No check |
| Surprised | Detection radius | `dist` |
| Scared | Detection radius | `dist` (inner check fixed at 3.0) |
| Curious | Stop distance | Fixed 10.0 |
| Threatening | Blocking-point lead distance | Fixed 4.5 |

Under threatening, `dist` is not a detection radius: below 0.8 the pedestrian collides with the
robot, above 2.5 it overshoots and no longer blocks.

### Visibility-check limitations (trees with a check only)

1. Uses centre-to-centre distance, not distance to the robot's path. On a wide detour the robot
   may drop out of the branch mid-way, appearing as the pedestrian losing interest and leaving.
2. Requires the robot within ±99.74° of the pedestrian's heading (shared by all pedestrians, not
   configurable). **No raytracing** — a robot behind a wall still counts as visible.
3. The spawn heading comes from `pose`, so a pedestrian spawned facing away must turn before the
   check can fire.

### Threatening characteristics

- The blocking point is recomputed every tick from the robot's current pose, so the pedestrian
  tracks the robot rather than aiming at a fixed intercept; it re-blocks when the robot detours,
  and returns to its original route as soon as the branch stops holding.
- During the chase the speed is overridden to a fixed 2.0 m/s, making the configured
  `desired_velocity` irrelevant.
- The upstream blocking-point formula has `sin`/`cos` swapped relative to the usual convention,
  so "directly in front of the robot" is actually rotated 90° and the pedestrian cuts in from the
  side. **Do not fix this locally** — doing so would silently diverge from every other
  threatening scenario.
- With `once: false` the pedestrian re-blocks indefinitely and the robot cannot get through.

---

## 5. Robot fields

| Field | Description |
|---|---|
| `name` | Must match the managed robot. |
| `pose` | Start `[x, y, yaw_deg]`. |
| `waypoints` | **Only the last** is the goal; intermediate points are not via-points. |
| `social_attributes` | `passive` \| `neutral` \| `aggressive`, see §6. Defaults to `neutral`. |
| `social_yielding` | Proactive-yielding pipeline toggle. |
| `behavior` | Free text, not read by the code. |

`social_yielding` precedence, highest first: launch argument → `robot.social_yielding` →
`false`. In a scenario file this field can only be declared inside the `robot:` block.

The launch argument is a tri-state string (`auto` / `true` / `false`), where `auto` means
unspecified. If an existing script passes `social_yielding:=false` to mean "use the default",
change it to `auto` or drop it, otherwise it overrides a `true` in the scenario.

The field is not actually per-robot (a single latched topic gates the pipeline). With several
robots the last one declaring it wins, and conflicting values raise a warning. The current
format supports a single `robot:` key, so this has no practical impact today.

---

## 6. Social profiles

`social_attributes` selects a parameter bundle, pushed at reset to controller_server and
global_costmap. It takes effect without a restart.

| Parameter | passive | neutral | aggressive | Description |
|---|---|---|---|---|
| `social_weight` | 700.0 | 550.0 | 400.0 | SFM social work; accounts for velocity and angle |
| `proxemics_weight` | 150.0 | 110.0 | 70.0 | Personal-space bubble strength |
| `proxemics_d0` | 1.0 | 0.8 | 0.6 | Bubble range, effective radius roughly `1.5*d0` |
| `proxemics_alpha` | 3.0 | 3.0 | 3.0 | Peak scaling; use `proxemics_weight` to change strength |
| `social_clear_distance` | 2.0 | 1.9 | 1.8 | Hard cutoff; beyond it the pedestrian is skipped entirely |
| `social_safety_distance` | 0.9 | 0.82 | 0.75 | Inside it the near-gain engages |
| `social_mid_gain` | 1.0 | 1.0 | 1.0 | |
| `social_near_gain` | 6.0 | 5.0 | 4.0 | |
| `max_linear_velocity` | 0.7 | 0.8 | 0.7 | vx **ceiling**, not a cruise speed |
| `desired_linear_vel` | 0.3 | 0.4 | 0.6 | The actual speed parameter |

The three differ only in the MPC critics and the cruise speed; the pedestrian cost bubbles
(`global_social_layer`) are identical verbatim. The global costmap has `obstacle_layer` disabled,
making this layer the planner's only pedestrian input — shrinking the bubbles would route the
global plan too close to pedestrians and leave the MPC permanently fighting its own path.

`max_linear_velocity` is only the optimizer's solution bound, not a speed target (the target is
`desired_linear_vel`), so raising it does not by itself make the robot faster. A separate clamp
in velocity_smoother (currently 0.8) is outside profile control, so values above it have no
effect.

**Selection guidance.** passive brakes early and often, making real interaction negotiation hard
to produce; aggressive largely drives its planned route. Choose neutral when the robot should
demonstrably react to pedestrians without becoming unable to make progress. Note its 0.4 cruise
leaves only a small differential against typical 0.3–0.4 pedestrian speeds, which matters when
designing overtaking scenarios (see §7).

Prefer keeping `social_clear_distance` large for early reaction while lowering the gains, rather
than the reverse.

### Modification constraints

- **Pedestrian-facing parameters only.** Never write `obstacle_weight` or `inflation_radius`;
  both are shared across all obstacles and would change how the robot treats walls.
- **All three profiles must define the same keys.** A switch only pushes keys present in the
  target profile, so a key unique to one profile retains its previous value.
- **A new profile directory requires a rebuild** (symlinks are per-file, not per-directory);
  editing existing files does not.
- **Cost scales roughly as weight² × scale⁴.** Change one parameter at a time and watch the
  `[MOTION-DIAG]` log.

---

## 7. Designing interaction scenarios

Interaction comes from speed differentials and geometry, not from any single parameter.

- **Overtaking** requires the robot behind the pedestrian and genuinely faster. aggressive (0.6)
  against a 0.30 pedestrian closes at 0.30 m/s; raise the pedestrian to 0.60 and it drops to
  zero, making an overtake impossible. passive (0.3) is slower than most pedestrians and cannot
  overtake at all.
- **Head-on** requires opposing headings in the same corridor with enough run-up on both sides.
- **Crossing** requires two paths meeting the same point at the same time; derive it from the
  respective speeds and distances.
- **Spawn spacing** must be at least `robot_radius + ped_radius` (0.55 m for jackal and the
  default pedestrian). Overlapping spawns make the result meaningless from tick 0. Check the
  occupancy grid before changing poses — a corridor that looks empty on paper may already lie
  inside the inflation layer.

---

## 8. Verifying that settings take effect

Do not rely on the RViz label text: the suffix is concatenated straight from `behavior.type` and
shows regardless of whether the logic ran. It only proves the config was parsed.

| Check | Command / location | Meaning |
|---|---|---|
| Robot pose reaching hunav | `ros2 topic echo <ns>/robot_states` | Position non-zero and changing |
| Special branch actually running | `agents[0].behavior.state` on `<ns>/human_states` | `== 1` means it ran this cycle |
| Pedestrian reacting to the robot | Blue arrow (`socialForce`) in `sfm_forces` | With one pedestrian the robot is the only contributor; the arrow grows on approach |
| Threatening entering the chase | Observe the speed | Clearly above `desired_velocity` (overridden to 2.0) |
| Robot-side speed and cost | `[MOTION-DIAG]` in the controller log | Commanded/measured speed, cruise target, ceiling, per-critic cost |

Editing a scenario or a profile requires no rebuild. Editing Isaac-side Python needs no rebuild
either, but the Isaac process must be restarted.

---

## 9. Related files

- `configs/nav2/controllers/social_mpc/controller_config.yaml`: MPC weights and limits
- `docs/social_mpc_internals.md`: derivations, history and topic topology
- `arena_bringup/configs/hunav/default.yaml`: pedestrian field fallbacks
- `hunav_agent_manager/behavior_trees/BT*.xml`: the behavior trees themselves
