# Architecture

## High-level execution graph

The benchmark stack is composed of the following layers:

1. **Launch orchestration**
2. **Simulator startup**
3. **Task generation and reset control**
4. **Robot-specific Nav2 / controller configuration**
5. **Planner or model inference loop**
6. **Data recording and metrics/video post-processing**

## Main packages and responsibilities

### `arena_bringup`

This package contains the top-level launch entrypoints and the new InternNav eval runner.

- `arena.launch.py` composes the global runtime
- `internnav_eval.py` is the specialized evaluation orchestrator for benchmark-like InternNav runs

### `arena_simulation_setup`

This package provides:

- world definitions
- scenarios
- shared environment tree utilities
- Nav2 configuration merging
- robot launch composition

It is where planners, controllers, worlds, and robot-specific params are wired together.

### `task_generator`

This package manages:

- scenario resets
- random or scenario-based obstacle spawning
- start/goal management
- benchmark suite progression
- simulator abstraction

It is the operational core that turns a static launch into repeated benchmark episodes.

### `arena_evaluation`

This package handles:

- topic recording
- CSV output for legacy metrics flow
- metrics generation
- plotting
- compatibility helpers for new output directory layouts

### `arena_isaac`

This package provides the Isaac-specific runtime:

- scene loading
- spawning walls, floors, pedestrians, and robots
- Isaac ROS bridge services
- data logging helpers

### `arena_robots`

This package contains:

- robot model params
- simulator mappings
- URDF or USD robot assets
- robot-specific sensor frame definitions

### `arena_vln_models`

This package contains the model-sim wrapper runtime pieces:

- backend abstractions
- shared observation / decision normalization
- debug visualization
- InternNav integration
- `internnav_server`

## Isaac + InternNav data path

For an Isaac InternNav run, the simplified path is:

1. Isaac publishes robot sensors
2. `internnav_server` subscribes to pose, goal, RGB, depth, and camera info
3. the configured backend produces a command or trajectory hint
4. the local planner consumes `get_command`
5. task generator advances the scenario
6. evaluation records outputs and closes when `finished` is observed

## Why the recent Isaac fixes matter

The stabilized Isaac path depends on three compatibility details:

1. camera subscriptions must accept `BEST_EFFORT`
2. Jazzy Nav2 plugin names must use `::`
3. Isaac must load `/opt/isaac_bridge_msgs` instead of stale host-built Python 3.11 message overlays

Without these three, the benchmark may launch but still fail in model startup, planner initialization, or video capture.
