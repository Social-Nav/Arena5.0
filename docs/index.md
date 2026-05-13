# Arena Benchmark Documentation

This site is the initial English documentation set for the Arena benchmark stack.

Arena is a ROS 2 benchmark framework for comparing navigation systems across:

- **Simulators**: Gazebo and Isaac Sim
- **Robots**: differential, omni-directional, and USD-based robots such as `Ai2_Bot2`
- **Planners**: Nav2 planners, RL planners, and the new InternNav evaluation flow
- **Worlds and scenarios**: map-based evaluation scenes, including `hospital_1`

## What this documentation covers

- How the benchmark stack is organized in this repository
- How to install the minimum dependencies for benchmark execution
- How suites, contests, task generators, planners, evaluation, and video outputs fit together
- How to run benchmark and InternNav evaluation jobs
- Where to find outputs, metrics, videos, and troubleshooting signals

## Repository areas you will touch most often

- `arena_bringup/`: launch entrypoints and evaluation orchestration
- `arena_simulation_setup/`: worlds, scenarios, Nav2 config, robot launch composition
- `task_generator/`: task reset, scenario/random obstacle logic, benchmark module
- `arena_evaluation/`: recorder, metrics, plots, output post-processing
- `arena_isaac/`: Isaac runtime and ROS bridge services
- `arena_robots/`: robot models, mappings, and model parameters
- `arena_vln_models/`: InternNav wrapper, visualization, and model adapter integration

## Benchmark block diagram

```text
+-------------------+        +--------------------------+
| Benchmark Config  |        |   World / Robot Assets   |
| contests + suites |        | maps, scenarios, USDs    |
+---------+---------+        +------------+-------------+
          |                               |
          v                               v
+--------------------------------------------------------+
|               arena_bringup / launch graph             |
| selects simulator, planner instance, robot namespace   |
+-------------------------+------------------------------+
                          |
                          v
+--------------------------------------------------------+
|                task_generator runtime                  |
| reset episodes, publish goals, scenario/task control   |
+----------------+-------------------+-------------------+
                 |                   |                   |
                 v                   v                   v
        +---------------+   +----------------+   +------------------+
        |   Nav2 stack  |   | Simulator      |   | arena_evaluation |
        | controller    |   | Gazebo / Isaac |   | metrics / video  |
        +-------+-------+   +--------+-------+   +------------------+
                |                    |
                | get_command        | RGB / depth / pose / odom
                v                    |
        +-------------------------------+ 
        | arena_vln_models wrapper      |
        | InternNavServer               |
        | - generic model-sim base      |
        | - InternNav-specific subclass |
        | - dual_vln as model instance  |
        +-------------------------------+
```

## Module relationships and interface design

- **`arena_bringup`** owns orchestration and reproducible eval entrypoints.
- **`task_generator`** owns reset semantics, scenario progression, and goal publication.
- **`arena_simulation_setup`** owns world composition, robot launch wiring, and Nav2 parameter assembly.
- **`arena_vln_models`** exposes the model-simulator wrapper layer.
  - A **generic base wrapper** handles ROS subscriptions, command service, status publishing, and visualization.
  - An **InternNav-specific subclass** defines the camera readiness behavior and model-family-specific runtime logic.
  - **`dual_vln` is treated as a model/planner instance**, not as the architectural name of the wrapper layer.
- **`arena_evaluation`** consumes runtime outputs and produces metrics, manifests, and H.264 videos.

## Interface boundaries

- **Task generator → Nav2 / wrapper**: goals, instructions, task reset, finished topics
- **Simulator → wrapper**: RGB, depth, camera info, odometry, pose
- **Wrapper → Nav2**: `get_command` service returning `Twist`
- **Wrapper → evaluation**: status topic, debug overlay images, model diagnostics
- **Evaluation → artifact directory**: run manifest, metrics, video index, encoded videos

## Recommended reading order

1. **Getting Started**
2. **Benchmark Overview**
3. **Architecture**
4. **Running Benchmarks**
5. **Outputs and Metrics**
6. **Troubleshooting**

## Scope of this first version

This is an initial documentation scaffold. It is intentionally focused on the benchmark pipeline and the recently stabilized Isaac + InternNav flow rather than trying to document every package in the repository at once.
