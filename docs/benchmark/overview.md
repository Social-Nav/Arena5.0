# Benchmark Overview

## Core benchmark concepts

Arena separates benchmarking into a few reusable pieces:

- **Contest**: a list of planner contestants to compare
- **Suite**: a list of benchmark stages (map, robot, episodes, task mode, seed, timeout)
- **Task generator**: resets tasks, spawns robots/obstacles, and applies scenario or random policies
- **Simulator backend**: Gazebo or Isaac Sim
- **Evaluation**: data recording, metrics generation, plots, and now H.264 video export for InternNav runs

## Benchmark configuration layout

The classical benchmark configuration lives in:

- `arena_bringup/configs/benchmark/config.yaml`
- `arena_bringup/configs/benchmark/contests/`
- `arena_bringup/configs/benchmark/suites/`

The top-level config selects:

- which contest file to use
- which suite file to use
- which simulator backend to run

## Contest files

Contest files define the compared navigation systems. A contestant typically includes:

- `name`
- `local_planner`
- `inter_planner`
- optionally `agent_name` for learned policies

Example use cases:

- compare `teb` vs `dwb`
- compare RL agents against Nav2 baselines
- compare a new local controller under the same suite and simulator

## Suite files

Suite files define the benchmark workload. A stage usually specifies:

- map/world
- robot
- episode count
- task mode (`scenario`, `random`, etc.)
- obstacle generation or scenario file config

This makes suites reusable across multiple contests.

## InternNav evaluation as a benchmark path

The new InternNav evaluation path is slightly different from the legacy CSV-only benchmark flow:

- it launches the full stack from `internnav_eval.py`
- it records a run manifest and video index
- it can export H.264 videos
- it supports an InternNav-backed real model adapter
- it can run on Isaac Sim with robot camera topics and debug overlays

So you can think of InternNav evaluation as a specialized benchmark entrypoint built on top of the same world, robot, Nav2, and task generator foundations.
