# Run Current Benchmarks

This page is the minimal runbook for current Arena benchmark work. It assumes the
workspace is mounted at `/home/ubuntu/arena_jazzy_ws` on the host and
`/opt/arena_ws` inside the Arena containers.

## Preflight

Run these checks from the host. They do not touch the ROS graph.

```bash
cd /home/ubuntu/arena_jazzy_ws
nvidia-smi

docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

docker exec arena-arena_jazzy_ws-isaac-1 \
  ps -eo pid,ppid,stat,comm,args

docker exec arena-arena_jazzy_ws-arena-1 \
  ps -eo pid,ppid,stat,comm,args

docker exec arena-arena_jazzy_ws-internnav-1 \
  ps -eo pid,ppid,stat,comm,args
```

Look for old `run_isaacsim`, `task_generator_node`, Nav2, `internnav_eval`,
`dual_vln_eval`, `dual_vln_server`, or `internnav_server` processes. Clean only
confirmed old runs.

## Preferred One-Command Entry Point

The benchmark feature owns the three-container startup order, readiness checks,
InternNav server lifetime, evaluator placement, and versioned benchmark profile.
Inspect its effective configuration and plan before the first run, then run the
current strict default case:

```bash
cd /home/ubuntu/arena_jazzy_ws
src/Arena/_meta/docker/features/benchmark/main config
src/Arena/_meta/docker/features/benchmark/main doctor
src/Arena/_meta/docker/features/benchmark/main plan run
src/Arena/_meta/docker/features/benchmark/main run
```

The default case is `isaac_eval`, `grscenes_20_v1/default_2`, `Ai2_Bot2`, one
episode with a 300-second simulation timeout, external InternNav direct control,
social evaluation, and video capture.
The entry point is intentionally fixed to `isaac_eval`. Use `--case
WORLD/SCENARIO`, or the individual `--world`, `--scenario`, `--robot`, and
`--timeout` options, to change the core case. Strict runs
require `--episodes 1`, because current metrics and videos represent one episode
per run directory. Run multiple independent commands and aggregate their result
directories for repeated trials. Arguments after `--` are passed to
`internnav_eval`:

```bash
src/Arena/_meta/docker/features/benchmark/main run \
  --case grscenes_20_v1/default_4 \
  --timeout 300 \
  -- --output-prefix review/default_4
```

## Configuration Layers

`ros2 run` does not automatically load a default YAML for an executable. The
benchmark entry point therefore loads a dedicated orchestration profile before
calling `internnav_eval`:

```text
CLI/case override > machine-local paths/device > benchmark profile > legacy code defaults
```

The tracked default is
`arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml`. It contains
settings that must stay identical across benchmark machines: Isaac/HuNav mode,
planner and InternNav topology, model/server topics, external direct control,
DDS contract, timing, video, and artifact-related settings. `config` validates
the schema and prints its SHA-256 plus the effective core values without touching
Docker.

Case-specific values can be supplied with `--case`/named CLI arguments or a
benchmark-level scenario YAML. Host paths and proxy credentials stay in the
workspace `.env`; the selected GPU may use `ARENA_BENCHMARK_DEVICE` when it is
machine-specific. These values do not belong in a tracked benchmark profile.

To inspect the evaluator's post-profile values inside `arena-1` without starting
ROS nodes or Isaac Sim:

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
  cd /opt/arena_ws &&
  source /opt/ros/jazzy/setup.bash &&
  source install/setup.bash &&
  ros2 run arena_bringup internnav_eval --print-effective-config
'
```

To review or test another versioned profile explicitly:

```bash
src/Arena/_meta/docker/features/benchmark/main config \
  --profile /absolute/container-visible/path/to/profile.yaml

src/Arena/_meta/docker/features/benchmark/main plan run \
  --profile /absolute/container-visible/path/to/profile.yaml
```

Each run records the profile id, schema version, path, SHA-256, a copied snapshot,
and the resolved core parameters under `benchmark_profile` in
`run_manifest.yaml`.

## Advanced: Build Arena Packages

All build and ROS commands run inside `arena-1`.

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
  cd /opt/arena_ws &&
  source /opt/ros/jazzy/setup.bash &&
  colcon build --merge-install --symlink-install \
    --packages-select \
      arena_bringup arena_evaluation arena_simulation_setup \
      task_generator arena_vln_models dual_vln &&
  source install/setup.bash
'
```

## Advanced: Expanded InternNav Command

The profile-backed entry point can expose or run only the model side when
debugging. The real model still runs in `internnav-1`, not in `arena-1`:

```bash
cd /home/ubuntu/arena_jazzy_ws
src/Arena/_meta/docker/features/benchmark/main plan serve
src/Arena/_meta/docker/features/benchmark/main serve
```

`plan serve` prints the complete profile-derived server command when the exact
topic/model settings are needed for diagnosis. Stop it from another host shell
with `src/Arena/_meta/docker/features/benchmark/main stop`.

## Advanced: Expanded Eval Command

Use `internnav_eval` directly only while debugging the resolved low-level command
inside `arena-1`. The default profile supplies the cross-machine values, so the
equivalent manual command is short:

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
  cd /opt/arena_ws &&
  source /opt/ros/jazzy/setup.bash &&
  source install/setup.bash &&
  export ROS_DOMAIN_ID=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 &&
  ros2 run arena_bringup internnav_eval \
    --benchmark-profile \
      /opt/arena_ws/src/Arena/arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml \
    --world grscenes_20_v1 \
    --robot Ai2_Bot2 \
    --episodes 1 \
    --timeout 300 \
    --scenario-file default_2 \
    --internnav-device cuda:0 \
    --output-prefix manual_benchmark/grscenes_20_v1_default_2
'
```

Use `benchmark/main plan eval` to generate the current resolved container command
instead of copying this expansion into automation. The default profile enables
social evaluation, which produces `vln_task_metrics.json`, `social_metrics.json`,
and `artifact_validation.json`.

`--internnav-status-topic` should be the robot-scoped absolute topic shown above.
The runner also normalizes relative `internnav/status` to that topic, but the
absolute form is clearer in benchmark scripts.

## Postprocess An Existing Run

Every run writes `postprocess_commands.txt`. To re-run metrics manually, pass
the run's container path into `arena-1`:

```bash
CONTAINER_RUN_DIR=/opt/arena_ws/outputs/<prefix>/<run>

docker exec -e RUN_DIR="$CONTAINER_RUN_DIR" \
  arena-arena_jazzy_ws-arena-1 bash -lc '
  cd /opt/arena_ws &&
  source /opt/ros/jazzy/setup.bash &&
  source install/setup.bash &&
  ros2 run arena_evaluation metrics --dir "$RUN_DIR" &&
  ros2 run arena_evaluation vln_task_metrics --dir "$RUN_DIR" &&
  ros2 run arena_evaluation social_metrics --dir "$RUN_DIR" &&
  ros2 run arena_bringup social_nav_validation --dir "$RUN_DIR"; validation_rc=$?
  ros2 run arena_bringup benchmark_result --dir "$RUN_DIR"
  exit $validation_rc
'
```

## Build This Documentation

```bash
cd /home/ubuntu/arena_jazzy_ws/src/Arena
uv run --frozen --only-group docs mkdocs build --strict
```
