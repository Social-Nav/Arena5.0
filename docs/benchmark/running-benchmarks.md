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
InternNav server lifetime, and evaluator placement. Inspect its plan before the
first run, then run the current strict default case:

```bash
cd /home/ubuntu/arena_jazzy_ws
src/Arena/_meta/docker/features/benchmark/main doctor
src/Arena/_meta/docker/features/benchmark/main plan run
src/Arena/_meta/docker/features/benchmark/main run
```

The default case is `isaac_eval`, `grscenes_20_v1/default_2`, `Ai2_Bot2`, one
episode with a 300-second simulation timeout, external InternNav direct control,
social evaluation, and video capture.
The entry point is intentionally fixed to `isaac_eval`; use `--world`,
`--scenario`, `--robot`, and `--timeout` to change the core case. Strict runs
require `--episodes 1`, because current metrics and videos represent one episode
per run directory. Run multiple independent commands and aggregate their result
directories for repeated trials. Arguments after `--` are passed to
`internnav_eval`:

```bash
src/Arena/_meta/docker/features/benchmark/main run \
  --world grscenes_20_v1 \
  --scenario default_4 \
  --timeout 300 \
  -- --output-prefix review/default_4
```

## Build Arena Packages

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

## Start InternNav

Start the real model from the host through the Docker feature script. The model
runs in `internnav-1`, not in `arena-1`.

```bash
cd /home/ubuntu/arena_jazzy_ws
HOST_ARENA_WS_DIR=/home/ubuntu/arena_jazzy_ws \
ARENA_PROJECT_NAME=arena-arena_jazzy_ws \
ARENA_IMAGE=arena:dev \
ROS_DOMAIN_ID=1 \
RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
src/Arena/_meta/docker/features/internnav/main launch \
  --robot Ai2_Bot2 \
  --mode internnav \
  --model-path /opt/arena_ws/deps/models/InternVLA-N1-DualVLN \
  --device cuda:0 \
  --rgb-topic head_camera/image \
  --depth-topic head_camera/depth \
  --camera-info-topic head_camera/camera_info \
  --odom-topic odom \
  --raw-cmd-vel-topic internnav/raw_cmd_vel \
  --status-topic /task_generator_node/Ai2_Bot2/internnav/status \
  --visualization-topic internnav/debug_image \
  --action-visualization-topic internnav/action_image \
  --model-output-topic internnav/model_output \
  --model-output-policy trajectory \
  --enable-visualization \
  --planning-rate-hz 3.3333333333 \
  --inference-timeout-sec 120.0
```

## Run A Benchmark Episode

Use `internnav_eval` from `arena-1`. Change `world`, `scenario-file`,
instruction, and output prefix for the specific case.

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
  cd /opt/arena_ws &&
  source /opt/ros/jazzy/setup.bash &&
  source install/setup.bash &&
  export ROS_DOMAIN_ID=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 &&
  ros2 run arena_bringup internnav_eval \
    --sim isaac_eval \
    --human hunav \
    --world grscenes_20_v1 \
    --robot Ai2_Bot2 \
    --local-planner dual_vln \
    --inter-planner navigate_to_pose_w_replanning_and_recovery \
    --global-planner navfn \
    --episodes 1 \
    --timeout 300 \
    --timeout-wall-factor 5.0 \
    --timeout-wall-sec 1800 \
    --tm-robots scenario \
    --tm-obstacles scenario \
    --scenario-file default_2 \
    --social-eval \
    --headless 2 \
    --log-level info \
    --vln-instruction "Walk straight forward down the aisle." \
    --internnav-mode internnav \
    --internnav-model-path /opt/arena_ws/deps/models/InternVLA-N1-DualVLN \
    --internnav-device cuda:0 \
    --internnav-inference-rate-hz 3.3333333333 \
    --internnav-inference-timeout-sec 120 \
    --internnav-rgb-topic head_camera/image \
    --internnav-depth-topic head_camera/depth \
    --internnav-camera-info-topic head_camera/camera_info \
    --internnav-adapter-target arena_vln_models.internnav:load_internvla_realworld_http_adapter \
    --internnav-model-output-policy trajectory \
    --internnav-enable-visualization \
    --internnav-external-server \
    --internnav-official-client \
    --internnav-direct-cmd-vel \
    --internnav-visualization-topic internnav/debug_image \
    --internnav-action-visualization-topic internnav/action_image \
    --internnav-visualization-rate-hz 5.0 \
    --internnav-model-output-topic internnav/model_output \
    --internnav-timing-mode wall \
    --internnav-model-latency-sec 0.3 \
    --internnav-latency-policy fixed \
    --internnav-raw-cmd-vel-topic internnav/raw_cmd_vel \
    --internnav-status-topic /task_generator_node/Ai2_Bot2/internnav/status \
    --save-eval-video \
    --eval-video-fps 10.0 \
    --eval-video-top-down-size-px 640 \
    --eval-video-top-down-window-m 10.0 \
    --eval-video-sim-top-down-topic /task_generator_node/Ai2_Bot2/top_down_camera/image \
    --eval-video-debug-overlay-topic /task_generator_node/Ai2_Bot2/internnav/debug_image \
    --external-server-preflight-timeout-sec 15 \
    --launch-timeout-sec 2400 \
    --shutdown-grace-period-sec 20 \
    --output-prefix manual_benchmark/grscenes_20_v1_default_2
'
```

`--social-eval` is required when the run should produce `vln_task_metrics.json`,
`social_metrics.json`, and `artifact_validation.json`.

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
