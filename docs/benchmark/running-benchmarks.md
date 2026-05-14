# Running Benchmarks

## Classical benchmark flow

The repository keeps the classic benchmark configuration under `arena_bringup/configs/benchmark/`.

Conceptually, the run is determined by:

- a simulator backend
- a suite
- a contest
- robot and map selections per suite stage

You usually launch Arena with the desired planner stack and let the task generator benchmark module manage resets and stage progression.

## Useful benchmark configuration files

- `arena_bringup/configs/benchmark/config.yaml`
- `arena_bringup/configs/benchmark/contests/basic.yaml`
- `arena_bringup/configs/benchmark/suites/basic.yaml`

These files are the right starting point when you want to define new benchmark workloads or planner comparisons.

## Running Isaac Sim directly

Example:

```bash
cd ~/arena_ws
source arena
arena launch sim:=isaac robot:=Ai2_Bot2 world:=hospital_1
```

## Running an InternNav evaluation

The specialized entrypoint is `arena_bringup/arena_bringup/internnav_eval.py`.

Typical Isaac example:

```bash
ros2 run arena_bringup internnav_eval \
  --sim isaac \
  --human hunav \
  --world hospital_1 \
  --robot Ai2_Bot2 \
  --episodes 1 \
  --timeout 240 \
  --tm-robots scenario \
  --headless 2 \
  --vln-instruction "navigate to the goal" \
  --internnav-mode internnav \
  --internnav-device cpu \
  --internnav-adapter-target arena_vln_models.internnav:load_internnav_adapter \
  --internnav-model-path /opt/arena_ws/deps/models/InternVLA-N1-DualVLN \
  --internnav-enable-visualization \
  --save-eval-video
```

## Important InternNav arguments

- `--internnav-mode`: selects the backend mode
- `--internnav-adapter-target`: points to the adapter loader
- `--internnav-model-path`: selects the model directory
- `--internnav-device`: selects CPU or CUDA runtime
- `--save-eval-video`: enables video recorder subprocess
- `--output-prefix`: groups outputs under a chosen benchmark label

## Choosing the robot and world

For the recently verified Isaac path, the most relevant validation combination is:

- world: `hospital_1`
- robot: `Ai2_Bot2`

That combination exercises:

- USD robot loading
- Isaac bridge messages
- camera subscription QoS
- Nav2 controller integration
- video export and H.264 validation

### Ai2_Bot2 Isaac-specific checks

Ai2_Bot2 is a USD robot with embedded OmniGraph controllers.  Arena disables
the embedded `ROS2SubscribeTwist` graph and installs a namespaced Arena
diff-drive graph so `/task_generator_node/Ai2_Bot2/cmd_vel` drives the robot
consistently.  During validation, check both of these conditions instead of only
checking that an episode finished:

- `odom.csv` should show motion after non-empty `cmd_vel.csv` commands, not just
  a one-time slide/fall during initialization.
- `videos/episode_0000/sim_top_down.mp4` should stay centered above the robot;
  the camera prim is a standalone `/World/vln_top_down_camera_*` prim and must
  be explicitly moved on every task reset.

## Running the documentation site locally

```bash
cd /home/ubuntu/arena_jazzy_ws/src/Arena
uv run --group docs mkdocs serve
```
