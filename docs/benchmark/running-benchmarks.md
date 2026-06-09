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

## Dynamic Social VLN scenario config

Dynamic Social VLN runs can be launched from a top-level scenario YAML overlay.
The sample config is:

```text
arena_bringup/configs/social_nav_scenarios/hospital_1_demo_001.yaml
```

This file is an overlay, not an Arena world replacement.  It references the
native Arena assets and adds benchmark-level semantics:

| Section | Purpose |
| --- | --- |
| `schema_version`, `id`, `name`, `description` | Stable identity and human-readable metadata. |
| `world` | World name, `world.yaml`, `map.yaml`, native Arena scenario file, and optional semantic region names. |
| `robot` | Robot name, planner, start pose, goal pose, tolerance, and expected command/odom/goal topics. |
| `language` | Natural-language VLN instruction, instruction type, and optional rephrases. |
| `humans` | Human simulator (`hunav`), expected actor count, source scenario, and behavior-tree references. |
| `task_spec` | BDDL-like predicates for success, social constraints, and failure tags. |
| `evaluation` | Timeout, repetitions, metric groups, and pass criteria. |
| `artifacts_required` | Files/videos that must exist for an accepted run. |

Validate the overlay before running a long simulation:

```bash
cd /home/ubuntu/arena_jazzy_ws
PYTHONPATH=src/Arena/arena_bringup \
python3 -m arena_bringup.social_nav_scenario \
  src/Arena/arena_bringup/configs/social_nav_scenarios/hospital_1_demo_001.yaml
```

After the package is built and sourced, the installed console script is:

```bash
ros2 run arena_bringup social_nav_scenario_validate \
  src/Arena/arena_bringup/configs/social_nav_scenarios/hospital_1_demo_001.yaml
```

For CI or launch review, use `--dry-run` to inspect the derived `internnav_eval`
arguments without starting Isaac or ROS launch processes:

```bash
ros2 run arena_bringup social_nav_scenario_eval \
  --scenario-config src/Arena/arena_bringup/configs/social_nav_scenarios/hospital_1_demo_001.yaml \
  --dry-run \
  --no-save-eval-video \
  -- --internnav-external-server --headless 2
```

The wrapper validates the YAML and translates it into the existing
`internnav_eval` contract, including `--social-eval`, `--world`, `--robot`,
`--local-planner`, `--scenario-file`, `--internnav-mode internnav`, the VLN
instruction, and scenario metadata stored later in `run_manifest.yaml`.

Run ROS/eval commands from the Arena container, not from the host shell.  The
validated Docker image is built from `osrf/ros:jazzy-desktop` and contains
`/opt/ros/jazzy/setup.bash`; a host machine may still have only `/opt/ros/humble`
installed.  The container roles are fixed: `arena-arena_jazzy_ws-arena-1` runs
evaluator/ROS scripts, `arena-arena_jazzy_ws-isaac-1` runs Isaac Sim, and
`arena-arena_jazzy_ws-internnav-1` runs the GPU InternNav model.  A quick sanity
check is:

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc \
  'source /opt/ros/jazzy/setup.bash && echo $ROS_DISTRO && python3 --version'
```

To run the full scenario with required video artifacts, omit `--dry-run` and
`--no-save-eval-video`:

```bash
ARENA_INTERNNAV_EXTERNAL_SERVER=1 \
ros2 run arena_bringup social_nav_scenario_eval \
  --scenario-config src/Arena/arena_bringup/configs/social_nav_scenarios/hospital_1_demo_001.yaml \
  --output-prefix hospital_1_demo_001 \
  -- --internnav-external-server --headless 2
```

Any arguments after `--` are appended to `internnav_eval`, so this is where you
pass backend/runtime controls such as `--internnav-external-server`,
`--internnav-device`, `--internnav-model-path`, `--internnav-python-executable`,
or explicit camera topics.

The latest validated video rerun used this path:

```text
/home/ubuntu/arena_jazzy_ws/outputs/social_nav_video_rerun_20260517/20260517_081450_hospital_1_Ai2_Bot2_internnav
```

It produced all four required videos under `videos/episode_0000/` and passed
`artifact_validation.json` with `overall_pass=true` and `social_nav_ready=true`.

### Recommended three-container flow

For GPU-backed InternNav inference, run the model server from the dedicated
`internnav-1` container and tell Arena not to launch a local server:

```bash
cd /home/ubuntu/arena_jazzy_ws
source src/Arena/_meta/docker/source

# Build/install or refresh the third container runtime when needed.
src/Arena/_meta/docker/features/internnav/main install
src/Arena/_meta/docker/features/internnav/main install-runtime

# Start the external server in /task_generator_node/Ai2_Bot2.  This command
# enters arena-arena_jazzy_ws-internnav-1; do not start the model in arena-1.
src/Arena/_meta/docker/features/internnav/main launch \
  --robot Ai2_Bot2 \
  --mode internnav \
  --device cuda:0
```

Then run the evaluation from `arena-arena_jazzy_ws-arena-1` with external-server mode:

```bash
ARENA_INTERNNAV_EXTERNAL_SERVER=1 \
ros2 run arena_bringup internnav_eval \
  --sim isaac \
  --human hunav \
  --world hospital_1 \
  --robot Ai2_Bot2 \
  --local-planner dual_vln \
  --episodes 1 \
  --timeout 60 \
  --headless 2 \
  --log-level warn \
  --internnav-mode internnav \
  --internnav-device cuda:0 \
  --internnav-external-server
```

The important launch contract is `--internnav-external-server`: it forwards
`internnav_external_server:=true` and `dual_vln_external_server:=true`, causing
the robot launch to suppress any local model `dual_vln_server` in arena-1.  The model container
continues to provide `/task_generator_node/Ai2_Bot2/get_command` and
`/task_generator_node/Ai2_Bot2/internnav/status`.

Use `--save-eval-video --internnav-enable-visualization` when video artifacts are
required.  Without `--save-eval-video`, the run still records CSVs and metrics but
does not create `videos/episode_xxxx/*.mp4`.

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
  --internnav-inference-rate-hz 10 \
  --internnav-inference-timeout-sec 30 \
  --internnav-invert-discrete-turns auto \
  --internnav-enable-visualization \
  --save-eval-video
```

This command writes the normal manifest/video artifacts plus InternNav diagnostics:

- `internnav_trace.jsonl` for per-inference and fallback/camera-gate decisions
- `internnav_diagnostic_summary.json` for action distribution, progress, and fault-candidate flags
- `videos/episode_0000/ego_debug_overlay.mp4` for frame-by-frame action and command visualization

If a legacy or baseline run has `ego_debug_overlay.mp4` but no trace/summary
files, it predates the current tracing path or was launched without the current
eval runner parameters.  Use a fresh run before comparing turn-sign A/B results.

## Important InternNav arguments

- `--internnav-mode`: selects the backend mode
- `--internnav-adapter-target`: points to the adapter loader
- `--internnav-model-path`: selects the model directory
- `--internnav-device`: selects CPU or CUDA runtime
- `--internnav-inference-rate-hz`: controls how often model inference is requested
- `--internnav-inference-timeout-sec`: bounds model/backend inference before reporting timeout diagnostics
- `--internnav-invert-discrete-turns`: `auto` enables the current Isaac + Ai2_Bot2 turn-sign correction; `true` or `false` force reproducible A/B runs
- `--internnav-enable-visualization`: publishes the debug overlay image stream recorded as `ego_debug_overlay.mp4` when video recording is enabled
- `--save-eval-video`: enables video recorder subprocess
- `--output-prefix`: groups outputs under a chosen benchmark label

## Where the InternNav environment lives

For Isaac evaluations, keep Isaac Sim and InternNav model inference in separate
runtime environments.  The preferred layout is the three-container architecture:

- the Isaac container runs Isaac Sim, rendering, physics, and the Isaac ROS bridge
- the Arena container runs ROS Jazzy, Nav2, recording, metrics, and evaluation orchestration
- the InternNav container runs the external `dual_vln_server` and model subprocess
  from `/opt/internnav_venv`

For older two-container debugging, the heavy InternNav model can still run from a
Python environment reachable inside the Arena container, selected with
`--internnav-python-executable` or `ARENA_INTERNNAV_PYTHON`.

For the current workspace layout, the isolated Arena-side model environment can
live at:

```bash
/home/ubuntu/arena_jazzy_ws/.venvs/internnav/bin/python
```

The model subprocess also needs the local Depth-Anything checkpoint available to
the Arena container, typically via:

```bash
export INTERNNAV_DEPTH_ANYTHING_CKPT=/home/ubuntu/arena_jazzy_ws/deps/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth
```

`internnav_server` stays in the ROS Python interpreter so `rclpy` and generated
message type-support match the ROS installation.  When a different InternNav
Python is configured, the server starts it as a subprocess and exchanges only
JSON plus temporary NumPy files.  This avoids mixing the ROS Python ABI with the
model stack (`torch`, `transformers`, `flash_attn`, `diffusers`, and related
packages).  Installing the model stack in the Arena image is fine when it is
isolated in a conda/venv and passed via `--internnav-python-executable`; avoid
installing it into the system ROS Python unless the versions are known to be
compatible.

Recently validated Arena-side isolated interpreter:

- `/home/ubuntu/arena_jazzy_ws/.venvs/internnav/bin/python`

This path successfully ran a full `hospital_1 + Ai2_Bot2 + InternNav` eval
through launch, recording, metrics, trace generation, and video export.

Example with an isolated Arena venv:

```bash
export INTERNNAV_DEPTH_ANYTHING_CKPT=/home/ubuntu/arena_jazzy_ws/deps/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth

ros2 run arena_bringup internnav_eval \
  --sim isaac \
  --human hunav \
  --world hospital_1 \
  --robot Ai2_Bot2 \
  --episodes 1 \
  --timeout 180 \
  --headless 2 \
  --internnav-mode internnav \
  --internnav-device cpu \
  --internnav-model-path /home/ubuntu/arena_jazzy_ws/deps/models/InternVLA-N1-DualVLN \
  --internnav-python-executable /home/ubuntu/arena_jazzy_ws/.venvs/internnav/bin/python \
  --internnav-adapter-target arena_vln_models.internnav:load_internnav_adapter \
  --internnav-rgb-topic head_camera/image \
  --internnav-depth-topic head_camera/depth \
  --internnav-camera-info-topic head_camera/camera_info \
  --internnav-require-real-backend \
  --internnav-enable-visualization \
  --save-eval-video
```

If the run ends with `camera_timeout`, check the camera topics before debugging
the InternNav model environment itself:

```bash
ros2 topic info -v /task_generator_node/Ai2_Bot2/head_camera/image
ros2 topic info -v /task_generator_node/Ai2_Bot2/head_camera/depth
ros2 topic info -v /task_generator_node/Ai2_Bot2/head_camera/camera_info
```

For a healthy Isaac run, those topics must have active publishers. If topic
names exist but `Publisher count: 0`, the remaining fault is on the Isaac side
(sensor/bridge/rendering startup), not in the Arena-side InternNav venv.

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
- `internnav_diagnostic_summary.json` should show non-trivial forward progress
  and should not repeatedly flag `possible_action_or_yaw_sign_mismatch` after the
  scoped turn-sign correction is enabled.

### Interpreting recent InternNav Isaac retests

The isolated-venv retest at
`outputs/internnav_hospital1_ai2_isolatedvenv_retest/20260515_064643_hospital_1_Ai2_Bot2_internnav`
established these important points:

- the end-to-end eval pipeline is healthy again: camera topics published, videos
  were recorded and closed cleanly, `internnav_trace.jsonl` was generated, and
  `run_manifest.yaml` ended with `end_reason: finished`
- the earlier `camera_timeout` diagnosis was not the steady-state blocker once
  the active eval container was checked; the active run had fresh RGB/depth/
  camera-info timestamps and live publishers
- after fixing the Isaac-side control / odometry graph, Ai2_Bot2 regained real
  physical motion; subsequent retests showed meaningful odom velocity and path
  length instead of a stationary base under non-zero `cmd_vel`
- with motion restored, a turn-sign A/B rerun showed that `--internnav-invert-discrete-turns false`
  materially outperformed the old inverted behavior for Isaac + Ai2_Bot2

In practice, for this robot/world pair you should treat the following as the key
post-run triage split:

1. **Camera / observation failure**
   - `internnav_status.json` shows missing or stale RGB/depth/camera info
   - `internnav_trace.jsonl` is dominated by `camera_timeout` or missing-input
     diagnostics
   - videos are absent or nearly empty

2. **Motion / controller failure**
   - videos and trace artifacts are present and normal
   - `cmd_vel.csv` contains non-zero commands
   - `odom.csv` shows little or no true motion after reset, or zero reported
     twist while commands continue
   - Nav2 emits `Failed to make progress`

The first May 2026 isolated-venv retest fell into category 2. After the motion
fix, the remaining tuning issue shifted back to turn/action interpretation.

### Current Isaac + Ai2_Bot2 turn-sign guidance

For the current validated setup, `--internnav-invert-discrete-turns auto`
resolves to **no inversion** for Isaac + Ai2_Bot2.

This is based on the A/B comparison between:

- `internnav_hospital1_ai2_motionfix/...` with inversion enabled, which still
  regressed in goal distance and flagged `rotate_heavy_low_progress`
- `internnav_hospital1_ai2_motionfix_noinvert/...` with inversion disabled,
  which finished with meaningful forward progress and improved goal distance

If you need to reproduce the comparison explicitly, run one eval with:

```bash
--internnav-invert-discrete-turns true
```

and a second with:

```bash
--internnav-invert-discrete-turns false
```

Then compare `internnav_diagnostic_summary.json`:

- `goal_distance.progress_first_minus_last` should be positive for a better run
- `fault_candidates.flags` should ideally omit `rotate_heavy_low_progress`
- `odom.csv` should show real motion alongside non-zero `cmd_vel.csv`

## Running the documentation site locally

```bash
cd /home/ubuntu/arena_jazzy_ws/src/Arena
uv run --group docs mkdocs serve
```
