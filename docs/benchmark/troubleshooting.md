# Troubleshooting

## Isaac crashes while creating ROS services

### Symptom

Isaac starts, but service creation fails with Python, typesupport, or FastCDR-related crashes.

### Most likely cause

The runtime is loading incompatible host-built Python 3.11 ROS message artifacts instead of the image-bundled bridge messages.

### Action

- ensure the Isaac launch path uses `/opt/isaac_bridge_msgs`
- rebuild the Isaac image after message definition changes
- avoid injecting cross-distro `install_py311_msgs` paths into Isaac

## InternNav waits forever for camera input

### Symptom

The model backend never leaves the initial waiting-for-camera state.

### Most likely cause

Isaac camera publishers use `BEST_EFFORT`, while the consumer expects the default `RELIABLE` QoS.

### Action

Use `BEST_EFFORT` subscriptions for RGB, depth, and camera info in the InternNav wrapper server.

## External InternNav server is not used

### Symptom

The dedicated `internnav` container is running, but the Arena launch still starts
a local `dual_vln_server`, or the robot never calls the external service.

### Most likely causes

- `--internnav-external-server` was omitted from `internnav_eval`
- an older task-generator YAML default overrode the launch argument
- the containers are not sharing the same ROS domain / host network discovery

### Action

- launch eval with `--internnav-external-server`; the runner forwards both
  `internnav_external_server:=true` and `dual_vln_external_server:=true`
- keep `ARENA_INTERNNAV_EXTERNAL_SERVER=1` in the eval environment when validating
  legacy configs
- verify the arena container has no local server process:

```bash
ps -ef | grep -E "dual_vln_server|internnav_server" | grep -v grep
```

- verify the external ROS contract is visible:

```bash
ros2 service list | grep /task_generator_node/Ai2_Bot2/get_command
ros2 topic list | grep /task_generator_node/Ai2_Bot2/internnav/status
```

If those endpoints are visible and the local process is absent, external-server
suppression is working.

## InternNav container has no CUDA

### Symptom

The model loads on CPU, `torch.cuda.is_available()` is false, or GPU memory does
not increase when the external server starts.

### Most likely causes

- the `internnav` service was started without the feature compose overlay
- the container does not have `gpus: all`
- the runtime venv installed CPU-only PyTorch wheels

### Action

- check the active compose services include `internnav`
- verify `NVIDIA_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility` in the service environment
- reinstall the runtime with the CUDA wheel index, for example
  `ARENA_INTERNNAV_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124`
- confirm with `nvidia-smi` from inside the `internnav` container while the model
  server is running

## Nav2 plugin not found on Jazzy

### Symptom

Planner or behavior server fails during lifecycle configure with plugin lookup errors.

### Most likely cause

Plugin names still use the older `/` separator.

### Action

On Jazzy, use:

- `nav2_navfn_planner::NavfnPlanner`
- `nav2_behaviors::Spin`
- `nav2_behaviors::BackUp`
- `nav2_behaviors::Wait`

## Video files exist but are not usable

### Symptom

The run produces metadata files but no valid `.mp4`, or the videos are encoded with the wrong codec.

### Action

- verify `ffmpeg` and `ffprobe` are available
- inspect `video_recording_error.txt`
- inspect `video_index.json`
- confirm the input RGB topic really receives robot camera frames

## InternNav rotates in place or makes little progress

### Symptom

The eval data path is alive and videos are produced, but the robot repeatedly
turns instead of moving toward the goal.

### Most likely causes

- model output is dominated by turn actions
- discrete left/right action conversion is inverted for the current robot/sim
- odom yaw and model/camera frame conventions disagree
- RGB/depth frames are stale or missing during inference
- fallback commands are being used for long periods while model inference is slow

### Action

- inspect `internnav_diagnostic_summary.json`
  - `rotate_heavy_low_progress` means turn commands dominate while goal distance does not improve
  - `possible_action_or_yaw_sign_mismatch` means turn commands often have the opposite sign from `goal.yaw_error`
  - `possible_stale_observations` or `missing_camera_inputs` point to camera freshness/QoS issues
- inspect `internnav_trace.jsonl` for `event_type`, `action.selected`, `action.effective_label`, `command.angular_z`, and `goal.yaw_error`
- inspect `videos/episode_0000/ego_debug_overlay.mp4` to verify the selected action, converted command, action history, and freshness indicators are readable frame by frame
- for Isaac + Ai2_Bot2, the eval runner defaults `--internnav-invert-discrete-turns auto`, which enables a scoped turn-sign correction; use `--internnav-invert-discrete-turns true|false` to force the behavior during A/B validation
- confirm `run_manifest.yaml` records `parameters.internnav_invert_discrete_turns_resolved` and compare it with `action.invert_discrete_turns` / `invert_discrete_turns_values` in the trace and summary

## Ego video shows a synthetic color gradient

### Symptom

The video looks like a fixed test pattern rather than a real robot view.

### Most likely cause

The recorder captured an old fallback Isaac image instead of the real camera stream.

### Action

Make sure the fallback publisher only writes to fallback topics and that the actual `head_camera/*` topics are produced by Isaac render products.

## Metrics generation fails on newer output layouts

### Action

Run metrics through the package entrypoint and point it at the run directory. The current resolver tries to bridge between manifest-based directories and the legacy recorder data layout.
