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
