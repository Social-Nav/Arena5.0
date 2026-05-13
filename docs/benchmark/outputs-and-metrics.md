# Outputs and Metrics

## Legacy evaluation outputs

The legacy evaluation pipeline writes topic-wise CSV files such as:

- `odom.csv`
- `scan.csv`
- `episode.csv`
- `start_goal.csv`
- `metrics.csv`

These are consumed by `arena_evaluation` scripts for offline metrics and plotting.

## InternNav evaluation outputs

The newer InternNav path adds structured run artifacts on top of metrics:

- `run_manifest.yaml`
- `internnav_status.json`
- `video_index.json`
- `video_recording_error.txt` (when needed)
- `videos/episode_xxxx/*.mp4`
- snapshot config files used for the run

## Output root behavior

The current InternNav runner defaults user-facing outputs to a workspace-level `outputs/` directory instead of mixing them into package `install/` content.

This is important because benchmark outputs are generated artifacts, not installed resources.

## Video artifacts

Depending on topic availability, a run may produce:

- `ego_observation.mp4`
- `ego_debug_overlay.mp4`
- `map_top_down_follow.mp4`
- `sim_top_down.mp4`

## Codec policy

The current recorder prefers H.264 MP4 output:

1. try `imageio` with `libx264`
2. fallback to OpenCV MP4 writer
3. if the detected codec is not H.264, try `ffmpeg` transcoding
4. verify the final file with `ffprobe`

This makes the benchmark outputs easier to archive, inspect, and share.

## Metrics compatibility

`arena_evaluation/get_metrics.py` includes a directory resolver so that metrics can still be generated when the recorder output layout and the newer manifest-based layout coexist.

## What to inspect after a run

After a benchmark or InternNav eval run, check these first:

- did the launch finish cleanly?
- is `run_manifest.yaml` present?
- is `end_reason` equal to `finished`?
- is `video_recorder_returncode` equal to `0` when video recording is enabled?
- do the generated videos have frames and the expected codec?
- did `metrics.csv` get generated when metrics were requested?
