# Arena Benchmark

This site documents the current Arena benchmark workflow used in this workspace:
Docker orchestration, Isaac Sim rendering, HuNav pedestrians, and InternNav /
Dual-VLN evaluation.

The supported runtime topology is three containers:

| Container | Responsibility |
| --- | --- |
| `arena-arena_jazzy_ws-arena-1` | ROS 2 Jazzy evaluator, launch, metrics, video recording, and tests. |
| `arena-arena_jazzy_ws-isaac-1` | Isaac Sim simulation and rendering. |
| `arena-arena_jazzy_ws-internnav-1` | GPU InternNav model service and official realworld client. |

Host commands are limited to Docker orchestration, file editing, Git operations,
log inspection, and GPU/process checks. Run ROS 2, colcon, metrics, and eval
commands inside `arena-1`.

## Current benchmark contract

- Simulator: `isaac_eval`
- Human simulator: `hunav`
- Robot: commonly `Ai2_Bot2`
- Model path: `/opt/arena_ws/deps/models/InternVLA-N1-DualVLN`
- InternNav execution: external server in `internnav-1`
- Direct-control eval: `--internnav-direct-cmd-vel` /
  `robot_launch_file:=internnav_async_eval.launch.py`
- Required evidence for accepted runs:
  `benchmark_result.json`, `run_manifest.yaml`, `video_index.json`, videos,
  metrics, and run-local InternNav status/trace evidence.

## Pages

- [Installation](installation.md)
- [Run Current Benchmarks](benchmark/running-benchmarks.md)
- [Artifacts and Validation](benchmark/outputs-and-metrics.md)
- [Troubleshooting](benchmark/troubleshooting.md)
