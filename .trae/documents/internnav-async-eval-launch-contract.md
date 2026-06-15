# InternNav async eval launch contract

This repository has two different InternNav external execution paths:

1. **Current async/direct `cmd_vel` eval**: selected by `internnav_eval --internnav-direct-cmd-vel`, which appends `robot_launch_file:=internnav_async_eval.launch.py`.
2. **Legacy Nav2 service-contract external-server eval**: selected by plain `--internnav-external-server`; this suppresses the local model server but can still use `robot.launch.py`.

For the current InternNav official realworld ROS2 client, use path 1 only.

## Source-of-truth code locations

- `arena_bringup/arena_bringup/internnav_eval.py`: `--internnav-direct-cmd-vel` sets `internnav_external_server=True` and appends `robot_launch_file:=internnav_async_eval.launch.py`.
- `arena_simulation_setup/launch/internnav_async_eval.launch.py`: case-specific launch file for async/direct eval; intentionally skips Nav2 and local `dual_vln_server`.
- `task_generator/task_generator/manager/robot_manager/robot_manager.py`: includes whichever `robot_launch_file` was passed through the launch arguments.

## Operator checklist

Before considering an async/direct eval valid, confirm the generated launch command includes:

```text
internnav_external_server:=true
dual_vln_external_server:=true
internnav_direct_cmd_vel:=true
dual_vln_direct_cmd_vel:=true
robot_launch_file:=internnav_async_eval.launch.py
```

If `robot.launch.py` is selected for an async/direct eval, the invocation is wrong. Fix the command first; do not spend time debugging the absence of a local `dual_vln_server` executable in `arena-1`.
