# InternNav Async Eval

Use this skill when running or debugging the current InternNav official realworld ROS2 client / async direct `cmd_vel` evaluation in Arena.

## Non-negotiable launch contract

- Run ROS/eval commands only inside `arena-arena_jazzy_ws-arena-1`.
- The async/direct eval entry is `ros2 run arena_bringup internnav_eval ... --internnav-direct-cmd-vel`.
- `--internnav-direct-cmd-vel` automatically forces external-server mode and causes the generated launch command to include `robot_launch_file:=internnav_async_eval.launch.py`.
- `internnav_async_eval.launch.py` intentionally skips Nav2, rosnav_rl, and the local Arena `dual_vln_server` wrapper. It only starts robot-side eval recording/readiness pieces needed by the external direct `cmd_vel` client.
- Do not treat plain `--internnav-external-server` as the async launch selector. Plain external-server mode is the legacy Nav2 service-contract path and can still use `robot.launch.py` with local server suppression.

## Correct command shape

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
  cd /opt/arena_ws &&
  source /opt/ros/jazzy/setup.bash &&
  source install/setup.bash &&
  ros2 run arena_bringup internnav_eval \
    --sim isaac \
    --human hunav \
    --world <world> \
    --robot Ai2_Bot2 \
    --local-planner dual_vln \
    --episodes <n> \
    --timeout <sec> \
    --timeout-wall-sec <wall_sec> \
    --headless 2 \
    --internnav-mode internnav \
    --internnav-direct-cmd-vel
'
```

## Required verification

After a run starts or after inspecting `manifest.json` / `postprocess_commands.txt`, verify:

```text
robot_launch_file:=internnav_async_eval.launch.py
internnav_external_server:=true
dual_vln_external_server:=true
internnav_direct_cmd_vel:=true
dual_vln_direct_cmd_vel:=true
```

If the target is async/direct eval and the generated command lacks `robot_launch_file:=internnav_async_eval.launch.py`, stop and fix the invocation before investigating model-server or Nav2 errors.

## Common hallucination trap

Do not debug `dual_vln_server` missing in `arena-1` for async/direct eval. That executable should not be started there. The usual root cause is using the legacy `robot.launch.py` path by omitting `--internnav-direct-cmd-vel` or by relying on stale docs that only mention `--internnav-external-server`.
