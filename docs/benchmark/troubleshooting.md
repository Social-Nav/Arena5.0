# Troubleshooting

## GPU Memory Stays High

Check before cleaning. Do not reboot the host as a first step.

```bash
nvidia-smi
docker exec arena-arena_jazzy_ws-isaac-1 ps -eo pid,ppid,stat,comm,args
docker exec arena-arena_jazzy_ws-arena-1 ps -eo pid,ppid,stat,comm,args
docker exec arena-arena_jazzy_ws-internnav-1 ps -eo pid,ppid,stat,comm,args
pgrep -af 'run_isaacsim|isaac-sim|kit/kit|internnav_eval|dual_vln_eval|dual_vln_server|internnav_server|task_generator_node|controller_server|planner_server|bt_navigator' || true
```

Only clean processes that are confirmed to be old eval/model runs. Prefer
container-local `pkill -TERM`, wait, then `pkill -KILL`.

## Social Metrics Are Missing

If `vln_task_metrics.json`, `social_metrics.json`, and
`artifact_validation.json` are absent, first check `run_manifest.yaml`:

```yaml
parameters:
  social_eval: false
```

When `social_eval` is `false`, those files are not expected. Re-run with
`--social-eval` for strict benchmark acceptance.

## InternNav Status Or Trace Is Missing

Direct official-client runs need the status topic to match the robot namespace:

```bash
--internnav-status-topic /task_generator_node/Ai2_Bot2/internnav/status
```

Relative `internnav/status` is normalized by the current runner, but older run
scripts may have produced manifests with `internnav_status_present: false` and
`internnav_trace_present: false`. Re-run after updating the script or pass the
absolute topic explicitly.

## Timing Summary Shows Zero Raw Commands

In direct official-client mode, `internnav_timing_manager` does not relay
`internnav/raw_cmd_vel`. These fields can be zero:

```json
{
  "raw_cmd_count": 0,
  "emitted_cmd_count": 0,
  "input_publisher_seen": false
}
```

Do not treat that alone as model inactivity. Check run-local status/trace,
`feature_trace.jsonl`, `cmd_vel.csv`, and videos.

## Videos Are Missing Or Empty

Confirm the eval used:

```bash
--save-eval-video
--internnav-enable-visualization
--eval-video-debug-overlay-topic /task_generator_node/Ai2_Bot2/internnav/debug_image
--eval-video-sim-top-down-topic /task_generator_node/Ai2_Bot2/top_down_camera/image
```

Then inspect `video_index.json`:

- `finalization_status` should be `complete`
- each episode should have non-zero frame counts
- required paths should point to existing `.mp4` files

## Humans Are Missing

For HuNav social navigation, check:

- `human:=hunav`
- `tm_obstacles:=scenario` when strict social validation is expected
- `human_states.csv` is non-empty
- `pedsim_agents_data.csv` is non-empty
- `artifact_validation.checks.humans.pass` is true

The synthetic HuNav robot agent should be filtered from published human states.

## Run Directory Has Feature Trace Outside The Run

Some benchmark wrappers place `feature_trace.jsonl` under the output prefix root
instead of the individual run directory. For long sweeps, copy or write that file
into the run directory so the evidence stays self-contained.
