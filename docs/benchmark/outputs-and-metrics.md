# Artifacts And Validation

Current benchmark runs are accepted only when the runtime evidence is complete
enough to audit the episode after ROS and Isaac have exited.

## Core Run Directory

Each run is written under `outputs/<prefix>/<timestamp>_<world>_<robot>_<mode>/`.
Important files include:

| Artifact | Purpose |
| --- | --- |
| `benchmark_result.json` | Canonical compact result for automation, comparison, and release gates. |
| `run_manifest.yaml` | Launch command, parameters, runtime adjustments, artifact paths, and return codes. |
| `postprocess_commands.txt` | Commands needed to regenerate metrics and validation. |
| `episode_outcome.json` | Episode completion reason and timing. |
| `metrics.csv` | Base navigation metrics. |
| `human_states.csv` | HuNav human state stream used for social checks. |
| `pedsim_agents_data.csv` | Pedestrian stream compatibility output. |
| `cmd_vel.csv` | Command stream produced during the episode. |
| `odom.csv` | Robot odometry stream. |
| `video_index.json` | Video paths, frame counts, codec metadata, and finalization status. |

`run_manifest.yaml` records the versioned benchmark profile under
`benchmark_profile`: profile id, schema version, source path, SHA-256, copied
snapshot, precedence rule, and the resolved core parameters. This preserves exact
configuration provenance even though the normal launch command is intentionally
short.

## Canonical Result

Use `benchmark_result.json` as the stable machine-readable interface. Detailed
JSON and CSV files remain the evidence from which it is derived. The result is
written automatically at every normal or classified early-failure exit from
`internnav_eval`; generate it for an older run inside `arena-1` with:
The versioned [JSON Schema](benchmark-result.schema.json) is published with this
site for downstream validation.

```bash
ros2 run arena_bringup benchmark_result --dir "$RUN_DIR"
```

The top-level groups are:

| Group | Meaning |
| --- | --- |
| `run` | Simulator, world, scenario, robot, planner, and requested episode count. |
| `provenance` | Git commit/branch/dirty state, launch argv, runtime adjustments, environment, instruction contract, and source schema versions. |
| `execution` | End reason, evaluator return code, finished signal, and timeout state. |
| `metrics.task` | Goal reached, duration, navigation/oracle error, path lengths, SPL, nDTW, SDTW, progress, static collision, and stuck time. |
| `metrics.social` | Human presence/motion, interaction overlap, clearance, personal-space time, near misses, and collisions. |
| `metrics.model` | Trace/control evidence, action counts, stale observations, and real-time factor statistics. |
| `artifacts` | Validation status plus presence, size, and SHA-256 for source artifacts. |
| `verdict` | `passed`, `failed`, or `invalid`, with failure and diagnostic tags. |

The current `run_manifest.yaml` schema is version 2. Version 2 adds Git
provenance and canonical-result artifact metadata; the aggregate reader remains
compatible with older manifests that do not contain those fields.

`valid_run` means the environment, human motion, model/control evidence, videos,
and metric files are complete enough to score. A valid run may still be
`failed` because the robot missed the goal or violated a social constraint. An
`invalid` run is excluded from model-quality interpretation because its runtime
or evidence contract failed. `benchmark_ready` is true only when the run is
valid and all strict task, social, and artifact gates pass.

The task metrics use the native scenario goal as the current scoring target.
Navigation error is final-to-goal distance and oracle error is the closest
sampled distance. SPL uses an 8-connected A* path over the static occupancy map
inflated by the robot radius; diagonal corner cutting is forbidden. nDTW compares
distance-resampled executed and A* reference paths, and SDTW sets nDTW to zero on
task failure. The exact path source and resolution are recorded under
`metrics.task.reference_path_*`; the default goal tolerance is 0.75 m. Social
These path scores measure geometric route quality against the static map; they
do not by themselves prove that every landmark or language clause was followed.
Social success requires a moving, interacting HuNav scene and rejects footprint
collisions, footprint/point near misses,
personal-space violations, static-map collisions, commanded-stuck intervals,
and large odometry teleports. Exact thresholds are embedded under
`metrics.thresholds` in every canonical result.

## InternNav Evidence

Direct official-client mode does not rely on the timing manager to relay
`internnav/raw_cmd_vel`. A zero `raw_cmd_count` in
`internnav_timing_summary.json` is not sufficient evidence that the model was
inactive.

Use these files together:

| Artifact | Check |
| --- | --- |
| `internnav_status.json` | Latest official client status for the run. |
| `internnav_trace.jsonl` | Run-local status history after reset. |
| `internnav_diagnostic_summary.json` | Aggregated model/control diagnostics when generated. |
| `feature_trace.jsonl` | External feature/client trace when the benchmark wrapper captures it. |
| `cmd_vel.csv` | Command output actually observed by Arena. |
| Videos | Visual confirmation of motion, observation, and top-down scene state. |

For reproducible sweeps, keep external feature/client traces inside the run
directory or copy them there during cleanup.

## Social-Eval Outputs

`--social-eval` is required for the strict benchmark acceptance chain. Without
it, the run will not produce:

- `vln_task_metrics.json`
- `social_metrics.json`
- `artifact_validation.json`

This is expected behavior, not a postprocess failure.

## Required Videos

Video-enabled runs should include four videos for each episode:

| Video | Meaning |
| --- | --- |
| `ego_observation.mp4` | Robot ego RGB observation. |
| `ego_debug_overlay.mp4` | Ego view with InternNav action and command diagnostics. |
| `map_top_down_follow.mp4` | Abstract map/odom/goal/scan follow view. |
| `sim_top_down.mp4` | Isaac-rendered top-down scene view. |

At minimum, `video_index.json` should report non-zero frame counts and
`finalization_status: complete`.

## Fast Acceptance Check

Generate and inspect the canonical result inside `arena-1`:

```bash
docker exec -e RUN_DIR=/opt/arena_ws/outputs/<prefix>/<run> \
  arena-arena_jazzy_ws-arena-1 bash -lc '
  cd /opt/arena_ws &&
  source /opt/ros/jazzy/setup.bash &&
  source install/setup.bash &&
  ros2 run arena_bringup benchmark_result --dir "$RUN_DIR" --require-ready
'
```

For strict benchmark pass, require:

- launch and metrics return codes are `0`
- video index exists and all required videos have frames
- run-local InternNav status/trace evidence exists for direct-client runs
- `artifact_validation.json` exists
- `overall_pass` is `true`
- `failed_checks` is empty

## Aggregate Runs

Aggregate result directories into one CSV and one JSON summary inside
`arena-1`. `--min-runs` prevents an empty or partial discovery from silently
passing; `--require-ready` returns non-zero if any run is failed or invalid.

```bash
docker exec arena-arena_jazzy_ws-arena-1 bash -lc '
  cd /opt/arena_ws &&
  source /opt/ros/jazzy/setup.bash &&
  source install/setup.bash &&
  ros2 run arena_bringup social_nav_metrics_aggregate \
    --root /opt/arena_ws/outputs/<batch> \
    --output-csv /opt/arena_ws/outputs/<batch>/benchmark_results.csv \
    --summary-json /opt/arena_ws/outputs/<batch>/benchmark_summary.json \
    --failure-csv /opt/arena_ws/outputs/<batch>/benchmark_failures.csv \
    --min-runs 10 \
    --require-ready
'
```

The aggregate JSON reports total/valid/invalid/ready counts, task and social
success rates over all runs and over valid runs only, SPL/nDTW/SDTW and distance
means, per-metric sample counts/standard deviations/95% confidence intervals,
collision and near-miss run rates, failure counts, and per-world/per-scenario
breakdowns. Use the `*_valid_runs` fields for model-quality reporting and report
`invalid_run_count` separately.
