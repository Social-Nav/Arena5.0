# NaVILA in Arena

NaVILA runs in its own model container. An Arena ROS client sends RGB history and
the episode instruction over TCP, then converts the model's action text into
`Twist` commands for Ai2_Bot2. Arena owns simulation, recording and scoring.

The local NaVILA source, weights and model service are reused unchanged. The
adapter follows NaVILA-Bench `e9d2db12ce5788c0f987d734c0094100b6bc0d3a` action rules:
forward 25/50/75 cm at 0.5 m/s, or left/right 15/30/45 degrees at pi/6 rad/s,
for 0.5/1/1.5 simulation seconds. STOP publishes zero. Odometry measures the
execution error; it does not extend the command duration to compensate.

## Commands

Run from `/home/kaizhen/arena_ws`. Stop the currently selected algorithm through
its benchmark profile before starting another controller. Active evaluations and
existing velocity publishers are checked; a competing controller is refused.

```bash
src/Arena/_meta/docker/features/benchmark/main doctor \
  --profile src/Arena/arena_bringup/configs/benchmark/profiles/navila_grscenes.yaml

src/Arena/_meta/docker/features/benchmark/main serve \
  --profile src/Arena/arena_bringup/configs/benchmark/profiles/navila_grscenes.yaml
```

In a second terminal:

```bash
cd /home/kaizhen/arena_ws
src/Arena/_meta/docker/features/benchmark/main eval \
  --profile src/Arena/arena_bringup/configs/benchmark/profiles/navila_grscenes.yaml \
  --case grscenes_20_v1/default_2 --episodes 1 --timeout 300 \
  -- --output-prefix navila_eval/restaurant
```

`run` starts the selected service, evaluates, and cleans up its model/client.
`serve` plus consecutive `eval` commands retains the model and client between
episodes. `stop` targets this profile's model/client and preserves shared
Arena/Isaac containers. The existing InternNav default and LightNav profile remain
available. `config`, `plan run`, `status`, and `logs` use the same profile argument.

To override the scene instruction, append, for example,
`--vln-instruction "Keep moving straight ahead."` after `--`. This changes the
model instruction, not the scenario's evaluation goal or termination criteria.

## Deployment and behavior

- Native feature install: `src/Arena/_meta/docker/features/navila/main install`.
- Model project: `arena-navila`; model image: `navila-isaac-vlm:cu122`.
- Weights: `/home/kaizhen/vln_ws/NaVILA/artifacts/checkpoints/navila-llama3-8b-8f`.
- Endpoint: `tcp://127.0.0.1:54321`; protocol: eight-byte big-endian length and JSON.
- Source/weight/image overrides: `NAVILA_SOURCE_PATH`, `NAVILA_MODEL_PATH`,
  `NAVILA_IMAGE`; the endpoint and device are selected by the profile.
- Native image sampling: one new RGB per 0.5 simulation seconds, seven uniformly
  sampled historical frames plus the current frame; black padding when needed.
- Velocity publication: 20 Hz simulation time. The independent wall watchdog
  stops commands within 0.5 seconds of clock/odometry/control inactivity.
  The shared profile's `execution.timing_mode: wall` disables Arena's command
  relay; NaVILA action durations still use `/clock`, not wall time.
- During inference the robot stops while the scene continues. The original
  synchronous Isaac evaluator pauses simulation during inference; this is an
  intentional integration difference, recorded in provenance.
- Invalid outputs, network errors and stale predictions cannot become motion.
  Reset and clock rollback discard old history, requests and actions.
- The model receives neither the evaluation goal nor reference trajectories.

Results are under `/home/kaizhen/arena_ws/outputs/<output-prefix>/<run>/`:
`benchmark_result.json`, `navila_trace.jsonl`, `navila_diagnostic_summary.json`,
trajectory/ROS bag, navigation/social metrics and four videos. Diagnostics link
each nonzero command to its model request and accepted action. Model STOP does
not imply arrival. Integration, goal arrival and social performance are reported
separately.
