# Arena5.0

Arena5.0 is the ROS 2 Jazzy social-navigation benchmark used in this workspace.
The maintained evaluation path is Docker + Isaac Sim + an external native
Nav2 FollowPath client. Gazebo and host-side ROS execution are not supported benchmark paths.

Arena manages two simulation containers; model repositories own their runtime:

| Container | Responsibility |
| --- | --- |
| `arena-arena_jazzy_ws-arena-1` | ROS 2 launch, evaluation, recording, metrics, and tests |
| `arena-arena_jazzy_ws-isaac-1` | Isaac Sim physics, sensors, and rendering |

Requirements are Docker, Docker Compose, NVIDIA Container Toolkit, a compatible
GPU/driver, and the external GRScenes assets. See the
[installation guide](docs/installation.md) for a clean-machine setup and
[dataset notes](datasets.md) for the runtime layout.

## Run the benchmark

Run the entry point from the host. It places ROS commands in `arena-1` and
connects to Isaac Sim. `eval` uses an already running external client; `run`
optionally starts the profile’s `external.command` and cleans up only that process.

```bash
cd /home/ubuntu/arena_jazzy_ws
src/Arena/_meta/docker/features/benchmark/main config
src/Arena/_meta/docker/features/benchmark/main doctor
src/Arena/_meta/docker/features/benchmark/main plan run
src/Arena/_meta/docker/features/benchmark/main run
```

The strict default is one `Ai2_Bot2` episode in
`grscenes_20_v1/default_2`, using `isaac_eval`, HuNav, native FollowPath/RPP,
social metrics, and three videos. Model debug overlays are optional. Use one run directory per episode
and aggregate independent runs afterward.

Common overrides:

```bash
src/Arena/_meta/docker/features/benchmark/main run \
  --case grscenes_20_v1/default_4 \
  --timeout 300 \
  -- --output-prefix review/default_4
```

See [Model Integration and Evaluation](docs/Model_Evaluation.md) for the
architecture, model repository downloads, NaVILA, InternNav and LightNav setup, and the shared
FollowPath interface. InternNav, LightNav and NaVILA profiles use the same runner;
All three models have completed client integration; their task and safety results are recorded separately.

Cross-machine settings are versioned in
`arena_bringup/configs/benchmark/profiles/internnav_grscenes.yaml`; host paths
and credentials remain in the workspace `.env`. CLI case options override the
profile without changing it.

See [benchmark commands](docs/Model_Evaluation.md#benchmark-commands) for running
evaluations and [troubleshooting](docs/Model_Evaluation.md#troubleshooting) for
startup and execution failures.

## Results and metrics

Each completed or classified early-failure run writes
`benchmark_result.json`, the stable machine-readable result. It includes:

- source revision and runtime provenance;
- execution status and failure classification;
- goal error, path length, SPL, nDTW, and SDTW;
- human motion, clearance, near-miss, collision, and personal-space metrics;
- model/control evidence and real-time factor statistics;
- artifact presence, size, and SHA-256 checksums.

Detailed source artifacts remain alongside it for audit. See
[Artifacts and Validation](docs/benchmark/outputs-and-metrics.md) for the schema,
acceptance rules, and batch aggregation commands.

## Documentation

Start with [Installation](docs/installation.md) on a clean machine.
Continue with [Model Integration and Evaluation](docs/Model_Evaluation.md) to
download and run a model or connect a new client.

Build the same site that GitHub Actions publishes:

```bash
cd /home/ubuntu/arena_jazzy_ws/src/Arena
UV_PROJECT_ENVIRONMENT=.venv-docs \
  uv run --python 3.12 --frozen --only-group docs mkdocs build --strict
```

The published site is <https://social-nav.github.io/Arena5.0/>. Pushes that
change docs or the docs build configuration on `jazzy`, `main`, `master`, or the
temporary preview branch `feat/internnav-eval-progress` trigger the Pages
workflow.
