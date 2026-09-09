# Arena5.0

Arena5.0 is the ROS 2 Jazzy social-navigation benchmark used in this workspace.
The maintained evaluation path is Docker + Isaac Sim + the external InternNav
service. Gazebo and host-side ROS execution are not supported benchmark paths.

The runtime has three containers:

| Container | Responsibility |
| --- | --- |
| `arena-arena_jazzy_ws-arena-1` | ROS 2 launch, evaluation, recording, metrics, and tests |
| `arena-arena_jazzy_ws-isaac-1` | Isaac Sim physics, sensors, and rendering |
| `arena-arena_jazzy_ws-internnav-1` | InternNav model server and official realworld client |

Requirements are Docker, Docker Compose, NVIDIA Container Toolkit, a compatible
GPU/driver, and the GRScenes assets described in [datasets.md](datasets.md).

## Run the benchmark

Run the entry point from the host. It places ROS commands in `arena-1`, starts
the model in `internnav-1`, connects to Isaac Sim, and stops the model server
when evaluation exits.

```bash
cd /home/ubuntu/arena_jazzy_ws
src/Arena/_meta/docker/features/benchmark/main doctor
src/Arena/_meta/docker/features/benchmark/main plan run
src/Arena/_meta/docker/features/benchmark/main run
```

The strict default is one `Ai2_Bot2` episode in
`grscenes_20_v1/default_2`, using `isaac_eval`, HuNav, external InternNav direct
control, social metrics, and all four videos. Use one run directory per episode
and aggregate independent runs afterward.

Common overrides:

```bash
src/Arena/_meta/docker/features/benchmark/main run \
  --world grscenes_20_v1 \
  --scenario default_4 \
  --timeout 300 \
  -- --output-prefix review/default_4
```

See the [benchmark runbook](docs/benchmark/running-benchmarks.md) for build,
manual launch, cleanup, and postprocessing commands.

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

Build the same site that GitHub Actions publishes:

```bash
cd /home/ubuntu/arena_jazzy_ws/src/Arena
uv run --frozen --only-group docs mkdocs build --strict
```

The published site is <https://social-nav.github.io/Arena5.0/>. Pushes that
change docs or the docs build configuration on `jazzy`, `main`, `master`, or the
temporary preview branch `feat/internnav-eval-progress` trigger the Pages
workflow.
