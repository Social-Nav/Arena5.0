# Datasets

This project keeps large GRScenes assets outside the Arena package tree and
loads them from a shared dataset mount inside the Docker containers.  Do not
copy a full GRScenes USD asset tree into `arena_simulation_setup/worlds/`; USD
files usually contain relative references to sibling `Materials`, `models`, and
texture assets, so moving one file without its original layout can make Isaac
load an incomplete scene.

## GRScenes Layout

Arena world configs for GRScenes live under:

```text
arena_simulation_setup/worlds/grscenes_<id>/world.yaml
```

Each `world.yaml` should point `usd_scene.path` at the USD file as it is visible
inside both the Arena and Isaac containers.  The standard dataset mount is:

```text
/data/scenes/commercial_scenes/scenes/<scene_id>_usd/
```

The active GRScenes benchmark worlds use trimmed navigation USDs:

| Arena world | Dataset scene directory | USD filename |
| --- | --- | --- |
| `grscenes_1` | `MV4AFHQKTKJZ2AABAAAAADQ8_usd` | `trimed_navigation.usd` |
| `grscenes_3` | `MV4AFHQKTKJZ2AABAAAAAEA8_usd` | `trimed_navigation.usd` |
| `grscenes_5` | `MV5M25QKTKJZ2AABAAAAAAA8_usd` | `trimed_navigation.usd` |
| `grscenes_6` | `MV5M25QKTKJZ2AABAAAAAAI8_usd` | `trimed_navigation.usd` |
| `grscenes_9` | `MV5M25QKTKJZ2AABAAAAAEI8_usd` | `trimed_navigation.usd` |

For example, the `grscenes_5` trimmed scene is configured as:

```yaml
world_type: "usd"

usd_scene:
  path: "/data/scenes/commercial_scenes/scenes/MV5M25QKTKJZ2AABAAAAAAA8_usd/trimed_navigation.usd"
  scale: 0.01
  position: [0.0, 0.0, 0.0]
  orientation: [0.0, 0.0, 0.0, 1.0]
```

GRScenes coordinates are authored in centimeters, so the Arena world config uses
`scale: 0.01` to convert the scene to meters.

## Trimmed USD Files

When a scene needs a cleaned version of the original USD, place the cleaned file
next to the original asset inside the dataset scene directory:

```text
/data/scenes/commercial_scenes/scenes/<scene_id>_usd/start_result_navigation.usd
/data/scenes/commercial_scenes/scenes/<scene_id>_usd/trimed_navigation.usd
```

Then update only that world's `usd_scene.path` to the cleaned USD.  Keeping the
trimmed USD beside the original preserves its relative references and makes
future dataset updates straightforward:

1. Replace or regenerate only `trimed_navigation.usd` in the dataset directory.
2. Keep the original `start_result_navigation.usd` available for comparison.
3. Leave package-local `world.yaml`, scenarios, maps, and metadata under git.

In this workspace, the cleaned uploads are staged under:

```text
/home/ubuntu/arena_jazzy_ws/grs/grscenes_<id>/start_result_navigation.usd
```

Copy each cleaned file into the matching dataset scene directory as
`trimed_navigation.usd`.  The Arena and Isaac containers both need to see the
same container path under `/data/scenes/...`; if the host does not mount `/data`,
use `docker cp` into both containers or update the dataset mount before running
eval.

Avoid committing large binary USD replacements into
`arena_simulation_setup/worlds/grscenes_<id>/` unless the file is intentionally a
small self-contained fixture.  Production GRScenes assets should remain in the
external dataset mount.

## Scenario Files

Native Arena scenarios live under:

```text
arena_simulation_setup/worlds/grscenes_<id>/scenarios/<scenario_name>/
```

The active GRScenes scenarios are synchronized from:

```text
/home/ubuntu/arena_jazzy_ws/grscenes_/grscenes_<id>/scenarios/
```

into:

```text
arena_simulation_setup/worlds/grscenes_<id>/scenarios/
```

The current benchmark scenarios are named `default` and `default_1`.  Older
`uploaded_*` or `recorded_*` scenario directories should not be kept alongside
the synchronized set unless they are intentionally preserved as separate
benchmark cases.

A scenario directory usually contains:

```text
scenario.yaml
BTRegularNav.xml
```

`scenario.yaml` defines robot start/goal and HuNav pedestrians:

```yaml
dynamic:
- name: hunav_1
  pose: [-3.71374, -2.76407, 104.98163]
  behavior_tree: ./BTRegularNav.xml
  velocity: 0.82
  desired_velocity: 1.02
  waypoints:
  - [-3.93816, -1.92543, 158.1986]

robots:
- start: [-7.1864, 3.50799, 99.71574]
  goal: [-8.00142, 8.26814, 99.71574]
```

Keep `BTRegularNav.xml` in the same scenario directory and reference it with a
`./` prefix, for example `behavior_tree: ./BTRegularNav.xml`.  The HuNav adapter
resolves `./...` paths relative to the scenario directory, which makes the
scenario self-contained and avoids relying on a global behavior-tree lookup.

## Container Checks

Before running an Isaac eval, check that the configured USD paths exist in both
containers:

```sh
for scene in \
  MV4AFHQKTKJZ2AABAAAAADQ8_usd \
  MV4AFHQKTKJZ2AABAAAAAEA8_usd \
  MV5M25QKTKJZ2AABAAAAAAA8_usd \
  MV5M25QKTKJZ2AABAAAAAAI8_usd \
  MV5M25QKTKJZ2AABAAAAAEI8_usd
do
  docker exec arena-arena_jazzy_ws-arena-1 \
    test -f /data/scenes/commercial_scenes/scenes/${scene}/trimed_navigation.usd
  docker exec arena-arena_jazzy_ws-isaac-1 \
    test -f /data/scenes/commercial_scenes/scenes/${scene}/trimed_navigation.usd
done
```

ROS, colcon, eval, metrics, topic, service, and launch commands must run from
the Arena container, not the host shell.

## Updating A Scene

Use this workflow when replacing a GRScenes asset version:

1. Put the new USD into the external dataset scene directory, preferably as a
   new filename such as `trimed_navigation.usd`.
2. Open `arena_simulation_setup/worlds/grscenes_<id>/world.yaml`.
3. Set `usd_scene.path` to the container-visible absolute dataset path.
4. Keep `scale: 0.01` unless the replacement USD was explicitly re-authored in
   meters.
5. Sync the current HuNav episode data from
   `/home/ubuntu/arena_jazzy_ws/grscenes_/grscenes_<id>/scenarios/` into
   `worlds/grscenes_<id>/scenarios/`.
6. Keep or add scenario-local HuNav files under
   `worlds/grscenes_<id>/scenarios/<scenario_name>/`.
7. Run the full Docker + Isaac eval pipeline and inspect `sim_top_down.mp4` and
   `ego_debug_overlay.mp4` before treating the scene as benchmark-ready.
