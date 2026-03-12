# GRScenes World — Configuration & Usage Guide

---

## 1. Download the GRScenes-100 Dataset

The GRScenes dataset is **not** included in this repository. You must download
it separately.



After downloading, extract the archives. The expected directory structure is:

```
<GRSCENES_ROOT>/
├── commercial_scenes/
│   └── scenes/
│       ├── <SCENE_ID_1>/
│       │   ├── start_result_navigation.usd   ← used for navigation tasks
│       │   ├── start_result_interaction.usd
│       │   ├── start_result_raw.usd
│       │   ├── Materials/
│       │   ├── models/
│       │   ├── metadata.json
│       │   └── interactive_obj_list.json
│       ├── <SCENE_ID_2>/
│       └── ...
└── home_scenes/
    └── scenes/
        └── ...
```

Scene IDs look like: `MV4AFHQKTKJZ2AABAAAAADQ8_usd`

---

## 2. Set the USD Scene Path in `world.yaml`

Open [`world.yaml`](world.yaml) and update the `path` field to point to your
local copy of the GRScenes dataset:

```yaml
world_type: "usd"

usd_scene:
  # ↓ Replace this with your actual path
  path: "/path/to/GRScenes-100/commercial_scenes/scenes/<SCENE_ID>/start_result_navigation.usd"
  scale: 0.01          # DO NOT change — converts cm → meters
  position: [0.0, 0.0, 0.0]
  orientation: [0.0, 0.0, 0.0, 1.0]
```
---

## 3. Directory Structure of This World

```
grscenes_test/
├── world.yaml
├── map/
│   ├── map.yaml
│   └── map.png
└── scenarios/
    └── default/
        ├── scenario.yaml
        ├── hunav_1_behavior_tree.xml
        ├── hunav_2_behavior_tree.xml
        └── hunav_3_behavior_tree.xml
```



### `map/map.png`
The current file is **entirely white**, meaning
the whole area is declared free space. This is intentional for USD/GRScenes
worlds: the actual walls and furniture are inside the USD file and provide
physical collision via PhysX — the 2D map is only needed to give the nav2
global planner a reference frame and rough boundary, not to encode wall geometry.

If you want nav2 to plan around walls (e.g. for a different planner), you would
need to replace `map.png` with an actual top-down occupancy image of the scene.


### `scenarios/default/hunav_N_behavior_tree.xml`
Per-pedestrian BehaviorTree XML files. Currently **not used** by `scenario.yaml`
(which uses the shared `BTRegularNav.xml` from `arena_simulation_setup` instead).
They are kept here as templates — you can reference them via the
`behavior_tree:` field in `scenario.yaml` if you need custom per-agent logic.

---

## 4. Launch the Simulation

```sh
cd ~/arena5_ws          # replace with your actual workspace path
source arena

arena launch sim:=isaac world:=grscenes_test
```

## 5. Adjust Pedestrian Spawn Poses & Waypoints

Edit [`scenarios/default/scenario.yaml`](scenarios/default/scenario.yaml) to
change where pedestrians start and walk to.

All coordinates are in **meters** after the `scale: 0.01` transform.

```yaml
dynamic:
  - name: hunav_1
    model: female_adult_business_02  
    pose: [1.0, -4.0, 0.0]           #spawn point [x, y, yaw_degrees]
    behavior_tree: BTRegularNav.xml
    velocity: 0.8
    desired_velocity: 1.0
    waypoints:
      - [5.0, -3.0, 0.0]
      - [1.0, -8.0, 0.0]
```

