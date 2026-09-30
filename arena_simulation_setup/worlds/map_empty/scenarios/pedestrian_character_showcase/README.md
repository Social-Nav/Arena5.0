# Pedestrian character showcase

This scenario is the Arena/HuNav acceptance scene for the curated ten-person
Isaac character pool. Each `model` in `scenario.yaml` is a stable key from
`pedestrian.simulator.logic.people.character_assets`.

The rows in the standalone review image are ordered left to right as follows:

- Back row: `female_medical_01`, `male_medical_01`,
  `female_business_01`, `female_casual_01`, `female_senior_01`.
- Front row: `male_casual_01`, `male_casual_02`, `male_senior_01`,
  `female_young_adult_01`, `male_young_adult_01`.

To select the scenario in a running `map_empty` Arena session:

```bash
ros2 param set /task_generator_node task.scenario.file pedestrian_character_showcase
```

Then trigger a task reset. The normal launch must use
`tm_robots:=scenario tm_obstacles:=scenario`.

For a faster asset-only regression check inside the Isaac container, run:

```bash
PYTHONPATH=/opt/arena_ws/src/Arena/arena_isaac/arena_isaac \
  /isaac-sim/python.sh \
  /opt/arena_ws/src/Arena/arena_isaac/arena_isaac/scripts/validate_character_showcase.py \
  --output-dir /opt/arena_ws/log/pedestrian_character_showcase
```

The script writes `validation.json` and an RGB image. First execution is slow
because DH assets trigger large USD/texture downloads and RTX shader caching.

To keep the Isaac Sim GUI open for interactive visual inspection, append
`--keep-open`. In this mode the people continuously shuttle left and right so
their walk cycles and turns can be inspected. Close the GUI window or press
`Ctrl+C` in the terminal to exit.

Scenario files may use one of the reviewed pools directly:

```yaml
model: pedestrian
model: doctor
model: police
model: construction_worker
```

The Isaac spawn service resolves a pool to a concrete identity (and a DH color
variant), uses each identity once before repeating it, and caches the result by
pedestrian name. Set `ARENA_PEDESTRIAN_MODEL_SEED` before launching Isaac for a
reproducible selection. Exact names such as `classic_female_medical_01` and
`dh_3809dbd8_v07` remain supported.
