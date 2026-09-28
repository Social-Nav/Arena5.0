# GRScenes reference trajectory lengths

`reference_trajectory_lengths.csv.xlsx` is the source table supplied for the
selected social-navigation evaluation episodes. The checked-in workbook has
SHA-256:

```text
4f7d00d34e317a7d86e60bfaa9356de83eb5e0cd2e74e70536442cc3702baac9
```

Despite the expected 150 episodes (30 worlds x 5 scenarios), this workbook has
134 physical rows: one header row and 133 episode data rows. Import it with:

```bash
ros2 run arena_simulation_setup import_reference_trajectory_lengths \
  <share>/arena_simulation_setup/reference_data/reference_trajectory_lengths.csv.xlsx \
  --worlds-dir <source>/arena_simulation_setup/worlds
```

The importer writes one `episode_metadata.json` beside each matching
`scenario.yaml` and rewrites the package-root `test_split.json` as a plain JSON
array of those 133 relative metadata paths. `benchmark/main full-eval` treats
that split as its only allowlist, so the remaining 17 standard GRScenes cases
are skipped. The VLN metric uses `reference_trajectory.length_m` as the SPL
reference length while retaining the occupancy-grid A* polyline for nDTW/sDTW.
Scenarios without an annotation continue to report the explicit
`occupancy_grid_astar` SPL denominator source.

The importer preserves `robot_scale` as provenance and does not apply another
scale factor: `length_m` is the supplied metric-space value. As a defensive
metric invariant, if an imported value is shorter than the scenario start-goal
Euclidean distance, the effective SPL reference length is clamped to that lower
bound and the output source is suffixed with
`clamped_to_endpoint_distance`. The original value remains visible as
`reference_path.annotated_length_m`.
