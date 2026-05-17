## Why

The social-navigation benchmark needs a repeatable scenario-level entrypoint that
binds Arena's native world/scenario files to language instructions, social
constraints, required artifacts, and aggregation.  Running `internnav_eval`
directly is powerful but makes it easy to omit `tm_obstacles=scenario`,
InternNav mode, social metric validation, or scenario metadata in the manifest.

## What Changes

- Add a benchmark-level Dynamic Social VLN scenario YAML validator and runner wrapper.
- Add a sample `hospital_1 + Ai2_Bot2 + HuNav + InternNav` scenario overlay.
- Add manifest fields for scenario config ID/path.
- Add an aggregate CLI that summarizes social-navigation runs and failure tags.
- Document how to validate scenarios, launch eval with videos, and use metrics/aggregation.

## Impact

- Affects `arena_bringup` CLI entrypoints and `internnav_eval.py` manifest schema.
- Preserves the existing `internnav_eval` launch contract; the scenario runner only derives arguments and appends user-provided overrides.
- Requires social scenario eval commands to be run inside the Arena Jazzy container for ROS 2 compatibility.
