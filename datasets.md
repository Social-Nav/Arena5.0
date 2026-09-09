# Datasets

Arena keeps the current `grscenes_*_v1` world definitions, maps, and episode
scenarios in Git. Large GRScenes USD assets remain outside the repository and are
mounted into both the Arena and Isaac containers.

The standard host layout is:

```text
<HOST_SCENES_DIR>/
└── commercial_scenes/
    ├── Materials/
    ├── models/
    └── scenes/
        └── <scene-id>_usd/
            └── start_result_navigation.usd
```

The workspace `.env` contains the absolute host path:

```bash
HOST_SCENES_DIR=/absolute/path/to/GRScenes-100
```

Docker mounts that directory at `/data/scenes`. The 30 current world configs
therefore resolve assets such as:

```text
/data/scenes/commercial_scenes/scenes/MWF4WLIKTIFZIAABAAAAACA8_usd/start_result_navigation.usd
```

Download the GRScenes commercial archive, overlay the Arena-cleaned navigation
USDs, and run all path checks by following the
[Installation guide](docs/installation.md#4-install-grscenes-commercial-assets).
Do not commit archives, extracted USD trees, cleaned binary USDs, or model
checkpoints to this repository. Keep the upstream `Materials`, `models`, texture,
and metadata trees in place so relative USD references remain valid.
