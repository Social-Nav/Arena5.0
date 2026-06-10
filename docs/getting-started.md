# Getting Started

## Prerequisites

Before running Arena benchmarks, make sure you have:

- Ubuntu-based Linux environment
- Docker installed
- NVIDIA drivers and `nvidia-container-toolkit` if you plan to use Isaac Sim
- A ROS 2 workspace cloned with this repository under `src/Arena`

## Basic installation

The repository already provides installation helpers:

```bash
curl -fsSL https://raw.githubusercontent.com/Social-Nav/Arena5.0/feat/internnav-eval-progress/install.sh -o install.sh
bash install.sh
```

For host dependencies used by evaluation and video export, ensure the install flow includes:

- `ffmpeg`
- `python3-imageio`

These are required by the current H.264 benchmark video path.

## Optional feature installation

After the workspace has been initialized:

```bash
cd ~/arena_ws
source arena
arena feature isaac install
arena feature gazebo install
arena feature training install
```

## Isaac-specific note

Isaac Sim must use the message packages bundled inside the image at `/opt/isaac_bridge_msgs`.

Do **not** point the Isaac runtime to ad-hoc host-built Python 3.11 message overlays from another ROS distro. The current Arena integration assumes that Isaac uses the bridge-compatible image-bundled packages.

If you changed any of these message packages:

- `arena_isaac/isaacsim_msgs`
- `utils/msgs/arena_people_msgs`
- `hunav_msgs` used by the image build

rebuild the Isaac feature image before running again:

```bash
cd ~/arena_ws
source arena
arena feature isaac update
```

## Local documentation preview

This repository now includes an initial MkDocs site.

If you want to preview it locally:

```bash
cd /home/ubuntu/arena_jazzy_ws/src/Arena
uv run --group docs mkdocs serve
```

Or build it once:

```bash
cd /home/ubuntu/arena_jazzy_ws/src/Arena
uv run --group docs mkdocs build
```
