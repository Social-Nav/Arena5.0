# Arena-Rosnav

A modular ROS 2 (Humble) platform for researching and benchmarking autonomous robot navigation in 2D and 3D simulated environments. It supports classical planners (Nav2), deep-RL planners ([rosnav_rl](https://github.com/Arena-Rosnav/rosnav-rl)), and a variety of simulators (Gazebo, Isaac Sim).

---

## Installation

Preqeuisites: [Docker](https://docs.docker.com/engine/install/) installation with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support. Current user must be in group `docker`.
Afterwards, run the following commands to install Arena:

### Basic Installation

```sh
curl https://raw.githubusercontent.com/voshch/Arena/jazzy/install.sh > install.sh
bash install.sh
```
and follow the prompts. This will create a ROS 2 workspace at your target location and instruct you how to proceed (yellow text).


### Optional Features
```sh
cd ~/arena_ws # replace with your actual workspace path
source arena
arena feature isaac install # optional
arena feature gazebo install # optional
arena feature training install # optional
```

We recommend installing at least one simulator.

### Isaac Sim clean-clone notes

For Isaac Sim runs, two extra pieces are required beyond a normal workspace build:

1. `arena feature isaac install` builds the Isaac image and embeds Python 3.11 ROS bridge messages into `/opt/isaac_bridge_msgs` inside the Isaac container.
2. The host workspace may also need a Humble-compatible eval overlay at `install_humble_eval/` for packages that otherwise pull in binaries requiring a newer glibc than Ubuntu 22.04 provides.

If you change anything under `arena_isaac/isaacsim_msgs`, `utils/msgs/arena_people_msgs`, or `src/deps/hunav/hunav_sim/hunav_msgs`, rebuild the Isaac image before launching again:

```sh
cd ~/arena_ws # replace with your actual workspace path
source arena
arena feature isaac update
```

Do not point Isaac launch at host-built `install_py311_msgs/` artifacts from another ROS distro. Isaac must load the image-bundled `/opt/isaac_bridge_msgs` packages so that its Python 3.11 runtime, `rclpy`, and FastDDS/FastCDR libraries stay ABI-compatible.

## Usage

```sh
cd ~/arena_ws # replace with your actual workspace path
source arena
arena launch sim:=isaac                          # Isaac Sim
arena launch local_planner:=rosnav_rl agent_name:=<your_agent>  # DRL planner
arena launch sim:=gazebo local_planner:=rosnav_rl env_n:=2 train_config:=<path to config.yaml> # DRL training 
```

### DRL quick-start
Place your trained agent folder inside `Arena/arena_training/agents/<agent_name>/` (must contain `training_config.yaml` and `best_model.zip`), then launch with `local_planner:=rosnav_rl agent_name:=<agent_name>`. Refer to the [arena_training](arena_training/README.md) for training instructions.


## Troubleshooting

### Unknown runtime speficied 'nvidia'

```sh
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

### Isaac eval crashes or videos are not produced

- Ensure the Isaac feature image has been rebuilt after any message definition changes: `arena feature isaac update`.
- Ensure `ffmpeg` is installed on the host/container path; eval video export and H.264 verification depend on it via `arena_bringup/arena_bringup/dual_vln_eval.py:376`.
- If Isaac starts but crashes during ROS service creation, verify it is loading `/opt/isaac_bridge_msgs` from the container rather than stale host-side Python 3.11 message builds.
- On ROS 2 Jazzy, Nav2 plugin class names in YAML must use the `::` form (for example `nav2_navfn_planner::NavfnPlanner` and `nav2_behaviors::Spin`), otherwise planner or behavior server bringup can fail even when Isaac itself is healthy.
