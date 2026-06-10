# Arena5.0

A modular ROS 2 Jazzy platform for researching and benchmarking autonomous robot navigation in 2D and 3D simulated environments. It supports classical planners (Nav2), deep-RL planners ([rosnav_rl](https://github.com/Arena-Rosnav/rosnav-rl)), and a variety of simulators (Gazebo, Isaac Sim).

---

## Installation

Prerequisites: [Docker](https://docs.docker.com/engine/install/) installation with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support. Current user must be in group `docker`.

The commands below assume the workspace is mounted at `~/arena_ws`.  Network-related environment variables are forwarded from the host into the main Arena container and into feature containers, so set proxy and mirror variables **before** sourcing `arena` or running feature installs.

### Network proxy and mirrors

If your network requires a proxy, set both upper-case and lower-case proxy variables:

```sh
export HTTP_PROXY=http://100.68.161.151:3128
export HTTPS_PROXY=http://100.68.161.151:3128
export NO_PROXY=localhost,127.0.0.1,::1
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export no_proxy="$NO_PROXY"
```

To use the Tsinghua ROS 2 mirror and the Tsinghua Python source mirror during clean builds:

```sh
export ROS_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu
export PYTHON_BUILD_MIRROR_URL=https://mirrors.tuna.tsinghua.edu.cn/python
export PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=1
```

To return to direct network access and the default upstream mirrors, unset them before re-running install/update commands:

```sh
unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy
unset ROS_APT_MIRROR PYTHON_BUILD_MIRROR_URL PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM
```

Afterwards, run the following commands to install Arena:

### Basic Installation

```sh
curl -fsSL https://raw.githubusercontent.com/Social-Nav/Arena5.0/feat/internnav-eval-progress/install.sh -o install.sh
bash install.sh
```
and follow the prompts. This will create a ROS 2 workspace at your target location and instruct you how to proceed (yellow text).


### Optional Features
```sh
cd ~/arena_ws # replace with your actual workspace path
source arena
arena update
arena feature isaac install # optional
arena feature gazebo install # optional
arena feature training install # optional
```

We recommend installing at least one simulator.

After installation, run the minimum workspace check from the host shell:

```sh
cd ~/arena_ws
source arena
arena minimum-test --include-installed-features --with-robot-launch --robot Ai2_Bot2 --world hospital_1 --launch-timeout 30
```

### Isaac Sim clean-clone notes

For Isaac Sim runs, two extra pieces are required beyond a normal workspace build:

1. `arena feature isaac install` builds the Isaac image and embeds Python 3.11 ROS bridge messages into `/opt/isaac_bridge_msgs` inside the Isaac container.
2. Isaac launches execute inside the Isaac feature container and use `/opt/arena_ws` paths, so keep the workspace mounted through the standard Arena Docker flow instead of pointing Isaac at host-only build artifacts.

If you change anything under `arena_isaac/isaacsim_msgs`, `utils/msgs/arena_people_msgs`, or `src/deps/hunav/hunav_sim/hunav_msgs`, rebuild the Isaac image before launching again:

```sh
cd ~/arena_ws # replace with your actual workspace path
source arena
arena feature isaac update
```

Do not point Isaac launch at host-built message overlays from another ROS distro. Isaac must load the image-bundled `/opt/isaac_bridge_msgs` packages so that its Python 3.11 runtime, `rclpy`, and FastDDS/FastCDR libraries stay ABI-compatible.

## Usage

```sh
cd ~/arena_ws # replace with your actual workspace path
source arena
arena launch sim:=isaac                          # Isaac Sim
arena launch local_planner:=rosnav_rl agent_name:=<your_agent>  # DRL planner
arena launch sim:=gazebo local_planner:=rosnav_rl env_n:=2 train_config:=<path to config.yaml> # DRL training 
```

### Minimum Isaac smoke test

For the minimal Isaac + `hospital_1` + `Ai2_Bot2` validation, start from the workspace and run the launch through the Arena container:

```sh
cd ~/arena_ws
source arena
arena feature isaac install

arena launch \
  sim:=isaac \
  robot:=Ai2_Bot2 \
  world:=hospital_1 \
  headless:=2 \
  tm_robots:=scenario \
  episodes:=1 \
  timeout:=10 \
  timeout_wall_sec:=45.0 \
  auto_reset:=false \
  local_planner:=dwb \
  global_planner:=navfn
```

`tm_robots:=scenario` is required for this smoke test.  The default `tm_robots:=explore` uses the demo robot setup, which currently spawns `jackal` robots instead of the requested `Ai2_Bot2`.

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
- Ensure `ffmpeg` is installed on the host/container path; eval video export and H.264 verification depend on it via `arena_bringup/arena_bringup/internnav_eval.py:376`.
- If Isaac starts but crashes during ROS service creation, verify it is loading `/opt/isaac_bridge_msgs` from the container rather than stale host-side Python 3.11 message builds.
- On ROS 2 Jazzy, Nav2 plugin class names in YAML must use the `::` form (for example `nav2_navfn_planner::NavfnPlanner` and `nav2_behaviors::Spin`), otherwise planner or behavior server bringup can fail even when Isaac itself is healthy.
