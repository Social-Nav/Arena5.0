# --- Stage 1: Builder (Strict Python 3.11 + ROS Jazzy) ---
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

FROM osrf/ros:jazzy-desktop AS builder

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# 1. Setup Deadsnakes for Python 3.11 (Jazzy is native 3.12)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update

RUN apt-get install -y \
    build-essential cmake ffmpeg git curl \
    python3.11 python3.11-dev python3.11-venv \
    python3-colcon-common-extensions && rm -rf /var/lib/apt/lists/*

# 2. Create Venv and SET PATH (Forced Activation)
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# 3. STRICT CHECK: Fail build if version is not 3.11
RUN python --version && \
    python -c "import sys; assert sys.version_info[:2] == (3, 11), 'ERROR: Python version is not 3.11!'"

WORKDIR /opt/msg_build

ADD src/Arena/arena_isaac/isaacsim_msgs src/isaacsim_msgs
ADD src/deps/hunav/hunav_sim/hunav_msgs src/hunav_msgs
ADD src/Arena/utils/msgs/arena_people_msgs src/arena_people_msgs

# 4. Install build dependencies (Including NumPy for the C-extensions)
RUN . /opt/ros/jazzy/setup.sh && \
    pip install --upgrade pip && \
    pip install catkin_pkg empy==3.3.4 lark numpy

# 5. Build - Forcing the directory name to python3.11
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build \
    --packages-select isaacsim_msgs hunav_msgs arena_people_msgs \
    --cmake-args \
    -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.11.so.1.0 \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.11 \
    -DROSIDL_GENERATOR_PY_PYTHON_EXECUTABLE=/opt/venv/bin/python

# --- Stage 2: Final Isaac Sim Image ---
FROM nvcr.io/nvidia/isaac-sim:5.1.0

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# 6. Switch to root to fix system libs and paths
USER root

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install runtime Python dependencies directly into Isaac's internal site-packages to ensure they persist
RUN /isaac-sim/python.sh -m pip install \
    aiofiles \
    attrs \
    cattrs \
    imageio \
    numpy \
    pandas \
    pillow \
    pydantic \
    pyyaml \
    requests \
    scipy \
    shapely \
    typing_extensions && \
    /isaac-sim/python.sh - <<'PY'
import importlib

for module in (
    'aiofiles',
    'attrs',
    'cattrs',
    'imageio',
    'numpy',
    'pandas',
    'PIL',
    'pydantic',
    'requests',
    'scipy',
    'shapely',
    'typing_extensions',
    'yaml',
):
    importlib.import_module(module)

print('Isaac runtime Python dependency smoke check passed.')
PY

# Copy the libpython3.11 library from builder (missing in Isaac 5.1.0 base)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libpython3.11.so.1.0 /usr/lib/x86_64-linux-gnu/
RUN ldconfig

# Copy the built messages
COPY --from=builder /opt/msg_build/install /opt/isaac_bridge_msgs

# 7. Persistent Environment Setup
# We write to /root/.bashrc because Isaac Sim's launch scripts often wipe 'ENV' variables
RUN echo 'export ROS_DISTRO=jazzy' >> /root/.bashrc && \
    echo 'export ACCEPT_EULA=Y' >> /root/.bashrc && \
    echo 'export PRIVACY_CONSENT=Y' >> /root/.bashrc && \
    echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' >> /root/.bashrc && \
    echo 'export AMENT_PREFIX_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy:$AMENT_PREFIX_PATH' >> /root/.bashrc && \
    echo 'export ROS_DOMAIN_ID=1' >> /root/.bashrc && \
    echo 'export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/python:$PYTHONPATH' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy/:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib/:$LD_LIBRARY_PATH' >> /root/.bashrc

# Set standard ENV as well for non-interactive shells
ENV ROS_DISTRO=jazzy \
    AMENT_PREFIX_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy \
    PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/python \
    LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy/:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib/

RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python3 && chmod +x /usr/local/bin/python3
RUN printf '#!/bin/bash\nsource /opt/isaac_bridge_msgs/local_setup.bash > /dev/null 2>&1\nexec python3 "$@"\n' > /usr/local/bin/python3-arena && chmod +x /usr/local/bin/python3-arena

ENTRYPOINT [ "bash" ]
