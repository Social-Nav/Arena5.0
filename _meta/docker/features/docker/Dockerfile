ARG ROS_DISTRO=jazzy
FROM osrf/ros:${ROS_DISTRO}-desktop AS base
ARG APT_TIMEOUT_SEC=3600
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG ROS_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu
ARG PYTHON_BUILD_MIRROR_URL=https://mirrors.tuna.tsinghua.edu.cn/python
ARG PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=1
ENV PYTHON_BUILD_MIRROR_URL=${PYTHON_BUILD_MIRROR_URL}
ENV PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=${PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM}

RUN if [ -z "${HTTP_PROXY}${HTTPS_PROXY}${http_proxy}${https_proxy}" ]; then \
        echo "WARNING: no HTTP(S) proxy is configured; Docker build requires working internet access." >&2; \
    fi

RUN if [ -n "${ROS_APT_MIRROR}" ]; then \
        rm -f /etc/apt/sources.list.d/ros2.sources /etc/apt/sources.list.d/ros2.list && \
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros2-archive-keyring.gpg] ${ROS_APT_MIRROR} $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
            > /etc/apt/sources.list.d/ros2.list; \
    fi

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    timeout ${APT_TIMEOUT_SEC}s apt-get update && \
    timeout ${APT_TIMEOUT_SEC}s apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    wget \
    curl \
    python-is-python3 \
    python3-pip \
    python3-pandas \
    pipx \
    python3-colcon-common-extensions \
    python3-rosdep \
    libgps-dev \
    ros-${ROS_DISTRO}-fastcdr \
    ros-${ROS_DISTRO}-fastrtps \
    ros-${ROS_DISTRO}-fastrtps-cmake-module \
    ros-${ROS_DISTRO}-irobot-create-description \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-map-server \
    ros-${ROS_DISTRO}-nav2-amcl \
    ros-${ROS_DISTRO}-nav2-behaviors \
    ros-${ROS_DISTRO}-nav2-bt-navigator \
    ros-${ROS_DISTRO}-nav2-collision-monitor \
    ros-${ROS_DISTRO}-nav2-common \
    ros-${ROS_DISTRO}-nav2-controller \
    ros-${ROS_DISTRO}-nav2-dwb-controller \
    ros-${ROS_DISTRO}-nav2-lifecycle-manager \
    ros-${ROS_DISTRO}-nav2-navfn-planner \
    ros-${ROS_DISTRO}-nav2-planner \
    ros-${ROS_DISTRO}-nav2-smoother \
    ros-${ROS_DISTRO}-nav2-velocity-smoother \
    ros-${ROS_DISTRO}-nav2-waypoint-follower \
    ros-${ROS_DISTRO}-nav2-msgs \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
    ros-${ROS_DISTRO}-rmw-fastrtps-shared-cpp \
    ros-${ROS_DISTRO}-rosidl-dynamic-typesupport-fastrtps \
    ros-${ROS_DISTRO}-rosidl-typesupport-fastrtps-c \
    ros-${ROS_DISTRO}-rosidl-typesupport-fastrtps-cpp \
    ros-${ROS_DISTRO}-tf-transformations \
    ros-${ROS_DISTRO}-topic-tools \
    sudo

RUN pipx install git+https://github.com/voshch/vcstool.git

WORKDIR /opt/arena_ws

# install python deps
FROM base AS python_deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build
COPY src/Arena/pyproject.toml uv.lock* ./

RUN uv venv /opt/venv


# merge
FROM base AS final
ARG APT_TIMEOUT_SEC=3600
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG username=arena
ARG group=arena
ARG uid=1000
ARG gid=1000

COPY --from=python_deps --chown=${uid}:${gid} /opt/venv /opt/venv
COPY --from=python_deps /usr/local/bin/uv /usr/local/bin/uv


# install docker cli
RUN  --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    timeout ${APT_TIMEOUT_SEC}s apt-get update && \
    timeout ${APT_TIMEOUT_SEC}s apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
    && install -m 0755 -d /etc/apt/keyrings \
        && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
        && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && timeout ${APT_TIMEOUT_SEC}s apt-get update \
    && timeout ${APT_TIMEOUT_SEC}s apt-get install -y --no-install-recommends docker-ce-cli docker-buildx-plugin docker-compose-plugin

# install lightsfm
RUN git clone --depth 1 https://github.com/robotics-upo/lightsfm.git /lightsfm && \
    cd /lightsfm && make && make install



# set user perms
RUN (getent passwd ${uid} && userdel -rf $(getent passwd ${uid} | cut -d: -f1)) 2>/dev/null || true && \
    (getent group ${gid} && groupdel -f $(getent group ${gid} | cut -d: -f1)) 2>/dev/null || true && \
    groupadd -g ${gid} ${group} && \
    useradd -m -u ${uid} -g ${gid} -s /bin/bash ${username} && \
    passwd -d ${username} && \
    passwd -d root && \
    groupadd -r sudo || true && \
    usermod -aG sudo ${username}

    
WORKDIR /opt/arena_ws

RUN mkdir -p build
RUN mkdir -p install
RUN mkdir -p log
RUN chown -R ${uid}:${gid} /opt/arena_ws


COPY src/Arena/_meta/docker/features/docker/entrypoint.sh /entrypoint.sh
RUN ln -fs src/Arena/_meta/docker/features/docker/source.container ./source

RUN chmod +x /entrypoint.sh
USER ${username}
RUN if [ -z "${HTTP_PROXY}${HTTPS_PROXY}${http_proxy}${https_proxy}" ]; then \
        echo "WARNING: no HTTP(S) proxy is configured; pyenv install requires working internet access." >&2; \
    fi && \
    curl -fsSL https://pyenv.run | bash

ENTRYPOINT ["/entrypoint.sh"]


# cleanup
# RUN rm -rf /var/lib/apt/lists/*
