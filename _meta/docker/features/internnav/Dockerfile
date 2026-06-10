ARG ROS_DISTRO=jazzy
FROM osrf/ros:${ROS_DISTRO}-desktop

ARG ROS_DISTRO=jazzy
ARG username=arena
ARG group=arena
ARG uid=1000
ARG gid=1000
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG ROS_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu

ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

RUN if [ -n "${ROS_APT_MIRROR}" ]; then \
        rm -f /etc/apt/sources.list.d/ros2.sources /etc/apt/sources.list.d/ros2.list && \
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros2-archive-keyring.gpg] ${ROS_APT_MIRROR} $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
            > /etc/apt/sources.list.d/ros2.list; \
    fi

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        python-is-python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        sudo \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN (getent passwd ${uid} && userdel -rf $(getent passwd ${uid} | cut -d: -f1)) 2>/dev/null || true && \
    (getent group ${gid} && groupdel -f $(getent group ${gid} | cut -d: -f1)) 2>/dev/null || true && \
    groupadd -g ${gid} ${group} && \
    useradd -m -u ${uid} -g ${gid} -s /bin/bash ${username} && \
    passwd -d ${username} && \
    passwd -d root && \
    groupadd -r sudo || true && \
    usermod -aG sudo ${username}

WORKDIR /opt/arena_ws
COPY src/Arena/_meta/docker/features/internnav/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /opt/internnav_venv && chown -R ${uid}:${gid} /opt/internnav_venv /home/${username}

USER ${username}
ENTRYPOINT ["/entrypoint.sh"]
