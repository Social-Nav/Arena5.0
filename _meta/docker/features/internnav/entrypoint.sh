#!/usr/bin/env bash
set -e

if [ -f "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" ]; then
    # shellcheck disable=SC1090
    source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
fi

if [ -f "/opt/arena_ws/install/local_setup.bash" ]; then
    # shellcheck disable=SC1091
    source "/opt/arena_ws/install/local_setup.bash"
fi

if [ -f "/opt/arena_ws/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "/opt/arena_ws/.env"
    set +a
fi

export ARENA_WS="${ARENA_WS:-/opt/arena_ws}"
export ARENA_WORKSPACE="${ARENA_WORKSPACE:-/opt/arena_ws}"
export ARENA_INTERNNAV_VENV="${ARENA_INTERNNAV_VENV:-/opt/internnav_venv}"
export ARENA_VLN_MODEL_PYTHON="${ARENA_VLN_MODEL_PYTHON:-${ARENA_INTERNNAV_VENV}/bin/python}"
export ARENA_INTERNNAV_PYTHON="${ARENA_INTERNNAV_PYTHON:-${ARENA_INTERNNAV_VENV}/bin/python}"
export PYTHONUNBUFFERED=1

exec "$@"
