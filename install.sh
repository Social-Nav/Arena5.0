#!/bin/bash -i
set -e

export ARENA_REPO=${ARENA_REPO:-https://github.com/Social-Nav/Arena5.0.git}
export ARENA_BRANCH=${ARENA_BRANCH:-feature/social_yielding}
export ARENA_ROS_DISTRO=${ARENA_ROS_DISTRO:-jazzy}

read_default(){
    local prompt=$1
    local default=$2
    local result
    
    if [[ -t 0 ]]; then
        read -rp "$prompt [$default]: " result
        echo "${result:-$default}"
    else
        echo "$default"
    fi
}

# == read inputs ==
echo 'Configuring Arena...'

ARENA_WS_DIR=$(realpath "$(eval echo "$(read_default "Arena workspace directory" "${ARENA_WS_DIR:-~/arena_ws}")")")
export ARENA_WS_DIR

echo "installing ${ARENA_REPO}:${ARENA_BRANCH} on ROS2 ${ARENA_ROS_DISTRO} to $ARENA_WS_DIR"
sudo echo 'confirmed'
mkdir -p "$ARENA_WS_DIR"
cd "$ARENA_WS_DIR"

# set up
mkdir -p src
if [ ! -d src/Arena ]; then
    git clone "$ARENA_REPO" -b "$ARENA_BRANCH" src/Arena
fi

ln -rsf "$ARENA_WS_DIR/src/Arena/_meta/docker/source" ./arena
ln -rsf "$ARENA_WS_DIR/src/Arena/_meta/tools/Arena.code-workspace" ./ws-arena.code-workspace

echo 'Building Arena...'
cd $ARENA_WS_DIR
printf 'exit\n' | source arena

echo 'Installed Arena'
echo 'run the following to get started:'
echo "  cd $ARENA_WS_DIR && source arena"