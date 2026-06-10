#!/bin/bash -i
set -e

export ARENA_REPO=${ARENA_REPO:-https://github.com/Social-Nav/Arena5.0.git}
export ARENA_BRANCH=${ARENA_BRANCH:-feat/internnav-eval-progress}
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

_RCFILE="${RCFILE:-$HOME/.$(ps -p $$ -o comm=)rc}"
if [ -z "${RCFILE+x}" ]; then
    echo "${_RCFILE}"
    if [ ! -f "${_RCFILE}" ] ; then
        echo RCFILE is not set, failed to autodetect
        echo pass manually using 'RCFILE=/my/rcfile sh install.sh'
        exit 1
    else
        echo "detected rcfile as ${_RCFILE}"
    fi
fi

# == read inputs ==
echo 'Configuring Arena...'

ARENA_WS_DIR=$(realpath "$(eval echo "$(read_default "Arena workspace directory" "${ARENA_WS_DIR:-~/arena5_ws}")")")
export ARENA_WS_DIR

echo "installing ${ARENA_REPO}:${ARENA_BRANCH} on ROS2 ${ARENA_ROS_DISTRO} to $ARENA_WS_DIR"
sudo echo 'confirmed'
mkdir -p "$ARENA_WS_DIR"
cd "$ARENA_WS_DIR"
sudo apt update # some deps require initial update on fresh systems

# == remove ros problems ==
files=$( (grep -l "/ros" /etc/apt/sources.list.d/* | grep -v "ros2") || echo '')

if [ -n "$files" ]; then
    echo "The following files can cause some problems to installer:"
    echo "$files"
    
    if [[ "$(read_default "Do you want to delete these files? (Y/n)" "Y")" =~ ^[Yy]$ ]]; then
        sudo rm -f $files
        echo "Deleted $files"
    fi
fi

# == python deps ==

# pyenv
if [ ! -d "$HOME/.pyenv" ] ; then
    rm -rf "$HOME/.pyenv"
    curl https://pyenv.run | "$(ps -p $$ -o comm=)"
    # shellcheck disable=SC2016
    {     echo 'export PYENV_ROOT="$HOME/.pyenv"';
        echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"';
        echo 'eval "$(pyenv init -)"';
    } >> "${_RCFILE}"
    
    # shellcheck source=/dev/null
    . "${_RCFILE}"
    
    # resourcing does not work in the same shell
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    which pyenv || (echo 'open a completely new shell'; exit 1)
fi

# UV
if ! which uv ; then
    echo "Installing uv...:"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    . "$HOME/.local/bin/env" # source uv
fi

# for building python
echo "Installing Python deps..."
sudo apt-get install -y build-essential python3-pip zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev tk-dev python3-dev g++-9 gcc-9

# == basic arena install ==
if [ ! -d "$ARENA_WS_DIR/src/Arena" ] ; then
    mkdir -p "$ARENA_WS_DIR/src/"
    git clone --branch "${ARENA_BRANCH}" --single-branch "${ARENA_REPO}" "$ARENA_WS_DIR/src/Arena"
fi
. "$ARENA_WS_DIR/src/Arena/_meta/tools/source"
ROSDEP=0 arena update

# == compile ros ==
sudo add-apt-repository universe -y
sudo apt-get update || echo 0
sudo apt-get install -y curl

echo "Installing tzdata...:"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get install -y tzdata libompl-dev
sudo dpkg-reconfigure --frontend noninteractive tzdata

# ROS
echo "Setting up ROS2 ${ARENA_ROS_DISTRO}..."

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Getting Packages
echo "Installing deps...:"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ffmpeg \
    libasio-dev \
    libtinyxml2-dev \
    libcunit1-dev \
    libpcl-dev \
    libboost-python-dev \
    python3-rosdep \
    python3-imageio \
    libgps-dev \
    graphicsmagick \
    libgraphicsmagick1-dev \
    nlohmann-json3-dev \
    libxtensor-dev \
    libceres-dev \
    libsuitesparse-dev \
    libxaw7-dev

# Check if the default ROS sources.list file already exists
ros_sources_list="/etc/ros/rosdep/sources.list.d/20-default.list"
if [[ -f "$ros_sources_list" ]]; then
    echo "rosdep appears to be already initialized"
    echo "Default ROS sources.list file already exists:"
    echo "$ros_sources_list"
else
    sudo rosdep init
fi

rosdep update --rosdistro "${ARENA_ROS_DISTRO}"

if [ ! -f src/ros2/compiled ] ; then
    # install ros2
    
    mkdir -p src/ros2
    curl "https://raw.githubusercontent.com/ros2/ros2/${ARENA_ROS_DISTRO}/ros2.repos" > ros2.repos
    vcs import src/ros2 < ros2.repos
    
    RTI_NC_LICENSE_ACCEPTED=yes \
    rosdep install \
        --from-paths src/ros2 \
        --ignore-src \
        --rosdistro "${ARENA_ROS_DISTRO}" \
        -y \
    || echo 'rosdep failed to install all dependencies'
    
    # fix rosidl error that was caused upstream https://github.com/ros2/rosidl/issues/822#issuecomment-2403368061
    pushd src/ros2/ros2/rosidl
        git -c user.name='Arena' -c user.email='anonymous@arena-rosnav.org' cherry-pick 654d6f5658b59009147b9fad9b724919633f38fe || echo 'already cherry picked'
    popd

    BASE_PATHS=src/ros2 BUILD_ALL=1 SKIP_OLD=0 arena build
    touch src/ros2/compiled
fi

# == install arena on top of ros2 ==

if [ ! -f "$INSTALLED" ] ; then
    # tmp
    # mv src/Arena src/.Arena
    
    # echo "cloning Arena..."
    # git clone --branch "${ARENA_BRANCH}" "${ARENA_REPO}" src/Arena

    # mv -n src/Arena/* src/.Arena/* src/Arena || true
    # rm -rf src/.Arena/

    arena update
fi

touch "$INSTALLED"

if [ ! -d /usr/local/include/lightsfm ] ; then
    git clone https://github.com/robotics-upo/lightsfm.git lightsfm
    ( (cd lightsfm && make && sudo make install) || rm -rf lightsfm)
    rm -rf lightsfm || echo 'failed to install lightsfm'
fi


BUILD_ALL=1 arena build
echo 'installation finished'
