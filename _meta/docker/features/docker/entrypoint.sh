#!/bin/bash

source ~/.bashrc

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init - bash)"
else
    echo "pyenv not found; continuing with container default python" >&2
fi

cd /opt/arena_ws
source source

set -e

export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf"
export GIT_CONFIG_VALUE_0="git@github.com:"
export GIT_CONFIG_KEY_1="url.https://github.com/.insteadOf"
export GIT_CONFIG_VALUE_1="ssh://git@github.com/"

if [ ! -f /.built ]; then
    echo "Running initial setup..."
    if [ -f /opt/arena_ws/src/Arena/.gitmodules ]; then
        echo "Initializing git submodules..."
        if [ -z "${HTTP_PROXY}${HTTPS_PROXY}${http_proxy}${https_proxy}" ]; then
            echo "WARNING: no HTTP(S) proxy is configured; git submodule update requires working internet access." >&2
        fi
        git -C /opt/arena_ws/src/Arena submodule sync
        git -C /opt/arena_ws/src/Arena submodule update --init
        if [ -f /opt/arena_ws/src/Arena/arena_robots/.gitmodules ]; then
            git -C /opt/arena_ws/src/Arena/arena_robots submodule sync
            # Reproduce the exact dependency revisions recorded by the
            # arena_robots superproject.  --remote makes a clean bootstrap
            # depend on mutable (and sometimes deleted) branch names.
            git -C /opt/arena_ws/src/Arena/arena_robots \
                submodule update --init \
                deps/jackal \
                deps/turtlebot4 \
                deps/robot_usds
        fi
    fi
    arena update
    BUILD_ALL=1 arena build
    sudo touch /.built
    echo 'Initial setup complete.'
    echo -e '\033[0mRun \033[01;33marena feature docker commit\033[0m to save this state.'
fi

set +e

exec "$@"
