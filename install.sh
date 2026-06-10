#!/bin/bash -i
set -e

export ARENA_REPO=${ARENA_REPO:-https://github.com/Social-Nav/Arena5.0.git}
export ARENA_BRANCH=${ARENA_BRANCH:-feat/internnav-eval-progress}
export ARENA_ROS_DISTRO=${ARENA_ROS_DISTRO:-jazzy}

declare -a ARENA_OPTIONAL_FEATURES=()
ARENA_CLEAN_INSTALL=${ARENA_CLEAN_INSTALL:-1}
ARENA_VALIDATE_BOOTSTRAP=${ARENA_VALIDATE_BOOTSTRAP:-1}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-isaac)
            ARENA_OPTIONAL_FEATURES+=("isaac")
            shift
        ;;
        --install-internnav)
            ARENA_OPTIONAL_FEATURES+=("internnav")
            shift
        ;;
        --install-training)
            ARENA_OPTIONAL_FEATURES+=("training")
            shift
        ;;
        --install-feature)
            if [[ $# -lt 2 ]]; then
                echo "Missing feature name after --install-feature" >&2
                exit 1
            fi
            ARENA_OPTIONAL_FEATURES+=("$2")
            shift 2
        ;;
        --no-clean-install)
            ARENA_CLEAN_INSTALL=0
            shift
        ;;
        --skip-bootstrap-validation)
            ARENA_VALIDATE_BOOTSTRAP=0
            shift
        ;;
        --help|-h)
            cat <<'EOF'
Usage: install.sh [options]

Options:
  --install-isaac         Install the Isaac Docker feature after bootstrap.
  --install-internnav     Install the InternNav model runtime feature after bootstrap.
  --install-training      Install the training feature after bootstrap.
  --install-feature NAME  Install an arbitrary Arena feature after bootstrap.
  --no-clean-install      Reuse existing build/install/.venv state instead of forcing a clean bootstrap.
  --skip-bootstrap-validation  Skip post-bootstrap validation checks.
EOF
            exit 0
        ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
        ;;
    esac
done

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

if [ "$ARENA_CLEAN_INSTALL" = 1 ]; then
    echo 'Enforcing clean Arena bootstrap (build/install/log/.venv reset)'
    rm -rf "$ARENA_WS_DIR/build" "$ARENA_WS_DIR/install" "$ARENA_WS_DIR/log"
    rm -rf "$ARENA_WS_DIR/src/Arena/.venv"
fi

echo 'Building Arena...'
cd $ARENA_WS_DIR
bash -lc "source \"$ARENA_WS_DIR/arena\" -c 'arena build'"

if [ "$ARENA_VALIDATE_BOOTSTRAP" = 1 ]; then
    bash -lc "source \"$ARENA_WS_DIR/arena\" -c 'arena validate bootstrap'"
fi

if [ ${#ARENA_OPTIONAL_FEATURES[@]} -gt 0 ]; then
    echo 'Installing optional Arena features...'
    for feature in "${ARENA_OPTIONAL_FEATURES[@]}"; do
        echo "  - $feature"
        bash -lc "source \"$ARENA_WS_DIR/arena\" -c 'arena feature '$feature' install'"
    done
fi

if [ "$ARENA_VALIDATE_BOOTSTRAP" = 1 ]; then
    bash -lc "source \"$ARENA_WS_DIR/arena\" -c 'arena validate bootstrap'"
fi

echo 'Installed Arena'
echo 'run the following to get started:'
echo "  cd $ARENA_WS_DIR && source arena"
