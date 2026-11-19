#!/usr/bin/env bash
# Single-GPU local Arena collection wrapper.
#
# This script stays on the host. By default it reuses the workspace's existing
# Arena/Isaac Compose project (the same one `source arena` uses), starts/ensures
# the Arena service, then runs the per-world sequential collector inside it.
# Set REUSE_EXISTING_PROJECT=false to use the old isolated-project behaviour.
#
# The underlying sequential collector owns resume state, so interrupting and
# rerunning this script skips completed worlds and completed scenarios.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$(cd "$ARENA_DIR/../.." && pwd)"
COMPOSE_FILE="$ARENA_DIR/_meta/docker/docker-compose.yaml"
SEQUENTIAL_SCRIPT="/opt/arena_ws/src/Arena/scripts/colleague_collect/collect_worlds_sequentially.sh"

GPU_ID="${GPU_ID:-0}"
PROJECT_PREFIX="${PROJECT_PREFIX:-local-4090-collect}"
REUSE_EXISTING_PROJECT="${REUSE_EXISTING_PROJECT:-true}"
EXISTING_PROJECT_NAME="${EXISTING_PROJECT_NAME:-arena-$(basename "$WORKSPACE")}"
DOMAIN_BASE="${DOMAIN_BASE:-180}"
COLLECT_ARENA_IMAGE="${COLLECT_ARENA_IMAGE:-arena:dev}"
COLLECT_ROS_DISTRO="${COLLECT_ROS_DISTRO:-jazzy}"

ROBOT="${ROBOT:-Ai2_Bot2}"
KEEP_RAW="${KEEP_RAW:-false}"
SKIP_PREVIEW_VIDEO="${SKIP_PREVIEW_VIDEO:-true}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-}"
RETRIES="${RETRIES:-2}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-true}"
READY_TIMEOUT="${READY_TIMEOUT:-1500}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-20}"
ARENA_RESET_SERVICE_TIMEOUT="${ARENA_RESET_SERVICE_TIMEOUT:-1500}"

MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
STOP_FREE_MIB="${STOP_FREE_MIB:-2500}"
MONITOR_SECONDS="${MONITOR_SECONDS:-15}"

DEFAULT_SCENARIOS_CSV="default,default_1,default_2,default_3,default_4"
SCENARIOS_CSV="${SCENARIOS_CSV:-$DEFAULT_SCENARIOS_CSV}"
SCENARIO_GLOB="${SCENARIO_GLOB:-}"
DATASET_ROOT="${ARENA_DATASET_ROOT:-src/social_gen/traj_data/grscenes}"
DRY_RUN=false
CLEANUP_FIRST=false
MANIFEST_FILES=()
WORLD_LIST=""
RANGE_GIVEN=false
WORLDS_GIVEN=false

declare -A WORLD_SCENARIOS=()
WORLD_ORDER=()

usage() {
    cat <<'EOF'
Usage:
  bash src/Arena/scripts/colleague_collect/collect_worlds_local.sh [options]

Options:
  --manifest FILE       Two-column world/scenario file; may be repeated to merge manifests
  --range FIRST LAST    Collect grscenes_FIRST through grscenes_LAST
  --worlds LIST         Whitespace-separated world names
  --scenarios LIST      Comma-separated scenario names
  --scenario-glob GLOB  Discover matching scenario directories in every world
  --dataset-root PATH   Dataset root inside the container
  --instruction-type TYPE
                        Assign this fixed instruction to every episode after postprocess.
                        Use 'auto' to infer per-episode type from scenario names.
  --keep-raw            Keep raw JSON/NPY after postprocess (default: false)
  --no-keep-raw         Delete raw JSON/NPY after successful postprocess
  --skip-preview-video  Skip RGB preview MP4 generation (default: true)
  --no-skip-preview-video
                        Generate RGB preview MP4 (requires ffmpeg)
  --gpu ID              Local NVIDIA GPU id (default 0)
  --cleanup-first       Remove the previous Compose project before starting
                        (only applies when REUSE_EXISTING_PROJECT=false)
  --dry-run             Validate and print the plan without starting anything
  -h, --help            Show this help

Environment controls:
  GPU_ID=0              Local GPU id
  PROJECT_PREFIX=local-4090-collect
  REUSE_EXISTING_PROJECT=true  Reuse the existing arena/isaac project
  DOMAIN_BASE=180       ROS domain for the isolated worker
  RETRIES=2             Retries after the first scenario attempt
  CONTINUE_ON_FAIL=true Continue with remaining scenarios/worlds after failures
  MIN_FREE_MIB=9000     Refuse to start below this free VRAM

Examples:
  # Use selection.txt, one world and all listed scenarios.
  bash src/Arena/scripts/colleague_collect/collect_worlds_local.sh \
    --manifest selection.txt \
    --dataset-root /opt/arena_ws/src/social_gen/traj_data/grscenes_10k

  # Collect default scenarios for grscenes_1 through grscenes_10.
  bash src/Arena/scripts/colleague_collect/collect_worlds_local.sh \
    --range 1 10 --scenarios default

  # Collect all scenarios matching gpt56_* in every world 1..30.
  bash src/Arena/scripts/colleague_collect/collect_worlds_local.sh \
    --range 1 30 --scenario-glob 'gpt56_*' \
    --dataset-root /opt/arena_ws/src/social_gen/traj_data/grscenes_10k_gpt56

  # Human-interaction batch: skip preview MP4, delete raw data, assign one fixed instruction.
  bash src/Arena/scripts/colleague_collect/collect_worlds_local.sh \
    --manifest selection_perpendicular.txt \
    --dataset-root /opt/arena_ws/src/human_interaction/traj_data/grscenes \
    --instruction-type perpendicular_traffic

  # Merge several interaction-type manifests into one batch.
  bash src/Arena/scripts/colleague_collect/collect_worlds_local.sh \
    --manifest selection_perpendicular.txt \
    --manifest selection_following.txt \
    --manifest selection_overtaking.txt \
    --dataset-root /opt/arena_ws/src/human_interaction/traj_data/grscenes
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

safe_tag() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

while (($#)); do
    case "$1" in
        --manifest)
            (($# >= 2)) || die "--manifest requires a value"
            MANIFEST_FILES+=("$2")
            shift 2
            ;;
        --range)
            (($# >= 3)) || die "--range requires FIRST LAST"
            FIRST_WORLD="$2"
            LAST_WORLD="$3"
            RANGE_GIVEN=true
            shift 3
            ;;
        --worlds)
            (($# >= 2)) || die "--worlds requires a value"
            WORLD_LIST="$2"
            WORLDS_GIVEN=true
            shift 2
            ;;
        --scenarios)
            (($# >= 2)) || die "--scenarios requires a value"
            SCENARIOS_CSV="$2"
            shift 2
            ;;
        --scenario-glob)
            (($# >= 2)) || die "--scenario-glob requires a value"
            SCENARIO_GLOB="$2"
            shift 2
            ;;
        --dataset-root)
            (($# >= 2)) || die "--dataset-root requires a value"
            DATASET_ROOT="$2"
            shift 2
            ;;
        --instruction-type)
            (($# >= 2)) || die "--instruction-type requires a value"
            INSTRUCTION_TYPE="$2"
            shift 2
            ;;
        --keep-raw)
            KEEP_RAW=true
            shift
            ;;
        --no-keep-raw)
            KEEP_RAW=false
            shift
            ;;
        --skip-preview-video)
            SKIP_PREVIEW_VIDEO=true
            shift
            ;;
        --no-skip-preview-video)
            SKIP_PREVIEW_VIDEO=false
            shift
            ;;
        --gpu)
            (($# >= 2)) || die "--gpu requires a value"
            GPU_ID="$2"
            shift 2
            ;;
        --cleanup-first)
            CLEANUP_FIRST=true
            shift
            ;;
        --isolated)
            REUSE_EXISTING_PROJECT=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

is_uint "$GPU_ID" || die "GPU id must be an integer"
is_uint "$DOMAIN_BASE" || die "DOMAIN_BASE must be an integer"
is_uint "$MIN_FREE_MIB" && is_uint "$STOP_FREE_MIB" || die "VRAM thresholds must be integers"
((STOP_FREE_MIB < MIN_FREE_MIB)) || die "STOP_FREE_MIB must be below MIN_FREE_MIB"
[[ -n "$SCENARIOS_CSV" ]] || die "scenario list must not be empty"
[[ -z "$SCENARIO_GLOB" || "$SCENARIOS_CSV" == "$DEFAULT_SCENARIOS_CSV" ]] || \
    die "--scenario-glob cannot be combined with a custom --scenarios list"

if ((${#MANIFEST_FILES[@]} > 0)); then
    [[ "$RANGE_GIVEN" == false && "$WORLDS_GIVEN" == false ]] || \
        die "--manifest cannot be combined with --range or --worlds"
    for manifest_file in "${MANIFEST_FILES[@]}"; do
        [[ -f "$manifest_file" ]] || die "manifest file not found: $manifest_file"
        manifest_file="$(realpath "$manifest_file")"
        line_number=0
        while IFS= read -r line || [[ -n "$line" ]]; do
            line_number=$((line_number + 1))
            line="${line%$'\r'}"
            [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
            read -r world_name scenario_name extra <<<"$line"
            [[ -n "$world_name" && -n "$scenario_name" && -z "$extra" ]] || \
                die "$manifest_file line $line_number must contain exactly WORLD SCENARIO"
            [[ "$world_name" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || \
                die "invalid world name on $manifest_file line $line_number: $world_name"
            [[ "$scenario_name" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || \
                die "invalid scenario name on $manifest_file line $line_number: $scenario_name"
            [[ -f "$ARENA_DIR/arena_simulation_setup/worlds/$world_name/scenarios/$scenario_name/scenario.yaml" ]] || \
                die "scenario.yaml not found: $world_name/$scenario_name"
            if [[ -z "${WORLD_SCENARIOS[$world_name]:-}" ]]; then
                WORLD_ORDER+=("$world_name")
                WORLD_SCENARIOS[$world_name]="$scenario_name"
            elif [[ ",${WORLD_SCENARIOS[$world_name]}," != *",$scenario_name,"* ]]; then
                WORLD_SCENARIOS[$world_name]+=",$scenario_name"
            fi
        done < "$manifest_file"
    done
    ((${#WORLD_ORDER[@]} > 0)) || die "manifest contains no selected scenarios"
    SELECTION_TAG="manifest-$(safe_tag "$(
        for world_name in "${WORLD_ORDER[@]}"; do
            printf '%s:%s\n' "$world_name" "${WORLD_SCENARIOS[$world_name]}"
        done | LC_ALL=C sort | sha256sum | cut -c1-16
    )")"
else
    if [[ "$RANGE_GIVEN" == true ]]; then
        is_uint "$FIRST_WORLD" && is_uint "$LAST_WORLD" || die "world range must be integers"
        ((FIRST_WORLD >= 1 && FIRST_WORLD <= LAST_WORLD && LAST_WORLD <= 30)) || \
            die "world range must satisfy 1 <= FIRST <= LAST <= 30"
        for ((i = FIRST_WORLD; i <= LAST_WORLD; ++i)); do
            WORLD_ORDER+=("grscenes_${i}")
        done
    elif [[ "$WORLDS_GIVEN" == true ]]; then
        read -r -a WORLD_ORDER <<<"$WORLD_LIST"
    else
        die "provide --manifest, --range, or --worlds"
    fi

    for world_name in "${WORLD_ORDER[@]}"; do
        [[ "$world_name" =~ ^grscenes_[0-9]+$ ]] || die "expected grscenes_N world name: $world_name"
        [[ -d "$ARENA_DIR/arena_simulation_setup/worlds/$world_name/scenarios" ]] || \
            die "world directory not found: $world_name"
        if [[ -n "$SCENARIO_GLOB" ]]; then
            WORLD_SCENARIOS[$world_name]="__glob__"
        else
            WORLD_SCENARIOS[$world_name]="$SCENARIOS_CSV"
        fi
    done

    if [[ -n "$SCENARIO_GLOB" ]]; then
        SELECTION_TAG="glob-$(safe_tag "$SCENARIO_GLOB")"
    else
        SELECTION_TAG="scenarios-$(safe_tag "$SCENARIOS_CSV")"
    fi
fi

DATASET_TAG="$(safe_tag "${DATASET_ROOT##*/}")"
STATE_TAG="${SELECTION_TAG}-dataset-${DATASET_TAG}"
if [[ "$REUSE_EXISTING_PROJECT" == "true" ]]; then
    PROJECT_NAME="$EXISTING_PROJECT_NAME"
else
    PROJECT_NAME="${PROJECT_PREFIX}-gpu${GPU_ID}"
fi
STATE_SCOPE="$PROJECT_NAME"
HOST_STATE_ROOT="$WORKSPACE/log/local_collect/state/${STATE_SCOPE}/${STATE_TAG}"
CONTAINER_STATE_ROOT="/opt/arena_ws/log/local_collect/state/${STATE_SCOPE}/${STATE_TAG}"
COLLECTED_STATE_FILE="$CONTAINER_STATE_ROOT/collected_worlds.txt"
COMPLETED_STATE_FILE="$CONTAINER_STATE_ROOT/completed_worlds.txt"
SCENARIO_STATE_ROOT="$CONTAINER_STATE_ROOT/scenario_state"
SCENARIO_TIMING_ROOT="$CONTAINER_STATE_ROOT/timings"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
HOST_RUN_LOG="$WORKSPACE/log/local_collect/$RUN_ID"
CONTAINER_LOG_ROOT="/opt/arena_ws/log/local_collect/$RUN_ID"

exec_args=(arena-worker)
for world_name in "${WORLD_ORDER[@]}"; do
    exec_args+=("$world_name" "${WORLD_SCENARIOS[$world_name]}")
done

gpu_free_mib() {
    nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits \
        | tr -d '[:space:]'
}

print_plan() {
    echo "Workspace: $WORKSPACE"
    echo "GPU: $GPU_ID"
    echo "Compose project: $PROJECT_NAME"
    echo "ROS domain: $DOMAIN_BASE"
    echo "Dataset root: $DATASET_ROOT"
    echo "State root: $HOST_STATE_ROOT"
    echo "Run log: $HOST_RUN_LOG"
    for world_name in "${WORLD_ORDER[@]}"; do
        echo "  $world_name: ${WORLD_SCENARIOS[$world_name]}"
    done
    echo "KEEP_RAW=$KEEP_RAW SKIP_PREVIEW_VIDEO=$SKIP_PREVIEW_VIDEO INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-<none>} RETRIES=$RETRIES CONTINUE_ON_FAIL=$CONTINUE_ON_FAIL"
}

print_plan
command -v docker >/dev/null || die "docker is not available"
command -v nvidia-smi >/dev/null || die "nvidia-smi is not available"
[[ -f "$COMPOSE_FILE" ]] || die "compose file not found: $COMPOSE_FILE"
[[ -f "$WORKSPACE/.env" ]] || die "workspace .env not found: $WORKSPACE/.env"

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
((GPU_ID < gpu_count)) || die "GPU $GPU_ID does not exist (found $gpu_count GPUs)"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
export HOST_UID HOST_GID

compose() {
    env \
        ARENA_PROJECT_NAME="$PROJECT_NAME" \
        ARENA_IMAGE="$COLLECT_ARENA_IMAGE" \
        ARENA_GPU_ID="$GPU_ID" \
        ROS_DOMAIN_ID="$DOMAIN_BASE" \
        ARENA_ROS_DISTRO="$COLLECT_ROS_DISTRO" \
        HOST_UID="$HOST_UID" \
        HOST_GID="$HOST_GID" \
        HOST_ARENA_WS_DIR="$WORKSPACE" \
        ARENA_DATASET_ROOT="$DATASET_ROOT" \
        docker compose \
            -f "$COMPOSE_FILE" \
            --project-directory "$WORKSPACE" \
            --project-name "$PROJECT_NAME" \
            "$@"
}

if [[ "$REUSE_EXISTING_PROJECT" != "true" && "$CLEANUP_FIRST" == true ]]; then
    compose down --remove-orphans >/dev/null 2>&1 || true
fi

if [[ "$REUSE_EXISTING_PROJECT" != "true" ]] && compose ps -aq | grep -q .; then
    die "existing Compose project found: $PROJECT_NAME; use --cleanup-first"
fi

free="$(gpu_free_mib)"
if ((free < MIN_FREE_MIB)); then
    die "GPU $GPU_ID has ${free}MiB free; need ${MIN_FREE_MIB}MiB"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN: no containers started."
    exit 0
fi

mkdir -p "$HOST_STATE_ROOT" "$HOST_RUN_LOG"
for world_name in "${WORLD_ORDER[@]}"; do
    mkdir -p "$HOST_RUN_LOG/$world_name"
    touch "$HOST_RUN_LOG/$world_name/launch.log"
done
compose up -d --remove-orphans arena

worker_pid=""
launch_tail_pid=""
cleanup() {
    local status=$?
    if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
        kill -INT "$worker_pid" 2>/dev/null || true
    fi
    if [[ -n "$launch_tail_pid" ]] && kill -0 "$launch_tail_pid" 2>/dev/null; then
        kill "$launch_tail_pid" 2>/dev/null || true
    fi
    if [[ "$REUSE_EXISTING_PROJECT" != "true" ]]; then
        compose down --remove-orphans >/dev/null 2>&1 || true
    fi
    return "$status"
}
trap cleanup EXIT INT TERM

compose exec -T \
    -e SEQUENTIAL_SCRIPT="$SEQUENTIAL_SCRIPT" \
    -e DEFAULT_SCENARIOS_CSV="$DEFAULT_SCENARIOS_CSV" \
    -e ROBOT="$ROBOT" \
    -e KEEP_RAW="$KEEP_RAW" \
    -e SKIP_PREVIEW_VIDEO="$SKIP_PREVIEW_VIDEO" \
    -e INSTRUCTION_TYPE="$INSTRUCTION_TYPE" \
    -e RETRIES="$RETRIES" \
    -e CONTINUE_ON_FAIL="$CONTINUE_ON_FAIL" \
    -e READY_TIMEOUT="$READY_TIMEOUT" \
    -e COOLDOWN_SECONDS="$COOLDOWN_SECONDS" \
    -e SCENARIO_GLOB="$SCENARIO_GLOB" \
    -e ARENA_RESET_SERVICE_TIMEOUT="$ARENA_RESET_SERVICE_TIMEOUT" \
    -e ARENA_DATASET_ROOT="$DATASET_ROOT" \
    -e BATCH_LOG_ROOT="$CONTAINER_LOG_ROOT" \
    -e COLLECTED_STATE_FILE="$COLLECTED_STATE_FILE" \
    -e COMPLETED_STATE_FILE="$COMPLETED_STATE_FILE" \
    -e SCENARIO_STATE_ROOT="$SCENARIO_STATE_ROOT" \
    -e SCENARIO_TIMING_ROOT="$SCENARIO_TIMING_ROOT" \
    arena /entrypoint.sh bash -lc '
        overall=0
        while (($#)); do
            world="$1"
            scenarios="$2"
            shift 2
            if [[ "$scenarios" == "__glob__" ]]; then
                export SCENARIOS_CSV="$DEFAULT_SCENARIOS_CSV"
                export SCENARIO_GLOB="${SCENARIO_GLOB:-}"
            else
                export SCENARIOS_CSV="$scenarios"
                export SCENARIO_GLOB=""
            fi
            bash "$SEQUENTIAL_SCRIPT" "$world" || overall=1
        done
        exit "$overall"
    ' "${exec_args[@]}" >>"$HOST_RUN_LOG/worker.log" 2>&1 &
worker_pid=$!

# Show Isaac/Arena launch output live while silently enforcing the VRAM stop gate.
launch_logs=("$HOST_RUN_LOG"/*/launch.log)
tail -n +1 -q -F "${launch_logs[@]}" &
launch_tail_pid=$!

while kill -0 "$worker_pid" 2>/dev/null; do
    sleep "$MONITOR_SECONDS"
    free="$(gpu_free_mib)"
    if ((free < STOP_FREE_MIB)); then
        echo "GPU $GPU_ID fell below ${STOP_FREE_MIB}MiB; stopping worker." >&2
        kill -INT "$worker_pid" 2>/dev/null || true
        break
    fi
done

status=0
wait "$worker_pid" || status=$?
worker_pid=""
kill "$launch_tail_pid" 2>/dev/null || true
wait "$launch_tail_pid" 2>/dev/null || true
launch_tail_pid=""
trap - EXIT INT TERM
if [[ "$REUSE_EXISTING_PROJECT" != "true" ]]; then
    compose down --remove-orphans >/dev/null 2>&1 || true
fi
exit "$status"
