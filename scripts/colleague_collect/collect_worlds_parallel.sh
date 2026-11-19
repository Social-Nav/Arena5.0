#!/usr/bin/env bash
# Host-side multi-GPU launcher for Arena collection.
#
# One shared workspace/image is used. Each worker gets an isolated Compose
# project, tmpfs volume, ROS domain, GPU, log/state files, and disjoint worlds.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$(cd "$ARENA_DIR/../.." && pwd)"
COMPOSE_FILE="$ARENA_DIR/_meta/docker/docker-compose.yaml"
SEQUENTIAL_SCRIPT="/opt/arena_ws/src/Arena/scripts/colleague_collect/collect_worlds_sequentially.sh"

WORKER_COUNT=1
GPU_CSV="0,1,2,3,4,5,6,7"
FIRST_WORLD=1
LAST_WORLD=30
DOMAIN_BASE="${DOMAIN_BASE:-180}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
STOP_FREE_MIB="${STOP_FREE_MIB:-2500}"
MIN_ROOT_FREE_MIB="${MIN_ROOT_FREE_MIB:-30720}"
STOP_ROOT_FREE_MIB="${STOP_ROOT_FREE_MIB:-20480}"
MONITOR_SECONDS="${MONITOR_SECONDS:-15}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-90}"
PROJECT_PREFIX="${PROJECT_PREFIX:-local-4090-collect}"
COLLECT_ARENA_IMAGE="${COLLECT_ARENA_IMAGE:-arena:dev}"
COLLECT_ISAAC_IMAGE="${COLLECT_ISAAC_IMAGE:-arena_isaac}"
COLLECT_ROS_DISTRO="${COLLECT_ROS_DISTRO:-jazzy}"
DEFAULT_SCENARIOS_CSV="default,default_1,default_2,default_3,default_4"
SCENARIOS_CSV="${SCENARIOS_CSV:-$DEFAULT_SCENARIOS_CSV}"
SCENARIO_GLOB="${SCENARIO_GLOB:-}"
DATASET_ROOT="${ARENA_DATASET_ROOT:-}"
DRY_RUN=false
MANIFEST_FILE=""
MANIFEST_HASH=""
RANGE_GIVEN=false
SCENARIOS_GIVEN=false
declare -A MANIFEST_SCENARIOS=() MANIFEST_COUNTS=() SELECTED_PAIRS=()
MANIFEST_WORLDS=()

usage() {
    cat <<'EOF'
Usage: bash src/Arena/scripts/collect_worlds_parallel.sh [options]

Options:
  --workers N        Number of concurrent GPU workers (default: 1, max: 8)
  --gpus LIST        Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
  --range FIRST LAST Collect grscenes_FIRST through grscenes_LAST (default: 1 30)
  --scenarios LIST   Comma-separated scenario names (default: all five)
  --scenario-glob P  Discover scenario directories per world (for example gpt56_*)
  --manifest FILE    Two-column world/scenario list; collect only these pairs
  --dataset-root P   Dataset output root inside containers
  --dry-run          Print partition and GPU checks without starting containers
  -h, --help         Show this help

Environment controls:
  MIN_FREE_MIB=9000       Refuse to start a worker below this free VRAM
  STOP_FREE_MIB=2500      Stop only that worker if free VRAM drops below this
  MIN_ROOT_FREE_MIB=30720 Refuse to start below this much root-disk free space
  STOP_ROOT_FREE_MIB=20480 Stop workers if root-disk free space drops below this
  START_DELAY_SECONDS=90  Delay between worker starts to avoid an init storm
  MONITOR_SECONDS=15      VRAM monitor interval
  DOMAIN_BASE=180         Worker ROS domains are DOMAIN_BASE + slot
  PROJECT_PREFIX=local-4090-collect
  COLLECT_ARENA_IMAGE=arena:dev   Use your own Arena image when sharing a server
  COLLECT_ISAAC_IMAGE=arena_isaac Use your own compatible Isaac image
  COLLECT_ROS_DISTRO=jazzy

Examples:
  # Inspect the eight-way split without starting anything.
  bash src/Arena/scripts/collect_worlds_parallel.sh --workers 8 --dry-run

  # Start two workers on GPUs 1 and 4.
  MIN_FREE_MIB=12000 bash src/Arena/scripts/collect_worlds_parallel.sh \
    --workers 2 --gpus 1,4

  # Collect one scenario in one world on GPU 0.
  bash src/Arena/scripts/collect_worlds_parallel.sh --workers 1 --gpus 0 \
    --range 1 1 --scenarios default

  # Explicitly start all eight workers after checking the dry run.
  bash src/Arena/scripts/collect_worlds_parallel.sh --workers 8

  # Collect the installed GPT-5.6 10k scenario set with per-scenario resume.
  bash src/Arena/scripts/collect_worlds_parallel.sh --workers 8 \
    --gpus 0,1,2,3,4,5,6,7 --range 1 30 --scenario-glob 'gpt56_*' \
    --dataset-root /opt/arena_ws/src/social_gen/traj_data/grscenes_10k_gpt56
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

while (($#)); do
    case "$1" in
        --workers)
            (($# >= 2)) || die "--workers requires a value"
            WORKER_COUNT="$2"
            shift 2
            ;;
        --gpus)
            (($# >= 2)) || die "--gpus requires a value"
            GPU_CSV="$2"
            shift 2
            ;;
        --range)
            (($# >= 3)) || die "--range requires FIRST LAST"
            FIRST_WORLD="$2"
            LAST_WORLD="$3"
            RANGE_GIVEN=true
            shift 3
            ;;
        --scenarios)
            (($# >= 2)) || die "--scenarios requires a comma-separated list"
            SCENARIOS_CSV="$2"
            SCENARIOS_GIVEN=true
            shift 2
            ;;
        --scenario-glob)
            (($# >= 2)) || die "--scenario-glob requires a pattern"
            SCENARIO_GLOB="$2"
            SCENARIOS_GIVEN=true
            shift 2
            ;;
        --manifest)
            (($# >= 2)) || die "--manifest requires a file"
            MANIFEST_FILE="$2"
            shift 2
            ;;
        --dataset-root)
            (($# >= 2)) || die "--dataset-root requires a path"
            DATASET_ROOT="$2"
            shift 2
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

is_uint "$WORKER_COUNT" || die "worker count must be an integer"
((WORKER_COUNT >= 1 && WORKER_COUNT <= 8)) || die "worker count must be 1..8"
is_uint "$FIRST_WORLD" && is_uint "$LAST_WORLD" || die "world range must be integers"
((FIRST_WORLD >= 1 && FIRST_WORLD <= LAST_WORLD && LAST_WORLD <= 30)) || \
    die "world range must satisfy 1 <= FIRST <= LAST <= 30"
is_uint "$DOMAIN_BASE" || die "DOMAIN_BASE must be an integer"
((DOMAIN_BASE + WORKER_COUNT - 1 <= 232)) || die "ROS domain would exceed 232"
is_uint "$MIN_FREE_MIB" && is_uint "$STOP_FREE_MIB" || \
    die "VRAM thresholds must be integers"
((STOP_FREE_MIB < MIN_FREE_MIB)) || die "STOP_FREE_MIB must be below MIN_FREE_MIB"
is_uint "$MIN_ROOT_FREE_MIB" && is_uint "$STOP_ROOT_FREE_MIB" || \
    die "root disk thresholds must be integers"
((STOP_ROOT_FREE_MIB < MIN_ROOT_FREE_MIB)) || \
    die "STOP_ROOT_FREE_MIB must be below MIN_ROOT_FREE_MIB"
IFS=',' read -r -a REQUESTED_SCENARIOS <<<"$SCENARIOS_CSV"
((${#REQUESTED_SCENARIOS[@]} > 0)) || die "scenario list must not be empty"
for scenario in "${REQUESTED_SCENARIOS[@]}"; do
    [[ -n "$scenario" ]] || die "scenario list contains an empty name"
done
[[ -z "$SCENARIO_GLOB" || "$SCENARIOS_CSV" == "$DEFAULT_SCENARIOS_CSV" ]] || \
    die "--scenario-glob cannot be combined with --scenarios"
if [[ -n "$DATASET_ROOT" && "$DATASET_ROOT" != /* ]]; then
    die "--dataset-root must be an absolute container path"
fi

if [[ -n "$MANIFEST_FILE" ]]; then
    [[ "$RANGE_GIVEN" == false && "$SCENARIOS_GIVEN" == false ]] || \
        die "--manifest cannot be combined with --range, --scenarios or --scenario-glob"
    [[ -z "$SCENARIO_GLOB" && "$SCENARIOS_CSV" == "$DEFAULT_SCENARIOS_CSV" ]] || \
        die "unset SCENARIOS_CSV/SCENARIO_GLOB when using --manifest"
    [[ -f "$MANIFEST_FILE" ]] || die "manifest file not found: $MANIFEST_FILE"
    [[ -n "$DATASET_ROOT" ]] || die "--manifest requires a distinct --dataset-root"
    MANIFEST_FILE="$(realpath "$MANIFEST_FILE")"
    line_number=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line_number=$((line_number + 1))
        line="${line%$'\r'}"
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        read -r world_name scenario_name extra <<<"$line"
        [[ -n "$world_name" && -n "$scenario_name" && -z "$extra" ]] || \
            die "manifest line $line_number must contain exactly WORLD SCENARIO"
        [[ "$world_name" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ && \
           "$scenario_name" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || \
            die "invalid world/scenario name at manifest line $line_number"
        pair="$world_name/$scenario_name"
        [[ -z "${SELECTED_PAIRS[$pair]:-}" ]] || die "duplicate manifest entry: $pair"
        [[ -f "$ARENA_DIR/arena_simulation_setup/worlds/$world_name/scenarios/$scenario_name/scenario.yaml" ]] || \
            die "scenario.yaml not found for selected entry: $pair"
        SELECTED_PAIRS[$pair]=1
        if [[ -z "${MANIFEST_SCENARIOS[$world_name]:-}" ]]; then
            MANIFEST_WORLDS+=("$world_name")
            MANIFEST_SCENARIOS[$world_name]="$scenario_name"
            MANIFEST_COUNTS[$world_name]=1
        else
            MANIFEST_SCENARIOS[$world_name]+=",$scenario_name"
            MANIFEST_COUNTS[$world_name]=$((MANIFEST_COUNTS[$world_name] + 1))
        fi
    done < "$MANIFEST_FILE"
    ((${#MANIFEST_WORLDS[@]} > 0)) || die "manifest contains no selected scenarios"
    MANIFEST_HASH="$(printf '%s\n' "${!SELECTED_PAIRS[@]}" | LC_ALL=C sort | sha256sum | cut -c1-16)"
    if ((WORKER_COUNT > ${#MANIFEST_WORLDS[@]})); then
        WORKER_COUNT="${#MANIFEST_WORLDS[@]}"
        echo "Only $WORKER_COUNT selected world(s); starting $WORKER_COUNT worker(s)."
    fi
fi

command -v docker >/dev/null || die "docker is not available"
command -v nvidia-smi >/dev/null || die "nvidia-smi is not available"
[[ -f "$COMPOSE_FILE" ]] || die "compose file not found: $COMPOSE_FILE"
[[ -f "$WORKSPACE/.env" ]] || die "workspace .env not found: $WORKSPACE/.env"

IFS=',' read -r -a ALL_GPUS <<<"$GPU_CSV"
((${#ALL_GPUS[@]} >= WORKER_COUNT)) || die "fewer GPU IDs than workers"

declare -A SEEN_GPUS=()
GPUS=()
for ((slot = 0; slot < WORKER_COUNT; ++slot)); do
    gpu="${ALL_GPUS[$slot]}"
    is_uint "$gpu" || die "invalid GPU ID: $gpu"
    [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || die "duplicate GPU ID: $gpu"
    SEEN_GPUS[$gpu]=1
    GPUS+=("$gpu")
done

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
for gpu in "${GPUS[@]}"; do
    ((gpu < GPU_COUNT)) || die "GPU $gpu does not exist (found $GPU_COUNT GPUs)"
done

scenario_count_for_world() {
    local world_name="$1"
    find "$ARENA_DIR/arena_simulation_setup/worlds/$world_name/scenarios" \
        -mindepth 1 -maxdepth 1 -type d -name "$SCENARIO_GLOB" \
        -exec test -f '{}/scenario.yaml' ';' -print | wc -l
}

declare -a WORKER_WORLDS
declare -a WORKER_LOAD
declare -a WORKER_ITEMS
for ((slot = 0; slot < WORKER_COUNT; ++slot)); do
    WORKER_WORLDS[$slot]=""
    WORKER_LOAD[$slot]=0
    WORKER_ITEMS[$slot]=0
done
if [[ -n "$MANIFEST_FILE" ]]; then
    mapfile -t WEIGHTED_WORLDS < <(
        for world_name in "${MANIFEST_WORLDS[@]}"; do
            printf '%s %s\n' "${MANIFEST_COUNTS[$world_name]}" "$world_name"
        done | sort -k1,1nr -k2,2V
    )
    for entry in "${WEIGHTED_WORLDS[@]}"; do
        read -r count world_name <<<"$entry"
        lightest_slot=0
        for ((slot = 1; slot < WORKER_COUNT; ++slot)); do
            if ((WORKER_LOAD[slot] < WORKER_LOAD[lightest_slot])); then lightest_slot=$slot; fi
        done
        WORKER_WORLDS[$lightest_slot]+=" $world_name"
        WORKER_LOAD[$lightest_slot]=$((WORKER_LOAD[lightest_slot] + count))
    done
elif [[ -n "$SCENARIO_GLOB" ]]; then
    mapfile -t WEIGHTED_WORLDS < <(
        for ((world = FIRST_WORLD; world <= LAST_WORLD; ++world)); do
            world_name="grscenes_${world}"
            printf '%s %s\n' "$(scenario_count_for_world "$world_name")" "$world_name"
        done | sort -k1,1nr -k2,2V
    )
    world_total="${#WEIGHTED_WORLDS[@]}"
    base_items=$((world_total / WORKER_COUNT))
    long_slots=$((world_total % WORKER_COUNT))
    short_slots=$((WORKER_COUNT - long_slots))
    short_item_count=$((short_slots * base_items))

    # Slots receiving one fewer world get the heaviest worlds. This avoids the
    # long tail produced by ordinary LPT when 30 worlds are split over 8 GPUs.
    for ((item = 0; item < short_item_count; ++item)); do
        read -r count world_name <<<"${WEIGHTED_WORLDS[$item]}"
        ((count > 0)) || die "$world_name has no scenarios matching $SCENARIO_GLOB"
        lightest_slot=-1
        for ((slot = 0; slot < short_slots; ++slot)); do
            if ((WORKER_ITEMS[slot] < base_items)) \
                && ((lightest_slot < 0 || WORKER_LOAD[slot] < WORKER_LOAD[lightest_slot])); then
                lightest_slot=$slot
            fi
        done
        ((lightest_slot >= 0)) || die "internal short-slot partition error"
        WORKER_WORLDS[$lightest_slot]+=" $world_name"
        WORKER_LOAD[$lightest_slot]=$((WORKER_LOAD[lightest_slot] + count))
        WORKER_ITEMS[$lightest_slot]=$((WORKER_ITEMS[lightest_slot] + 1))
    done

    for ((item = short_item_count; item < world_total; ++item)); do
        read -r count world_name <<<"${WEIGHTED_WORLDS[$item]}"
        ((count > 0)) || die "$world_name has no scenarios matching $SCENARIO_GLOB"
        lightest_slot=-1
        for ((slot = short_slots; slot < WORKER_COUNT; ++slot)); do
            if ((WORKER_ITEMS[slot] < base_items + 1)) \
                && ((lightest_slot < 0 || WORKER_LOAD[slot] < WORKER_LOAD[lightest_slot])); then
                lightest_slot=$slot
            fi
        done
        ((lightest_slot >= 0)) || die "internal long-slot partition error"
        WORKER_WORLDS[$lightest_slot]+=" $world_name"
        WORKER_LOAD[$lightest_slot]=$((WORKER_LOAD[lightest_slot] + count))
        WORKER_ITEMS[$lightest_slot]=$((WORKER_ITEMS[lightest_slot] + 1))
    done
else
    for ((world = FIRST_WORLD; world <= LAST_WORLD; ++world)); do
        slot=$(((world - FIRST_WORLD) % WORKER_COUNT))
        WORKER_WORLDS[$slot]+=" grscenes_${world}"
    done
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
HOST_RUN_LOG="$WORKSPACE/log/colleague_collect/$RUN_ID"
TIMING_TAG_INPUT="${SCENARIO_GLOB:-$SCENARIOS_CSV}-${DATASET_ROOT##*/}"
[[ -z "$MANIFEST_FILE" ]] || TIMING_TAG_INPUT="manifest-${MANIFEST_HASH}-${DATASET_ROOT##*/}"
TIMING_TAG="$(printf '%s' "$TIMING_TAG_INPUT" | sed 's/[^A-Za-z0-9._-]/_/g')"
TIMING_ROOT_REL="log/colleague_collect/timings/$TIMING_TAG"
HOST_TIMING_ROOT="$WORKSPACE/$TIMING_ROOT_REL"
CONTAINER_TIMING_ROOT="/opt/arena_ws/$TIMING_ROOT_REL"
mkdir -p "$HOST_RUN_LOG"

gpu_free_mib() {
    nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits \
        | tr -d '[:space:]'
}

root_free_mib() {
    df -Pm / | awk 'NR == 2 {print $4}'
}

print_plan() {
    echo "Workspace: $WORKSPACE"
    echo "Run ID: $RUN_ID"
    echo "Workers: $WORKER_COUNT"
    if [[ -n "$MANIFEST_FILE" ]]; then
        echo "Manifest: $MANIFEST_FILE (selection $MANIFEST_HASH)"
        for world_name in "${MANIFEST_WORLDS[@]}"; do
            echo "Selected $world_name: ${MANIFEST_SCENARIOS[$world_name]}"
        done
    elif [[ -n "$SCENARIO_GLOB" ]]; then
        echo "Scenarios: directories matching $SCENARIO_GLOB in each world"
    else
        echo "Scenarios: $SCENARIOS_CSV"
    fi
    echo "Dataset root: ${DATASET_ROOT:-default social_gen/traj_data/grscenes}"
    echo "Timing CSV root: $HOST_TIMING_ROOT"
    echo "Isaac cache root: $WORKSPACE/.cache/isaac/gpuN"
    echo "VRAM start/stop thresholds: ${MIN_FREE_MIB}/${STOP_FREE_MIB} MiB"
    echo "Root disk start/stop thresholds: ${MIN_ROOT_FREE_MIB}/${STOP_ROOT_FREE_MIB} MiB"
    local total_scenarios=0
    for ((slot = 0; slot < WORKER_COUNT; ++slot)); do
        gpu="${GPUS[$slot]}"
        domain=$((DOMAIN_BASE + slot))
        project="${PROJECT_PREFIX}-gpu${gpu}"
        free="$(gpu_free_mib "$gpu")"
        local scenario_summary=""
        if [[ -n "$SCENARIO_GLOB" || -n "$MANIFEST_FILE" ]]; then
            local worker_scenarios="${WORKER_LOAD[$slot]}"
            total_scenarios=$((total_scenarios + worker_scenarios))
            scenario_summary=" scenarios=$worker_scenarios"
        fi
        echo "slot=$slot gpu=$gpu free=${free}MiB domain=$domain project=$project${scenario_summary} worlds:${WORKER_WORLDS[$slot]}"
    done
    if [[ -n "$SCENARIO_GLOB" || -n "$MANIFEST_FILE" ]]; then
        echo "Total matched scenarios: $total_scenarios"
    fi
}

print_plan | tee "$HOST_RUN_LOG/plan.log"
if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
export HOST_UID HOST_GID

worker_main() (
    local slot="$1"
    local gpu="${GPUS[$slot]}"
    local domain=$((DOMAIN_BASE + slot))
    local project="${PROJECT_PREFIX}-gpu${gpu}"
    local host_isaac_cache="$WORKSPACE/.cache/isaac/gpu${gpu}"
    local worker_state_rel="log/colleague_collect/state/gpu${gpu}"
    local scenario_state_rel=""
    if [[ -n "$MANIFEST_FILE" ]]; then
        local dataset_state_tag
        dataset_state_tag="$(printf '%s' "$DATASET_ROOT" | sed 's/[^A-Za-z0-9._-]/_/g')"
        local state_base_rel="log/colleague_collect/state/manifest-${MANIFEST_HASH}-dataset-${dataset_state_tag}"
        worker_state_rel="$state_base_rel/gpu${gpu}"
        scenario_state_rel="$state_base_rel/by-world"
    elif [[ -n "$SCENARIO_GLOB" ]]; then
        local scenario_glob_tag
        scenario_glob_tag="$(printf '%s' "$SCENARIO_GLOB" | sed 's/[^A-Za-z0-9._-]/_/g')"
        local dataset_state_tag
        dataset_state_tag="$(printf '%s' "${DATASET_ROOT:-default}" | sed 's/[^A-Za-z0-9._-]/_/g')"
        local state_base_rel="log/colleague_collect/state/scenario-glob-${scenario_glob_tag}-dataset-${dataset_state_tag}"
        worker_state_rel="$state_base_rel/gpu${gpu}"
        # Per-world scenario progress is deliberately independent of GPU so a
        # future repartition can resume without duplicating completed episodes.
        scenario_state_rel="$state_base_rel/by-world"
    elif [[ "$SCENARIOS_CSV" != "$DEFAULT_SCENARIOS_CSV" ]]; then
        local scenario_state_tag
        scenario_state_tag="$(printf '%s' "$SCENARIOS_CSV" | sed 's/[^A-Za-z0-9._-]/_/g')"
        worker_state_rel="log/colleague_collect/state/scenarios-${scenario_state_tag}/gpu${gpu}"
    fi
    if [[ -z "$scenario_state_rel" ]]; then
        scenario_state_rel="$worker_state_rel/scenarios"
    fi
    local worker_state_host="$WORKSPACE/$worker_state_rel"
    local worker_state_container="/opt/arena_ws/$worker_state_rel"
    local scenario_state_host="$WORKSPACE/$scenario_state_rel"
    local scenario_state_container="/opt/arena_ws/$scenario_state_rel"
    local collected="$worker_state_container/collected_worlds.txt"
    local completed="$worker_state_container/completed_worlds.txt"
    local collect_pid=""
    local threshold_hit=false
    local project_owned=false
    local -a worlds
    read -r -a worlds <<<"${WORKER_WORLDS[$slot]}"
    local -a world_selections=()
    for selected_world in "${worlds[@]}"; do
        if [[ -n "$MANIFEST_FILE" ]]; then
            world_selections+=("$selected_world" "${MANIFEST_SCENARIOS[$selected_world]}")
        else
            world_selections+=("$selected_world" "$SCENARIOS_CSV")
        fi
    done

    compose() {
        env \
            ARENA_PROJECT_NAME="$project" \
            ARENA_IMAGE="$COLLECT_ARENA_IMAGE" \
            ARENA_ISAAC_IMAGE="$COLLECT_ISAAC_IMAGE" \
            ARENA_GPU_ID="$gpu" \
            ROS_DOMAIN_ID="$domain" \
            ARENA_ROS_DISTRO="$COLLECT_ROS_DISTRO" \
            HOST_UID="$HOST_UID" \
            HOST_GID="$HOST_GID" \
            HOST_ARENA_WS_DIR="$WORKSPACE" \
            ARENA_DATASET_ROOT="$DATASET_ROOT" \
            docker compose \
                -f "$COMPOSE_FILE" \
                --project-directory "$WORKSPACE" \
                --project-name "$project" \
                "$@"
    }

    cleanup_worker() {
        local status=$?
        if [[ -n "$collect_pid" ]] && kill -0 "$collect_pid" 2>/dev/null; then
            kill -INT "$collect_pid" 2>/dev/null || true
        fi
        if [[ "$project_owned" == true ]]; then
            compose stop -t 30 isaac >/dev/null 2>&1 || true
            compose down -t 30 >/dev/null 2>&1 || true
        fi
        return "$status"
    }
    trap cleanup_worker EXIT
    trap 'exit 130' INT TERM

    mkdir -p "$worker_state_host" "$scenario_state_host" "$HOST_TIMING_ROOT" \
        "$host_isaac_cache"/{kit,ov,pip,glcache,computecache,logs,data,documents}

    if compose ps -aq | grep -q .; then
        echo "Refusing to reuse existing Compose project: $project" >&2
        exit 73
    fi

    local free
    local root_free
    free="$(gpu_free_mib "$gpu")"
    if ((free < MIN_FREE_MIB)); then
        echo "GPU $gpu has only ${free}MiB free; need ${MIN_FREE_MIB}MiB. Skipping." >&2
        exit 75
    fi
    root_free="$(root_free_mib)"
    if ((root_free < MIN_ROOT_FREE_MIB)); then
        echo "Root disk has only ${root_free}MiB free; need ${MIN_ROOT_FREE_MIB}MiB. Skipping." >&2
        exit 76
    fi

    echo "Starting Arena worker on GPU $gpu (ROS domain $domain)."
    project_owned=true
    compose up -d --remove-orphans arena

    compose exec -T \
        -e READY_TIMEOUT="${READY_TIMEOUT:-1500}" \
        -e RETRIES="${RETRIES:-0}" \
        -e CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-true}" \
        -e ROBOT="${ROBOT:-Ai2_Bot2}" \
        -e KEEP_RAW="${KEEP_RAW:-true}" \
        -e COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-20}" \
        -e ARENA_RESET_SERVICE_TIMEOUT="${ARENA_RESET_SERVICE_TIMEOUT:-1500}" \
        arena /entrypoint.sh bash -lc '
        collected="$1"
        completed="$2"
        scenarios_csv="$3"
        scenario_glob="$4"
        scenario_state_root="$5"
        scenario_timing_root="$6"
        shift 6
        export COLLECTED_STATE_FILE="$collected"
        export COMPLETED_STATE_FILE="$completed"
        export SCENARIOS_CSV="$scenarios_csv"
        export SCENARIO_GLOB="$scenario_glob"
        export SCENARIO_STATE_ROOT="$scenario_state_root"
        export SCENARIO_TIMING_ROOT="$scenario_timing_root"
        overall=0
        while (($#)); do
            world="$1"
            export SCENARIOS_CSV="$2"
            shift 2
            bash /opt/arena_ws/src/Arena/scripts/colleague_collect/collect_worlds_sequentially.sh "$world" || overall=1
        done
        exit "$overall"
    ' arena-worker "$collected" "$completed" "$SCENARIOS_CSV" \
        "$SCENARIO_GLOB" "$scenario_state_container" \
        "$CONTAINER_TIMING_ROOT" "${world_selections[@]}" &
    collect_pid=$!

    while kill -0 "$collect_pid" 2>/dev/null; do
        sleep "$MONITOR_SECONDS"
        free="$(gpu_free_mib "$gpu")"
        root_free="$(root_free_mib)"
        echo "GPU $gpu free VRAM: ${free}MiB; root disk free: ${root_free}MiB"
        if ((free < STOP_FREE_MIB)); then
            echo "GPU $gpu fell below ${STOP_FREE_MIB}MiB; stopping only project $project." >&2
            threshold_hit=true
            compose stop -t 30 isaac || true
            kill -INT "$collect_pid" 2>/dev/null || true
            break
        fi
        if ((root_free < STOP_ROOT_FREE_MIB)); then
            echo "Root disk fell below ${STOP_ROOT_FREE_MIB}MiB; stopping project $project." >&2
            threshold_hit=true
            compose stop -t 30 isaac || true
            kill -INT "$collect_pid" 2>/dev/null || true
            break
        fi
    done

    local status=0
    wait "$collect_pid" || status=$?
    collect_pid=""
    if [[ "$threshold_hit" == true ]]; then
        exit 86
    fi
    exit "$status"
)

PIDS=()
SLOTS=()

stop_workers() {
    local status=$?
    trap - EXIT INT TERM
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait || true
    exit "$status"
}
trap stop_workers EXIT INT TERM

for ((slot = 0; slot < WORKER_COUNT; ++slot)); do
    worker_log="$HOST_RUN_LOG/gpu${GPUS[$slot]}.log"
    worker_main "$slot" >"$worker_log" 2>&1 &
    PIDS+=("$!")
    SLOTS+=("$slot")
    if ((slot + 1 < WORKER_COUNT)); then
        sleep "$START_DELAY_SECONDS"
    fi
done

overall=0
for ((i = 0; i < ${#PIDS[@]}; ++i)); do
    pid="${PIDS[$i]}"
    slot="${SLOTS[$i]}"
    if wait "$pid"; then
        echo "Worker GPU ${GPUS[$slot]} completed."
    else
        status=$?
        echo "Worker GPU ${GPUS[$slot]} failed with status $status; see $HOST_RUN_LOG/gpu${GPUS[$slot]}.log" >&2
        overall=1
    fi
done

trap - EXIT INT TERM
exit "$overall"
