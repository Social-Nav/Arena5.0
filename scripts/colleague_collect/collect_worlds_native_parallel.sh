#!/usr/bin/env bash
# Native multi-GPU Arena collector.
#
# Run this INSIDE the Arena/Isaac-capable container. It does not use Docker or
# the host daemon; each worker starts its own Arena ROS2 process group and native
# Isaac process, with isolated GPU, ROS domain, logs, state and output paths.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="${ARENA_WS_DIR:-$(cd "$ARENA_DIR/../.." && pwd)}"
SEQUENTIAL_SCRIPT="$ARENA_DIR/scripts/colleague_collect/collect_worlds_sequentially.sh"

WORKER_COUNT=1
GPU_CSV="0,1,2,3,4,5,6,7"
FIRST_WORLD=1
LAST_WORLD=30
DOMAIN_BASE="${DOMAIN_BASE:-180}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
STOP_FREE_MIB="${STOP_FREE_MIB:-2500}"
MONITOR_SECONDS="${MONITOR_SECONDS:-15}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-30}"

ROBOT="${ROBOT:-Ai2_Bot2}"
KEEP_RAW="${KEEP_RAW:-false}"
SKIP_PREVIEW_VIDEO="${SKIP_PREVIEW_VIDEO:-true}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-}"
RETRIES="${RETRIES:-2}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-true}"
READY_TIMEOUT="${READY_TIMEOUT:-1500}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-20}"
DEFAULT_SCENARIOS_CSV="default,default_1,default_2,default_3,default_4"
SCENARIOS_CSV="${SCENARIOS_CSV:-$DEFAULT_SCENARIOS_CSV}"
SCENARIO_GLOB="${SCENARIO_GLOB:-}"
DATASET_ROOT="${ARENA_DATASET_ROOT:-src/social_gen/traj_data/grscenes}"
DRY_RUN=false
MANIFEST_FILE=""
MANIFEST_HASH=""
RANGE_GIVEN=false
SCENARIOS_GIVEN=false
declare -A MANIFEST_SCENARIOS=()
MANIFEST_WORLDS=()

usage() {
    cat <<'EOF'
Usage:
  bash src/Arena/scripts/colleague_collect/collect_worlds_native_parallel.sh [options]

Options:
  --workers N        Number of concurrent GPU workers
  --gpus LIST        Comma-separated GPU IDs
  --range FIRST LAST Collect grscenes_FIRST through grscenes_LAST
  --scenarios LIST   Comma-separated scenario names
  --scenario-glob P  Discover matching scenario directories
  --manifest FILE    Two-column world/scenario manifest
  --dataset-root P   Dataset output root
  --instruction-type TYPE
                     Fixed instruction type, or 'auto'
  --keep-raw         Keep raw JSON/NPY after postprocess
  --no-keep-raw      Delete raw JSON/NPY after postprocess
  --skip-preview-video
  --no-skip-preview-video
  --dry-run          Print worker plan and exit
  -h, --help         Show this help

Environment:
  DOMAIN_BASE, ROBOT, KEEP_RAW, SKIP_PREVIEW_VIDEO, INSTRUCTION_TYPE,
  RETRIES, CONTINUE_ON_FAIL, READY_TIMEOUT, COOLDOWN_SECONDS
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
is_uint "$MIN_FREE_MIB" && is_uint "$STOP_FREE_MIB" || die "VRAM thresholds must be integers"
((STOP_FREE_MIB < MIN_FREE_MIB)) || die "STOP_FREE_MIB must be below MIN_FREE_MIB"
[[ -f "$SEQUENTIAL_SCRIPT" ]] || die "sequential collector not found: $SEQUENTIAL_SCRIPT"
[[ -z "$SCENARIO_GLOB" || "$SCENARIOS_CSV" == "$DEFAULT_SCENARIOS_CSV" ]] || \
    die "--scenario-glob cannot be combined with --scenarios"

if [[ -n "$MANIFEST_FILE" ]]; then
    [[ "$RANGE_GIVEN" == false && "$SCENARIOS_GIVEN" == false ]] || \
        die "--manifest cannot be combined with --range/--scenarios/--scenario-glob"
    [[ -f "$MANIFEST_FILE" ]] || die "manifest file not found: $MANIFEST_FILE"
    line_number=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        line_number=$((line_number + 1))
        line="${line%$'\r'}"
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        read -r world_name scenario_name extra <<<"$line"
        [[ -n "$world_name" && -n "$scenario_name" && -z "$extra" ]] || \
            die "manifest line $line_number must contain exactly WORLD SCENARIO"
        if [[ -z "${MANIFEST_SCENARIOS[$world_name]:-}" ]]; then
            MANIFEST_WORLDS+=("$world_name")
            MANIFEST_SCENARIOS[$world_name]="$scenario_name"
        else
            MANIFEST_SCENARIOS[$world_name]+=",$scenario_name"
        fi
    done < "$MANIFEST_FILE"
    ((${#MANIFEST_WORLDS[@]} > 0)) || die "manifest contains no selected scenarios"
    MANIFEST_HASH="$(printf '%s\n' "${MANIFEST_WORLDS[@]}" | LC_ALL=C sort | sha256sum | cut -c1-16)"
    ((WORKER_COUNT <= ${#MANIFEST_WORLDS[@]})) || WORKER_COUNT="${#MANIFEST_WORLDS[@]}"
fi

command -v nvidia-smi >/dev/null || die "nvidia-smi is not available"
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

declare -a WORKER_WORLDS=()
for ((slot = 0; slot < WORKER_COUNT; ++slot)); do
    WORKER_WORLDS[$slot]=""
done

if [[ -n "$MANIFEST_FILE" ]]; then
    slot=0
    for world_name in "${MANIFEST_WORLDS[@]}"; do
        WORKER_WORLDS[$slot]+=" $world_name"
        slot=$(((slot + 1) % WORKER_COUNT))
    done
elif [[ -n "$SCENARIO_GLOB" ]]; then
    slot=0
    for ((world = FIRST_WORLD; world <= LAST_WORLD; ++world)); do
        world_name="grscenes_${world}"
        ((scenario_count_for_world "$world_name" > 0)) || continue
        WORKER_WORLDS[$slot]+=" $world_name"
        slot=$(((slot + 1) % WORKER_COUNT))
    done
else
    slot=0
    for ((world = FIRST_WORLD; world <= LAST_WORLD; ++world)); do
        WORKER_WORLDS[$slot]+=" grscenes_${world}"
        slot=$(((slot + 1) % WORKER_COUNT))
    done
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
HOST_RUN_LOG="$WORKSPACE/log/native_collect/$RUN_ID"
STATE_ROOT="$WORKSPACE/log/native_collect/state"
TIMING_ROOT="$WORKSPACE/log/native_collect/timings"
mkdir -p "$HOST_RUN_LOG" "$STATE_ROOT" "$TIMING_ROOT"

gpu_free_mib() {
    nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]'
}

print_plan() {
    echo "Workspace: $WORKSPACE"
    echo "Run ID: $RUN_ID"
    echo "Workers: $WORKER_COUNT"
    echo "Dataset root: $DATASET_ROOT"
    echo "Instruction type: ${INSTRUCTION_TYPE:-<none>}"
    echo "KEEP_RAW=$KEEP_RAW SKIP_PREVIEW_VIDEO=$SKIP_PREVIEW_VIDEO"
    for ((slot = 0; slot < WORKER_COUNT; ++slot)); do
        gpu="${GPUS[$slot]}"
        domain=$((DOMAIN_BASE + slot))
        free="$(gpu_free_mib "$gpu")"
        echo "slot=$slot gpu=$gpu free=${free}MiB domain=$domain worlds:${WORKER_WORLDS[$slot]}"
    done
}

print_plan | tee "$HOST_RUN_LOG/plan.log"
if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

worker_main() {
    local slot="$1"
    local gpu="${GPUS[$slot]}"
    local domain=$((DOMAIN_BASE + slot))
    local worker_state="$STATE_ROOT/gpu${gpu}"
    local timing_state="$TIMING_ROOT/gpu${gpu}"
    local collect_pid=""
    local threshold_hit=false
    local -a worlds
    read -r -a worlds <<<"${WORKER_WORLDS[$slot]}"

    mkdir -p "$worker_state" "$timing_state"

    cleanup_worker() {
        local status=$?
        if [[ -n "$collect_pid" ]] && kill -0 "$collect_pid" 2>/dev/null; then
            kill -INT -- "-$collect_pid" 2>/dev/null || kill -INT "$collect_pid" 2>/dev/null || true
            sleep 2
            kill -TERM -- "-$collect_pid" 2>/dev/null || kill -TERM "$collect_pid" 2>/dev/null || true
        fi
        return "$status"
    }
    trap cleanup_worker EXIT
    trap 'exit 130' INT TERM

    local free
    free="$(gpu_free_mib "$gpu")"
    if ((free < MIN_FREE_MIB)); then
        echo "GPU $gpu has only ${free}MiB free; need ${MIN_FREE_MIB}MiB. Skipping." >&2
        return 75
    fi

    for world_name in "${worlds[@]}"; do
        [[ -n "$world_name" ]] || continue
        local scenarios
        if [[ -n "$MANIFEST_FILE" ]]; then
            scenarios="${MANIFEST_SCENARIOS[$world_name]}"
        else
            scenarios="$SCENARIOS_CSV"
        fi

        echo "Starting native worker on GPU $gpu, ROS domain $domain, world $world_name."
        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            ROS_DOMAIN_ID="$domain" \
            ARENA_WS_DIR="$WORKSPACE" \
            ARENA_DATASET_ROOT="$DATASET_ROOT" \
            ISAAC_LAUNCH_MODE=native \
            ROBOT="$ROBOT" \
            KEEP_RAW="$KEEP_RAW" \
            SKIP_PREVIEW_VIDEO="$SKIP_PREVIEW_VIDEO" \
            INSTRUCTION_TYPE="$INSTRUCTION_TYPE" \
            RETRIES="$RETRIES" \
            CONTINUE_ON_FAIL="$CONTINUE_ON_FAIL" \
            READY_TIMEOUT="$READY_TIMEOUT" \
            COOLDOWN_SECONDS="$COOLDOWN_SECONDS" \
            SCENARIOS_CSV="$scenarios" \
            SCENARIO_GLOB="$SCENARIO_GLOB" \
            SCENARIO_STATE_ROOT="$worker_state/scenario_state" \
            SCENARIO_TIMING_ROOT="$timing_state" \
            COLLECTED_STATE_FILE="$worker_state/collected_worlds.txt" \
            COMPLETED_STATE_FILE="$worker_state/completed_worlds.txt" \
            bash "$SEQUENTIAL_SCRIPT" "$world_name" &
        collect_pid=$!

        while kill -0 "$collect_pid" 2>/dev/null; do
            sleep "$MONITOR_SECONDS"
            free="$(gpu_free_mib "$gpu")"
            echo "GPU $gpu free VRAM: ${free}MiB; worker=$world_name"
            if ((free < STOP_FREE_MIB)); then
                echo "GPU $gpu fell below ${STOP_FREE_MIB}MiB; stopping worker." >&2
                threshold_hit=true
                kill -INT -- "-$collect_pid" 2>/dev/null || kill -INT "$collect_pid" 2>/dev/null || true
                break
            fi
        done

        local status=0
        wait "$collect_pid" || status=$?
        collect_pid=""
        if ((status != 0)); then
            echo "Worker GPU $gpu failed for $world_name with status $status." >&2
            return "$status"
        fi
        if [[ "$threshold_hit" == true ]]; then
            return 86
        fi
    done
}

PIDS=()
for ((slot = 0; slot < WORKER_COUNT; ++slot)); do
    worker_log="$HOST_RUN_LOG/gpu${GPUS[$slot]}.log"
    worker_main "$slot" >"$worker_log" 2>&1 &
    PIDS+=("$!")
    if ((slot + 1 < WORKER_COUNT)); then
        sleep "$START_DELAY_SECONDS"
    fi
done

overall=0
for ((i = 0; i < ${#PIDS[@]}; ++i)); do
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        echo "Worker GPU ${GPUS[$i]} completed."
    else
        status=$?
        echo "Worker GPU ${GPUS[$i]} failed with status $status; see $HOST_RUN_LOG/gpu${GPUS[$i]}.log" >&2
        overall=1
    fi
done

exit "$overall"
