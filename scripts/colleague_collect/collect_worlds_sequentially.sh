#!/usr/bin/env bash
# Sequential, per-world Arena data collection.
#
# Run this INSIDE the Arena container from /opt/arena_ws, for example:
#   bash src/Arena/scripts/collect_worlds_sequentially.sh --range 13 30
#
# It intentionally collects one world per Isaac lifetime:
#   launch -> batch scenarios -> graceful shutdown -> cooldown -> stage-1 postprocess.
# A scenario is retried, then recorded and skipped if it remains unavailable;
# collection proceeds with the remaining scenarios and worlds.

set -Eeuo pipefail

WORKSPACE="/opt/arena_ws"
ARENA_DIR="$WORKSPACE/src/Arena"
BATCH_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/batch_collect_scenarios.py"
POSTPROCESS_SCRIPT="$ARENA_DIR/arena_isaac/arena_isaac/arena_isaac/data_logging/postprocess/process_raw_to_dataset.py"
ANNOTATE_SCRIPT="$ARENA_DIR/arena_isaac/arena_isaac/arena_isaac/data_logging/postprocess/annotate_with_instructions.py"
BUILD_INSTRUCTION_MAP_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_instruction_map_from_state.py"
INSTRUCTION_CONFIG="$ARENA_DIR/arena_isaac/arena_isaac/arena_isaac/data_logging/postprocess/social_interaction_instructions.yaml"
DATA_ROOT="${ARENA_DATASET_ROOT:-src/social_gen/traj_data/grscenes}"
ISAAC_LAUNCH_MODE="${ISAAC_LAUNCH_MODE:-docker}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
ISAAC_RUN_SCRIPT="${ISAAC_RUN_SCRIPT:-/opt/arena_ws/build/arena_isaac/arena_isaac/run_isaacsim.py}"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-20}"
READY_TIMEOUT="${READY_TIMEOUT:-1500}"
BATCH_SETTLE_SECONDS="${BATCH_SETTLE_SECONDS:-30}"
RETRIES="${RETRIES:-2}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-true}"
ROBOT="${ROBOT:-Ai2_Bot2}"
KEEP_RAW="${KEEP_RAW:-false}"
SKIP_PREVIEW_VIDEO="${SKIP_PREVIEW_VIDEO:-true}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-}"
SCENARIOS_CSV="${SCENARIOS_CSV:-default,default_1,default_2,default_3,default_4}"
SCENARIO_GLOB="${SCENARIO_GLOB:-}"
SCENARIO_STATE_ROOT="${SCENARIO_STATE_ROOT:-$WORKSPACE/log/batch_collect/scenario_state}"
SCENARIO_TIMING_ROOT="${SCENARIO_TIMING_ROOT:-$SCENARIO_STATE_ROOT/timings}"
SCENARIOS=()
if [[ -z "$SCENARIO_GLOB" ]]; then
    IFS=',' read -r -a SCENARIOS <<<"$SCENARIOS_CSV"
    if ((${#SCENARIOS[@]} == 0)); then
        echo "SCENARIOS_CSV must contain at least one scenario." >&2
        exit 2
    fi
    for scenario in "${SCENARIOS[@]}"; do
        if [[ -z "$scenario" ]]; then
            echo "SCENARIOS_CSV contains an empty scenario: $SCENARIOS_CSV" >&2
            exit 2
        fi
    done
fi

usage() {
    cat <<'EOF'
Usage:
  bash src/Arena/scripts/collect_worlds_sequentially.sh WORLD [WORLD ...]
  bash src/Arena/scripts/collect_worlds_sequentially.sh --range FIRST LAST
  bash src/Arena/scripts/collect_worlds_sequentially.sh --all

Run this inside the Arena container (/opt/arena_ws).  --range 13 30 expands to
grscenes_13 through grscenes_30; --all expands to grscenes_1 through
grscenes_30.  The script uses headless:=2 and runs stage-1 postprocessing.
When INSTRUCTION_TYPE is set, it also assigns that fixed instruction after
conversion; VLM instruction generation is still not included.

Environment overrides:
  COOLDOWN_SECONDS=20  READY_TIMEOUT=1500  RETRIES=2
  CONTINUE_ON_FAIL=true  ROBOT=Ai2_Bot2
  KEEP_RAW=false         Delete raw NPY/JSON after verified postprocessing
  SKIP_PREVIEW_VIDEO=true  Skip RGB preview MP4 generation (requires no ffmpeg)
  INSTRUCTION_TYPE=perpendicular_traffic  Assign this fixed instruction after postprocess
  ISAAC_LAUNCH_MODE=native  Run Isaac as a native process instead of via Docker
  ISAAC_PYTHON=/isaac-sim/python.sh  Native Isaac Python entrypoint
  ISAAC_RUN_SCRIPT=/path/run_isaacsim.py  Native Isaac main script
  SCENARIOS_CSV=default  Comma-separated scenarios to collect
  SCENARIO_GLOB=gpt56_*  Discover matching scenario directories per world
  SCENARIO_STATE_ROOT=/path  Per-world, per-scenario resume state directory
  SCENARIO_TIMING_ROOT=/path Per-world timing CSV directory
  COLLECTED_STATE_FILE=/path/to/collected_worlds.txt
  COMPLETED_STATE_FILE=/path/to/completed_worlds.txt

Collection and postprocessing state are recorded separately, one world per line:
  /opt/arena_ws/log/batch_collect/collected_worlds.txt
  /opt/arena_ws/log/batch_collect/completed_worlds.txt

`collected` worlds skip launch/batch but still run stage-1 postprocessing.
`completed` worlds skip the whole pipeline.  To adopt worlds collected manually,
add their names to collected_worlds.txt once.

RETRIES is the number of retries after the first attempt (2 = three attempts).
With CONTINUE_ON_FAIL=true, a scenario that still fails is recorded in each
world's failed_scenarios.tsv and the remaining scenarios/worlds continue.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
elif [[ "$1" == "--all" ]]; then
    if [[ $# -ne 1 ]]; then
        echo "--all cannot be combined with explicit world names." >&2
        exit 2
    fi
    WORLDS=()
    for i in {1..30}; do
        WORLDS+=("grscenes_${i}")
    done
elif [[ "$1" == "--range" ]]; then
    if [[ $# -ne 3 ]] || ! [[ "$2" =~ ^[1-9][0-9]*$ && "$3" =~ ^[1-9][0-9]*$ ]] || (( $2 > $3 )); then
        echo "Usage: $0 --range FIRST LAST   (for example: --range 13 30)" >&2
        exit 2
    fi
    WORLDS=()
    for ((i = $2; i <= $3; ++i)); do
        WORLDS+=("grscenes_${i}")
    done
else
    WORLDS=("$@")
fi

cd "$WORKSPACE"

    if [[ ! -f "$BATCH_SCRIPT" || ! -f "$POSTPROCESS_SCRIPT" || ! -f "$ANNOTATE_SCRIPT" || ! -f "$BUILD_INSTRUCTION_MAP_SCRIPT" ]]; then
    echo "Arena source tree is incomplete under $ARENA_DIR." >&2
    exit 2
fi

# A script launched from an already-sourced Arena shell inherits these functions.
# Source the container setup only when it does not, so direct `bash script.sh` works too.
if ! declare -F arena >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$WORKSPACE/source"
fi

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_ROOT="${BATCH_LOG_ROOT:-$WORKSPACE/log/batch_collect/$RUN_ID}"
COLLECTED_STATE_FILE="${COLLECTED_STATE_FILE:-$WORKSPACE/log/batch_collect/collected_worlds.txt}"
COMPLETED_STATE_FILE="${COMPLETED_STATE_FILE:-$WORKSPACE/log/batch_collect/completed_worlds.txt}"
mkdir -p "$LOG_ROOT"
mkdir -p "$(dirname "$COLLECTED_STATE_FILE")"
touch "$COLLECTED_STATE_FILE" "$COMPLETED_STATE_FILE"

LAUNCH_PID=""
ISAAC_PID=""
CURRENT_WORLD=""

log() {
    printf '[%(%F %T)T] %s\n' -1 "$*"
}

native_isaac_running() {
    [[ -n "$ISAAC_PID" ]] && kill -0 "$ISAAC_PID" 2>/dev/null
}

start_native_isaac() {
    local world="$1"
    local robot="${2:-$ROBOT}"
    local headless="${3:-2}"
    local isaac_log="$LOG_ROOT/$world/isaac.log"
    mkdir -p "$(dirname "$isaac_log")"

    if [[ ! -x "$ISAAC_PYTHON" ]]; then
        echo "Native Isaac Python not found or not executable: $ISAAC_PYTHON" >&2
        return 1
    fi
    if [[ ! -f "$ISAAC_RUN_SCRIPT" ]]; then
        echo "Native Isaac run script not found: $ISAAC_RUN_SCRIPT" >&2
        return 1
    fi

    log "${world}: starting native Isaac process."
    local headless_bool=false
    if (( headless > 0 )); then
        headless_bool=true
    fi
    setsid env \
        ARENA_DATASET_ROOT="$DATA_ROOT" \
        ARENA_ROBOT="$robot" \
        ARENA_WORLD="$world" \
        ARENA_HEADLESS="$headless_bool" \
        ARENA_SAVE_DATA=true \
        "$ISAAC_PYTHON" "$ISAAC_RUN_SCRIPT" \
            --save-data true \
            --robot "$robot" \
            --world "$world" \
            --headless "$headless_bool" \
            >>"$isaac_log" 2>&1 &
    ISAAC_PID=$!
}

stop_native_isaac() {
    if [[ -z "$ISAAC_PID" ]]; then
        return 0
    fi
    if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
        ISAAC_PID=""
        return 0
    fi

    log "Stopping native Isaac process group."
    kill -INT -- "-$ISAAC_PID" 2>/dev/null || kill -INT "$ISAAC_PID" 2>/dev/null || true
    local deadline=$((SECONDS + 60))
    while native_isaac_running && (( SECONDS < deadline )); do
        sleep 2
    done
    if native_isaac_running; then
        log "Native Isaac did not stop after SIGINT; sending SIGTERM/SIGKILL."
        kill -TERM -- "-$ISAAC_PID" 2>/dev/null || kill -TERM "$ISAAC_PID" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$ISAAC_PID" 2>/dev/null || kill -KILL "$ISAAC_PID" 2>/dev/null || true
    fi
    wait "$ISAAC_PID" 2>/dev/null || true
    ISAAC_PID=""
}

isaac_running() {
    arena_docker_compose ps -q isaac 2>/dev/null | grep -q .
}

wait_for_isaac_stop() {
    local timeout_s="${1:-180}"
    local deadline=$((SECONDS + timeout_s))
    while isaac_running; do
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for Isaac container to stop." >&2
            return 1
        fi
        sleep 2
    done
}

capture_isaac_state() {
    local cid
    cid="$(arena_docker_compose ps -q isaac 2>/dev/null | head -1 || true)"
    [[ -n "$cid" ]] || return 0
    local state_file="$LOG_ROOT/$CURRENT_WORLD/isaac_state.log"
    mkdir -p "$(dirname "$state_file")"
    docker inspect "$cid" \
        --format 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} Status={{.State.Status}} RestartCount={{.RestartCount}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}}' \
        | tee -a "$state_file"
}

stop_current_launch() {
    if [[ "$ISAAC_LAUNCH_MODE" == "native" ]]; then
        stop_native_isaac
        if [[ -n "$LAUNCH_PID" ]] && kill -0 -- "-$LAUNCH_PID" 2>/dev/null; then
            log "Stopping native arena launch for ${CURRENT_WORLD} (SIGINT to launch process group)..."
            kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true
            local deadline=$((SECONDS + 180))
            while kill -0 -- "-$LAUNCH_PID" 2>/dev/null; do
                if (( SECONDS >= deadline )); then
                    echo "arena launch process group did not exit within 180 s." >&2
                    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
                    local term_deadline=$((SECONDS + 30))
                    while kill -0 -- "-$LAUNCH_PID" 2>/dev/null && (( SECONDS < term_deadline )); do
                        sleep 2
                    done
                    kill -KILL -- "-$LAUNCH_PID" 2>/dev/null || true
                    wait "$LAUNCH_PID" 2>/dev/null || true
                    LAUNCH_PID=""
                    return 1
                fi
                sleep 2
            done
            wait "$LAUNCH_PID" || true
        fi
        LAUNCH_PID=""
        return 0
    fi

    capture_isaac_state
    if [[ -n "$LAUNCH_PID" ]] && kill -0 -- "-$LAUNCH_PID" 2>/dev/null; then
        log "Stopping arena launch for ${CURRENT_WORLD} (SIGINT to launch process group)..."
        # launch is started through setsid below, so its PID is also the process-group
        # ID.  Signalling -PID reaches the wrapper, ros2 launch and the Isaac feature,
        # letting the latter's EXIT trap stop the Isaac container.
        kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true

        local deadline=$((SECONDS + 180))
        while kill -0 -- "-$LAUNCH_PID" 2>/dev/null; do
            if (( SECONDS >= deadline )); then
                echo "arena launch process group did not exit within 180 s." >&2
                log "Stopping Isaac defensively, then terminating the remaining launch group."
                arena_docker_compose stop isaac || true
                kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
                local term_deadline=$((SECONDS + 30))
                while kill -0 -- "-$LAUNCH_PID" 2>/dev/null && (( SECONDS < term_deadline )); do
                    sleep 2
                done
                if kill -0 -- "-$LAUNCH_PID" 2>/dev/null; then
                    log "Launch process group survived SIGTERM; sending SIGKILL."
                    kill -KILL -- "-$LAUNCH_PID" 2>/dev/null || true
                fi
                wait "$LAUNCH_PID" 2>/dev/null || true
                LAUNCH_PID=""
                return 1
            fi
            sleep 2
        done
        wait "$LAUNCH_PID" || true
    fi
    LAUNCH_PID=""

    # arena launch normally stops this service via its EXIT trap.  Keep this as a
    # defensive fallback if its shutdown path was interrupted.
    if isaac_running; then
        capture_isaac_state
        log "Isaac is still running; requesting compose stop."
        arena_docker_compose stop isaac
    fi
    wait_for_isaac_stop
}

cleanup() {
    local status=$?
    if [[ "$ISAAC_LAUNCH_MODE" == "native" ]]; then
        if [[ -n "$LAUNCH_PID" ]] || native_isaac_running; then
            log "Cleanup after interruption/failure."
            stop_current_launch || true
        fi
    elif [[ -n "$LAUNCH_PID" ]] || isaac_running; then
        log "Cleanup after interruption/failure."
        stop_current_launch || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

run_logged() {
    local logfile="$1"
    shift
    PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$logfile"
    return "${PIPESTATUS[0]}"
}

world_in_state_file() {
    local world="$1"
    local state_file="$2"
    grep -Fqx "$world" "$state_file"
}

mark_world_in_state_file() {
    local world="$1"
    local state_file="$2"
    local tmp
    tmp="$(mktemp "${state_file}.XXXXXX")"
    {
        cat "$state_file"
        printf '%s\n' "$world"
    } | awk 'NF && !seen[$0]++' >"$tmp"
    mv "$tmp" "$state_file"
    log "${world}: recorded in $state_file"
}

collect_world() {
    local world="$1"
    CURRENT_WORLD="$world"
    local world_log_dir="$LOG_ROOT/$world"
    mkdir -p "$world_log_dir"
    local launch_log="$world_log_dir/launch.log"
    local batch_log="$world_log_dir/batch.log"
    local failed_scenarios_file="$world_log_dir/failed_scenarios.tsv"
    local timings_file="$SCENARIO_TIMING_ROOT/${world}.csv"
    local postprocess_log="$world_log_dir/postprocess.log"
    local instruction_log="$world_log_dir/instruction.log"
    local scenario_state_file="$SCENARIO_STATE_ROOT/${world}.txt"
    local -a world_scenarios

    if [[ -n "$SCENARIO_GLOB" ]]; then
        local scenarios_dir="$ARENA_DIR/arena_simulation_setup/worlds/$world/scenarios"
        mapfile -t world_scenarios < <(
            find "$scenarios_dir" -mindepth 1 -maxdepth 1 -type d \
                -name "$SCENARIO_GLOB" -printf '%f\n' | sort -V
        )
        if ((${#world_scenarios[@]} == 0)); then
            echo "${world}: no scenario directories match '$SCENARIO_GLOB'." >&2
            return 1
        fi
        log "${world}: discovered ${#world_scenarios[@]} scenario(s) matching $SCENARIO_GLOB."
    else
        world_scenarios=("${SCENARIOS[@]}")
    fi
    mkdir -p "$SCENARIO_STATE_ROOT" "$SCENARIO_TIMING_ROOT"

    if world_in_state_file "$world" "$COLLECTED_STATE_FILE"; then
        log "===== ${world}: already collected; skipping launch/batch ====="
    else
        log "===== ${world}: launching Isaac/headless collection ====="
        # A new session is essential: killing a background shell alone does not
        # reliably interrupt its ros2-launch child.  The session/group is stopped
        # together by stop_current_launch().
        if [[ "$ISAAC_LAUNCH_MODE" == "native" ]]; then
            setsid bash -c '
                source /opt/arena_ws/source
                ros2 launch arena_bringup arena.launch.py "$@"
            ' arena-launch \
                sim:=isaac \
                robot:="$ROBOT" \
                world:="$world" \
                tm_robots:=scenario \
                tm_obstacles:=scenario \
                startup_reset:=true \
                save_data:=true \
                start_sim:=false \
                headless:=2 >>"$launch_log" 2>&1 &
        else
            setsid bash -c '
                source /opt/arena_ws/source
                arena launch "$@"
            ' arena-launch \
                sim:=isaac \
                robot:="$ROBOT" \
                world:="$world" \
                tm_robots:=scenario \
                tm_obstacles:=scenario \
                startup_reset:=true \
                save_data:=true \
                headless:=2 >>"$launch_log" 2>&1 &
        fi
        LAUNCH_PID=$!

        if [[ "$ISAAC_LAUNCH_MODE" == "native" ]]; then
            start_native_isaac "$world" "$ROBOT" 2 || {
                stop_current_launch
                return 1
            }
        fi

        log "${world}: waiting for services/Nav2 through batch collector."
        local episode_base
        episode_base="$(python3 - "$DATA_ROOT/$world" <<'PY'
import os
import re
import sys

base = sys.argv[1]
indices = set()
if os.path.isdir(base):
    for name in os.listdir(base):
        match = re.match(r"episode_(\d+)\.json$", name)
        if match:
            indices.add(int(match.group(1)))
        chunk_dir = os.path.join(base, name)
        if name.startswith("chunk-") and os.path.isdir(chunk_dir):
            for file_name in os.listdir(chunk_dir):
                pm = re.match(r"episode_(\d+)\.parquet$", file_name)
                if pm:
                    indices.add(int(pm.group(1)))
print(max(indices) + 1 if indices else 0)
PY
)"
        local episode_map_file="$SCENARIO_STATE_ROOT/${world}.episode_map.tsv"
        local batch_args=(
            --world "$world"
            --robot "$ROBOT"
            --scenarios "${world_scenarios[@]}"
            --episode-base "$episode_base"
            --episode-map-file "$episode_map_file"
            --skip-rviz-check
            --ready-timeout "$READY_TIMEOUT"
            --settle-s "$BATCH_SETTLE_SECONDS"
            --retries "$RETRIES"
            --failed-scenarios-file "$failed_scenarios_file"
            --completed-scenarios-file "$scenario_state_file"
            --timings-file "$timings_file"
        )
        case "${CONTINUE_ON_FAIL,,}" in
            1|true|yes)
                batch_args+=(--continue-on-fail)
                ;;
            0|false|no)
                ;;
            *)
                echo "CONTINUE_ON_FAIL must be true or false, got: $CONTINUE_ON_FAIL" >&2
                return 2
                ;;
        esac

        if ! run_logged "$batch_log" \
            python3 "$BATCH_SCRIPT" \
                "${batch_args[@]}"; then
            echo "${world}: batch collection failed; see $world_log_dir" >&2
            stop_current_launch
            return 1
        fi

        # With --continue-on-fail the batch collector deliberately exits zero
        # after recording skipped scenarios. Successful scenarios are already
        # durable in scenario_state_file and will be skipped on a later run.
        local collection_failed=false
        if [[ -s "$failed_scenarios_file" ]]; then
            collection_failed=true
            echo "${world}: one or more scenarios failed; see $failed_scenarios_file" >&2
        else
            mark_world_in_state_file "$world" "$COLLECTED_STATE_FILE"
        fi

        log "${world}: batch collection completed; shutting down Isaac."
        stop_current_launch

        log "${world}: cooling down for ${COOLDOWN_SECONDS}s before postprocessing."
        sleep "$COOLDOWN_SECONDS"
    fi

    log "${world}: stage-1 postprocessing."
    local postprocess_args=(
        --data-path "$DATA_ROOT/$world"
    )
    case "${KEEP_RAW,,}" in
        1|true|yes)
            postprocess_args+=(--keep-src)
            log "${world}: KEEP_RAW=true; preserving raw NPY/JSON inputs."
            ;;
        0|false|no)
            log "${world}: KEEP_RAW=false; verified raw inputs may be removed by postprocessing."
            ;;
        *)
            echo "KEEP_RAW must be true or false, got: $KEEP_RAW" >&2
            return 2
            ;;
    esac
    case "${SKIP_PREVIEW_VIDEO,,}" in
        1|true|yes)
            postprocess_args+=(--skip-video)
            log "${world}: SKIP_PREVIEW_VIDEO=true; skipping RGB preview MP4 generation."
            ;;
        0|false|no)
            ;;
        *)
            echo "SKIP_PREVIEW_VIDEO must be true or false, got: $SKIP_PREVIEW_VIDEO" >&2
            return 2
            ;;
    esac
    if ! run_logged "$postprocess_log" \
        python3 "$POSTPROCESS_SCRIPT" \
            "${postprocess_args[@]}"; then
        echo "${world}: postprocessing failed; see $world_log_dir" >&2
        return 1
    fi

    if [[ "$INSTRUCTION_TYPE" == "auto" ]]; then
        local instruction_map="$world_log_dir/instruction_map.json"
        log "${world}: building per-episode instructions from scenario names."
        local instruction_map_args=(
            --instruction-config "$INSTRUCTION_CONFIG"
            --output "$instruction_map"
        )
        if [[ -f "$episode_map_file" ]]; then
            instruction_map_args+=(--episode-map "$episode_map_file")
        else
            instruction_map_args+=(--scenario-state "$scenario_state_file")
        fi
        if ! run_logged "$instruction_log" \
            python3 "$BUILD_INSTRUCTION_MAP_SCRIPT" \
                "${instruction_map_args[@]}"; then
            echo "${world}: instruction map generation failed; see $world_log_dir" >&2
            return 1
        fi
        if ! run_logged "$instruction_log" \
            python3 "$ANNOTATE_SCRIPT" \
                --data-path "$DATA_ROOT/$world" \
                --instructions "$instruction_map"; then
            echo "${world}: instruction annotation failed; see $world_log_dir" >&2
            return 1
        fi
    elif [[ -n "$INSTRUCTION_TYPE" ]]; then
        log "${world}: assigning fixed instruction '${INSTRUCTION_TYPE}'."
        if ! run_logged "$instruction_log" \
            python3 "$ANNOTATE_SCRIPT" \
                --data-path "$DATA_ROOT/$world" \
                --instruction-type "$INSTRUCTION_TYPE"; then
            echo "${world}: instruction annotation failed; see $world_log_dir" >&2
            return 1
        fi
    fi

    if [[ "${collection_failed:-false}" == true ]]; then
        log "${world}: postprocessed successful episodes; incomplete scenarios remain."
        return 1
    fi

    log "${world}: postprocessing completed; cooling down for ${COOLDOWN_SECONDS}s."
    mark_world_in_state_file "$world" "$COMPLETED_STATE_FILE"
    sleep "$COOLDOWN_SECONDS"
    log "===== ${world}: complete ====="
    CURRENT_WORLD=""
}

log "Run ID: $RUN_ID"
log "Worlds: ${WORLDS[*]}"
log "Logs: $LOG_ROOT"
log "Collection state: $COLLECTED_STATE_FILE"
log "Completion state: $COMPLETED_STATE_FILE"

for world in "${WORLDS[@]}"; do
    if world_in_state_file "$world" "$COMPLETED_STATE_FILE"; then
        log "===== ${world}: already complete; skipping ====="
        continue
    fi
    collect_world "$world"
done

trap - EXIT INT TERM
log "All requested worlds completed successfully."
