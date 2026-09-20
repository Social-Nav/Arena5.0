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
BATCH_SCRIPT="$ARENA_DIR/scripts/batch_collect_scenarios.py"
POSTPROCESS_SCRIPT="$ARENA_DIR/arena_isaac/arena_isaac/arena_isaac/data_logging/postprocess/process_raw_to_dataset.py"
DATA_ROOT="src/social_gen/traj_data/grscenes"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-20}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
RETRIES="${RETRIES:-2}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-true}"
ROBOT="${ROBOT:-Ai2_Bot2}"
SOCIAL_YIELDING="${SOCIAL_YIELDING:-false}"
SCENARIOS=(default default_1 default_2 default_3 default_4)

usage() {
    cat <<'EOF'
Usage:
  bash src/Arena/scripts/collect_worlds_sequentially.sh WORLD [WORLD ...]
  bash src/Arena/scripts/collect_worlds_sequentially.sh --range FIRST LAST
  bash src/Arena/scripts/collect_worlds_sequentially.sh --all

Run this inside the Arena container (/opt/arena_ws).  --range 13 30 expands to
grscenes_13 through grscenes_30; --all expands to grscenes_1 through
grscenes_30.  The script uses headless:=2 and runs stage-1 postprocessing
only; instruction/VLM generation is deliberately not included.

Environment overrides:
  COOLDOWN_SECONDS=20  READY_TIMEOUT=600  RETRIES=2
  CONTINUE_ON_FAIL=true  ROBOT=Ai2_Bot2  SOCIAL_YIELDING=false
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

if [[ ! -f "$BATCH_SCRIPT" || ! -f "$POSTPROCESS_SCRIPT" ]]; then
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
LOG_ROOT="$WORKSPACE/log/batch_collect/$RUN_ID"
COLLECTED_STATE_FILE="${COLLECTED_STATE_FILE:-$WORKSPACE/log/batch_collect/collected_worlds.txt}"
COMPLETED_STATE_FILE="${COMPLETED_STATE_FILE:-$WORKSPACE/log/batch_collect/completed_worlds.txt}"
mkdir -p "$LOG_ROOT"
mkdir -p "$(dirname "$COLLECTED_STATE_FILE")"
touch "$COLLECTED_STATE_FILE" "$COMPLETED_STATE_FILE"

LAUNCH_PID=""
CURRENT_WORLD=""

log() {
    printf '[%(%F %T)T] %s\n' -1 "$*"
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

stop_current_launch() {
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
        log "Isaac is still running; requesting compose stop."
        arena_docker_compose stop isaac
    fi
    wait_for_isaac_stop
}

cleanup() {
    local status=$?
    if [[ -n "$LAUNCH_PID" ]] || isaac_running; then
        log "Cleanup after interruption/failure."
        stop_current_launch || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

run_logged() {
    local logfile="$1"
    shift
    "$@" 2>&1 | tee -a "$logfile"
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
    local postprocess_log="$world_log_dir/postprocess.log"

    if world_in_state_file "$world" "$COLLECTED_STATE_FILE"; then
        log "===== ${world}: already collected; skipping launch/batch ====="
    else
        log "===== ${world}: launching Isaac/headless collection ====="
        # A new session is essential: killing a background shell alone does not
        # reliably interrupt its ros2-launch child.  The session/group is stopped
        # together by stop_current_launch().
        setsid bash -c '
            source /opt/arena_ws/source
            arena launch "$@"
        ' arena-launch \
            sim:=isaac \
            robot:="$ROBOT" \
            world:="$world" \
            tm_robots:=scenario \
            tm_obstacles:=scenario \
            social_yielding:="$SOCIAL_YIELDING" \
            save_data:=true \
            headless:=2 >>"$launch_log" 2>&1 &
        LAUNCH_PID=$!

        log "${world}: waiting for services/Nav2 through batch collector."
        local batch_args=(
            --robot "$ROBOT"
            --scenarios "${SCENARIOS[@]}"
            --skip-rviz-check
            --ready-timeout "$READY_TIMEOUT"
            --retries "$RETRIES"
            --failed-scenarios-file "$failed_scenarios_file"
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

        mark_world_in_state_file "$world" "$COLLECTED_STATE_FILE"
        log "${world}: batch collection completed; shutting down Isaac."
        stop_current_launch

        log "${world}: cooling down for ${COOLDOWN_SECONDS}s before postprocessing."
        sleep "$COOLDOWN_SECONDS"
    fi

    log "${world}: stage-1 postprocessing."
    if ! run_logged "$postprocess_log" \
        python3 "$POSTPROCESS_SCRIPT" \
            --data-path "$DATA_ROOT/$world"; then
        echo "${world}: postprocessing failed; see $world_log_dir" >&2
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
