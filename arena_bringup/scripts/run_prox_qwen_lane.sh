#!/usr/bin/env bash
# Stable repo-owned wrapper for the LISN/Qwen + ProxMPC A-track lane.
#
# Runtime status: IMPLEMENTED_UNVERIFIED for the in-repo copy.  The 2035 run
# used the outputs copy with the same argv; see
# docs/benchmark/prox_mpc_a_track_runbook.md.
#
# Safety properties:
# - shared GPU lock; a busy lock never writes into an existing run directory
# - exclusive mkdir of a timestamped run directory; collision fails
# - model-server start rc is checked before any eval
# - exactly one resolved container IPv4 (optional EXPECTED_CONTAINER_IP)
# - cleanup uses recorded owned PIDs and the exact Qwen container only
# - cleanup errors go to cleanup_errors.log, never silently swallowed
#
# Env overrides:
#   MODE=check|diag|episode
#   LANE_ROOT, PREFIX, RUN_TAG, TIMEOUT, WORKTREE, TARGET_INSTALL_ROOT,
#   MODEL_STAGE, QWEN_CONTAINER, EXPECTED_CONTAINER_IP
set -uo pipefail

ROOT=${ARENA_WS_DIR:-/home/ubuntu/arena_jazzy_ws}
WORKTREE=${WORKTREE:-$ROOT/.trae/worktrees/vln-target-merged-20261013}
LANE_ROOT=${LANE_ROOT:-$ROOT/outputs/arena5_prox_isaac_a_20261102}
TARGET_INSTALL_ROOT=${TARGET_INSTALL_ROOT:-$ROOT/outputs/arena_seed21_four_agent_batch_20261026}
MODEL_STAGE=${MODEL_STAGE:-/mnt/bn/navigation-hl/mlx/users/chenjunting/arena_models/_staging/qwen3vl_fp8_20261030}
QWEN_CONTAINER=${QWEN_CONTAINER:-arena-qwen3vl-fp8}
LOCK=${LOCK:-/tmp/arena_jazzy_ws_isaac_gpu_runtime.lock}
MODE=${MODE:-diag}
PROFILE=${PROFILE:-$WORKTREE/arena_bringup/configs/lisn/slow_loop_grounding_scan_prox_v1.json}
STAMP=$(date +%Y%m%d_%H%M%S)

case "$MODE" in
  check) TIMEOUT=${TIMEOUT:-30}; RUN_TAG=${RUN_TAG:-qwen_check_$STAMP} ;;
  diag) TIMEOUT=${TIMEOUT:-59}; RUN_TAG=${RUN_TAG:-qwen_diag_v2_$STAMP} ;;
  episode) TIMEOUT=${TIMEOUT:-300}; RUN_TAG=${RUN_TAG:-qwen_episode_$STAMP} ;;
  *) echo "unsupported MODE=$MODE" >&2; exit 64 ;;
esac
PREFIX=${PREFIX:-arena5_prox_isaac_a_20261102/$RUN_TAG}
RUN_DIR="$LANE_ROOT/$RUN_TAG"

exec 9>"$LOCK"
if ! flock -n 9; then
  mkdir -p "$LANE_ROOT/logs"
  echo "LOCK_BUSY" > "$LANE_ROOT/lock_busy.$STAMP"
  exit 75
fi

if ! mkdir "$RUN_DIR" 2>/dev/null; then
  echo "OUTPUT_DIR_EXISTS:$RUN_DIR" >&2
  exit 74
fi
mkdir -p "$LANE_ROOT/logs"
echo "lock_acquired_epoch=$(date +%s) pid=$$ mode=$MODE" > "$RUN_DIR/lock.txt"

if [ "${DRY_RUN:-0}" = "1" ]; then
  case "${FAULT_INJECT:-none}" in
    none) DRY_RC=0 ;;
    model_start) DRY_RC=22 ;;
    readiness) DRY_RC=21 ;;
    eval) DRY_RC=3 ;;
    render) DRY_RC=2 ;;
    *) echo "unsupported FAULT_INJECT=${FAULT_INJECT}" >&2; DRY_RC=64 ;;
  esac
  echo "dry_run=1 fault_inject=${FAULT_INJECT:-none} rc=$DRY_RC" >> "$RUN_DIR/lane.rc"
  exit "$DRY_RC"
fi
: > "$RUN_DIR/timing.raw"
: > "$RUN_DIR/OWNED_PIDS.txt"
cat > "$RUN_DIR/RUN_OWNER.json" <<JSON
{"schema":"arena.prox_diag.run_owner.v1","owner":"${OWNER:-OVE_bot}","mode":"$MODE","run_tag":"$RUN_TAG","prefix":"$PREFIX","script":"$0","pid":$$,"start_wall":$(date +%s)}
JSON

RECORDER_PID=""
EVAL_HOST_PID=""
EVAL_CONTAINER_PID=""
ISAAC_PID=""

cleanup() {
  echo "cleanup_start_wall=$(date +%s.%N)" >> "$RUN_DIR/timing.raw"
  : > "$RUN_DIR/cleanup_errors.log"
  kill_owned() {
    local container="$1" pid="$2" pattern="$3" label="$4"
    if [ -z "$pid" ]; then
      echo "${label}_pid_unavailable" >> "$RUN_DIR/cleanup_errors.log"
      return 0
    fi
    if docker exec "$container" ps -p "$pid" -o args= 2>/dev/null | grep -q "$pattern"; then
      docker exec "$container" kill -TERM "$pid" 2>>"$RUN_DIR/cleanup_errors.log" \
        || echo "${label}_kill_failed:$pid" >> "$RUN_DIR/cleanup_errors.log"
    else
      echo "${label}_pid_identity_mismatch:$pid" >> "$RUN_DIR/cleanup_errors.log"
    fi
  }
  kill_owned arena-arena_jazzy_ws-arena-1 "$RECORDER_PID" prox_diag_recorder recorder
  kill_owned arena-arena_jazzy_ws-arena-1 "$EVAL_CONTAINER_PID" 'arena_vln_models.runner' eval_container
  if [ -n "$EVAL_HOST_PID" ]; then
    kill -TERM "$EVAL_HOST_PID" 2>>"$RUN_DIR/cleanup_errors.log" \
      || echo "eval_host_kill_failed:$EVAL_HOST_PID" >> "$RUN_DIR/cleanup_errors.log"
  fi
  kill_owned arena-arena_jazzy_ws-isaac-1 "$ISAAC_PID" run_isaacsim isaac
  if [ -n "$ISAAC_PID" ]; then
    sleep 2
    docker exec -u root arena-arena_jazzy_ws-isaac-1 kill -KILL "$ISAAC_PID" \
      2>>"$RUN_DIR/cleanup_errors.log" || true
  fi
  bash "$MODEL_STAGE/stop_server.sh" 2>>"$RUN_DIR/cleanup_errors.log" \
    || echo "qwen_stop_failed" >> "$RUN_DIR/cleanup_errors.log"
  echo "cleanup_done_wall=$(date +%s.%N)" >> "$RUN_DIR/timing.raw"
}
trap cleanup EXIT INT TERM

bash "$MODEL_STAGE/start_server.sh" > "$LANE_ROOT/logs/${MODE}_docker_run.log" 2>&1
START_RC=$?
echo "docker_run_rc=$START_RC" >> "$RUN_DIR/lane.rc"
if [ "$START_RC" -ne 0 ]; then
  echo "MODEL_START_FAILED:$START_RC" >&2
  exit 22
fi

IP=$(docker inspect "$QWEN_CONTAINER" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null)
if ! printf '%s' "$IP" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "NETWORK_RESOLVE_FAILED:$IP" >&2
  exit 23
fi
if [ -n "${EXPECTED_CONTAINER_IP:-}" ] && [ "$IP" != "$EXPECTED_CONTAINER_IP" ]; then
  echo "NETWORK_MISMATCH:$IP!=$EXPECTED_CONTAINER_IP" >&2
  exit 23
fi
echo "container_ip=$IP" > "$RUN_DIR/identity.txt"

READY=0
for _ in $(seq 1 180); do
  if docker exec -e NO_PROXY="$IP,127.0.0.1,localhost" -e no_proxy="$IP,127.0.0.1,localhost" \
      arena-arena_jazzy_ws-arena-1 bash -lc \
      "curl -s --noproxy '*' --max-time 3 http://$IP:8000/v1/models" 2>/dev/null \
      | grep -q 'Qwen/Qwen3-VL-8B-Instruct-FP8'; then
    READY=1; break
  fi
  sleep 2
done
echo "server_ready=$READY" >> "$RUN_DIR/lane.rc"
if [ "$READY" -ne 1 ]; then exit 21; fi

if [ "$MODE" = "diag" ]; then
  docker exec -d arena-arena_jazzy_ws-arena-1 bash -lc "
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
source $TARGET_INSTALL_ROOT/target_install/setup.bash
source /opt/arena_ws/install_prox/setup.bash
export ROS_DOMAIN_ID=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export PYTHONPATH=$WORKTREE/arena_bringup:\$PYTHONPATH
exec /opt/venv/bin/python3 $WORKTREE/arena_bringup/arena_bringup/prox_diag_recorder.py --ros-args \
  -p use_sim_time:=true \
  -p robot_namespace:=/task_generator_node/Ai2_Bot2 \
  -p record_dir:=$RUN_DIR \
  -p costmap_hz:=5.0 -p ego_hz:=5.0 \
  > $LANE_ROOT/logs/${MODE}_recorder.log 2>&1
"
  sleep 2
  RECORDER_PID=$(docker exec arena-arena_jazzy_ws-arena-1 \
    pgrep -f "prox_diag_recorder.*record_dir:=$RUN_DIR" | head -1)
  echo "recorder_pid=$RECORDER_PID" >> "$RUN_DIR/OWNED_PIDS.txt"
else
  docker exec -d arena-arena_jazzy_ws-arena-1 bash -lc "
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
source $TARGET_INSTALL_ROOT/target_install/setup.bash
source /opt/arena_ws/install_prox/setup.bash
export ROS_DOMAIN_ID=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
exec /opt/venv/bin/python3 $LANE_ROOT/tracker_age_monitor.py --ros-args \
  -p use_sim_time:=true -p robot_namespace:=/task_generator_node/Ai2_Bot2 \
  -p record_dir:=$RUN_DIR -p obstacle_timeout_sec:=1.2 \
  > $LANE_ROOT/logs/${MODE}_monitor.log 2>&1
"
fi

EVAL_START_WALL=$(date +%s.%N)
echo "eval_start_wall=$EVAL_START_WALL" >> "$RUN_DIR/timing.raw"
docker exec \
  -e NO_PROXY="$IP,127.0.0.1,localhost" \
  -e no_proxy="$IP,127.0.0.1,localhost" \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -e ARENA_HUNAV_CONFIG=default_sf60.yaml \
  -e ARENA_LISN_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8 \
  -e ARENA_LISN_VLM_MODEL_REVISION=9cdc6310a8cb770ce18efaf4e9935334512aee45 \
  -e ARENA_LISN_VLM_IMAGE_ID=sha256:c1975562dbe35cf5201d55ba84b8e6093043e415ce0728c0c0bcf6d57ee3ff82 \
  -e ARENA_LISN_VLM_ENFORCE_MODEL_IDENTITY=1 \
  -e ARENA_LISN_VLM_JSON_FORMAT=1 \
  -e ARENA_LISN_VLM_MAX_TOKENS=512 \
  arena-arena_jazzy_ws-arena-1 bash -lc "
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
source $TARGET_INSTALL_ROOT/target_install/setup.bash
source /opt/arena_ws/install_prox/setup.bash
export ARENA_WS_DIR=$ROOT
export HOST_ARENA_WS_DIR=$ROOT
export PYTHONPATH=$WORKTREE/arena_vln_models:\$PYTHONPATH
cd $WORKTREE
/opt/venv/bin/python3 -m arena_vln_models.runner eval --local \
  --base-profile base_follow_path_grscenes_task34.yaml \
  --model internnav --robot Ai2_Bot2 --case grscenes_20_v1/default_2 \
  --episodes 1 --pedestrian-seed 0 --timeout $TIMEOUT \
  --lisn-slow-loop --lisn-wiring-live --prox-mpc \
  --lisn-producer-endpoint http://$IP:8000/v1/chat/completions \
  --lisn-policy $PROFILE --lisn-grounding-scan \
  --lisn-scan-topic /task_generator_node/Ai2_Bot2/laser_scan \
  --lisn-scan-max-age-sec 0.6 --lisn-max-scan-range-m 12.0 \
  --lisn-evidence $RUN_DIR/evidence.jsonl \
  --lisn-cruise-vel 0.8 \
  --lisn-grounded-goal-topic /task_generator_node/Ai2_Bot2/lisn_slow_loop/grounded_goal \
  -- --output-prefix $PREFIX
" > "$LANE_ROOT/logs/${MODE}_eval.stdout.log" 2> "$LANE_ROOT/logs/${MODE}_eval.stderr.log" &
EVAL_HOST_PID=$!
echo "eval_host_pid=$EVAL_HOST_PID" >> "$RUN_DIR/OWNED_PIDS.txt"
sleep 5
EVAL_CONTAINER_PID=$(docker exec arena-arena_jazzy_ws-arena-1 \
  pgrep -f "arena_vln_models.runner eval.*--output-prefix $PREFIX" | head -1)
ISAAC_CANDIDATES=$(docker exec arena-arena_jazzy_ws-isaac-1 pgrep -f run_isaacsim || true)
ISAAC_COUNT=$(printf '%s\n' "$ISAAC_CANDIDATES" | grep -c . || true)
if [ "$ISAAC_COUNT" -eq 1 ]; then
  ISAAC_PID="$ISAAC_CANDIDATES"
else
  ISAAC_PID=""
  echo "isaac_pid_ambiguous:$ISAAC_COUNT" >> "$RUN_DIR/cleanup_errors.log"
fi
echo "eval_container_pid=$EVAL_CONTAINER_PID" >> "$RUN_DIR/OWNED_PIDS.txt"
echo "isaac_pid=$ISAAC_PID" >> "$RUN_DIR/OWNED_PIDS.txt"
wait "$EVAL_HOST_PID"
EVAL_RC=$?
FINAL_RC=$EVAL_RC
echo "eval_rc=$EVAL_RC" >> "$RUN_DIR/lane.rc"
EVAL_DONE_WALL=$(date +%s.%N)
echo "eval_done_wall=$EVAL_DONE_WALL" >> "$RUN_DIR/timing.raw"

if [ "$MODE" = "diag" ]; then
  if [ -n "$RECORDER_PID" ]; then
    docker exec arena-arena_jazzy_ws-arena-1 kill -TERM "$RECORDER_PID" 2>>"$RUN_DIR/cleanup_errors.log" || true
  fi
  sleep 2
  RENDER_START_WALL=$(date +%s.%N)
  echo "render_start_wall=$RENDER_START_WALL" >> "$RUN_DIR/timing.raw"
  docker exec arena-arena_jazzy_ws-arena-1 bash -lc "
source /opt/ros/jazzy/setup.bash
source /opt/arena_ws/install/setup.bash
source $TARGET_INSTALL_ROOT/target_install/setup.bash
source /opt/arena_ws/install_prox/setup.bash
export PYTHONPATH=$WORKTREE/arena_bringup:\$PYTHONPATH
exec /opt/venv/bin/python3 $WORKTREE/arena_bringup/arena_bringup/render_prox_diag_video.py \
  --diag-dir $RUN_DIR --out-dir $RUN_DIR --fps 5.0 --name-suffix v2
" > "$LANE_ROOT/logs/${MODE}_render.log" 2>&1
  RENDER_RC=$?
  echo "render_rc=$RENDER_RC" >> "$RUN_DIR/lane.rc"
  if [ "$RENDER_RC" -ne 0 ]; then FINAL_RC=$RENDER_RC; fi
  echo "render_done_wall=$(date +%s.%N)" >> "$RUN_DIR/timing.raw"
fi
echo "final_rc=$FINAL_RC" >> "$RUN_DIR/lane.rc"
echo "done" >> "$RUN_DIR/lane.rc"
cat > "$RUN_DIR/RUN_RESULT.json" <<JSON
{"schema":"arena.prox_diag.run_result.v1","eval_rc":$EVAL_RC,"render_rc":${RENDER_RC:-0},"final_rc":$FINAL_RC,"run_dir":"$RUN_DIR"}
JSON
exit "$FINAL_RC"
