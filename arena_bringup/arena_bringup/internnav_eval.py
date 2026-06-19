import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime

import yaml
from ament_index_python.packages import get_package_share_directory


REALWORLD_HTTP_ADAPTER_TARGET = 'arena_vln_models.internnav:load_internvla_realworld_http_adapter'
LEGACY_NATIVE_ADAPTER_TARGET = 'arena_vln_models.internnav:load_internnav_adapter'
DEFAULT_INTERNNAV_ADAPTER_TARGET = REALWORLD_HTTP_ADAPTER_TARGET
LEGACY_INTERNNAV_ADAPTER_TARGETS = {
    LEGACY_NATIVE_ADAPTER_TARGET: REALWORLD_HTTP_ADAPTER_TARGET,
    'internnav.agent.internvla_n1_agent_realworld.InternVLAN1AsyncAgent': REALWORLD_HTTP_ADAPTER_TARGET,
}
HTTP_ADAPTER_REPLACED_TARGETS = {
    LEGACY_NATIVE_ADAPTER_TARGET,
    *LEGACY_INTERNNAV_ADAPTER_TARGETS.keys(),
}
GENERIC_VLN_INSTRUCTIONS = {'', 'navigate', 'go', 'start', 'default', 'none', 'null'}


def _write_yaml(path: str, data) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_text(path: str, data: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


def _copy_if_exists(src: str, dst: str) -> str | None:
    if not src or not os.path.exists(src):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _workspace_root_from_share(package_share_dir: str) -> str:
    """Return the Arena workspace root used for human-facing eval outputs.

    ROS package share directories live under ``<ws>/install/<pkg>/share/<pkg>``.
    Historical Arena evaluation code used that share path as its data root, which
    made generated videos appear inside ``install/``.  Eval artifacts are user
    outputs, not installed package resources, so default them to ``<ws>/outputs``.
    """
    for env_name in ('ARENA_OUTPUT_WORKSPACE', 'ARENA_WS_DIR', 'HOST_ARENA_WS_DIR', 'COLCON_PREFIX_PATH'):
        value = os.environ.get(env_name, '').strip()
        if not value:
            continue
        candidate = value.split(os.pathsep)[0] if env_name == 'COLCON_PREFIX_PATH' else value
        if env_name == 'COLCON_PREFIX_PATH' and os.path.basename(candidate) == 'install':
            candidate = os.path.dirname(candidate)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    marker = os.path.join(os.sep, 'install', 'arena_evaluation', 'share', 'arena_evaluation')
    abs_share = os.path.abspath(package_share_dir)
    if abs_share.endswith(marker):
        return abs_share[:-len(marker)] or os.path.sep

    return os.getcwd()


def _resolve_output_root(output_root_arg: str, package_share_dir: str) -> str:
    if output_root_arg:
        root = output_root_arg
        if not os.path.isabs(root):
            root = os.path.join(_workspace_root_from_share(package_share_dir), root)
        return os.path.abspath(root)

    return os.path.join(_workspace_root_from_share(package_share_dir), 'outputs')


def _first_env_value(*names: str) -> tuple[str, str]:
    for name in names:
        value = str(os.environ.get(name, '')).strip()
        if value:
            return value, name
    return '', ''


def _eval_python_executable(env: dict[str, str]) -> str:
    for candidate in (
        str(env.get('ARENA_EVAL_PYTHON', '')).strip(),
        str(sys.executable).strip(),
        '/usr/bin/python3',
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return '/usr/bin/python3'


def _start_finished_watcher(
    env: dict[str, str],
    topic: str,
    task_reset_topic: str,
    scenario_reset_topic: str,
) -> subprocess.Popen:
    python_bin = _eval_python_executable(env)
    watcher_code = r'''
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Empty
from std_msgs.msg import Int16

topic = sys.argv[1]
task_reset_topic = sys.argv[2]
scenario_reset_topic = sys.argv[3]
rclpy.init()
node = Node('internnav_eval_finished_watcher')
qos = QoSProfile(depth=1)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
done = {'seen': False, 'reset_seen': False, 'last_reset_monotonic': 0.0}

def _cb(_msg):
    if done['reset_seen'] and (time.monotonic() - done['last_reset_monotonic']) >= 2.0:
        done['seen'] = True

def _on_reset(_msg):
    done['reset_seen'] = True
    done['last_reset_monotonic'] = time.monotonic()

node.create_subscription(Empty, topic, _cb, qos)
node.create_subscription(Int16, task_reset_topic, _on_reset, 10)
node.create_subscription(Int16, scenario_reset_topic, _on_reset, 10)
try:
    while rclpy.ok() and not done['seen']:
        rclpy.spin_once(node, timeout_sec=0.5)
finally:
    node.destroy_node()
    rclpy.shutdown()

raise SystemExit(0 if done['seen'] else 1)
'''

    return subprocess.Popen(
        [python_bin, '-c', watcher_code, topic, task_reset_topic, scenario_reset_topic],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _start_status_watcher(
    env: dict[str, str],
    topic: str,
    output_path: str,
    history_path: str | None = None,
    *,
    task_reset_topic: str = '',
    scenario_reset_topic: str = '',
    require_reset_for_history: bool = False,
) -> subprocess.Popen:
    python_bin = _eval_python_executable(env)
    watcher_code = r'''
import json
import sys
import rclpy
from pathlib import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Int16
from std_msgs.msg import String

topic = sys.argv[1]
output_path = Path(sys.argv[2])
history_path = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
task_reset_topic = sys.argv[4] if len(sys.argv) > 4 else ''
scenario_reset_topic = sys.argv[5] if len(sys.argv) > 5 else ''
require_reset_for_history = (sys.argv[6].strip().lower() in {'1', 'true', 'yes'}) if len(sys.argv) > 6 else False
rclpy.init()
node = Node('internnav_eval_status_watcher')
qos = QoSProfile(depth=1)
qos.reliability = ReliabilityPolicy.RELIABLE
qos.durability = DurabilityPolicy.VOLATILE
state = {'last': None, 'reset_seen': False}

def _on_reset(_msg):
    state['reset_seen'] = True

def _cb(msg):
    state['last'] = msg.data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(msg.data, encoding='utf-8')
    if history_path is not None:
        if require_reset_for_history and not state['reset_seen']:
            return
        history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            parsed = json.loads(msg.data)
        except Exception:
            parsed = None
        event_type = 'model_result'
        if isinstance(parsed, dict) and parsed.get('status'):
            event_type = str(parsed.get('status'))
        record = {
            'wall_time': node.get_clock().now().nanoseconds / 1e9,
            'topic': topic,
            'event_type': event_type,
            'raw': msg.data,
            'parsed': parsed,
        }
        with history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

node.create_subscription(String, topic, _cb, qos)
if task_reset_topic:
    node.create_subscription(Int16, task_reset_topic, _on_reset, 10)
if scenario_reset_topic and scenario_reset_topic != task_reset_topic:
    node.create_subscription(Int16, scenario_reset_topic, _on_reset, 10)
try:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()

raise SystemExit(0)
'''

    return subprocess.Popen(
        [
            python_bin,
            '-c',
            watcher_code,
            topic,
            output_path,
            history_path or '',
            task_reset_topic,
            scenario_reset_topic,
            '1' if require_reset_for_history else '0',
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _read_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _read_text_if_exists(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def _is_generic_vln_instruction(value: str | None) -> bool:
    return str(value or '').strip().lower() in GENERIC_VLN_INSTRUCTIONS


def _workspace_root_from_runtime() -> str:
    for env_name in ('ARENA_WS_DIR', 'HOST_ARENA_WS_DIR', 'ARENA_OUTPUT_WORKSPACE'):
        value = str(os.environ.get(env_name, '') or '').strip()
        if value and os.path.isdir(value):
            return os.path.abspath(value)
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, 'src', 'Arena')):
        return os.path.abspath(cwd)
    return os.path.abspath(cwd)


def _scenario_key(value: str | None) -> str:
    raw = str(value or '').strip()
    if not raw:
        return ''
    base = os.path.basename(raw)
    if base in {'scenario.yaml', 'scenario.yml', 'scenario.json'}:
        parent = os.path.basename(os.path.dirname(raw.rstrip(os.sep)))
        return parent or os.path.splitext(base)[0]
    stem, ext = os.path.splitext(base)
    return stem if ext in {'.yaml', '.yml'} else base


def _candidate_grscenes_instruction_manifests(workspace_root: str) -> list[str]:
    candidates = []
    for value in (
        os.environ.get('ARENA_GRSCENES_INSTRUCTION_MANIFEST', ''),
        os.environ.get('GRSCENES_INSTRUCTION_MANIFEST', ''),
    ):
        value = str(value or '').strip()
        if value:
            candidates.append(value)
    candidates.extend([
        os.path.join(
            workspace_root,
            'data',
            'grscenes_trajectories',
            '20260609_uploaded_instructions_txt',
            'uploaded_grscenes_test_entries.json',
        ),
        os.path.join(
            workspace_root,
            'data',
            'grscenes_trajectories',
            'uploaded_grscenes_test_entries.json',
        ),
    ])
    unique = []
    seen = set()
    for path in candidates:
        normalized = os.path.abspath(os.path.expanduser(path))
        if normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return unique


def _resolve_existing_manifest_path(manifest_arg: str, workspace_root: str) -> tuple[str, list[str]]:
    attempts = []
    if manifest_arg:
        raw = os.path.expanduser(str(manifest_arg).strip())
        candidates = [raw if os.path.isabs(raw) else os.path.join(workspace_root, raw)]
    else:
        candidates = _candidate_grscenes_instruction_manifests(workspace_root)
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        attempts.append(normalized)
        if os.path.exists(normalized):
            return normalized, attempts
    return '', attempts


def _lookup_grscenes_instruction_from_manifest(
    manifest_path: str,
    *,
    world: str,
    scenario: str,
    episode: str,
    timestamp: str,
) -> dict:
    data = _read_json_if_exists(manifest_path)
    if not isinstance(data, list):
        return {'ok': False, 'reason': 'manifest_not_list', 'manifest_path': manifest_path}

    world_key = str(world or '').strip()
    scenario_key = _scenario_key(scenario)
    episode_key = str(episode or '').strip()
    timestamp_key = str(timestamp or '').strip()

    matches = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if world_key and str(entry.get('world') or '').strip() != world_key:
            continue
        if scenario_key and str(entry.get('scenario') or '').strip() != scenario_key:
            continue
        if episode_key and str(entry.get('episode') or '').strip() != episode_key:
            continue
        if timestamp_key and str(entry.get('timestamp') or '').strip() != timestamp_key:
            continue
        instruction = str(entry.get('instruction') or '').strip()
        if instruction:
            matches.append(entry)

    if not matches:
        return {
            'ok': False,
            'reason': 'no_matching_instruction',
            'manifest_path': manifest_path,
            'world': world_key,
            'scenario': scenario_key,
            'episode': episode_key,
            'timestamp': timestamp_key,
        }

    selected = matches[0]
    return {
        'ok': True,
        'manifest_path': manifest_path,
        'match_count': len(matches),
        'ambiguous': len(matches) > 1,
        'world': str(selected.get('world') or ''),
        'scenario': str(selected.get('scenario') or ''),
        'episode': str(selected.get('episode') or ''),
        'timestamp': str(selected.get('timestamp') or ''),
        'instruction_file': str(selected.get('instruction_file') or ''),
        'instruction': str(selected.get('instruction') or '').strip(),
    }


def _wait_for_file(path: str, timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.05)
    return os.path.exists(path)


def _normalize_external_ros_env(env: dict[str, str]) -> dict[str, dict[str, str]]:
    """Resolve ROS discovery defaults used by the three-container eval path."""
    defaults = {
        'ROS_DOMAIN_ID': '1',
        'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
        'ROS_LOCALHOST_ONLY': '0',
    }
    resolved: dict[str, dict[str, str]] = {}
    for key, default in defaults.items():
        current = str(env.get(key, '')).strip()
        if current:
            resolved[key] = {'value': current, 'source': 'environment'}
            continue
        env[key] = default
        resolved[key] = {'value': default, 'source': 'default'}
    return resolved


def _truncate_preflight_output(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f'\n... <truncated {omitted} chars>'


def _run_ros_list_command(env: dict[str, str], args: list[str], timeout_sec: float) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_sec), 0.1),
        )
        return {
            'command': args,
            'returncode': result.returncode,
            'stdout': _truncate_preflight_output(result.stdout),
            'stderr': _truncate_preflight_output(result.stderr),
            'duration_sec': time.monotonic() - started,
            'timed_out': False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        stderr = exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
        return {
            'command': args,
            'returncode': None,
            'stdout': _truncate_preflight_output(stdout),
            'stderr': _truncate_preflight_output(stderr),
            'duration_sec': time.monotonic() - started,
            'timed_out': True,
        }
    except Exception as exc:
        return {
            'command': args,
            'returncode': None,
            'stdout': '',
            'stderr': repr(exc),
            'duration_sec': time.monotonic() - started,
            'timed_out': False,
        }


def _run_external_rclpy_discovery_probe(
    env: dict[str, str],
    *,
    expected_service: str,
    candidate_services: list[str] | None = None,
    expected_status_topic: str,
    timeout_sec: float,
) -> dict:
    python_bin = _eval_python_executable(env)
    probe_code = r'''
import json
import sys
import time

import rclpy
from rclpy.node import Node
from rosnav_rl_msgs.srv import GetCommand

expected_service = sys.argv[1]
candidate_services = [name for name in json.loads(sys.argv[2]) if name]
expected_status_topic = sys.argv[3]
timeout_sec = float(sys.argv[4])

rclpy.init(args=None)
node = Node('internnav_eval_external_preflight_probe')
deadline = time.monotonic() + max(timeout_sec, 0.0)
attempts = 0
services = []
topics = []
publisher_count = 0
observed_service = ''
service_response_ok = False
checks = {
    'service_visible': False,
    'status_topic_visible': False,
    'service_responds': False,
}

try:
    while True:
        attempts += 1
        rclpy.spin_once(node, timeout_sec=0.05)
        services = sorted(name for name, _types in node.get_service_names_and_types())
        topics = sorted(name for name, _types in node.get_topic_names_and_types())
        publisher_count = int(node.count_publishers(expected_status_topic) or 0)
        observed_service = next((name for name in candidate_services if name in services), '')
        service_response_ok = False
        if observed_service:
            try:
                client = node.create_client(GetCommand, observed_service)
                if client.wait_for_service(timeout_sec=0.0):
                    future = client.call_async(GetCommand.Request())
                    call_deadline = time.monotonic() + 1.0
                    while time.monotonic() < call_deadline:
                        rclpy.spin_once(node, timeout_sec=0.05)
                        if future.done():
                            try:
                                response = future.result()
                                service_response_ok = response is not None and getattr(response, 'twist', None) is not None
                            except Exception:
                                service_response_ok = False
                            break
                node.destroy_client(client)
            except Exception:
                service_response_ok = False
        checks = {
            'service_visible': bool(observed_service),
            'status_topic_visible': expected_status_topic in topics and publisher_count > 0,
            'service_responds': service_response_ok,
        }
        if all(checks.values()) or time.monotonic() >= deadline:
            break
        time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))
finally:
    node.destroy_node()
    rclpy.shutdown()

print(json.dumps({
    'attempts': attempts,
    'checks': checks,
    'candidate_services': candidate_services,
    'observed_service': observed_service,
    'service_response_ok': service_response_ok,
    'status_topic_publisher_count': publisher_count,
    'services': services,
    'topics': topics,
}))
'''

    started = time.monotonic()
    service_candidates = candidate_services or [expected_service]
    encoded_candidates = json.dumps(service_candidates)
    try:
        result = subprocess.run(
            [python_bin, '-c', probe_code, expected_service, encoded_candidates, expected_status_topic, str(timeout_sec)],
            env=env,
            capture_output=True,
            text=True,
            timeout=max(float(timeout_sec), 0.1) + 5.0,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        stderr = exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
        return {
            'command': [python_bin, '-c', '<rclpy_external_preflight_probe>', expected_service, expected_status_topic, str(timeout_sec)],
            'returncode': None,
            'stdout': _truncate_preflight_output(stdout),
            'stderr': _truncate_preflight_output(stderr),
            'duration_sec': time.monotonic() - started,
            'timed_out': True,
            'probe_error': 'timeout',
        }
    except Exception as exc:
        return {
            'command': [python_bin, '-c', '<rclpy_external_preflight_probe>', expected_service, expected_status_topic, str(timeout_sec)],
            'returncode': None,
            'stdout': '',
            'stderr': repr(exc),
            'duration_sec': time.monotonic() - started,
            'timed_out': False,
            'probe_error': repr(exc),
        }

    payload = None
    stdout_text = str(result.stdout or '')
    stdout_lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    json_candidates = []
    if stdout_text.strip():
        json_candidates.append(stdout_text.strip())
    json_candidates.extend(reversed(stdout_lines))
    for candidate in json_candidates:
        if not candidate.startswith('{'):
            continue
        try:
            payload = json.loads(candidate)
            break
        except Exception:
            continue

    return {
        'command': [python_bin, '-c', '<rclpy_external_preflight_probe>', expected_service, '<candidate_services>', expected_status_topic, str(timeout_sec)],
        'returncode': result.returncode,
        'stdout': _truncate_preflight_output(result.stdout),
        'stderr': _truncate_preflight_output(result.stderr),
        'duration_sec': time.monotonic() - started,
        'timed_out': False,
        'payload': payload,
    }


def _listed_names(command_result: dict) -> set[str]:
    return {
        line.strip()
        for line in str(command_result.get('stdout') or '').splitlines()
        if line.strip()
    }


def _topic_publisher_count(command_result: dict) -> int | None:
    for line in str(command_result.get('stdout') or '').splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith('publisher count:'):
            continue
        try:
            return int(stripped.split(':', 1)[1].strip())
        except Exception:
            return None
    return None


def _run_external_internnav_preflight(
    env: dict[str, str],
    *,
    expected_service: str,
    candidate_services: list[str] | None = None,
    expected_status_topic: str,
    timeout_sec: float,
) -> dict:
    probe_result = _run_external_rclpy_discovery_probe(
        env,
        expected_service=expected_service,
        candidate_services=candidate_services,
        expected_status_topic=expected_status_topic,
        timeout_sec=timeout_sec,
    )
    payload = probe_result.get('payload') if isinstance(probe_result.get('payload'), dict) else {}
    services = payload.get('services') if isinstance(payload.get('services'), list) else []
    topics = payload.get('topics') if isinstance(payload.get('topics'), list) else []
    checks = payload.get('checks') if isinstance(payload.get('checks'), dict) else {
        'service_visible': expected_service in services,
        'status_topic_visible': expected_status_topic in topics,
        'service_responds': False,
    }
    attempts = int(payload.get('attempts') or 1)
    observed_service = str(payload.get('observed_service') or '')
    service_candidates = payload.get('candidate_services') if isinstance(payload.get('candidate_services'), list) else (candidate_services or [expected_service])
    status_topic_publisher_count = payload.get('status_topic_publisher_count')
    if status_topic_publisher_count is None and checks.get('status_topic_visible'):
        status_topic_publisher_count = 1

    missing = [name for name, passed in checks.items() if not passed]
    return {
        'pass': not missing,
        'timeout_sec': timeout_sec,
        'attempts': attempts,
        'expected_service': expected_service,
        'candidate_services': service_candidates,
        'observed_service': observed_service,
        'expected_status_topic': expected_status_topic,
        'status_topic_publisher_count': status_topic_publisher_count,
        'checks': checks,
        'missing_checks': missing,
        'service_list': {
            'backend': 'rclpy_discovery_probe',
            'command': probe_result.get('command'),
            'returncode': probe_result.get('returncode'),
            'stdout': probe_result.get('stdout'),
            'stderr': probe_result.get('stderr'),
            'duration_sec': probe_result.get('duration_sec'),
            'timed_out': probe_result.get('timed_out'),
            'observed_count': len(services),
            'sample': services[:50],
        },
        'topic_list': {
            'backend': 'rclpy_discovery_probe',
            'command': probe_result.get('command'),
            'returncode': probe_result.get('returncode'),
            'stdout': probe_result.get('stdout'),
            'stderr': probe_result.get('stderr'),
            'duration_sec': probe_result.get('duration_sec'),
            'timed_out': probe_result.get('timed_out'),
            'observed_count': len(topics),
            'sample': topics[:50],
        },
        'topic_info': {
            'backend': 'rclpy_discovery_probe',
            'command': probe_result.get('command'),
            'returncode': probe_result.get('returncode'),
            'stdout': probe_result.get('stdout'),
            'stderr': probe_result.get('stderr'),
            'duration_sec': probe_result.get('duration_sec'),
            'timed_out': probe_result.get('timed_out'),
            'publisher_count': status_topic_publisher_count,
            'probe_error': probe_result.get('probe_error'),
        },
    }


def _external_command_service_candidates(canonical_service: str, status_topic: str) -> list[str]:
    candidates: list[str] = []

    def add(name: str) -> None:
        normalized = '/' + str(name or '').strip().strip('/')
        if normalized != '/' and normalized not in candidates:
            candidates.append(normalized)

    add(canonical_service)
    status = '/' + str(status_topic or '').strip().strip('/')
    if status.endswith('/internnav/status'):
        add(status[: -len('/internnav/status')] + '/get_command')
    if status.endswith('/status'):
        add(status[: -len('/status')] + '/get_command')
    add('/get_command')
    add('/internnav_server/get_command')
    return candidates


def _read_jsonl(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _write_internnav_diagnostic_summary(trace_path: str, output_path: str) -> dict | None:
    records = _read_jsonl(trace_path)
    if not records:
        return None

    action_counts: dict[str, int] = {}
    no_action_status_counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    rotate_count = 0
    forward_count = 0
    stop_count = 0
    control_record_count = 0
    model_command_record_count = 0
    pre_episode_ready_records = 0
    yaw_sign_mismatch = 0
    yaw_sign_check_count = 0
    stale_records = 0
    missing_rgb = 0
    missing_depth = 0
    official_primitive_count = 0
    official_discrete_selected_count = 0
    queued_action_tail_record_count = 0
    dropped_action_tail_record_count = 0
    dropped_action_tail_count = 0
    action_effect_count = 0
    yaw_sign_action_effect_match = 0
    yaw_sign_action_effect_mismatch = 0
    llm_raw_present_count = 0
    llm_raw_has_digits_count = 0
    llm_digits_present_count = 0
    llm_digits_non_empty_count = 0
    llm_raw_has_digits_but_digits_empty_count = 0
    llm_symbolic_output_count = 0
    llm_pixel_goal_output_count = 0
    goal_distances = []
    final_goal_distances = []
    yaw_errors = []
    commands = []
    event_counts: dict[str, int] = {}
    for rec in records:
        event_type = str(rec.get('event_type') or rec.get('event') or 'model_result')
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        payload = rec.get('parsed') if isinstance(rec.get('parsed'), dict) else rec
        debug = payload.get('debug') if isinstance(payload.get('debug'), dict) else {}
        llm = payload.get('llm') if isinstance(payload.get('llm'), dict) else {}
        raw_llm = (
            llm.get('raw_output_text')
            or debug.get('raw_output_text')
            or debug.get('subprocess_llm_output')
            or debug.get('adapter_llm_output')
            or debug.get('llm_output')
        )
        llm_digits = llm.get('llm_digits') if 'llm_digits' in llm else debug.get('llm_digits', debug.get('digit_groups'))
        if raw_llm:
            llm_raw_present_count += 1
        raw_has_digits = bool(raw_llm and re.search(r'\d+', str(raw_llm)))
        if raw_has_digits:
            llm_raw_has_digits_count += 1
        if llm_digits is not None:
            llm_digits_present_count += 1
        digits_non_empty = isinstance(llm_digits, list) and len(llm_digits) > 0
        if digits_non_empty:
            llm_digits_non_empty_count += 1
        if raw_has_digits and not digits_non_empty:
            llm_raw_has_digits_but_digits_empty_count += 1
        output_mode = str(llm.get('output_mode') or debug.get('model_generation_output_mode') or '')
        if output_mode == 'symbolic_action':
            llm_symbolic_output_count += 1
        elif output_mode == 'pixel_goal':
            llm_pixel_goal_output_count += 1
        status = str(payload.get('status', ''))
        statuses[status] = statuses.get(status, 0) + 1

        # ``backend_ready`` status samples are intentionally emitted by the
        # external InternNav server before the episode starts.  They prove the
        # backend is alive, but they often contain old sensor-age diagnostics
        # from the previous run and no command/action.  Keep them in the raw
        # status counts, but exclude them from command and fault ratios so the
        # summary reflects episode-control health instead of startup readiness.
        if status == 'backend_ready':
            pre_episode_ready_records += 1
            continue

        control_record_count += 1
        if status == 'internnav_command':
            model_command_record_count += 1
        action_info = payload.get('action') if isinstance(payload.get('action'), dict) else {}
        if action_info.get('official_discrete_selected') or debug.get('official_discrete_selected'):
            official_discrete_selected_count += 1
        if action_info.get('official_discrete_primitive') or debug.get('official_discrete_primitive'):
            official_primitive_count += 1
        queued_tail = action_info.get('queued_action_sequence_tail', debug.get('queued_action_sequence_tail'))
        if isinstance(queued_tail, list):
            queued_action_tail_record_count += 1
        dropped_tail = action_info.get('dropped_action_sequence_tail', debug.get('dropped_action_sequence_tail'))
        if isinstance(dropped_tail, list):
            dropped_action_tail_record_count += 1
            dropped_action_tail_count += len(dropped_tail)
        action_effect = payload.get('action_effect', debug.get('action_effect'))
        if isinstance(action_effect, dict):
            action_effect_count += 1
            sign_match = action_effect.get('yaw_sign_matches_action')
            if sign_match is True:
                yaw_sign_action_effect_match += 1
            elif sign_match is False:
                yaw_sign_action_effect_mismatch += 1
        action = action_info.get('selected')
        if action is None:
            action = debug.get('selected_action')
        if action is None:
            no_action_status_counts[status] = no_action_status_counts.get(status, 0) + 1
        else:
            key = str(action)
            action_counts[key] = action_counts.get(key, 0) + 1
        cmd = payload.get('command') if isinstance(payload.get('command'), dict) else {}
        if not cmd and ('linear_x' in payload or 'angular_z' in payload):
            cmd = payload
        if not cmd and ('desired_v' in payload or 'desired_w' in payload):
            cmd = {
                'linear_x': payload.get('desired_v', 0.0),
                'angular_z': payload.get('desired_w', 0.0),
            }
        vx = float(cmd.get('linear_x', 0.0) or 0.0)
        wz = float(cmd.get('angular_z', 0.0) or 0.0)
        commands.append({'linear_x': vx, 'angular_z': wz})
        if abs(wz) > 0.05 and abs(vx) < 0.45:
            rotate_count += 1
        if vx > 0.05:
            forward_count += 1
        if abs(vx) <= 0.01 and abs(wz) <= 0.01:
            stop_count += 1
        goal = payload.get('goal') if isinstance(payload.get('goal'), dict) else {}
        dist = goal.get('goal_distance', debug.get('goal_distance'))
        final_dist = debug.get('final_goal_distance')
        yaw = goal.get('yaw_error', debug.get('yaw_error'))
        if isinstance(dist, (float, int)):
            goal_distances.append(float(dist))
        if isinstance(final_dist, (float, int)):
            final_goal_distances.append(float(final_dist))
        if isinstance(yaw, (float, int)):
            yaw_errors.append(float(yaw))
            if abs(float(yaw)) > 0.25 and abs(wz) > 0.05:
                yaw_sign_check_count += 1
                if (float(yaw) * wz) < 0.0:
                    yaw_sign_mismatch += 1
        obs = payload.get('observation') if isinstance(payload.get('observation'), dict) else {}
        missing_inputs = debug.get('missing_inputs') if isinstance(debug.get('missing_inputs'), list) else []
        rgb_available = obs.get('rgb_available', debug.get('rgb_available'))
        depth_available = obs.get('depth_available', debug.get('depth_available'))
        if rgb_available is False or 'rgb' in missing_inputs:
            missing_rgb += 1
        if depth_available is False or 'depth' in missing_inputs:
            missing_depth += 1
        ages = obs.get('sensor_ages_sec') if isinstance(obs.get('sensor_ages_sec'), dict) else {}
        if not ages and isinstance(debug.get('sensor_ages_sec'), dict):
            ages = debug.get('sensor_ages_sec')
        stale_after = float(obs.get('stale_after_sec', debug.get('stale_after_sec', 0.0)) or 0.0)
        if stale_after > 0.0:
            for value in ages.values():
                if isinstance(value, (float, int)) and float(value) > stale_after:
                    stale_records += 1
                    break

    total = len(records)
    first_distance = goal_distances[0] if goal_distances else None
    last_distance = goal_distances[-1] if goal_distances else None
    min_distance = min(goal_distances) if goal_distances else None
    progress = (first_distance - last_distance) if first_distance is not None and last_distance is not None else None
    first_final_distance = final_goal_distances[0] if final_goal_distances else None
    last_final_distance = final_goal_distances[-1] if final_goal_distances else None
    min_final_distance = min(final_goal_distances) if final_goal_distances else None
    final_progress = (
        first_final_distance - last_final_distance
        if first_final_distance is not None and last_final_distance is not None
        else None
    )
    ratio_total = control_record_count or total
    rotate_ratio = rotate_count / ratio_total if ratio_total else 0.0
    forward_ratio = forward_count / ratio_total if ratio_total else 0.0
    yaw_sign_mismatch_ratio = yaw_sign_mismatch / yaw_sign_check_count if yaw_sign_check_count else 0.0
    flags = []
    if rotate_ratio >= 0.65 and (progress is None or progress < 0.5):
        flags.append('rotate_heavy_low_progress')
    if yaw_sign_mismatch_ratio >= 0.45:
        flags.append('possible_action_or_yaw_sign_mismatch')
    if ratio_total and stale_records / ratio_total >= 0.2:
        flags.append('possible_stale_observations')
    if missing_rgb or missing_depth:
        flags.append('missing_camera_inputs')
    if official_discrete_selected_count and official_primitive_count == 0:
        flags.append('missing_official_discrete_primitive_conversion')
    if dropped_action_tail_count:
        flags.append('official_discrete_action_tail_dropped')
    if llm_raw_has_digits_but_digits_empty_count:
        flags.append('llm_digits_drop_suspected')

    summary = {
        'trace_path': trace_path,
        'record_count': total,
        'control_record_count': control_record_count,
        'model_command_record_count': model_command_record_count,
        'pre_episode_ready_records': pre_episode_ready_records,
        'event_counts': event_counts,
        'action_counts': action_counts,
        'no_action_status_counts': no_action_status_counts,
        'status_counts': statuses,
        'command_stats': {
            'rotate_count': rotate_count,
            'rotate_ratio': rotate_ratio,
            'forward_count': forward_count,
            'forward_ratio': forward_ratio,
            'stop_count': stop_count,
        },
        'official_discrete_stats': {
            'official_discrete_selected_count': official_discrete_selected_count,
            'official_discrete_primitive_count': official_primitive_count,
            'queued_action_tail_record_count': queued_action_tail_record_count,
            'dropped_action_tail_record_count': dropped_action_tail_record_count,
            'dropped_action_tail_count': dropped_action_tail_count,
            'action_effect_count': action_effect_count,
            'yaw_sign_action_effect_match_count': yaw_sign_action_effect_match,
            'yaw_sign_action_effect_mismatch_count': yaw_sign_action_effect_mismatch,
        },
        'llm_trace_stats': {
            'raw_llm_present_count': llm_raw_present_count,
            'raw_llm_has_digits_count': llm_raw_has_digits_count,
            'llm_digits_present_count': llm_digits_present_count,
            'llm_digits_non_empty_count': llm_digits_non_empty_count,
            'raw_has_digits_but_llm_digits_empty_count': llm_raw_has_digits_but_digits_empty_count,
            'symbolic_output_count': llm_symbolic_output_count,
            'pixel_goal_output_count': llm_pixel_goal_output_count,
        },
        'goal_distance': {
            'first': first_distance,
            'last': last_distance,
            'min': min_distance,
            'progress_first_minus_last': progress,
        },
        'final_goal_distance': {
            'first': first_final_distance,
            'last': last_final_distance,
            'min': min_final_distance,
            'progress_first_minus_last': final_progress,
        },
        'fault_candidates': {
            'flags': flags,
            'yaw_sign_check_count': yaw_sign_check_count,
            'yaw_sign_mismatch_count': yaw_sign_mismatch,
            'yaw_sign_mismatch_ratio': yaw_sign_mismatch_ratio,
            'stale_record_count': stale_records,
            'missing_rgb_count': missing_rgb,
            'missing_depth_count': missing_depth,
        },
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def _is_internnav_run(args) -> bool:
    adapter_target = str(args.dual_vln_adapter_target or '').lower()
    mode = str(args.dual_vln_mode or '').lower()
    model_path = str(args.dual_vln_model_path or '').lower()
    http_url = str(
        getattr(args, 'dual_vln_http_url', '')
        or os.environ.get('ARENA_EVAL_INTERNNAV_HTTP_URL')
        or os.environ.get('ARENA_INTERNNAV_HTTP_URL')
        or ''
    ).strip()
    return (
        'internnav' in adapter_target
        or mode == 'internnav'
        or ('internvla' in model_path or 'internnav' in model_path)
        or bool(http_url)
    )


def _normalize_internnav_adapter_target(adapter_target: str) -> tuple[str, str | None]:
    normalized = str(adapter_target or '').strip()
    if not normalized:
        return DEFAULT_INTERNNAV_ADAPTER_TARGET, 'default'

    mapped = LEGACY_INTERNNAV_ADAPTER_TARGETS.get(normalized)
    if mapped is not None:
        return mapped, f'legacy:{normalized}'

    return normalized, None


def _default_vision_topics(robot: str) -> tuple[str, str, str] | None:
    normalized = str(robot or '').strip().lower()
    if normalized == 'turtlebot':
        return ('rgbd_camera/image', 'rgbd_camera/depth_image', 'rgbd_camera/camera_info')
    if normalized == 'ai2_bot2':
        return ('head_camera/image', 'head_camera/depth', 'head_camera/camera_info')
    if normalized == 'linkhou_s2':
        return ('head_camera/image', 'head_camera/depth', 'head_camera/camera_info')
    return None


def _default_eval_video_sim_top_down_topic(robot: str) -> str:
    normalized = str(robot or '').strip().lower()
    if normalized in {'turtlebot', 'ai2_bot2', 'linkhou_s2'}:
        return 'top_down_camera/image'
    return ''


def _default_eval_video_debug_overlay_topic() -> str:
    return 'internnav/debug_image'


def _task_root_from_topic(topic: str) -> str:
    normalized = '/' + str(topic or '').strip().strip('/')
    if normalized.endswith('/task_reset'):
        return normalized[: -len('/task_reset')]
    if normalized.endswith('/finished'):
        return normalized[: -len('/finished')]
    return normalized.rstrip('/')


def _robot_topic(task_reset_topic: str, robot: str, topic: str) -> str:
    normalized_topic = str(topic or '').strip()
    if not normalized_topic:
        return ''
    if normalized_topic.startswith('/'):
        return normalized_topic
    root = _task_root_from_topic(task_reset_topic)
    return f'{root}/{robot}/{normalized_topic.strip("/")}'


def _scenario_reset_topic(task_reset_topic: str, robot: str) -> str:
    # The current Arena benchmark path publishes task resets on the task
    # generator root topic and does not provide a reliable per-robot
    # scenario_reset stream during eval.  Reuse task_reset as the canonical
    # episode boundary so watchers/recorders do not wait on a dead topic.
    return task_reset_topic


def _world_map_yaml_path(sim_setup_share: str, world: str) -> str:
    return os.path.join(sim_setup_share, 'worlds', world, 'map', 'map.yaml')


def _start_eval_video_recorder(
    env: dict[str, str],
    *,
    output_dir: str,
    map_yaml_path: str,
    task_reset_topic: str,
    scenario_reset_topic: str,
    finished_topic: str,
    ego_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    debug_overlay_topic: str,
    sim_top_down_topic: str,
    odom_topic: str,
    goal_topic: str,
    scan_topic: str,
    fps: float,
    top_down_size_px: int,
    top_down_window_m: float,
) -> subprocess.Popen:
    python_bin = _eval_python_executable(env)
    recorder_code = r'''
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from PIL import ImageDraw
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Empty, Int16


try:
    _PIL_BILINEAR = PILImage.Resampling.BILINEAR
except AttributeError:
    _PIL_BILINEAR = PILImage.BILINEAR


def _load_video_backend():
    # Prefer OpenCV when available. Some container images include the imageio
    # package but not the ffmpeg/pyav writer plugins, which makes get_writer()
    # fail only after the episode starts and leaves empty video artifacts.
    try:
        import cv2  # type: ignore
        return 'cv2', cv2
    except Exception:
        pass
    try:
        import imageio.v2 as imageio  # type: ignore
        return 'imageio', imageio
    except Exception:
        pass
    try:
        import imageio  # type: ignore
        return 'imageio', imageio
    except Exception:
        pass
    return None, None


BACKEND_NAME, BACKEND_MODULE = _load_video_backend()


def _codec_is_h264(codec_name):
    if not codec_name:
        return False
    normalized = str(codec_name).strip().lower().replace('.', '')
    return normalized in {'h264', 'avc1', 'x264', 'libx264'}


def _probe_video_codec(path: Path):
    ffprobe_bin = shutil.which('ffprobe')
    if ffprobe_bin is None or not path.exists():
        return None
    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    for line in result.stdout.splitlines():
        value = line.strip()
        if value:
            return value
    return None


def _transcode_to_h264(path: Path):
    ffmpeg_bin = shutil.which('ffmpeg')
    if ffmpeg_bin is None:
        return False, 'ffmpeg not found on PATH'
    if not path.exists():
        return False, f'video file does not exist: {path}'
    temp_path = path.with_name(f'{path.stem}.h264tmp{path.suffix}')
    if temp_path.exists():
        temp_path.unlink()
    try:
        result = subprocess.run(
            [
                ffmpeg_bin,
                '-y',
                '-i', str(path),
                '-an',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                str(temp_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or '').strip()
        return False, stderr or f'ffmpeg failed with return code {exc.returncode}'
    except Exception as exc:
        return False, str(exc)

    if not temp_path.exists():
        return False, 'ffmpeg did not create transcoded output'

    temp_path.replace(path)
    detected = _probe_video_codec(path)
    if _codec_is_h264(detected):
        return True, detected or 'h264'

    stderr = (result.stderr or '').strip()
    return False, stderr or f'transcoded file codec is {detected!r}, expected h264'


class VideoWriterWrapper:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = max(float(fps), 1.0)
        self._writer = None
        self._size = None
        self.codec = 'libx264' if BACKEND_NAME == 'imageio' else None
        self.actual_codec = None
        self.transcode_error = None

    def _ensure_writer(self, frame: np.ndarray):
        if self._writer is not None:
            return
        height, width = frame.shape[:2]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._size = (width, height)
        if BACKEND_NAME == 'imageio':
            self._writer = BACKEND_MODULE.get_writer(
                str(self.path),
                fps=self.fps,
                codec='libx264',
                macro_block_size=1,
                ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'],
            )
            return
        if BACKEND_NAME == 'cv2':
            for codec in ('avc1', 'H264', 'X264', 'mp4v'):
                fourcc = BACKEND_MODULE.VideoWriter_fourcc(*codec)
                writer = BACKEND_MODULE.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
                if writer.isOpened():
                    self._writer = writer
                    self.codec = codec
                    return
                writer.release()
            raise RuntimeError(f'failed to open cv2 mp4 writer for {self.path}')
            return
        raise RuntimeError('no supported video backend available (need cv2 or imageio[ffmpeg])')

    def write(self, frame_rgb: np.ndarray):
        frame = np.asarray(frame_rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('expected RGB frame with shape HxWx3')
        self._ensure_writer(frame)
        if BACKEND_NAME == 'imageio':
            self._writer.append_data(frame)
            return
        self._writer.write(frame[:, :, ::-1])

    def close(self):
        if self._writer is None:
            return
        if BACKEND_NAME == 'cv2':
            self._writer.release()
        else:
            self._writer.close()
        self._writer = None
        detected_codec = _probe_video_codec(self.path) or self.codec
        self.actual_codec = detected_codec
        self.codec = detected_codec
        self.transcode_error = None
        if not _codec_is_h264(detected_codec):
            ok, detail = _transcode_to_h264(self.path)
            if ok:
                self.actual_codec = _probe_video_codec(self.path) or 'h264'
                self.codec = self.actual_codec
            else:
                self.transcode_error = (
                    f'failed to transcode {self.path.name} to h264 from codec={detected_codec!r}: {detail}'
                )


def _is_static_fallback_gradient(image: np.ndarray) -> bool:
    """Detect task_generator's old synthetic Isaac fallback RGB image.

    The deprecated fallback publisher produced a fixed 640x480 RGB test pattern:
    R=x%256, G=y%256, B=96. Recording that pattern makes the ego video look like
    meaningless 2x3 color blocks, so treat it as invalid camera input.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return False
    height, width = image.shape[:2]
    if height < 16 or width < 16:
        return False
    sample = image[: min(height, 480), : min(width, 640), :]
    yy, xx = np.mgrid[0:sample.shape[0], 0:sample.shape[1]]
    expected_r = (xx % 256).astype(np.uint8)
    expected_g = (yy % 256).astype(np.uint8)
    blue = sample[..., 2]
    return (
        np.array_equal(sample[..., 0], expected_r)
        and np.array_equal(sample[..., 1], expected_g)
        and bool(np.all((blue >= 94) & (blue <= 98)))
    )


def image_msg_to_numpy(message: Image):
    data = np.frombuffer(message.data, dtype=np.uint8)
    if message.encoding in ('rgb8', 'bgr8'):
        image = data.reshape((message.height, message.step // 3, 3))[:, : message.width, :].copy()
        if message.encoding == 'bgr8':
            image = image[:, :, ::-1]
        return image
    if message.encoding in ('rgba8', 'bgra8'):
        image = data.reshape((message.height, message.step // 4, 4))[:, : message.width, :4].copy()
        if message.encoding == 'bgra8':
            image = image[:, :, [2, 1, 0, 3]]
        return image[:, :, :3]
    if message.encoding == 'mono8':
        mono = data.reshape((message.height, message.step))[:, : message.width].copy()
        return np.repeat(mono[:, :, None], 3, axis=2)
    return None


def _yaw_from_quat(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_scan_ranges(scan: LaserScan):
    points = []
    angle = float(scan.angle_min)
    for raw in scan.ranges:
        rng = float(raw)
        if math.isfinite(rng) and scan.range_min <= rng <= scan.range_max:
            points.append((rng, angle))
        angle += float(scan.angle_increment)
    return points


class EvalVideoRecorder(Node):
    def __init__(self, *, output_dir, map_yaml_path, task_reset_topic, scenario_reset_topic, finished_topic, ego_topic, depth_topic, camera_info_topic, debug_overlay_topic, sim_top_down_topic, odom_topic, goal_topic, scan_topic, fps, top_down_size_px, top_down_window_m):
        super().__init__('internnav_eval_video_recorder')
        self.output_dir = Path(output_dir)
        self.videos_dir = self.output_dir / 'videos'
        self.index_path = self.output_dir / 'video_index.json'
        self.error_path = self.output_dir / 'video_recording_error.txt'
        self.fps = max(float(fps), 1.0)
        self.frame_period = 1.0 / self.fps
        self.top_down_size_px = max(int(top_down_size_px), 128)
        self.top_down_window_m = max(float(top_down_window_m), 2.0)
        self.map_yaml_path = str(map_yaml_path)
        self.ego_topic = ego_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.debug_overlay_topic = debug_overlay_topic
        self.sim_top_down_topic = sim_top_down_topic
        self.odom_topic = odom_topic
        self.goal_topic = goal_topic
        self.scan_topic = scan_topic
        self.finished = False
        self.current_episode = None
        self.current_episode_info = None
        self.ego_writer = None
        self.top_writer = None
        self.debug_overlay_writer = None
        self.sim_top_down_writer = None
        self.last_frame_time = 0.0
        self.reset_generation = 0
        self.latest_rgb = None
        self.latest_rgb_generation = -1
        self.depth_ready = False
        self.depth_ready_generation = -1
        self.camera_info_ready = False
        self.camera_info_generation = -1
        self.latest_debug_overlay = None
        self.latest_debug_overlay_generation = -1
        self.latest_sim_top_down = None
        self.latest_sim_top_down_generation = -1
        self.latest_pose = None
        self.latest_pose_generation = -1
        self.latest_goal = None
        self.latest_scan = []
        self.trajectory_world = []
        self.reset_seen = False
        self.last_reset_wall_time = 0.0
        self.index = {
            'video_backend': BACKEND_NAME,
            'config': {
                'map_yaml_path': self.map_yaml_path,
                'task_reset_topic': task_reset_topic,
                'scenario_reset_topic': scenario_reset_topic,
                'ego_topic': ego_topic,
                'depth_topic': depth_topic,
                'camera_info_topic': camera_info_topic,
                'debug_overlay_topic': debug_overlay_topic,
                'sim_top_down_topic': sim_top_down_topic,
                'odom_topic': odom_topic,
                'goal_topic': goal_topic,
                'scan_topic': scan_topic,
                'fps': self.fps,
                'top_down_size_px': self.top_down_size_px,
                'top_down_window_m': self.top_down_window_m,
            },
            'format': {
                'container': 'mp4',
                'preferred_codec': 'libx264',
                'file_extension': '.mp4',
            },
            'episodes': [],
        }

        self.map_image, self.map_resolution, self.map_origin = self._load_map(self.map_yaml_path)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self._write_index()

        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        event_qos = QoSProfile(depth=1)
        event_qos.reliability = ReliabilityPolicy.RELIABLE
        # The eval recorder is started before the launch process, so it should
        # observe the new run's task_reset live.  Using TRANSIENT_LOCAL here can
        # replay a stale latched reset from a previous task_generator process on
        # the same topic, causing videos to start during scene/robot/pedestrian
        # loading instead of at the benchmark episode boundary.
        event_qos.durability = DurabilityPolicy.VOLATILE

        self.create_subscription(Int16, task_reset_topic, self._on_task_reset, event_qos)
        if scenario_reset_topic and scenario_reset_topic != task_reset_topic:
            self.create_subscription(Int16, scenario_reset_topic, self._on_task_reset, event_qos)
        self.create_subscription(Empty, finished_topic, self._on_finished, event_qos)
        self.create_subscription(Image, ego_topic, self._on_ego_image, sensor_qos)
        if depth_topic:
            self.create_subscription(Image, depth_topic, self._on_depth_image, sensor_qos)
        if camera_info_topic:
            self.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, sensor_qos)
        if debug_overlay_topic:
            self.create_subscription(Image, debug_overlay_topic, self._on_debug_overlay_image, sensor_qos)
        if sim_top_down_topic:
            self.create_subscription(Image, sim_top_down_topic, self._on_sim_top_down_image, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, event_qos)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, sensor_qos)

    def _load_map(self, map_yaml_path):
        try:
            map_yaml = Path(map_yaml_path)
            if not map_yaml.exists():
                return None, 0.05, (-0.0, -0.0)
            metadata = yaml.safe_load(map_yaml.read_text(encoding='utf-8')) or {}
            image_path = map_yaml.parent / str(metadata.get('image', ''))
            if not image_path.exists():
                return None, float(metadata.get('resolution', 0.05)), tuple(metadata.get('origin', [0.0, 0.0])[:2])
            image = PILImage.open(image_path).convert('L')
            rgb = PILImage.merge('RGB', (image, image, image))
            return np.asarray(rgb), float(metadata.get('resolution', 0.05)), tuple(metadata.get('origin', [0.0, 0.0])[:2])
        except Exception as exc:
            self._record_error(f'failed to load map: {exc}')
            return None, 0.05, (0.0, 0.0)

    def _record_error(self, message):
        self.error_path.write_text(str(message), encoding='utf-8')

    def _clear_transient_error(self):
        if not self.error_path.exists():
            return
        try:
            message = self.error_path.read_text(encoding='utf-8')
        except Exception:
            return
        if message.startswith('waiting for task_reset before opening video writers') or message.startswith(
            'waiting for fresh post-reset camera+odom messages before recording episode '
        ):
            try:
                self.error_path.unlink()
            except Exception:
                pass

    def _write_index(self):
        self.index_path.write_text(json.dumps(self.index, ensure_ascii=False, indent=2), encoding='utf-8')

    def _episode_dir(self, episode):
        return self.videos_dir / f'episode_{int(episode):04d}'

    def _close_episode(self, *, reason):
        ego_writer = self.ego_writer
        top_writer = self.top_writer
        debug_overlay_writer = self.debug_overlay_writer
        sim_top_down_writer = self.sim_top_down_writer
        if ego_writer is not None:
            ego_writer.close()
            self.ego_writer = None
        if top_writer is not None:
            top_writer.close()
            self.top_writer = None
        if debug_overlay_writer is not None:
            debug_overlay_writer.close()
            self.debug_overlay_writer = None
        if sim_top_down_writer is not None:
            sim_top_down_writer.close()
            self.sim_top_down_writer = None
        if self.current_episode_info is not None:
            if ego_writer is not None:
                self.current_episode_info['ego_video_codec'] = getattr(ego_writer, 'codec', None)
                self.current_episode_info['ego_video_codec_detected'] = getattr(ego_writer, 'actual_codec', None)
            if top_writer is not None:
                self.current_episode_info['top_down_video_codec'] = getattr(top_writer, 'codec', None)
                self.current_episode_info['top_down_video_codec_detected'] = getattr(top_writer, 'actual_codec', None)
                self.current_episode_info['map_top_down_video_codec'] = getattr(top_writer, 'codec', None)
            if debug_overlay_writer is not None:
                self.current_episode_info['debug_overlay_video_codec'] = getattr(debug_overlay_writer, 'codec', None)
                self.current_episode_info['debug_overlay_video_codec_detected'] = getattr(debug_overlay_writer, 'actual_codec', None)
            if sim_top_down_writer is not None:
                self.current_episode_info['sim_top_down_video_codec'] = getattr(sim_top_down_writer, 'codec', None)
                self.current_episode_info['sim_top_down_video_codec_detected'] = getattr(sim_top_down_writer, 'actual_codec', None)

            transcode_errors = [
                getattr(writer, 'transcode_error', None)
                for writer in (ego_writer, top_writer, debug_overlay_writer, sim_top_down_writer)
                if writer is not None and getattr(writer, 'transcode_error', None)
            ]
            if transcode_errors:
                self.current_episode_info['video_transcode_errors'] = transcode_errors
                self._record_error('\n'.join(transcode_errors))
            self.current_episode_info['close_reason'] = reason
            self.current_episode_info['finished_at_wall_time'] = time.time()
            self._write_index()
        self.current_episode = None
        self.current_episode_info = None
        self.trajectory_world = []

    def _reset_episode_stream_state(self):
        self.last_frame_time = 0.0
        self.latest_rgb = None
        self.latest_rgb_generation = -1
        self.depth_ready = False
        self.depth_ready_generation = -1
        self.camera_info_ready = False
        self.camera_info_generation = -1
        self.latest_debug_overlay = None
        self.latest_debug_overlay_generation = -1
        self.latest_sim_top_down = None
        self.latest_sim_top_down_generation = -1
        self.latest_pose = None
        self.latest_pose_generation = -1
        self.latest_goal = None
        self.latest_scan = []
        self.trajectory_world = []

    def _has_fresh_rgb(self):
        return self.latest_rgb is not None and self.latest_rgb_generation == self.reset_generation

    def _has_fresh_pose(self):
        return self.latest_pose is not None and self.latest_pose_generation == self.reset_generation

    def _ensure_episode(self):
        if not self.reset_seen:
            self._record_error(
                'waiting for task_reset before opening video writers; '
                f'camera streams may already exist but the episode has not started on {self.ego_topic}'
            )
            return False
        if self.current_episode is None:
            if not self._streams_ready_for_episode():
                return False
            next_episode = 0
            if self.index['episodes']:
                try:
                    next_episode = int(self.index['episodes'][-1].get('episode', -1)) + 1
                except Exception:
                    next_episode = 0
            self.current_episode = next_episode
        if not self._streams_ready_for_episode():
            self._record_error(
                'waiting for fresh post-reset camera+odom messages before recording episode '
                f'{self.current_episode}: ego={self.ego_topic}, depth={self.depth_topic or "<disabled>"}, '
                f'camera_info={self.camera_info_topic or "<disabled>"}, odom={self.odom_topic}'
            )
            return False
        if self.current_episode_info is not None:
            return True
        episode_dir = self._episode_dir(self.current_episode)
        episode_dir.mkdir(parents=True, exist_ok=True)
        ego_path = episode_dir / 'ego_observation.mp4'
        top_path = episode_dir / 'map_top_down_follow.mp4'
        debug_overlay_path = episode_dir / 'ego_debug_overlay.mp4'
        sim_top_down_path = episode_dir / 'sim_top_down.mp4'
        self.ego_writer = VideoWriterWrapper(ego_path, self.fps)
        self.top_writer = VideoWriterWrapper(top_path, self.fps)
        self.debug_overlay_writer = VideoWriterWrapper(debug_overlay_path, self.fps) if self.debug_overlay_topic else None
        self.sim_top_down_writer = VideoWriterWrapper(sim_top_down_path, self.fps) if self.sim_top_down_topic else None
        self.current_episode_info = {
            'episode': int(self.current_episode),
            'directory': str(episode_dir),
            'ego_video': str(ego_path),
            'top_down_video': str(top_path),
            'map_top_down_video': str(top_path),
            'debug_overlay_video': str(debug_overlay_path) if self.debug_overlay_topic else None,
            'sim_top_down_video': str(sim_top_down_path) if self.sim_top_down_topic else None,
            'ego_frames': 0,
            'top_down_frames': 0,
            'debug_overlay_frames': 0,
            'sim_top_down_frames': 0,
            'container': 'mp4',
            'started_at_wall_time': time.time(),
        }
        self.index['episodes'].append(self.current_episode_info)
        self._write_index()
        return True

    def _camera_ready(self):
        return (
            self._has_fresh_rgb()
            and (not self.depth_topic or (self.depth_ready and self.depth_ready_generation == self.reset_generation))
            and (not self.camera_info_topic or (self.camera_info_ready and self.camera_info_generation == self.reset_generation))
        )

    def _streams_ready_for_episode(self):
        return self._camera_ready() and self._has_fresh_pose()

    def _map_world_to_pixel(self, x, y):
        if self.map_image is None:
            return None
        width = self.map_image.shape[1]
        height = self.map_image.shape[0]
        px = int(round((float(x) - float(self.map_origin[0])) / self.map_resolution))
        py = int(round(height - ((float(y) - float(self.map_origin[1])) / self.map_resolution)))
        return px, py

    def _pose_pixel_to_crop(self, pixel, center_pixel, crop_size_px):
        if pixel is None:
            return None
        left = center_pixel[0] - crop_size_px
        top = center_pixel[1] - crop_size_px
        rel_x = (pixel[0] - left) * (self.top_down_size_px / (crop_size_px * 2))
        rel_y = (pixel[1] - top) * (self.top_down_size_px / (crop_size_px * 2))
        return int(rel_x), int(rel_y)

    def _render_top_down(self):
        size = self.top_down_size_px
        if self.latest_pose is None:
            return np.zeros((size, size, 3), dtype=np.uint8)

        if self.map_image is None:
            canvas = PILImage.new('RGB', (size, size), color=(25, 25, 25))
        else:
            center_pixel = self._map_world_to_pixel(self.latest_pose['x'], self.latest_pose['y'])
            crop_radius_px = max(int(round((self.top_down_window_m / self.map_resolution) / 2.0)), 32)
            left = center_pixel[0] - crop_radius_px
            top = center_pixel[1] - crop_radius_px
            right = center_pixel[0] + crop_radius_px
            bottom = center_pixel[1] + crop_radius_px
            map_image = PILImage.fromarray(self.map_image)
            if left < 0 or top < 0 or right > map_image.width or bottom > map_image.height:
                padded = PILImage.new('RGB', (max(right, map_image.width) - min(left, 0), max(bottom, map_image.height) - min(top, 0)), color=(180, 180, 180))
                padded.paste(map_image, (-min(left, 0), -min(top, 0)))
                crop = padded.crop((left + min(left, 0), top + min(top, 0), right + min(left, 0), bottom + min(top, 0)))
            else:
                crop = map_image.crop((left, top, right, bottom))
            canvas = crop.resize((size, size), _PIL_BILINEAR)

        draw = ImageDraw.Draw(canvas)
        center = (size // 2, size // 2)
        draw.ellipse((center[0] - 8, center[1] - 8, center[0] + 8, center[1] + 8), fill=(255, 80, 80), outline=(255, 255, 255), width=2)

        heading = float(self.latest_pose['yaw'])
        arrow_length = max(size // 8, 24)
        arrow_end = (
            int(center[0] + math.cos(heading) * arrow_length),
            int(center[1] - math.sin(heading) * arrow_length),
        )
        draw.line([center, arrow_end], fill=(255, 80, 80), width=4)

        if self.latest_goal is not None and self.map_image is not None:
            center_pixel = self._map_world_to_pixel(self.latest_pose['x'], self.latest_pose['y'])
            crop_radius_px = max(int(round((self.top_down_window_m / self.map_resolution) / 2.0)), 32)
            goal_pixel = self._map_world_to_pixel(self.latest_goal['x'], self.latest_goal['y'])
            goal_crop = self._pose_pixel_to_crop(goal_pixel, center_pixel, crop_radius_px)
            if goal_crop is not None:
                gx, gy = goal_crop
                draw.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), outline=(80, 255, 80), width=3)
                draw.text((gx + 10, gy - 16), 'goal', fill=(80, 255, 80))

        if self.trajectory_world and self.map_image is not None:
            center_pixel = self._map_world_to_pixel(self.latest_pose['x'], self.latest_pose['y'])
            crop_radius_px = max(int(round((self.top_down_window_m / self.map_resolution) / 2.0)), 32)
            traj = []
            for x, y in self.trajectory_world[-80:]:
                pixel = self._map_world_to_pixel(x, y)
                crop_pixel = self._pose_pixel_to_crop(pixel, center_pixel, crop_radius_px)
                if crop_pixel is not None:
                    traj.append(crop_pixel)
            if len(traj) >= 2:
                draw.line(traj, fill=(64, 200, 255), width=3)

        for rng, angle in self.latest_scan[:720]:
            rel_angle = heading + angle
            wx = self.latest_pose['x'] + math.cos(rel_angle) * rng
            wy = self.latest_pose['y'] + math.sin(rel_angle) * rng
            if self.map_image is not None:
                center_pixel = self._map_world_to_pixel(self.latest_pose['x'], self.latest_pose['y'])
                crop_radius_px = max(int(round((self.top_down_window_m / self.map_resolution) / 2.0)), 32)
                pixel = self._map_world_to_pixel(wx, wy)
                scan_crop = self._pose_pixel_to_crop(pixel, center_pixel, crop_radius_px)
            else:
                scale = self.top_down_size_px / self.top_down_window_m
                scan_crop = (
                    int(center[0] + math.cos(rel_angle) * rng * scale),
                    int(center[1] - math.sin(rel_angle) * rng * scale),
                )
            if scan_crop is not None:
                draw.point(scan_crop, fill=(255, 220, 80))

        draw.rectangle((8, 8, size - 8, 58), outline=(0, 255, 0), width=2)
        draw.text((16, 16), f'episode: {self.current_episode if self.current_episode is not None else "idle"}', fill=(255, 255, 255))
        draw.text((16, 34), f'pose: x={self.latest_pose["x"]:.2f} y={self.latest_pose["y"]:.2f}', fill=(255, 255, 255))
        return np.asarray(canvas, dtype=np.uint8)

    def _maybe_write_frame(self):
        if self.latest_rgb is None or not self._ensure_episode():
            return
        now = time.monotonic()
        if (now - self.last_frame_time) < self.frame_period:
            return
        ego_frame = np.asarray(self.latest_rgb, dtype=np.uint8)
        if _is_static_fallback_gradient(ego_frame):
            self._record_error(f'skipped synthetic fallback gradient frame from {self.ego_topic}; waiting for real Isaac camera frames')
            return
        top_frame = self._render_top_down()
        self.ego_writer.write(ego_frame)
        self.top_writer.write(top_frame)
        self._clear_transient_error()
        self.current_episode_info['ego_frames'] += 1
        self.current_episode_info['top_down_frames'] += 1
        self.current_episode_info['ego_video_codec'] = getattr(self.ego_writer, 'codec', None)
        self.current_episode_info['top_down_video_codec'] = getattr(self.top_writer, 'codec', None)
        if (
            self.debug_overlay_writer is not None
            and self.latest_debug_overlay is not None
            and self.latest_debug_overlay_generation == self.reset_generation
        ):
            debug_overlay_frame = np.asarray(self.latest_debug_overlay, dtype=np.uint8)
            if not _is_static_fallback_gradient(debug_overlay_frame):
                self.debug_overlay_writer.write(debug_overlay_frame)
                self.current_episode_info['debug_overlay_frames'] += 1
                self.current_episode_info['debug_overlay_video_codec'] = getattr(self.debug_overlay_writer, 'codec', None)
        elif self.debug_overlay_writer is not None:
            fallback_overlay = PILImage.fromarray(ego_frame).convert('RGB')
            draw = ImageDraw.Draw(fallback_overlay)
            draw.rectangle((8, 8, min(fallback_overlay.width - 8, 430), 58), fill=(0, 0, 0), outline=(255, 180, 0), width=2)
            draw.text((16, 18), 'InternNav debug overlay unavailable', fill=(255, 220, 80))
            draw.text((16, 36), 'showing ego camera fallback', fill=(255, 255, 255))
            self.debug_overlay_writer.write(np.asarray(fallback_overlay, dtype=np.uint8))
            self.current_episode_info['debug_overlay_frames'] += 1
            self.current_episode_info['debug_overlay_video_codec'] = getattr(self.debug_overlay_writer, 'codec', None)
            self.current_episode_info['debug_overlay_fallback'] = True
        if (
            self.sim_top_down_writer is not None
            and self.latest_sim_top_down is not None
            and self.latest_sim_top_down_generation == self.reset_generation
        ):
            # Isaac's top-down Replicator stream can emit one pre-settled frame
            # right after task_reset while the camera/render product is being
            # positioned.  Do not let that stale/incorrect frame become t=0 of
            # sim_top_down.mp4; the visual validator samples t=0 as the episode
            # baseline and expects an actual top-down camera view.
            sim_started_at = float(self.current_episode_info.get('started_at_wall_time') or 0.0)
            if time.time() - sim_started_at < 0.5:
                self.current_episode_info['sim_top_down_warmup_skipped'] = True
                self.current_episode_info['sim_top_down_warmup_sec'] = 0.5
                self.current_episode_info['last_frame_wall_time'] = time.time()
                self.last_frame_time = now
                self._write_index()
                return
            sim_top_down_frame = np.asarray(self.latest_sim_top_down, dtype=np.uint8)
            self.sim_top_down_writer.write(sim_top_down_frame)
            self.current_episode_info['sim_top_down_frames'] += 1
        self.current_episode_info['last_frame_wall_time'] = time.time()
        self.last_frame_time = now
        self._write_index()

    def _on_task_reset(self, msg: Int16):
        episode = int(msg.data)
        self.reset_seen = True
        self.last_reset_wall_time = time.time()
        if self.current_episode == episode:
            return
        if self.current_episode_info is not None:
            self._close_episode(reason='task_reset')
        self.reset_generation += 1
        self.current_episode = episode
        self.current_episode_info = None
        self._reset_episode_stream_state()
        if self._streams_ready_for_episode():
            self._ensure_episode()

    def _on_finished(self, _msg: Empty):
        # /finished uses transient-local QoS in the task generator.  A recorder
        # that starts after a previous run can receive that stale latched finish
        # near the next task_reset and exit before any navigation frames are
        # written.  Only accept finish after this process has observed a reset
        # and the episode has been alive long enough to be from the current run.
        if self.reset_seen:
            if (time.time() - self.last_reset_wall_time) < 2.0:
                return
        else:
            if self.current_episode_info is None:
                return
            if int(self.current_episode_info.get('ego_frames', 0) or 0) <= 0:
                return
        self.finished = True
        self._close_episode(reason='finished')

    def _on_ego_image(self, msg: Image):
        image = image_msg_to_numpy(msg)
        if image is None:
            return
        self.latest_rgb = image
        self.latest_rgb_generation = self.reset_generation if self.reset_seen else -1
        self._maybe_write_frame()

    def _on_depth_image(self, _msg: Image):
        self.depth_ready = True
        self.depth_ready_generation = self.reset_generation if self.reset_seen else -1
        self._maybe_write_frame()

    def _on_camera_info(self, _msg: CameraInfo):
        self.camera_info_ready = True
        self.camera_info_generation = self.reset_generation if self.reset_seen else -1
        self._maybe_write_frame()

    def _on_debug_overlay_image(self, msg: Image):
        image = image_msg_to_numpy(msg)
        if image is None:
            return
        self.latest_debug_overlay = image
        self.latest_debug_overlay_generation = self.reset_generation if self.reset_seen else -1

    def _on_sim_top_down_image(self, msg: Image):
        image = image_msg_to_numpy(msg)
        if image is None:
            return
        self.latest_sim_top_down = image
        self.latest_sim_top_down_generation = self.reset_generation if self.reset_seen else -1

    def _on_odom(self, msg: Odometry):
        pose = msg.pose.pose
        quat = pose.orientation
        self.latest_pose = {
            'x': float(pose.position.x),
            'y': float(pose.position.y),
            'yaw': _yaw_from_quat(float(quat.x), float(quat.y), float(quat.z), float(quat.w)),
        }
        self.latest_pose_generation = self.reset_generation if self.reset_seen else -1
        self.trajectory_world.append((self.latest_pose['x'], self.latest_pose['y']))
        if len(self.trajectory_world) > 512:
            self.trajectory_world = self.trajectory_world[-512:]
        # Camera frames in Isaac can arrive before the episode reset marker and
        # then stay latched/static for a while.  Odom is the reliable stream
        # during navigation, so use it to drive video frame capture once the
        # first valid camera/depth/camera_info sample has made the recorder
        # ready.  This guarantees a top-down trajectory video even if no new
        # RGB callback occurs after task_reset.
        self._maybe_write_frame()

    def _on_goal(self, msg: PoseStamped):
        self.latest_goal = {
            'x': float(msg.pose.position.x),
            'y': float(msg.pose.position.y),
        }

    def _on_scan(self, msg: LaserScan):
        self.latest_scan = _normalize_scan_ranges(msg)


OUTPUT_DIR = sys.argv[1]
MAP_YAML_PATH = sys.argv[2]
TASK_RESET_TOPIC = sys.argv[3]
SCENARIO_RESET_TOPIC = sys.argv[4]
FINISHED_TOPIC = sys.argv[5]
EGO_TOPIC = sys.argv[6]
DEPTH_TOPIC = sys.argv[7]
CAMERA_INFO_TOPIC = sys.argv[8]
DEBUG_OVERLAY_TOPIC = sys.argv[9]
SIM_TOP_DOWN_TOPIC = sys.argv[10]
ODOM_TOPIC = sys.argv[11]
GOAL_TOPIC = sys.argv[12]
SCAN_TOPIC = sys.argv[13]
FPS = float(sys.argv[14])
TOP_DOWN_SIZE_PX = int(sys.argv[15])
TOP_DOWN_WINDOW_M = float(sys.argv[16])

rclpy.init()
node = EvalVideoRecorder(
    output_dir=OUTPUT_DIR,
    map_yaml_path=MAP_YAML_PATH,
    task_reset_topic=TASK_RESET_TOPIC,
    scenario_reset_topic=SCENARIO_RESET_TOPIC,
    finished_topic=FINISHED_TOPIC,
    ego_topic=EGO_TOPIC,
    depth_topic=DEPTH_TOPIC,
    camera_info_topic=CAMERA_INFO_TOPIC,
    debug_overlay_topic=DEBUG_OVERLAY_TOPIC,
    sim_top_down_topic=SIM_TOP_DOWN_TOPIC,
    odom_topic=ODOM_TOPIC,
    goal_topic=GOAL_TOPIC,
    scan_topic=SCAN_TOPIC,
    fps=FPS,
    top_down_size_px=TOP_DOWN_SIZE_PX,
    top_down_window_m=TOP_DOWN_WINDOW_M,
)


def _shutdown(*_args):
    try:
        node._close_episode(reason='shutdown')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

exit_code = 0
try:
    while rclpy.ok() and not node.finished:
        try:
            rclpy.spin_once(node, timeout_sec=0.5)
        except KeyboardInterrupt:
            break
        except BaseException:
            if not rclpy.ok():
                break
            raise
except BaseException:
    node._record_error(traceback.format_exc())
    exit_code = 1
finally:
    if not node.index.get('episodes'):
        node._record_error(
            'video recorder exited without opening an episode; '
            f'reset_seen={node.reset_seen}, current_episode={node.current_episode}, '
            f'reset_generation={node.reset_generation}, latest_rgb={node.latest_rgb is not None}, '
            f'rgb_generation={node.latest_rgb_generation}, depth_ready={node.depth_ready}, '
            f'depth_generation={node.depth_ready_generation}, camera_info_ready={node.camera_info_ready}, '
            f'camera_info_generation={node.camera_info_generation}, latest_pose={node.latest_pose is not None}, '
            f'pose_generation={node.latest_pose_generation}'
        )
    node._close_episode(reason='process_exit' if exit_code == 0 else 'exception')
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

raise SystemExit(exit_code)
'''

    return subprocess.Popen(
        [
            python_bin,
            '-c',
            recorder_code,
            output_dir,
            map_yaml_path,
            task_reset_topic,
            scenario_reset_topic,
            finished_topic,
            ego_topic,
            depth_topic,
            camera_info_topic,
            debug_overlay_topic,
            sim_top_down_topic,
            odom_topic,
            goal_topic,
            scan_topic,
            str(fps),
            str(top_down_size_px),
            str(top_down_window_m),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _apply_runtime_defaults(args) -> dict:
    adjustments = {}

    if (
        not str(getattr(args, 'vln_instruction_file', '') or '').strip()
        and _is_generic_vln_instruction(getattr(args, 'vln_instruction', ''))
        and str(getattr(args, 'world', '') or '').strip().startswith('grscenes_')
    ):
        workspace_root = _workspace_root_from_runtime()
        manifest_path, manifest_attempts = _resolve_existing_manifest_path(
            str(getattr(args, 'vln_instruction_manifest', '') or ''),
            workspace_root,
        )
        if manifest_path:
            lookup = _lookup_grscenes_instruction_from_manifest(
                manifest_path,
                world=getattr(args, 'world', ''),
                scenario=getattr(args, 'scenario_file', ''),
                episode=getattr(args, 'vln_instruction_episode', ''),
                timestamp=getattr(args, 'vln_instruction_timestamp', ''),
            )
            if lookup.get('ok') and lookup.get('instruction'):
                args.vln_instruction = str(lookup['instruction'])
                args.vln_instruction_manifest = manifest_path
                adjustments['vln_instruction'] = {
                    'source': 'grscenes_manifest',
                    'manifest_path': manifest_path,
                    'world': lookup.get('world'),
                    'scenario': lookup.get('scenario'),
                    'episode': lookup.get('episode'),
                    'timestamp': lookup.get('timestamp'),
                    'instruction_file': lookup.get('instruction_file'),
                    'match_count': lookup.get('match_count'),
                    'ambiguous': lookup.get('ambiguous'),
                }
            else:
                adjustments['vln_instruction_manifest_lookup_failed'] = lookup
        else:
            adjustments['vln_instruction_manifest_not_found'] = manifest_attempts

    env_python, env_python_name = _first_env_value(
        'ARENA_VLN_MODEL_PYTHON',
        'ARENA_INTERNNAV_PYTHON',
        'ARENA_PYTHON',
    )
    if env_python and not getattr(args, 'dual_vln_python_executable', ''):
        args.dual_vln_python_executable = env_python
        adjustments['dual_vln_python_executable'] = f'{env_python} ({env_python_name})'
    elif getattr(args, 'dual_vln_python_executable', ''):
        configured_python = str(getattr(args, 'dual_vln_python_executable', '')).strip()
        if configured_python and not os.path.exists(configured_python):
            if env_python and os.path.exists(env_python):
                args.dual_vln_python_executable = env_python
                adjustments['dual_vln_python_executable'] = (
                    f'{env_python} ({env_python_name}; replaced missing configured path {configured_python})'
                )
            else:
                adjustments['dual_vln_python_executable_missing'] = configured_python

    if getattr(args, 'dual_vln_status_topic', '') in {
        '/task_generator_node/dual_vln/status',
        '/task_generator_node/internnav/status',
    }:
        args.dual_vln_status_topic = f'/task_generator_node/{args.robot}/internnav/status'
        adjustments['dual_vln_status_topic'] = args.dual_vln_status_topic

    if _is_internnav_run(args):
        env_http_url, env_http_url_name = _first_env_value(
            'ARENA_EVAL_INTERNNAV_HTTP_URL',
            'ARENA_INTERNNAV_HTTP_URL',
        )
        if env_http_url and not getattr(args, 'dual_vln_http_url', ''):
            args.dual_vln_http_url = env_http_url
            adjustments['dual_vln_http_url'] = f'{env_http_url} ({env_http_url_name})'
        env_http_timeout, env_http_timeout_name = _first_env_value(
            'ARENA_EVAL_INTERNNAV_HTTP_TIMEOUT_SEC',
            'ARENA_INTERNNAV_HTTP_TIMEOUT_SEC',
        )
        if env_http_timeout and float(getattr(args, 'dual_vln_http_timeout_sec', 0.0) or 0.0) <= 0.0:
            try:
                args.dual_vln_http_timeout_sec = float(env_http_timeout)
                adjustments['dual_vln_http_timeout_sec'] = f'{args.dual_vln_http_timeout_sec} ({env_http_timeout_name})'
            except (TypeError, ValueError):
                adjustments['dual_vln_http_timeout_sec_invalid'] = f'{env_http_timeout} ({env_http_timeout_name})'
        if getattr(args, 'dual_vln_http_url', ''):
            raw_adapter_target = str(getattr(args, 'dual_vln_adapter_target', '') or '').strip()
            normalized_http_target, _ = _normalize_internnav_adapter_target(raw_adapter_target)
            if not raw_adapter_target or raw_adapter_target in HTTP_ADAPTER_REPLACED_TARGETS or normalized_http_target == DEFAULT_INTERNNAV_ADAPTER_TARGET:
                args.dual_vln_adapter_target = REALWORLD_HTTP_ADAPTER_TARGET
                adjustments['dual_vln_adapter_target'] = f'{REALWORLD_HTTP_ADAPTER_TARGET} (http-url)'
            if str(getattr(args, 'dual_vln_mode', '')).strip().lower() == 'heuristic':
                args.dual_vln_mode = 'adapter'
                adjustments['dual_vln_mode'] = 'adapter (http-url)'
        normalized_adapter_target, adapter_target_source = _normalize_internnav_adapter_target(
            getattr(args, 'dual_vln_adapter_target', '')
        )
        if (
            not getattr(args, 'dual_vln_adapter_target', '')
            or normalized_adapter_target != getattr(args, 'dual_vln_adapter_target', '')
        ):
            args.dual_vln_adapter_target = normalized_adapter_target
            adjustments['dual_vln_adapter_target'] = (
                normalized_adapter_target if adapter_target_source is None else f'{normalized_adapter_target} ({adapter_target_source})'
            )

        env_model_path, env_model_path_name = _first_env_value(
            'ARENA_INTERNNAV_MODEL_PATH',
            'INTERNNAV_MODEL_PATH',
            'ARENA_VLN_MODEL_PATH',
        )
        if env_model_path and not getattr(args, 'dual_vln_model_path', ''):
            args.dual_vln_model_path = env_model_path
            adjustments['dual_vln_model_path'] = f'{env_model_path} ({env_model_path_name})'
        default_topics = _default_vision_topics(args.robot)
        if default_topics is not None:
            rgb_topic, depth_topic, camera_info_topic = default_topics
            if not args.dual_vln_rgb_topic:
                args.dual_vln_rgb_topic = rgb_topic
                adjustments['dual_vln_rgb_topic'] = rgb_topic
            if not args.dual_vln_depth_topic:
                args.dual_vln_depth_topic = depth_topic
                adjustments['dual_vln_depth_topic'] = depth_topic
            if not args.dual_vln_camera_info_topic:
                args.dual_vln_camera_info_topic = camera_info_topic
                adjustments['dual_vln_camera_info_topic'] = camera_info_topic

    if getattr(args, 'dual_vln_require_real_backend', False):
        default_topics = _default_vision_topics(args.robot)
        if default_topics is not None:
            rgb_topic, depth_topic, camera_info_topic = default_topics
            if not args.dual_vln_rgb_topic:
                args.dual_vln_rgb_topic = rgb_topic
                adjustments['dual_vln_rgb_topic'] = rgb_topic
            if not args.dual_vln_depth_topic:
                args.dual_vln_depth_topic = depth_topic
                adjustments['dual_vln_depth_topic'] = depth_topic
            if not args.dual_vln_camera_info_topic:
                args.dual_vln_camera_info_topic = camera_info_topic
                adjustments['dual_vln_camera_info_topic'] = camera_info_topic

    if not getattr(args, 'eval_video_sim_top_down_topic', ''):
        sim_top_down_topic = _default_eval_video_sim_top_down_topic(args.robot)
        if sim_top_down_topic:
            args.eval_video_sim_top_down_topic = sim_top_down_topic
            adjustments['eval_video_sim_top_down_topic'] = sim_top_down_topic

    if not getattr(args, 'eval_video_debug_overlay_topic', ''):
        debug_overlay_topic = _default_eval_video_debug_overlay_topic()
        if debug_overlay_topic:
            args.eval_video_debug_overlay_topic = debug_overlay_topic
            adjustments['eval_video_debug_overlay_topic'] = debug_overlay_topic

    if getattr(args, 'save_eval_video', False):
        args.dual_vln_enable_visualization = True
        adjustments.setdefault('dual_vln_enable_visualization', True)

    device_name = str(args.dual_vln_device).strip().lower()
    real_backend_requested = bool(
        getattr(args, 'dual_vln_require_real_backend', False)
        or str(getattr(args, 'dual_vln_adapter_target', '')).strip()
        or str(getattr(args, 'dual_vln_mode', '')).strip().lower() == 'internnav'
    )
    model_inference_requested = bool(
        str(getattr(args, 'dual_vln_adapter_target', '')).strip()
        or str(getattr(args, 'dual_vln_mode', '')).strip().lower()
        in {'internnav', 'model', 'torchscript', 'adapter', 'python', 'python_adapter'}
    )
    if real_backend_requested and args.dual_vln_inference_timeout_sec <= 0.2:
        args.dual_vln_inference_timeout_sec = 120.0
        adjustments['dual_vln_inference_timeout_sec'] = 120.0
    if model_inference_requested and device_name == 'cpu' and args.dual_vln_inference_rate_hz >= 10.0:
        args.dual_vln_inference_rate_hz = 0.5
        adjustments['dual_vln_inference_rate_hz'] = 0.5

    return adjustments


def _classify_end_reason(
    *,
    finished_observed: bool,
    launch_returncode: int | None,
    timed_out: bool,
    internnav_status,
    internnav_diagnostic_summary=None,
):
    if finished_observed and not timed_out and launch_returncode in (None, 0):
        if isinstance(internnav_diagnostic_summary, dict):
            final_distance = internnav_diagnostic_summary.get('final_goal_distance')
            if isinstance(final_distance, dict):
                min_final = final_distance.get('min')
                if isinstance(min_final, (float, int)) and float(min_final) > 0.75:
                    return 'finished_without_goal_reached'
        return 'finished'
    if isinstance(internnav_status, dict):
        status = str(internnav_status.get('status', ''))
        degraded = bool(internnav_status.get('degraded', False))
        debug = internnav_status.get('debug', {}) if isinstance(internnav_status.get('debug'), dict) else {}
        missing_inputs = debug.get('missing_inputs') if isinstance(debug.get('missing_inputs'), list) else []
        if status in {
            'adapter_exception',
            'backend_unavailable',
            'camera_timeout',
            'invalid_adapter_output',
            'invalid_model_output',
            'model_unavailable',
            'internnav_missing_rgb',
            'internnav_missing_depth',
            'internnav_empty_output',
            'empty_model_output',
        }:
            return 'adapter_failure'
        if degraded and status in {'waiting_for_camera', 'stale_camera'}:
            if any(name in {'tf', 'odom', 'camera_info', 'rgb', 'depth'} for name in missing_inputs):
                return 'required_inputs_not_ready'
            return 'camera_not_ready'
        if degraded and status in {'inference_timeout', 'exception'}:
            return 'infrastructure_exception'
        if debug.get('safe_stop'):
            return 'safe_stop'

    if finished_observed:
        return 'finished'
    if timed_out:
        return 'timeout'
    if launch_returncode not in (None, 0):
        return 'infrastructure_exception'
    return 'completed_without_finished_topic'


def _terminate_process_tree(proc: subprocess.Popen, *, grace_period_sec: float = 20.0) -> int:
    if proc.poll() is not None:
        return proc.returncode

    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return proc.wait(timeout=1.0)

    deadline = time.monotonic() + max(grace_period_sec, 0.0)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(0.25)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.wait(timeout=1.0)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(0.25)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return proc.wait(timeout=5.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Run a reproducible Arena InternNav eval from arena-1. '
            'Real InternNav model inference is always external and must be served by internnav-1.'
        )
    )
    parser.add_argument('--sim', default='isaac_eval')
    parser.add_argument('--human', default='hunav')
    parser.add_argument('--world', default='map_empty')
    parser.add_argument('--robot', default='jackal')
    parser.add_argument('--local-planner', default='dual_vln')
    parser.add_argument('--inter-planner', default='navigate_to_pose_w_replanning_and_recovery')
    parser.add_argument('--global-planner', default='navfn')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument(
        '--timeout-wall-factor',
        type=float,
        default=5.0,
        help='Wall-clock timeout multiplier forwarded to task_generator for slow Isaac/model evals.',
    )
    parser.add_argument(
        '--timeout-wall-sec',
        type=float,
        default=0.0,
        help='Explicit wall-clock timeout forwarded to task_generator; 0 lets timeout_wall_factor decide.',
    )
    parser.add_argument('--tm-robots', default='random')
    parser.add_argument('--tm-obstacles', default='random')
    parser.add_argument('--scenario-file', default='')
    parser.add_argument('--scenario-config-id', default='')
    parser.add_argument('--scenario-config-path', default='')
    parser.add_argument('--social-eval', action='store_true', help='Enable stricter social-navigation metrics and artifact validation expectations.')
    parser.add_argument('--headless', default='2')
    parser.add_argument('--log-level', default='warn')
    parser.add_argument('--vln-instruction', default='navigate')
    parser.add_argument('--vln-instruction-file', default='')
    parser.add_argument(
        '--vln-instruction-manifest',
        default='',
        help=(
            'Optional GRScenes uploaded test entries manifest. When vln_instruction is generic and no '
            'vln_instruction_file is set, internnav_eval will auto-select the matching entry by world, '
            'scenario_file, episode, and optional timestamp.'
        ),
    )
    parser.add_argument(
        '--vln-instruction-episode',
        default='episode_00',
        help='Episode key used when auto-selecting a GRScenes instruction from --vln-instruction-manifest.',
    )
    parser.add_argument(
        '--vln-instruction-timestamp',
        default='',
        help='Optional timestamp disambiguator for GRScenes instruction manifest lookup.',
    )
    parser.add_argument('--internnav-mode', '--dual-vln-mode', dest='dual_vln_mode', default='heuristic')
    parser.add_argument('--internnav-model-path', '--dual-vln-model-path', dest='dual_vln_model_path', default='')
    parser.add_argument('--internnav-device', '--dual-vln-device', dest='dual_vln_device', default='cpu')
    parser.add_argument(
        '--internnav-inference-rate-hz',
        '--dual-vln-inference-rate-hz',
        '--internnav-planning-rate-hz',
        '--dual-vln-planning-rate-hz',
        dest='dual_vln_inference_rate_hz',
        type=float,
        default=3.3333333333,
        help=(
            'Outer InternNav realworld client planning request rate. '
            'System-2 cadence is controlled separately by plan_step_gap in the InternNav server.'
        ),
    )
    parser.add_argument('--internnav-inference-timeout-sec', '--dual-vln-inference-timeout-sec', dest='dual_vln_inference_timeout_sec', type=float, default=0.2)
    parser.add_argument('--internnav-rgb-topic', '--dual-vln-rgb-topic', dest='dual_vln_rgb_topic', default='')
    parser.add_argument('--internnav-depth-topic', '--dual-vln-depth-topic', dest='dual_vln_depth_topic', default='')
    parser.add_argument('--internnav-camera-info-topic', '--dual-vln-camera-info-topic', dest='dual_vln_camera_info_topic', default='')
    parser.add_argument('--internnav-python-executable', '--dual-vln-python-executable', dest='dual_vln_python_executable', default='')
    parser.add_argument('--internnav-adapter-target', '--dual-vln-adapter-target', dest='dual_vln_adapter_target', default='')
    parser.add_argument(
        '--internnav-http-url',
        '--dual-vln-http-url',
        dest='dual_vln_http_url',
        default='',
        help='HTTP URL for InternVLA realworld /eval_dual adapter, e.g. http://127.0.0.1:5801/eval_dual.',
    )
    parser.add_argument(
        '--internnav-http-timeout-sec',
        '--dual-vln-http-timeout-sec',
        dest='dual_vln_http_timeout_sec',
        type=float,
        default=0.0,
        help='HTTP request timeout for InternVLA realworld adapter. Defaults to inference timeout when unset.',
    )
    parser.add_argument('--internnav-require-real-backend', '--dual-vln-require-real-backend', dest='dual_vln_require_real_backend', action='store_true')
    parser.add_argument('--internnav-strict-device', '--dual-vln-strict-device', dest='dual_vln_strict_device', action='store_true')
    parser.add_argument(
        '--internnav-model-output-policy',
        '--dual-vln-model-output-policy',
        dest='dual_vln_model_output_policy',
        choices=['trajectory', 'discrete', 'raw'],
        default='trajectory',
        help='Select how InternNav outputs are converted: trajectory prefers output_trajectory->cmd_vel; discrete forces action ids; raw keeps legacy precedence.',
    )
    parser.add_argument('--internnav-look-down', '--dual-vln-look-down', dest='dual_vln_look_down', action='store_true')
    parser.add_argument('--internnav-enable-visualization', '--dual-vln-enable-visualization', dest='dual_vln_enable_visualization', action='store_true')
    parser.add_argument(
        '--internnav-external-server',
        '--dual-vln-external-server',
        dest='internnav_external_server',
        action='store_true',
        help='Use the dedicated internnav-1 model server. This is the required/default mode for real InternNav eval.',
    )
    parser.add_argument(
        '--internnav-official-client',
        '--internnav-direct-cmd-vel',
        '--dual-vln-direct-cmd-vel',
        dest='internnav_direct_cmd_vel',
        action='store_true',
        help='Use upstream InternNav realworld ROS2 client publishing cmd_vel directly; skip Arena get_command/status wrapper checks.',
    )
    parser.add_argument('--internnav-command-service', '--dual-vln-command-service', dest='dual_vln_command_service', default='')
    parser.add_argument('--internnav-visualization-topic', '--dual-vln-visualization-topic', dest='dual_vln_visualization_topic', default='internnav/debug_image')
    parser.add_argument('--internnav-action-visualization-topic', '--dual-vln-action-visualization-topic', dest='dual_vln_action_visualization_topic', default='internnav/action_image')
    parser.add_argument('--internnav-visualization-rate-hz', '--dual-vln-visualization-rate-hz', dest='dual_vln_visualization_rate_hz', type=float, default=5.0)
    parser.add_argument('--internnav-model-output-topic', '--dual-vln-model-output-topic', dest='dual_vln_model_output_topic', default='internnav/model_output')
    parser.add_argument('--save-eval-video', action='store_true')
    parser.add_argument('--eval-video-fps', type=float, default=10.0)
    parser.add_argument('--eval-video-top-down-size-px', type=int, default=640)
    parser.add_argument('--eval-video-top-down-window-m', type=float, default=10.0)
    parser.add_argument('--eval-video-sim-top-down-topic', default='')
    parser.add_argument('--eval-video-debug-overlay-topic', default='')
    parser.add_argument('--internnav-status-topic', '--dual-vln-status-topic', dest='dual_vln_status_topic', default='/task_generator_node/internnav/status')
    parser.add_argument(
        '--external-server-preflight-timeout-sec',
        type=float,
        default=15.0,
        help='Bounded discovery timeout for external InternNav get_command/status preflight checks.',
    )
    parser.add_argument(
        '--skip-external-server-preflight',
        action='store_true',
        help='Skip fail-fast external InternNav discovery preflight. Intended only for diagnostics.',
    )
    parser.add_argument('--finished-topic', default='/task_generator_node/finished')
    parser.add_argument('--task-reset-topic', default='/task_generator_node/task_reset')
    parser.add_argument('--launch-timeout-sec', type=float, default=0.0)
    parser.add_argument('--shutdown-grace-period-sec', type=float, default=20.0)
    parser.add_argument('--output-prefix', default='internnav_eval')
    parser.add_argument(
        '--output-root',
        default='',
        help=(
            'Root directory for eval artifacts. Defaults to <workspace>/outputs. '
            'Use an absolute path to write elsewhere, or a relative path under the workspace root.'
        ),
    )
    parser.add_argument('--skip-metrics', action='store_true')
    parser.add_argument('extra_launch_args', nargs='*', help='Additional KEY:=VALUE launch arguments')
    args = parser.parse_args()
    runtime_adjustments = _apply_runtime_defaults(args)
    if args.internnav_direct_cmd_vel:
        args.internnav_external_server = True
        args.skip_external_server_preflight = True
        args.dual_vln_require_real_backend = False
        args.dual_vln_strict_device = False
    if str(args.dual_vln_mode).strip().lower() == 'internnav' and not str(getattr(args, 'dual_vln_http_url', '') or '').strip():
        args.internnav_external_server = True
        args.dual_vln_require_real_backend = False
        args.dual_vln_strict_device = False
    if args.social_eval:
        if args.tm_robots == 'random':
            args.tm_robots = 'scenario'
        if args.tm_obstacles == 'random':
            args.tm_obstacles = 'scenario'
        if not args.scenario_file:
            args.scenario_file = 'normal'
    if args.scenario_file:
        normalized_scenario_file = _scenario_key(args.scenario_file)
        if normalized_scenario_file and normalized_scenario_file != args.scenario_file:
            runtime_adjustments['scenario_file'] = {
                'source': 'normalized_scenario_identifier',
                'input': args.scenario_file,
                'value': normalized_scenario_file,
            }
            args.scenario_file = normalized_scenario_file
    manifest_binding_failure = (
        str(getattr(args, 'world', '') or '').strip().startswith('grscenes_')
        and not str(getattr(args, 'vln_instruction_file', '') or '').strip()
        and _is_generic_vln_instruction(getattr(args, 'vln_instruction', ''))
        and (
            'vln_instruction_manifest_lookup_failed' in runtime_adjustments
            or 'vln_instruction_manifest_not_found' in runtime_adjustments
        )
    )

    arena_eval_share = get_package_share_directory('arena_evaluation')
    bringup_share = get_package_share_directory('arena_bringup')
    sim_setup_share = get_package_share_directory('arena_simulation_setup')

    timestamp = os.environ.get('ARENA_EVAL_FIXED_TIMESTAMP') or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{timestamp}_{args.world}_{args.robot}_{args.dual_vln_mode}'
    relative_dir = os.path.join(args.output_prefix, run_name)
    output_root = _resolve_output_root(args.output_root, arena_eval_share)
    output_dir = os.path.join(output_root, relative_dir)
    snapshots_dir = os.path.join(output_dir, 'snapshots')
    os.makedirs(snapshots_dir, exist_ok=True)
    dual_vln_status_path = os.path.join(output_dir, 'internnav_status.json')
    internnav_trace_path = os.path.join(output_dir, 'internnav_trace.jsonl')
    internnav_diagnostic_summary_path = os.path.join(output_dir, 'internnav_diagnostic_summary.json')
    postprocess_commands_path = os.path.join(output_dir, 'postprocess_commands.txt')
    videos_dir = os.path.join(output_dir, 'videos')
    video_index_path = os.path.join(output_dir, 'video_index.json')
    video_error_path = os.path.join(output_dir, 'video_recording_error.txt')

    env = os.environ.copy()
    if args.scenario_file:
        env['ARENA_SCENARIO_FILE'] = str(args.scenario_file)
    if str(args.world or '').startswith('grscenes_'):
        env.setdefault('ARENA_ISAAC_LOAD_USD_TIMEOUT_SEC', '1800.0')
    resolved_ros_env = _normalize_external_ros_env(env) if args.internnav_external_server else {
        key: {'value': str(env.get(key, '')).strip(), 'source': 'environment' if str(env.get(key, '')).strip() else 'unset'}
        for key in ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'ROS_LOCALHOST_ONLY')
    }

    if args.save_eval_video and not args.dual_vln_rgb_topic:
        raise SystemExit(
            'save_eval_video requires a resolvable RGB topic. '
            'Pass --internnav-rgb-topic explicitly or use a robot/mode with runtime defaults.'
        )

    robot_ego_topic = _robot_topic(args.task_reset_topic, args.robot, args.dual_vln_rgb_topic)
    robot_depth_topic = _robot_topic(args.task_reset_topic, args.robot, args.dual_vln_depth_topic)
    robot_camera_info_topic = _robot_topic(args.task_reset_topic, args.robot, args.dual_vln_camera_info_topic)
    robot_debug_overlay_topic = _robot_topic(args.task_reset_topic, args.robot, args.eval_video_debug_overlay_topic)
    robot_model_output_topic = _robot_topic(args.task_reset_topic, args.robot, args.dual_vln_model_output_topic)
    robot_sim_top_down_topic = _robot_topic(args.task_reset_topic, args.robot, args.eval_video_sim_top_down_topic)
    robot_odom_topic = _robot_topic(args.task_reset_topic, args.robot, 'odom')
    robot_goal_topic = _robot_topic(args.task_reset_topic, args.robot, 'episode_goal_pose')
    robot_scan_topic = _robot_topic(args.task_reset_topic, args.robot, 'scan')
    robot_command_service = str(args.dual_vln_command_service or '').strip() or _robot_topic(args.task_reset_topic, args.robot, 'get_command')
    robot_scenario_reset_topic = _scenario_reset_topic(args.task_reset_topic, args.robot)
    map_yaml_path = _world_map_yaml_path(sim_setup_share, args.world)

    snapshot_files = {}
    for label, src in {
        'task_generator': os.path.join(bringup_share, 'configs', 'task_generator.yaml'),
        'internnav_controller': os.path.join(sim_setup_share, 'configs', 'nav2', 'controllers', 'dual_vln', 'controller_config.yaml'),
    }.items():
        copied = _copy_if_exists(src, os.path.join(snapshots_dir, os.path.basename(src)))
        if copied is not None:
            snapshot_files[label] = copied

    launch_cmd = [
        'ros2', 'launch', 'arena_bringup', 'arena.launch.py',
        f'sim:={args.sim}',
        f'human:={args.human}',
        f'world:={args.world}',
        f'robot:={args.robot}',
        f'local_planner:={args.local_planner}',
        f'inter_planner:={args.inter_planner}',
        f'global_planner:={args.global_planner}',
        *([] if args.skip_metrics else [f'record_data_dir:={output_dir}']),
        f'episodes:={args.episodes}',
        'auto_reset:=true',
        f'tm_robots:={args.tm_robots}',
        f'tm_obstacles:={args.tm_obstacles}',
        f'timeout:={args.timeout}',
        f'timeout_wall_factor:={args.timeout_wall_factor}',
        f'timeout_wall_sec:={args.timeout_wall_sec}',
        f'headless:={args.headless}',
        f'log_level:={args.log_level}',
        # HuNav pedestrian state publication is part of episode readiness for
        # all Isaac/Gazebo eval runs that use the human simulator, not only the
        # stricter social-metrics mode.  Otherwise task_reset can be released
        # before pedestrians exist in the active episode, which matches the
        # regression the user observed.
        f'require_human_states_ready:={str(args.human == "hunav").lower()}',
        'human_states_ready_timeout_sec:=20.0',
        'episode_start_delay_sec:=1.0',
        f'vln_instruction:={args.vln_instruction}',
        f'dual_vln_mode:={args.dual_vln_mode}',
        f'dual_vln_device:={args.dual_vln_device}',
        f'dual_vln_inference_rate_hz:={args.dual_vln_inference_rate_hz}',
        f'dual_vln_inference_timeout_sec:={args.dual_vln_inference_timeout_sec}',
        f'dual_vln_enable_visualization:={str(args.dual_vln_enable_visualization).lower()}',
        f'internnav_external_server:={str(args.internnav_external_server).lower()}',
        f'dual_vln_external_server:={str(args.internnav_external_server).lower()}',
        f'internnav_direct_cmd_vel:={str(args.internnav_direct_cmd_vel).lower()}',
        f'dual_vln_direct_cmd_vel:={str(args.internnav_direct_cmd_vel).lower()}',
        f'dual_vln_require_real_backend:={str(args.dual_vln_require_real_backend).lower()}',
        f'dual_vln_strict_device:={str(args.dual_vln_strict_device).lower()}',
        f'dual_vln_visualization_topic:={args.dual_vln_visualization_topic}',
        f'dual_vln_action_visualization_topic:={args.dual_vln_action_visualization_topic}',
        f'dual_vln_visualization_rate_hz:={args.dual_vln_visualization_rate_hz}',
        f'dual_vln_model_output_topic:={args.dual_vln_model_output_topic}',
    ]
    if args.local_planner == 'dual_vln':
        launch_cmd.append('enable_collision_monitor:=false')
    if args.internnav_direct_cmd_vel:
        launch_cmd.append('robot_launch_file:=internnav_async_eval.launch.py')
    if args.dual_vln_rgb_topic:
        launch_cmd.append(f'dual_vln_rgb_topic:={args.dual_vln_rgb_topic}')
    if args.dual_vln_depth_topic:
        launch_cmd.append(f'dual_vln_depth_topic:={args.dual_vln_depth_topic}')
    if args.dual_vln_camera_info_topic:
        launch_cmd.append(f'dual_vln_camera_info_topic:={args.dual_vln_camera_info_topic}')
    if args.dual_vln_python_executable:
        launch_cmd.append(f'dual_vln_python_executable:={args.dual_vln_python_executable}')
    if args.dual_vln_adapter_target:
        launch_cmd.append(f'dual_vln_adapter_target:={args.dual_vln_adapter_target}')
    if args.dual_vln_http_url:
        launch_cmd.append(f'dual_vln_http_url:={args.dual_vln_http_url}')
    if args.dual_vln_http_timeout_sec and args.dual_vln_http_timeout_sec > 0.0:
        launch_cmd.append(f'dual_vln_http_timeout_sec:={args.dual_vln_http_timeout_sec}')
    if args.dual_vln_model_output_policy:
        launch_cmd.append(f'dual_vln_model_output_policy:={args.dual_vln_model_output_policy}')
    if args.dual_vln_command_service:
        launch_cmd.append(f'dual_vln_command_service:={args.dual_vln_command_service}')
    if args.dual_vln_status_topic:
        launch_cmd.append(f'dual_vln_status_topic:={args.dual_vln_status_topic}')
    if args.dual_vln_look_down:
        launch_cmd.append('dual_vln_look_down:=true')
    if args.vln_instruction_file:
        launch_cmd.append(f'vln_instruction_file:={args.vln_instruction_file}')
    if args.dual_vln_model_path:
        launch_cmd.append(f'dual_vln_model_path:={args.dual_vln_model_path}')
    if args.scenario_file:
        launch_cmd.append(f'scenario_file:={args.scenario_file}')
    launch_cmd.extend(args.extra_launch_args)

    metrics_cmd = ['ros2', 'run', 'arena_evaluation', 'metrics', '--dir', output_dir]
    social_metrics_cmd = ['ros2', 'run', 'arena_evaluation', 'social_metrics', '--dir', output_dir]
    artifact_validation_cmd = ['ros2', 'run', 'arena_bringup', 'social_nav_validation', '--dir', output_dir]
    postprocess_commands = [
        ' '.join(launch_cmd),
        ' '.join(metrics_cmd),
    ]
    if args.social_eval:
        postprocess_commands.extend([
            ' '.join(social_metrics_cmd),
            ' '.join(artifact_validation_cmd),
        ])
    _write_text(postprocess_commands_path, '\n'.join(postprocess_commands) + '\n')

    manifest = {
        'timestamp': timestamp,
        'result_dir_relative': relative_dir,
        'result_dir_absolute': output_dir,
        'output_root': output_root,
        'launch_command': launch_cmd,
        'metrics_command': None if args.skip_metrics else metrics_cmd,
        'postprocess_commands_file': postprocess_commands_path,
        'parameters': {
            'sim': args.sim,
            'human': args.human,
            'world': args.world,
            'robot': args.robot,
            'local_planner': args.local_planner,
            'inter_planner': args.inter_planner,
            'global_planner': args.global_planner,
            'episodes': args.episodes,
            'timeout': args.timeout,
            'timeout_wall_factor': args.timeout_wall_factor,
            'timeout_wall_sec': args.timeout_wall_sec,
            'tm_robots': args.tm_robots,
            'tm_obstacles': args.tm_obstacles,
            'scenario_file': args.scenario_file,
            'scenario_config_id': args.scenario_config_id,
            'scenario_config_path': args.scenario_config_path,
            'social_eval': args.social_eval,
            'social_eval_expectations': {
                'world': args.world,
                'robot': args.robot,
                'human': 'hunav',
                'tm_obstacles': 'scenario',
                'scenario_file': args.scenario_file or 'normal',
                'required_videos': [
                    'ego_observation',
                    'ego_debug_overlay',
                    'sim_top_down',
                    'map_top_down_follow',
                ],
                'required_metrics': [
                    'metrics.csv',
                    'social_metrics.json',
                    'artifact_validation.json',
                ],
            } if args.social_eval else None,
            'vln_instruction': args.vln_instruction,
            'vln_instruction_file': args.vln_instruction_file,
            'vln_instruction_manifest': args.vln_instruction_manifest,
            'vln_instruction_episode': args.vln_instruction_episode,
            'vln_instruction_timestamp': args.vln_instruction_timestamp,
            'dual_vln_mode': args.dual_vln_mode,
            'dual_vln_model_path': args.dual_vln_model_path,
            'dual_vln_device': args.dual_vln_device,
            'internnav_planning_rate_hz': args.dual_vln_inference_rate_hz,
            'dual_vln_inference_rate_hz': args.dual_vln_inference_rate_hz,
            'dual_vln_inference_timeout_sec': args.dual_vln_inference_timeout_sec,
            'dual_vln_rgb_topic': args.dual_vln_rgb_topic,
            'dual_vln_depth_topic': args.dual_vln_depth_topic,
            'dual_vln_camera_info_topic': args.dual_vln_camera_info_topic,
            'dual_vln_python_executable': args.dual_vln_python_executable,
            'eval_python_executable': sys.executable,
            'dual_vln_adapter_target': args.dual_vln_adapter_target,
            'dual_vln_http_url': args.dual_vln_http_url,
            'dual_vln_http_timeout_sec': args.dual_vln_http_timeout_sec,
            'dual_vln_require_real_backend': args.dual_vln_require_real_backend,
            'dual_vln_strict_device': args.dual_vln_strict_device,
            'dual_vln_model_output_policy': args.dual_vln_model_output_policy,
            'dual_vln_look_down': args.dual_vln_look_down,
            'dual_vln_enable_visualization': args.dual_vln_enable_visualization,
            'internnav_external_server': args.internnav_external_server,
            'internnav_direct_cmd_vel': args.internnav_direct_cmd_vel,
            'external_server_preflight_timeout_sec': args.external_server_preflight_timeout_sec,
            'skip_external_server_preflight': args.skip_external_server_preflight,
            'dual_vln_visualization_topic': args.dual_vln_visualization_topic,
            'dual_vln_action_visualization_topic': args.dual_vln_action_visualization_topic,
            'dual_vln_visualization_rate_hz': args.dual_vln_visualization_rate_hz,
            'dual_vln_model_output_topic': args.dual_vln_model_output_topic,
            'dual_vln_model_output_topic_resolved': robot_model_output_topic,
            'save_eval_video': args.save_eval_video,
            'eval_video_fps': args.eval_video_fps,
            'eval_video_top_down_size_px': args.eval_video_top_down_size_px,
            'eval_video_top_down_window_m': args.eval_video_top_down_window_m,
            'eval_video_sim_top_down_topic': robot_sim_top_down_topic if args.save_eval_video else None,
            'eval_video_ego_topic': robot_ego_topic if args.save_eval_video else None,
            'eval_video_depth_topic': robot_depth_topic if args.save_eval_video else None,
            'eval_video_camera_info_topic': robot_camera_info_topic if args.save_eval_video else None,
            'eval_video_debug_overlay_topic': robot_debug_overlay_topic if args.save_eval_video else None,
            'eval_video_odom_topic': robot_odom_topic if args.save_eval_video else None,
            'eval_video_goal_topic': robot_goal_topic if args.save_eval_video else None,
            'eval_video_scan_topic': robot_scan_topic if args.save_eval_video else None,
            'eval_video_scenario_reset_topic': robot_scenario_reset_topic if args.save_eval_video else None,
            'eval_video_map_yaml_path': map_yaml_path if args.save_eval_video else None,
            'dual_vln_status_topic': args.dual_vln_status_topic,
            'dual_vln_model_output_topic': robot_model_output_topic,
            'dual_vln_command_service': robot_command_service,
            'finished_topic': args.finished_topic,
            'task_reset_topic': args.task_reset_topic,
            'launch_timeout_sec': args.launch_timeout_sec,
            'shutdown_grace_period_sec': args.shutdown_grace_period_sec,
            'output_prefix': args.output_prefix,
            'output_root': output_root,
            'isaac_load_usd_timeout_sec': env.get('ARENA_ISAAC_LOAD_USD_TIMEOUT_SEC'),
        },
        'runtime_adjustments': runtime_adjustments,
        'runtime_environment': {
            'ros_discovery': resolved_ros_env,
        },
        'snapshots': snapshot_files,
        'artifacts': {
            'snapshots_dir': snapshots_dir,
            'dual_vln_status_path': dual_vln_status_path,
            'internnav_trace_path': internnav_trace_path,
            'internnav_diagnostic_summary_path': internnav_diagnostic_summary_path,
            'social_metrics_path': os.path.join(output_dir, 'social_metrics.json') if args.social_eval else None,
            'artifact_validation_path': os.path.join(output_dir, 'artifact_validation.json') if args.social_eval else None,
            'postprocess_commands_file': postprocess_commands_path,
            'videos_dir': videos_dir if args.save_eval_video else None,
            'video_index_path': video_index_path if args.save_eval_video else None,
            'video_recording_error_path': video_error_path if args.save_eval_video else None,
        },
        'result': {
            'finished_observed': False,
            'launch_returncode': None,
            'metrics_returncode': None,
            'social_metrics_returncode': None,
            'artifact_validation_returncode': None,
            'end_reason': 'running',
            'video_recorder_returncode': None,
            'external_server_preflight': None,
        },
    }

    manifest_path = os.path.join(output_dir, 'run_manifest.yaml')
    _write_yaml(manifest_path, manifest)
    if manifest_binding_failure:
        manifest['result'].update(
            {
                'launch_returncode': None,
                'metrics_returncode': None,
                'social_metrics_returncode': None,
                'artifact_validation_returncode': None,
                'timed_out': False,
                'end_reason': 'vln_instruction_manifest_lookup_failed',
            }
        )
        _write_yaml(manifest_path, manifest)
        print(
            (
                'GRScenes VLN instruction manifest lookup failed; refusing to run '
                f'with generic instruction {args.vln_instruction!r}. '
                f'See {manifest_path} runtime_adjustments for lookup diagnostics.'
            ),
            file=sys.stderr,
        )
        return 2

    env.setdefault('RCUTILS_LOGGING_BUFFERED_STREAM', '1')
    env.setdefault('ARENA_EVAL_PYTHON', sys.executable)
    env['ARENA_EVAL_INTERNNAV_MODE'] = str(args.dual_vln_mode)
    env['ARENA_EVAL_INTERNNAV_MODEL_PATH'] = str(args.dual_vln_model_path or '')
    env['ARENA_EVAL_INTERNNAV_DEVICE'] = str(args.dual_vln_device)
    env['ARENA_EVAL_INTERNNAV_RGB_TOPIC'] = str(args.dual_vln_rgb_topic or '')
    env['ARENA_EVAL_INTERNNAV_DEPTH_TOPIC'] = str(args.dual_vln_depth_topic or '')
    env['ARENA_EVAL_INTERNNAV_CAMERA_INFO_TOPIC'] = str(args.dual_vln_camera_info_topic or '')
    env['ARENA_EVAL_INTERNNAV_ADAPTER_TARGET'] = str(args.dual_vln_adapter_target or '')
    if args.dual_vln_http_url:
        env['ARENA_EVAL_INTERNNAV_HTTP_URL'] = str(args.dual_vln_http_url)
    else:
        # Do not export an empty high-priority override: launch files also look
        # at ARENA_INTERNNAV_HTTP_URL, and an empty ARENA_EVAL_* value would mask
        # that fallback in EnvironmentVariable substitutions.
        env.pop('ARENA_EVAL_INTERNNAV_HTTP_URL', None)
    env['ARENA_EVAL_INTERNNAV_HTTP_TIMEOUT_SEC'] = str(
        args.dual_vln_http_timeout_sec
        if float(args.dual_vln_http_timeout_sec or 0.0) > 0.0
        else args.dual_vln_inference_timeout_sec
    )
    env['ARENA_EVAL_INTERNNAV_REQUIRE_REAL_BACKEND'] = str(bool(args.dual_vln_require_real_backend)).lower()
    env['ARENA_EVAL_INTERNNAV_STRICT_DEVICE'] = str(bool(args.dual_vln_strict_device)).lower()
    env['ARENA_EVAL_INTERNNAV_MODEL_OUTPUT_POLICY'] = str(args.dual_vln_model_output_policy or 'trajectory')
    env['ARENA_EVAL_INTERNNAV_LOOK_DOWN'] = str(bool(args.dual_vln_look_down)).lower()
    env['ARENA_EVAL_INTERNNAV_ENABLE_VISUALIZATION'] = str(bool(args.dual_vln_enable_visualization)).lower()
    env['ARENA_EVAL_INTERNNAV_VISUALIZATION_TOPIC'] = str(args.dual_vln_visualization_topic)
    env['ARENA_EVAL_INTERNNAV_ACTION_VISUALIZATION_TOPIC'] = str(args.dual_vln_action_visualization_topic)
    env['ARENA_EVAL_INTERNNAV_VISUALIZATION_RATE_HZ'] = str(args.dual_vln_visualization_rate_hz)
    env['ARENA_EVAL_INTERNNAV_MODEL_OUTPUT_TOPIC'] = str(args.dual_vln_model_output_topic)
    env['ARENA_EVAL_INTERNNAV_PLANNING_RATE_HZ'] = str(args.dual_vln_inference_rate_hz)
    env['ARENA_EVAL_INTERNNAV_INFERENCE_RATE_HZ'] = str(args.dual_vln_inference_rate_hz)
    env['ARENA_EVAL_INTERNNAV_INFERENCE_TIMEOUT_SEC'] = str(args.dual_vln_inference_timeout_sec)
    env['ARENA_EVAL_INTERNNAV_TRACE_PATH'] = internnav_trace_path
    env['ARENA_INTERNNAV_EXTERNAL_SERVER'] = '1' if args.internnav_external_server else '0'
    env['ARENA_INTERNNAV_DIRECT_CMD_VEL'] = '1' if args.internnav_direct_cmd_vel else '0'
    if args.dual_vln_python_executable:
        # The InternNav adapter itself looks for these variables when deciding
        # whether to launch the heavy model in a separate Python environment.
        # Set them in the ros2 launch process environment as a robust fallback
        # to launch_ros Node.additional_env, which can be bypassed by legacy
        # launch aliases in nested robot launch files.
        env['ARENA_VLN_MODEL_PYTHON'] = str(args.dual_vln_python_executable)
        env['ARENA_INTERNNAV_PYTHON'] = str(args.dual_vln_python_executable)
        env['ARENA_PYTHON'] = str(args.dual_vln_python_executable)

    if args.internnav_external_server and not args.skip_external_server_preflight:
        command_service_candidates = _external_command_service_candidates(robot_command_service, args.dual_vln_status_topic)
        external_preflight = _run_external_internnav_preflight(
            env,
            expected_service=robot_command_service,
            candidate_services=command_service_candidates,
            expected_status_topic=args.dual_vln_status_topic,
            timeout_sec=args.external_server_preflight_timeout_sec,
        )
        manifest['result']['external_server_preflight'] = external_preflight
        _write_yaml(manifest_path, manifest)
        if not external_preflight.get('pass'):
            manifest['result'].update(
                {
                    'launch_returncode': None,
                    'metrics_returncode': None,
                    'social_metrics_returncode': None,
                    'artifact_validation_returncode': None,
                    'timed_out': False,
                    'end_reason': 'external_preflight_failed',
                }
            )
            _write_yaml(manifest_path, manifest)
            return 2
        observed_service = str(external_preflight.get('observed_service') or '').strip()
        if observed_service and observed_service != robot_command_service:
            args.dual_vln_command_service = observed_service
            robot_command_service = observed_service
            launch_cmd.append(f'dual_vln_command_service:={observed_service}')
            manifest['parameters']['dual_vln_command_service'] = observed_service
            postprocess_commands[0] = ' '.join(launch_cmd)
            _write_text(postprocess_commands_path, '\n'.join(postprocess_commands) + '\n')
            _write_yaml(manifest_path, manifest)
    elif args.internnav_external_server:
        manifest['result']['external_server_preflight'] = {
            'pass': None,
            'skipped': True,
            'reason': 'internnav_direct_cmd_vel' if args.internnav_direct_cmd_vel else 'skip_external_server_preflight',
            'expected_service': robot_command_service,
            'expected_status_topic': args.dual_vln_status_topic,
            'timeout_sec': args.external_server_preflight_timeout_sec,
        }
        _write_yaml(manifest_path, manifest)

    launch_timeout_sec = args.launch_timeout_sec
    if launch_timeout_sec <= 0.0:
        timeout_wall_sec = float(args.timeout_wall_sec or 0.0)
        if timeout_wall_sec <= 0.0:
            timeout_wall_factor = max(float(args.timeout_wall_factor or 1.0), 1.0)
            timeout_wall_sec = max(float(args.timeout) * timeout_wall_factor, float(args.timeout) + 120.0)
        # The wrapper timeout must include pre-episode Isaac readiness time in
        # addition to the task_generator's per-episode wall timeout.  Large
        # GRScenes USD loads routinely spend several minutes in LoadUsdScene
        # before task_reset is published, so a small fixed 180s startup margin
        # can kill the eval before the episode is released even though the
        # task_generator's own episode timeout has not started yet.
        launch_timeout_sec = max(timeout_wall_sec * max(args.episodes, 1) + 600.0, 300.0)

    video_proc = None
    if args.save_eval_video:
        video_proc = _start_eval_video_recorder(
            env,
            output_dir=output_dir,
            map_yaml_path=map_yaml_path,
            task_reset_topic=args.task_reset_topic,
            scenario_reset_topic=robot_scenario_reset_topic,
            finished_topic=args.finished_topic,
            ego_topic=robot_ego_topic,
            depth_topic=robot_depth_topic,
            camera_info_topic=robot_camera_info_topic,
            debug_overlay_topic=robot_debug_overlay_topic,
            sim_top_down_topic=robot_sim_top_down_topic,
            odom_topic=robot_odom_topic,
            goal_topic=robot_goal_topic,
            scan_topic=robot_scan_topic,
            fps=args.eval_video_fps,
            top_down_size_px=args.eval_video_top_down_size_px,
            top_down_window_m=args.eval_video_top_down_window_m,
        )
        if not _wait_for_file(video_index_path, timeout_sec=5.0):
            video_error = f'eval video recorder did not create video_index.json within 5s: {video_index_path}'
            _write_text(video_error_path, video_error)
            manifest['artifacts']['video_recorder_start_error'] = video_error
            _write_yaml(manifest_path, manifest)
    finished_proc = _start_finished_watcher(
        env,
        args.finished_topic,
        args.task_reset_topic,
        robot_scenario_reset_topic,
    )
    # Direct official-client mode does not require wrapper status for pass/fail,
    # but the official ROS client publishes a JSON status stream.  Record it per
    # run so batch sweeps get run-local model/control evidence even when the
    # long-lived external client was started with a different trace path.
    status_proc = _start_status_watcher(
        env,
        args.dual_vln_status_topic,
        dual_vln_status_path,
        internnav_trace_path,
        task_reset_topic=args.task_reset_topic,
        scenario_reset_topic=robot_scenario_reset_topic,
        require_reset_for_history=bool(args.internnav_direct_cmd_vel),
    )
    launch_proc = subprocess.Popen(
        launch_cmd,
        env=env,
        start_new_session=True,
    )

    deadline = time.monotonic() + launch_timeout_sec
    launch_returncode = None
    metrics_returncode = None
    social_metrics_returncode = None
    artifact_validation_returncode = None
    finished_observed = False
    timed_out = False

    def run_social_postprocess() -> int:
        nonlocal social_metrics_returncode, artifact_validation_returncode
        if not args.social_eval:
            return 0
        social_metrics_result = subprocess.run(social_metrics_cmd, env=env)
        social_metrics_returncode = social_metrics_result.returncode
        artifact_validation_result = subprocess.run(artifact_validation_cmd, env=env)
        artifact_validation_returncode = artifact_validation_result.returncode
        manifest['result']['social_metrics_returncode'] = social_metrics_returncode
        manifest['result']['artifact_validation_returncode'] = artifact_validation_returncode
        manifest['artifacts']['social_metrics_present'] = os.path.exists(os.path.join(output_dir, 'social_metrics.json'))
        manifest['artifacts']['artifact_validation_present'] = os.path.exists(os.path.join(output_dir, 'artifact_validation.json'))
        _write_yaml(manifest_path, manifest)
        if social_metrics_returncode != 0:
            return social_metrics_returncode
        return artifact_validation_returncode or 0

    try:
        while True:
            launch_returncode = launch_proc.poll()
            if launch_returncode is not None:
                break

            finished_returncode = finished_proc.poll()
            if finished_returncode == 0:
                finished_observed = True
                launch_returncode = _terminate_process_tree(
                    launch_proc,
                    grace_period_sec=args.shutdown_grace_period_sec,
                )
                break

            if time.monotonic() >= deadline:
                timed_out = True
                launch_returncode = _terminate_process_tree(
                    launch_proc,
                    grace_period_sec=args.shutdown_grace_period_sec,
                )
                break

            time.sleep(1.0)
    finally:
        if finished_proc.poll() is None:
            _terminate_process_tree(finished_proc, grace_period_sec=2.0)
        if status_proc is not None and status_proc.poll() is None:
            _terminate_process_tree(status_proc, grace_period_sec=2.0)
        if video_proc is not None and video_proc.poll() is None:
            _terminate_process_tree(video_proc, grace_period_sec=5.0)

    dual_vln_status = _read_json_if_exists(dual_vln_status_path)
    internnav_diagnostic_summary = _write_internnav_diagnostic_summary(
        internnav_trace_path,
        internnav_diagnostic_summary_path,
    )
    video_index = _read_json_if_exists(video_index_path) if args.save_eval_video else None
    video_error = _read_text_if_exists(video_error_path) if args.save_eval_video else None
    video_returncode = video_proc.returncode if video_proc is not None else None
    video_artifacts_complete = None
    video_artifact_issues: list[str] = []
    if args.save_eval_video:
        video_artifacts_complete = False
        episodes = video_index.get('episodes') if isinstance(video_index, dict) else []
        if not episodes:
            video_artifact_issues.append('missing_video_episodes')
        else:
            video_artifacts_complete = True
            for episode in episodes:
                if not isinstance(episode, dict):
                    continue
                for field in ('ego_frames', 'debug_overlay_frames', 'top_down_frames'):
                    if int(episode.get(field, 0) or 0) <= 0:
                        video_artifacts_complete = False
                        video_artifact_issues.append(f'{field}_empty')
    if internnav_diagnostic_summary is not None and video_artifact_issues:
        flags = internnav_diagnostic_summary.setdefault('fault_candidates', {}).setdefault('flags', [])
        if 'missing_or_empty_video_artifacts' not in flags:
            flags.append('missing_or_empty_video_artifacts')
        with open(internnav_diagnostic_summary_path, 'w', encoding='utf-8') as f:
            json.dump(internnav_diagnostic_summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    end_reason = _classify_end_reason(
        finished_observed=finished_observed,
        launch_returncode=launch_returncode,
        timed_out=timed_out,
        internnav_status=dual_vln_status,
        internnav_diagnostic_summary=internnav_diagnostic_summary,
    )

    manifest['artifacts']['internnav_status_present'] = dual_vln_status is not None
    manifest['artifacts']['internnav_trace_present'] = os.path.exists(internnav_trace_path)
    manifest['artifacts']['internnav_diagnostic_summary_present'] = internnav_diagnostic_summary is not None
    manifest['artifacts']['snapshot_files'] = sorted(snapshot_files.values())
    manifest['artifacts']['video_index_present'] = video_index is not None
    manifest['artifacts']['video_index'] = video_index
    manifest['artifacts']['video_recording_error'] = video_error
    manifest['artifacts']['video_artifacts_complete'] = video_artifacts_complete
    manifest['artifacts']['video_artifact_issues'] = video_artifact_issues
    manifest['result'].update(
        {
            'finished_observed': finished_observed,
            'launch_returncode': launch_returncode,
            'metrics_returncode': metrics_returncode,
            'social_metrics_returncode': social_metrics_returncode,
            'artifact_validation_returncode': artifact_validation_returncode,
            'timed_out': timed_out,
            'end_reason': end_reason,
            'dual_vln_status': dual_vln_status,
            'internnav_diagnostic_summary': internnav_diagnostic_summary,
            'video_recorder_returncode': video_returncode,
        }
    )
    _write_yaml(manifest_path, manifest)

    if finished_observed and launch_returncode not in (None, 0):
        launch_returncode = 0

    if timed_out:
        run_social_postprocess()
        manifest['result']['launch_returncode'] = launch_returncode
        _write_yaml(manifest_path, manifest)
        return 124 if launch_returncode == 0 else launch_returncode

    if launch_returncode != 0:
        run_social_postprocess()
        manifest['result']['launch_returncode'] = launch_returncode
        _write_yaml(manifest_path, manifest)
        return launch_returncode

    if args.skip_metrics:
        social_postprocess_returncode = run_social_postprocess()
        _write_yaml(manifest_path, manifest)
        return social_postprocess_returncode

    metrics_result = subprocess.run(metrics_cmd, env=env)
    metrics_returncode = metrics_result.returncode
    manifest['result']['metrics_returncode'] = metrics_returncode
    if metrics_returncode != 0 and manifest['result']['end_reason'] == 'finished':
        manifest['result']['end_reason'] = 'metrics_failed'
    social_postprocess_returncode = run_social_postprocess()
    _write_yaml(manifest_path, manifest)
    if metrics_returncode != 0:
        return metrics_returncode
    return social_postprocess_returncode


if __name__ == '__main__':
    raise SystemExit(main())
