"""NaVILA provenance, model probes and evidence used by the shared evaluator."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
from urllib.parse import urlparse


def check_weights(root):
    root = Path(root)
    index = json.loads((root / 'llm/model.safetensors.index.json').read_text())
    names = {f'llm/{name}' for name in index['weight_map'].values()}
    names.update(('mm_projector/model.safetensors', 'vision_tower/model.safetensors'))
    result = {}
    for name in sorted(names):
        path = root / name
        size = path.stat().st_size
        with path.open('rb') as f:
            header_size = int.from_bytes(f.read(8), 'little')
            if not 0 < header_size < min(size, 16777216):
                raise ValueError(f'invalid safetensors header: {path}')
            header = json.loads(f.read(header_size))
        ends = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
        if not ends or max(ends) + header_size + 8 != size:
            raise ValueError(f'incomplete model tensor file: {path}')
        result[name] = size
    config = json.loads((root / 'config.json').read_text())
    if config.get('num_video_frames') != 8:
        raise ValueError('NaVILA adapter requires the eight-frame model')
    return {'files': result, 'total_bytes': sum(result.values()), 'num_video_frames': 8}


def model_smoke(endpoint):
    # The production model itself is probed; this is never a robot control request.
    import base64
    import io
    from PIL import Image
    image = io.BytesIO()
    Image.new('RGB', (512, 512)).save(image, 'JPEG')
    request = json.dumps({'images': [base64.b64encode(image.getvalue()).decode()] * 8,
                          'query': 'Stop.'}).encode()
    address = urlparse(endpoint)
    with socket.create_connection((address.hostname, address.port), timeout=3) as connection:
        connection.settimeout(60)
        connection.sendall(len(request).to_bytes(8, 'big') + request)
        def receive(n):
            data = bytearray()
            while len(data) < n:
                chunk = connection.recv(n - len(data))
                if not chunk:
                    raise ConnectionError('incomplete NaVILA smoke response')
                data.extend(chunk)
            return data
        size = int.from_bytes(receive(8), 'big')
        if not 0 < size < 65536:
            raise ValueError('invalid NaVILA smoke response length')
        output = json.loads(receive(size))
    if not isinstance(output, str) or output.lower().startswith('server error:'):
        raise ValueError(f'NaVILA smoke failed: {output!r}')
    print(json.dumps({'navila_real_inference_smoke': output}), flush=True)
    return output


def runtime_identity(robot='Ai2_Bot2'):
    from ament_index_python.packages import get_package_share_directory
    import yaml
    raw = subprocess.check_output(['docker', 'inspect', '--format',
        '{"image_id":{{json .Image}},"mounts":{{json .Mounts}}}', 'arena-navila-navila-1'], text=True)
    identity = json.loads(raw)
    robot_path = Path(get_package_share_directory('arena_robots')) / 'robots' / robot / 'model_params.yaml'
    config = yaml.safe_load(robot_path.read_text())
    identity.update(controller='navila_action_velocity', execution_mode='external_direct_cmd_vel',
        model_protocol='tcp_length8_json', model_frames=8, history_sample_period_sim_sec=.5,
        command_rate_sim_hz=20., watchdog_timebase='wall', command_timeout_sec=.5,
        forward_speed_m_s=.5, turn_speed_rad_s=0.5235987755982988,
        action_durations_sim_sec=[.5, 1., 1.5], inference_wait='robot_stopped_scene_continues',
        robot_limits={'config_path': str(robot_path), 'sha256': hashlib.sha256(robot_path.read_bytes()).hexdigest(),
                      'direct_max_velocity': config['velocity_smoother_max_velocity'],
                      'direct_min_velocity': config['velocity_smoother_min_velocity']})
    return identity


def diagnostic_summary(trace_path, output_path=None):
    from arena_bringup.lightnav_support import read_trace
    records = read_trace(trace_path)
    counts = Counter(r.get('status') for r in records)
    predictions = [r for r in records if r.get('status') == 'navila_inference']
    commands = [r for r in records if r.get('status') == 'navila_command']
    def key(r):
        return (r.get('run_token'), r.get('episode'), r.get('generation'), r.get('request_id'))
    accepted = {key(r) for r in records if r.get('status') == 'action_accepted'}
    inferred = {key(r) for r in predictions}
    moving = [r for r in commands if abs(r.get('linear_x', 0)) + abs(r.get('angular_z', 0)) > 1e-6]
    unlinked = [r for r in moving if key(r) not in accepted & inferred]
    sessions = {}
    for r in predictions:
        sessions.setdefault((r.get('episode'), r.get('generation')), []).append(r['capture_stamp_ns'])
    latency = [r['latency_ms'] for r in predictions]
    result = dict(backend='navila', controller='navila_action_velocity', execution_mode='external_direct_cmd_vel',
        trace_record_count=len(records), inference_count=len(predictions), command_count=len(commands),
        nonzero_command_count=len(moving), action_accepted_count=counts['action_accepted'],
        action_completed_count=counts['action_completed'], error_count=counts['error'],
        unlinked_command_count=len(unlinked), commands_linked_to_inference=bool(moving) and not unlinked,
        unique_image_count=sum(len(set(v)) for v in sessions.values()),
        duplicate_capture_count=sum(len(v) - len(set(v)) for v in sessions.values()),
        mean_client_latency_ms=statistics.mean(latency) if latency else None,
        command_stats={'forward_count': sum(r.get('linear_x', 0) > 0 for r in moving),
                       'rotate_count': sum(abs(r.get('angular_z', 0)) > 0 for r in moving),
                       'stop_count': counts['model_stop']}, event_counts=dict(counts),
        action_measurements=[r for r in records if r.get('status') in {'action_completed', 'action_aborted'}],
        fault_candidates={'flags': ['unlinked_control'] if unlinked else []})
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2) + '\n')
    return result
