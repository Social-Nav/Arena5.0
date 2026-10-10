"""LightNav identity and diagnostic artifacts for the shared evaluator."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def runtime_identity(robot='Ai2_Bot2'):
    import yaml
    from ament_index_python.packages import get_package_share_directory
    root = Path('/opt/arena_ws/deps/lightnav-client')
    runtime = json.loads((root / 'runtime.json').read_text())
    source = json.loads((Path(runtime['stage']) / 'source.json').read_text())
    raw = subprocess.check_output(['docker', 'inspect', '--format',
        '{"image_id":{{json .Image}},"mounts":{{json .Mounts}},"labels":{{json .Config.Labels}}}',
        'arena-lightnav-lightnav-1'], text=True, timeout=10)
    identity = json.loads(raw)
    robot_path = Path(get_package_share_directory('arena_robots')) / 'robots' / robot / 'model_params.yaml'
    robot_config = yaml.safe_load(robot_path.read_text())
    identity.update(client_source=source, client_python='/opt/arena_ws/deps/lightnav-client/venv/bin/python',
        controller='lightnav_mpc', execution_mode='external_direct_cmd_vel',
        mpc={'control_rate_hz': 10., 'horizon': 5, 'dt_sec': .1,
             'timing_source': 'simulation', 'command_rate_hz': 20., 'watchdog_timebase': 'wall',
             'waypoint_timebase': None,
             'q_weights': [10., 10., 1.], 'r_weights': [.1, .1], 'output_scale': [1., 1.]},
        native_client={'response_timeout_sec': 3., 'jpeg_size': [480, 270], 'jpeg_quality': 80})
    identity['robot_limits'] = {
        'config_path': str(robot_path), 'sha256': hashlib.sha256(robot_path.read_bytes()).hexdigest(),
        'max_linear_velocity_m_s': robot_config['max_vel_x'],
        'max_angular_velocity_rad_s': robot_config['max_vel_theta'],
        'linear_acceleration_limit_m_s2': min(abs(robot_config['velocity_smoother_max_accel'][0]),
                                            abs(robot_config['velocity_smoother_max_decel'][0])),
        'angular_acceleration_limit_rad_s2': min(abs(robot_config['velocity_smoother_max_accel'][2]),
                                               abs(robot_config['velocity_smoother_max_decel'][2])),
        'base_frame': robot + '/' + robot_config.get('robot_base_frame', 'base_link'),
        'odom_frame': robot + '/' + robot_config.get('robot_odom_frame', 'odom'),
    }
    return identity


def read_trace(path):
    records = []
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try:
                record = json.loads(line)
                payload = record.get('parsed', record)
                if isinstance(payload, dict):
                    records.append(payload)
            except (ValueError, TypeError):
                continue
    return records


def diagnostic_summary(trace_path, output_path=None):
    records = read_trace(trace_path)
    counts = Counter(record.get('status') for record in records)
    predictions = [r for r in records if r.get('status') == 'lightnav_inference']
    commands = [r for r in records if r.get('status') == 'lightnav_command']
    accepted = {(r.get('episode'), r.get('native_episode'), r.get('request_id'))
                for r in records if r.get('status') == 'path_accepted'}
    inferred = {(r.get('episode'), r.get('native_episode'), r.get('request_id')) for r in predictions}
    nonzero = [r for r in commands if abs(r.get('linear_x', 0)) + abs(r.get('angular_z', 0)) > 1e-6]
    unlinked = [r for r in nonzero if (r.get('episode'), r.get('native_episode'), r.get('request_id')) not in accepted & inferred]
    latencies = [r['response']['latency_ms'] for r in predictions]
    frame_sessions = {}
    for record in predictions:
        key = (record.get('episode'), record.get('native_episode'))
        stamps = frame_sessions.setdefault(key, [])
        stamp = record['response'].get('capture_stamp_ns')
        if type(stamp) is int:
            stamps.append(stamp)
    frame_stats = [dict(episode=episode, native_episode=native_episode,
                        stamped_inference_count=len(stamps), unique_image_count=len(set(stamps)),
                        duplicate_capture_count=len(stamps)-len(set(stamps)))
                   for (episode, native_episode), stamps in frame_sessions.items()]
    summary = dict(backend='lightnav', controller='lightnav_mpc', execution_mode='external_direct_cmd_vel',
        trace_record_count=len(records),
        inference_count=len(predictions), command_count=len(commands), nonzero_command_count=len(nonzero),
        path_accepted_count=counts['path_accepted'], path_rejected_count=counts['path_rejected'],
        error_count=counts['error'], unlinked_command_count=len(unlinked),
        commands_linked_to_inference=bool(nonzero) and not unlinked,
        command_stats={'sample_count': len(commands), 'nonzero_count': len(nonzero),
                       'forward_count': sum(r.get('linear_x', 0.) > 1e-6 for r in commands),
                       'rotate_count': sum(abs(r.get('angular_z', 0.)) > 1e-6 for r in commands),
                       'stop_count': sum(bool(r['response'].get('stop')) for r in predictions)},
        mean_client_latency_ms=statistics.mean(latencies) if latencies else None,
        unique_image_count=sum(s['unique_image_count'] for s in frame_stats),
        duplicate_capture_count=sum(s['duplicate_capture_count'] for s in frame_stats),
        missing_capture_stamp_count=len(predictions)-sum(s['stamped_inference_count'] for s in frame_stats),
        episode_frame_stats=frame_stats,
        episodes_with_inference=sorted({r['episode'] for r in predictions}),
        event_counts=dict(counts), fault_candidates={'flags': []})
    if output_path:
        Path(output_path).write_text(json.dumps(summary, indent=2) + '\n')
    return summary
