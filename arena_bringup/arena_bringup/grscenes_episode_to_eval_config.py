"""Convert recorded GRScenes episodes into Arena eval scenario configs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_GRSCENES_ROOT = Path('/home/ubuntu/arena_jazzy_ws/grscenes_')
DEFAULT_ARENA_WORLDS_DIR = Path('/home/ubuntu/arena_jazzy_ws/src/Arena/arena_simulation_setup/worlds')
DEFAULT_OUTPUT_DIR = Path('/home/ubuntu/arena_jazzy_ws/data/grscenes_eval_configs')
DEFAULT_ROBOT = 'Ai2_Bot2'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Read GRScenes recorded episode params.parquet and instruction.json, then emit '
            'an Arena native scenario.yaml plus a benchmark/eval config YAML.'
        )
    )
    parser.add_argument(
        'input',
        nargs='?',
        default=str(DEFAULT_GRSCENES_ROOT),
        help='Episode dir, params.parquet path, or GRScenes root to scan. Defaults to /home/ubuntu/arena_jazzy_ws/grscenes_.',
    )
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Directory for generated eval config YAML files.')
    parser.add_argument('--arena-worlds-dir', default=str(DEFAULT_ARENA_WORLDS_DIR), help='Arena source worlds directory.')
    parser.add_argument('--robot', default=DEFAULT_ROBOT, help='Robot name written into eval config.')
    parser.add_argument('--timeout-sec', type=float, default=120.0, help='Evaluation timeout_sec in generated config.')
    parser.add_argument('--goal-tolerance-m', type=float, default=0.45, help='Goal tolerance in generated config.')
    parser.add_argument('--start-jump-threshold-m', type=float, default=0.5, help='Warmup teleport jump threshold for auto start frame.')
    parser.add_argument('--warmup-scan-frames', type=int, default=10, help='Only treat jumps before this frame as warmup teleports.')
    parser.add_argument(
        '--yaw-source',
        choices=('source_scenario', 'camera_forward', 'camera_x', 'trajectory'),
        default='source_scenario',
        help='Yaw source for generated start/goal poses. source_scenario falls back to camera_forward if unavailable.',
    )
    parser.add_argument(
        '--write-native-to-arena-world',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If the Arena world exists, write native scenario.yaml under arena_simulation_setup/worlds/<world>/scenarios/<id>.',
    )
    parser.add_argument('--dry-run', action='store_true', help='Analyze and print JSON without writing files.')
    args = parser.parse_args(argv)

    episodes = list(discover_episode_dirs(Path(args.input)))
    if not episodes:
        raise SystemExit(f'No GRScenes episode dirs found under: {args.input}')

    results = []
    for episode_dir in episodes:
        results.append(
            convert_episode(
                episode_dir,
                output_dir=Path(args.output_dir),
                arena_worlds_dir=Path(args.arena_worlds_dir),
                robot=args.robot,
                timeout_sec=args.timeout_sec,
                goal_tolerance_m=args.goal_tolerance_m,
                start_jump_threshold_m=args.start_jump_threshold_m,
                warmup_scan_frames=args.warmup_scan_frames,
                yaw_source=args.yaw_source,
                write_native_to_arena_world=args.write_native_to_arena_world,
                dry_run=args.dry_run,
            )
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def discover_episode_dirs(input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.name != 'params.parquet':
            return []
        return [path.parent.parent]
    if (path / 'data' / 'params.parquet').is_file():
        return [path]
    return sorted({p.parent.parent for p in path.glob('**/data/params.parquet')})


def convert_episode(
    episode_dir: Path,
    *,
    output_dir: Path,
    arena_worlds_dir: Path,
    robot: str,
    timeout_sec: float,
    goal_tolerance_m: float,
    start_jump_threshold_m: float,
    warmup_scan_frames: int,
    yaw_source: str,
    write_native_to_arena_world: bool,
    dry_run: bool,
) -> dict[str, Any]:
    episode_dir = episode_dir.expanduser().resolve()
    params_path = episode_dir / 'data' / 'params.parquet'
    if not params_path.is_file():
        raise FileNotFoundError(params_path)

    info = infer_episode_info(episode_dir)
    trajectory = read_camera_trajectory(params_path)
    start_index = choose_start_index(
        trajectory,
        jump_threshold_m=start_jump_threshold_m,
        warmup_scan_frames=warmup_scan_frames,
    )
    goal_index = len(trajectory) - 1

    source_scenario = find_source_scenario(episode_dir, info)
    source_robot_pose = read_source_robot_pose(source_scenario)
    source_dynamic = read_source_dynamic(source_scenario)

    start_pose = pose_from_record(trajectory[start_index], yaw_kind='camera_forward')
    goal_pose = pose_from_record(trajectory[goal_index], yaw_kind='camera_forward')

    requested_yaw_source = yaw_source
    effective_yaw_source = yaw_source
    if yaw_source == 'source_scenario':
        if source_robot_pose is not None:
            start_pose[2] = source_robot_pose['start'][2]
            goal_pose[2] = source_robot_pose['goal'][2]
        else:
            effective_yaw_source = 'camera_forward'
    elif yaw_source in ('camera_forward', 'camera_x'):
        start_pose = pose_from_record(trajectory[start_index], yaw_kind=yaw_source)
        goal_pose = pose_from_record(trajectory[goal_index], yaw_kind=yaw_source)
    elif yaw_source == 'trajectory':
        start_pose[2] = trajectory_yaw(trajectory, start_index, forward=True)
        goal_pose[2] = trajectory_yaw(trajectory, goal_index, forward=False)

    peds = read_pedestrian_summary(params_path)
    native_dynamic = source_dynamic if source_dynamic is not None else dynamic_from_peds(peds)
    scenario_id = make_scenario_id(info)
    native_scenario = {
        'dynamic': native_dynamic,
        'robots': [
            {
                'start': round_pose(start_pose),
                'goal': round_pose(goal_pose),
            }
        ],
    }

    world_dir = arena_worlds_dir / info['world']
    world_exists = world_dir.is_dir()
    if write_native_to_arena_world and world_exists:
        native_scenario_path = world_dir / 'scenarios' / scenario_id / 'scenario.yaml'
        native_scenario_ref = f'package://arena_simulation_setup/worlds/{info["world"]}/scenarios/{scenario_id}/scenario.yaml'
    else:
        native_scenario_path = output_dir / 'native_scenarios' / info['world'] / scenario_id / 'scenario.yaml'
        native_scenario_ref = str(native_scenario_path)

    instruction = read_instruction(episode_dir)
    eval_config = build_eval_config(
        info=info,
        world_dir=world_dir if world_exists else None,
        scenario_id=scenario_id,
        native_scenario_ref=native_scenario_ref,
        native_scenario_name=scenario_id,
        robot=robot,
        start_pose=start_pose,
        goal_pose=goal_pose,
        instruction=instruction,
        expected_humans=len(native_dynamic),
        timeout_sec=timeout_sec,
        goal_tolerance_m=goal_tolerance_m,
    )
    eval_config_path = output_dir / f'{scenario_id}.yaml'

    report = {
        'episode_dir': str(episode_dir),
        'params_path': str(params_path),
        'instruction': instruction,
        'world': info['world'],
        'source_scenario': info['source_scenario'],
        'timestamp': info['timestamp'],
        'episode': info['episode'],
        'scenario_id': scenario_id,
        'rows': len(trajectory),
        'start_index': start_index,
        'goal_index': goal_index,
        'raw_first_pose_xy_yaw_camera_forward': round_pose(pose_from_record(trajectory[0], yaw_kind='camera_forward')),
        'selected_start_pose_xy_yaw': round_pose(start_pose),
        'selected_goal_pose_xy_yaw': round_pose(goal_pose),
        'yaw_source_requested': requested_yaw_source,
        'yaw_source_effective': effective_yaw_source,
        'source_robot_pose_xy_yaw': source_robot_pose,
        'human_count': len(native_dynamic),
        'arena_world_exists': world_exists,
        'native_scenario_path': str(native_scenario_path),
        'eval_config_path': str(eval_config_path),
        'dry_run': dry_run,
    }

    if not dry_run:
        native_scenario_path.parent.mkdir(parents=True, exist_ok=True)
        native_scenario_path.write_text(
            '# Generated from GRScenes recorded episode params.parquet\n'
            + yaml.safe_dump(native_scenario, sort_keys=False, allow_unicode=True),
            encoding='utf-8',
        )
        copy_behavior_trees(native_dynamic, native_scenario_path.parent, source_scenario=source_scenario)
        eval_config_path.parent.mkdir(parents=True, exist_ok=True)
        eval_config_path.write_text(yaml.safe_dump(eval_config, sort_keys=False, allow_unicode=True), encoding='utf-8')

    return report


def infer_episode_info(episode_dir: Path) -> dict[str, str]:
    # Expected: <root>/<world>/<scenario>/<timestamp>/<episode>/...
    parts = episode_dir.parts
    if len(parts) < 4:
        raise ValueError(f'Cannot infer GRScenes episode fields from path: {episode_dir}')
    return {
        'world': episode_dir.parents[2].name,
        'source_scenario': episode_dir.parents[1].name,
        'timestamp': episode_dir.parent.name,
        'episode': episode_dir.name,
    }


def make_scenario_id(info: dict[str, str]) -> str:
    timestamp = info['timestamp'].replace('-', '').replace(':', '').replace('_', '_')
    raw = f"recorded_{info['source_scenario']}_{timestamp}_{info['episode']}"
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', raw).strip('_')


def read_camera_trajectory(params_path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError('pandas + pyarrow are required to read params.parquet') from exc

    df = pd.read_parquet(params_path)
    if 'observation.camera_state' not in df.columns:
        raise ValueError(f'{params_path} is missing observation.camera_state')

    rows = []
    for _, row in df.iterrows():
        state = row['observation.camera_state']
        if state is None or len(state) != 16:
            continue
        rows.append({'frame_index': int(row.get('frame_index', len(rows))), 'matrix': [float(v) for v in state]})
    if not rows:
        raise ValueError(f'{params_path} contains no valid camera_state rows')
    return rows


def read_pedestrian_summary(params_path: Path) -> dict[str, dict[str, Any]]:
    try:
        import pandas as pd
    except Exception:
        return {}
    df = pd.read_parquet(params_path, columns=['observation.peds_state'])
    if df.empty:
        return {}
    first = df.iloc[0]['observation.peds_state'] or {}
    last = df.iloc[-1]['observation.peds_state'] or {}
    out: dict[str, dict[str, Any]] = {}
    for key, mat in first.items():
        if not isinstance(mat, (list, tuple)) or len(mat) != 16:
            continue
        # GRScenes stores each pedestrian twice in many episodes: body at z≈0 and head/collider at z≈0.875.
        # Keep the z≈0 entries as HuNav agents.
        if abs(float(mat[11])) > 0.25:
            continue
        last_mat = last.get(key, mat)
        out[str(key)] = {
            'start': [float(mat[3]), float(mat[7]), yaw_from_matrix(mat, 'camera_x')],
            'goal': [float(last_mat[3]), float(last_mat[7]), yaw_from_matrix(last_mat, 'camera_x')],
        }
    return out


def choose_start_index(trajectory: list[dict[str, Any]], *, jump_threshold_m: float, warmup_scan_frames: int) -> int:
    limit = max(0, min(len(trajectory) - 1, warmup_scan_frames))
    for idx in range(limit):
        if xy_distance(trajectory[idx], trajectory[idx + 1]) >= jump_threshold_m:
            return idx + 1
    return 0


def pose_from_record(record: dict[str, Any], *, yaw_kind: str) -> list[float]:
    m = record['matrix']
    return [float(m[3]), float(m[7]), yaw_from_matrix(m, yaw_kind)]


def yaw_from_matrix(matrix: list[float], yaw_kind: str) -> float:
    if yaw_kind == 'camera_x':
        return normalize_degrees(math.degrees(math.atan2(float(matrix[4]), float(matrix[0]))))
    # The camera optical/forward axis is stored in the third matrix column for the recorded GRScenes data.
    return normalize_degrees(math.degrees(math.atan2(float(matrix[6]), float(matrix[2]))))


def trajectory_yaw(trajectory: list[dict[str, Any]], index: int, *, forward: bool) -> float:
    if forward:
        current = trajectory[index]
        candidates = trajectory[index + 1:]
        sign = 1.0
    else:
        current = trajectory[index]
        candidates = reversed(trajectory[:index])
        sign = -1.0
    x0, y0 = current['matrix'][3], current['matrix'][7]
    for candidate in candidates:
        dx = (candidate['matrix'][3] - x0) * sign
        dy = (candidate['matrix'][7] - y0) * sign
        if math.hypot(dx, dy) > 0.05:
            return normalize_degrees(math.degrees(math.atan2(dy, dx)))
    return pose_from_record(current, yaw_kind='camera_forward')[2]


def xy_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a['matrix'][3]) - float(b['matrix'][3]), float(a['matrix'][7]) - float(b['matrix'][7]))


def normalize_degrees(value: float) -> float:
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    return value


def round_pose(pose: list[float]) -> list[float]:
    return [round(float(pose[0]), 5), round(float(pose[1]), 5), round(normalize_degrees(float(pose[2])), 5)]


def find_source_scenario(episode_dir: Path, info: dict[str, str]) -> Path | None:
    root = episode_dir.parents[3]
    path = root / info['world'] / 'scenarios' / info['source_scenario'] / 'scenario.yaml'
    return path if path.is_file() else None


def read_source_robot_pose(path: Path | None) -> dict[str, list[float]] | None:
    if path is None:
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception:
        return None
    robots = data.get('robots') if isinstance(data, dict) else None
    if not isinstance(robots, list) or not robots:
        return None
    first = robots[0]
    if not isinstance(first, dict):
        return None
    start = first.get('start')
    goal = first.get('goal')
    if is_pose(start) and is_pose(goal):
        return {'start': round_pose(start), 'goal': round_pose(goal)}
    return None


def read_source_dynamic(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception:
        return None
    dynamic = data.get('dynamic') if isinstance(data, dict) else None
    return dynamic if isinstance(dynamic, list) else None


def dynamic_from_peds(peds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    models = ['female_adult_business_02', 'male_adult_medical_01', 'F_Business_02', 'M_Young_02']
    dynamic = []
    for index, key in enumerate(sorted(peds)):
        ped = peds[key]
        dynamic.append(
            {
                'name': f'hunav_{index + 1}',
                'model': models[index % len(models)],
                'pose': round_pose(ped['start']),
                'behavior_tree': 'BTRegularNav.xml',
                'velocity': 0.8,
                'desired_velocity': 1.0,
                'waypoints': [round_pose(ped['goal'])],
            }
        )
    return dynamic


def copy_behavior_trees(dynamic: list[dict[str, Any]], scenario_dir: Path, *, source_scenario: Path | None) -> None:
    names = sorted(
        {
            str(agent.get('behavior_tree'))
            for agent in dynamic
            if isinstance(agent, dict) and str(agent.get('behavior_tree', '')).strip()
        }
    )
    if not names:
        return

    fallback_dir = DEFAULT_ARENA_WORLDS_DIR.parent / 'configs' / 'hunav' / 'behavior_trees'
    source_dir = source_scenario.parent if source_scenario is not None else None
    for name in names:
        target = scenario_dir / name
        if target.exists():
            continue
        candidates = []
        if source_dir is not None:
            candidates.append(source_dir / name)
        candidates.append(fallback_dir / name)
        for candidate in candidates:
            if candidate.is_file():
                target.write_text(candidate.read_text(encoding='utf-8'), encoding='utf-8')
                break


def read_instruction(episode_dir: Path) -> str:
    path = episode_dir / 'instruction' / 'instruction.json'
    if not path.is_file():
        return 'navigate'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return 'navigate'
    parsed = data.get('parsed_result') if isinstance(data, dict) else None
    if isinstance(parsed, dict) and str(parsed.get('instruction', '')).strip():
        return str(parsed['instruction']).strip()
    if isinstance(data, dict) and str(data.get('instruction', '')).strip():
        return str(data['instruction']).strip()
    return 'navigate'


def build_eval_config(
    *,
    info: dict[str, str],
    world_dir: Path | None,
    scenario_id: str,
    native_scenario_ref: str,
    native_scenario_name: str,
    robot: str,
    start_pose: list[float],
    goal_pose: list[float],
    instruction: str,
    expected_humans: int,
    timeout_sec: float,
    goal_tolerance_m: float,
) -> dict[str, Any]:
    world = info['world']
    if world_dir is not None:
        world_config = f'package://arena_simulation_setup/worlds/{world}/world.yaml'
        map_yaml = f'package://arena_simulation_setup/worlds/{world}/map/map.yaml'
    else:
        world_config = f'package://arena_simulation_setup/worlds/{world}/world.yaml'
        map_yaml = f'package://arena_simulation_setup/worlds/{world}/map/map.yaml'
    return {
        'id': scenario_id,
        'schema_version': 0.1,
        'source': {
            'dataset': 'grscenes_recorded_episode',
            'world': world,
            'scenario': info['source_scenario'],
            'timestamp': info['timestamp'],
            'episode': info['episode'],
        },
        'world': {
            'name': world,
            'world_config': world_config,
            'map_yaml': map_yaml,
            'native_scenario': {
                'name': native_scenario_name,
                'file': native_scenario_ref,
            },
        },
        'robot': {
            'name': robot,
            'local_planner': 'dual_vln',
            'start': {'pose_xy_yaw': round_pose(start_pose)},
            'goal': {'pose_xy_yaw': round_pose(goal_pose), 'tolerance_m': float(goal_tolerance_m)},
            'command_interface': {
                'cmd_vel_topic': f'/task_generator_node/{robot}/cmd_vel',
                'odom_topic': f'/task_generator_node/{robot}/odom',
                'goal_topic': f'/task_generator_node/{robot}/episode_goal_pose',
            },
        },
        'language': {
            'instruction_type': 'recorded_episode_instruction',
            'instruction': instruction,
            'rephrases': [],
        },
        'humans': {
            'simulator': 'hunav',
            'source': 'native_scenario',
            'expected_count': int(expected_humans),
            'native_scenario_file': native_scenario_ref,
        },
        'task_spec': {
            'type': 'grscenes_recorded_vln_episode',
            'predicates': {
                'entities': {'robot': robot, 'humans': 'all_hunav_pedestrians', 'goal': 'robot.goal'},
                'success': ['goal_reached(robot, goal)', 'robot_moved(robot, min_path_length_m=0.1)'],
                'social_constraints': [
                    'human_collision_count == 0',
                    'near_miss_count == 0',
                    'personal_space_violation_time_sec == 0.0',
                    'min_human_distance_m >= 0.25',
                ],
                'failure': ['timeout', 'no_motion', 'collision', 'missing_humans', 'missing_required_artifacts'],
            },
        },
        'evaluation': {
            'timeout_sec': float(timeout_sec),
            'repetitions': 1,
            'random_seed': None,
        },
    }


def is_pose(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except Exception:
        return False


if __name__ == '__main__':
    raise SystemExit(main())
