import os
from pathlib import Path

import launch
import launch_ros.actions
import launch_ros.parameter_descriptions
from ament_index_python.packages import get_package_share_directory


def _default_map_yaml() -> str:
    world_name = os.environ.get('ARENA_WORLD', 'map_empty').strip() or 'map_empty'
    candidates = []

    host_ws_dir = os.environ.get('HOST_ARENA_WS_DIR', '').strip()
    if host_ws_dir:
        candidates.append(
            Path(host_ws_dir) / 'src' / 'Arena' / 'arena_simulation_setup' / 'worlds' / world_name / 'map' / 'map.yaml'
        )

    candidates.append(
        Path('/opt/arena_ws/src/Arena/arena_simulation_setup/worlds') / world_name / 'map' / 'map.yaml'
    )

    try:
        share_dir = Path(get_package_share_directory('arena_simulation_setup'))
        candidates.append(share_dir / 'worlds' / world_name / 'map' / 'map.yaml')
    except Exception:
        pass

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    fallback = Path('/opt/arena_ws/src/Arena/arena_simulation_setup/worlds/map_empty/map/map.yaml')
    return str(fallback if fallback.exists() else candidates[0])


def generate_launch_description():
    default_map_yaml = _default_map_yaml()

    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'yaml_filename',
            default_value=default_map_yaml,
        ),
        launch_ros.actions.Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{
                'topic_name': 'map',
                'frame_id': 'map',
                'yaml_filename': launch_ros.parameter_descriptions.ParameterValue(
                    launch.substitutions.LaunchConfiguration('yaml_filename'),
                    value_type=str,
                ),
                'use_sim_time': True,
            }],
        ),
        launch_ros.actions.Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map_server',
            parameters=[{
                'node_names': ['map_server'],
                # WorldManagerROS drives configure / activate explicitly on Humble.
                # Keeping lifecycle_manager available without autostart avoids a
                # startup race where lifecycle_manager sends ACTIVATE before the
                # map server has completed CONFIGURE.
                'autostart': False,
                'use_sim_time': True,
                'bond_timeout': 0.0,
            }]
        ),
    ])

    return ld


if __name__ == '__main__':
    generate_launch_description()
