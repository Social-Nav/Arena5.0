import os

import launch.actions
import launch.launch_description_sources
import launch.substitutions
from ament_index_python.packages import get_package_share_directory

import launch
from arena_bringup.substitutions import LaunchArgument, SelectAction


def generate_launch_description():

    ld = []
    LaunchArgument.auto_append(ld)

    use_sim_time = LaunchArgument(
        name='use_sim_time',
    )

    headless = LaunchArgument(
        name='headless',
        default_value='False',
    )
    save_data = LaunchArgument(
        name='save_data',
        default_value='false',
        choices=['true', 'false'],
        description='Enable VLN dataset logging'
    )
    session_tag = LaunchArgument(
        name='session_tag',
        default_value='',
        description='Label prepended to collected_data subdirectory',
    )

    # TODO temporary
    world = LaunchArgument(
        name='world'
    )

    launch_simulator = SelectAction(launch.substitutions.LaunchConfiguration('simulator'))

    launch_simulator.add(
        'dummy',
        launch.actions.GroupAction([])
    )

    launch_simulator.add(
        'gazebo',
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'arena_bringup'), 'launch/simulator/sim/gazebo/gazebo.launch.py')
            ),
            launch_arguments={
                **use_sim_time.dict,
                **headless.dict,
                **world.dict,
            }.items(),
        )
    )

    launch_simulator.add(
        'isaac',
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'arena_bringup'), 'launch/simulator/sim/isaac/isaac.launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time.substitution,
                'save_data': save_data.substitution,
                'session_tag': session_tag.substitution,
                # 'headless': headless.substitution
            }.items(),
        )
    )

    simulator = LaunchArgument(
        name='simulator',
        choices=launch_simulator.keys,
    )

    ld = launch.LaunchDescription([
        *ld,
        launch_simulator,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
