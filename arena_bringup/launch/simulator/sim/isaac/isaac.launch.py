import launch.actions
import launch.substitutions

from arena_bringup.substitutions import LaunchArgument
from launch import LaunchDescription


def generate_launch_description():
    ld = []
    LaunchArgument.auto_append(ld)
    log_level = LaunchArgument(
        name='log_level',
        default_value='debug',
        description='Logging level',
    )
    save_data = LaunchArgument(
        name='save_data',
        default_value='false',
        description='data saving flag',
    )
    robot = LaunchArgument(
        name='robot',
        default_value='jackal',
        description='robot model name (used for nav topic namespace)',
    )
    world = LaunchArgument(
        name='world',
        default_value='',
        description='world being loaded, e.g. grscenes_1. Selects the dataset directory: '
                    'social_gen/traj_data/grscenes/<world>/{data,meta,videos}/',
    )
    return LaunchDescription([
        *ld,
        launch.actions.SetEnvironmentVariable(
            name='ARENA_LOG_LEVEL',
            value=log_level.substitution,
        ),
        launch.actions.SetEnvironmentVariable(
            name='ARENA_SAVE_DATA',
            value=save_data.substitution,
        ),
        launch.actions.SetEnvironmentVariable(
            name='ARENA_ROBOT',
            value=robot.substitution,
        ),
        launch.actions.SetEnvironmentVariable(
            name='ARENA_WORLD',
            value=world.substitution,
        ),
        launch.actions.ExecuteProcess(
            cmd=[
                'bash','-c','arena feature isaac launch --save-data "$ARENA_SAVE_DATA" --robot "$ARENA_ROBOT" --world "$ARENA_WORLD"'
            ],
            sigterm_timeout=launch.substitutions.LaunchConfiguration('sigterm_timeout', default='60'),
            sigkill_timeout=launch.substitutions.LaunchConfiguration('sigkill_timeout', default='30'),
            on_exit=[launch.actions.Shutdown()],
            output='log',
        )
    ])

