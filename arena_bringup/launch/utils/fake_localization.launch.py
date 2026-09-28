
import launch
import launch_ros.actions
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='global_frame_id',
            default_value='map'
        ),
        launch.actions.DeclareLaunchArgument(
            name='odom_frame_id',
            default_value='odom'
        ),
        launch.actions.DeclareLaunchArgument(
            name='use_sim_time',
            default_value='true'
        ),
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tfpublisher',
            # Without use_sim_time this stamps map->odom with wall time while everything
            # else runs on /clock, so every lookup lands "in the future" and nav2 logs
            # Extrapolation Error. tf2_monitor reports the delay as the raw epoch (~1.8e9).
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            arguments=['0', '0', '0', '0', '0', '0', LaunchConfiguration('global_frame_id'), LaunchConfiguration('odom_frame_id')]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
