from launch import LaunchDescription
from launch_ros.actions import Node
from arena_bringup.substitutions import LaunchArgument


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchArgument('use_sim_time', default_value='true')
    namespace = LaunchArgument('namespace')

    return LaunchDescription([
        use_sim_time,
        namespace,

        # Agent Manager Node
        Node(
            package='hunav_agent_manager',
            executable='arena_hunav_agent_manager',
            namespace=namespace.substitution,
            name='hunav_agent_manager',
            output='screen',
            parameters=[
                use_sim_time.param(bool)
            ]
        ),

        # Bridge: arena_people_msgs/Pedestrians → people_msgs/People
        Node(
            package='arena_hunav_sim_bridge',
            executable='arena_peds_to_people_bridge',
            name='arena_peds_to_people_bridge',
            output='screen',
            parameters=[
                use_sim_time.param(bool),
                {'input_topic': 'arena_peds'},
                {'output_topic': '/people'},
            ],
            remappings=[
                ('arena_peds', [namespace.substitution, '/arena_peds']),
            ],
        ),
    ])
