import os

from arena_bringup.future import PythonExpression
from arena_bringup.substitutions import (
    LaunchArgument,
    YAMLFileSubstitution,
    YAMLMergeSubstitution,
    YAMLReplaceSubstitution,
    YAMLRetrieveSubstitution,
)
from launch.actions import GroupAction
from launch.conditions import IfCondition
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml

from launch import LaunchDescription


def generate_launch_description():
    ss_root = FindPackageShare('arena_simulation_setup')
    robots_root = FindPackageShare('arena_robots')
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    robot = LaunchArgument('robot')
    task_generator_node = LaunchArgument('task_generator_node')
    namespace = LaunchArgument('namespace')
    frame = LaunchArgument('frame')
    use_sim_time = LaunchArgument('use_sim_time')
    global_planner = LaunchArgument('global_planner')
    local_planner = LaunchArgument('local_planner')
    inter_planner = LaunchArgument('inter_planner')
    enable_collision_monitor = LaunchArgument('enable_collision_monitor', default_value='true')

    controller_config_dir = PythonExpression(
        ['"model_wrapper" if "', local_planner.substitution, '" == "dual_vln" else "', local_planner.substitution, '"']
    )

    amcl = LaunchArgument('amcl')
    train_mode = LaunchArgument('train_mode', default_value='false')

    substitutions = YAMLMergeSubstitution(
        # Load default model params
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'defaults',
                'model_params.yaml'
            ])
        ),
        # Load robot-specific model params
        YAMLFileSubstitution(
            PathJoinSubstitution([
                robots_root,
                'robots',
                robot.substitution,
                'model_params.yaml'
            ])
        ),
        # Load default controller params
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'defaults',
                'controller_config.yaml'
            ])
        ),
        # Load controller-specific configuration based on local_planner argument
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'controllers',
                controller_config_dir,
                'controller_config.yaml'
            ])
        ),
        # Load default planner params
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'defaults',
                'planner_config.yaml'
            ])
        ),
        # Load planner-specific configuration based on global_planner argument
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'planners',
                global_planner.substitution,
                'planner_config.yaml'
            ])
        ),
        # Load default interplanner params
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'defaults',
                'interplanner_config.yaml'
            ])
        ),
        # Load interplanner-specific configuration based on inter_planner argument
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'interplanners',
                inter_planner.substitution,
                'interplanner_config.yaml'
            ])
        ),
        YAMLFileSubstitution.from_dict(
            {
                'frame': frame.substitution,
                **task_generator_node.dict,
                'namespace': namespace.substitution,
                # In train_mode the RL environment publishes cmd_vel directly.
                # Redirect the collision_monitor output to a dead topic so it
                # never overwrites the RL agent's velocity commands.
                'cmd_vel_out_topic': PythonExpression(
                    ['"cmd_vel_sink" if "', train_mode.substitution, '" == "true" else "cmd_vel"']
                ),
                'default_nav_to_pose_bt_xml': YAMLRetrieveSubstitution(
                    YAMLFileSubstitution(
                        PathJoinSubstitution([
                            ss_root,
                            'configs',
                            'nav2',
                            'interplanners',
                            inter_planner.substitution,
                            'interplanner_config.yaml'
                        ])
                    ),
                    'bt_navigator/ros__parameters/default_nav_to_pose_bt_xml'
                ),
                'default_nav_through_poses_bt_xml': YAMLRetrieveSubstitution(
                    YAMLFileSubstitution(
                        PathJoinSubstitution([
                            ss_root,
                            'configs',
                            'nav2',
                            'interplanners',
                            inter_planner.substitution,
                            'interplanner_config.yaml'
                        ])
                    ),
                    'bt_navigator/ros__parameters/default_nav_through_poses_bt_xml'
                ),
                'plugin_lib_names': YAMLRetrieveSubstitution(
                    YAMLFileSubstitution(
                        PathJoinSubstitution([
                            ss_root,
                            'configs',
                            'nav2',
                            'interplanners',
                            inter_planner.substitution,
                            'interplanner_config.yaml'
                        ])
                    ),
                    'bt_navigator/ros__parameters/plugin_lib_names'
                ),
            },
            substitute=True
        ),
    )

    substituted_parameters = YAMLReplaceSubstitution(
        obj=YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'nav2.yaml'
            ])
        ),
        substitutions=YAMLFileSubstitution(substitutions)
    )

    nav2_configured_params = ParameterFile(
        RewrittenYaml(
            source_file=substituted_parameters,
            root_key=namespace.substitution,
            param_rewrites={
                'use_sim_time': use_sim_time.substitution,
            },
            convert_types=True
        ),
        allow_substs=True,
    )

    robot_base_frame = YAMLRetrieveSubstitution(
        YAMLFileSubstitution(substitutions),
        os.path.join('robot_base_frame'),
    )

    robot_odom_frame = YAMLRetrieveSubstitution(
        YAMLFileSubstitution(substitutions),
        os.path.join('robot_odom_frame'),
    )

    first_observation_source_topic = YAMLRetrieveSubstitution(
        YAMLFileSubstitution(
            YAMLReplaceSubstitution(
                obj=YAMLFileSubstitution(substitutions),
                substitutions=YAMLFileSubstitution(substitutions)
            )
        ),
        PathJoinSubstitution([
            'observation_sources_dict',
            YAMLRetrieveSubstitution(
                YAMLFileSubstitution(substitutions),
                os.path.join('observation_sources', '0'),
            ),
            'topic',
        ]),
    )

    remappings = [
        ('map_server', '/map_server'),
        ('/tf', '/tf'),
        ('/tf_static', '/tf_static'),
        ('map', PathJoinSubstitution([task_generator_node.substitution, 'map'])),
    ]

    lifecycle_nodes = [
        'controller_server',
        'smoother_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'velocity_smoother',
    ]

    collision_monitor_enabled = IfCondition(enable_collision_monitor.substitution)

    bringup_cmd_group = GroupAction([
        *(SetRemap(src=r[0], dst=r[1]) for r in remappings),
        Node(
            package='topic_tools',
            executable='relay',
            name='goal_pose_relay',
            arguments=['/goal_pose', 'goal_pose'],
        ),
        # nav2 nodes
        Node(
            package='nav2_controller', executable='controller_server', name='controller_server',
            output='screen', parameters=[nav2_configured_params]
        ),
        Node(
            package='nav2_smoother', executable='smoother_server', name='smoother_server',
            output='screen', parameters=[nav2_configured_params]
        ),
        Node(
            package='nav2_planner', executable='planner_server', name='planner_server',
            output='screen', parameters=[nav2_configured_params]
        ),
        Node(
            package='nav2_behaviors', executable='behavior_server', name='behavior_server',
            output='screen', parameters=[nav2_configured_params]
        ),
        Node(
            package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
            output='screen', parameters=[nav2_configured_params]
        ),
        Node(
            package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
            output='screen', parameters=[nav2_configured_params]
        ),
        Node(
            package='nav2_collision_monitor', executable='collision_monitor', name='collision_monitor',
            output='screen', parameters=[nav2_configured_params],
            condition=collision_monitor_enabled,
        ),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_collision_monitor',
            output='screen',
            parameters=[
                {'autostart': True},
                {'node_names': ['collision_monitor']},
                nav2_configured_params
            ],
            condition=collision_monitor_enabled,
        ),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation',
            output='screen',
            parameters=[
                {'autostart': True},
                {'node_names': lifecycle_nodes},
                nav2_configured_params
            ]
        ),
    ])

    # Create the launch description and populate
    ld = LaunchDescription([
        *ld_items,
        bringup_cmd_group,
    ])

    return ld
