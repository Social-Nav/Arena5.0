import os

import launch_ros
from arena_bringup.future import PythonExpression
from arena_bringup.substitutions import LaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare

import launch
import launch.actions
import launch.launch_description_sources
import launch.substitutions


def generate_launch_description():

    ss_path = FindPackageShare('arena_simulation_setup')

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    use_sim_time = LaunchArgument("use_sim_time")

    task_generator_node = LaunchArgument('task_generator_node')
    namespace = LaunchArgument("namespace")
    robot = LaunchArgument("robot")
    frame = LaunchArgument("frame")

    global_planner = LaunchArgument("global_planner")
    local_planner = LaunchArgument("local_planner")
    inter_planner = LaunchArgument("inter_planner", default_value="navigate_to_pose")

    record_data_dir = LaunchArgument('record_data_dir', default_value='')
    amcl = LaunchArgument('amcl', default_value='false')
    train_mode = LaunchArgument('train_mode', default_value='false')
    enable_collision_monitor = LaunchArgument('enable_collision_monitor', default_value='true')
    agents_dir = LaunchArgument(
        'agents_dir',
        default_value=launch.substitutions.EnvironmentVariable('ROSNAV_AGENTS_DIR', default_value=''),
        description=(
            'Base directory for agent artifacts. '
            'Forwarded as ROSNAV_AGENTS_DIR to the action server. '
            'Defaults to the ROSNAV_AGENTS_DIR env var.'
        ),
    )

    # Include the Nav2 launch file
    nav2_launch = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            launch.substitutions.PathJoinSubstitution(
                [
                    ss_path,
                    "launch",
                    "nav2.launch.py",
                ]
            )),
        launch_arguments={
            **use_sim_time.dict,
            **robot.dict,
            **task_generator_node.dict,
            **namespace.dict,
            **global_planner.dict,
            **local_planner.dict,
            **inter_planner.dict,
            **frame.dict,
            **amcl.dict,
            **train_mode.dict,
            **enable_collision_monitor.dict,
        }.items(),
    )

    # launch robot control
    state_pub_launch = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            launch.substitutions.PathJoinSubstitution(
                [
                    ss_path,
                    "launch",
                    "state_publisher.launch.py",
                ]
            )),
        launch_arguments={
            **use_sim_time.dict,
            **frame.dict,
            **namespace.dict,
            **robot.dict,
        }.items(),
    )

    data_recorder = launch_ros.actions.Node(
        package='arena_evaluation',
        executable='record',
        name=PythonExpression(['"data_recorder" + "', namespace.substitution, '".replace("/","_")']),
        arguments=['--dir', record_data_dir.substitution],
        parameters=[
            {
                'local_planner': local_planner.substitution,
                'inter_planner': inter_planner.substitution,
                'agent_name': launch.substitutions.LaunchConfiguration('agent_name'),
                # task_generator publishes task_reset under its fully-qualified name;
                # recorders use this to attribute samples to episodes.
                'scenario_reset_topic': PythonExpression(['"', task_generator_node.substitution, '/task_reset"']),
                # HuNav publishes a shared human state stream at the task-generator
                # namespace, not below each robot namespace.
                'human_states_topic': PythonExpression(['"', task_generator_node.substitution, '/human_states"']),
                'start_topic': 'episode_start_pose',
                'goal_topic': 'episode_goal_pose_metadata',
            }
        ],
        condition=launch.conditions.IfCondition(PythonExpression(['bool("', record_data_dir.substitution, '")'])),
    )

    # Launch the rosnav_rl action server when using DRL local planner
    rosnav_rl_action_server = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            launch.substitutions.PathJoinSubstitution([
                FindPackageShare('rosnav_rl'),
                'launch',
                'action_server.launch.py',
            ])
        ),
        launch_arguments={
            'agent_name': launch.substitutions.LaunchConfiguration('agent_name'),
            'namespace': namespace.substitution,
            'agents_dir': agents_dir.substitution,
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", local_planner.substitution, "' == 'rosnav_rl' and '", train_mode.substitution, "' == 'false'"])
        ),
    )

    ld = launch.LaunchDescription([
        *ld_items,
        launch.actions.DeclareLaunchArgument(
            name='agent_name',
            default_value='',
            description='DRL agent name to be deployed'
        ),
        launch.actions.DeclareLaunchArgument(
            name='complexity',
            default_value='1'
        ),
        PushRosNamespace(namespace=namespace.substitution),
        # robot_localization_node,
        nav2_launch,
        # state_pub_launch,
        rosnav_rl_action_server,
        data_recorder,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
