import launch_ros
from ament_index_python.packages import get_package_share_directory
from arena_bringup.future import PythonExpression
from arena_bringup.substitutions import LaunchArgument
from launch_ros.actions import PushRosNamespace

import launch
import launch.actions
import launch.launch_description_sources
import launch.substitutions


def generate_launch_description():

    ss_path = get_package_share_directory('arena_simulation_setup')

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
        arguments=[
            ['--dir', ' ', record_data_dir.substitution],
        ],
        remappings=[
            # HuNav Agent Manager is launched in the task generator namespace, not the robot namespace.
            # Without this, the recorder (running in /<namespace>) will subscribe to /<namespace>/human_states
            # which stays empty.
            (
                'human_states',
                launch.substitutions.PathJoinSubstitution([
                    task_generator_node.substitution,
                    'human_states',
                ]),
            ),
        ],
        condition=launch.conditions.IfCondition(PythonExpression(['bool("', record_data_dir.substitution, '")'])),
    )

    ld = launch.LaunchDescription([
        *ld_items,
        launch.actions.DeclareLaunchArgument(
            name='train_mode',
            default_value='false',
            description='If false, start the Rosnav Deployment launch.actions.Nodes'
        ),
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
        data_recorder,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
