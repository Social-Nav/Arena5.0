import launch
import launch.actions
import launch.substitutions
import launch_ros
from arena_bringup.future import PythonExpression
from arena_bringup.substitutions import LaunchArgument
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    """Robot-level launch for InternNav official async/direct-cmd_vel eval.

    This run case intentionally does not include Nav2, rosnav_rl, or the local
    Arena InternNav wrapper.  The upstream InternNav realworld ROS2 client runs
    in the internnav container and publishes cmd_vel directly.
    """

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    use_sim_time = LaunchArgument('use_sim_time', default_value='true')
    task_generator_node = LaunchArgument('task_generator_node')
    namespace = LaunchArgument('namespace')
    robot = LaunchArgument('robot')
    local_planner = LaunchArgument('local_planner', default_value='dual_vln')
    inter_planner = LaunchArgument('inter_planner', default_value='navigate_to_pose')
    record_data_dir = LaunchArgument('record_data_dir', default_value='')
    internnav_direct_cmd_vel = LaunchArgument('internnav_direct_cmd_vel', default_value='true')
    dual_vln_direct_cmd_vel = LaunchArgument('dual_vln_direct_cmd_vel', default_value=internnav_direct_cmd_vel.substitution)

    data_recorder = launch_ros.actions.Node(
        package='arena_evaluation',
        executable='record',
        name=PythonExpression(['"data_recorder" + "', namespace.substitution, '".replace("/","_")']),
        arguments=['--dir', record_data_dir.substitution],
        parameters=[
            {
                'use_sim_time': use_sim_time.param_value(bool),
                'local_planner': local_planner.substitution,
                'inter_planner': inter_planner.substitution,
                'agent_name': launch.substitutions.LaunchConfiguration('agent_name'),
                'scenario_reset_topic': PythonExpression(['"', task_generator_node.substitution, '/task_reset"']),
                'human_states_topic': PythonExpression(['"', task_generator_node.substitution, '/human_states"']),
                'start_topic': 'episode_start_pose',
                'goal_topic': 'episode_goal_pose_metadata',
            }
        ],
        condition=launch.conditions.IfCondition(PythonExpression(['bool("', record_data_dir.substitution, '")'])),
    )

    return launch.LaunchDescription([
        *ld_items,
        launch.actions.DeclareLaunchArgument(
            name='agent_name',
            default_value='',
            description='DRL agent name to be deployed',
        ),
        launch.actions.LogInfo(msg=[
            'internnav_async_eval.launch.py: skipping Nav2 and using external direct cmd_vel InternNav client for ',
            robot.substitution,
        ]),
        PushRosNamespace(namespace=namespace.substitution),
        data_recorder,
    ])


if __name__ == '__main__':
    generate_launch_description()
