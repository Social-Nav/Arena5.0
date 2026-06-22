import os

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
    internnav_timing_mode = LaunchArgument(
        'internnav_timing_mode',
        default_value=os.environ.get('ARENA_EVAL_INTERNNAV_TIMING_MODE', 'wall'),
    )
    dual_vln_timing_mode = LaunchArgument('dual_vln_timing_mode', default_value=internnav_timing_mode.substitution)
    internnav_model_latency_sec = LaunchArgument(
        'internnav_model_latency_sec',
        default_value=os.environ.get('ARENA_EVAL_INTERNNAV_MODEL_LATENCY_SEC', '0.3'),
    )
    dual_vln_model_latency_sec = LaunchArgument(
        'dual_vln_model_latency_sec', default_value=internnav_model_latency_sec.substitution
    )
    internnav_latency_policy = LaunchArgument(
        'internnav_latency_policy',
        default_value=os.environ.get('ARENA_EVAL_INTERNNAV_LATENCY_POLICY', 'fixed'),
    )
    dual_vln_latency_policy = LaunchArgument('dual_vln_latency_policy', default_value=internnav_latency_policy.substitution)
    internnav_planning_period_sec = LaunchArgument('internnav_planning_period_sec', default_value='0.3')
    dual_vln_planning_period_sec = LaunchArgument(
        'dual_vln_planning_period_sec', default_value=internnav_planning_period_sec.substitution
    )
    internnav_raw_cmd_vel_topic = LaunchArgument(
        'internnav_raw_cmd_vel_topic',
        default_value=os.environ.get('ARENA_EVAL_INTERNNAV_RAW_CMD_VEL_TOPIC', 'internnav/raw_cmd_vel'),
    )
    dual_vln_raw_cmd_vel_topic = LaunchArgument(
        'dual_vln_raw_cmd_vel_topic', default_value=internnav_raw_cmd_vel_topic.substitution
    )

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
    timing_manager = launch_ros.actions.Node(
        package='arena_bringup',
        executable='internnav_timing_manager',
        name='internnav_timing_manager',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'timing_mode': dual_vln_timing_mode.substitution,
                'latency_policy': dual_vln_latency_policy.substitution,
                'model_latency_sec': dual_vln_model_latency_sec.param_value(float),
                'planning_period_sec': dual_vln_planning_period_sec.param_value(float),
                'input_cmd_vel_topic': dual_vln_raw_cmd_vel_topic.substitution,
                'output_cmd_vel_topic': 'cmd_vel',
                'status_topic': 'internnav/status',
                'task_reset_topic': PythonExpression(['"', task_generator_node.substitution, '/task_reset"']),
                'eval_ready_topic': PythonExpression(['"', task_generator_node.substitution, '/eval_ready"']),
                'record_data_dir': record_data_dir.substitution,
            }
        ],
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
        timing_manager,
    ])


if __name__ == '__main__':
    generate_launch_description()
