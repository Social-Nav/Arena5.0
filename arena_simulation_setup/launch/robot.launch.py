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
from launch.conditions import UnlessCondition


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
    dual_vln_mode = LaunchArgument('dual_vln_mode', default_value='heuristic')
    dual_vln_model_path = LaunchArgument(
        'dual_vln_model_path',
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_INTERNNAV_MODEL_PATH',
            default_value=launch.substitutions.EnvironmentVariable(
                'INTERNNAV_MODEL_PATH',
                default_value=launch.substitutions.EnvironmentVariable('ARENA_VLN_MODEL_PATH', default_value=''),
            ),
        ),
    )
    dual_vln_device = LaunchArgument('dual_vln_device', default_value='cpu')
    dual_vln_inference_rate_hz = LaunchArgument('dual_vln_inference_rate_hz', default_value='10.0')
    dual_vln_inference_timeout_sec = LaunchArgument('dual_vln_inference_timeout_sec', default_value='0.2')
    dual_vln_rgb_topic = LaunchArgument('dual_vln_rgb_topic', default_value='')
    dual_vln_depth_topic = LaunchArgument('dual_vln_depth_topic', default_value='')
    dual_vln_camera_info_topic = LaunchArgument('dual_vln_camera_info_topic', default_value='')
    dual_vln_python_executable = LaunchArgument(
        'dual_vln_python_executable',
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_VLN_MODEL_PYTHON',
            default_value=launch.substitutions.EnvironmentVariable(
                'ARENA_INTERNNAV_PYTHON',
                default_value=launch.substitutions.EnvironmentVariable('ARENA_PYTHON', default_value=''),
            ),
        ),
    )
    dual_vln_adapter_target = LaunchArgument('dual_vln_adapter_target', default_value='')
    dual_vln_require_real_backend = LaunchArgument('dual_vln_require_real_backend', default_value='false')
    dual_vln_strict_device = LaunchArgument('dual_vln_strict_device', default_value='false')
    dual_vln_look_down = LaunchArgument('dual_vln_look_down', default_value='false')
    dual_vln_enable_visualization = LaunchArgument('dual_vln_enable_visualization', default_value='false')
    dual_vln_visualization_topic = LaunchArgument('dual_vln_visualization_topic', default_value='dual_vln/debug_image')
    dual_vln_visualization_rate_hz = LaunchArgument('dual_vln_visualization_rate_hz', default_value='5.0')
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
                'goal_topic': 'episode_goal_pose',
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

    # Launch the dual_vln server when using dual_vln local planner
    dual_vln_server_parameters = [
        {
            'namespace': namespace.substitution,
            'mode': dual_vln_mode.substitution,
            'model_path': dual_vln_model_path.substitution,
            'device': dual_vln_device.substitution,
            'goal_topic': 'episode_goal_pose',
            'instruction_topic': PythonExpression(['"', task_generator_node.substitution, '/vln_instruction"']),
            'rgb_topic': dual_vln_rgb_topic.substitution,
            'depth_topic': dual_vln_depth_topic.substitution,
            'camera_info_topic': dual_vln_camera_info_topic.substitution,
            'adapter_target': dual_vln_adapter_target.substitution,
            'require_real_backend': launch_ros.parameter_descriptions.ParameterValue(
                dual_vln_require_real_backend.substitution, value_type=bool
            ),
            'strict_device': launch_ros.parameter_descriptions.ParameterValue(
                dual_vln_strict_device.substitution, value_type=bool
            ),
            'look_down': launch_ros.parameter_descriptions.ParameterValue(
                dual_vln_look_down.substitution, value_type=bool
            ),
            'enable_visualization': launch_ros.parameter_descriptions.ParameterValue(
                dual_vln_enable_visualization.substitution, value_type=bool
            ),
            'visualization_topic': dual_vln_visualization_topic.substitution,
            'visualization_rate_hz': launch_ros.parameter_descriptions.ParameterValue(
                dual_vln_visualization_rate_hz.substitution, value_type=float
            ),
            'inference_rate_hz': launch_ros.parameter_descriptions.ParameterValue(
                dual_vln_inference_rate_hz.substitution, value_type=float
            ),
            'inference_timeout_sec': launch_ros.parameter_descriptions.ParameterValue(
                dual_vln_inference_timeout_sec.substitution, value_type=float
            ),
        }
    ]
    dual_vln_enabled = IfCondition(
        PythonExpression(["'", local_planner.substitution, "' == 'dual_vln' and '", train_mode.substitution, "' == 'false'"])
    )
    dual_vln_server = launch_ros.actions.Node(
        package='arena_vln_models',
        executable='dual_vln_server',
        name='dual_vln_server',
        output='screen',
        parameters=dual_vln_server_parameters,
        additional_env={
            'ARENA_PYTHON': dual_vln_python_executable.substitution,
        },
        condition=dual_vln_enabled,
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
        dual_vln_server,
        data_recorder,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
