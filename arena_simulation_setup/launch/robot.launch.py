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


def _requires_missing_local_internnav_server(
    *,
    local_planner: str,
    train_mode: str,
    internnav_external_server: str,
    dual_vln_external_server: str,
    internnav_direct_cmd_vel: str,
    dual_vln_direct_cmd_vel: str,
    env_external_server: str,
) -> bool:
    enabled = {'1', 'true', 'yes', 'on'}
    external = any(
        str(value).strip().lower() in enabled
        for value in (
            internnav_external_server,
            dual_vln_external_server,
            internnav_direct_cmd_vel,
            dual_vln_direct_cmd_vel,
            env_external_server,
        )
    )
    return (
        str(local_planner).strip() == 'dual_vln'
        and str(train_mode).strip().lower() != 'true'
        and not external
    )


def generate_launch_description():

    ss_path = FindPackageShare('arena_simulation_setup')

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    def declare_legacy_alias(name: str, target: LaunchArgument) -> LaunchArgument:
        return LaunchArgument(
            name=name,
            default_value=target.substitution,
        )

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
    internnav_mode = LaunchArgument('internnav_mode', default_value='heuristic')
    dual_vln_mode = declare_legacy_alias('dual_vln_mode', internnav_mode)
    internnav_model_path = LaunchArgument(
        'internnav_model_path',
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_INTERNNAV_MODEL_PATH',
            default_value=launch.substitutions.EnvironmentVariable(
                'INTERNNAV_MODEL_PATH',
                default_value=launch.substitutions.EnvironmentVariable('ARENA_VLN_MODEL_PATH', default_value=''),
            ),
        ),
    )
    dual_vln_model_path = declare_legacy_alias('dual_vln_model_path', internnav_model_path)
    internnav_device = LaunchArgument('internnav_device', default_value='cpu')
    dual_vln_device = declare_legacy_alias('dual_vln_device', internnav_device)
    internnav_inference_rate_hz = LaunchArgument('internnav_inference_rate_hz', default_value='3.3333333333')
    dual_vln_inference_rate_hz = declare_legacy_alias('dual_vln_inference_rate_hz', internnav_inference_rate_hz)
    internnav_inference_timeout_sec = LaunchArgument('internnav_inference_timeout_sec', default_value='0.2')
    dual_vln_inference_timeout_sec = declare_legacy_alias('dual_vln_inference_timeout_sec', internnav_inference_timeout_sec)
    internnav_rgb_topic = LaunchArgument('internnav_rgb_topic', default_value='')
    dual_vln_rgb_topic = declare_legacy_alias('dual_vln_rgb_topic', internnav_rgb_topic)
    internnav_depth_topic = LaunchArgument('internnav_depth_topic', default_value='')
    dual_vln_depth_topic = declare_legacy_alias('dual_vln_depth_topic', internnav_depth_topic)
    internnav_camera_info_topic = LaunchArgument('internnav_camera_info_topic', default_value='')
    dual_vln_camera_info_topic = declare_legacy_alias('dual_vln_camera_info_topic', internnav_camera_info_topic)
    internnav_python_executable = LaunchArgument(
        'internnav_python_executable',
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_VLN_MODEL_PYTHON',
            default_value=launch.substitutions.EnvironmentVariable(
                'ARENA_INTERNNAV_PYTHON',
                default_value=launch.substitutions.EnvironmentVariable('ARENA_PYTHON', default_value=''),
            ),
        ),
    )
    dual_vln_python_executable = declare_legacy_alias('dual_vln_python_executable', internnav_python_executable)
    internnav_adapter_target = LaunchArgument('internnav_adapter_target', default_value='')
    dual_vln_adapter_target = declare_legacy_alias('dual_vln_adapter_target', internnav_adapter_target)
    internnav_http_url = LaunchArgument(
        'internnav_http_url',
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_EVAL_INTERNNAV_HTTP_URL',
            default_value=launch.substitutions.EnvironmentVariable('ARENA_INTERNNAV_HTTP_URL', default_value=''),
        ),
    )
    dual_vln_http_url = declare_legacy_alias('dual_vln_http_url', internnav_http_url)
    internnav_http_timeout_sec = LaunchArgument(
        'internnav_http_timeout_sec',
        # Keep launch-time float conversion deterministic; adapter code reads
        # ARENA_*_HTTP_TIMEOUT_SEC directly and can recover from invalid values.
        default_value='0.0',
    )
    dual_vln_http_timeout_sec = declare_legacy_alias('dual_vln_http_timeout_sec', internnav_http_timeout_sec)
    internnav_require_real_backend = LaunchArgument('internnav_require_real_backend', default_value='false')
    dual_vln_require_real_backend = declare_legacy_alias('dual_vln_require_real_backend', internnav_require_real_backend)
    internnav_strict_device = LaunchArgument('internnav_strict_device', default_value='false')
    dual_vln_strict_device = declare_legacy_alias('dual_vln_strict_device', internnav_strict_device)
    internnav_look_down = LaunchArgument('internnav_look_down', default_value='false')
    dual_vln_look_down = declare_legacy_alias('dual_vln_look_down', internnav_look_down)
    internnav_model_output_policy = LaunchArgument('internnav_model_output_policy', default_value='trajectory')
    dual_vln_model_output_policy = declare_legacy_alias('dual_vln_model_output_policy', internnav_model_output_policy)
    internnav_enable_visualization = LaunchArgument('internnav_enable_visualization', default_value='false')
    dual_vln_enable_visualization = declare_legacy_alias('dual_vln_enable_visualization', internnav_enable_visualization)
    internnav_visualization_topic = LaunchArgument('internnav_visualization_topic', default_value='internnav/debug_image')
    dual_vln_visualization_topic = declare_legacy_alias('dual_vln_visualization_topic', internnav_visualization_topic)
    internnav_action_visualization_topic = LaunchArgument('internnav_action_visualization_topic', default_value='internnav/action_image')
    dual_vln_action_visualization_topic = declare_legacy_alias('dual_vln_action_visualization_topic', internnav_action_visualization_topic)
    internnav_visualization_rate_hz = LaunchArgument('internnav_visualization_rate_hz', default_value='5.0')
    dual_vln_visualization_rate_hz = declare_legacy_alias('dual_vln_visualization_rate_hz', internnav_visualization_rate_hz)
    internnav_model_output_topic = LaunchArgument('internnav_model_output_topic', default_value='internnav/model_output')
    dual_vln_model_output_topic = declare_legacy_alias('dual_vln_model_output_topic', internnav_model_output_topic)
    internnav_external_server = LaunchArgument('internnav_external_server', default_value='false')
    dual_vln_external_server = declare_legacy_alias('dual_vln_external_server', internnav_external_server)
    internnav_direct_cmd_vel = LaunchArgument('internnav_direct_cmd_vel', default_value='false')
    dual_vln_direct_cmd_vel = declare_legacy_alias('dual_vln_direct_cmd_vel', internnav_direct_cmd_vel)
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
            **internnav_direct_cmd_vel.dict,
            **dual_vln_direct_cmd_vel.dict,
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

    def require_external_internnav(context):
        if _requires_missing_local_internnav_server(
            local_planner=local_planner.substitution.perform(context),
            train_mode=train_mode.substitution.perform(context),
            internnav_external_server=internnav_external_server.substitution.perform(context),
            dual_vln_external_server=dual_vln_external_server.substitution.perform(context),
            internnav_direct_cmd_vel=internnav_direct_cmd_vel.substitution.perform(context),
            dual_vln_direct_cmd_vel=dual_vln_direct_cmd_vel.substitution.perform(context),
            env_external_server=os.environ.get('ARENA_INTERNNAV_EXTERNAL_SERVER', ''),
        ):
            raise RuntimeError(
                'dual_vln requires the dedicated internnav-1 service; pass '
                'internnav_external_server:=true or use the case-specific '
                'internnav_async_eval.launch.py direct-control path'
            )
        return []

    external_internnav_guard = launch.actions.OpaqueFunction(
        function=require_external_internnav
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
        external_internnav_guard,
        # robot_localization_node,
        nav2_launch,
        # state_pub_launch,
        rosnav_rl_action_server,
        data_recorder,
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
