import os
import typing

import launch.event_handlers
import launch.launch_description_sources
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from arena_bringup.future import PythonExpression
from arena_bringup.substitutions import CurrentNamespaceSubstitution, LaunchArgument

import launch


def generate_launch_description():

    bringup_dir = get_package_share_directory("arena_bringup")

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    def declare_legacy_alias(name: str, target: LaunchArgument) -> LaunchArgument:
        return LaunchArgument(
            name=name,
            default_value=target.substitution,
        )

    namespace = LaunchArgument(name="namespace", default_value="task_generator_node")

    sim = LaunchArgument(name="sim", description="[dummy, gazebo, isaac]")
    human = LaunchArgument(name="human", description="[dummy]")
    robot = LaunchArgument(
        name="robot",
        description="robot type [burger, jackal, ridgeback, agvota, rto, ...]",
    )

    tm_robots = LaunchArgument(
        name="tm_robots",
    )
    tm_obstacles = LaunchArgument(
        name="tm_obstacles",
    )
    scenario_file = LaunchArgument(
        name="scenario_file",
        default_value=launch.substitutions.EnvironmentVariable('ARENA_SCENARIO_FILE', default_value='default'),
    )
    tm_modules = LaunchArgument(
        name="tm_modules",
    )
    world = LaunchArgument(
        name="world",
    )

    local_planner = LaunchArgument(
        name="local_planner",
    )
    inter_planner = LaunchArgument(
        name="inter_planner",
    )
    global_planner = LaunchArgument(
        name="global_planner",
    )
    record_data_dir = LaunchArgument(
        name="record_data_dir",
        default_value="",
    )
    episodes = LaunchArgument(
        name="episodes",
        default_value="2",
    )
    auto_reset = LaunchArgument(
        name="auto_reset",
        default_value="true",
    )
    timeout = LaunchArgument(
        name="timeout",
        default_value="120.0",
    )
    timeout_wall_factor = LaunchArgument(
        name="timeout_wall_factor",
        default_value="5.0",
    )
    timeout_wall_sec = LaunchArgument(
        name="timeout_wall_sec",
        default_value="0.0",
    )
    vln_instruction = LaunchArgument(
        name="vln_instruction",
        default_value="navigate",
    )
    vln_instruction_file = LaunchArgument(
        name="vln_instruction_file",
        default_value="",
    )
    internnav_mode = LaunchArgument(
        name="internnav_mode",
        default_value="heuristic",
    )
    dual_vln_mode = declare_legacy_alias("dual_vln_mode", internnav_mode)
    internnav_model_path = LaunchArgument(
        name="internnav_model_path",
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_INTERNNAV_MODEL_PATH',
            default_value=launch.substitutions.EnvironmentVariable(
                'INTERNNAV_MODEL_PATH',
                default_value=launch.substitutions.EnvironmentVariable('ARENA_VLN_MODEL_PATH', default_value=''),
            ),
        ),
    )
    dual_vln_model_path = declare_legacy_alias("dual_vln_model_path", internnav_model_path)
    internnav_device = LaunchArgument(
        name="internnav_device",
        default_value="cpu",
    )
    dual_vln_device = declare_legacy_alias("dual_vln_device", internnav_device)
    internnav_inference_rate_hz = LaunchArgument(
        name="internnav_inference_rate_hz",
        default_value="3.3333333333",
    )
    dual_vln_inference_rate_hz = declare_legacy_alias("dual_vln_inference_rate_hz", internnav_inference_rate_hz)
    internnav_inference_timeout_sec = LaunchArgument(
        name="internnav_inference_timeout_sec",
        default_value="0.2",
    )
    dual_vln_inference_timeout_sec = declare_legacy_alias("dual_vln_inference_timeout_sec", internnav_inference_timeout_sec)
    internnav_rgb_topic = LaunchArgument(
        name="internnav_rgb_topic",
        default_value="",
    )
    dual_vln_rgb_topic = declare_legacy_alias("dual_vln_rgb_topic", internnav_rgb_topic)
    internnav_depth_topic = LaunchArgument(
        name="internnav_depth_topic",
        default_value="",
    )
    dual_vln_depth_topic = declare_legacy_alias("dual_vln_depth_topic", internnav_depth_topic)
    internnav_camera_info_topic = LaunchArgument(
        name="internnav_camera_info_topic",
        default_value="",
    )
    dual_vln_camera_info_topic = declare_legacy_alias("dual_vln_camera_info_topic", internnav_camera_info_topic)
    internnav_python_executable = LaunchArgument(
        name="internnav_python_executable",
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_VLN_MODEL_PYTHON',
            default_value=launch.substitutions.EnvironmentVariable(
                'ARENA_INTERNNAV_PYTHON',
                default_value=launch.substitutions.EnvironmentVariable('ARENA_PYTHON', default_value=''),
            ),
        ),
    )
    dual_vln_python_executable = declare_legacy_alias("dual_vln_python_executable", internnav_python_executable)
    internnav_adapter_target = LaunchArgument(
        name="internnav_adapter_target",
        default_value="",
    )
    dual_vln_adapter_target = declare_legacy_alias("dual_vln_adapter_target", internnav_adapter_target)
    internnav_http_url = LaunchArgument(
        name="internnav_http_url",
        default_value=launch.substitutions.EnvironmentVariable(
            "ARENA_EVAL_INTERNNAV_HTTP_URL",
            default_value=launch.substitutions.EnvironmentVariable("ARENA_INTERNNAV_HTTP_URL", default_value=""),
        ),
    )
    dual_vln_http_url = declare_legacy_alias("dual_vln_http_url", internnav_http_url)
    internnav_http_timeout_sec = LaunchArgument(
        name="internnav_http_timeout_sec",
        # Avoid launch-time float conversion of arbitrary env strings; runtime
        # nodes read ARENA_*_HTTP_TIMEOUT_SEC and apply their own fallback.
        default_value="0.0",
    )
    dual_vln_http_timeout_sec = declare_legacy_alias("dual_vln_http_timeout_sec", internnav_http_timeout_sec)
    internnav_require_real_backend = LaunchArgument(
        name="internnav_require_real_backend",
        default_value="false",
    )
    dual_vln_require_real_backend = declare_legacy_alias("dual_vln_require_real_backend", internnav_require_real_backend)
    internnav_strict_device = LaunchArgument(
        name="internnav_strict_device",
        default_value="false",
    )
    dual_vln_strict_device = declare_legacy_alias("dual_vln_strict_device", internnav_strict_device)
    internnav_look_down = LaunchArgument(
        name="internnav_look_down",
        default_value="false",
    )
    dual_vln_look_down = declare_legacy_alias("dual_vln_look_down", internnav_look_down)
    internnav_model_output_policy = LaunchArgument(
        name="internnav_model_output_policy",
        default_value="trajectory",
    )
    dual_vln_model_output_policy = declare_legacy_alias(
        "dual_vln_model_output_policy", internnav_model_output_policy
    )
    internnav_enable_visualization = LaunchArgument(
        name="internnav_enable_visualization",
        default_value="false",
    )
    dual_vln_enable_visualization = declare_legacy_alias("dual_vln_enable_visualization", internnav_enable_visualization)
    internnav_visualization_topic = LaunchArgument(
        name="internnav_visualization_topic",
        default_value="internnav/debug_image",
    )
    dual_vln_visualization_topic = declare_legacy_alias("dual_vln_visualization_topic", internnav_visualization_topic)
    internnav_action_visualization_topic = LaunchArgument(
        name="internnav_action_visualization_topic",
        default_value="internnav/action_image",
    )
    dual_vln_action_visualization_topic = declare_legacy_alias(
        "dual_vln_action_visualization_topic", internnav_action_visualization_topic
    )
    internnav_visualization_rate_hz = LaunchArgument(
        name="internnav_visualization_rate_hz",
        default_value="5.0",
    )
    dual_vln_visualization_rate_hz = declare_legacy_alias("dual_vln_visualization_rate_hz", internnav_visualization_rate_hz)
    internnav_model_output_topic = LaunchArgument(
        name="internnav_model_output_topic",
        default_value="internnav/model_output",
    )
    dual_vln_model_output_topic = declare_legacy_alias("dual_vln_model_output_topic", internnav_model_output_topic)
    internnav_external_server = LaunchArgument(
        name="internnav_external_server",
        default_value="false",
    )
    dual_vln_external_server = declare_legacy_alias("dual_vln_external_server", internnav_external_server)
    internnav_direct_cmd_vel = LaunchArgument(
        name="internnav_direct_cmd_vel",
        default_value="false",
    )
    dual_vln_direct_cmd_vel = declare_legacy_alias("dual_vln_direct_cmd_vel", internnav_direct_cmd_vel)
    internnav_command_service = LaunchArgument(
        name="internnav_command_service",
        default_value="",
    )
    dual_vln_command_service = declare_legacy_alias("dual_vln_command_service", internnav_command_service)
    internnav_status_topic = LaunchArgument(
        name="internnav_status_topic",
        default_value="",
    )
    dual_vln_status_topic = declare_legacy_alias("dual_vln_status_topic", internnav_status_topic)
    enable_collision_monitor = LaunchArgument(
        name="enable_collision_monitor",
        default_value="true",
    )
    robot_launch_file = LaunchArgument(
        name="robot_launch_file",
        default_value="robot.launch.py",
    )

    parameter_file = LaunchArgument(name="parameter_file")

    headless = LaunchArgument(
        name="headless",
        default_value="False",
    )
    reference = LaunchArgument(
        name="reference",
        default_value="[0, 0]",
    )
    prefix = LaunchArgument(
        name="prefix",
        default_value="",
    )
    debug = LaunchArgument(
        name="debug",
        default_value="False",
    )
    train_mode = LaunchArgument(
        name="train_mode",
        default_value="false",
    )
    require_human_states_ready = LaunchArgument(
        name="require_human_states_ready",
        default_value="false",
    )
    human_states_ready_timeout_sec = LaunchArgument(
        name="human_states_ready_timeout_sec",
        default_value="10.0",
    )
    episode_start_delay_sec = LaunchArgument(
        name="episode_start_delay_sec",
        default_value="0.0",
    )

    map_server_node = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, "launch/utils/map_server.launch.py")
        )
    )

    # Hunavsim Pedestrians in rviz
    pedestrian_marker_node = launch_ros.actions.Node(
        package="rviz_utils",
        executable="pedestrian_marker_publisher",
        name="pedestrian_marker_publisher",
        parameters=[
            {"use_sim_time": True},
            {"body_height": 1.6},
            {"body_radius": 0.25},
            {"head_radius": 0.15},
            {"arrow_length": 0.6},
            {"show_labels": True},
            {"show_velocity_arrows": True},
            {"show_orientation_arrows": True},
            {"namespace": namespace.substitution},
        ],
        output="screen",
        condition=launch.conditions.IfCondition(
            PythonExpression(['"', human.substitution, '" == "hunav"'])
        ),
    )
    # Start the rviz config generator which launches also rviz2 with desired config file
    rviz_node = launch_ros.actions.Node(
        package="rviz_utils",
        executable="rviz_config",
        name="rviz_config_generator",
        arguments=[
            CurrentNamespaceSubstitution(),
        ],
        parameters=[
            {
                "use_sim_time": True,
                "origin": reference.param_value(typing.List[float]),
            }
        ],
        output="screen",
        condition=launch.conditions.UnlessCondition(headless.substitution),
    )

    task_generator_node = launch_ros.actions.Node(
        package="task_generator",
        executable="task_generator_node",
        namespace=PythonExpression(
            ['os.path.dirname("', namespace.substitution, '")'], ["os"]
        ),
        name=PythonExpression(
            ['os.path.basename("', namespace.substitution, '")'], ["os"]
        ),
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                **sim.str_param,
                **human.str_param,
                **robot.str_param,
                **tm_robots.str_param,
                **tm_obstacles.str_param,
                **tm_modules.str_param,
                **world.str_param,
                **inter_planner.str_param,
                **local_planner.str_param,
                **global_planner.str_param,
                **record_data_dir.str_param,
                **episodes.param(int),
                **auto_reset.param(bool),
                **timeout.param(int),
                **timeout_wall_factor.param(float),
                **timeout_wall_sec.param(float),
                **vln_instruction.str_param,
                **vln_instruction_file.str_param,
                'internnav_mode': dual_vln_mode.param_value(str),
                'internnav_model_path': dual_vln_model_path.param_value(str),
                'internnav_device': dual_vln_device.param_value(str),
                'internnav_inference_rate_hz': dual_vln_inference_rate_hz.param_value(float),
                'internnav_inference_timeout_sec': dual_vln_inference_timeout_sec.param_value(float),
                'internnav_rgb_topic': dual_vln_rgb_topic.param_value(str),
                'internnav_depth_topic': dual_vln_depth_topic.param_value(str),
                'internnav_camera_info_topic': dual_vln_camera_info_topic.param_value(str),
                'internnav_python_executable': dual_vln_python_executable.param_value(str),
                'internnav_adapter_target': dual_vln_adapter_target.param_value(str),
                'internnav_http_url': dual_vln_http_url.param_value(str),
                'internnav_http_timeout_sec': dual_vln_http_timeout_sec.param_value(float),
                'internnav_require_real_backend': dual_vln_require_real_backend.param_value(bool),
                'internnav_strict_device': dual_vln_strict_device.param_value(bool),
                'internnav_look_down': dual_vln_look_down.param_value(bool),
                'internnav_model_output_policy': dual_vln_model_output_policy.param_value(str),
                'internnav_enable_visualization': dual_vln_enable_visualization.param_value(bool),
                'internnav_visualization_topic': dual_vln_visualization_topic.param_value(str),
                'internnav_action_visualization_topic': dual_vln_action_visualization_topic.param_value(str),
                'internnav_visualization_rate_hz': dual_vln_visualization_rate_hz.param_value(float),
                'internnav_model_output_topic': dual_vln_model_output_topic.param_value(str),
                'internnav_external_server': internnav_external_server.param_value(bool),
                'internnav_direct_cmd_vel': internnav_direct_cmd_vel.param_value(bool),
                'internnav_command_service': dual_vln_command_service.param_value(str),
                'internnav_status_topic': dual_vln_status_topic.param_value(str),
                **dual_vln_mode.str_param,
                **dual_vln_model_path.str_param,
                **dual_vln_device.str_param,
                **dual_vln_inference_rate_hz.param(float),
                **dual_vln_inference_timeout_sec.param(float),
                **dual_vln_rgb_topic.str_param,
                **dual_vln_depth_topic.str_param,
                **dual_vln_camera_info_topic.str_param,
                **dual_vln_python_executable.str_param,
                **dual_vln_adapter_target.str_param,
                **dual_vln_http_url.str_param,
                **dual_vln_http_timeout_sec.param(float),
                **dual_vln_require_real_backend.param(bool),
                **dual_vln_strict_device.param(bool),
                **dual_vln_look_down.param(bool),
                **dual_vln_model_output_policy.str_param,
                **dual_vln_enable_visualization.param(bool),
                **dual_vln_visualization_topic.str_param,
                **dual_vln_action_visualization_topic.str_param,
                **dual_vln_visualization_rate_hz.param(float),
                **dual_vln_model_output_topic.str_param,
                **dual_vln_command_service.str_param,
                **dual_vln_status_topic.str_param,
                **internnav_external_server.param(bool),
                **dual_vln_external_server.param(bool),
                **internnav_direct_cmd_vel.param(bool),
                **dual_vln_direct_cmd_vel.param(bool),
                **enable_collision_monitor.param(bool),
                **robot_launch_file.str_param,
                **reference.param(typing.List[float]),
                **prefix.str_param,
                **debug.param(bool),
                **train_mode.param(bool),
                **require_human_states_ready.param(bool),
                **human_states_ready_timeout_sec.param(float),
                **episode_start_delay_sec.param(float),
            },
            {"use_sim_time": False},
            parameter_file.substitution,
            {
                # Keep CLI/launch-time InternNav selection authoritative even
                # when the task_generator YAML contains older defaults.  This
                # block intentionally comes after ``parameter_file`` because ROS
                # 2 parameter sources are applied in order and the YAML carries
                # legacy empty dual_vln_* defaults that otherwise erase the
                # runtime camera/model arguments before robot_manager launches
                # the InternNav wrapper.
                'task': {
                    'scenario': {
                        'file': scenario_file.param_value(str),
                    },
                },
                'internnav_mode': dual_vln_mode.param_value(str),
                'internnav_model_path': dual_vln_model_path.param_value(str),
                'internnav_device': dual_vln_device.param_value(str),
                'internnav_inference_rate_hz': dual_vln_inference_rate_hz.param_value(float),
                'internnav_inference_timeout_sec': dual_vln_inference_timeout_sec.param_value(float),
                'internnav_rgb_topic': dual_vln_rgb_topic.param_value(str),
                'internnav_depth_topic': dual_vln_depth_topic.param_value(str),
                'internnav_camera_info_topic': dual_vln_camera_info_topic.param_value(str),
                'internnav_python_executable': dual_vln_python_executable.param_value(str),
                'internnav_adapter_target': dual_vln_adapter_target.param_value(str),
                'internnav_http_url': dual_vln_http_url.param_value(str),
                'internnav_http_timeout_sec': dual_vln_http_timeout_sec.param_value(float),
                'internnav_require_real_backend': dual_vln_require_real_backend.param_value(bool),
                'internnav_strict_device': dual_vln_strict_device.param_value(bool),
                'internnav_look_down': dual_vln_look_down.param_value(bool),
                'internnav_model_output_policy': dual_vln_model_output_policy.param_value(str),
                'internnav_enable_visualization': dual_vln_enable_visualization.param_value(bool),
                'internnav_visualization_topic': dual_vln_visualization_topic.param_value(str),
                'internnav_action_visualization_topic': dual_vln_action_visualization_topic.param_value(str),
                'internnav_visualization_rate_hz': dual_vln_visualization_rate_hz.param_value(float),
                'internnav_model_output_topic': dual_vln_model_output_topic.param_value(str),
                'internnav_external_server': internnav_external_server.param_value(bool),
                'internnav_direct_cmd_vel': internnav_direct_cmd_vel.param_value(bool),
                'internnav_command_service': dual_vln_command_service.param_value(str),
                'internnav_status_topic': dual_vln_status_topic.param_value(str),
                **dual_vln_mode.str_param,
                **dual_vln_model_path.str_param,
                **dual_vln_device.str_param,
                **dual_vln_inference_rate_hz.param(float),
                **dual_vln_inference_timeout_sec.param(float),
                **dual_vln_rgb_topic.str_param,
                **dual_vln_depth_topic.str_param,
                **dual_vln_camera_info_topic.str_param,
                **dual_vln_python_executable.str_param,
                **dual_vln_adapter_target.str_param,
                **dual_vln_http_url.str_param,
                **dual_vln_http_timeout_sec.param(float),
                **dual_vln_require_real_backend.param(bool),
                **dual_vln_strict_device.param(bool),
                **dual_vln_look_down.param(bool),
                **dual_vln_model_output_policy.str_param,
                **dual_vln_enable_visualization.param(bool),
                **dual_vln_visualization_topic.str_param,
                **dual_vln_action_visualization_topic.str_param,
                **dual_vln_visualization_rate_hz.param(float),
                **dual_vln_model_output_topic.str_param,
                **dual_vln_command_service.str_param,
                **dual_vln_status_topic.str_param,
                'dual_vln_external_server': dual_vln_external_server.param_value(bool),
                'dual_vln_direct_cmd_vel': dual_vln_direct_cmd_vel.param_value(bool),
                **enable_collision_monitor.param(bool),
                **robot_launch_file.str_param,
            },
        ],
    )

    debug_window_cb = launch.event_handlers.OnProcessStart(
        target_action=task_generator_node,
        on_start=[
            launch.actions.ExecuteProcess(
                cmd=[
                    "/usr/bin/x-terminal-emulator",
                    "-e",
                    'bash -c "sleep 5; python -m aiomonitor.cli"',
                ],
                output="screen",
            )
        ],
    )

    ld = launch.LaunchDescription(
        [
            *ld_items,
            launch.actions.GroupAction(
                [
                    launch_ros.actions.PushRosNamespace(
                        namespace=namespace.substitution
                    ),
                    map_server_node,
                    pedestrian_marker_node,
                    rviz_node,
                ]
            ),
            launch.actions.RegisterEventHandler(
                debug_window_cb,
                condition=launch.conditions.IfCondition(debug.substitution),
            ),
            task_generator_node,
            # launch_ros.actions.Node(
            #     package='task_generator',
            #     executable='server',
            #     name='task_generator_server',
            #     output='screen'
            # ),
            # launch_ros.actions.Node(
            #     package='task_generator',
            #     executable='filewatcher',
            #     name='task_generator_filewatcher',
            #     output='screen'
            # )
        ]
    )
    return ld


if __name__ == "__main__":
    generate_launch_description()
