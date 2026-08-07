import os
import typing

import launch.conditions
import launch.launch_description_sources
import launch.utilities
import launch.utilities.type_utils
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution

import launch
from arena_bringup.actions import IsolatedGroupAction
from arena_bringup.extensions.NodeLogLevelExtension import SetGlobalLogLevelAction
from arena_bringup.future import IfElseSubstitution, PythonExpression
from arena_bringup.substitutions import LaunchArgument


def generate_launch_description():
    bringup_dir = get_package_share_directory('arena_bringup')

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    def declare_legacy_alias(name: str, target: LaunchArgument) -> LaunchArgument:
        return LaunchArgument(
            name=name,
            default_value=target.substitution,
            description=f'Legacy alias for {target.name}'
        )

    log_level = LaunchArgument(
        name='log_level',
        default_value='warn',
        choices=['debug', 'info', 'warn', 'error', 'fatal'],
        description='Set the log level for all nodes'
    )

    robot = LaunchArgument(
        name='robot',
        default_value='jackal',
        description='robot model type'
    )
    inter_planner = LaunchArgument(
        name='inter_planner',
        default_value='navigate_to_pose_w_replanning_and_recovery',
        description='inter planner type (Behavior Tree)'
    )
    local_planner = LaunchArgument(
        name='local_planner',
        default_value='dwb',
        description='local planner type [teb, dwa, mpc, rlca, arena, rosnav, cohan]'
    )
    global_planner = LaunchArgument(
        name='global_planner',
        default_value='navfn',
        description='global planner type [navfn]'
    )
    sim = LaunchArgument(
        name='sim',
        default_value='dummy',  # todo select first installed simulator
    )
    headless = LaunchArgument(
        name='headless',
        default_value='0',
        choices=['-1', '0', '1', '2'],
        description='-1 = show all environments, 0 = show all, 1 = show only rviz, 2 = show nothing'
    )
    human = LaunchArgument(
        name='human',
        description='human simulator to use',
        default_value=PythonExpression([str({"dummy": "dummy", "gazebo": "hunav", "isaac": "hunav", "isaac_eval": "hunav"}), '.get("', sim.substitution, '", "dummy")']),
    )
    complexity = LaunchArgument(
        name='complexity',
        default_value='1',
        description='1 = Map known, Position known; 2 = Map known, Position unknown (AMCL); 3 = Map unknown, Position unknown (SLAM)'
    )
    agent_name = LaunchArgument(
        name='agent_name',
        default_value=robot.substitution,
        description='DRL agent name to be deployed'
    )
    record_data_dir = LaunchArgument(
        name='record_data_dir',
        default_value=''
    )
    episodes = LaunchArgument(
        name='episodes',
        default_value='2',
        description='Number of episodes to execute before task_generator publishes finished'
    )
    auto_reset = LaunchArgument(
        name='auto_reset',
        default_value='true',
        choices=['true', 'false'],
        description='Automatically reset tasks between episodes during evaluation'
    )
    timeout = LaunchArgument(
        name='timeout',
        default_value='120.0',
        description='Task timeout in seconds'
    )
    timeout_wall_factor = LaunchArgument(
        name='timeout_wall_factor',
        default_value='5.0',
        description='Wall-clock timeout multiplier for slow simulator/model evals'
    )
    timeout_wall_sec = LaunchArgument(
        name='timeout_wall_sec',
        default_value='0.0',
        description='Explicit wall-clock timeout in seconds; 0 derives it from timeout_wall_factor'
    )
    vln_instruction = LaunchArgument(
        name='vln_instruction',
        default_value='navigate',
        description='Default text instruction published for each episode'
    )
    vln_instruction_file = LaunchArgument(
        name='vln_instruction_file',
        default_value='',
        description='Optional file containing the instruction text to publish'
    )
    internnav_mode = LaunchArgument(
        name='internnav_mode',
        default_value='heuristic',
        description='InternNav wrapper mode: heuristic or model backend instance'
    )
    dual_vln_mode = declare_legacy_alias('dual_vln_mode', internnav_mode)
    internnav_model_path = LaunchArgument(
        name='internnav_model_path',
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_INTERNNAV_MODEL_PATH',
            default_value=launch.substitutions.EnvironmentVariable(
                'INTERNNAV_MODEL_PATH',
                default_value=launch.substitutions.EnvironmentVariable('ARENA_VLN_MODEL_PATH', default_value=''),
            ),
        ),
        description='Path to a model checkpoint when the InternNav wrapper uses model mode'
    )
    dual_vln_model_path = declare_legacy_alias('dual_vln_model_path', internnav_model_path)
    internnav_device = LaunchArgument(
        name='internnav_device',
        default_value='cpu',
        description='Inference device for the InternNav wrapper model mode'
    )
    dual_vln_device = declare_legacy_alias('dual_vln_device', internnav_device)
    internnav_inference_rate_hz = LaunchArgument(
        name='internnav_inference_rate_hz',
        default_value='3.3333333333',
        description='Outer InternNav realworld client planning request rate in Hz'
    )
    dual_vln_inference_rate_hz = declare_legacy_alias('dual_vln_inference_rate_hz', internnav_inference_rate_hz)
    internnav_inference_timeout_sec = LaunchArgument(
        name='internnav_inference_timeout_sec',
        default_value='0.2',
        description='Discard model inference outputs slower than this timeout'
    )
    dual_vln_inference_timeout_sec = declare_legacy_alias('dual_vln_inference_timeout_sec', internnav_inference_timeout_sec)
    internnav_rgb_topic = LaunchArgument(
        name='internnav_rgb_topic',
        default_value='',
        description='Optional RGB topic for the InternNav wrapper input / debug visualization'
    )
    dual_vln_rgb_topic = declare_legacy_alias('dual_vln_rgb_topic', internnav_rgb_topic)
    internnav_depth_topic = LaunchArgument(
        name='internnav_depth_topic',
        default_value='',
        description='Optional depth topic for the InternNav wrapper model input'
    )
    dual_vln_depth_topic = declare_legacy_alias('dual_vln_depth_topic', internnav_depth_topic)
    internnav_camera_info_topic = LaunchArgument(
        name='internnav_camera_info_topic',
        default_value='',
        description='Optional CameraInfo topic for InternNav native vision backends'
    )
    dual_vln_camera_info_topic = declare_legacy_alias('dual_vln_camera_info_topic', internnav_camera_info_topic)
    internnav_python_executable = LaunchArgument(
        name='internnav_python_executable',
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_VLN_MODEL_PYTHON',
            default_value=launch.substitutions.EnvironmentVariable(
                'ARENA_INTERNNAV_PYTHON',
                default_value=launch.substitutions.EnvironmentVariable('ARENA_PYTHON', default_value=''),
            ),
        ),
        description='Optional Python interpreter used to launch internnav_server'
    )
    dual_vln_python_executable = declare_legacy_alias('dual_vln_python_executable', internnav_python_executable)
    internnav_adapter_target = LaunchArgument(
        name='internnav_adapter_target',
        default_value='',
        description='Optional Python adapter target (module:attr) for custom model backends inside the InternNav wrapper'
    )
    dual_vln_adapter_target = declare_legacy_alias('dual_vln_adapter_target', internnav_adapter_target)
    internnav_http_url = LaunchArgument(
        name='internnav_http_url',
        default_value=launch.substitutions.EnvironmentVariable(
            'ARENA_EVAL_INTERNNAV_HTTP_URL',
            default_value=launch.substitutions.EnvironmentVariable('ARENA_INTERNNAV_HTTP_URL', default_value=''),
        ),
        description='HTTP URL for InternVLA realworld /eval_dual adapter'
    )
    dual_vln_http_url = declare_legacy_alias('dual_vln_http_url', internnav_http_url)
    internnav_http_timeout_sec = LaunchArgument(
        name='internnav_http_timeout_sec',
        # Keep launch-time float conversion deterministic; internnav_server and
        # adapter read ARENA_*_HTTP_TIMEOUT_SEC directly and can recover from
        # invalid env values.
        default_value='0.0',
        description='HTTP timeout in seconds for InternVLA realworld /eval_dual adapter'
    )
    dual_vln_http_timeout_sec = declare_legacy_alias('dual_vln_http_timeout_sec', internnav_http_timeout_sec)
    internnav_require_real_backend = LaunchArgument(
        name='internnav_require_real_backend',
        default_value='false',
        description='Fail fast instead of using heuristic/mock fallback when the requested wrapper backend cannot load'
    )
    dual_vln_require_real_backend = declare_legacy_alias('dual_vln_require_real_backend', internnav_require_real_backend)
    internnav_strict_device = LaunchArgument(
        name='internnav_strict_device',
        default_value='false',
        description='Fail fast instead of falling back to CPU when the requested wrapper device is unavailable'
    )
    dual_vln_strict_device = declare_legacy_alias('dual_vln_strict_device', internnav_strict_device)
    internnav_look_down = LaunchArgument(
        name='internnav_look_down',
        default_value='false',
        description='Forward a look_down hint to native InternNav backends'
    )
    dual_vln_look_down = declare_legacy_alias('dual_vln_look_down', internnav_look_down)
    internnav_model_output_policy = LaunchArgument(
        name='internnav_model_output_policy',
        default_value='trajectory',
        choices=['trajectory', 'discrete', 'raw'],
        description='How InternNav model outputs are converted: trajectory prefers output_trajectory->cmd_vel; discrete forces action ids; raw keeps legacy adapter precedence'
    )
    dual_vln_model_output_policy = declare_legacy_alias('dual_vln_model_output_policy', internnav_model_output_policy)
    internnav_enable_visualization = LaunchArgument(
        name='internnav_enable_visualization',
        default_value='false',
        description='Publish annotated debug images for the InternNav wrapper when true'
    )
    dual_vln_enable_visualization = declare_legacy_alias('dual_vln_enable_visualization', internnav_enable_visualization)
    internnav_visualization_topic = LaunchArgument(
        name='internnav_visualization_topic',
        default_value='internnav/debug_image',
        description='Topic for annotated InternNav debug images'
    )
    dual_vln_visualization_topic = declare_legacy_alias('dual_vln_visualization_topic', internnav_visualization_topic)
    internnav_action_visualization_topic = LaunchArgument(
        name='internnav_action_visualization_topic',
        default_value='internnav/action_image',
        description='Topic for InternNav ego-centric action visualization images'
    )
    dual_vln_action_visualization_topic = declare_legacy_alias(
        'dual_vln_action_visualization_topic', internnav_action_visualization_topic
    )
    internnav_visualization_rate_hz = LaunchArgument(
        name='internnav_visualization_rate_hz',
        default_value='5.0',
        description='Maximum debug image publish rate for InternNav visualization'
    )
    dual_vln_visualization_rate_hz = declare_legacy_alias('dual_vln_visualization_rate_hz', internnav_visualization_rate_hz)
    internnav_model_output_topic = LaunchArgument(
        name='internnav_model_output_topic',
        default_value='internnav/model_output',
        description='Topic for InternNav raw model output JSON diagnostics'
    )
    dual_vln_model_output_topic = declare_legacy_alias('dual_vln_model_output_topic', internnav_model_output_topic)
    internnav_external_server = LaunchArgument(
        name='internnav_external_server',
        default_value='false',
        choices=['true', 'false'],
        description='Use the dedicated internnav-1 InternNav server instead of starting a model server in arena-1'
    )
    dual_vln_external_server = declare_legacy_alias('dual_vln_external_server', internnav_external_server)
    internnav_direct_cmd_vel = LaunchArgument(
        name='internnav_direct_cmd_vel',
        default_value='false',
        choices=['true', 'false'],
        description='Use upstream InternNav realworld ROS2 client publishing cmd_vel directly; no Arena get_command wrapper.'
    )
    dual_vln_direct_cmd_vel = declare_legacy_alias('dual_vln_direct_cmd_vel', internnav_direct_cmd_vel)
    internnav_command_service = LaunchArgument(
        name='internnav_command_service',
        default_value='',
        description='Optional external InternNav get_command service name. Defaults to the robot namespace service.'
    )
    dual_vln_command_service = declare_legacy_alias('dual_vln_command_service', internnav_command_service)
    internnav_status_topic = LaunchArgument(
        name='internnav_status_topic',
        default_value='',
        description='Optional external InternNav status topic. Defaults to the robot namespace status topic.'
    )
    dual_vln_status_topic = declare_legacy_alias('dual_vln_status_topic', internnav_status_topic)
    internnav_timing_mode = LaunchArgument(
        name='internnav_timing_mode',
        default_value='wall',
        choices=['wall', 'sim_time_realworld'],
        description='Command timing emulation mode for direct official-client InternNav eval'
    )
    dual_vln_timing_mode = declare_legacy_alias('dual_vln_timing_mode', internnav_timing_mode)
    internnav_model_latency_sec = LaunchArgument(
        name='internnav_model_latency_sec',
        default_value='0.3',
        description='Sim-time command delay used by internnav_timing_manager'
    )
    dual_vln_model_latency_sec = declare_legacy_alias('dual_vln_model_latency_sec', internnav_model_latency_sec)
    internnav_latency_policy = LaunchArgument(
        name='internnav_latency_policy',
        default_value='fixed',
        choices=['fixed', 'measured'],
        description='Latency source used by internnav_timing_manager'
    )
    dual_vln_latency_policy = declare_legacy_alias('dual_vln_latency_policy', internnav_latency_policy)
    internnav_raw_cmd_vel_topic = LaunchArgument(
        name='internnav_raw_cmd_vel_topic',
        default_value='internnav/raw_cmd_vel',
        description='Raw official-client cmd_vel topic consumed by internnav_timing_manager'
    )
    dual_vln_raw_cmd_vel_topic = declare_legacy_alias('dual_vln_raw_cmd_vel_topic', internnav_raw_cmd_vel_topic)
    enable_collision_monitor = LaunchArgument(
        name='enable_collision_monitor',
        default_value='true',
        description='Enable Nav2 collision_monitor for robot navigation launch'
    )
    robot_launch_file = LaunchArgument(
        name='robot_launch_file',
        default_value='robot.launch.py',
        description='Robot-level launch file in arena_simulation_setup/launch for the selected run case'
    )
    tm_robots = LaunchArgument(
        name='tm_robots',
        default_value='explore'
    )
    tm_obstacles = LaunchArgument(
        name='tm_obstacles',
        default_value='random'
    )
    scenario_file = LaunchArgument(
        name='scenario_file',
        default_value='default',
        description='Scenario file/name forwarded to task.scenario.file'
    )
    tm_modules = LaunchArgument(
        name='tm_modules',
        default_value='rviz_ui'  # TODO breaks launch if empty
    )
    world = LaunchArgument(
        name='world',
        default_value='map_empty',
        description='world to load'
    )
    use_sim_time = LaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )
    env_n = LaunchArgument(
        name='env_n',
        default_value='1',
        description='Number of environments to spawn within simulator'
    )
    env_d = LaunchArgument(
        name='env_d',
        default_value='50',
        description='space between environments'
    )
    debug = LaunchArgument(
        name='debug',
        default_value='False',
        description='Enable debug features'
    )
    save_data = LaunchArgument(
        name='save_data',
        default_value='false',
        choices=['true', 'false'],
        description='Enable VLN dataset logging'
    )
    train_config = LaunchArgument(
        name='train_config',
        default_value='',
        description='Path to training config YAML. When provided, train_mode is implied true and train_agent.py is started automatically.'
    )
    train_mode = LaunchArgument(
        name='train_mode',
        default_value=PythonExpression(['"', train_config.substitution, '" != ""']),
        description='If true, RL env publishes cmd_vel directly; nav2 controller output is silenced. Implied when train_config is provided.'
    )
    require_human_states_ready = LaunchArgument(
        name='require_human_states_ready',
        default_value='false',
        description='If true, delay task_reset and goal release until a non-empty HuNav human_states message is observed.'
    )
    human_states_ready_timeout_sec = LaunchArgument(
        name='human_states_ready_timeout_sec',
        default_value='10.0',
        description='Maximum wait for non-empty HuNav human_states before releasing an episode.'
    )
    episode_start_delay_sec = LaunchArgument(
        name='episode_start_delay_sec',
        default_value='0.0',
        description='Additional post-reset delay before publishing task_reset and releasing navigation goals.'
    )
    pedestrian_goal_traversal = LaunchArgument(
        name='pedestrian_goal_traversal',
        default_value='',
        description=(
            'Run-level pedestrian waypoint traversal mode: once (walk 0->N then stop, '
            'the default), cyclic (return to waypoint 0 and repeat forward), or '
            'reciprocate (ping-pong 0->N->0->N for the whole episode). Empty means use '
            'configs/hunav/default.yaml. Per-pedestrian scenario keys take precedence. '
            'reciprocate changes scene dynamics, so its social metrics are NOT '
            'comparable with once-mode runs.'
        )
    )

    def create_task_generators(
        context: launch.LaunchContext,
        *,
        n_substitution: launch.SomeSubstitutionsType,
        d_substitution: launch.SomeSubstitutionsType,
    ) -> typing.Optional[typing.List[launch.LaunchDescriptionEntity]]:
        n = launch.utilities.type_utils.perform_typed_substitution(
            context,
            launch.utilities.normalize_to_list_of_substitutions(n_substitution),
            int,
        )
        n = typing.cast(int, n)
        d = launch.utilities.type_utils.perform_typed_substitution(
            context,
            launch.utilities.normalize_to_list_of_substitutions(d_substitution),
            float,
        )
        d = typing.cast(float, d)
        scenario_file_value = scenario_file.substitution

        # Log env_n value
        launch.actions.LogInfo(
            msg=[
                TextSubstitution(text="env_n value: "),
                TextSubstitution(text=str(n))
            ]
        ).execute(context)

        if n < 1:
            return None

        task_generators = []
        base_namespace = 'task_generator_node'
        base_prefix = 'env'
        references = snail_grid(d)

        if n == 1:
            task_generators.append(
                create_task_generator(
                    headlessness=PythonExpression([headless.substitution, '>1']),
                    namespace=base_namespace,
                    prefix='',
                    reference=list(next(references)),
                    scenario_file_value=scenario_file_value,
                )
            )

        else:
            for i in range(n):
                prefix = base_prefix + str(i)
                task_generators.append(
                    create_task_generator(
                        headlessness=PythonExpression([headless.substitution, '>-1']),
                        namespace=os.path.join(base_namespace, prefix),
                        prefix=prefix,
                        reference=list(next(references)),
                        scenario_file_value=scenario_file_value,
                    )
                )

        # Log total task generators
        launch.actions.LogInfo(
            msg=[
                TextSubstitution(text="Total task_generator nodes spawned: "),
                TextSubstitution(text=str(len(task_generators)))
            ]
        ).execute(context)

        return task_generators

    def create_task_generator(
        headlessness,
        namespace: str,
        prefix: str,
        reference: typing.List[float],
        scenario_file_value,
    ):
        return IsolatedGroupAction([
            launch.actions.SetEnvironmentVariable(
                name='ARENA_SCENARIO_FILE',
                value=scenario_file.substitution,
            ),
            LogInfo(msg=[
                TextSubstitution(text="Spawning task_generator with namespace: "),
                TextSubstitution(text=namespace)
            ]),
            launch.actions.IncludeLaunchDescription(
                launch.launch_description_sources.PythonLaunchDescriptionSource(
                    os.path.join(bringup_dir, 'launch/simulator/human/human.launch.py')
                ),
                launch_arguments={
                    'simulator': human.substitution,
                    'namespace': namespace,
                }.items()
            ),
            launch.actions.IncludeLaunchDescription(
                launch.launch_description_sources.PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('task_generator'),
                        'launch/task_generator.launch.py'
                    )
                ),
                launch_arguments={
                    **sim.dict,
                    **human.dict,
                    **tm_obstacles.dict,
                    **tm_robots.dict,
                    'scenario_file': scenario_file_value,
                    **tm_modules.dict,
                    **robot.dict,
                    **inter_planner.dict,
                    **local_planner.dict,
                    **global_planner.dict,
                    **world.dict,
                    **record_data_dir.dict,
                    **episodes.dict,
                    **auto_reset.dict,
                    **timeout.dict,
                    **timeout_wall_factor.dict,
                    **timeout_wall_sec.dict,
                    **vln_instruction.dict,
                    **vln_instruction_file.dict,
                    **dual_vln_mode.dict,
                    **dual_vln_model_path.dict,
                    **dual_vln_device.dict,
                    **dual_vln_inference_rate_hz.dict,
                    **dual_vln_inference_timeout_sec.dict,
                    **dual_vln_rgb_topic.dict,
                    **dual_vln_depth_topic.dict,
                    **dual_vln_camera_info_topic.dict,
                    **dual_vln_python_executable.dict,
                    **dual_vln_adapter_target.dict,
                    **dual_vln_http_url.dict,
                    **dual_vln_http_timeout_sec.dict,
                    **dual_vln_require_real_backend.dict,
                    **dual_vln_strict_device.dict,
                    **dual_vln_model_output_policy.dict,
                    **dual_vln_look_down.dict,
                    **dual_vln_enable_visualization.dict,
                    **dual_vln_visualization_topic.dict,
                    **dual_vln_action_visualization_topic.dict,
                    **dual_vln_visualization_rate_hz.dict,
                    **dual_vln_model_output_topic.dict,
                    **internnav_external_server.dict,
                    **dual_vln_external_server.dict,
                    **internnav_direct_cmd_vel.dict,
                    **dual_vln_direct_cmd_vel.dict,
                    **dual_vln_command_service.dict,
                    **dual_vln_status_topic.dict,
                    **dual_vln_timing_mode.dict,
                    **dual_vln_model_latency_sec.dict,
                    **dual_vln_latency_policy.dict,
                    **dual_vln_raw_cmd_vel_topic.dict,
                    **enable_collision_monitor.dict,
                    **robot_launch_file.dict,
                    **debug.dict,
                    **save_data.dict,
                    'namespace': namespace,
                    'headless': headlessness,
                    'reference': str(reference),
                    'prefix': prefix,
                    'parameter_file': os.path.join(get_package_share_directory('arena_bringup'), 'configs', 'task_generator.yaml'),
                    **train_mode.dict,
                    **require_human_states_ready.dict,
                    **human_states_ready_timeout_sec.dict,
                    **episode_start_delay_sec.dict,
                    **pedestrian_goal_traversal.dict,
                }.items(),
            )
        ])

    launch_task_generators = launch.actions.OpaqueFunction(
        function=create_task_generators,
        kwargs={
            'n_substitution': env_n.substitution,
            'd_substitution': env_d.substitution,
        },
    )

    launch_simulator = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch/simulator/sim/sim.launch.py')
        ),
        launch_arguments={
            **use_sim_time.dict,
            'simulator': sim.substitution,
            **world.dict,
            **save_data.dict,
            'headless': PythonExpression([headless.substitution, '>0']),
        }.items(),
    )

    world_generator_node = launch_ros.actions.Node(
        package='arena_simulation_setup',
        executable='world_generator',
        name='world_generator',
        output='screen',
    )

    ld = launch.LaunchDescription([
        *ld_items,
        LogInfo(
            msg=[
                TextSubstitution(text="Starting arena bringup with env_n="),
                env_n.substitution,
                TextSubstitution(text=" task_generator_node(s)")
            ]
        ),
        SetGlobalLogLevelAction(log_level.substitution),
        launch_task_generators,
        IsolatedGroupAction([launch_simulator]),
        world_generator_node,
        launch.actions.ExecuteProcess(
            cmd=['ros2', 'run', 'arena_training', 'train_agent.py',
                 '--config', train_config.substitution],
            output='screen',
            condition=launch.conditions.IfCondition(
                PythonExpression(['"', train_config.substitution, '" != ""'])
            ),
        ),
    ])
    return ld


def snail_grid(d: float, initial=None):
    if initial is None:
        initial = (0, 0)
    x, y = map(float, initial)

    step: int = 0
    while True:
        yield x, y

        for _ in range(step):
            y -= d
            yield x, y

        for _ in range(step):
            x -= d
            yield x, y

        for _ in range(step):
            y += d
            yield x, y

        for _ in range(step):
            x += d
            yield x, y

        x += d
        y += d
        step += 2


if __name__ == '__main__':
    generate_launch_description()
