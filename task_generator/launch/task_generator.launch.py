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
    execution_mode = LaunchArgument('execution_mode', default_value='navigation')
    vln_rgb_topic = LaunchArgument('vln_rgb_topic', default_value='')
    model_heartbeat_timeout_sec = LaunchArgument('model_heartbeat_timeout_sec', default_value='2.5')
    vln_instruction = LaunchArgument(
        name="vln_instruction",
        default_value="navigate",
    )
    vln_instruction_file = LaunchArgument(
        name="vln_instruction_file",
        default_value="",
    )
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
    expected_human_count = LaunchArgument(
        name="expected_human_count",
        default_value="-1",
    )
    episode_start_delay_sec = LaunchArgument(
        name="episode_start_delay_sec",
        default_value="0.0",
    )
    # Must be declared HERE, not only in arena.launch.py.  A launch argument
    # forwarded into this included description is inert unless this file BOTH
    # declares it AND lists it in the task generator node's `parameters`
    # allowlist below.  Forwarding alone crosses no boundary, and the omission is
    # silent: the node's `rosparam[str].get(...)` simply returns its default, the
    # run completes, and pedestrians behave exactly as they did before.  That is
    # how one case03 slot was spent on a void run.
    pedestrian_goal_traversal = LaunchArgument(
        name="pedestrian_goal_traversal",
        default_value="",
    )
    social_yielding = LaunchArgument(
        name="social_yielding",
        # TRI-STATE string, not a bool: "auto" = not specified (defer to the scenario file),
        # "true"/"false" = explicit override that beats the scenario. A bool cannot express this,
        # since a default of false is indistinguishable from a user passing false.
        default_value="auto",
        choices=["auto", "true", "false"],
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
                **execution_mode.str_param,
                **vln_rgb_topic.str_param,
                **model_heartbeat_timeout_sec.param(float),
                **vln_instruction.str_param,
                **vln_instruction_file.str_param,
                **enable_collision_monitor.param(bool),
                **robot_launch_file.str_param,
                **reference.param(typing.List[float]),
                **prefix.str_param,
                **debug.param(bool),
                **train_mode.param(bool),
                **require_human_states_ready.param(bool),
                **human_states_ready_timeout_sec.param(float),
                **expected_human_count.param(int),
                **episode_start_delay_sec.param(float),
                **pedestrian_goal_traversal.str_param,
                # str, not bool: carries the tri-state ("auto"/"true"/"false") through to
                # tasks/robots/scenario.py, which resolves the precedence.
                **social_yielding.str_param,
            },
            {"use_sim_time": False},
            parameter_file.substitution,
            {
                # Explicit launch settings override defaults in the parameter file.
                **execution_mode.str_param,
                **vln_rgb_topic.str_param,
                **model_heartbeat_timeout_sec.param(float),
                'task': {
                    'scenario': {
                        'file': scenario_file.param_value(str),
                    },
                },
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
