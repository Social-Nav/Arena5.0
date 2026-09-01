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
        default_value='social_mpc',
        description='local planner type [teb, dwa, mpc, rlca, arena, rosnav, cohan, social_mpc]'
    )
    global_planner = LaunchArgument(
        name='global_planner',
        default_value='navfn',
        description='global planner type [navfn, smac_2d, smac_hybrid, smac_state_lattice]'
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
        default_value=PythonExpression([str({"dummy": "dummy", "gazebo": "hunav", "isaac": "hunav"}), '.get("', sim.substitution, '", "dummy")']),
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
    tm_robots = LaunchArgument(
        name='tm_robots',
        default_value='explore'
    )
    tm_obstacles = LaunchArgument(
        name='tm_obstacles',
        default_value='random'
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
    task_generator_param_file = LaunchArgument(
        name='task_generator_param_file',
        default_value=os.path.join(get_package_share_directory('arena_bringup'), 'configs', 'task_generator.yaml'),
        description='Override path for task_generator parameter yaml (default: arena_bringup configs/task_generator.yaml)',
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
    save_snapshots = LaunchArgument(
        name='save_snapshots',
        default_value='false',
        choices=['true', 'false'],
        description='Keep the yielding snapshot dirs (head/back/topdown PNGs + depth npy) after '
                    'the replan has read them. NOT a "skip capture" switch: the replan pipeline '
                    'reads the snapshot back off disk, so yielding cannot work without it being '
                    'written. Default false deletes each dir once its replan is done; true keeps '
                    'them for debugging. With world:= set they land in '
                    'social_gen/traj_data/grscenes/<world>/snapshots/, beside data/ and videos/.',
    )
    social_yielding = LaunchArgument(
        name='social_yielding',
        default_value='auto',
        choices=['auto', 'true', 'false'],
        description='Enable state for the proactive-yielding pipeline (trigger + orchestrator, '
                    'which always launch but stay inert until enabled). OVERRIDE: an explicit '
                    'true/false here wins over the scenario file. `auto` (the default) means "not '
                    'specified" and defers to the scenario file\'s robots[].social_yielding, then '
                    'false. Tri-state because a plain boolean cannot distinguish "not passed" from '
                    '"passed false".',
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
                    reference=list(next(references))
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
                        reference=list(next(references))
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
        reference: typing.List[float]
    ):
        return IsolatedGroupAction([
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
                    **tm_modules.dict,
                    **robot.dict,
                    **inter_planner.dict,
                    **local_planner.dict,
                    **global_planner.dict,
                    **world.dict,
                    **record_data_dir.dict,
                    **debug.dict,
                    **save_data.dict,
                    **social_yielding.dict,
                    'namespace': namespace,
                    'headless': headlessness,
                    'reference': str(reference),
                    'prefix': prefix,
                    'parameter_file': task_generator_param_file.substitution,
                    **train_mode.dict,
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
            **robot.dict,
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
        # Always launch both yielding nodes; they stay INERT (no polling) until enabled at reset.
        # Enable resolution: the launch arg if explicitly true/false, else scenario.yaml's
        # `robot.social_yielding`, else false (task_generator resolves + publishes it).
        # Both yielding nodes read ARENA_ROBOT to build their namespace and base frame.
        launch.actions.SetEnvironmentVariable(
            name='ARENA_ROBOT',
            value=robot.substitution,
        ),
        # Read by the orchestrator (SAVE_SNAPSHOTS) to decide whether each snapshot dir
        # survives its replan. Passed as an env var, not a ROS param, to match how
        # ARENA_ROBOT already reaches these two plain-python nodes.
        launch.actions.SetEnvironmentVariable(
            name='ARENA_SAVE_SNAPSHOTS',
            value=save_snapshots.substitution,
        ),
        launch.actions.ExecuteProcess(
            cmd=['bash', '-c',
                 'source /opt/arena_ws/src/Arena/_meta/tools/source && '
                 'python3 /opt/arena_ws/src/Arena/arena_isaac/arena_isaac/arena_isaac/social_yielding/proactive_yielding_trigger.py'],
            output='screen',
        ),
        launch.actions.ExecuteProcess(
            cmd=['bash', '-c',
                 'source /opt/arena_ws/src/Arena/_meta/tools/source && '
                 'python3 /opt/arena_ws/src/Arena/arena_isaac/arena_isaac/arena_isaac/social_yielding/social_yielding_orchestrator.py'],
            output='screen',
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
