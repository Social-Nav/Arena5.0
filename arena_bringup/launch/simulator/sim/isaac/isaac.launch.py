import launch.actions
import launch.substitutions

from arena_bringup.substitutions import LaunchArgument
from launch import LaunchDescription


def generate_launch_description():
    ld = []
    LaunchArgument.auto_append(ld)
    log_level = LaunchArgument(
        name='log_level',
        default_value='debug',
        description='Logging level',
    )
    save_data = LaunchArgument(
        name='save_data',
        default_value='false',
        description='data saving flag',
    )
    robot = LaunchArgument(
        name='robot',
        default_value='jackal',
        description='robot model name (used for nav topic namespace)',
    )

    isaac_pythonpath = ':'.join([
        '/opt/isaac_bridge_msgs/arena_people_msgs/lib/python3.11/site-packages',
        '/opt/isaac_bridge_msgs/hunav_msgs/lib/python3.11/site-packages',
        '/opt/isaac_bridge_msgs/isaacsim_msgs/lib/python3.11/site-packages',
        '/home/ubuntu/arena_jazzy_ws/install_humble_eval/arena_isaac/lib/python3.10/site-packages',
        '/home/ubuntu/arena_jazzy_ws/src/Arena/arena_simulation_setup/src',
        '/home/ubuntu/arena_jazzy_ws/src/Arena/arena_isaac/arena_isaac',
        '/home/ubuntu/arena_jazzy_ws/src/Arena/arena_robots/arena_robots',
        '/home/ubuntu/arena_jazzy_ws/src/Arena/arena_robots',
        '/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/python',
        '/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy',
    ])
    isaac_ld_library_path = ':'.join([
        '/opt/isaac_bridge_msgs/arena_people_msgs/lib',
        '/opt/isaac_bridge_msgs/hunav_msgs/lib',
        '/opt/isaac_bridge_msgs/isaacsim_msgs/lib',
        '/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib',
        '/lib/x86_64-linux-gnu',
        '/usr/lib/x86_64-linux-gnu',
    ])
    isaac_ament_prefix_path = ':'.join([
        '/home/ubuntu/arena_jazzy_ws/install/arena_bringup',
        '/home/ubuntu/arena_jazzy_ws/install/arena_robots',
        '/home/ubuntu/arena_jazzy_ws/install/arena_simulation_setup',
        '/home/ubuntu/arena_jazzy_ws/install/arena_isaac',
        '/home/ubuntu/arena_jazzy_ws/install_humble_eval/arena_bringup',
        '/home/ubuntu/arena_jazzy_ws/install_humble_eval/arena_robots',
        '/home/ubuntu/arena_jazzy_ws/install_humble_eval/arena_simulation_setup',
        '/home/ubuntu/arena_jazzy_ws/install_humble_eval/arena_isaac',
        '/home/ubuntu/arena_jazzy_ws/install',
        '/opt/ros/jazzy',
    ])
    isaac_cmd = (
        # A previous docker exec can leave Isaac Python alive after the host
        # launch tree is interrupted.  Multiple live PublishTime graphs publish
        # interleaved /clock epochs, which makes Nav2 repeatedly clear TF and
        # abort with disconnected map/odom/base trees.  Force the Isaac
        # container to be single-source before starting a new eval run.
        'pkill -INT -f ^/isaac-sim/kit/python/bin/python3.*/run_isaacsim.py 2>/dev/null || true; '
        'pkill -INT -f ^/bin/bash.*/isaac-sim/python.sh.*/run_isaacsim.py 2>/dev/null || true; '
        'sleep 2; '
        'pkill -KILL -f ^/isaac-sim/kit/python/bin/python3.*/run_isaacsim.py 2>/dev/null || true; '
        'pkill -KILL -f ^/bin/bash.*/isaac-sim/python.sh.*/run_isaacsim.py 2>/dev/null || true; '
        'source /opt/isaac_bridge_msgs/local_setup.bash >/dev/null 2>&1; '
        'export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"; '
        'export ROS_LOCALHOST_ONLY=0; '
        'export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET; '
        'export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; '
        f'export AMENT_PREFIX_PATH=\'{isaac_ament_prefix_path}\'; '
        f'export PYTHONPATH=\'{isaac_pythonpath}\'; '
        f'export LD_LIBRARY_PATH=\'{isaac_ld_library_path}\':"${{LD_LIBRARY_PATH}}"; '
        'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; '
        'export ARENA_ISAAC_HEADLESS="${ARENA_ISAAC_HEADLESS:-1}"; '
        'export ARENA_ISAAC_RENDERER="${ARENA_ISAAC_RENDERER:-RTXLinear}"; '
        'export ARENA_DISABLE_ISAAC_ODOM_GRAPH="${ARENA_DISABLE_ISAAC_ODOM_GRAPH:-0}"; '
        'export ARENA_SPAWN_USD_ROBOT_ENABLE_ISAAC_ODOM_GRAPH="${ARENA_SPAWN_USD_ROBOT_ENABLE_ISAAC_ODOM_GRAPH:-1}"; '
        'python3 /home/ubuntu/arena_jazzy_ws/src/Arena/arena_isaac/arena_isaac/arena_isaac/run_isaacsim.py '
        '--save-data "$ARENA_SAVE_DATA" --robot "$ARENA_ROBOT" 2>&1 | tee /tmp/isaac_sim.log'
    )

    return LaunchDescription([
        *ld,
        launch.actions.SetEnvironmentVariable(
            name='ARENA_LOG_LEVEL',
            value=log_level.substitution,
        ),
        launch.actions.SetEnvironmentVariable(
            name='ARENA_SAVE_DATA',
            value=save_data.substitution,
        ),
        launch.actions.SetEnvironmentVariable(
            name='ARENA_ROBOT',
            value=robot.substitution,
        ),
        launch.actions.SetEnvironmentVariable(
            name='ROS_DOMAIN_ID',
            value=launch.substitutions.EnvironmentVariable('ROS_DOMAIN_ID', default_value='0'),
        ),
        launch.actions.SetEnvironmentVariable(
            name='ROS_LOCALHOST_ONLY',
            value='0',
        ),
        launch.actions.SetEnvironmentVariable(
            name='ROS_AUTOMATIC_DISCOVERY_RANGE',
            value='SUBNET',
        ),
        launch.actions.SetEnvironmentVariable(
            name='RMW_IMPLEMENTATION',
            value='rmw_fastrtps_cpp',
        ),
        launch.actions.SetEnvironmentVariable(
            name='FASTDDS_BUILTIN_TRANSPORTS',
            value='UDPv4',
        ),
        launch.actions.ExecuteProcess(
            cmd=['bash', '-c', f'docker exec docker-isaac-1 bash -lc "{isaac_cmd}"'],
            sigterm_timeout=launch.substitutions.LaunchConfiguration('sigterm_timeout', default='60'),
            sigkill_timeout=launch.substitutions.LaunchConfiguration('sigkill_timeout', default='30'),
            on_exit=[launch.actions.Shutdown()],
            output='log',
        )
    ])
