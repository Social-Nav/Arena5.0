"""Case-specific launch for the LISN slow-loop engineering sidecar.

This is deliberately separate from the shared ``arena.launch.py``: it adds one
sidecar node and does not change the shared fast FollowPath chain.  The node
refuses to start while the engineering profile is not runtime-ready.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory('arena_bringup')
    policy = LaunchConfiguration('policy')
    evidence = LaunchConfiguration('evidence')
    controller_node = LaunchConfiguration('controller_node')
    proposal_topic = LaunchConfiguration('proposal_topic')
    target_frame = LaunchConfiguration('target_frame')
    episode_id = LaunchConfiguration('episode_id')
    producer_endpoint = LaunchConfiguration('producer_endpoint')

    return LaunchDescription([
        DeclareLaunchArgument(
            'policy',
            default_value=f'{share}/configs/lisn/slow_loop_engineering_v1.json',
            description='Engineering slow-loop policy (new identity, frozen contract untouched).',
        ),
        DeclareLaunchArgument('evidence', default_value='/tmp/lisn_slow_loop/evidence.jsonl'),
        DeclareLaunchArgument(
            'controller_node',
            default_value='/task_generator_node/Ai2_Bot2/controller_server',
        ),
        DeclareLaunchArgument(
            'proposal_topic',
            default_value='/task_generator_node/Ai2_Bot2/lisn_slow_loop/proposal',
        ),
        DeclareLaunchArgument('target_frame', default_value='map'),
        DeclareLaunchArgument('episode_id', default_value='lisn-episode'),
        DeclareLaunchArgument(
            'producer_endpoint',
            default_value='',
            description='Real LISN VLM JSON-HTTP endpoint; empty means proposal-topic mode.',
        ),
        LogInfo(msg='LISN slow-loop engineering sidecar; runtime_ready=false refuses production start'),
        Node(
            package='arena_bringup',
            executable='lisn_slow_loop_bridge',
            name='lisn_slow_loop_bridge',
            output='screen',
            arguments=[
                '--engineering',
                '--policy', policy,
                '--evidence', evidence,
                '--controller-node', controller_node,
                '--proposal-topic', proposal_topic,
                '--target-frame', target_frame,
                '--episode-id', episode_id,
                '--producer-endpoint', producer_endpoint,
            ],
        ),
    ])
