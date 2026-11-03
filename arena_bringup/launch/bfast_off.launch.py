"""Explicit B off arm: strict RGB-only, fast guard disabled.

Both B arms use the same RGB topic, model, prompt, and common executor action.
This launch starts the common FollowPath controller/executor with no veto topic
and no guard node.  It does not start Isaac or a model server.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    follow = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('arena_vln_models'),
            'launch', 'follow_path.launch.py')),
        launch_arguments={
            'robot': LaunchConfiguration('robot'),
            'namespace': LaunchConfiguration('namespace'),
            'frame': LaunchConfiguration('frame'),
            'local_planner': 'regulated_pure_pursuit',
            'model_adapter': LaunchConfiguration('model_adapter'),
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'fast_guard_enabled': 'false',
            'fast_guard_veto_topic': '',
            'fast_guard_required': 'false',
            'bfast_decision_topic': LaunchConfiguration('bfast_decision_topic'),
        }.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument('robot', default_value='Ai2_Bot2'),
        DeclareLaunchArgument('namespace', default_value='/task_generator_node/Ai2_Bot2'),
        DeclareLaunchArgument('frame', default_value='Ai2_Bot2/'),
        DeclareLaunchArgument('model_adapter', default_value='internnav'),
        DeclareLaunchArgument(
            'rgb_topic', default_value='/task_generator_node/Ai2_Bot2/head_camera/image'),
        DeclareLaunchArgument(
            'bfast_decision_topic',
            default_value='/task_generator_node/Ai2_Bot2/bfast/decision'),
        follow,
    ])
