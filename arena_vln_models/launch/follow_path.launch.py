"""Native FollowPath tracking only. TaskGenerator owns controller lifecycle."""
import os

from ament_index_python.packages import get_package_share_directory
from arena_bringup.substitutions import YAMLFileSubstitution, YAMLMergeSubstitution
from arena_vln_models.tracking_config import tracking_parameters
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_nodes(context):
    get = lambda name: LaunchConfiguration(name).perform(context)
    setup = get_package_share_directory('arena_simulation_setup')
    robots = get_package_share_directory('arena_robots')
    merged = YAMLMergeSubstitution(
        YAMLFileSubstitution(os.path.join(setup, 'configs/nav2/defaults/model_params.yaml')),
        YAMLFileSubstitution(os.path.join(robots, 'robots', get('robot'), 'model_params.yaml')),
    )
    robot = YAMLFileSubstitution(merged).perform_load(context)
    params = YAMLFileSubstitution.from_dict({get('namespace'): tracking_parameters(robot, get('frame'))})
    params_file = params.perform(context)
    nodes = [
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             namespace=get('namespace'), parameters=[params_file], output='screen',
             remappings=[('cmd_vel', 'cmd_vel_nav')]),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
             namespace=get('namespace'), parameters=[params_file], output='screen',
             remappings=[('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_smoother', namespace=get('namespace'), output='screen',
             parameters=[{'autostart': True, 'bond_timeout': 0.0,
                          'node_names': ['velocity_smoother'], 'use_sim_time': False}]),
    ]
    if get('record_data_dir'):
        nodes.append(Node(package='arena_vln_models', executable='timing_observer',
                          namespace=get('namespace'), parameters=[{'record_data_dir': get('record_data_dir')}]))
        nodes.append(Node(
            package='arena_evaluation', executable='record', name='data_recorder',
            namespace=get('namespace'), arguments=['--dir', get('record_data_dir')],
            parameters=[{'use_sim_time': True, 'local_planner': 'regulated_pure_pursuit',
                         'inter_planner': 'none', 'agent_name': get('agent_name'),
                         'scenario_reset_topic': get('task_generator_node') + '/task_reset',
                         'human_states_topic': get('task_generator_node') + '/human_states',
                         'start_topic': 'episode_start_pose', 'goal_topic': 'episode_goal_pose_metadata'}]))
    return nodes


def generate_launch_description():
    defaults = {'robot': None, 'namespace': None, 'frame': None,
                'task_generator_node': '/task_generator_node', 'record_data_dir': '', 'agent_name': ''}
    return LaunchDescription([
        *(DeclareLaunchArgument(key, **({} if value is None else {'default_value': value}))
          for key, value in defaults.items()),
        OpaqueFunction(function=launch_nodes),
    ])
