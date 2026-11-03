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
    local_planner = get('local_planner')
    lisn_grounding_live = os.environ.get('ARENA_LISN_GROUNDING_LIVE', '0').strip().lower() in {
        '1', 'true', 'yes', 'on'
    }
    lisn_grounded_goal_topic = os.environ.get('ARENA_LISN_GROUNDED_GOAL_TOPIC', '').strip()
    cruise_env = os.environ.get('ARENA_LISN_CRUISE_LINEAR_VEL', '').strip()
    cruise_linear_vel = float(cruise_env) if cruise_env else None
    prox_scan_topic = (
        str(get('namespace')).rstrip('/') + '/laser_scan'
        if local_planner == 'prox_mpc' else None
    )
    params = YAMLFileSubstitution.from_dict(
        {get('namespace'): tracking_parameters(
            robot, get('frame'), local_planner,
            cruise_linear_vel=cruise_linear_vel,
            scan_topic=prox_scan_topic,
        )}
    )
    params_file = params.perform(context)
    smoother_remaps = [('cmd_vel', 'cmd_vel_nav')]
    if local_planner != 'prox_mpc':
        # Legacy LISN/SocialMPC path: the smoother output is the final cmd_vel.
        # Prox inserts the real-scan collision_monitor between smoother and base.
        smoother_remaps.append(('cmd_vel_smoothed', 'cmd_vel'))
    nodes = [
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             namespace=get('namespace'), parameters=[params_file], output='screen',
             remappings=[('cmd_vel', 'cmd_vel_nav')]),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
             namespace=get('namespace'), parameters=[params_file], output='screen',
             remappings=smoother_remaps),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_smoother', namespace=get('namespace'), output='screen',
             parameters=[{'autostart': True, 'bond_timeout': 0.0,
                          'node_names': ['velocity_smoother'], 'use_sim_time': False}]),
    ]
    if get('model_adapter'):
        # Exactly one in-process executor.  When model_adapter is empty the
        # legacy external-client flow keeps owning FollowPath goals; this avoids
        # two senders racing for the same action server.
        nodes.append(Node(
            package='arena_vln_models', executable='model_executor', name='arena_model_executor',
            namespace=get('namespace'), output='screen',
            parameters=[{
                'model_adapter': get('model_adapter'),
                'control_mode': get('control_mode'),
                'task_namespace': get('task_generator_node'),
                'rgb_topic': get('rgb_topic'),
                'depth_topic': get('depth_topic'),
                'camera_info_topic': get('camera_info_topic'),
                'odom_topic': get('odom_topic'),
                'odom_frame': get('odom_frame'),
                'base_frame': get('base_frame'),
                'use_tf_exact_stamp': get('use_tf_exact_stamp').lower() in {'1', 'true', 'yes', 'on'},
                'lisn_grounding_live': lisn_grounding_live,
                'lisn_grounded_goal_topic': lisn_grounded_goal_topic,
                'fast_guard_veto_topic': get('fast_guard_veto_topic'),
                'fast_guard_veto_timeout_sec': float(get('fast_guard_veto_timeout_sec')),
                'fast_guard_required': get('fast_guard_required').lower() in {'1', 'true', 'yes', 'on'},
                'eval_ready_topic': get('task_generator_node') + '/eval_ready',
                'task_reset_topic': get('task_generator_node') + '/task_reset',
                'instruction_topic': get('task_generator_node') + '/vln_instruction',
                'model_status_topic': 'model/status',
                'follow_path_action': 'follow_path',
                'inference_hz': float(get('inference_hz')),
                'request_timeout_sec': float(get('request_timeout_sec')),
                'observation_max_age_sec': float(get('observation_max_age_sec')),
                'server_url': get('model_server_url'),
                'trace_dir': get('model_trace_dir'),
                'model_params_json': get('model_params_json'),
            }],
        ))
    if get('fast_guard_enabled').lower() in {'1', 'true', 'yes', 'on'}:
        veto_topic = get('fast_guard_veto_topic')
        if not veto_topic:
            raise RuntimeError('fast_guard_enabled requires fast_guard_veto_topic')
        rgb_topic = get('rgb_topic')
        if not rgb_topic.startswith('/'):
            rgb_topic = get('namespace').rstrip('/') + '/' + rgb_topic.lstrip('/')
        decision_topic = get('bfast_decision_topic') or (
            get('namespace').rstrip('/') + '/bfast/decision')
        nodes.append(Node(
            package='arena_bringup', executable='bfast_guard', name='bfast_guard',
            namespace=get('namespace'), output='screen',
            parameters=[{
                'rgb_topic': rgb_topic,
                'veto_topic': veto_topic,
                'decision_topic': decision_topic,
                'guard_enabled': True,
                'risk_threshold': float(get('bfast_risk_threshold')),
                'max_frame_gap_sec': float(get('bfast_max_frame_gap_sec')),
            }],
        ))
    if local_planner == 'prox_mpc':
        # A-track: real LiDAR -> generic cluster tracks -> ProxMPC solver.
        # The tracker is a self-bring-up lifecycle node; no external lifecycle
        # manager is needed.  It never emits human labels or People.
        nodes.append(Node(
            package='prox_mpc_obstacle_tracker', executable='obstacle_tracker',
            name='prox_mpc_obstacle_tracker', namespace=get('namespace'), output='screen',
            parameters=[{
                'scan_topic': get('scan_topic'),
                'output_topic': 'tracked_obstacles',
                'tracking_frame': get('odom_frame') or 'odom',
                'max_detection_range': 3.0,
                'max_cluster_radius': 0.6,
                'cluster_center_offset_gain': 0.5,
                'confirm_count': 3,
                'drop_count': 3,
                'max_tracks': 10,
                'prediction_steps': 25,
                'prediction_dt': 0.1,
                'log_level': 'info',
            }],
            arguments=['--ros-args', '--log-level', 'prox_mpc_obstacle_tracker:=info'],
        ))
        # Independent real-scan safety backstop.  It sits between the smoother
        # and the base so a stale/missing lidar or a stop polygon gates motion.
        base = str(get('frame')) + robot.get('robot_base_frame', 'base_link')
        odom = str(get('frame')) + robot.get('robot_odom_frame', 'odom')
        source_names = list(robot.get('collision_monitor_sources', []) or [])
        source_dict = {}
        for name in source_names:
            source = dict((robot.get('observation_sources_dict') or {}).get(name) or {})
            if not source:
                continue
            if 'scan' in str(source.get('topic', '')):
                source['topic'] = prox_scan_topic
            # nav2_collision_monitor selects the source class from `type`, not
            # the costmap's `data_type`; without this the node fails configure
            # and the lifecycle barrier times out.
            source['type'] = 'scan'
            source['enabled'] = True
            source_dict[name] = source
        collision_params = {
            'enabled': True,
            'observation_sources': source_names,
            **source_dict,
            'polygons': robot.get('polygons', []),
            **(robot.get('polygons_dict') or {}),
            'base_frame_id': base,
            'odom_frame_id': odom,
            'cmd_vel_in_topic': 'cmd_vel_smoothed',
            'cmd_vel_out_topic': 'cmd_vel',
            'use_sim_time': True,
            # Same measured-period bound as Prox obstacle_timeout (~2 x 0.549 s).
            'source_timeout': 1.2,
            'stop_pub_timeout': 0.2,
            'transform_tolerance': 1.0,
            'base_shift_correction': True,
            'state_topic': 'collision_monitor_state',
        }
        nodes.append(Node(
            package='nav2_collision_monitor', executable='collision_monitor',
            name='collision_monitor', namespace=get('namespace'), output='screen',
            parameters=[collision_params],
        ))
        nodes.append(Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_collision_monitor', namespace=get('namespace'),
            output='screen',
            parameters=[{
                'autostart': True, 'bond_timeout': 0.0,
                'node_names': ['collision_monitor'], 'use_sim_time': True,
            }],
        ))
    if get('record_data_dir'):
        # Keep the timing summary honest about which control path produced it;
        # a LISN SocialMPC/ProxMPC run must never be recorded as RPP.
        timing_control_path = (
            'nav2_follow_path_prox_mpc' if local_planner == 'prox_mpc'
            else 'nav2_follow_path_social_mpc'
            if lisn_grounding_live or local_planner == 'social_mpc'
            else 'nav2_follow_path_rpp'
        )
        nodes.append(Node(package='arena_vln_models', executable='timing_observer',
                          namespace=get('namespace'), parameters=[{
                              'record_data_dir': get('record_data_dir'),
                              'execution_mode': 'nav2_follow_path',
                              'control_path': timing_control_path,
                          }]))
        nodes.append(Node(
            package='arena_vln_models', executable='cmd_vel_trace', name='task34_cmd_vel_trace',
            namespace=get('namespace'), output='screen',
            parameters=[{'record_data_dir': get('record_data_dir'),
                         'namespace': get('namespace'), 'use_sim_time': True}]))
        nodes.append(Node(
            package='arena_evaluation', executable='record', name='data_recorder',
            namespace=get('namespace'), arguments=['--dir', get('record_data_dir')],
            parameters=[{'use_sim_time': True, 'local_planner': local_planner,
                         'inter_planner': 'none', 'agent_name': get('agent_name'),
                         'scenario_reset_topic': get('task_generator_node') + '/task_reset',
                         'human_states_topic': get('task_generator_node') + '/human_states',
                         'start_topic': 'episode_start_pose', 'goal_topic': 'episode_goal_pose_metadata'}]))
    return nodes


def generate_launch_description():
    defaults = {
        'robot': None, 'namespace': None, 'frame': None,
        'task_generator_node': '/task_generator_node', 'record_data_dir': '', 'agent_name': '',
        # Tracking identity selected by the task_generator. Default keeps the
        # historical regulated_pure_pursuit behaviour; LISN explicitly passes
        # social_mpc and gets RotationShim + SocialMPC instead.
        'local_planner': 'regulated_pure_pursuit',
        # Optional unified in-process executor. Empty keeps the existing
        # external-client flow unchanged.
        'model_adapter': '', 'control_mode': 'arena_follow_path',
        'rgb_topic': 'head_camera/image', 'depth_topic': 'head_camera/depth',
        'camera_info_topic': 'head_camera/camera_info', 'odom_topic': 'odom',
        'scan_topic': 'laser_scan',
        'inference_hz': '4.0', 'request_timeout_sec': '10.0',
        'observation_max_age_sec': '2.0', 'model_server_url': '',
        'model_trace_dir': '', 'model_params_json': '{}',
        'odom_frame': '', 'base_frame': '', 'use_tf_exact_stamp': 'true',
        # Strict RGB-only B fast guard.  Empty veto topic keeps A unchanged.
        'fast_guard_enabled': 'false', 'fast_guard_veto_topic': '',
        'fast_guard_veto_timeout_sec': '1.0', 'fast_guard_required': 'false',
        'bfast_decision_topic': '', 'bfast_risk_threshold': '0.5',
        'bfast_max_frame_gap_sec': '1.0',
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(key, **({} if value is None else {'default_value': value}))
          for key, value in defaults.items()),
        OpaqueFunction(function=launch_nodes),
    ])
