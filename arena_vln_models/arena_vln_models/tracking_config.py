"""Resolved tracking-only Nav2 parameters; no map or obstacle input."""

RPP_PLUGIN = 'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController'
SOCIAL_MPC_WRAPPER_PLUGIN = 'nav2_rotation_shim_controller::RotationShimController'
SOCIAL_MPC_PRIMARY_PLUGIN = 'nav2_social_mpc_controller::SocialMPCController'
PROX_MPC_PLUGIN = 'prox_mpc_controller::ProxMpcController'


def _prox_mpc_follow_path(base_frame, cruise_linear_vel=None):
    """FollowPath block for the A-track ProxMPC identity (RGB + real LiDAR).

    Parameters mirror ``src/deps/prox_mpc/prox_mpc_controller/config/
    prox_mpc_controller.yaml``.  The only deviations are Arena-specific:
    ``robot_radius`` comes from the Ai2_Bot2 footprint, the cruise target is
    0.8 m/s, and ``costmap_cost_threshold`` keeps the source default 200
    (the demo override 253 is recorded in the manifest, not copied blindly).
    """
    desired_cruise = 0.8 if cruise_linear_vel is None else float(cruise_linear_vel)
    v_max = max(desired_cruise, 0.0)
    return {
        'plugin': PROX_MPC_PLUGIN,
        'model_plugin': 'prox_mpc_core/Unicycle',
        'model_params': {
            'L': 1.6,
            'v_max': v_max,
            'v_min': -0.1,
        },
        'np': 20,
        'nc': 20,
        'dt': 0.1,
        'desired_linear_vel': desired_cruise,
        'curvature_gain': 0.0,
        'allow_reversing': False,
        'reverse_from_plan_orientation': False,
        'direction_switch_standstill_speed_mps': 0.05,
        'direction_switch_dwell_s': 0.5,
        'goal_settle_hysteresis_m': 0.10,
        'brake_period_s': 0.0,
        'q_pos': 10.0,
        'q_theta': 1.0,
        's_factor': 2.0,
        'r_weight': 0.1,
        'w_weight': 1000.0,
        'max_int_iter_qp': 1500,
        'max_ext_iter_qp': 10000,
        'max_iter_sqp': 1,
        'max_solve_time': 0.0,
        'qp_type': False,
        'guess': True,
        'warm_start': True,
        'max_solver_failures': 3,
        'max_obstacles': 4,
        'safety_margin': 0.1,
        'robot_radius': 0.48,
        'cbf_gamma': 1.0,
        'costmap_cost_threshold': 200,
        'obstacle_cluster_radius': 0.3,
        'max_obstacle_scan_cells': 50,
        'predict_obstacles': True,
        'obstacle_topic': 'tracked_obstacles',
        # Bounded by the measured RTX scan period, not by the configured
        # 20 Hz.  Live canary: 33 scans / 18.12 sim s (~1.82 Hz, period
        # ~0.549 s), tracker ages up to 0.55 s.  Two periods is the bound;
        # the episode receipt records max stale age and fallback cycles.
        'obstacle_timeout': 1.2,
        'dynamic_speed_threshold': 0.1,
        'prediction_uncertainty_growth': 0.0,
        'prediction_forward_shadow_s': 0.0,
        'obstacle_yield_band_m': 0.0,
        'obstacle_yield_caps_speed': False,
        'max_dynamic_obstacles': 2,
        'max_dynamic_obstacle_radius': 0.0,
        'publish_diagnostics': True,
        'log_level': 'info',
    }


def _prox_mpc_costmap(base_frame, sensor_frame, scan_topic=None):
    """Rolling geometry costmap fed only by the real Isaac LaserScan.

    No static/oracle map, no semantic layer, no GT people.  ``track_unknown_space``
    stays true so unobserved cells are NO_INFORMATION; ProxMPC skips
    NO_INFORMATION by source design, so this config alone does not turn unknown
    into an obstacle.  The lane records that fact and relies on the scan-freshness
    guard, never on an assumption that unknown is free.

    ``scan_topic`` must be the absolute robot-namespaced topic when supplied:
    the local_costmap node lives at ``<robot>/local_costmap``, so a relative
    ``laser_scan`` would resolve to ``<robot>/local_costmap/laser_scan`` and
    never receive the real scan.
    """
    return {
        'plugins': ['obstacle_layer', 'inflation_layer'],
        'obstacle_layer': {
            'plugin': 'nav2_costmap_2d::ObstacleLayer',
            'enabled': True,
            'combination_method': 1,
            'observation_sources': 'scan',
            'scan': {
                # The Ai2_Bot2 Isaac bridge publishes the RTX LaserScan at
                # <namespace>/laser_scan (USD topicName=/laser_scan + nodeNamespace),
                # not <namespace>/scan.  Verified live: 1600 beams, finite ranges.
                # The caller passes the absolute robot-namespaced name; the
                # fallback climbs one namespace from <robot>/local_costmap.
                'topic': scan_topic or '../laser_scan',
                'sensor_frame': sensor_frame,
                'data_type': 'LaserScan',
                'marking': True,
                'clearing': True,
                'max_obstacle_height': 2.0,
                'raytrace_max_range': 8.0,
                'raytrace_min_range': 0.0,
                'obstacle_max_range': 3.0,
                'obstacle_min_range': 0.0,
                'observation_persistence': 0.0,
            },
        },
        'inflation_layer': {
            'plugin': 'nav2_costmap_2d::InflationLayer',
            'cost_scaling_factor': 3.0,
            'inflation_radius': 0.55,
        },
    }


def _social_mpc_follow_path(base_frame, cruise_linear_vel=None):
    """FollowPath block for the LISN social_mpc identity.

    Mirrors ``arena_simulation_setup/configs/nav2/controllers/social_mpc/
    controller_config.yaml`` (RotationShim wraps SocialMPC).  The default
    ``nav2_follow_path_rpp`` identity is untouched: this block is selected only
    when the caller explicitly asks for ``local_planner=social_mpc``.

    ``cruise_linear_vel`` overrides only the trajectorizer's cruise TARGET;
    the optimizer's ``max_linear_velocity`` remains the hard ceiling.
    """
    desired_cruise = 0.35 if cruise_linear_vel is None else float(cruise_linear_vel)
    return {
        # LISN uses SocialMPC directly.  The RotationShim wrapper is the only
        # in-place rotation mechanism and it kept the robot turning for the
        # whole 10s slow-loop window while subgoals changed; the optimizer
        # itself drives the path and turns while moving.
        'plugin': SOCIAL_MPC_PRIMARY_PLUGIN,
        'primary_controller': SOCIAL_MPC_PRIMARY_PLUGIN,
        # LISN uses all real HuNav people around the robot, not only the front
        # 45-degree cone; the costmap still rejects out-of-map people.
        'fov_angle': 3.14159,
        'angular_dist_threshold': 3.14,
        'angular_disengage_threshold': 0.1745,
        'forward_sampling_distance': 0.5,
        'rotate_to_heading_angular_vel': 0.8,
        'max_angular_accel': 3.2,
        'simulate_ahead_time': 1.0,
        'rotate_to_goal_heading': False,
        'closed_loop': True,
        'transform_tolerance': 1.0,
        'trajectorizer': {
            'omnidirectional': False,
            'desired_linear_vel': desired_cruise,
            'lookahead_dist': 0.3,
            'max_angular_vel': 1.5,
            'rotate_to_heading_min_angle': 3.14,
            'base_frame': base_frame,
            'time_step': 0.05,
            'max_time': 1.2,
        },
        'optimizer': {
            'linear_solver_type': 'DENSE_SCHUR',
            'param_tol': 1.0e-9,
            'fn_tol': 1.0e-5,
            'gradient_tol': 1.0e-8,
            'max_iterations': 50,
            'solver_time_budget': 0.05,
            'max_angular_velocity': 1.4,
            'min_linear_velocity': -0.1,
            'max_linear_velocity': 0.8,
            'control_horizon': 24,
            'parameter_block_length': 6,
            'discretization': 1,
            'debug_optimizer': False,
            'current_path_weight': 1.0,
            'current_cmds_weight': 0.3,
            'weights': {
                'distance_weight': 50.0,
                'social_weight': 700.0,
                'proxemics_weight': 150.0,
                'proxemics_d0': 1.0,
                'proxemics_alpha': 3.0,
                'velocity_weight': 40.0,
                'angle_weight': 60.0,
                'agent_angle_weight': 0.0,
                'velocity_feasibility_weight': 5.0,
                'goal_align_weight': 8.0,
                'obstacle_weight': 0.05,
                'social_clear_distance': 2.0,
                'social_safety_distance': 0.9,
                'social_mid_gain': 1.0,
                'social_near_gain': 6.0,
            },
        },
    }


def tracking_parameters(robot, frame, local_planner='regulated_pure_pursuit',
                        cruise_linear_vel=None, scan_topic=None):
    base = frame + robot.get('robot_base_frame', 'base_link')
    odom = frame + robot.get('robot_odom_frame', 'odom')
    limits = robot['velocity_smoother_max_velocity']
    v = min(float(robot['max_vel_x']), float(limits[0]))
    w = min(float(robot['max_vel_theta']), float(limits[2]),
            -float(robot['velocity_smoother_min_velocity'][2]))
    accel = [float(x) for x in robot['velocity_smoother_max_accel']]
    decel = [float(x) for x in robot['velocity_smoother_max_decel']]
    planner = str(local_planner or '').strip()
    if planner == 'social_mpc':
        follow_path = _social_mpc_follow_path(base, cruise_linear_vel)
        # social_mpc's optimizer owns the vx ceiling (0.8); the canonical
        # controller config documents it must stay <= velocity_smoother
        # max_velocity[0].  RPP instead keeps the min(max_vel_x, limit) target.
        smoother_v = float(limits[0])
    elif planner == 'prox_mpc':
        follow_path = _prox_mpc_follow_path(base, cruise_linear_vel)
        # ProxMPC's model v_max and the smoother ceiling must agree.  The
        # robot config already caps the smoother at 0.8; keep that exact value.
        smoother_v = float(limits[0])
    else:
        smoother_v = v
        follow_path = {
            'plugin': RPP_PLUGIN,
            'desired_linear_vel': v, 'min_approach_linear_velocity': 0.05,
            'approach_velocity_scaling_dist': 0.2,
            'rotate_to_heading_angular_vel': min(w, 1.5), 'max_angular_accel': accel[2],
            'use_velocity_scaled_lookahead_dist': True, 'lookahead_time': 1.5,
            'min_lookahead_dist': 0.4, 'max_lookahead_dist': 1.2,
            'use_rotate_to_heading': True, 'rotate_to_heading_min_angle': 0.5,
            'use_regulated_linear_velocity_scaling': True,
            'regulated_linear_scaling_min_radius': 0.6,
            'regulated_linear_scaling_min_speed': min(v, 0.2),
            'use_cost_regulated_linear_velocity_scaling': False,
            'use_collision_detection': False, 'use_cancel_deceleration': False,
            'allow_reversing': False, 'transform_tolerance': 0.3,
            'max_robot_pose_search_dist': 10.0,
        }
    controller = dict(use_sim_time=True, controller_frequency=20.0, bond_heartbeat_period=0.0,
        enable_stamped_cmd_vel=False, odom_topic='odom',
        progress_checker_plugins=['progress_checker'], goal_checker_plugins=['goal_checker'],
        controller_plugins=['FollowPath'], publish_zero_velocity=True,
        failure_tolerance=0.0, costmap_update_timeout=2.0,
        progress_checker={'plugin': 'nav2_controller::PoseProgressChecker',
                          'required_movement_radius': 0.05, 'required_movement_angle': 0.1,
                          'movement_time_allowance': 20.0},
        goal_checker={'plugin': 'nav2_controller::SimpleGoalChecker',
                      'xy_goal_tolerance': 0.03, 'yaw_goal_tolerance': 0.03, 'stateful': False},
        FollowPath=follow_path)
    costmap = dict(use_sim_time=True, update_frequency=20.0, publish_frequency=2.0,
        global_frame=odom, robot_base_frame=base, rolling_window=True,
        width=15, height=15, resolution=0.1, track_unknown_space=False,
        # Jazzy's YAML parser rejects an untyped empty array. A disabled layer
        # supplies a typed plugin list without adding any obstacle information.
        plugins=['tracking_background'],
        tracking_background={'plugin': 'nav2_costmap_2d::InflationLayer', 'enabled': False,
                             'inflation_radius': 0.55},
        always_send_full_costmap=True, transform_tolerance=0.5,
        footprint=robot['footprint'])
    if planner == 'prox_mpc':
        # Real Isaac LaserScan -> ObstacleLayer marking/clearing + Inflation.
        # No static/oracle map, no HuNav people, no semantic grid.
        sensor_frame = frame + robot.get('robot_sensor_frame', 'lidar_frame')
        costmap.update(_prox_mpc_costmap(base, sensor_frame, scan_topic=scan_topic))
        costmap['track_unknown_space'] = True
    elif planner == 'social_mpc':
        # LISN-only graded semantic cost.  The layer consumes the sidecar's
        # same-source semantic grid; real people remain the optimizer's HuNav
        # tracks and are never synthesized here.
        costmap['plugins'] = ['tracking_background', 'semantic_layer']
        costmap['semantic_layer'] = {
            'plugin': 'arena_semantic_costmap_plugin::SemanticLayer',
            # Start disabled so the controller activation does not race the
            # semantic grid; the lane enables it after the episode-start barrier.
            'enabled': False,
            'semantic_enabled': True,
            'semantic_topic': '/task_generator_node/Ai2_Bot2/lisn_semantic_cost',
            'semantic_ttl_sec': 35.0,
            'status_topic': '/lisn_semantic_layer_status',
        }
    smoother = dict(bond_heartbeat_period=0.0, use_sim_time=False, enable_stamped_cmd_vel=False,
        smoothing_frequency=20.0, feedback='OPEN_LOOP', scale_velocities=True,
        max_velocity=[smoother_v, 0.0, w], min_velocity=[0.0, 0.0, -w],
        max_accel=accel, max_decel=decel, odom_topic='odom',
        velocity_timeout=0.25, deadband_velocity=[0.0, 0.0, 0.0])
    return {'controller_server': {'ros__parameters': controller},
            'local_costmap': {'local_costmap': {'ros__parameters': costmap}},
            'velocity_smoother': {'ros__parameters': smoother}}
