"""Resolved tracking-only Nav2 parameters; no map or obstacle input."""


def tracking_parameters(robot, frame):
    base = frame + robot.get('robot_base_frame', 'base_link')
    odom = frame + robot.get('robot_odom_frame', 'odom')
    limits = robot['velocity_smoother_max_velocity']
    v = min(float(robot['max_vel_x']), float(limits[0]))
    w = min(float(robot['max_vel_theta']), float(limits[2]),
            -float(robot['velocity_smoother_min_velocity'][2]))
    accel = [float(x) for x in robot['velocity_smoother_max_accel']]
    decel = [float(x) for x in robot['velocity_smoother_max_decel']]
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
        FollowPath={
            'plugin': 'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
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
        })
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
    smoother = dict(bond_heartbeat_period=0.0, use_sim_time=False, enable_stamped_cmd_vel=False,
        smoothing_frequency=20.0, feedback='OPEN_LOOP', scale_velocities=True,
        max_velocity=[v, 0.0, w], min_velocity=[0.0, 0.0, -w],
        max_accel=accel, max_decel=decel, odom_topic='odom',
        velocity_timeout=0.25, deadband_velocity=[0.0, 0.0, 0.0])
    return {'controller_server': {'ros__parameters': controller},
            'local_costmap': {'local_costmap': {'ros__parameters': costmap}},
            'velocity_smoother': {'ros__parameters': smoother}}
