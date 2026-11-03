"""FollowPath controller identity must follow the declared local_planner.

The native ``nav2_follow_path`` tracking path used to hardcode
RegulatedPurePursuit, so a LISN run that declared ``social_mpc`` still loaded
RPP.  These tests pin both sides of the contract:

* the default identity is still byte-for-byte RPP;
* ``local_planner=social_mpc`` selects RotationShim + SocialMPC and exposes the
  ``optimizer.max_linear_velocity`` hot binding the slow loop applies.
"""

from pathlib import Path

from arena_vln_models.tracking_config import tracking_parameters

ROBOT = {
    'robot_base_frame': 'base_footprint',
    'robot_odom_frame': 'odom',
    'max_vel_x': 0.4,
    'max_vel_theta': 1.5,
    'velocity_smoother_max_velocity': [0.8, 0.0, 1.5],
    'velocity_smoother_min_velocity': [0.0, 0.0, -1.5],
    'velocity_smoother_max_accel': [2.5, 0.0, 3.2],
    'velocity_smoother_max_decel': [-2.5, 0.0, -3.2],
    'footprint': '[[0.42, 0.48], [0.42, -0.48], [-0.42, -0.48], [-0.42, 0.48]]',
}


def _follow_path(local_planner):
    params = tracking_parameters(ROBOT, 'Ai2_Bot2/', local_planner)
    controller = params['controller_server']['ros__parameters']
    assert controller['controller_plugins'] == ['FollowPath']
    return controller['FollowPath']


def test_default_identity_is_still_rpp():
    follow_path = _follow_path('regulated_pure_pursuit')
    assert follow_path['plugin'] == (
        'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController'
    )
    assert 'primary_controller' not in follow_path
    assert follow_path['desired_linear_vel'] == 0.4


def test_omitted_local_planner_keeps_rpp():
    params = tracking_parameters(ROBOT, 'Ai2_Bot2/')
    follow_path = params['controller_server']['ros__parameters']['FollowPath']
    assert follow_path['plugin'] == (
        'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController'
    )


def test_social_mpc_selects_controller_directly_for_lisn():
    follow_path = _follow_path('social_mpc')
    assert follow_path['plugin'] == 'nav2_social_mpc_controller::SocialMPCController'
    assert follow_path['primary_controller'] == 'nav2_social_mpc_controller::SocialMPCController'
    assert follow_path['optimizer']['max_linear_velocity'] == 0.8
    assert follow_path['optimizer']['max_angular_velocity'] == 1.4
    assert follow_path['trajectorizer']['base_frame'] == 'Ai2_Bot2/base_footprint'


def test_social_mpc_velocity_bound_stays_within_smoother_limit():
    follow_path = _follow_path('social_mpc')
    smoother = tracking_parameters(ROBOT, 'Ai2_Bot2/', 'social_mpc')[
        'velocity_smoother'
    ]['ros__parameters']
    assert follow_path['optimizer']['max_linear_velocity'] <= smoother['max_velocity'][0]
    assert follow_path['optimizer']['max_angular_velocity'] <= smoother['max_velocity'][2]


def test_social_mpc_cruise_override_changes_target_not_ceiling():
    follow_path = tracking_parameters(
        ROBOT, 'Ai2_Bot2/', 'social_mpc', cruise_linear_vel=0.8
    )['controller_server']['ros__parameters']['FollowPath']
    assert follow_path['trajectorizer']['desired_linear_vel'] == 0.8
    assert follow_path['optimizer']['max_linear_velocity'] == 0.8
    default = _follow_path('social_mpc')
    assert default['trajectorizer']['desired_linear_vel'] == 0.35


def test_follow_path_launch_forwards_local_planner():
    source = (
        Path(__file__).resolve().parents[1] / 'launch' / 'follow_path.launch.py'
    ).read_text()
    assert "'local_planner': 'regulated_pure_pursuit'" in source
    assert 'cruise_linear_vel=cruise_linear_vel' in source
    assert "'local_planner': local_planner" in source
