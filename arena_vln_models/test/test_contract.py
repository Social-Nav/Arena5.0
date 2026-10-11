import json
from pathlib import Path

import pytest

from arena_vln_models.profile import BenchmarkProfileError, evaluation_defaults, load_profile, validate_profile
from arena_vln_models.session import Session
from arena_vln_models.tracking_config import tracking_parameters


def test_session_filters_old_stop_and_terminal_cannot_reopen():
    state = Session()
    state.reset('first', 0)
    assert not state.receive({'session_id': 'first', 'episode': 0, 'status': 'stop'}, 10.)
    assert state.receive({'session_id': 'first', 'episode': 0, 'status': 'ready'}, 10.)
    assert state.ready(12.5) and not state.ready(12.51)
    state.released = True
    assert state.stop_reason(13.) == 'model_heartbeat_timeout'
    assert state.receive({'session_id': 'first', 'episode': 0, 'status': 'hold'}, 13.)
    assert state.stop_reason(13.) == 'model_hold'
    state.reset('second', 1)
    state.released = True
    assert not state.receive({'session_id': 'first', 'episode': 0, 'status': 'stop'}, 14.)
    assert not state.receive({'session_id': 'second', 'episode': True, 'status': 'stop'}, 14.)
    assert state.receive({'session_id': 'second', 'episode': 1, 'status': 'stop'}, 15.)
    assert state.stop_reason(15.) == 'model_stop'
    assert not state.receive({'session_id': 'second', 'episode': 1, 'status': 'running'}, 16.)


def test_error_is_terminal_and_not_a_model_stop():
    state = Session('current', 3)
    assert state.receive(dict(session_id='current', episode=3, status='error', reason='unavailable'), 1.)
    assert state.stop_reason(1.) == 'model_error'
    assert state.reason == 'unavailable'


@pytest.mark.parametrize('model', ['internnav', 'lightnav', 'navila'])
def test_profiles_share_one_controller(model):
    arena = Path(__file__).resolve().parents[2]
    profile = load_profile(arena / f'arena_bringup/configs/benchmark/profiles/{model}_grscenes.yaml')
    defaults = evaluation_defaults(profile)
    assert defaults['model_backend'] == model
    assert defaults['execution_mode'] == 'nav2_follow_path'
    assert defaults['local_planner'] == 'regulated_pure_pursuit'
    assert defaults['eval_video_debug_overlay_topic'] == ''
    assert profile['external']['command'] == []
    for old in ('direct_cmd_vel', 'lightnav_native_mpc', 'navila_action_velocity', 'adapter_target'):
        assert old not in json.dumps(profile)


def test_old_profiles_require_migration():
    for version in (1, 2):
        with pytest.raises(BenchmarkProfileError, match='migrate'):
            validate_profile({'schema_version': version})


def test_tracking_uses_robot_limits_and_rotation_goal_checker():
    robot = dict(max_vel_x=.65, max_vel_theta=2., velocity_smoother_max_velocity=[.4, 0., 1.5],
                 velocity_smoother_min_velocity=[0., 0., -1.],
                 velocity_smoother_max_accel=[.5, 0., 2.], velocity_smoother_max_decel=[-.5, 0., -2.],
                 footprint='[[.1,.1],[.1,-.1],[-.1,-.1],[-.1,.1]]')
    cfg = tracking_parameters(robot, 'test/')
    controller = cfg['controller_server']['ros__parameters']
    smoother = cfg['velocity_smoother']['ros__parameters']
    assert controller['FollowPath']['desired_linear_vel'] == .4
    assert controller['FollowPath']['rotate_to_heading_angular_vel'] == 1.
    assert controller['goal_checker']['yaw_goal_tolerance'] == .03
    assert controller['goal_checker']['xy_goal_tolerance'] < .25
    assert controller['progress_checker']['plugin'].endswith('PoseProgressChecker')
    assert not controller['FollowPath']['use_collision_detection']
    assert not controller['FollowPath']['use_cost_regulated_linear_velocity_scaling']
    assert not smoother['use_sim_time'] and smoother['velocity_timeout'] == .25
