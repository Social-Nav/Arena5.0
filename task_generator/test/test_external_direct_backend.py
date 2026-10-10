import asyncio
from types import SimpleNamespace

import pytest

from task_generator.manager.robot_manager.robot_manager import RobotManager


@pytest.mark.parametrize('backend,expected', [
    ('lightnav', '/task_generator_node/Ai2_Bot2/lightnav/status'),
    ('navila', '/task_generator_node/Ai2_Bot2/navila/status'),
    ('internnav', '/task_generator_node/Ai2_Bot2/internnav/status'),
])
def test_backend_status_selection_overrides_the_nonempty_legacy_default(monkeypatch, backend, expected):
    monkeypatch.setenv('ARENA_MODEL_BACKEND', backend)
    monkeypatch.delenv('ARENA_MODEL_STATUS_TOPIC', raising=False)
    subscribed = []
    manager = SimpleNamespace(
        _dual_vln_status_subscription=None,
        _is_dual_vln_robot=lambda: True,
        _get_compat_rosparam=lambda *a, **k: '/task_generator_node/Ai2_Bot2/internnav/status',
        namespace=lambda *parts: '/task_generator_node/Ai2_Bot2/' + '/'.join(parts),
        _on_dual_vln_status=lambda msg: None,
        node=SimpleNamespace(create_subscription=lambda kind, topic, callback, qos: subscribed.append(topic)),
    )
    RobotManager._ensure_dual_vln_status_subscription(manager)
    assert subscribed == [expected]
    assert manager._dual_vln_status_topic == expected


def test_direct_clients_do_not_wait_for_absent_nav2_services():
    manager = SimpleNamespace(_internnav_direct_cmd_vel_enabled=lambda: True)
    assert asyncio.run(RobotManager._clear_global_costmap(manager))
    assert asyncio.run(RobotManager.set_social_attributes(manager, 'neutral'))
    assert asyncio.run(RobotManager.set_desired_linear_vel(manager, .3))


@pytest.mark.parametrize('mode', ['external_direct_cmd_vel', 'legacy_internnav', 'nav2'])
def test_robot_setup_gives_direct_clients_exclusive_velocity_publisher(monkeypatch, mode):
    from unittest.mock import AsyncMock
    import task_generator.manager.robot_manager.robot_manager as module

    monkeypatch.setenv('ARENA_MODEL_EXECUTION_MODE', mode)
    monkeypatch.setenv('ARENA_INTERNNAV_DIRECT_CMD_VEL', '1' if mode == 'legacy_internnav' else '0')
    monkeypatch.setattr(module.rclpy.action, 'ActionClient', lambda *a: object())
    topics = []
    robot = SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(z=0.)))
    manager = SimpleNamespace(
        _robot=robot, _robot_radius=.25, _cmd_vel_pub=None,
        _config=SimpleNamespace(model_params=SimpleNamespace(z_offset=0.)),
        _environment_manager=SimpleNamespace(spawn_robot=AsyncMock(return_value=[robot])),
        namespace=lambda *parts: '/robot/' + '/'.join(parts),
        _get_compat_rosparam=lambda *a: False,
        _stop_vel_timer_cb=lambda: None, _robot_pos_callback=lambda msg: None,
        _goal_status_callback=lambda msg: None,
        _setup_camera_readiness_subscriptions=lambda: None,
        _ensure_dual_vln_status_subscription=lambda: None,
        _launch_robot=AsyncMock(),
        node=SimpleNamespace(
            create_publisher=lambda kind, topic, qos: topics.append(topic) or object(),
            create_subscription=lambda *a: None, create_timer=lambda *a: None,
            rosparam={float: {}}),
    )
    manager._internnav_direct_cmd_vel_enabled = lambda: RobotManager._internnav_direct_cmd_vel_enabled(manager)
    asyncio.run(RobotManager.set_up_robot(manager, set()))
    assert ('/robot/cmd_vel' in topics) == (mode == 'nav2')
    assert (manager._cmd_vel_pub is None) == (mode != 'nav2')
