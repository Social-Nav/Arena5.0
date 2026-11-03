import asyncio
import time
from types import SimpleNamespace

import pytest

from arena_rclpy_mixins.shared import Namespace
from arena_vln_models.session import Session
from task_generator.manager.robot_manager.robot_manager import RobotManager
from task_generator.node import TaskGenerator


def _task_generator_with_lifecycle(lifecycle):
    robot = SimpleNamespace(
        namespace=Namespace('/task_generator_node/Ai2_Bot2'),
        _waypoint_lifecycle=lifecycle,
    )
    node = TaskGenerator.__new__(TaskGenerator)
    node._robots_manager = SimpleNamespace(robots={'Ai2_Bot2': robot})
    return node


def test_follow_path_barrier_requires_current_heartbeat_and_active_controller():
    lifecycle = SimpleNamespace(
        session=Session('current', 0),
        controller_active=True,
    )
    node = _task_generator_with_lifecycle(lifecycle)
    channels = node._model_channels()

    assert channels == {
        'command_services': [],
        'status_topics': ['/task_generator_node/Ai2_Bot2/model/status'],
    }
    assert node._model_ready_state(channels)['ready'] is False
    assert lifecycle.session.receive(
        {'session_id': 'current', 'episode': 0, 'status': 'ready'},
        time.monotonic(),
    )
    assert node._model_ready_state(channels)['ready'] is True
    lifecycle.controller_active = False
    assert node._model_ready_state(channels)['ready'] is False


def test_follow_path_barrier_surfaces_model_error():
    lifecycle = SimpleNamespace(
        session=Session('current', 0),
        controller_active=True,
    )
    node = _task_generator_with_lifecycle(lifecycle)
    lifecycle.session.receive(
        {
            'session_id': 'current',
            'episode': 0,
            'status': 'error',
            'reason': 'client unavailable',
        },
        time.monotonic(),
    )

    with pytest.raises(RuntimeError, match='client unavailable'):
        node._model_ready_state(node._model_channels())


def test_robot_manager_propagates_follow_path_terminal_reason():
    manager = RobotManager.__new__(RobotManager)
    session = Session('current', 0)
    session.released = True
    session.receive(
        {'session_id': 'current', 'episode': 0, 'status': 'error', 'reason': 'failed'},
        time.monotonic(),
    )

    async def poll():
        return 'model_error'

    manager._waypoint_lifecycle = SimpleNamespace(
        session=session,
        poll=poll,
        error='',
    )
    manager._model_declared_stop = False
    manager._model_declared_stop_payload = {}
    manager._is_goal_reached = False
    manager._NodeInterface__node = SimpleNamespace(
        _publish_eval_ready=lambda *args, **kwargs: None,
    )

    assert asyncio.run(manager.is_done) is True
    assert manager.done_reason == 'model_error'


def test_robot_manager_never_reports_heartbeat_loss_as_goal_reached():
    manager = RobotManager.__new__(RobotManager)
    session = Session('current', 0)
    session.released = True

    async def poll():
        return 'model_heartbeat_timeout'

    manager._waypoint_lifecycle = SimpleNamespace(
        session=session,
        poll=poll,
        error='',
    )
    manager._model_declared_stop = False
    manager._model_declared_stop_payload = {}
    manager._is_goal_reached = False
    manager._NodeInterface__node = SimpleNamespace(
        _publish_eval_ready=lambda *args, **kwargs: None,
    )

    assert asyncio.run(manager.is_done) is True
    assert manager.done_reason == 'control_error'
    assert manager._model_declared_stop_payload['detail'] == 'model_heartbeat_timeout'
