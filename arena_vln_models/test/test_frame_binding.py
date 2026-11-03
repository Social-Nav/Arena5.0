"""Authoritative robot frame binding for the unified executor."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_ai2bot2_authoritative_frames():
    data = yaml.safe_load(
        (ROOT / 'arena_robots/arena_robots/robots/Ai2_Bot2/model_params.yaml').read_text()
    )
    assert data['robot_base_frame'] == 'base_footprint'
    assert data['robot_odom_frame'] == 'odom'


def test_robot_manager_uses_model_params_frames_not_robot_attributes():
    text = (
        ROOT / 'task_generator/task_generator/manager/robot_manager/robot_manager.py'
    ).read_text()
    assert 'self.frame(self._config.model_params.odom_frame)' in text
    assert 'self.frame(self._config.model_params.base_frame)' in text
    # The old attribute fallback would build ``<frame>/base_link`` for Ai2_Bot2.
    assert 'getattr(self._robot, "robot_base_frame"' not in text
    assert 'getattr(self._robot, "robot_odom_frame"' not in text
