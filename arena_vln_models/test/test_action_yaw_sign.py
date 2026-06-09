"""
Yaw sign and action direction consistency tests.

Verifies that discrete action mapping, trajectory synthesis, and heuristic
control all agree on which direction is "left" vs "right" in ROS coordinates.

ROS convention: +angular_z = counter-clockwise = LEFT turn
                 -angular_z = clockwise = RIGHT turn

InternNav native convention (camera frame):
  action 2 = native "turn_left"  → +angular_z in generic discrete mapping
  action 3 = native "turn_right" → -angular_z in generic discrete mapping

Arena official-discrete primitive convention (Isaac/Ai2_Bot2):
  action 2 = native "turn_left"  → -cmd_vel angular_z, observed as +odom yaw
  action 3 = native "turn_right" → +cmd_vel angular_z, observed as -odom yaw

The mapping is hard-coded (no invert_discrete_turns parameter) to match the
trajectory policy's passthrough yaw sign.  A mismatched sign between discrete
and trajectory policies was the root cause of yaw_sign_mismatch_ratio reaching
0.73–0.96 in earlier eval runs.
"""

import math
import sys
from types import SimpleNamespace

import numpy as np

# Install ROS stubs if needed (same pattern as test_arena_vln_models.py)
try:
    import rclpy  # noqa: F401
except ModuleNotFoundError:
    import types as _types
    _ns = SimpleNamespace
    _rclpy = _types.ModuleType('rclpy')
    _rclpy_node = _types.ModuleType('rclpy.node')
    _rclpy_node.Node = type('Node', (), {})
    _rclpy_param = _types.ModuleType('rclpy.parameter')
    _rclpy_param.Parameter = lambda n, v=None: SimpleNamespace(name=n, value=v)
    _rclpy_qos = _types.ModuleType('rclpy.qos')
    _rclpy_qos.DurabilityPolicy = _ns(TRANSIENT_LOCAL=1, VOLATILE=2)
    _rclpy_qos.ReliabilityPolicy = _ns(RELIABLE=1, BEST_EFFORT=2)
    _rclpy_qos.QoSProfile = lambda *a, **kw: _ns(**kw)
    _geom = _types.ModuleType('geometry_msgs')
    _geom_msg = _types.ModuleType('geometry_msgs.msg')
    _geom_msg.PoseStamped = type('PoseStamped', (), {})
    _geom_msg.Twist = type('Twist', (), {'__init__': lambda s: setattr(s, 'linear', _ns(x=0.0)) or setattr(s, 'angular', _ns(z=0.0))})
    _nav = _types.ModuleType('nav_msgs')
    _nav_msg = _types.ModuleType('nav_msgs.msg')
    _nav_msg.Odometry = type('Odometry', (), {})
    _sensor = _types.ModuleType('sensor_msgs')
    _sensor_msg = _types.ModuleType('sensor_msgs.msg')
    _sensor_msg.CameraInfo = type('CameraInfo', (), {})
    _sensor_msg.Image = type('Image', (), {'__init__': lambda s: setattr(s, 'header', _ns(frame_id=''))})
    _rosnav = _types.ModuleType('rosnav_rl_msgs')
    _rosnav_srv = _types.ModuleType('rosnav_rl_msgs.srv')
    _rosnav_srv.GetCommand = type('GetCommand', (), {'Request': type('Request', (), {}), 'Response': type('Response', (), {})})
    _std = _types.ModuleType('std_msgs')
    _std_msg = _types.ModuleType('std_msgs.msg')
    _std_msg.String = type('String', (), {'__init__': lambda s: setattr(s, 'data', '')})
    sys.modules.update({
        'rclpy': _rclpy, 'rclpy.node': _rclpy_node, 'rclpy.parameter': _rclpy_param,
        'rclpy.qos': _rclpy_qos, 'geometry_msgs': _geom, 'geometry_msgs.msg': _geom_msg,
        'nav_msgs': _nav, 'nav_msgs.msg': _nav_msg, 'sensor_msgs': _sensor,
        'sensor_msgs.msg': _sensor_msg, 'rosnav_rl_msgs': _rosnav,
        'rosnav_rl_msgs.srv': _rosnav_srv, 'std_msgs': _std, 'std_msgs.msg': _std_msg,
    })

from arena_vln_models.backends import (
    HeuristicBackend,
    ModelSimDecision,
    Pose2D,
    PythonAdapterBackend,
    DualVLNObservation,
    _action_to_command,
    _official_discrete_primitive,
    _trajectory_control_step,
)


# ── Discrete action full matrix ──────────────────────────────────────────────

def _discrete_params():
    return {
        'max_linear': 1.0,
        'max_angular': 2.0,
        'discrete_arc_turn': True,
    }


class TestDiscreteActionFullMatrix:
    """Every discrete action (0-5) with the hard-coded mapping."""

    def test_action_0_stop(self):
        lin, ang, status, _debug = _action_to_command(0, _discrete_params())
        assert lin == 0.0
        assert ang == 0.0
        assert status == 'discrete_stop'

    def test_action_1_forward(self):
        lin, ang, status, _debug = _action_to_command(1, _discrete_params())
        assert lin > 0.0
        assert ang == 0.0
        assert status == 'discrete_forward'

    def test_action_2_native_left(self):
        """action 2 (native "turn_left") → +angular_z (ROS left / CCW)."""
        lin, ang, status, debug = _action_to_command(2, _discrete_params())
        assert ang > 0.0, f"Expected positive angular_z (ROS left), got {ang}"
        assert status == 'discrete_turn_left'
        assert debug['native_action_label'] == 'turn_left'
        assert debug['effective_action_label'] == 'turn_left'
        assert debug['arc_turn'] is True
        assert lin > 0.0  # arc turn preserves forward motion

    def test_action_3_native_right(self):
        """action 3 (native "turn_right") → -angular_z (ROS right / CW)."""
        lin, ang, status, debug = _action_to_command(3, _discrete_params())
        assert ang < 0.0, f"Expected negative angular_z (ROS right), got {ang}"
        assert status == 'discrete_turn_right'
        assert debug['native_action_label'] == 'turn_right'
        assert debug['effective_action_label'] == 'turn_right'

    def test_action_5_look_down(self):
        lin, ang, status, debug = _action_to_command(5, _discrete_params())
        assert lin == 0.0
        assert ang == 0.0
        assert status == 'look_down_requested'
        assert debug['look_down_requested'] is True

    def test_action_2_and_3_are_opposites(self):
        """Action 2 and 3 should produce opposite angular_z signs."""
        _lin2, ang2, _s2, _d2 = _action_to_command(2, _discrete_params())
        _lin3, ang3, _s3, _d3 = _action_to_command(3, _discrete_params())
        assert ang2 * ang3 < 0.0, (
            f"action 2 ang={ang2}, action 3 ang={ang3} — should have opposite signs"
        )

    def test_official_discrete_turn_primitives_are_in_place_bounded_and_compensated(self):
        lin2, ang2, status2, debug2 = _official_discrete_primitive(2, _discrete_params())
        lin3, ang3, status3, debug3 = _official_discrete_primitive(3, _discrete_params())
        assert lin2 == 0.0
        assert lin3 == 0.0
        assert ang2 < 0.0
        assert ang3 > 0.0
        assert status2 == 'official_discrete_turn_left_primitive'
        assert status3 == 'official_discrete_turn_right_primitive'
        assert debug2['official_discrete_primitive'] is True
        assert debug3['primitive_interface'] == 'single_cmd_vel_tick'

    def test_official_discrete_forward_primitive_moves_forward_only(self):
        lin, ang, status, debug = _official_discrete_primitive(1, _discrete_params())
        assert lin > 0.0
        assert ang == 0.0
        assert status == 'official_discrete_forward_primitive'
        assert debug['primitive_forward_speed'] <= _discrete_params()['max_linear']


# ── Trajectory control step ──────────────────────────────────────────────────

class TestTrajectoryControlStep:
    """Verify _trajectory_control_step extracts correct waypoint."""

    def test_last_waypoint_used(self):
        x, y, yaw = _trajectory_control_step([[0.1, 0.0, 0.05], [0.3, 0.2, -0.5]])
        assert x == 0.3
        assert y == 0.2
        assert yaw == -0.5

    def test_single_waypoint(self):
        x, y, yaw = _trajectory_control_step([[0.5, 0.0, 0.3]])
        assert x == 0.5
        assert y == 0.0
        assert yaw == 0.3

    def test_yaw_fallback_to_bearing(self):
        """When only x,y provided, yaw = atan2(y, x)."""
        x, y, yaw = _trajectory_control_step([[1.0, 1.0]])
        assert x == 1.0
        assert y == 1.0
        assert abs(yaw - math.pi / 4) < 0.001

    def test_empty_trajectory_returns_none(self):
        assert _trajectory_control_step([]) is None

    def test_yaw_passthrough_no_inversion(self):
        """Trajectory yaw passes through directly — no sign inversion."""
        # Positive yaw → positive angular_z (left turn in ROS)
        x, y, yaw = _trajectory_control_step([[0.3, 0.0, 0.8]])
        assert yaw == 0.8
        # Negative yaw → negative angular_z (right turn in ROS)
        x2, y2, yaw2 = _trajectory_control_step([[0.3, 0.0, -0.8]])
        assert yaw2 == -0.8


# ── Dual-mode consistency (trajectory vs discrete) ───────────────────────────

class TestDualModeConsistency:
    """Verify trajectory and discrete policies produce directionally consistent Twist."""

    def _make_backend(self, policy):
        backend = PythonAdapterBackend.__new__(PythonAdapterBackend)
        backend._params = {
            'max_linear': 0.5,
            'max_angular': 0.5,
            'model_output_policy': policy,
        }
        return backend

    def test_same_turn_direction_across_policies(self):
        """Both policies should produce the same turn direction for consistent model output."""
        # Model outputs: action=2 (native left) + trajectory with positive yaw
        output = {
            'discrete_action': 2,
            'output_trajectory': [[0.3, 0.0, 0.25]],  # positive yaw = left in ROS
        }

        traj_backend = self._make_backend('trajectory')
        disc_backend = self._make_backend('discrete')

        traj_decision = traj_backend._coerce_output(output)
        disc_decision = disc_backend._coerce_output(output)

        # Trajectory policy: uses trajectory yaw directly → angular_z = +0.25
        assert traj_decision.status == 'trajectory_command'
        assert traj_decision.angular_z == 0.25

        # Discrete policy: action 2 → +angular_z (ROS left)
        assert disc_decision.status == 'discrete_turn_left'
        assert disc_decision.angular_z > 0.0

        # Both produce positive angular_z (left turn) — consistent!
        assert traj_decision.angular_z > 0.0
        assert disc_decision.angular_z > 0.0

    def test_trajectory_policy_no_yaw_inversion(self):
        """Trajectory policy never inverts yaw — passthrough."""
        backend = self._make_backend('trajectory')
        decision = backend._coerce_output({
            'output_trajectory': [[0.3, 0.0, 0.5]],
        })
        assert decision.angular_z == 0.5

    def test_discrete_policy_hard_coded_mapping(self):
        """Discrete policy: action 2 → +angular_z (left), action 3 → -angular_z (right)."""
        decision_2 = self._make_backend('discrete')._coerce_output({'discrete_action': 2})
        decision_3 = self._make_backend('discrete')._coerce_output({'discrete_action': 3})

        assert decision_2.angular_z > 0.0  # action 2 = left
        assert decision_3.angular_z < 0.0  # action 3 = right

    def test_trajectory_fallback_when_discrete_missing(self):
        """When discrete_action is missing, trajectory policy still works."""
        backend = self._make_backend('trajectory')
        decision = backend._coerce_output({
            'output_trajectory': [[0.2, 0.1, -0.3]],
        })
        assert decision.status == 'trajectory_command'
        assert decision.linear_x == pytest.approx(math.hypot(0.2, 0.1))
        assert decision.angular_z == -0.3

    def test_discrete_fallback_when_trajectory_missing(self):
        """When output_trajectory is missing, discrete policy still works."""
        backend = self._make_backend('discrete')
        decision = backend._coerce_output({'discrete_action': 1})
        assert decision.status == 'discrete_forward'
        assert decision.linear_x > 0.0
        assert decision.angular_z == 0.0


# ── Heuristic vs model yaw convention ────────────────────────────────────────

class TestHeuristicYawConvention:
    """Heuristic backend yaw convention matches ROS standard."""

    def test_goal_left_produces_positive_angular_z(self):
        """Goal to the left → positive angular_z (counter-clockwise)."""
        params = {
            'max_linear': 1.0, 'max_angular': 1.5,
            'k_lin': 1.0, 'k_ang': 2.0,
            'goal_tolerance': 0.35, 'angle_tolerance': 0.25,
            'min_lin_when_aligned': 0.05,
        }
        decision = HeuristicBackend(None, params).compute(DualVLNObservation(
            pose=Pose2D(0.0, 0.0, 0.0),
            goal=Pose2D(0.0, 1.0, 0.0),  # goal is to the LEFT
            instruction='turn left',
        ))
        assert decision.angular_z > 0.0, (
            f"Goal left should produce positive angular_z, got {decision.angular_z}"
        )

    def test_goal_right_produces_negative_angular_z(self):
        """Goal to the right → negative angular_z (clockwise)."""
        params = {
            'max_linear': 1.0, 'max_angular': 1.5,
            'k_lin': 1.0, 'k_ang': 2.0,
            'goal_tolerance': 0.35, 'angle_tolerance': 0.25,
            'min_lin_when_aligned': 0.05,
        }
        decision = HeuristicBackend(None, params).compute(DualVLNObservation(
            pose=Pose2D(0.0, 0.0, 0.0),
            goal=Pose2D(0.0, -1.0, 0.0),  # goal is to the RIGHT
            instruction='turn right',
        ))
        assert decision.angular_z < 0.0, (
            f"Goal right should produce negative angular_z, got {decision.angular_z}"
        )

    def test_goal_behind_produces_large_rotation(self):
        """Goal behind → pure rotation with correct sign."""
        params = {
            'max_linear': 1.0, 'max_angular': 1.5,
            'k_lin': 1.0, 'k_ang': 2.0,
            'goal_tolerance': 0.35, 'angle_tolerance': 0.25,
            'min_lin_when_aligned': 0.05,
        }
        # Goal at (-1, 0) with pose at (0,0,0): goal is behind, yaw_err ≈ π
        decision = HeuristicBackend(None, params).compute(DualVLNObservation(
            pose=Pose2D(0.0, 0.0, 0.0),
            goal=Pose2D(-1.0, 0.0, 0.0),
            instruction='turn around',
        ))
        assert decision.status == 'rotate_to_goal'
        assert decision.linear_x == 0.0
        # yaw_err ≈ π or -π depending on wrap; either way |angular_z| > 0
        assert abs(decision.angular_z) > 0.0

    def test_heuristic_and_discrete_agree_on_left(self):
        """Heuristic 'goal left' and discrete 'action 2' both produce positive angular_z."""
        params_h = {
            'max_linear': 1.0, 'max_angular': 1.5,
            'k_lin': 1.0, 'k_ang': 2.0,
            'goal_tolerance': 0.35, 'angle_tolerance': 0.25,
            'min_lin_when_aligned': 0.05,
        }
        h_decision = HeuristicBackend(None, params_h).compute(DualVLNObservation(
            pose=Pose2D(0.0, 0.0, 0.0),
            goal=Pose2D(0.0, 1.0, 0.0),
            instruction='turn left',
        ))
        _lin, ang_d, _status, _debug = _action_to_command(2, {
            'max_linear': 1.0, 'max_angular': 1.5,
        })

        # Both should be positive (left turn)
        assert h_decision.angular_z > 0.0
        assert ang_d > 0.0


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestActionEdgeCases:
    """Edge cases for action mapping."""

    def test_unsupported_action_returns_stop(self):
        lin, ang, status, debug = _action_to_command(99, _discrete_params())
        assert lin == 0.0
        assert ang == 0.0
        assert status == 'unsupported_discrete_action'
        assert debug['unsupported_action'] == 99

    def test_none_action_returns_none(self):
        assert _action_to_command(None, _discrete_params()) is None

    def test_string_action_returns_none(self):
        assert _action_to_command("not_an_int", _discrete_params()) is None

    def test_empty_list_action_returns_none(self):
        assert _action_to_command([], _discrete_params()) is None

    def test_list_action_uses_first_element(self):
        lin, ang, status, _debug = _action_to_command([2, 1, 3], _discrete_params())
        assert status == 'discrete_turn_left'
        assert ang > 0.0

    def test_forward_linear_is_bounded_by_max(self):
        lin, ang, _status, _debug = _action_to_command(1, {
            'max_linear': 0.3, 'max_angular': 2.0,
        })
        assert lin == pytest.approx(0.18)  # 0.3 * 0.6
        assert ang == 0.0

    def test_turn_linear_has_minimum(self):
        """Turn actions have a minimum forward speed so robot makes visible progress."""
        lin, ang, _status, _debug = _action_to_command(2, {
            'max_linear': 0.1, 'max_angular': 2.0, 'discrete_arc_turn': True,
        })
        assert lin == 0.12  # max(0.1*0.6, 0.12) = 0.12
        assert ang > 0.0


# ── pytest import for approx ─────────────────────────────────────────────────
import pytest
