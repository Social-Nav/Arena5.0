import threading
import sys
import types
from types import SimpleNamespace

import numpy as np


def _install_ros_import_stubs_if_needed():
    try:
        import rclpy  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _Node:
        pass

    class _Parameter:
        def __init__(self, name, value=None):
            self.name = name
            self.value = value

    class _Twist:
        def __init__(self):
            self.linear = SimpleNamespace(x=0.0)
            self.angular = SimpleNamespace(z=0.0)

    class _String:
        def __init__(self):
            self.data = ''

    class _Image:
        def __init__(self):
            self.header = SimpleNamespace(frame_id='')
            self.height = 0
            self.width = 0
            self.encoding = ''
            self.is_bigendian = 0
            self.step = 0
            self.data = b''

    rclpy_mod = types.ModuleType('rclpy')
    node_mod = types.ModuleType('rclpy.node')
    parameter_mod = types.ModuleType('rclpy.parameter')
    qos_mod = types.ModuleType('rclpy.qos')
    node_mod.Node = _Node
    parameter_mod.Parameter = _Parameter
    qos_mod.DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL=1, VOLATILE=2)
    qos_mod.ReliabilityPolicy = SimpleNamespace(RELIABLE=1, BEST_EFFORT=2)
    qos_mod.QoSProfile = lambda *args, **kwargs: SimpleNamespace(**kwargs)

    geometry_msgs_mod = types.ModuleType('geometry_msgs')
    geometry_msgs_msg_mod = types.ModuleType('geometry_msgs.msg')
    geometry_msgs_msg_mod.PoseStamped = type('PoseStamped', (), {})
    geometry_msgs_msg_mod.Twist = _Twist

    nav_msgs_mod = types.ModuleType('nav_msgs')
    nav_msgs_msg_mod = types.ModuleType('nav_msgs.msg')
    nav_msgs_msg_mod.Odometry = type('Odometry', (), {})

    sensor_msgs_mod = types.ModuleType('sensor_msgs')
    sensor_msgs_msg_mod = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg_mod.CameraInfo = type('CameraInfo', (), {})
    sensor_msgs_msg_mod.Image = _Image

    rosnav_mod = types.ModuleType('rosnav_rl_msgs')
    rosnav_srv_mod = types.ModuleType('rosnav_rl_msgs.srv')
    rosnav_srv_mod.GetCommand = type('GetCommand', (), {
        'Request': type('Request', (), {}),
        'Response': type('Response', (), {}),
    })

    std_msgs_mod = types.ModuleType('std_msgs')
    std_msgs_msg_mod = types.ModuleType('std_msgs.msg')
    std_msgs_msg_mod.String = _String

    sys.modules.update({
        'rclpy': rclpy_mod,
        'rclpy.node': node_mod,
        'rclpy.parameter': parameter_mod,
        'rclpy.qos': qos_mod,
        'geometry_msgs': geometry_msgs_mod,
        'geometry_msgs.msg': geometry_msgs_msg_mod,
        'nav_msgs': nav_msgs_mod,
        'nav_msgs.msg': nav_msgs_msg_mod,
        'sensor_msgs': sensor_msgs_mod,
        'sensor_msgs.msg': sensor_msgs_msg_mod,
        'rosnav_rl_msgs': rosnav_mod,
        'rosnav_rl_msgs.srv': rosnav_srv_mod,
        'std_msgs': std_msgs_mod,
        'std_msgs.msg': std_msgs_msg_mod,
    })


_install_ros_import_stubs_if_needed()

from arena_vln_models.core import (
    camera_intrinsic_matrix,
    normalize_backend_output,
    normalize_depth,
    normalize_rgb,
    pose_vector,
)
from arena_vln_models.backends import HeuristicBackend, ModelSimDecision, Pose2D, PythonAdapterBackend, DualVLNObservation, _action_to_command
from arena_vln_models import internnav as internnav_module
from arena_vln_models.internnav import InternNavAdapter, available_backends, load_internnav_adapter
from arena_vln_models.internnav_server import InternNavServer, _normalize_internnav_adapter_target
from arena_vln_models.visualization import image_msg_to_numpy, numpy_to_image_msg, render_debug_overlay


def _observation(**overrides):
    pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0)
    goal = SimpleNamespace(x=1.0, y=0.0, yaw=0.0)
    values = {
        'pose': pose,
        'goal': goal,
        'instruction': 'go forward',
        'rgb_image': np.zeros((4, 6, 3), dtype=np.uint8),
        'depth_image': np.ones((4, 6), dtype=np.float32),
        'camera_intrinsics': (1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0),
        'look_down': False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_package_imports_and_backend_metadata():
    assert InternNavAdapter is not None
    backends = available_backends()
    assert backends
    assert backends[0]['name'] == 'internnav_realworld_internvla_n1'


def test_observation_helpers_normalize_arrays_and_geometry():
    obs = _observation(rgb_image=np.ones((2, 3), dtype=np.float32) * 255.0)
    rgb = normalize_rgb(obs.rgb_image)
    assert rgb.shape == (2, 3, 3)
    assert rgb.dtype == np.uint8

    depth = normalize_depth(np.ones((2, 3, 1), dtype=np.uint16) * 1000)
    assert depth.shape == (2, 3)
    assert float(depth[0, 0]) == 1.0

    np.testing.assert_allclose(pose_vector(obs), np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    assert camera_intrinsic_matrix(obs, rgb).shape == (3, 3)


def test_output_normalization_supports_discrete_trajectory_and_pixel():
    output = normalize_backend_output({
        'output_action': 1,
        'trajectory': np.asarray([[0.2, 0.0, 0.0]], dtype=np.float32),
        'target_pixel': np.asarray([3, 2], dtype=np.int64),
        'debug': {'source': 'unit'},
    })
    assert output['discrete_action'] == 1
    assert output['output_trajectory'] == [[0.20000000298023224, 0.0, 0.0]]
    assert output['output_pixel'] == [3, 2]
    assert output['debug']['source'] == 'unit'


def test_internnav_mock_mode_returns_command_mapping():
    adapter = load_internnav_adapter(params={'model_path': 'mock'})
    result = adapter.compute(_observation())
    assert result['status'] == 'mock_internnav_command'
    assert result['discrete_action'] in {0, 1, 2, 3}
    assert result['debug']['wrapper_package'] == 'arena_vln_models'


def test_internnav_model_path_falls_back_to_env(monkeypatch, tmp_path):
    model_dir = tmp_path / 'InternVLA-N1'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')
    monkeypatch.setenv('ARENA_INTERNNAV_MODEL_PATH', str(model_dir))

    resolved, debug = internnav_module._resolve_model_path('')
    assert resolved == str(model_dir)
    assert debug['requested_model_path_source'] == 'env:ARENA_INTERNNAV_MODEL_PATH'


def test_internnav_missing_visual_input_degrades_when_real_adapter_present():
    adapter = load_internnav_adapter(params={'model_path': 'mock'})
    adapter._adapter = SimpleNamespace(step=lambda *args, **kwargs: None)  # noqa: SLF001 - intentional internal test hook
    adapter._mode = 'internnav_async_agent'  # noqa: SLF001 - intentional internal test hook

    result = adapter.compute(_observation(rgb_image=None))
    assert result['status'] == 'internnav_missing_rgb'
    assert result['degraded'] is True


def test_python_adapter_backend_loads_canonical_target_only():
    params = {
        'adapter_target': 'arena_vln_models.internnav:load_internnav_adapter',
        'require_real_backend': False,
        'model_path': 'mock',
        'device': 'cpu',
        'max_linear': 1.0,
        'max_angular': 1.5,
        'goal_tolerance': 0.35,
        'angle_tolerance': 0.25,
        'inference_timeout_sec': 30.0,
    }
    backend = PythonAdapterBackend(None, params)
    decision = backend.compute(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(1.0, 0.0, 0.0),
        instruction='go forward',
        rgb_image=np.zeros((4, 6, 3), dtype=np.uint8),
        depth_image=np.ones((4, 6), dtype=np.float32),
        camera_intrinsics=(1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0),
    ))
    assert decision.status == 'mock_internnav_command'
    assert decision.linear_x > 0.0
    assert decision.debug['adapter_target'] == 'arena_vln_models.internnav:load_internnav_adapter'


def test_discrete_turn_mapping_supports_inverted_effective_labels():
    params = {'max_linear': 1.0, 'max_angular': 2.0, 'invert_discrete_turns': True}
    linear_x, angular_z, status, debug = _action_to_command(2, params)
    assert status == 'discrete_turn_right'
    assert linear_x == 0.0
    assert angular_z < 0.0
    assert debug['native_action_label'] == 'turn_left'
    assert debug['effective_action_label'] == 'turn_right'
    assert debug['invert_discrete_turns'] is True
    assert debug['arc_turn'] is False


def test_discrete_turn_mapping_default_preserves_native_effective_labels():
    params = {'max_linear': 1.0, 'max_angular': 2.0, 'invert_discrete_turns': False}
    linear_x, angular_z, status, debug = _action_to_command(2, params)
    assert status == 'discrete_turn_left'
    assert linear_x == 0.0
    assert angular_z > 0.0
    assert debug['native_action_label'] == 'turn_left'
    assert debug['effective_action_label'] == 'turn_left'
    assert debug['invert_discrete_turns'] is False
    assert debug['arc_turn'] is False


def test_trajectory_output_uses_internnav_continuous_subgoal_interface():
    backend = PythonAdapterBackend.__new__(PythonAdapterBackend)
    backend._params = {'max_linear': 0.5, 'max_angular': 0.5}

    decision = backend._coerce_output({
        'output_trajectory': [
            [0.05, 0.0, 0.1],
            [0.3, 0.4, -0.8],
        ],
        'debug': {'source': 'unit'},
    })

    assert decision.status == 'trajectory_command'
    assert decision.linear_x == 0.5
    assert decision.angular_z == -0.5
    assert decision.debug['trajectory_control_step'] == [0.3, 0.4, -0.8]
    assert decision.debug['trajectory_first_step'] == [0.05, 0.0, 0.1]
    assert decision.debug['trajectory_command_interface'] == 'internnav_continuous_subgoal_vw'


def test_trajectory_policy_prefers_trajectory_when_action_is_also_present():
    backend = PythonAdapterBackend.__new__(PythonAdapterBackend)
    backend._params = {
        'max_linear': 0.5,
        'max_angular': 0.5,
        'model_output_policy': 'trajectory',
        'invert_discrete_turns': False,
    }

    decision = backend._coerce_output({
        'discrete_action': 2,
        'output_trajectory': [[0.4, 0.0, 0.25]],
    })

    assert decision.status == 'trajectory_command'
    assert decision.linear_x == 0.4
    assert decision.angular_z == 0.25
    assert decision.debug['selected_output_mode'] == 'trajectory'
    assert decision.debug['model_output_policy'] == 'trajectory'
    assert 'selected_action' not in decision.debug


def test_discrete_policy_forces_action_mapping_for_ablation():
    backend = PythonAdapterBackend.__new__(PythonAdapterBackend)
    backend._params = {
        'max_linear': 0.5,
        'max_angular': 0.5,
        'model_output_policy': 'discrete',
        'invert_discrete_turns': False,
    }

    decision = backend._coerce_output({
        'discrete_action': 2,
        'output_trajectory': [[0.4, 0.0, 0.25]],
    })

    assert decision.status == 'discrete_turn_left'
    assert decision.linear_x == 0.0
    assert decision.angular_z > 0.0
    assert decision.debug['selected_output_mode'] == 'discrete'
    assert decision.debug['model_output_policy'] == 'discrete'
    assert decision.debug['selected_action'] == 2


def test_internnav_server_defaults_empty_adapter_target_for_internnav_mode():
    adapter_target, source = _normalize_internnav_adapter_target('internnav', '')
    assert adapter_target == 'arena_vln_models.internnav:load_internnav_adapter'
    assert source == 'default'


def test_internnav_server_normalizes_legacy_native_adapter_target():
    adapter_target, source = _normalize_internnav_adapter_target(
        'internnav',
        'internnav.agent.internvla_n1_agent_realworld.InternVLAN1AsyncAgent',
    )
    assert adapter_target == 'arena_vln_models.internnav:load_internnav_adapter'
    assert source == 'legacy:internnav.agent.internvla_n1_agent_realworld.InternVLAN1AsyncAgent'


def test_internnav_cached_command_preserves_last_model_turn_direction():
    server = InternNavServer.__new__(InternNavServer)
    server._state_lock = threading.Lock()
    server._params = {'camera_stale_after_sec': 2.0, 'invert_discrete_turns': True}
    server._latest_rgb_ts = 0.0
    server._latest_depth_ts = 0.0
    server._camera_info_ts = 0.0
    server._last_model_decision = ModelSimDecision(
        linear_x=0.36,
        angular_z=-0.375,
        status='internnav_command',
        degraded=False,
        debug={
            'selected_action': 2,
            'native_action_label': 'turn_left',
            'effective_action_label': 'turn_right',
            'invert_discrete_turns': True,
        },
    )

    cached = server._cached_model_decision_while_computing(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(1.0, 0.0, 0.0),
        instruction='go forward',
    ))

    assert cached is not None
    assert cached.status == 'inference_in_progress_cached_internnav_command'
    assert cached.linear_x == 0.36
    assert cached.angular_z == -0.375
    assert cached.debug['cached_previous_model_command'] is True
    assert cached.debug['cached_selected_action'] == 2
    assert 'selected_action' not in cached.debug


def test_heuristic_backend_produces_forward_command():
    params = {
        'max_linear': 1.0,
        'max_angular': 1.5,
        'k_lin': 1.0,
        'k_ang': 1.5,
        'goal_tolerance': 0.35,
        'angle_tolerance': 0.25,
        'min_lin_when_aligned': 0.05,
    }
    decision = HeuristicBackend(None, params).compute(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(1.0, 0.0, 0.0),
        instruction='go forward',
    ))
    assert decision.status == 'drive_to_goal'
    assert decision.linear_x > 0.0


def test_visualization_image_round_trip_helpers():
    array = np.zeros((2, 3, 3), dtype=np.uint8)
    array[..., 1] = 128
    msg = numpy_to_image_msg(array, frame_id='camera')
    assert msg.height == 2
    assert msg.width == 3
    assert msg.encoding == 'rgb8'
    recovered = image_msg_to_numpy(msg)
    assert recovered.shape == (2, 3, 3)
    assert int(recovered[0, 0, 1]) == 128


def test_debug_overlay_renders_action_and_freshness_diagnostics():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    observation = DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(1.0, 0.0, 0.0),
        instruction='go forward',
        rgb_image=image,
        depth_image=np.ones((180, 240), dtype=np.float32),
        camera_frame_id='head_camera',
        metadata={
            'rgb_available': True,
            'depth_available': True,
            'camera_info_available': False,
            'stale_after_sec': 2.0,
        },
    )
    decision = ModelSimDecision(
        linear_x=0.36,
        angular_z=-0.375,
        status='internnav_command',
        debug={
            'selected_action': 2,
            'native_action_label': 'turn_left',
            'effective_action_label': 'turn_right',
            'invert_discrete_turns': True,
            'goal_distance': 1.0,
            'yaw_error': -0.5,
            'action_history_tail': [2, 2, 1],
            'sensor_ages_sec': {'rgb': 0.1, 'depth': 0.2, 'camera_info': None},
            'stale_after_sec': 2.0,
        },
    )
    overlay = render_debug_overlay(image, observation, decision, backend_name='python_adapter')
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert int(overlay.sum()) > 0
