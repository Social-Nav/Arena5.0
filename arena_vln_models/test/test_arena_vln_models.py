import threading
import json
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
    time_mod = types.ModuleType('rclpy.time')
    node_mod.Node = _Node
    parameter_mod.Parameter = _Parameter
    qos_mod.DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL=1, VOLATILE=2)
    qos_mod.ReliabilityPolicy = SimpleNamespace(RELIABLE=1, BEST_EFFORT=2)
    qos_mod.QoSProfile = lambda *args, **kwargs: SimpleNamespace(**kwargs)
    time_mod.Time = type('Time', (), {})

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

    tf2_ros_mod = types.ModuleType('tf2_ros')
    tf2_ros_mod.Buffer = type('Buffer', (), {})
    tf2_ros_mod.TransformListener = type('TransformListener', (), {})

    sys.modules.update({
        'rclpy': rclpy_mod,
        'rclpy.node': node_mod,
        'rclpy.parameter': parameter_mod,
        'rclpy.qos': qos_mod,
        'rclpy.time': time_mod,
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
        'tf2_ros': tf2_ros_mod,
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
from arena_vln_models.internnav import (
    InternNavAdapter,
    InternVLARealworldHttpAdapter,
    available_backends,
    load_internnav_adapter,
    load_internvla_realworld_http_adapter,
)
from arena_vln_models.internnav_server import (
    InternNavServer,
    _normalize_internnav_adapter_target,
    _resolve_adapter_target_for_http_adapter,
    _resolve_float,
    _resolve_mode_for_http_adapter,
)
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


def test_internvla_realworld_http_adapter_posts_arena_observation(monkeypatch):
    captured = {}

    class _Response:
        text = '{"output_trajectory": [[0.1, 0.0, 0.0]]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {
                'status': 'internvla_realworld_http_command',
                'output_trajectory': [[0.1, 0.0, 0.0]],
                'output_pixel': [2, 3],
                'debug': {'server_compute_sec': 0.01},
            }

    def _post(url, *, files, data, timeout):
        captured['url'] = url
        captured['files'] = files
        captured['data'] = json.loads(data['json'])
        captured['timeout'] = timeout
        return _Response()

    monkeypatch.setitem(sys.modules, 'requests', SimpleNamespace(post=_post))
    adapter = load_internvla_realworld_http_adapter(params={
        'internnav_http_url': 'http://internnav:5801/eval_dual',
        'internnav_http_timeout_sec': 3.0,
        'model_output_policy': 'trajectory',
    })

    result = adapter.compute(_observation())

    assert isinstance(adapter, InternVLARealworldHttpAdapter)
    assert captured['url'] == 'http://internnav:5801/eval_dual'
    assert captured['timeout'] == 3.0
    assert captured['data']['reset'] is True
    assert captured['data']['instruction'] == 'go forward'
    assert captured['data']['client'] == 'arena_ros2_get_command_async_worker'
    assert 'image' in captured['files']
    assert 'depth' in captured['files']
    assert result['output_trajectory'] == [[0.1, 0.0, 0.0]]
    assert result['debug']['realworld_http_client'] is True
    assert result['debug']['system1_http_server'] is True
    assert result['debug']['system2_http_server'] is True


def test_internvla_realworld_http_adapter_prefers_eval_env_and_positive_timeout(monkeypatch):
    monkeypatch.setenv('ARENA_INTERNNAV_HTTP_URL', 'http://generic:5801/eval_dual')
    monkeypatch.setenv('ARENA_EVAL_INTERNNAV_HTTP_URL', 'http://eval:5801/eval_dual')
    monkeypatch.setenv('ARENA_INTERNNAV_HTTP_TIMEOUT_SEC', '1.0')
    monkeypatch.setenv('ARENA_EVAL_INTERNNAV_HTTP_TIMEOUT_SEC', '2.0')

    adapter = load_internvla_realworld_http_adapter(params={})

    assert adapter._url == 'http://eval:5801/eval_dual'  # noqa: SLF001 - intentional internal test hook
    assert adapter._timeout == 2.0  # noqa: SLF001 - intentional internal test hook


def test_internvla_realworld_http_adapter_non_positive_timeout_falls_back_to_inference_timeout():
    adapter = load_internvla_realworld_http_adapter(params={
        'internnav_http_timeout_sec': '0.0',
        'inference_timeout_sec': 7.5,
    })

    assert adapter._timeout == 7.5  # noqa: SLF001 - intentional internal test hook


def test_internvla_realworld_http_adapter_missing_rgb_safe_stops_without_post(monkeypatch):
    called = {'post': False}

    def _post_eval_dual(*args, **kwargs):
        called['post'] = True
        raise AssertionError('HTTP should not be called when RGB is missing')

    monkeypatch.setattr(InternVLARealworldHttpAdapter, '_post_eval_dual', staticmethod(_post_eval_dual))
    adapter = load_internvla_realworld_http_adapter(params={
        'internnav_http_url': 'http://internnav:5801/eval_dual',
    })

    result = adapter.compute(_observation(rgb_image=None))

    assert called['post'] is False
    assert result['status'] == 'internnav_http_missing_rgb'
    assert result['degraded'] is True
    assert result['discrete_action'] == 0
    assert result['debug']['safe_stop'] is True
    assert result['debug']['realworld_http_client'] is True


def test_internvla_realworld_http_adapter_reset_flag_tracks_session(monkeypatch):
    resets = []
    request_ids = []

    def _post_eval_dual(url, *, files, payload, timeout):
        resets.append(payload['reset'])
        request_ids.append(payload['request_id'])
        return {'output_trajectory': [[0.1, 0.0, 0.0]]}

    monkeypatch.setattr(InternVLARealworldHttpAdapter, '_post_eval_dual', staticmethod(_post_eval_dual))
    adapter = load_internvla_realworld_http_adapter(params={
        'internnav_http_url': 'http://internnav:5801/eval_dual',
    })

    adapter.compute(_observation())
    adapter.compute(_observation())
    adapter.compute(_observation(instruction='turn right'))
    adapter.compute(_observation(goal=SimpleNamespace(x=2.0, y=0.0, yaw=0.0)))

    assert resets == [True, False, True, True]
    assert request_ids == [1, 2, 3, 4]


def test_internvla_realworld_http_adapter_urllib_fallback_posts_multipart(monkeypatch):
    import urllib.request

    captured = {}
    original_import_module = internnav_module.importlib.import_module

    def _import_module(name, *args, **kwargs):
        if name == 'requests':
            raise ModuleNotFoundError(name)
        return original_import_module(name, *args, **kwargs)

    class _UrlopenResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"output_trajectory": [[0.2, 0.0, 0.0]], "debug": {"source": "urllib-test"}}'

    def _urlopen(request, timeout):
        captured['request'] = request
        captured['timeout'] = timeout
        captured['body'] = request.data
        captured['headers'] = dict(request.header_items())
        return _UrlopenResponse()

    monkeypatch.setattr(internnav_module.importlib, 'import_module', _import_module)
    monkeypatch.setattr(urllib.request, 'urlopen', _urlopen)
    adapter = load_internvla_realworld_http_adapter(params={
        'internnav_http_url': 'http://internnav:5801/eval_dual',
        'internnav_http_timeout_sec': 3.0,
    })

    result = adapter.compute(_observation())

    assert captured['timeout'] == 3.0
    assert captured['request'].get_method() == 'POST'
    assert 'multipart/form-data' in captured['headers']['Content-type']
    assert b'name="json"' in captured['body']
    assert b'name="image"; filename="rgb_image.jpg"' in captured['body']
    assert b'name="depth"; filename="depth_image.png"' in captured['body']
    assert result['output_trajectory'] == [[0.2, 0.0, 0.0]]
    assert result['debug']['realworld_http_client'] is True


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


def test_discrete_turn_mapping_preserves_native_labels():
    """action 2 (native "turn_left") → +angular_z (ROS left / CCW)."""
    params = {'max_linear': 1.0, 'max_angular': 2.0}
    linear_x, angular_z, status, debug = _action_to_command(2, params)
    assert status == 'discrete_turn_left'
    assert linear_x == 0.0
    assert angular_z > 0.0
    assert debug['native_action_label'] == 'turn_left'
    assert debug['effective_action_label'] == 'turn_left'
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


def test_python_adapter_backend_promotes_top_level_llm_trace_fields():
    backend = PythonAdapterBackend.__new__(PythonAdapterBackend)
    backend._params = {
        'max_linear': 0.5,
        'max_angular': 0.5,
        'model_output_policy': 'auto',
    }

    decision = backend._coerce_output({
        'linear_x': 0.1,
        'angular_z': 0.0,
        'raw_output_text': '215 376',
        'llm_digits': [215, 376],
        'digit_groups': [215, 376],
        'generated_token_ids': [11, 22],
    })

    assert decision.debug['raw_output_text'] == '215 376'
    assert decision.debug['llm_digits'] == [215, 376]
    assert decision.debug['digit_groups'] == [215, 376]
    assert decision.debug['generated_token_ids'] == [11, 22]


def test_internnav_server_status_contains_llm_block():
    published = []
    server = InternNavServer.__new__(InternNavServer)
    server._status_publisher = SimpleNamespace(publish=lambda msg: published.append(msg.data))
    decision = ModelSimDecision(
        linear_x=0.0,
        angular_z=0.0,
        status='internnav_command',
        degraded=False,
        debug={
            'raw_output_text': '215 376',
            'llm_digits': [215, 376],
            'digit_groups': [215, 376],
            'model_generation_output_mode': 'pixel_goal',
            'pixel_goal': [376, 215],
        },
    )

    server._publish_status(decision)

    payload = json.loads(published[-1])
    assert payload['llm']['raw_output_text'] == '215 376'
    assert payload['llm']['llm_digits'] == [215, 376]
    assert payload['llm']['digit_groups'] == [215, 376]
    assert payload['llm']['output_mode'] == 'pixel_goal'
    assert payload['debug']['llm_digits'] == [215, 376]


def test_discrete_policy_forces_action_mapping_for_ablation():
    backend = PythonAdapterBackend.__new__(PythonAdapterBackend)
    backend._params = {
        'max_linear': 0.5,
        'max_angular': 0.5,
        'model_output_policy': 'discrete',
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


def test_trajectory_policy_symbolic_official_discrete_uses_bounded_primitive():
    backend = PythonAdapterBackend.__new__(PythonAdapterBackend)
    backend._params = {
        'max_linear': 0.6,
        'max_angular': 1.5,
        'model_output_policy': 'trajectory',
        'internnav_symbolic_fallback_policy': 'official_discrete',
        'official_discrete_forward_speed': 0.2,
        'official_discrete_turn_speed': 0.4,
    }

    decision = backend._coerce_output({
        'discrete_action': 2,
        'status': 'synthetic_symbolic_action',
        'debug': {'symbolic_fallback_policy': 'official_discrete'},
    })

    assert decision.status == 'synthetic_symbolic_action'
    assert decision.linear_x == 0.0
    assert decision.angular_z == -0.4
    assert decision.debug['selected_action'] == 2
    assert decision.debug['official_discrete_selected'] is True
    assert decision.debug['official_discrete_primitive'] is True
    assert decision.debug['primitive_interface'] == 'single_cmd_vel_tick'
    assert decision.debug['primitive_forward_speed'] == 0.2
    assert decision.debug['primitive_turn_speed'] == 0.4
    assert decision.debug['command_generation_stage'] == 'symbolic_action_to_cmd_vel'


def test_trajectory_policy_goal_guided_synthetic_trajectory_not_official_primitive():
    backend = PythonAdapterBackend.__new__(PythonAdapterBackend)
    backend._params = {
        'max_linear': 0.6,
        'max_angular': 1.5,
        'model_output_policy': 'trajectory',
        'internnav_symbolic_fallback_policy': 'goal_guided',
    }

    decision = backend._coerce_output({
        'discrete_action': 2,
        'output_trajectory': [[0.45, 0.0, 0.25]],
        'debug': {
            'symbolic_fallback_policy': 'goal_guided',
            'trajectory_synthetic_source': 'goal_guided_symbolic_fallback',
        },
    })

    assert decision.status == 'trajectory_command'
    assert decision.debug['selected_output_mode'] == 'trajectory'
    assert decision.debug['trajectory_synthetic_source'] == 'goal_guided_symbolic_fallback'
    assert 'official_discrete_primitive' not in decision.debug


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


def test_internnav_server_invalid_float_env_falls_back_to_raw_value(monkeypatch):
    monkeypatch.setenv('ARENA_EVAL_INTERNNAV_HTTP_TIMEOUT_SEC', 'not-a-float')

    value, source = _resolve_float(4.5, env_names=('ARENA_EVAL_INTERNNAV_HTTP_TIMEOUT_SEC',))

    assert value == 4.5
    assert source == 'invalid-env:ARENA_EVAL_INTERNNAV_HTTP_TIMEOUT_SEC'


def test_internnav_server_http_url_forces_internnav_mode():
    mode, source = _resolve_mode_for_http_adapter('heuristic', 'http://internnav:5801/eval_dual')

    assert mode == 'internnav'
    assert source == 'internnav_http_url'


def test_internnav_server_http_url_keeps_existing_internnav_mode():
    mode, source = _resolve_mode_for_http_adapter('internnav', 'http://internnav:5801/eval_dual')

    assert mode == 'internnav'
    assert source is None


def test_internnav_server_http_url_selects_realworld_http_adapter_for_empty_target():
    adapter_target, source = _resolve_adapter_target_for_http_adapter('', 'http://internnav:5801/eval_dual')

    assert adapter_target == 'arena_vln_models.internnav:load_internvla_realworld_http_adapter'
    assert source == 'internnav_http_url'


def test_internnav_server_http_url_replaces_legacy_local_adapter_target():
    adapter_target, source = _resolve_adapter_target_for_http_adapter(
        'internnav.agent.internvla_n1_agent_realworld.InternVLAN1AsyncAgent',
        'http://internnav:5801/eval_dual',
    )

    assert adapter_target == 'arena_vln_models.internnav:load_internvla_realworld_http_adapter'
    assert source == 'internnav_http_url'


def test_internnav_instruction_gate_blocks_generic_default_instruction():
    server = InternNavServer.__new__(InternNavServer)
    server._params = {'require_route_instruction': True}
    server.get_parameter = lambda name: SimpleNamespace(value='vln_instruction')

    decision = server._instruction_gate_decision(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(1.0, 0.0, 0.0),
        instruction='navigate',
    ))

    assert decision is not None
    assert decision.status == 'waiting_for_instruction'
    assert decision.degraded is True
    assert decision.linear_x == 0.0
    assert decision.angular_z == 0.0
    assert decision.debug['instruction_gate'] is True


def test_internnav_instruction_gate_allows_route_specific_instruction():
    server = InternNavServer.__new__(InternNavServer)
    server._params = {'require_route_instruction': True}
    server.get_parameter = lambda name: SimpleNamespace(value='vln_instruction')

    decision = server._instruction_gate_decision(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(1.0, 0.0, 0.0),
        instruction='Turn right and move through the open waiting area toward the corridor.',
    ))

    assert decision is None


def test_internnav_cached_command_preserves_last_model_turn_direction():
    server = InternNavServer.__new__(InternNavServer)
    server._state_lock = threading.Lock()
    server._params = {'camera_stale_after_sec': 2.0}
    server._latest_rgb_ts = 0.0
    server._latest_depth_ts = 0.0
    server._camera_info_ts = 0.0
    server._last_model_decision = ModelSimDecision(
        linear_x=0.36,
        angular_z=0.375,
        status='internnav_command',
        degraded=False,
        debug={
            'selected_action': 2,
            'native_action_label': 'turn_left',
            'effective_action_label': 'turn_left',
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
    assert cached.angular_z == 0.375
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


def test_heuristic_backend_caps_far_goal_rotation_and_arcs_forward():
    params = {
        'max_linear': 1.0,
        'max_angular': 1.5,
        'k_lin': 1.0,
        'k_ang': 2.0,
        'goal_tolerance': 0.35,
        'angle_tolerance': 0.25,
        'min_lin_when_aligned': 0.05,
    }
    decision = HeuristicBackend(None, params).compute(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(1.0, 1.0, 0.0),
        instruction='go diagonally',
    ))

    assert decision.status == 'arc_to_goal'
    assert decision.linear_x > 0.0
    assert abs(decision.angular_z) <= 0.6
    assert decision.debug['arc_turn'] is True
    assert decision.debug['effective_max_angular'] == 0.6


def test_heuristic_backend_slows_large_pure_rotations():
    params = {
        'max_linear': 1.0,
        'max_angular': 1.5,
        'k_lin': 1.0,
        'k_ang': 2.0,
        'goal_tolerance': 0.35,
        'angle_tolerance': 0.25,
        'min_lin_when_aligned': 0.05,
    }
    decision = HeuristicBackend(None, params).compute(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(-1.0, 0.0, 0.0),
        instruction='turn around',
    ))

    assert decision.status == 'rotate_to_goal'
    assert decision.linear_x == 0.0
    assert abs(decision.angular_z) <= 0.25
    assert decision.debug['arc_turn'] is False
    assert decision.debug['pure_rotate_max_angular'] == 0.25


def test_heuristic_backend_arcs_through_large_but_recoverable_heading_error():
    params = {
        'max_linear': 1.0,
        'max_angular': 1.5,
        'k_lin': 1.0,
        'k_ang': 2.0,
        'goal_tolerance': 0.35,
        'angle_tolerance': 0.25,
        'min_lin_when_aligned': 0.05,
    }
    decision = HeuristicBackend(None, params).compute(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(0.0, 1.0, 0.0),
        instruction='turn and drive',
    ))

    assert decision.status == 'arc_to_goal'
    assert decision.linear_x > 0.0
    assert abs(decision.angular_z) <= 0.6
    assert decision.debug['arc_yaw_limit'] == 2.4


def test_heuristic_backend_prefers_real_path_subgoal_when_final_goal_is_far():
    params = {
        'max_linear': 1.0,
        'max_angular': 1.5,
        'k_lin': 1.0,
        'k_ang': 2.0,
        'goal_tolerance': 0.35,
        'angle_tolerance': 0.25,
        'min_lin_when_aligned': 0.05,
    }
    decision = HeuristicBackend(None, params).compute(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(10.0, 10.0, 0.0),
        subgoal=Pose2D(1.2, 0.0, 0.0),
        instruction='follow the path',
    ))

    assert decision.status == 'drive_to_goal'
    assert decision.linear_x > 0.0
    assert decision.debug['target_source'] == 'subgoal'
    assert decision.debug['final_goal_distance'] > 1.0


def test_heuristic_backend_does_not_chase_reached_subgoal():
    params = {
        'max_linear': 1.0,
        'max_angular': 1.5,
        'k_lin': 1.0,
        'k_ang': 2.0,
        'goal_tolerance': 0.35,
        'angle_tolerance': 0.25,
        'min_lin_when_aligned': 0.05,
    }
    decision = HeuristicBackend(None, params).compute(DualVLNObservation(
        pose=Pose2D(0.0, 0.0, 0.0),
        goal=Pose2D(10.0, 10.0, 0.0),
        subgoal=Pose2D(0.5, 0.0, 0.0),
        instruction='follow the path',
    ))

    assert decision.debug['target_source'] == 'goal'


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


def test_debug_overlay_renders_current_action_visualization():
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
            'effective_action_label': 'turn_right',
        },
    )
    overlay = render_debug_overlay(image, observation, decision, backend_name='python_adapter')
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert int(overlay.sum()) > 0
