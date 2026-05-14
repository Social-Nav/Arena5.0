from types import SimpleNamespace

import numpy as np

from arena_vln_models.core import (
    camera_intrinsic_matrix,
    normalize_backend_output,
    normalize_depth,
    normalize_rgb,
    pose_vector,
)
from arena_vln_models.backends import HeuristicBackend, Pose2D, PythonAdapterBackend, DualVLNObservation
from arena_vln_models import internnav as internnav_module
from arena_vln_models.internnav import InternNavAdapter, available_backends, load_internnav_adapter
from arena_vln_models.internnav_server import _normalize_internnav_adapter_target
from arena_vln_models.visualization import image_msg_to_numpy, numpy_to_image_msg


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
