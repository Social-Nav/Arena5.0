"""Static/CPU wiring checks for the stable lane script and B guard node."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arena_bringup.bfast_guard_node import decode_rgb


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / 'scripts' / 'run_prox_qwen_lane.sh'
GUARD = ROOT / 'arena_bringup' / 'bfast_guard_node.py'
EXECUTOR = ROOT.parent / 'arena_vln_models' / 'arena_vln_models' / 'executor.py'


def test_stable_lane_script_syntax_and_safety_guards():
    result = subprocess.run(['bash', '-n', str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = SCRIPT.read_text()
    assert 'mkdir "$RUN_DIR"' in text
    assert 'OUTPUT_DIR_EXISTS' in text
    assert 'MODEL_START_FAILED' in text
    assert 'NETWORK_RESOLVE_FAILED' in text
    assert 'EXPECTED_CONTAINER_IP' in text
    assert 'cleanup_errors.log' in text
    assert 'OWNED_PIDS.txt' in text
    assert 'RUN_OWNER.json' in text
    assert 'RUN_RESULT.json' in text
    assert 'isaac_pid_ambiguous' in text
    assert 'pid_identity_mismatch' in text
    assert 'pkill -f' not in text


def test_bfast_guard_node_is_rgb_only_and_common_executor_hooked():
    guard = GUARD.read_text()
    executor = EXECUTOR.read_text()
    assert 'sensor_msgs.msg import Image' in guard
    assert 'LaserScan' not in guard
    assert 'PointCloud2' not in guard
    assert 'OccupancyGrid' not in guard
    assert 'create_subscription(Image' in guard
    assert 'create_publisher(Bool' in guard
    assert 'create_publisher(String' in guard
    assert 'compute_flow_risk_proxy' in guard
    assert 'fast_guard_veto_topic' in executor
    assert '_fast_guard_veto_active' in executor
    assert 'self.core.set_stop(guard_reason)' in executor
    assert 'bfast_guard_stale' in executor


def _image(data, height, width, step, encoding='rgb8'):
    return SimpleNamespace(
        height=height, width=width, step=step, encoding=encoding,
        data=data, header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)))


def test_decode_rgb_respects_step_and_encoding():
    height, width, step = 2, 3, 3 * 3 + 2  # two padding bytes per row
    raw = bytearray()
    for row in range(height):
        for col in range(width):
            raw.extend((row * 10 + col, 1, 2))
        raw.extend((0, 0))
    image = decode_rgb(_image(bytes(raw), height, width, step, 'rgb8'))
    assert image.shape == (2, 3, 3)
    assert int(image[1, 2, 0]) == 12


def test_decode_rgb_rejects_unsupported_encoding():
    import pytest
    from arena_bringup.paper_pipeline import PipelineRejected
    with pytest.raises(PipelineRejected):
        decode_rgb(_image(b'\x00' * 12, 2, 2, 6, '32FC1'))
