"""Shell-level fault injection for the stable lane wrapper.

DRY_RUN=1 exercises the stage->exit-code mapping without Docker, Isaac, a
model, or ROS.  Each fault gets its own temporary lane root and run tag.
"""
import os
import subprocess
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'run_prox_qwen_lane.sh'


def _run(fault: str) -> int:
    lane = tempfile.mkdtemp(prefix=f'bfast_lane_{fault}_')
    env = os.environ.copy()
    env.update({
        'DRY_RUN': '1',
        'FAULT_INJECT': fault,
        'LANE_ROOT': lane,
        'RUN_TAG': f'test_{fault}_{os.getpid()}',
        'LOCK': os.path.join(lane, 'lock'),
        'MODE': 'diag',
    })
    result = subprocess.run(['bash', str(SCRIPT)], env=env,
                            capture_output=True, text=True)
    return result.returncode


def test_stage_fault_injection_exit_codes():
    assert _run('none') == 0
    assert _run('model_start') == 22
    assert _run('readiness') == 21
    assert _run('eval') == 3
    assert _run('render') == 2
