"""A solver exception completing after reset must not stop the new episode."""
import ast
from concurrent.futures import Future
from pathlib import Path
import math
import time
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('generation,expected_error', [(2, False), (3, True)])
def test_solver_exception_respects_episode_generation(generation, expected_error):
    source = Path(__file__).resolve().parents[1] / 'arena_vln_models/lightnav_client.py'
    cls = next(n for n in ast.walk(ast.parse(source.read_text()))
               if isinstance(n, ast.ClassDef) and n.name == 'LightNavClient')
    tick = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'tick')
    context = dict(time=time, math=math)
    exec(compile(ast.Module(body=[tick], type_ignores=[]), str(source), 'exec'), context)
    future = Future(); future.set_exception(RuntimeError('old solver failed'))
    errors, zeros = [], []
    path = object()
    state = SimpleNamespace(generation=3, sequence=1, active=True, terminal=False)
    state.invalidate = lambda: setattr(state, 'generation', state.generation + 1)
    state.block_reason = lambda *a, **k: 'control_timeout' if state.generation != 3 else ''
    client = SimpleNamespace(state=state, path=path, pose=object(), sim=10., future=future,
        future_generation=generation, last_heartbeat=time.monotonic(),
        count_publishers=lambda topic: 1,
        native=SimpleNamespace(take_results=lambda: [], snapshot=lambda: SimpleNamespace(state='READY')),
        solve_cadence=SimpleNamespace(due=lambda sim: False),
        publish_cadence=SimpleNamespace(due=lambda sim: False),
        zero=zeros.append, emit=lambda status, **fields: errors.append(status))
    context['tick'](client)
    assert (errors == ['error']) == expected_error
    assert (client.path is None) == expected_error
    assert bool(zeros) == expected_error
    assert client.future is None
