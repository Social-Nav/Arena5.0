"""Package metadata contracts used by colcon's ament-python test runner."""

import ast
from pathlib import Path


def _setup_keywords():
    setup_path = Path(__file__).resolve().parents[1] / 'setup.py'
    tree = ast.parse(setup_path.read_text(encoding='utf-8'))
    setup_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'setup'
    ]
    assert len(setup_calls) == 1
    return {keyword.arg: keyword.value for keyword in setup_calls[0].keywords}


def test_colcon_selects_pytest_from_standard_test_extra():
    keywords = _setup_keywords()
    extras = ast.literal_eval(keywords['extras_require'])

    assert extras['test'] == ['pytest']
    assert 'tests_require' not in keywords
