"""The launch arguments this driver builds must be accepted by ros2launch.

`ros2launch` rejects a trailing `:=` outright -- `parse_launch_arguments` raises
`RuntimeError` on `count == 1 and argument.endswith(':=')` -- and it does so while
parsing the command line, before the launch service starts.  So an argument that
is emitted unconditionally from an argparse option whose default is the empty
string makes the whole evaluation unrunnable the moment nobody passes that flag.

That is not hypothetical: `pedestrian_goal_traversal:={args.pedestrian_goal_traversal}`
was emitted unconditionally with `default=''`, which aborted every fresh-overlay
run at HEAD before launch.  A batch survived only because it pinned an older
overlay.  The invariant below is the general form of that defect, so the next
option with an empty default is caught by name rather than by a dead run.

Everything is read from the shipped source with :mod:`ast`; `main()` is a single
long function that starts subprocesses, so it is not called here.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / 'arena_bringup' / 'internnav_eval.py'
)
ROBOT_LAUNCH = (
    Path(__file__).resolve().parents[2]
    / 'arena_simulation_setup'
    / 'launch'
    / 'robot.launch.py'
)

#: `name:=` with nothing after it is what ros2launch refuses.
_ARG_NAME = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):=$')


def _main_function():
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            return node
    raise AssertionError('internnav_eval.py has no main()')


def _argparse_defaults(main_node):
    """dest -> default, for every string-valued option main() declares."""
    defaults = {}
    for node in ast.walk(main_node):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        long_flags = [f for f in flags if isinstance(f, str) and f.startswith('--')]
        if not long_flags:
            continue
        dest = long_flags[0][2:].replace('-', '_')
        for kw in node.keywords:
            if kw.arg == 'dest' and isinstance(kw.value, ast.Constant):
                dest = kw.value.value
        for kw in node.keywords:
            if kw.arg == 'default':
                if isinstance(kw.value, ast.Constant):
                    defaults[dest] = kw.value.value
                else:
                    defaults[dest] = '<computed>'
    return defaults


def _unconditional_launch_arguments(main_node):
    """(launch_arg_name, args_attribute) for entries of the `launch_cmd = [...]` literal.

    Only the list literal is inspected.  Everything appended later sits behind an
    `if`, which is exactly the shape that makes an empty value safe.
    """
    for node in ast.walk(main_node):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.List)):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == 'launch_cmd' for t in node.targets
        ):
            continue
        found = []
        for element in node.value.elts:
            if not isinstance(element, ast.JoinedStr):
                continue
            parts = element.values
            if not parts or not isinstance(parts[0], ast.Constant):
                continue
            match = _ARG_NAME.match(str(parts[0].value))
            if not match:
                continue
            # The single interpolated value, when it is a plain `args.<dest>`.
            interpolations = [p for p in parts if isinstance(p, ast.FormattedValue)]
            if len(interpolations) != 1:
                continue
            value = interpolations[0].value
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == 'args'
            ):
                found.append((match.group(1), value.attr))
        return found
    raise AssertionError('main() has no `launch_cmd = [...]` list literal')


def test_no_unconditionally_emitted_launch_argument_can_be_empty():
    """The invariant: unconditional emission requires a non-empty default."""
    main_node = _main_function()
    defaults = _argparse_defaults(main_node)
    offenders = {}
    for launch_name, dest in _unconditional_launch_arguments(main_node):
        if dest not in defaults:
            continue
        default = defaults[dest]
        if default == '' or default is None:
            offenders[launch_name] = f'args.{dest} defaults to {default!r}'
    assert not offenders, (
        'these launch arguments are emitted unconditionally but default to an '
        f'empty value, so ros2launch would reject the command line: {offenders}. '
        'Append them behind an `if args.<dest>:` instead, as the block after '
        '`launch_cmd = [...]` already does for every optional argument.'
    )


def test_pedestrian_goal_traversal_is_emitted_only_when_requested():
    """The specific regression, pinned by name.

    It came in with the reciprocate work and is already pushed, so it is worth an
    assertion of its own rather than relying on the general invariant above.
    """
    main_node = _main_function()
    unconditional = dict(_unconditional_launch_arguments(main_node))
    assert 'pedestrian_goal_traversal' not in unconditional, (
        'pedestrian_goal_traversal is back in the unconditional launch_cmd list; '
        "its default is '' and ros2launch rejects a trailing ':='"
    )
    source = SOURCE.read_text()
    assert 'if args.pedestrian_goal_traversal:' in source, (
        'the conditional append that replaced it is gone'
    )


@pytest.mark.parametrize(
    'argument, accepted',
    [
        ('pedestrian_goal_traversal:=', False),
        ('pedestrian_goal_traversal:=once', True),
        ('pedestrian_goal_traversal', False),
    ],
)
def test_ros2launch_really_rejects_a_trailing_assignment(argument, accepted):
    """Positive control for the premise, using ros2launch's own parser.

    Without this the invariant above rests on a remembered claim about another
    package's behaviour.
    """
    parse = pytest.importorskip(
        'ros2launch.api.api', reason='ros2launch is only present in the arena container'
    ).parse_launch_arguments
    if accepted:
        assert list(parse([argument]))
    else:
        with pytest.raises(RuntimeError):
            parse([argument])


def test_shared_robot_launch_requires_external_internnav_service():
    source = ROBOT_LAUNCH.read_text(encoding='utf-8')

    assert "executable='dual_vln_server'" not in source
    assert 'require_external_internnav' in source
    assert 'internnav_external_server:=true' in source
    assert 'internnav_async_eval.launch.py' in source


def test_shared_robot_launch_external_internnav_guard():
    spec = importlib.util.spec_from_file_location('arena_robot_launch', ROBOT_LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check = module._requires_missing_local_internnav_server
    defaults = {
        'local_planner': 'dual_vln',
        'train_mode': 'false',
        'internnav_external_server': 'false',
        'dual_vln_external_server': 'false',
        'internnav_direct_cmd_vel': 'false',
        'dual_vln_direct_cmd_vel': 'false',
        'env_external_server': '',
    }

    assert check(**defaults) is True
    assert check(**{**defaults, 'internnav_external_server': 'true'}) is False
    assert check(**{**defaults, 'internnav_direct_cmd_vel': 'true'}) is False
    assert check(**{**defaults, 'local_planner': 'dwb'}) is False
    assert check(**{**defaults, 'train_mode': 'true'}) is False
