"""Regression tests: a failing startup parameter callback must be LOUD.

What broke, measured
--------------------
Four evaluation cases ask for worlds named ``grscenes_<i>_v1``.  A validation
overlay built without ``arena_simulation_setup`` makes ``ASS_DIR`` fall back to the
container's base install, whose ``worlds/`` still holds the 30 *pre-rename*
``grscenes_<i>_start_result_4200af90`` directories and none of the ``_v1`` names.
So ``grscenes_20_v1/map/map.yaml`` does not exist and
``WorldManagerROS._shift_map`` raises ``FileNotFoundError`` on its first statement.

That error is instant and precise.  Three evaluation attempts nonetheless burned
**599.9 wall seconds each** and then reported ``RuntimeError('Initial world
geometry did not report ready...')`` -- naming a subsystem that had never started.
Three separate defects converted an instant, precise error into ten minutes of
silence and a misleading message:

D1  ``arena_rclpy_mixins/ROSParamServer.py`` dispatched the *initial* parameter
    callback through ``executor.create_task(...)`` with no ``try/except``, while
    every *later* set went through ``ROSParamServer._callback``, which wraps the
    same callback and returns a formatted traceback.  ``rclpy`` re-raises a failed
    Task only on a *later* return from ``wait_for_ready_callbacks``
    (``rclpy/executors.py``), and ``task_generator_node`` at that moment has no
    timers and no inbound traffic, so that return never comes.  The exception was
    lost entirely.

D2  ``TaskGenerator.wait_for_world_geometry_ready`` waited 600 s in one opaque
    ``asyncio.wait_for``, logged nothing while waiting, and read
    ``_world_geometry_error`` only *before* waiting -- guaranteed too early,
    because the producer runs concurrently on an executor thread.

D3  ``FallbackResolver.resolve`` returns a path with no existence check, so
    ``Identifier.resolve_path``'s "not found among <resolvers>" diagnostic was
    unreachable for worlds and a missing world surfaced as an ``open()`` failure
    naming ``map.yaml``.

Test-design constraints, deliberate
-----------------------------------
* **No timers on the node under test.** A predecessor lane's probe added a 0.1 s
  timer "so ``wait_for_ready_callbacks`` returns often" and thereby produced a
  *false refutation of the correct answer*: the presence of a timer is exactly the
  variable that decides whether the exception ever surfaces.
  :func:`_assert_production_topology` fails the test if the node under test has
  any timer, subscription, service or client.
* **Every assertion is functional.** Each test that targets a defect fails against
  the unfixed code on a *number* or a *substring*, never on ``AttributeError``.
  The two exceptions are labelled ``structural`` in their own docstrings and named
  in the lane's pre-registration.
* **Anti-vacuity.** Tests that assert "N records were produced" assert ``N >= 1``
  against a recorder that is proven to work in the same test, and the latency test
  asserts elapsed time, so a test that never ran the code cannot pass.
* **Over-reach guards.** ``test_no_executor_branch_still_raises`` and
  ``test_missing_world_still_resolvable_for_generation`` pass against the *unfixed*
  code too; they exist to fail if the fix goes too far.
"""

import asyncio
import inspect
import threading
import time
from pathlib import Path

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor

from arena_rclpy_mixins import ROSParamServer
from arena_simulation_setup.tree import FallbackResolver
from arena_simulation_setup.tree.World import World as _WorldModule
from arena_simulation_setup.tree.World.World import WorldIdentifier
from task_generator.manager.world_manager.world_manager_ros import WorldManagerROS
from task_generator.node import TaskGenerator

#: A world name that cannot exist in any install.  Used instead of monkeypatching
#: ``ASS_DIR`` so the error messages under test name the *real* search roots.
ABSENT_WORLD = 'lane_a3_world_that_does_not_exist'


# --------------------------------------------------------------------------- #
# Recorders
# --------------------------------------------------------------------------- #


class SpyLogger:
    """Records what the code under test logs, and at which level.

    The code under test calls ``self.get_logger().error(...)`` /
    ``.warn(...)`` / ``self._logger.info(...)``, so recording the logger records
    exactly the contract.  Every message is also kept in :attr:`all`, so a test
    whose level of interest is empty can still prove the recorder itself worked.

    The record lists are exposed as *properties* rather than attributes named
    ``error``/``warn``: an earlier version of this helper stored them as instance
    attributes with those names, which shadowed the logging methods and made every
    call raise ``TypeError: 'list' object is not callable``.  A recorder that
    cannot record is indistinguishable from silence, so the shapes are kept apart.
    """

    def __init__(self) -> None:
        self._records: list[tuple[str, str]] = []

    def _record(self, level: str, message) -> None:
        self._records.append((level, str(message)))

    def error(self, message, *args, **kwargs):
        del args, kwargs
        self._record('error', message)

    def warn(self, message, *args, **kwargs):
        del args, kwargs
        self._record('warn', message)

    # rclpy's logger exposes both spellings
    warning = warn

    def info(self, message, *args, **kwargs):
        del args, kwargs
        self._record('info', message)

    def debug(self, message, *args, **kwargs):
        del args, kwargs
        self._record('debug', message)

    def at(self, level: str) -> list[str]:
        return [text for lvl, text in self._records if lvl == level]

    @property
    def errors(self) -> list[str]:
        return self.at('error')

    @property
    def warns(self) -> list[str]:
        return self.at('warn')

    @property
    def all(self) -> list[tuple[str, str]]:
        return list(self._records)


def _fresh_spy() -> SpyLogger:
    return SpyLogger()


# --------------------------------------------------------------------------- #
# D1 -- a startup parameter callback that raises must be reported at once
# --------------------------------------------------------------------------- #


@pytest.fixture
def ros_context():
    """A real rclpy context, so the Task dispatch under test is the real one."""
    context = rclpy.Context()
    context.init()
    try:
        yield context
    finally:
        try:
            context.try_shutdown()
        except Exception:  # pragma: no cover - shutdown races are not the subject
            pass


class _ParamProbeNode(ROSParamServer):
    """Minimal real ``ROSParamServer``: no timers, no topics, no services.

    Deliberately bare.  See the module docstring: node traffic is the variable
    that decides whether a lost Task exception ever surfaces, so a probe that adds
    any is measuring something other than production.
    """


def _assert_production_topology(node, context) -> None:
    """Fail if the probe node is busier than ``task_generator_node`` is.

    Only a **timer** can wake a quiet executor by itself, and a timer is precisely
    what made a predecessor lane's probe surface the exception in 0.57 s and report
    a false refutation.  Subscriptions, clients and services only wake the executor
    when traffic arrives, and none is produced here.

    Every ``rclpy.node.Node`` also registers seven built-in parameter services, so
    "zero services" is not a reachable state and would be the wrong bar.  The count
    is therefore compared against a bare node created in the same context, which is
    a positive control on the comparison itself.
    """
    timers = list(node.timers)
    subscriptions = list(node.subscriptions)
    clients = list(node.clients)
    assert not timers, (
        f'probe node has {len(timers)} timer(s); a timer makes '
        'wait_for_ready_callbacks return and would hide the defect under test'
    )
    assert not subscriptions, f'probe node has {len(subscriptions)} subscription(s)'
    assert not clients, f'probe node has {len(clients)} client(s)'

    baseline = rclpy.node.Node('lane_a3_bare_baseline', context=context)
    try:
        expected = len(list(baseline.services))
    finally:
        baseline.destroy_node()
    actual = len(list(node.services))
    assert actual == expected, (
        f'probe node has {actual} service(s) against a bare node\'s {expected}; '
        'it advertises something production does not'
    )


def _dispatch_failing_world_callback(node, path: str):
    """Register a ``world`` parameter whose callback raises, the production way.

    Returns the exception instance the callback raised, for message assertions.
    """
    raised: list[BaseException] = []

    def _callback(value):
        error = FileNotFoundError(2, 'No such file or directory', path)
        raised.append(error)
        raise error

    node.declare_parameter('world', 'grscenes_20_v1')
    node.rosparam[str].callback('world', _callback)
    return raised


def _spin_production_shaped(executor, seconds: float = 2.0) -> None:
    """Spin the way production spins: ``spin()`` on a quiet node, no timeout.

    This distinction is the whole test.  ``spin_once(timeout_sec=0.05)`` passes a
    timeout to ``wait_for_ready_callbacks``, so it returns even with no work
    pending, reaches ``executors.py:1006`` and re-raises the lost Task exception --
    which makes the defect invisible and is the same mistake, in a different guise,
    as the predecessor lane's 0.1 s timer.  ``spin()`` calls
    ``_spin_once_impl(None)``, whose wait is unbounded, so on a node with no timers
    and no traffic the loop blocks forever and ``future.result()`` is never reached
    again.  That is production.

    Shutdown does not rescue the exception either: ``wait_for_ready_callbacks``
    then raises ``ShutdownException``, which ``_spin_once_impl`` swallows before the
    ``else:`` branch that inspects futures.
    """
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        time.sleep(seconds)
    finally:
        executor.shutdown(timeout_sec=1.0)
        thread.join(timeout=5.0)


def test_startup_param_callback_failure_is_logged_with_traceback(ros_context):
    """D1 power: the failure must be reported when it happens.

    Unfixed: the exception vanishes into an unretrieved ``Task`` and **zero**
    records are produced, so this fails on ``len(spy.error) == 0``.
    """
    node = _ParamProbeNode('lane_a3_d1_log_probe', context=ros_context)
    spy = _fresh_spy()
    node.get_logger = lambda: spy  # type: ignore[method-assign]
    executor = MultiThreadedExecutor(context=ros_context, num_threads=2)
    executor.add_node(node)
    try:
        _assert_production_topology(node, ros_context)
        assert node.executor is not None, 'the executor branch must be the one taken'

        missing = '/opt/arena_ws/install/.../worlds/grscenes_20_v1/map/map.yaml'
        raised = _dispatch_failing_world_callback(node, missing)
        _spin_production_shaped(executor)

        # Anti-vacuity: prove the callback actually ran, so "nothing was logged"
        # cannot be "nothing happened".  True both before and after the fix.
        assert len(raised) == 1, (
            'the parameter callback never ran, so this test proves nothing about '
            'reporting; the harness is broken, not the code'
        )
        assert spy.errors, (
            'a startup parameter callback raised and NOTHING was logged at error '
            f'level; recorder saw {len(spy.all)} record(s) in total'
        )
        joined = '\n'.join(spy.errors)
        assert 'world' in joined, joined
        assert 'FileNotFoundError' in joined, joined
        assert missing in joined, joined
        assert 'Traceback (most recent call last)' in joined, (
            'the report must carry the callback\'s own traceback, not just its '
            f'message:\n{joined}'
        )
    finally:
        executor.remove_node(node)
        node.destroy_node()


def test_startup_param_callback_failure_is_recorded_for_later_waiters(ros_context):
    """D1 power: the failure must also be *observable*, not only printed.

    Unfixed: nothing records it, so ``param_callback_failures`` is empty and this
    fails on ``0 == 1``.  ``getattr`` with a default keeps the failure functional
    rather than an ``AttributeError``.
    """
    node = _ParamProbeNode('lane_a3_d1_record_probe', context=ros_context)
    node.get_logger = lambda: _fresh_spy()  # type: ignore[method-assign]
    executor = MultiThreadedExecutor(context=ros_context, num_threads=2)
    executor.add_node(node)
    try:
        _assert_production_topology(node, ros_context)
        missing = '/opt/arena_ws/install/.../worlds/grscenes_20_v1/map/map.yaml'
        raised = _dispatch_failing_world_callback(node, missing)
        _spin_production_shaped(executor)

        assert len(raised) == 1, (
            'the parameter callback never ran, so this test proves nothing'
        )
        failures = tuple(getattr(node, 'param_callback_failures', ()))
        assert len(failures) == 1, (
            'a startup parameter callback raised and left no record a later '
            f'waiter could find; recorded {len(failures)}'
        )
        failure = failures[0]
        assert failure.param_name == 'world'
        assert missing in failure.traceback_text
        assert 'FileNotFoundError' in failure.summary()
        assert 'world' in failure.summary()
    finally:
        executor.remove_node(node)
        node.destroy_node()


def test_no_executor_branch_still_raises(ros_context):
    """Over-reach guard: the no-executor branch must keep propagating.

    Without an executor the callback runs on the caller's stack, so its exception
    reaches a caller that can act on it -- that path is already loud and wrapping
    it would make diagnostics *worse*.  This test passes against the unfixed code
    too; it exists so that "removing the remaining asymmetry" fails here.
    """
    node = _ParamProbeNode('lane_a3_d1_direct_probe', context=ros_context)
    node.get_logger = lambda: _fresh_spy()  # type: ignore[method-assign]
    try:
        assert node.executor is None, 'this test is about the no-executor branch'
        with pytest.raises(FileNotFoundError):
            _dispatch_failing_world_callback(node, '/some/missing/map.yaml')
    finally:
        node.destroy_node()


# --------------------------------------------------------------------------- #
# D2 -- the world-geometry wait must observe a dead producer and say something
# --------------------------------------------------------------------------- #


class _GeometryWaiter:
    """Borrows the real waiter methods, with test-scale intervals.

    The methods are the production code objects, taken straight off
    ``TaskGenerator``; only the two cadence constants are shortened so the test
    does not have to sit through 30 s to observe one progress line.
    ``test_wait_cadence_constants_are_usable`` pins the production values.
    """

    _WORLD_GEOMETRY_POLL_INTERVAL_SEC = 0.05
    _WORLD_GEOMETRY_PROGRESS_INTERVAL_SEC = 0.10

    wait_for_world_geometry_ready = TaskGenerator.wait_for_world_geometry_ready
    _world_geometry_producer_failure = getattr(
        TaskGenerator, '_world_geometry_producer_failure', None)
    _world_geometry_wait_diagnosis = getattr(
        TaskGenerator, '_world_geometry_wait_diagnosis', None)

    def __init__(self, failures=(), spawn_started=False, error=''):
        self._world_geometry_ready = asyncio.Event()
        self._world_geometry_error = error
        self._world_geometry_spawn_started = spawn_started
        self.param_callback_failures = tuple(failures)
        self.spy = _fresh_spy()

    def get_logger(self):
        return self.spy


def _recorded_failure(param_name: str, path: str):
    """A ``ParamCallbackFailure``-shaped record, built without importing it.

    Structural duck-typing keeps this test functional against the unfixed code:
    importing the new type at module scope would turn the whole file into one
    ``ImportError``, which is not a functional failure.
    """
    error = FileNotFoundError(2, 'No such file or directory', path)
    traceback_text = (
        'Traceback (most recent call last):\n'
        '  File "world_manager_ros.py", line 127, in _shift_map\n'
        f'FileNotFoundError: [Errno 2] No such file or directory: {path!r}\n'
    )

    class _Failure:
        def __init__(self):
            self.param_name = param_name
            self.value = 'grscenes_20_v1'
            self.exception = error
            self.traceback_text = traceback_text

        def summary(self):
            return (
                f'parameter {param_name!r} callback raised '
                f'FileNotFoundError: {error}'
            )

    return _Failure()


def test_wait_returns_immediately_when_its_producer_already_died():
    """D2 power: latency.

    Unfixed: ``_world_geometry_error`` is empty and the recorded parameter-callback
    failure is never consulted, so the wait burns its whole budget.  This fails on
    elapsed time.
    """
    missing = '/opt/arena_ws/install/x/worlds/grscenes_20_v1/map/map.yaml'
    waiter = _GeometryWaiter(failures=(_recorded_failure('world', missing),))

    started = time.monotonic()
    ready = asyncio.run(waiter.wait_for_world_geometry_ready(timeout_s=6.0))
    elapsed = time.monotonic() - started

    assert ready is False
    assert elapsed < 2.0, (
        f'the wait took {elapsed:.2f}s to notice that the thing it waits for can '
        'never arrive; its producer had already failed before it started'
    )


def test_wait_names_the_real_cause_when_its_producer_died():
    """D2 power: message content.

    Unfixed: the wait logs nothing at all in this scenario, so this fails on
    ``len(spy.error) == 0`` -- not on a missing attribute.
    """
    missing = '/opt/arena_ws/install/x/worlds/grscenes_20_v1/map/map.yaml'
    waiter = _GeometryWaiter(failures=(_recorded_failure('world', missing),))

    asyncio.run(waiter.wait_for_world_geometry_ready(timeout_s=6.0))

    assert waiter.spy.errors, (
        'the wait failed and said nothing at error level; recorder saw '
        f'{len(waiter.spy.all)} record(s)'
    )
    joined = '\n'.join(waiter.spy.errors)
    assert 'world' in joined, joined
    assert missing in joined, joined
    assert 'geometry' in joined.lower(), joined
    assert 'Traceback (most recent call last)' in joined, (
        'the wait must carry the producer\'s traceback, so the reader does not '
        f'have to go looking for it:\n{joined}'
    )


def test_wait_reports_progress_while_it_waits():
    """D2 power: a long wait must not be silent.

    Unfixed: one opaque ``asyncio.wait_for`` produces zero records for the whole
    budget.  This fails on a count.
    """
    waiter = _GeometryWaiter()

    ready = asyncio.run(waiter.wait_for_world_geometry_ready(timeout_s=0.8))

    assert ready is False
    progress = [m for m in waiter.spy.warns if 'Still waiting' in m]
    assert progress, (
        'a wait that did not complete produced no progress record; recorder saw '
        f'{len(waiter.spy.all)} record(s)'
    )
    assert any('elapsed=' in m and 'remaining=' in m for m in progress), progress
    assert any('spawn_started=' in m for m in progress), progress


def test_wait_still_returns_true_when_geometry_becomes_ready():
    """Over-reach guard: the success path must be unchanged."""
    waiter = _GeometryWaiter()

    async def _drive():
        async def _set_soon():
            await asyncio.sleep(0.1)
            waiter._world_geometry_ready.set()

        setter = asyncio.ensure_future(_set_soon())
        result = await waiter.wait_for_world_geometry_ready(timeout_s=5.0)
        await setter
        return result

    assert asyncio.run(_drive()) is True


def test_wait_still_reports_a_spawn_error():
    """Over-reach guard: the pre-existing ``_world_geometry_error`` path survives."""
    waiter = _GeometryWaiter(error='world_geometry_spawn_failed', spawn_started=True)

    started = time.monotonic()
    ready = asyncio.run(waiter.wait_for_world_geometry_ready(timeout_s=6.0))
    elapsed = time.monotonic() - started

    assert ready is False
    assert elapsed < 2.0
    assert any('world_geometry_spawn_failed' in m for m in waiter.spy.errors), \
        waiter.spy.all


def test_wait_cadence_constants_are_usable():
    """The production cadence must actually produce records inside the budget.

    Read with ``getattr`` defaults so that against the unfixed code this fails on
    ``None`` -- "there is no cadence, the wait is one opaque sleep" -- rather than
    on an ``AttributeError``, which says nothing about behaviour.
    """
    poll = getattr(TaskGenerator, '_WORLD_GEOMETRY_POLL_INTERVAL_SEC', None)
    progress = getattr(TaskGenerator, '_WORLD_GEOMETRY_PROGRESS_INTERVAL_SEC', None)
    assert poll is not None, (
        'the wait has no poll cadence, so it cannot re-check its producer'
    )
    assert progress is not None, (
        'the wait has no progress cadence, so it cannot say anything while waiting'
    )
    assert 0 < poll <= 10.0, poll
    assert 0 < progress <= 60.0, progress
    # 600.0 is world_geometry_ready_timeout_sec's default (node.py).
    assert progress < 600.0 / 4, (
        'a progress interval this long cannot make a 600 s wait observable'
    )


def test_timeout_message_names_the_real_subsystem():
    """D2 power, *structural* (declared in the pre-registration).

    Asserts the raise site no longer emits the fixed string that named world
    *geometry* for a failure in which geometry spawn never started.  Fails against
    the unfixed source on a substring.
    """
    source = inspect.getsource(TaskGenerator._set_up_managers)
    assert 'Initial world geometry did not report ready before robot manager setup.' \
        not in source, (
            'the raise site still emits the message that sent three diagnoses to '
            'the wrong subsystem'
        )
    assert '_world_geometry_wait_diagnosis' in source, (
        'the raise site must interpolate a cause, not restate the symptom'
    )


def test_diagnosis_distinguishes_never_started_from_did_not_finish():
    """The two failures have different places to look, so they must read differently."""
    diagnose = getattr(TaskGenerator, '_world_geometry_wait_diagnosis', None)
    assert diagnose is not None, 'no diagnosis is produced at all'

    never_started = diagnose(_GeometryWaiter(spawn_started=False), 600.0)
    did_not_finish = diagnose(_GeometryWaiter(spawn_started=True), 600.0)

    assert never_started != did_not_finish
    assert 'never reached geometry spawn' in never_started, never_started
    assert 'did not complete' in did_not_finish, did_not_finish


# --------------------------------------------------------------------------- #
# D3 -- a missing world must be named as a missing world
# --------------------------------------------------------------------------- #


class _WorldCallbackHarness:
    """Borrows the real ``_world_callback`` and ``_shift_map``.

    Only the collaborators those two statements actually touch before the failure
    point are provided, so the harness cannot accidentally pass by stubbing out the
    code under test.
    """

    _world_callback = WorldManagerROS._world_callback
    _shift_map = WorldManagerROS._shift_map

    def __init__(self):
        self._world_name = ''
        self._origin = None
        self._logger = _fresh_spy()


def test_missing_world_is_reported_as_a_missing_world():
    """D3 power: the error must be about the world, not about ``map.yaml``.

    Unfixed: ``FallbackResolver`` hands back a non-existent path, the code walks on
    and fails inside ``_shift_map``'s ``open()``, so the message reads
    ``[Errno 2] No such file or directory: '.../map/map.yaml'`` -- it names the map
    file and never says which world is missing or where it was looked for.  This
    fails on substrings, not on an attribute.
    """
    harness = _WorldCallbackHarness()

    with pytest.raises(FileNotFoundError) as excinfo:
        harness._world_callback(ABSENT_WORLD)

    message = str(excinfo.value)
    assert 'not found among' in message, (
        'a missing world is still reported as a bare file-open failure:\n'
        f'{message}'
    )
    assert ABSENT_WORLD in message, message
    assert 'worlds' in message, (
        'the message must name the directory that was searched:\n' f'{message}'
    )
    assert 'map.yaml' not in message, (
        'the message still points at map.yaml instead of at the missing world:\n'
        f'{message}'
    )


def test_resolve_path_lists_every_searched_root_and_candidate():
    """D3 power, *structural* (declared in the pre-registration).

    ``require_exists`` does not exist in the unfixed API, so this fails with
    ``TypeError`` there.  It is kept separate from the functional tests for that
    reason.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        WorldIdentifier(ABSENT_WORLD).resolve_sync(require_exists=True)

    message = str(excinfo.value)
    assert ABSENT_WORLD in message, message
    assert 'FallbackResolver' in message, message
    assert '(does not exist)' in message, (
        'the message must state that a candidate path was produced and was not '
        f'there, otherwise a reader cannot tell where to look:\n{message}'
    )
    # The candidate path must be shown, so a wrong ASS_DIR is visible at a glance.
    assert str(Path('worlds') / ABSENT_WORLD) in message, message


def test_missing_world_still_resolvable_for_generation():
    """Over-reach guard: asset *generation* resolves a path that does not exist yet.

    ``arena_simulation_setup/utils/generative/world_generator.py:13`` does
    ``WorldIdentifier(out).resolve_sync().save(...)`` and ``World.save`` then does
    ``os.makedirs(..., exist_ok=True)``.  If the fix made existence unconditional,
    world generation would break.  Passes against the unfixed code too.
    """
    world = WorldIdentifier(ABSENT_WORLD).resolve_sync()
    assert world.path.name == ABSENT_WORLD
    assert not world.path.exists(), (
        'this test is only meaningful for a world that is genuinely absent'
    )


def test_fallback_resolver_documents_why_it_stays_permissive():
    """The permissiveness is deliberate; it must not read like an oversight."""
    doc = inspect.getdoc(FallbackResolver) or ''
    assert 'require_exists' in doc, (
        'FallbackResolver returns a non-existent path by design; its docstring '
        'must point readers at the opt-in strict path'
    )


def test_world_read_path_asks_for_existence():
    """The read path must opt in, or D3's diagnostic is unreachable again.

    Structural, and deliberately so: this is the one-line coupling between the
    generic mechanism and the caller that needs it, and nothing else in the file
    would notice if it were removed.
    """
    source = inspect.getsource(WorldManagerROS._world_callback)
    assert 'require_exists=True' in source, (
        'WorldManagerROS._world_callback resolves the world without requiring it '
        'to exist, so a missing world will again be reported as an open() failure'
    )
    assert _WorldModule is not None  # import is load-bearing for the harness above
