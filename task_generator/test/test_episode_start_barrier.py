"""Functional tests for the episode-start barrier and the pedestrian gate.

Every test here is written to fail against the pre-barrier code for a *behavioural*
reason, not because a name is missing.  The two that carry the load are:

* ``test_pedestrian_clock_holds_then_resumes_with_a_single_tick_step`` -- reproduces
  HuNav's own dt arithmetic (``bt_node.cpp:398-404``) and asserts that no route is
  consumed before release and that the first released step is one tick, not the
  whole gated interval.  The pre-barrier code passes raw ``/clock`` here, which
  makes the pre-release displacement non-zero.
* ``test_reset_publishes_task_reset_before_barrier_and_origin_after`` -- asserts the
  *ordering* of the real ``_reset_task_unlocked``: pedestrians and the timeout
  origin must be released after the stream-readiness report, not before
  ``task_reset``.  The pre-barrier ordering fails it.
"""

import asyncio
import ast
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

if importlib.util.find_spec('rclpy') is None:  # pragma: no cover - host fallback
    def _module(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    rclpy = _module('rclpy')
    rclpy.node = _module('rclpy.node')
    rclpy.node.Node = object

from task_generator.episode_barrier import (  # noqa: E402
    BarrierCondition,
    EpisodeStartBarrierTimeout,
    PedestrianEpisodeClock,
    await_episode_start_barrier,
)


# --------------------------------------------------------------------------- #
# The pedestrian gate
# --------------------------------------------------------------------------- #


def _hunav_route_progress(stamps_sec, speed_mps=1.0):
    """Replay HuNav's integration using its own dt rule, from its own source.

    ``bt_node.cpp:398-404``::

        double time_step_secs = (Time(ag->header.stamp) - prev_time_).seconds();
        if (time_step_secs < 0.0) time_step_secs = 0.0;
        tree_tick(time_step_secs);
        prev_time_ = Time(ag->header.stamp);

    The first call only initialises ``prev_time_`` and returns without ticking
    (``bt_node.cpp:373-384``).

    Returns:
        ``(total_distance_m, per_step_distances)``.
    """
    prev = None
    steps = []
    for stamp in stamps_sec:
        if prev is None:
            prev = stamp
            continue
        dt = stamp - prev
        if dt < 0.0:
            dt = 0.0
        steps.append(dt * speed_mps)
        prev = stamp
    return sum(steps), steps


def test_pedestrian_clock_holds_then_resumes_with_a_single_tick_step():
    """No route before release, and the first released step is one tick."""
    clock = PedestrianEpisodeClock()
    tick = 0.1

    # 25 s of startup at 10 Hz while the barrier is closed.
    stamps = []
    sim = 12.0
    for _ in range(250):
        stamps.append(clock.tick(sim))
        sim += tick

    gated_distance, _ = _hunav_route_progress(stamps)
    assert gated_distance == pytest.approx(0.0, abs=1e-9), (
        'HuNav must make zero route progress before the episode-start barrier; '
        f'got {gated_distance} m'
    )

    # Release and run 10 more ticks.
    clock.release()
    released = []
    for _ in range(10):
        released.append(clock.tick(sim))
        sim += tick

    _total, steps = _hunav_route_progress(stamps + released)
    first_released_step = steps[len(stamps) - 1]
    assert first_released_step == pytest.approx(tick, abs=1e-6), (
        'The first step after release must be one simulator tick, not the whole '
        f'gated interval; got {first_released_step} s worth of motion'
    )
    released_distance, _ = _hunav_route_progress(released)
    assert released_distance == pytest.approx(0.9, abs=1e-6)


def test_skipping_calls_without_the_clock_would_consume_the_route_in_one_step():
    """Control: this is why the fix is a clock and not just "skip the calls".

    Sending raw ``/clock`` after a gap makes HuNav integrate the entire gap in a
    single tick.  A 25 s gap at 1 m/s is 25 m -- more than any route in the
    benchmark (longest measured route: 18.14 m).
    """
    naive_stamps = [12.0, 37.0, 37.1]
    total, steps = _hunav_route_progress(naive_stamps)
    assert steps[0] == pytest.approx(25.0)
    assert total > 18.14, 'the naive approach consumes more than the longest benchmark route'


def test_prefix_stamp_source_consumes_the_route_before_the_barrier():
    """Control on the OLD behaviour: raw ``/clock`` stamps are what broke it.

    ``hunav.py`` used to set the ``compute_agents`` request stamp from
    ``self.node.sim_time``, so during a 25 s startup at 10 Hz HuNav integrated
    24.9 m of route -- more than the longest scenario route in the benchmark
    (18.14 m).  That is the whole defect: the walk was over before the review
    video's frame 0.  Paired with
    ``test_pedestrian_clock_holds_then_resumes_with_a_single_tick_step``, which
    asserts 0.0 m for the same window, this is the before/after measurement.
    """
    raw_clock_stamps = [12.0 + step * 0.1 for step in range(250)]
    distance, _ = _hunav_route_progress(raw_clock_stamps)
    assert distance == pytest.approx(24.9, abs=1e-6)
    assert distance > 18.14


def test_pedestrian_clock_is_monotone_across_a_backwards_clock_step():
    """The recorder's known backwards ``/clock`` steps must not rewind the gate."""
    clock = PedestrianEpisodeClock()
    clock.tick(100.0)
    clock.release()
    assert clock.tick(100.1) == pytest.approx(100.1)
    # /clock jumps backwards, as it does during scene load.
    assert clock.tick(40.0) == pytest.approx(100.1)
    assert clock.tick(40.1) == pytest.approx(100.2)


def test_pedestrian_clock_hold_after_release_freezes_again():
    """Episode rollover must be able to re-arm the gate."""
    clock = PedestrianEpisodeClock()
    clock.tick(5.0)
    clock.release()
    clock.tick(5.5)
    assert clock.value == pytest.approx(5.5)
    clock.hold()
    clock.tick(9.0)
    clock.tick(20.0)
    assert clock.value == pytest.approx(5.5)
    clock.release()
    clock.tick(20.1)
    assert clock.value == pytest.approx(5.6)


def test_pedestrian_clock_episode_elapsed_is_released_time_only():
    clock = PedestrianEpisodeClock()
    clock.tick(30.0)
    for step in range(1, 100):
        clock.tick(30.0 + step * 0.1)
    assert clock.episode_elapsed == pytest.approx(0.0)
    clock.release()
    for step in range(100, 130):
        clock.tick(30.0 + step * 0.1)
    assert clock.episode_elapsed == pytest.approx(3.0, abs=1e-6)


def test_pedestrian_clock_stamp_is_exact_in_nanoseconds():
    """Stamps must be exact: a rounding drift would show up as a bogus dt."""
    clock = PedestrianEpisodeClock()
    clock.tick_ns(123_456_789_012)
    clock.release()
    clock.tick_ns(123_456_789_012 + 33_333_333)
    assert clock.stamp_sec_nanosec() == (123, 490_122_345)


# --------------------------------------------------------------------------- #
# The barrier itself
# --------------------------------------------------------------------------- #


class _FakeTime:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.now += float(seconds)


def test_barrier_waits_for_a_late_condition_then_passes():
    clock = _FakeTime()
    flipped = {'value': False}

    def _late():
        if clock.now >= 1.0:
            flipped['value'] = True
        return flipped['value']

    report = asyncio.run(
        await_episode_start_barrier(
            [
                BarrierCondition(name='immediate', check=lambda: True),
                BarrierCondition(name='late', check=_late, detail=lambda: f't={clock.now}'),
            ],
            timeout_sec=10.0,
            poll_interval_sec=0.25,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )
    assert report.passed is True
    assert report.satisfied == ['immediate', 'late']
    assert report.waited_sec == pytest.approx(1.0)


def test_barrier_raises_with_the_unsatisfied_condition_named():
    clock = _FakeTime()
    with pytest.raises(EpisodeStartBarrierTimeout) as excinfo:
        asyncio.run(
            await_episode_start_barrier(
                [
                    BarrierCondition(name='ok', check=lambda: True),
                    BarrierCondition(
                        name='video_streams_ready',
                        check=lambda: False,
                        detail=lambda: 'ready_episode=None expected_episode=0',
                    ),
                ],
                timeout_sec=2.0,
                poll_interval_sec=0.5,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        )
    report = excinfo.value.report
    assert report.passed is False
    assert report.unsatisfied == ['video_streams_ready']
    message = str(excinfo.value)
    assert 'video_streams_ready' in message
    assert 'ready_episode=None expected_episode=0' in message
    assert 'Refusing to declare an episode origin that was never reached' in message


def test_barrier_never_waits_on_a_not_required_condition_but_reports_it():
    clock = _FakeTime()
    report = asyncio.run(
        await_episode_start_barrier(
            [
                BarrierCondition(name='ok', check=lambda: True),
                BarrierCondition(
                    name='video_streams_ready',
                    check=lambda: False,
                    required=False,
                    skip_reason='no_publisher_on=/task_generator_node/video_streams_ready',
                ),
            ],
            timeout_sec=5.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )
    assert report.passed is True
    assert report.skipped == {
        'video_streams_ready': 'no_publisher_on=/task_generator_node/video_streams_ready'
    }
    assert clock.now == pytest.approx(0.0), 'a not-required condition must not cost any wait'


def test_barrier_treats_a_raising_check_as_unsatisfied_not_as_an_abort():
    clock = _FakeTime()

    def _raises():
        raise RuntimeError('topic has no publisher yet')

    with pytest.raises(EpisodeStartBarrierTimeout) as excinfo:
        asyncio.run(
            await_episode_start_barrier(
                [BarrierCondition(name='flaky', check=_raises)],
                timeout_sec=1.0,
                poll_interval_sec=0.5,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
        )
    assert excinfo.value.report.unsatisfied == ['flaky']


def test_barrier_with_zero_timeout_evaluates_once_and_can_pass():
    clock = _FakeTime()
    report = asyncio.run(
        await_episode_start_barrier(
            [BarrierCondition(name='ok', check=lambda: True)],
            timeout_sec=0.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )
    assert report.passed is True
    assert clock.now == pytest.approx(0.0)


def test_barrier_report_round_trips_to_a_json_safe_dict():
    report = asyncio.run(
        await_episode_start_barrier(
            [BarrierCondition(name='ok', check=lambda: True, detail=lambda: 'ready=True')],
            timeout_sec=1.0,
            monotonic=_FakeTime().monotonic,
        )
    )
    payload = report.to_dict()
    import json

    assert json.loads(json.dumps(payload))['satisfied'] == ['ok']
    assert payload['details'] == {'ok': 'ready=True'}


# --------------------------------------------------------------------------- #
# HunavHumanSimulator's gate binding
# --------------------------------------------------------------------------- #


def _logger_stub(warnings):
    logger = SimpleNamespace(
        warn=warnings.append,
        warning=warnings.append,
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    logger.get_child = lambda _name: logger
    return logger


def _hunav_manager_stub(node, warnings=None):
    """Build a HunavHumanSimulator without running its ROS-dependent __init__."""
    from task_generator.simulators.human.hunav.hunav import HunavHumanSimulator

    warnings = [] if warnings is None else warnings
    node.get_logger = lambda: _logger_stub(warnings)
    manager = HunavHumanSimulator.__new__(HunavHumanSimulator)
    # ``NodeInterface.node`` is a read-only property backed by a name-mangled
    # attribute, matching the pattern already used in test_task_generator_stability.
    manager._NodeInterface__node = node
    manager._pedestrian_clock, manager._pedestrian_gate_source = manager._resolve_pedestrian_clock()
    return manager


def test_hunav_binds_to_the_node_clock_and_honours_the_gate():
    clock = PedestrianEpisodeClock()
    node = SimpleNamespace(
        pedestrian_episode_clock=clock,
        episode_motion_released=False,
        sim_time=SimpleNamespace(sec=7, nanosec=500_000_000),
    )
    manager = _hunav_manager_stub(node)
    assert manager._pedestrian_gate_source == 'task_generator_node'
    assert manager._pedestrian_motion_released() is False

    first = manager._pedestrian_clock_stamp()
    node.sim_time = SimpleNamespace(sec=12, nanosec=0)
    second = manager._pedestrian_clock_stamp()
    assert (first.sec, first.nanosec) == (second.sec, second.nanosec), (
        'a held gate must produce identical consecutive stamps so HuNav sees dt == 0'
    )
    assert (second.sec, second.nanosec) == (7, 500_000_000)

    node.episode_motion_released = True
    node.sim_time = SimpleNamespace(sec=12, nanosec=100_000_000)
    third = manager._pedestrian_clock_stamp()
    dt = (third.sec + third.nanosec / 1e9) - (second.sec + second.nanosec / 1e9)
    assert dt == pytest.approx(0.1, abs=1e-9), (
        f'the first released step must be one 10 Hz tick, got {dt}'
    )


def test_hunav_reports_loudly_and_runs_ungated_without_the_barrier_api():
    warnings = []
    node = SimpleNamespace(sim_time=SimpleNamespace(sec=1, nanosec=0))
    manager = _hunav_manager_stub(node, warnings)

    assert manager._pedestrian_gate_source == 'local_ungated'
    assert manager._pedestrian_motion_released() is True
    assert len(warnings) == 1
    assert 'pedestrian_episode_clock' in warnings[0]
    assert 'UNGATED' in warnings[0]


# --------------------------------------------------------------------------- #
# The real _reset_task_unlocked ordering
# --------------------------------------------------------------------------- #


class _RecordingPublisher:
    def __init__(self, log, label):
        self._log = log
        self._label = label

    def publish(self, msg):
        self._log.append((self._label, getattr(msg, 'data', None)))


class _RosParamAccessor:
    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, _type):
        return self

    def get(self, name, default):
        return self._values.get(name, default)

    def set(self, name, value):
        self._values[name] = value


def _task_generator_stub(events, *, streams_ready_publishers=1):
    """A TaskGenerator wired just far enough to run _reset_task_unlocked."""
    from task_generator.node import TaskGenerator

    node = TaskGenerator.__new__(TaskGenerator)
    node._sim_time = SimpleNamespace(sec=0, nanosec=0)
    node._start_time = None
    node._number_of_resets = 0
    node._episode_entities_ready = asyncio.Event()
    node._human_states_ready = asyncio.Event()
    node._world_geometry_ready = asyncio.Event()
    node._world_geometry_ready.set()
    node._world_geometry_error = ''
    node._episode_started = asyncio.Event()
    node._pedestrian_clock = PedestrianEpisodeClock()
    node._last_barrier_report = None
    node._robot_navigation_ready = False
    node._video_streams_ready_episode = None
    node._last_human_states_count = 0
    node._video_streams_ready_topic = '/task_generator_node/video_streams_ready'
    node._vln_instruction = 'go'
    node._vln_instruction_file = ''
    node._vln_instruction_republish_task = None
    node._robots_manager = SimpleNamespace(robots={})
    node.rosparam = _RosParamAccessor(
        {'episode_start_barrier_timeout_sec': 5.0, 'episode_start_recorder_discovery_sec': 0.0}
    )
    # Use the real enum: comparing against a look-alike stub would silently make
    # the pedestrians_spawned condition "not required" and weaken every assertion
    # below, which is exactly the class of stub-fidelity bug this lane is guarding.
    from task_generator.constants import Constants

    node.conf = SimpleNamespace(
        Arena=SimpleNamespace(HUMAN=SimpleNamespace(value=Constants.HumanSimulator.HUNAV))
    )
    node._pub_task_reset = _RecordingPublisher(events, 'task_reset')
    node._pub_episode_start = _RecordingPublisher(events, 'episode_start')
    node._pub_vln_instruction = _RecordingPublisher(events, 'vln_instruction')
    node._pub_eval_ready = _RecordingPublisher([], 'eval_ready')
    node.get_logger = lambda: _logger_stub([])
    node.count_publishers = lambda _topic: streams_ready_publishers
    node.get_service_names_and_types = lambda: []

    async def _noop():
        return None

    node._simulator = SimpleNamespace(
        before_reset_task=lambda: _noop(),
        after_reset_task=lambda: _noop(),
    )

    def _mark_started():
        events.append(('mark_episode_started', None))

    async def _task_reset(**_kwargs):
        events.append(('task.reset', None))
        # HuNav agents become visible during the task reset, as they do in production.
        node._human_states_ready.set()
        node._last_human_states_count = 2

    async def _wait_navigation_ready(timeout_s):
        del timeout_s
        events.append(('wait_navigation_ready', None))

    node._task = SimpleNamespace(
        reset=_task_reset,
        mark_episode_started=_mark_started,
        wait_navigation_ready=_wait_navigation_ready,
    )

    # The human simulator's gate binds to this node, exactly as in production.
    node._pedestrian_release_probe = []
    return node


def _run_reset_with_late_stream_readiness(events, delay_rounds=3):
    """Run _reset_task_unlocked while the recorder reports readiness late."""
    from task_generator.node import TaskGenerator

    node = _task_generator_stub(events)

    async def _drive():
        async def _late_readiness():
            for _ in range(delay_rounds):
                await asyncio.sleep(0)
                events.append(('pedestrian_released_probe', node.episode_motion_released))
            node._video_streams_ready_episode = node._number_of_resets
            events.append(('video_streams_ready', node._number_of_resets))

        readiness = asyncio.create_task(_late_readiness())
        await TaskGenerator._reset_task_unlocked(node)
        await readiness

    asyncio.run(_drive())
    return node


def test_reset_publishes_task_reset_before_barrier_and_origin_after():
    """Ordering is the fix: streams open first, then t=0 is declared.

    Pre-barrier ordering published the timeout origin and the instruction before
    any stream had converged, which is how the pedestrians' 4-18 s walk ended up
    inside sim_top_down's 20 s warm-up.
    """
    events = []
    node = _run_reset_with_late_stream_readiness(events)
    labels = [label for label, _payload in events]

    assert 'task_reset' in labels
    assert 'video_streams_ready' in labels
    for released_only_after in ('mark_episode_started', 'episode_start', 'vln_instruction'):
        assert labels.index(released_only_after) > labels.index('video_streams_ready'), (
            f'{released_only_after} must happen after the recorder reports stream readiness'
        )
    assert labels.index('task_reset') < labels.index('video_streams_ready'), (
        'task_reset must still open the recorder writers before the barrier waits on them'
    )
    assert labels.index('mark_episode_started') > labels.index('task_reset'), (
        'the timeout origin must no longer precede task_reset'
    )
    assert node.episode_motion_released is True
    assert node._pedestrian_clock.released is True


def test_pedestrians_stay_held_for_every_barrier_round():
    """Not one probe round may see motion released before the barrier passes."""
    events = []
    _run_reset_with_late_stream_readiness(events, delay_rounds=5)
    probes = [payload for label, payload in events if label == 'pedestrian_released_probe']
    assert probes, 'the probe must actually have run'
    assert all(released is False for released in probes), (
        f'pedestrian motion was released before the barrier passed: {probes}'
    )


def test_reset_raises_when_stream_readiness_never_arrives():
    """A barrier that cannot pass must abort loudly, never silently proceed."""
    from task_generator.node import TaskGenerator

    events = []
    node = _task_generator_stub(events)
    node.rosparam = _RosParamAccessor(
        {'episode_start_barrier_timeout_sec': 0.3, 'episode_start_recorder_discovery_sec': 0.0}
    )

    with pytest.raises(EpisodeStartBarrierTimeout) as excinfo:
        asyncio.run(TaskGenerator._reset_task_unlocked(node))

    labels = [label for label, _payload in events]
    assert 'task_reset' in labels
    assert 'mark_episode_started' not in labels, 'no timeout origin may be declared on failure'
    assert 'episode_start' not in labels, 'no episode origin may be published on failure'
    assert 'vln_instruction' not in labels, 'the model must not be started on failure'
    assert node.episode_motion_released is False
    assert node._pedestrian_clock.released is False
    assert excinfo.value.report.unsatisfied == ['video_streams_ready']


def test_a_recorder_discovered_late_still_makes_stream_readiness_required():
    """DDS discovery must not be able to silently drop the condition.

    Deciding "no recorder attached" while discovery is still in flight would make
    the barrier skip the one condition that fixes the defect.
    """
    from task_generator.node import TaskGenerator

    events = []
    node = _task_generator_stub(events, streams_ready_publishers=0)
    node.rosparam = _RosParamAccessor(
        {'episode_start_barrier_timeout_sec': 5.0, 'episode_start_recorder_discovery_sec': 5.0}
    )
    polls = {'count': 0}

    def _late_publisher(_topic):
        polls['count'] += 1
        return 1 if polls['count'] >= 3 else 0

    node.count_publishers = _late_publisher

    async def _drive():
        async def _readiness():
            while node._video_streams_ready_episode is None:
                await asyncio.sleep(0)
                if polls['count'] >= 3:
                    node._video_streams_ready_episode = node._number_of_resets
                    events.append(('video_streams_ready', node._number_of_resets))

        readiness = asyncio.create_task(_readiness())
        await TaskGenerator._reset_task_unlocked(node)
        await readiness

    asyncio.run(_drive())

    assert polls['count'] >= 3, 'the discovery grace period must actually re-poll'
    assert 'video_streams_ready' in node._last_barrier_report.required, (
        'a recorder that appears during the grace period must still be required'
    )
    assert 'video_streams_ready' not in node._last_barrier_report.skipped
    assert 'pedestrians_spawned' in node._last_barrier_report.required


def test_barrier_skips_stream_readiness_when_no_recorder_is_attached():
    """A run without a video recorder must not be blocked by the recorder condition."""
    from task_generator.node import TaskGenerator

    events = []
    node = _task_generator_stub(events, streams_ready_publishers=0)
    asyncio.run(TaskGenerator._reset_task_unlocked(node))

    labels = [label for label, _payload in events]
    assert 'episode_start' in labels
    assert node._last_barrier_report.skipped['video_streams_ready'].startswith('no_publisher_on=')
    assert node.episode_motion_released is True


# --------------------------------------------------------------------------- #
# Source-level ordering guards.
#
# These run against whichever node.py/hunav.py ARENA_TEST_TASK_GENERATOR_SRC
# points at, so the pre-barrier sources can be checked with the same assertions
# (`git show <sha>:task_generator/task_generator/node.py`).  They express
# behaviour, not names: "the timeout origin is declared after task_reset" is a
# control-flow fact, and it is false in the pre-barrier code.
# --------------------------------------------------------------------------- #

_SRC_ROOT = Path(
    os.environ.get('ARENA_TEST_TASK_GENERATOR_SRC', '')
    or Path(__file__).parents[1] / 'task_generator'
)


def _function_def(path, class_name, function_name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    scope = tree
    if class_name:
        scope = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    return next(
        node
        for node in scope.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )


def _attribute_lines(function_node, attr):
    """Every line in ``function_node`` that mentions ``attr``.

    Matches attribute access, bare names and string literals, because the
    pre-barrier code reached the origin hook through
    ``getattr(self._task, 'mark_episode_started', None)`` -- an attribute-only
    scan would miss it and the ordering assertion would fail for the wrong reason.
    """
    lines = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.Attribute) and node.attr == attr:
            lines.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == attr:
            lines.append(node.lineno)
        elif isinstance(node, ast.Constant) and node.value == attr:
            lines.append(node.lineno)
    return sorted(lines)


def test_the_timeout_origin_is_declared_after_task_reset_is_published():
    """Pre-barrier this is false: mark_episode_started ran before task_reset.

    ``task_reset`` has to come first because the recorders cannot open their
    writers or start their warm-up gates without it, and the barrier waits on
    the result.  Declaring the timeout origin before that point is what charged
    the recorders' entire warm-up to the episode.
    """
    reset = _function_def(_SRC_ROOT / 'node.py', 'TaskGenerator', '_reset_task_unlocked')
    task_reset_lines = _attribute_lines(reset, '_pub_task_reset')
    origin_lines = _attribute_lines(reset, 'mark_episode_started') + _attribute_lines(
        reset, '_release_episode_start'
    )
    assert task_reset_lines, 'task_reset must still be published from _reset_task_unlocked'
    assert origin_lines, 'the episode origin must be declared from _reset_task_unlocked'
    assert min(origin_lines) > max(task_reset_lines), (
        'the timeout origin must be declared AFTER task_reset opens the recorder writers; '
        f'origin at line(s) {origin_lines}, task_reset at {task_reset_lines}'
    )


def test_the_barrier_is_awaited_before_the_origin_and_the_model_start():
    reset = _function_def(_SRC_ROOT / 'node.py', 'TaskGenerator', '_reset_task_unlocked')
    barrier_lines = _attribute_lines(reset, '_await_episode_start_barrier')
    origin_lines = _attribute_lines(reset, '_release_episode_start')
    instruction_lines = _attribute_lines(reset, '_publish_vln_instruction_for_episode')
    assert barrier_lines, 'the barrier must be awaited from _reset_task_unlocked'
    assert min(origin_lines) > max(barrier_lines), 'the origin must follow the barrier'
    assert min(instruction_lines) > max(barrier_lines), (
        'the VLN instruction -- and therefore the model client -- must start after the barrier, '
        'so the model cold start is charged to startup rather than to the episode'
    )


def test_hunav_sends_the_episode_clock_not_raw_sim_time_to_compute_agents():
    """The request stamp is the gate; raw ``/clock`` there is the pre-fix defect."""
    hunav_path = _SRC_ROOT / 'simulators' / 'human' / 'hunav' / 'hunav.py'
    loop = _function_def(hunav_path, 'HunavHumanSimulator', '_publish_arena_peds_loop')
    sources = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == 'stamp'
            for target in node.targets
        )
    ]
    assert sources, 'the loop must still stamp the compute_agents request'
    stamp_calls = {
        node.func.attr
        for assignment in sources
        for node in ast.walk(assignment.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert '_pedestrian_clock_stamp' in stamp_calls, (
        'the compute_agents request stamp must come from the gated episode clock; '
        f'found {sorted(stamp_calls)}'
    )


def test_hunav_skips_the_compute_agents_call_entirely_while_held():
    """Belt and braces: the behaviour tree ticks only inside that service call."""
    hunav_path = _SRC_ROOT / 'simulators' / 'human' / 'hunav' / 'hunav.py'
    loop = _function_def(hunav_path, 'HunavHumanSimulator', '_publish_arena_peds_loop')
    guards = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and any(
            isinstance(inner, ast.Attribute) and inner.attr == '_pedestrian_motion_released'
            for inner in ast.walk(node.test)
        )
    ]
    assert len(guards) == 1, 'exactly one release guard must gate the compute_agents call'
    guard_end = max(getattr(node, 'lineno', guards[0].lineno) for node in ast.walk(guards[0]))
    call_lines = [
        node.lineno
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'call_timeout'
    ]
    assert call_lines, 'the loop must still be able to call compute_agents'
    assert min(call_lines) > guard_end, (
        'the compute_agents call must sit after the release guard so a held episode never ticks '
        'the behaviour tree'
    )


def test_model_reachability_is_required_and_reads_the_ros_graph():
    """The model condition must be satisfied by observable state, either channel.

    Two channels exist and which one is populated depends on the eval mode, so
    the barrier accepts an advertised ``get_command`` service OR a received
    InternNav status sample.  Production runs with ``--internnav-official-client``,
    where the wrapper's own service waits are skipped, so the status channel is
    the one that arms there.
    """
    from task_generator.node import TaskGenerator

    events = []
    node = _task_generator_stub(events)
    robot = SimpleNamespace(
        _is_dual_vln_robot=lambda: True,
        _dual_vln_command_service_name=lambda: '/task_generator_node/Ai2_Bot2/get_command',
        _dual_vln_status_topic='/task_generator_node/Ai2_Bot2/internnav/status',
        _dual_vln_status_wall_time=0.0,
    )
    node._robots_manager = SimpleNamespace(robots={'Ai2_Bot2': robot})
    node._video_streams_ready_episode = 0

    conditions = node._episode_start_barrier_conditions()
    model = next(c for c in conditions if c.name == 'model_reachable')
    assert model.required is True
    assert model.evaluate() is False, 'no channel has spoken yet'

    # Channel A: the service becomes visible on the graph.
    node.get_service_names_and_types = lambda: [
        ('/task_generator_node/Ai2_Bot2/get_command', ['x/srv/GetCommand'])
    ]
    assert model.evaluate() is True

    # Channel B alone is enough too, which is the official-client path.
    node.get_service_names_and_types = lambda: []
    assert model.evaluate() is False
    robot._dual_vln_status_wall_time = 123.4
    assert model.evaluate() is True


def test_model_reachability_can_be_declared_not_required_explicitly():
    """The escape hatch is a named rosparam, not a silent skip."""
    events = []
    node = _task_generator_stub(events)
    node._robots_manager = SimpleNamespace(
        robots={
            'Ai2_Bot2': SimpleNamespace(
                _is_dual_vln_robot=lambda: True,
                _dual_vln_command_service_name=lambda: '/get_command',
                _dual_vln_status_topic='/status',
                _dual_vln_status_wall_time=0.0,
            )
        }
    )
    node.rosparam = _RosParamAccessor(
        {
            'episode_start_barrier_timeout_sec': 5.0,
            'episode_start_recorder_discovery_sec': 0.0,
            'episode_start_require_model_ready': False,
        }
    )
    model = next(
        c for c in node._episode_start_barrier_conditions() if c.name == 'model_reachable'
    )
    assert model.required is False
    assert model.skip_reason == 'episode_start_require_model_ready=false'


# --------------------------------------------------------------------------- #
# Post-validation fixes (applied after all four cases ran on identical code)
# --------------------------------------------------------------------------- #


def test_arena_pedestrian_spawn_z_matches_hunav_and_the_ground_plane():
    """The spawn target's z must equal what HuNav publishes for the same field.

    ``arena_isaac/services/NavigatePedestrians.py:85`` builds a THREE-dimensional
    residual from this pose against ``person.state.position`` (z ~ 0) and compares
    it against the 0.25 m arrival dead band.  A spawn z of 1.25 made the residual
    permanently 1.25 m while the horizontal residual was ~1 mm, so a walk was
    commanded with zero required horizontal displacement and the animation health
    monitor logged a spurious ``NOT ADVANCING`` error in 4/4 validation runs.
    Every value HuNav subsequently publishes for this field is 0.0.
    """
    source = (
        Path(__file__).parents[1] / 'task_generator' / 'simulators' / 'human' / 'hunav' / 'hunav.py'
    ).read_text(encoding='utf-8')
    creator = _function_def(
        _SRC_ROOT / 'simulators' / 'human' / 'hunav' / 'hunav.py',
        'HunavHumanSimulator',
        '_create_arena_pedestrian',
    )
    def _chain(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        return '.'.join(reversed(parts))

    # Match only pose.position.z; twist.angular.z is a different field.
    zs = [
        node.value.value
        for node in ast.walk(creator)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(_chain(t).endswith('pose.position.z') for t in node.targets)
    ]
    assert zs == [0.0], (
        'the arena pedestrian spawn pose z must be 0.0 (ground plane), matching HuNav; '
        f'found {zs}. A non-zero value re-introduces a permanent 3-D residual above the '
        'dead band and a spurious NOT ADVANCING error.'
    )
    assert 'NavigatePedestrians.py:85' in source, (
        'keep the pointer to the 3-D residual that makes this value load-bearing'
    )


def test_barrier_arming_lines_are_logged_at_warn_not_info():
    """Production evals run ``--log-level warn``; Stage-0 evidence must survive that.

    In all four validation runs the "video recorder detected" and "required=[...]"
    lines were suppressed and had to be reconstructed from artifacts.
    """
    node_source = (_SRC_ROOT / 'node.py').read_text(encoding='utf-8')
    for marker in (
        'Video recorder detected on ',
        'Waiting for the episode-start barrier before releasing pedestrian motion',
        'Episode-start barrier passed after ',
    ):
        idx = node_source.index(marker)
        preceding = node_source[max(0, idx - 400):idx]
        call = preceding.rsplit('self.get_logger().', 1)[-1]
        assert call.startswith('warn('), (
            f'{marker!r} must be logged at warn so it is observable under '
            f'--log-level warn; found get_logger().{call[:12]}'
        )
