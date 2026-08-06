"""Regression tests for D7: the episode-start barrier starved the model client.

What broke, and what these tests assert
---------------------------------------
``<task_generator>/eval_ready`` is a ``depth=1, RELIABLE, TRANSIENT_LOCAL`` topic.
Two external consumers subscribe to it, both at ``depth=1``, and both discard
every sample whose ``stage`` is not ``'episode'``:

* ``deps/InternNav/scripts/realworld/http_internvla_client.py:882`` -- the official
  model client.  While its gate is unsatisfied it never plans and never publishes
  a velocity command.
* ``arena_bringup/arena_bringup/internnav_timing_manager.py:280`` -- publishes
  20 Hz zero velocity while its gate is unsatisfied.

The barrier began publishing ``stage='episode_start'`` immediately after the
``stage='episode', ready=True`` sample those consumers wait for.  Measured in the
production middleware with production QoS on both ends, 20 trials per case
(``tmp/lane_w2_eval_ready_fix/qos_probe_replicate.py``), *only the final sample on
such a topic is reliably obtainable*: with a later non-matching publication the
gate was satisfied 0/20, and raising the publisher's history depth to 10 or 50
changed nothing, because the binding constraint is the subscriber's own
``KEEP_LAST(1)`` cache.  Result in four evaluation runs: zero control ticks, zero
planning requests, robot travel 0.149-0.479 m against a baseline minimum of
2.130 m.

Why the pre-existing barrier suite did not catch it
--------------------------------------------------
``test_episode_start_barrier.py:474`` wires ``_pub_eval_ready`` to a recording
publisher whose log is a throwaway ``[]``.  Every eval_ready publication was
discarded by the harness, so no assertion could see the ordering.  The tests here
record that topic and model the consumers, so the assertion is about a subscriber
being able to obtain the sample it filters for.

Failure mode discipline
-----------------------
Every test drives the real ``TaskGenerator._reset_task_unlocked`` and asserts on
behaviour.  ``_publish_eval_ready`` swallows exceptions into a logger warning, so
a broken harness would look like "nothing was published" rather than an error;
:func:`_assert_publishing_was_healthy` therefore fails the test if any publish
warning was logged or if the contract topic saw nothing at all.
"""

import asyncio
import ast
import importlib.util
import os
import re
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

from task_generator.episode_barrier import PedestrianEpisodeClock  # noqa: E402
from task_generator.latched_stage_topic import (  # noqa: E402
    EVAL_READY_CONSUMER_STAGE,
    EVAL_READY_STATUS_TOPIC,
    EVAL_READY_TOPIC,
    LatchedStageTopicContract,
    obtainable_sample,
    starved_consumer_filters,
)

_SRC_ROOT = Path(
    os.environ.get('ARENA_TEST_TASK_GENERATOR_SRC', '')
    or Path(__file__).parents[1] / 'task_generator'
)

#: Where the two consumer filters live, so a reader can check them by hand.
_CLIENT_SOURCE_CANDIDATES = (
    Path('/opt/arena_ws/deps/InternNav/scripts/realworld/http_internvla_client.py'),
    Path(__file__).parents[4] / 'deps/InternNav/scripts/realworld/http_internvla_client.py',
)
_TIMING_MANAGER_CANDIDATES = (
    Path(__file__).parents[2] / 'arena_bringup/arena_bringup/internnav_timing_manager.py',
    Path('/opt/arena_ws/src/Arena/arena_bringup/arena_bringup/internnav_timing_manager.py'),
)


# --------------------------------------------------------------------------- #
# A faithful model of the two external subscribers
# --------------------------------------------------------------------------- #


class StageFilteringSubscriber:
    """The two external consumers' gate logic, at ``depth=1``.

    Reproduces ``http_internvla_client.py:877-896`` and
    ``internnav_timing_manager.py:275-287``: discard anything whose stage is not
    ``'episode'``, then set the gate from ``ready``.

    ``deliver_latched`` is the conservative delivery model calibrated by the live
    QoS measurement: a depth-1 subscriber is only *guaranteed* the final sample on
    the topic.  Using the guaranteed set rather than the optimistic one is the
    whole point -- the four broken runs are what optimism costs.
    """

    def __init__(self, accepted_stage: str = EVAL_READY_CONSUMER_STAGE):
        self.accepted_stage = accepted_stage
        self.episode_started = False
        self.seen_stages: list[str] = []

    def _callback(self, payload) -> None:
        self.seen_stages.append(str(payload.get('stage')))
        if payload.get('stage') != self.accepted_stage:
            return
        self.episode_started = bool(payload.get('ready'))

    def deliver_latched(self, publications) -> None:
        """Deliver only what a depth-1 subscriber is guaranteed to obtain."""
        latched = obtainable_sample(publications)
        if latched is not None:
            self._callback(latched)

    def deliver_every(self, publications) -> None:
        """Deliver every sample -- the optimistic case, for contrast only."""
        for payload in publications:
            self._callback(payload)


# --------------------------------------------------------------------------- #
# Harness: drive the real reset and record what lands on each topic
# --------------------------------------------------------------------------- #


class _JsonRecordingPublisher:
    def __init__(self, sink):
        self._sink = sink

    def publish(self, msg):
        import json

        self._sink.append(json.loads(getattr(msg, 'data', '{}')))


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


def _logger_stub(messages):
    def _record(level):
        def _log(message, *args):
            messages.append((level, str(message) % args if args else str(message)))

        return _log

    return SimpleNamespace(
        info=_record('info'),
        warn=_record('warn'),
        warning=_record('warn'),
        error=_record('error'),
        debug=_record('debug'),
    )


class _Harness:
    """One driven ``_reset_task_unlocked`` and everything it published."""

    def __init__(self):
        self.contract: list[dict] = []
        self.status: list[dict] = []
        self.events: list[tuple] = []
        self.log_messages: list[tuple] = []
        self.node = None

    @property
    def publish_warnings(self):
        return [
            message
            for level, message in self.log_messages
            if 'Failed to publish' in message
        ]


def _build_node(harness, *, streams_ready_publishers=1):
    """A TaskGenerator wired just far enough to run ``_reset_task_unlocked``.

    Both eval_ready publishers and the stage contract are installed
    unconditionally.  The pre-fix ``_publish_eval_ready`` simply never reads the
    two new attributes, so the same harness drives both revisions and the tests
    fail on behaviour rather than on ``AttributeError``.
    """
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

    from task_generator.constants import Constants

    node.conf = SimpleNamespace(
        Arena=SimpleNamespace(HUMAN=SimpleNamespace(value=Constants.HumanSimulator.HUNAV))
    )
    node._pub_task_reset = _RecordingPublisher(harness.events, 'task_reset')
    node._pub_episode_start = _RecordingPublisher(harness.events, 'episode_start')
    node._pub_vln_instruction = _RecordingPublisher(harness.events, 'vln_instruction')
    node._pub_eval_ready = _JsonRecordingPublisher(harness.contract)
    node._pub_eval_ready_status = _JsonRecordingPublisher(harness.status)
    node._eval_ready_contract = LatchedStageTopicContract.eval_ready()
    node.get_logger = lambda: _logger_stub(harness.log_messages)
    node.count_publishers = lambda _topic: streams_ready_publishers
    node.get_service_names_and_types = lambda: []

    async def _noop():
        return None

    node._simulator = SimpleNamespace(
        before_reset_task=lambda: _noop(),
        after_reset_task=lambda: _noop(),
    )

    async def _task_reset(**_kwargs):
        harness.events.append(('task.reset', None))
        node._human_states_ready.set()
        node._last_human_states_count = 2

    async def _wait_navigation_ready(timeout_s):
        del timeout_s
        harness.events.append(('wait_navigation_ready', None))

    node._task = SimpleNamespace(
        reset=_task_reset,
        mark_episode_started=lambda: harness.events.append(('mark_episode_started', None)),
        wait_navigation_ready=_wait_navigation_ready,
    )
    harness.node = node
    return node


def _run_reset(*, extra_publication=None, streams_ready_delay_rounds=2):
    """Drive one real reset; the recorder reports stream readiness a bit late.

    Args:
        extra_publication: Optional ``(stage, ready)`` published through the real
            ``_publish_eval_ready`` right after the episode origin is declared.
            Used to mutate the publication sequence and check the property still
            holds for a publication nobody has written yet.
        streams_ready_delay_rounds: How many scheduler rounds the barrier has to
            wait, so the barrier genuinely blocks rather than passing trivially.
    """
    from task_generator.node import TaskGenerator

    harness = _Harness()
    node = _build_node(harness)

    if extra_publication is not None:
        original_release = TaskGenerator._release_episode_start

        def _release_then_publish(self, report):
            original_release(self, report)
            self._publish_eval_ready(*extra_publication, reason='a_stage_added_next_year')

        # Bind the wrapper onto this instance only, so the real
        # ``_release_episode_start`` still runs and the mutation is purely additive.
        node._release_episode_start = types.MethodType(_release_then_publish, node)

    async def _drive():
        async def _late_readiness():
            for _ in range(streams_ready_delay_rounds):
                await asyncio.sleep(0)
            node._video_streams_ready_episode = node._number_of_resets
            harness.events.append(('video_streams_ready', node._number_of_resets))

        readiness = asyncio.create_task(_late_readiness())
        await TaskGenerator._reset_task_unlocked(node)
        await readiness

    asyncio.run(_drive())
    return harness


def _assert_publishing_was_healthy(harness):
    """Fail loudly if the harness -- not the code -- is the reason a log is empty.

    ``_publish_eval_ready`` catches every exception and turns it into a warning,
    so a stub that is missing something looks exactly like "the code published
    nothing".  This project has shipped a positive control that passed while
    nothing ran; refuse to interpret an empty log.
    """
    assert not harness.publish_warnings, (
        'the harness broke publishing, so no conclusion about ordering is available: '
        f'{harness.publish_warnings}'
    )
    assert harness.contract, (
        f'nothing at all was published on {EVAL_READY_TOPIC}; the reset did not run '
        'far enough for these assertions to mean anything'
    )
    assert any(event[0] == 'task_reset' for event in harness.events), (
        'task_reset was never published, so the reset did not reach the barrier'
    )
    assert any(event[0] == 'mark_episode_started' for event in harness.events), (
        'the episode origin was never declared, so the barrier never passed'
    )


# --------------------------------------------------------------------------- #
# The regression test
# --------------------------------------------------------------------------- #


def test_the_model_client_can_obtain_the_sample_it_filters_for():
    """THE regression test for D7.  Fails on the pre-fix source, functionally.

    Pre-fix, the last sample on ``eval_ready`` is
    ``stage='episode_start', ready=True``, which both consumers discard, so the
    modelled client's ``episode_started`` stays ``False`` -- exactly the observed
    ``missing=['eval_ready_episode']`` from the first line to the last in 4/4
    runs.  Post-fix the last sample is ``stage='episode', ready=True``.
    """
    harness = _run_reset()
    _assert_publishing_was_healthy(harness)

    client = StageFilteringSubscriber()
    client.deliver_latched(harness.contract)

    latched = obtainable_sample(harness.contract)
    assert client.episode_started, (
        'a depth-1 stage-filtering subscriber cannot obtain a usable sample: the '
        f'latched sample on {EVAL_READY_TOPIC} is stage='
        f'{latched.get("stage")!r} ready={latched.get("ready")!r}, and the official '
        f'InternNav client discards every stage that is not {EVAL_READY_CONSUMER_STAGE!r} '
        '(http_internvla_client.py:882), so it never plans and never commands the robot. '
        f'Full contract-topic sequence: {[p.get("stage") for p in harness.contract]}'
    )


def test_the_timing_manager_can_obtain_the_sample_it_filters_for():
    """The second, independent consumer: it floods 20 Hz zeros while starved.

    Asserted separately from the client because the two failures have different
    consequences -- a starved client issues no commands at all, a starved timing
    manager actively publishes zero velocity onto the robot's own ``cmd_vel``
    (``internnav_timing_manager.py:304-306``).
    """
    harness = _run_reset()
    _assert_publishing_was_healthy(harness)

    timing_manager = StageFilteringSubscriber()
    timing_manager.deliver_latched(harness.contract)

    assert timing_manager.episode_started, (
        'the timing manager cannot obtain a usable sample and would publish 20 Hz zero '
        f'velocity for the whole episode; latched stage='
        f'{obtainable_sample(harness.contract).get("stage")!r}'
    )


def test_no_registered_consumer_can_be_starved_by_the_reset_sequence():
    """State the property in the contract's own terms, over the real sequence."""
    harness = _run_reset()
    _assert_publishing_was_healthy(harness)

    starved = starved_consumer_filters(harness.contract, LatchedStageTopicContract.eval_ready())
    assert starved == [], (
        f'{len(starved)} registered consumer filter(s) cannot obtain any sample from the '
        f'real reset sequence {[p.get("stage") for p in harness.contract]}'
    )


def test_the_release_sample_is_the_last_one_on_the_contract_topic():
    """Not just "obtainable" but "obtainable *last*", which is what depth 1 means."""
    harness = _run_reset()
    _assert_publishing_was_healthy(harness)

    last = harness.contract[-1]
    assert last.get('stage') == EVAL_READY_CONSUMER_STAGE and last.get('ready') is True, (
        'the final sample on the contract topic must be the episode-ready one; it is '
        f'{last.get("stage")!r}/{last.get("ready")!r}'
    )


# --------------------------------------------------------------------------- #
# The general property: a future publication must not be able to starve anyone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    'stage',
    ['episode_start', 'world_geometry', 'startup', 'some_stage_invented_next_year'],
)
def test_a_publication_added_after_the_origin_cannot_starve_a_consumer(stage):
    """The defect class, pinned: adding a publication must not silently starve.

    D7 was not "someone published the wrong thing", it was "an ordering
    assumption nobody had written down".  So the assertion is not about the
    barrier's message specifically: an *arbitrary* extra publication, including a
    stage that does not exist yet, is injected immediately after the episode
    origin and the modelled consumers must still be able to obtain their sample.

    Pre-fix this fails for every parameter, because every stage lands on the one
    depth-1 topic and overwrites the latched sample.
    """
    harness = _run_reset(extra_publication=(stage, False))
    _assert_publishing_was_healthy(harness)

    client = StageFilteringSubscriber()
    client.deliver_latched(harness.contract)
    assert client.episode_started, (
        f'publishing stage={stage!r} after the episode origin starved a stage-filtering '
        f'subscriber; contract-topic sequence {[p.get("stage") for p in harness.contract]}'
    )
    assert starved_consumer_filters(
        harness.contract, LatchedStageTopicContract.eval_ready()
    ) == []


def test_an_added_publication_is_still_observable_on_the_status_topic():
    """Routing must not be a quiet way of dropping information.

    A fix that made the contract topic safe by discarding the barrier's own
    status would trade one silent failure for another, so assert the full
    lifecycle is still published somewhere.
    """
    harness = _run_reset(extra_publication=('some_stage_invented_next_year', False))
    _assert_publishing_was_healthy(harness)

    status_stages = [payload.get('stage') for payload in harness.status]
    assert 'some_stage_invented_next_year' in status_stages, (
        f'the added publication vanished; {EVAL_READY_STATUS_TOPIC} carried {status_stages}'
    )
    assert 'episode_start' in status_stages, (
        f'the barrier lifecycle vanished; {EVAL_READY_STATUS_TOPIC} carried {status_stages}'
    )
    # Nothing on the contract topic may be missing from the status topic.
    for payload in harness.contract:
        assert payload in harness.status, (
            'a contract sample was not mirrored onto the status topic: '
            f'{payload.get("stage")}/{payload.get("details")}'
        )


def test_every_sample_on_the_contract_topic_passes_every_consumer_filter():
    """The invariant in its strongest form, over the whole recorded sequence.

    If this holds then *no* ordering can starve a consumer, which is why the fix
    is routing rather than re-ordering: re-ordering would have to be re-argued
    every time a caller is added.
    """
    harness = _run_reset(extra_publication=('episode_start', True))
    _assert_publishing_was_healthy(harness)

    contract = LatchedStageTopicContract.eval_ready()
    offenders = [
        payload.get('stage')
        for payload in harness.contract
        if not contract.carries(str(payload.get('stage')))
    ]
    assert offenders == [], (
        f'{EVAL_READY_TOPIC} carried stage(s) a consumer discards: {offenders}. '
        'Any of them can become the latched sample and starve the gate.'
    )


def test_publisher_history_depth_is_not_the_lever_the_contract_relies_on():
    """Guard the reasoning, not just the code.

    The measurement (20 trials/case, production RMW and QoS) showed a depth-1
    subscriber is starved just the same with publisher depth 10 and 50, because
    the binding constraint is the subscriber's own ``KEEP_LAST(1)``.  So the
    model must keep treating only the final sample as obtainable; if someone
    "optimises" it into an optimistic model, the whole fix stops being justified.
    """
    published = [
        {'stage': 'episode', 'ready': True},
        {'stage': 'episode_start', 'ready': False},
    ]
    assert obtainable_sample(published) == published[-1]

    optimistic = StageFilteringSubscriber()
    optimistic.deliver_every(published)
    conservative = StageFilteringSubscriber()
    conservative.deliver_latched(published)
    assert optimistic.episode_started is True, 'sanity: an ideal subscriber would see it'
    assert conservative.episode_started is False, (
        'the conservative model must reproduce the measured starvation; if it does not, '
        'these tests would pass on the broken code'
    )


# --------------------------------------------------------------------------- #
# Keep the model honest against the real consumers and the real source
# --------------------------------------------------------------------------- #


def _first_existing(candidates):
    for path in candidates:
        if path.is_file():
            return path
    return None


def test_the_modelled_filter_matches_the_official_client_source():
    """The model is only evidence if it still matches the component it models."""
    path = _first_existing(_CLIENT_SOURCE_CANDIDATES)
    if path is None:
        pytest.skip('official InternNav client source not present in this tree')
    source = path.read_text(encoding='utf-8')
    assert re.search(
        r"payload\.get\(\s*'stage'\s*\)\s*!=\s*'episode'", source
    ), f'{path} no longer filters eval_ready on stage=="episode"; re-derive the contract'
    assert "missing.append('eval_ready_episode')" in source, (
        f'{path} no longer gates planning on the eval_ready episode sample'
    )


def test_the_modelled_filter_matches_the_timing_manager_source():
    path = _first_existing(_TIMING_MANAGER_CANDIDATES)
    if path is None:
        pytest.skip('timing manager source not present in this tree')
    source = path.read_text(encoding='utf-8')
    assert re.search(
        r"payload\.get\(\s*'stage'\s*\)\s*!=\s*'episode'", source
    ), f'{path} no longer filters eval_ready on stage=="episode"'


def test_the_release_sample_is_published_at_the_origin_not_at_task_reset():
    """Where the release sample is published is a t=0 property, not cosmetics.

    Publishing it at ``task_reset`` would release the model client ~100 s before
    the episode origin.  On the first episode the instruction gate hides that; on
    a second episode the instruction is still latched from the first, so the
    client could command the robot throughout the recorders' warm-up -- the exact
    thing the barrier exists to prevent.
    """
    tree = ast.parse((_SRC_ROOT / 'node.py').read_text(encoding='utf-8'))
    class_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'TaskGenerator'
    )

    def _ready_publications(function_name):
        function = next(
            node
            for node in class_def.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        found = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == '_publish_eval_ready'):
                continue
            args = [a.value if isinstance(a, ast.Constant) else None for a in node.args]
            if args[:2] == [EVAL_READY_CONSUMER_STAGE, True]:
                found.append(node.lineno)
        return found

    assert _ready_publications('_release_episode_start'), (
        "the episode-ready sample must be published from _release_episode_start, so the "
        'model client is released at t=0'
    )
    assert _ready_publications('_reset_task_unlocked') == [], (
        'the episode-ready sample must NOT be published from _reset_task_unlocked: that '
        'releases the model client before the episode origin'
    )
