"""Behavioural guard for the episode-start barrier inside the eval video recorder.

The defect this pins, measured on the four delivered runs: **the four video
streams never shared a ``t = 0``.**  ``ego_observation.mp4`` started writing at
``task_reset`` (its warm-up timers default to zero), while ``sim_top_down.mp4``
discarded the first 20 s of wall clock plus 5 frames on purpose.  With HuNav
consuming each pedestrian's whole one-way route in the first 4-18 s of the
episode, the review video's frame 0 therefore began 19-130 frames *after* the
walk had already finished, in 8/8 agent-runs.

The barrier makes ``task_reset`` the stream-open edge and a separate
``episode_start`` message the time origin: the recorder runs every warm-up gate,
reports readiness, and holds every stream's frame 0 until the origin arrives.

Power proof: point ``ARENA_TEST_INTERNNAV_EVAL_SOURCE`` at a pre-barrier copy of
``internnav_eval.py`` (``git show <sha>:...``) and the behavioural tests in this
file fail, because that source writes ego frames during the sim_top_down warm-up.
"""

import ast
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


DEFAULT_SOURCE_PATH = Path(__file__).parents[1] / 'arena_bringup' / 'internnav_eval.py'
SOURCE_PATH = Path(os.environ.get('ARENA_TEST_INTERNNAV_EVAL_SOURCE', '') or DEFAULT_SOURCE_PATH)


def _recorder_source():
    tree = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'))
    start = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == '_start_eval_video_recorder'
    )
    assignment = next(
        node
        for node in start.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'recorder_code' for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


# --------------------------------------------------------------------------- #
# Source-level guards.  These need no numpy, so the barrier cannot be deleted
# unnoticed on an interpreter without the scientific stack.
# --------------------------------------------------------------------------- #


def test_recorder_declares_the_barrier_symbols_once_at_the_expected_scope():
    tree = ast.parse(_recorder_source())
    recorder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EvalVideoRecorder'
    )
    methods = [node.name for node in recorder.body if isinstance(node, ast.FunctionDef)]
    for name in (
        '_episode_start_expected',
        '_announce_video_streams_ready',
        '_hold_frame_for_episode_start',
        '_on_episode_start',
        '_sim_top_down_gate',
    ):
        assert methods.count(name) == 1, f'{name} must be defined exactly once on EvalVideoRecorder'


def test_the_hold_runs_before_any_writer_is_touched():
    """No stream may write before the hold has had a chance to consume the frame."""
    tree = ast.parse(_recorder_source())
    recorder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EvalVideoRecorder'
    )
    write_frame = next(
        node for node in recorder.body if isinstance(node, ast.FunctionDef) and node.name == '_maybe_write_frame'
    )
    hold_line = min(
        node.lineno
        for node in ast.walk(write_frame)
        if isinstance(node, ast.Attribute) and node.attr == '_hold_frame_for_episode_start'
    )
    writer_lines = [
        node.lineno
        for node in ast.walk(write_frame)
        if isinstance(node, ast.Attribute)
        and node.attr in {'ego_writer', 'top_writer', 'debug_overlay_writer', 'sim_top_down_writer'}
    ]
    assert writer_lines, 'the write path must still reference the writers'
    assert hold_line < min(writer_lines), (
        'the episode-start hold must be evaluated before any writer is used, otherwise the '
        'streams do not share t=0'
    )


def test_sim_top_down_gate_has_exactly_one_implementation():
    """The barrier and the write path must agree on when the stream is ready."""
    source = _recorder_source()
    tree = ast.parse(source)
    assert source.count('_looks_like_corrupt_sim_top_down(') >= 1
    gate_defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == '_sim_top_down_gate'
    ]
    assert len(gate_defs) == 1
    # The warm-up comparison must live only inside that one gate.
    warmup_refs = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == 'sim_top_down_warmup_sec'
    ]
    gate_span = range(
        gate_defs[0].lineno,
        max(getattr(node, 'lineno', gate_defs[0].lineno) for node in ast.walk(gate_defs[0])) + 1,
    )
    outside = [
        line for line in warmup_refs
        if line not in gate_span
    ]
    # Remaining references are the constructor default and the index record.
    assert len(outside) <= 3, f'unexpected sim_top_down warm-up comparisons outside the gate: {outside}'


def test_video_index_declares_the_barrier_fields():
    source = _recorder_source()
    for field in (
        'episode_start_origin',
        'episode_start_topic',
        'pre_episode_start_held_frames',
        'video_streams_ready',
        'video_streams_ready_wall_time',
        'video_streams_ready_after_sec',
        'episode_start_hold_state',
    ):
        assert f"'{field}'" in source, f'{field} must be recorded in video_index.json'


def test_the_barrier_topics_come_from_the_environment_and_are_derived_once():
    """The recorder's argv contract must not change, and names must have one source."""
    outer = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'))
    derivers = [
        node for node in outer.body
        if isinstance(node, ast.FunctionDef) and node.name == '_episode_barrier_topics'
    ]
    assert len(derivers) == 1
    source = _recorder_source()
    assert "os.environ.get('ARENA_EVAL_EPISODE_START_TOPIC'" in source
    assert "os.environ.get('ARENA_EVAL_VIDEO_STREAMS_READY_TOPIC'" in source


# --------------------------------------------------------------------------- #
# Behavioural tests, driving the real _maybe_write_frame.
# --------------------------------------------------------------------------- #

np = pytest.importorskip('numpy', reason='barrier behaviour tests need real numpy')

FRAME_H, FRAME_W = 48, 64


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self.codec = 'libx264'

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(getattr(msg, 'data', None))


def _namespace():
    tree = ast.parse(_recorder_source())
    wanted_defs = {
        '_ego_chroma_noise_sigma',
        '_looks_like_unconverged_ego_render',
        '_is_static_fallback_gradient',
        '_looks_like_corrupt_sim_top_down',
        'EvalVideoRecorder',
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted_defs
    ]
    missing = wanted_defs - {node.name for node in selected}
    assert not missing, f'missing recorder definitions: {sorted(missing)}'
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {'EGO_RENDER_SETTLING_PREFIX', 'EPISODE_START_TIMEOUT_PREFIX'}
            for target in node.targets
        )
    ]
    module = ast.fix_missing_locations(ast.Module(body=constants + selected, type_ignores=[]))

    class _StubMessage:
        """Accepts keyword fields the way a generated ROS message does."""

        def __init__(self, **fields):
            self.__dict__.update(fields)

    message_type = _StubMessage
    namespace = {
        'np': np,
        'math': math,
        'json': json,
        'os': os,
        'time': time,
        'Path': Path,
        'Node': object,
        'traceback': __import__('traceback'),
        'PILImage': None,
        'ImageDraw': None,
        '_PIL_BILINEAR': None,
        'yaml': __import__('yaml'),
        'Int16': message_type,
        'Empty': message_type,
        'Image': message_type,
        'CameraInfo': message_type,
        'Odometry': message_type,
        'PoseStamped': message_type,
        'LaserScan': message_type,
        'BACKEND_NAME': 'cv2',
        'BACKEND_MODULE': None,
        'VideoWriterWrapper': object,
    }
    exec(compile(module, '<recorder-barrier-definitions>', 'exec'), namespace)
    return namespace


_CACHE = {}


def _recorder_class():
    if 'ns' not in _CACHE:
        _CACHE['ns'] = _namespace()
    return _CACHE['ns']['EvalVideoRecorder']


def _settled_frame(seed=0):
    """A smooth frame the ego content gate accepts (low chroma noise)."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 200, FRAME_W, dtype=np.float64)
    frame = np.repeat(base[None, :], FRAME_H, axis=0)
    frame = np.stack([frame, frame * 0.9, frame * 0.8], axis=2)
    frame += rng.normal(0.0, 0.05, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _build_recorder(tmp_path, *, episode_start_publishers, sim_top_down=True, warmup_sec=20.0):
    recorder = _recorder_class().__new__(_recorder_class())
    recorder.videos_dir = tmp_path / 'videos'
    recorder.videos_dir.mkdir(parents=True, exist_ok=True)
    recorder.index_path = tmp_path / 'video_index.json'
    recorder.error_path = tmp_path / 'video_error.txt'
    recorder.ego_topic = '/ego'
    recorder.depth_topic = ''
    recorder.camera_info_topic = ''
    recorder.debug_overlay_topic = ''
    recorder.sim_top_down_topic = '/top' if sim_top_down else ''
    recorder.odom_topic = '/odom'
    recorder.fps = 10.0
    recorder.frame_period = 0.0
    recorder.last_frame_time = 0.0

    recorder.reset_seen = True
    recorder.reset_generation = 1
    recorder.current_episode = 0

    recorder.latest_rgb = _settled_frame()
    recorder.latest_rgb_generation = 1
    recorder.latest_sim_top_down = _settled_frame(1)
    recorder.latest_sim_top_down_generation = 1
    recorder.latest_pose = (0.0, 0.0, 0.0)
    recorder.latest_pose_generation = 1
    recorder.latest_goal = None
    recorder.latest_scan = []
    recorder.trajectory_world = []

    recorder.ego_writer = _FakeWriter()
    recorder.top_writer = _FakeWriter()
    recorder.debug_overlay_writer = None
    recorder.sim_top_down_writer = _FakeWriter() if sim_top_down else None

    recorder.sim_top_down_skipped_frame_count = 0
    recorder.sim_top_down_corrupt_skip_count = 0
    recorder.sim_top_down_warmup_sec = warmup_sec
    recorder.sim_top_down_post_warmup_discard_frames = 2
    recorder.sim_top_down_post_warmup_discard_count = 0
    recorder.ego_skipped_frame_count = 0
    recorder.ego_noise_skip_count = 0
    recorder.ego_warmup_sec = 0.0
    recorder.ego_post_warmup_discard_frames = 0
    recorder.ego_post_warmup_discard_count = 0
    recorder.ego_noise_sigma_threshold = 1.0
    recorder.ego_settle_timeout_sec = 10.0
    recorder.ego_stream_open = False
    recorder.ego_settle_timed_out = False

    recorder.episode_start_topic = '/task_generator_node/episode_start'
    recorder.video_streams_ready_topic = '/task_generator_node/video_streams_ready'
    recorder.episode_start_wait_timeout_sec = 180.0
    recorder.episode_start_seen_episode = None
    recorder.streams_ready_episode = None
    recorder.streams_ready_wall_time = 0.0
    recorder.pre_episode_start_held_frames = 0

    recorder._streams_ready_pub = _FakePublisher()
    recorder.count_publishers = lambda _topic: episode_start_publishers
    recorder._render_top_down = lambda: _settled_frame(2)

    started_at = time.time()
    recorder.current_episode_info = {
        'episode': 0,
        'ego_frames': 0,
        'top_down_frames': 0,
        'debug_overlay_frames': 0,
        'sim_top_down_frames': 0,
        'sim_top_down_skipped_frames': 0,
        'sim_top_down_corrupt_skipped_frames': 0,
        'sim_top_down_warmup_sec': warmup_sec,
        'ego_skipped_frames': 0,
        'ego_noise_skipped_frames': 0,
        'debug_overlay_fallback': False,
        'started_at_wall_time': started_at,
        'episode_start_origin': 'pending' if episode_start_publishers else 'not_expected',
        'episode_start_topic': recorder.episode_start_topic,
        'pre_episode_start_held_frames': 0,
        'video_streams_ready': False,
    }
    recorder.index = {'episodes': [recorder.current_episode_info], 'finalization_errors': []}
    recorder._ensure_episode = lambda: True
    recorder._debug_overlay_source_diagnostics = lambda: 'disabled'
    return recorder


def _pump(recorder, frames):
    for index in range(frames):
        recorder.latest_rgb = _settled_frame(index)
        recorder.latest_sim_top_down = _settled_frame(1000 + index)
        recorder.latest_rgb_generation = recorder.reset_generation
        recorder.latest_sim_top_down_generation = recorder.reset_generation
        recorder._maybe_write_frame()


def test_streams_never_diverge_while_a_warmup_gate_is_still_running(tmp_path):
    """THE POWER PROOF. Uses no symbol introduced by the barrier.

    It asserts only the property the owner cares about -- every stream's frame
    count must stay equal while any stream is still warming up -- using two
    ``video_index.json`` fields that existed long before this change.  Run against
    the pre-barrier source it fails with ego at 6 frames and sim_top_down at 0,
    which is precisely the 20 s divergence that hid the pedestrian walk.
    """
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=20.0)
    _pump(recorder, 6)
    ego_frames = int(recorder.current_episode_info['ego_frames'])
    sim_frames = int(recorder.current_episode_info['sim_top_down_frames'])
    assert ego_frames == sim_frames, (
        'the streams diverged during a warm-up gate: '
        f'ego={ego_frames} sim_top_down={sim_frames}. Their t=0 is then '
        f'{ego_frames} frames apart, which is how the pedestrian walk fell outside the review video.'
    )
    assert ego_frames == 0, 'nothing may be recorded while a stream is still warming up'


def test_no_stream_writes_before_the_episode_origin_arrives(tmp_path):
    """The load-bearing assertion: t=0 is the barrier, for every stream.

    Against the pre-barrier source this fails immediately, because ego frames are
    written throughout the sim_top_down warm-up.
    """
    recorder = _build_recorder(tmp_path, episode_start_publishers=1)
    _pump(recorder, 12)

    assert recorder.ego_writer.frames == [], 'ego must not write before the episode origin'
    assert recorder.top_writer.frames == [], 'map overlay must not write before the episode origin'
    assert recorder.sim_top_down_writer.frames == [], 'sim_top_down must not write before the origin'
    assert recorder.pre_episode_start_held_frames == 12
    assert recorder.current_episode_info['episode_start_origin'] == 'pending'


def test_every_stream_starts_at_the_origin_frame(tmp_path):
    """After the origin all enabled streams begin on the same frame."""
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=0.0)
    _pump(recorder, 4)
    held = recorder.pre_episode_start_held_frames
    assert held == 4

    recorder._on_episode_start(SimpleNamespace(data=0))
    _pump(recorder, 5)

    assert len(recorder.ego_writer.frames) == 5
    assert len(recorder.top_writer.frames) == 5
    assert len(recorder.sim_top_down_writer.frames) == 5, (
        'sim_top_down must start on the same frame as ego, not 20 s later'
    )
    assert recorder.current_episode_info['episode_start_origin'] == 'barrier'
    assert recorder.current_episode_info['pre_episode_start_held_frames'] == held


def test_the_sim_top_down_warmup_is_spent_inside_the_hold_not_inside_the_episode(tmp_path):
    """The warm-up is preserved; it simply no longer eats episode time.

    With a live warm-up the stream reports "not ready" and the hold keeps every
    stream at frame 0, which is what lets the barrier absorb the warm-up.
    """
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=20.0)
    _pump(recorder, 6)
    assert recorder.sim_top_down_skipped_frame_count == 6, 'the warm-up gate must still run'
    assert recorder.streams_ready_episode is None, 'readiness must not be claimed during warm-up'
    assert recorder._streams_ready_pub.messages == []
    assert recorder.current_episode_info['episode_start_hold_state']['sim_top_down'] == 'warmup'


def test_readiness_is_announced_only_after_every_gate_passes(tmp_path):
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=0.0)
    # discard budget is 2, so the first two frames drain it
    _pump(recorder, 1)
    assert recorder._streams_ready_pub.messages == []
    _pump(recorder, 1)
    assert recorder._streams_ready_pub.messages == []
    _pump(recorder, 1)
    assert recorder._streams_ready_pub.messages == [0], (
        'readiness must be announced exactly once, on the first frame every stream would write'
    )
    _pump(recorder, 3)
    assert recorder._streams_ready_pub.messages == [0], 'readiness must not be re-announced'
    assert recorder.current_episode_info['video_streams_ready'] is True


def test_readiness_waits_for_a_stream_that_never_delivers_a_frame(tmp_path):
    """A stream configured but silent must block readiness, so the barrier fails loudly."""
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=0.0)
    for _ in range(5):
        recorder.latest_rgb = _settled_frame(7)
        recorder.latest_rgb_generation = recorder.reset_generation
        recorder.latest_sim_top_down = None
        recorder.latest_sim_top_down_generation = -1
        recorder._maybe_write_frame()
    assert recorder.ego_writer.frames == [], (
        'a silent stream must hold the whole episode, not let ego run ahead'
    )
    assert recorder._streams_ready_pub.messages == []
    assert recorder.current_episode_info['episode_start_hold_state']['sim_top_down'] == 'no_post_reset_frame'


def test_no_hold_when_no_task_generator_publishes_an_origin(tmp_path):
    """Backwards compatibility: without a barrier publisher, behaviour is unchanged."""
    recorder = _build_recorder(tmp_path, episode_start_publishers=0, warmup_sec=0.0)
    _pump(recorder, 4)
    assert len(recorder.ego_writer.frames) == 4, (
        'a run whose task generator does not publish an origin must record as before'
    )
    assert recorder.pre_episode_start_held_frames == 0
    assert recorder.current_episode_info['episode_start_origin'] == 'not_expected'


def test_a_missing_origin_fails_open_loudly_rather_than_losing_the_video(tmp_path):
    """Never hang, never silently proceed: record, and mark t=0 as not the barrier."""
    prefix = _CACHE.setdefault('ns', _namespace())['EPISODE_START_TIMEOUT_PREFIX']
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=0.0)
    recorder.episode_start_wait_timeout_sec = 0.05
    _pump(recorder, 3)
    assert recorder._streams_ready_pub.messages == [0]
    assert recorder.ego_writer.frames == []

    time.sleep(0.06)
    _pump(recorder, 2)

    assert len(recorder.ego_writer.frames) == 2, 'the video must not be lost'
    assert recorder.current_episode_info['episode_start_origin'] == 'timeout_fail_open'
    message = recorder.error_path.read_text(encoding='utf-8')
    assert message.startswith(prefix)
    assert 'is NOT the barrier' in message


def test_a_stale_origin_for_another_episode_does_not_release_this_one(tmp_path):
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=0.0)
    recorder._on_episode_start(SimpleNamespace(data=7))
    _pump(recorder, 3)
    assert recorder.ego_writer.frames == [], 'a latched origin from another episode must be ignored'
    assert recorder.episode_start_seen_episode is None


def test_mid_episode_corrupt_sim_top_down_frame_does_not_cost_an_ego_frame(tmp_path):
    """Post-origin behaviour must be exactly as it was before the barrier."""
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=0.0)
    recorder.sim_top_down_post_warmup_discard_frames = 0
    recorder._on_episode_start(SimpleNamespace(data=0))
    _pump(recorder, 2)
    assert len(recorder.ego_writer.frames) == 2
    assert len(recorder.sim_top_down_writer.frames) == 2

    # A pure-black frame is what _looks_like_corrupt_sim_top_down rejects.
    recorder.latest_rgb = _settled_frame(3)
    recorder.latest_rgb_generation = recorder.reset_generation
    recorder.latest_sim_top_down = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    recorder.latest_sim_top_down_generation = recorder.reset_generation
    recorder._maybe_write_frame()

    assert len(recorder.ego_writer.frames) == 3, (
        'a corrupt sim_top_down frame must still cost only the sim_top_down frame'
    )
    assert len(recorder.sim_top_down_writer.frames) == 2
    assert recorder.current_episode_info['sim_top_down_corrupt_skipped_frames'] == 1


def test_the_drawn_trajectory_also_starts_at_the_origin(tmp_path):
    """Startup poses must not appear in the episode's map overlay."""
    recorder = _build_recorder(tmp_path, episode_start_publishers=1, warmup_sec=0.0)
    recorder.trajectory_world = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0)]
    recorder._on_episode_start(SimpleNamespace(data=0))
    assert recorder.trajectory_world == [(0.3, 0.0)], (
        'the drawn trajectory must start at t=0, keeping only the current pose'
    )


def test_the_recorder_and_the_task_generator_derive_the_same_topic_names():
    """One prefix, two processes, no hand-maintained duplicate.

    The task generator publishes ``service_namespace('episode_start')`` and its
    node resolves to ``/task_generator_node`` (launched with
    ``name=basename('task_generator_node')``, ``namespace=dirname(...)`` in
    ``task_generator/launch/task_generator.launch.py:348-356``).  The recorder is
    handed the real ``task_reset`` topic, so stripping ``/task_reset`` from it must
    yield the same prefix -- otherwise the barrier is wired to a dead topic and,
    because absence of a publisher means "not required", it would silently do
    nothing at all.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location('_lane_t_eval_module', SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    topics = module._episode_barrier_topics('/task_generator_node/task_reset')
    assert topics['ARENA_EVAL_EPISODE_START_TOPIC'] == '/task_generator_node/episode_start'
    assert topics['ARENA_EVAL_VIDEO_STREAMS_READY_TOPIC'] == '/task_generator_node/video_streams_ready'

    node_source = (
        Path(__file__).parents[2] / 'task_generator' / 'task_generator' / 'node.py'
    ).read_text(encoding='utf-8')
    assert "EPISODE_START_TOPIC = 'episode_start'" in node_source
    assert "VIDEO_STREAMS_READY_TOPIC = 'video_streams_ready'" in node_source
    assert 'self.service_namespace(EPISODE_START_TOPIC)' in node_source
    assert 'self.service_namespace(VIDEO_STREAMS_READY_TOPIC)' in node_source

    # A non-default namespace must stay consistent across both halves.
    other = module._episode_barrier_topics('/bench_tg/task_reset')
    assert other['ARENA_EVAL_EPISODE_START_TOPIC'] == '/bench_tg/episode_start'
