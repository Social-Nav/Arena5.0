"""Regression guard for the ego stream render warm-up gate.

Isaac publishes a genuine ``sensor_msgs/msg/Image`` before its RTX/DLSS temporal
accumulation has converged: correct geometry and materials buried in dense chroma
speckle.  The recorder's only ego validity check,
``_is_static_fallback_gradient``, matches a single deprecated synthetic test
pattern and cannot detect noise, so those frames used to become t=0 of
``ego_observation.mp4`` -- and of ``ego_debug_overlay.mp4``, which composites the
ego frame on its fallback path.  Measured on the delivered runs: case01 frame 0
at 7.39x its own luma baseline and 31.3x chroma, case03 frame 0 at 17.4x / 11.8x,
while the pre-v1-dataset control is clean.

``sim_top_down`` already carries exactly this guard.  These tests pin the ego
equivalent: it must skip unconverged leading frames, skip nothing when the stream
is already clean, report every drop in ``video_index.json``, never touch a
mid-episode frame, and never leave the stream empty or truncated when the
renderer simply never settles.
"""

import ast
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE_PATH = Path(__file__).parents[1] / 'arena_bringup' / 'internnav_eval.py'


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


# ---------------------------------------------------------------------------
# Source-level guards.  These run without numpy so the gate cannot be removed
# unnoticed on an interpreter that lacks the scientific stack.
# ---------------------------------------------------------------------------
def test_ego_gate_symbols_exist_exactly_once_at_the_expected_scope():
    tree = ast.parse(_recorder_source())
    module_level = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert module_level.count('_ego_chroma_noise_sigma') == 1
    assert module_level.count('_looks_like_unconverged_ego_render') == 1

    recorder = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EvalVideoRecorder'
    )
    methods = [n.name for n in recorder.body if isinstance(n, ast.FunctionDef)]
    assert methods.count('_skip_unsettled_ego_frame') == 1

    # no shadowing copies hiding in a nested scope
    everywhere = sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name in {'_ego_chroma_noise_sigma', '_looks_like_unconverged_ego_render',
                       '_skip_unsettled_ego_frame'}
    )
    assert everywhere == 3


def test_ego_gate_is_consulted_before_the_ego_frame_is_written():
    """The gate must run before ego_writer.write, not after."""
    source = _recorder_source()
    tree = ast.parse(source)
    recorder = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EvalVideoRecorder'
    )
    write_frame = next(
        n for n in recorder.body if isinstance(n, ast.FunctionDef) and n.name == '_maybe_write_frame'
    )
    gate_line = min(
        n.lineno for n in ast.walk(write_frame)
        if isinstance(n, ast.Attribute) and n.attr == '_skip_unsettled_ego_frame'
    )
    ego_write_line = min(
        n.lineno for n in ast.walk(write_frame)
        if isinstance(n, ast.Attribute) and n.attr == 'ego_writer'
    )
    assert gate_line < ego_write_line, (
        'the warm-up gate must be evaluated before the ego frame reaches the writer'
    )


def test_video_index_declares_the_ego_skip_counters():
    """The skip must be auditable from the artifact, not only from the log."""
    source = _recorder_source()
    for field in (
        'ego_skipped_frames',
        'ego_noise_skipped_frames',
        'ego_warmup_sec',
        'ego_post_warmup_discard_frames',
        'ego_post_warmup_discarded_frames',
        'ego_noise_sigma_threshold',
        'ego_settle_timeout_sec',
        'ego_settle_timed_out',
    ):
        assert f"'{field}'" in source, f'{field} must be present in the episode index'


# ---------------------------------------------------------------------------
# Behavioural tests.  These drive the real _maybe_write_frame.
# ---------------------------------------------------------------------------
np = pytest.importorskip('numpy', reason='gate behaviour tests need real numpy')

SQRT_PI_2 = math.sqrt(math.pi / 2.0)
FRAME_H, FRAME_W = 120, 160


def _recorder_namespace():
    """exec the gate helpers and EvalVideoRecorder with real numpy bound."""
    tree = ast.parse(_recorder_source())
    wanted = {'_ego_chroma_noise_sigma', '_looks_like_unconverged_ego_render',
              '_is_static_fallback_gradient', 'EvalVideoRecorder'}
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    assert len(selected) == len(wanted), f'expected {len(wanted)} definitions, found {len(selected)}'
    constants = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'EGO_RENDER_SETTLING_PREFIX' for t in node.targets)
    ]
    assert len(constants) == 1, 'EGO_RENDER_SETTLING_PREFIX must be defined once at module scope'
    selected = constants + selected
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    message_type = type('Message', (), {})
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
        '_looks_like_corrupt_sim_top_down': lambda frame: False,
    }
    exec(compile(module, '<recorder-ego-gate-definitions>', 'exec'), namespace)
    return namespace


_NAMESPACE_CACHE = {}


def _gate_namespace():
    """Resolve the gate definitions lazily.

    Deliberately not evaluated at import time: if the gate is ever removed, the
    source-level guards above must still run and fail with a readable message
    rather than the whole module erroring during collection.
    """
    if 'ns' not in _NAMESPACE_CACHE:
        _NAMESPACE_CACHE['ns'] = _recorder_namespace()
    return _NAMESPACE_CACHE['ns']


@pytest.fixture(scope='module')
def gate():
    namespace = _gate_namespace()
    return SimpleNamespace(
        recorder_cls=namespace['EvalVideoRecorder'],
        chroma_noise_sigma=namespace['_ego_chroma_noise_sigma'],
        looks_unconverged=namespace['_looks_like_unconverged_ego_render'],
        settling_prefix=namespace['EGO_RENDER_SETTLING_PREFIX'],
    )


class FakeWriter:
    """Stand-in for VideoWriterWrapper that just records what it was given."""

    codec = 'h264'

    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))


def _scene(seed=0):
    """A deterministic textured 'scene': smooth gradients plus hard edges.

    Structure only, no per-pixel noise, so the opponent-plane statistic stays
    low the way a converged render does.
    """
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float64)
    base = 40.0 + 90.0 * (xx / FRAME_W) + 40.0 * (yy / FRAME_H)
    frame = np.empty((FRAME_H, FRAME_W, 3), dtype=np.float64)
    frame[..., 0] = base
    frame[..., 1] = base * 0.92 + 8.0
    frame[..., 2] = base * 0.80 + 16.0
    # a few hard-edged blocks, identical in every channel (achromatic structure)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        y0 = int(rng.integers(0, FRAME_H - 20))
        x0 = int(rng.integers(0, FRAME_W - 20))
        frame[y0:y0 + 18, x0:x0 + 18, :] += 55.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _noisy_scene(sigma_dn, seed=0):
    """The same scene with independent per-channel sampling noise added.

    Independent noise per channel is what an unconverged RTX/DLSS sample looks
    like; it inflates the R-G opponent plane by sigma*sqrt(2).
    """
    rng = np.random.default_rng(1000 + seed)
    frame = _scene(seed).astype(np.float64)
    frame += rng.normal(0.0, sigma_dn, size=frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _make_recorder(gate, tmp_path, **overrides):
    """A recorder carrying the real gate and the real _maybe_write_frame."""
    recorder = object.__new__(gate.recorder_cls)
    recorder.output_dir = tmp_path
    recorder.index_path = tmp_path / 'video_index.json'
    recorder.error_path = tmp_path / 'video_recording_error.txt'
    recorder.ego_topic = '/ego/rgb'
    recorder.fps = 10.0
    recorder.frame_period = 1.0 / 10.0
    recorder.last_frame_time = -1000.0
    recorder.top_down_size_px = 64
    recorder.top_down_window_m = 10.0
    recorder.map_image = None
    recorder.latest_pose = None          # _render_top_down short-circuits to zeros
    recorder.latest_goal = None
    recorder.latest_scan = []
    recorder.trajectory_world = []
    recorder.latest_rgb = None
    recorder.current_episode = 0
    recorder.ego_writer = FakeWriter()
    recorder.top_writer = FakeWriter()
    recorder.debug_overlay_writer = None
    recorder.sim_top_down_writer = None
    recorder.latest_sim_top_down = None
    recorder.latest_sim_top_down_generation = -1
    recorder.reset_generation = 0

    # gate configuration, mirroring the defaults in __init__
    recorder.ego_skipped_frame_count = 0
    recorder.ego_noise_skip_count = 0
    recorder.ego_warmup_sec = 0.0
    recorder.ego_post_warmup_discard_frames = 0
    recorder.ego_post_warmup_discard_count = 0
    recorder.ego_noise_sigma_threshold = 1.0
    recorder.ego_settle_timeout_sec = 10.0
    recorder.ego_stream_open = False
    recorder.ego_settle_timed_out = False

    # Episode-start barrier state.  These tests are about the ego gate, so the
    # barrier is configured absent (no publisher of the episode origin), which is
    # the documented "record exactly as before" path -- see
    # test_internnav_eval_episode_start_barrier.py for the barrier's own tests.
    recorder.episode_start_topic = ''
    recorder.video_streams_ready_topic = ''
    recorder.episode_start_wait_timeout_sec = 180.0
    recorder.episode_start_seen_episode = None
    recorder.streams_ready_episode = None
    recorder.streams_ready_wall_time = 0.0
    recorder.pre_episode_start_held_frames = 0
    recorder.count_publishers = lambda _topic: 0
    for key, value in overrides.items():
        setattr(recorder, key, value)

    recorder.current_episode_info = {
        'episode': 0,
        'ego_frames': 0,
        'top_down_frames': 0,
        'debug_overlay_frames': 0,
        'sim_top_down_frames': 0,
        'ego_skipped_frames': 0,
        'ego_noise_skipped_frames': 0,
        'ego_warmup_sec': recorder.ego_warmup_sec,
        'ego_post_warmup_discard_frames': recorder.ego_post_warmup_discard_frames,
        'ego_post_warmup_discarded_frames': 0,
        'ego_noise_sigma_threshold': recorder.ego_noise_sigma_threshold,
        'ego_settle_timeout_sec': recorder.ego_settle_timeout_sec,
        'ego_settle_timed_out': False,
        'started_at_wall_time': time.time(),
    }
    recorder.index = {'episodes': [recorder.current_episode_info], 'finalization_status': 'recording'}
    recorder._ensure_episode = lambda: True
    return recorder


def _tick(recorder, frame):
    """Deliver one camera frame, bypassing only the frame-rate throttle."""
    recorder.latest_rgb = frame
    recorder.last_frame_time = -1000.0
    recorder._maybe_write_frame()


# ---------------------------------------------------------------------------
def test_detector_matches_the_injected_noise_level(gate):
    """Sanity-check the estimator itself before relying on it."""
    clean = gate.chroma_noise_sigma(_scene())
    assert clean < 1.0, f'structure-only scene should read low, got {clean:.3f}'

    for sigma_dn in (1.0, 2.0, 4.0):
        measured = gate.chroma_noise_sigma(_noisy_scene(sigma_dn))
        expected = sigma_dn * math.sqrt(2.0)  # independent noise in R and in G
        assert abs(measured - expected) / expected < 0.15, (
            f'injected {sigma_dn} DN per channel -> expected ~{expected:.3f} on R-G, '
            f'measured {measured:.3f}'
        )


def test_detector_flags_unconverged_and_accepts_converged_frames(gate):
    assert gate.looks_unconverged(_noisy_scene(2.0), 1.0) is True
    assert gate.looks_unconverged(_scene(), 1.0) is False
    # a zero/negative threshold disables the content check entirely
    assert gate.looks_unconverged(_noisy_scene(4.0), 0.0) is False


def test_unconverged_leading_frames_are_skipped(gate, tmp_path):
    recorder = _make_recorder(gate, tmp_path)
    for i in range(3):
        _tick(recorder, _noisy_scene(2.0, seed=i))
    assert recorder.ego_writer.frames == [], 'unconverged leading frames must not be recorded'
    assert recorder.ego_skipped_frame_count == 3
    assert recorder.ego_noise_skip_count == 3
    assert recorder.current_episode_info['ego_frames'] == 0

    _tick(recorder, _scene())
    assert len(recorder.ego_writer.frames) == 1, 'the first converged frame must be recorded'
    assert recorder.current_episode_info['ego_frames'] == 1
    assert recorder.ego_skipped_frame_count == 3, 'admitting a frame must not change the count'


def test_no_frames_are_skipped_when_the_stream_is_already_clean(gate, tmp_path):
    recorder = _make_recorder(gate, tmp_path)
    for i in range(5):
        _tick(recorder, _scene(seed=i))
    assert len(recorder.ego_writer.frames) == 5
    assert recorder.ego_skipped_frame_count == 0, 'a clean stream must lose nothing'
    assert recorder.ego_noise_skip_count == 0
    assert recorder.current_episode_info['ego_skipped_frames'] == 0
    assert recorder.current_episode_info['ego_settle_timed_out'] is False


def test_skip_count_is_reported_in_video_index(gate, tmp_path):
    recorder = _make_recorder(gate, tmp_path)
    for i in range(4):
        _tick(recorder, _noisy_scene(2.0, seed=i))
    _tick(recorder, _scene())

    index = json.loads((tmp_path / 'video_index.json').read_text(encoding='utf-8'))
    episode = index['episodes'][0]
    assert episode['ego_skipped_frames'] == 4
    assert episode['ego_noise_skipped_frames'] == 4
    assert episode['ego_skip_reason'] == 'unconverged_render'
    assert episode['ego_noise_sigma_threshold'] == 1.0
    assert episode['ego_frames'] == 1


def test_gate_never_truncates_a_stream_that_never_converges(gate, tmp_path):
    """A permanently noisy renderer must still yield a video, and say so."""
    started = time.time() - 30.0  # already past the 10 s settle deadline
    recorder = _make_recorder(gate, tmp_path)
    recorder.current_episode_info['started_at_wall_time'] = started

    for i in range(3):
        _tick(recorder, _noisy_scene(4.0, seed=i))

    assert len(recorder.ego_writer.frames) == 3, (
        'past the settle deadline the gate must fail open rather than drop every frame'
    )
    assert recorder.ego_settle_timed_out is True
    index = json.loads((tmp_path / 'video_index.json').read_text(encoding='utf-8'))
    assert index['episodes'][0]['ego_settle_timed_out'] is True

    error = (tmp_path / 'video_recording_error.txt').read_text(encoding='utf-8')
    assert 'never settled' in error, f'expected an explicit error, got {error!r}'
    assert '/ego/rgb' in error


def test_gate_fails_open_only_after_the_deadline(gate, tmp_path):
    """Before the deadline the same frames are held back, so the test above is not vacuous."""
    recorder = _make_recorder(gate, tmp_path)
    recorder.current_episode_info['started_at_wall_time'] = time.time()
    for i in range(3):
        _tick(recorder, _noisy_scene(4.0, seed=i))
    assert recorder.ego_writer.frames == []
    assert recorder.ego_settle_timed_out is False
    # held back, not timed out: the transient explanation must be present but must
    # not claim the render failed to settle
    error = (tmp_path / 'video_recording_error.txt').read_text(encoding='utf-8')
    assert error.startswith(gate.settling_prefix)
    assert 'never settled' not in error


def test_gate_disarms_after_the_first_admitted_frame(gate, tmp_path):
    """No mid-episode frame may ever be dropped by the content check."""
    recorder = _make_recorder(gate, tmp_path)
    _tick(recorder, _scene())
    assert len(recorder.ego_writer.frames) == 1
    assert recorder.ego_stream_open is True

    # a noisy frame arriving later in the episode must still be recorded
    for i in range(3):
        _tick(recorder, _noisy_scene(4.0, seed=i))
    assert len(recorder.ego_writer.frames) == 4
    assert recorder.ego_skipped_frame_count == 0


def test_warmup_period_is_honoured_and_counted(gate, tmp_path):
    """The timer knob mirrors sim_top_down; it is off by default but must work."""
    recorder = _make_recorder(gate, tmp_path, ego_warmup_sec=60.0)
    recorder.current_episode_info['started_at_wall_time'] = time.time()
    for i in range(3):
        _tick(recorder, _scene(seed=i))  # clean frames, held back purely by the timer
    assert recorder.ego_writer.frames == []
    assert recorder.ego_skipped_frame_count == 3
    assert recorder.ego_noise_skip_count == 0, 'timer skips are not noise skips'
    index = json.loads((tmp_path / 'video_index.json').read_text(encoding='utf-8'))
    assert index['episodes'][0]['ego_skip_reason'] == 'warmup'
    assert index['episodes'][0]['ego_skipped_frames'] == 3


def test_post_warmup_discard_count_is_honoured_and_counted(gate, tmp_path):
    recorder = _make_recorder(gate, tmp_path, ego_post_warmup_discard_frames=2)
    for i in range(4):
        _tick(recorder, _scene(seed=i))
    assert len(recorder.ego_writer.frames) == 2, 'exactly two leading frames discarded'
    assert recorder.ego_skipped_frame_count == 2
    assert recorder.ego_post_warmup_discard_count == 2
    index = json.loads((tmp_path / 'video_index.json').read_text(encoding='utf-8'))
    assert index['episodes'][0]['ego_post_warmup_discarded_frames'] == 2


def test_skipped_tick_keeps_the_streams_in_lockstep(gate, tmp_path):
    """A skipped ego frame must not silently desynchronise the other videos."""
    recorder = _make_recorder(gate, tmp_path)
    for i in range(3):
        _tick(recorder, _noisy_scene(2.0, seed=i))
    assert recorder.ego_writer.frames == []
    assert recorder.top_writer.frames == [], (
        'the map view must not advance while the ego frame is held back, or the two '
        'videos stop sharing a frame index'
    )
    _tick(recorder, _scene())
    assert len(recorder.ego_writer.frames) == len(recorder.top_writer.frames) == 1
    assert recorder.current_episode_info['ego_frames'] == recorder.current_episode_info['top_down_frames']


def test_gate_is_rearmed_and_counters_zeroed_between_episodes(gate, tmp_path):
    recorder = _make_recorder(gate, tmp_path)
    _tick(recorder, _noisy_scene(2.0))
    _tick(recorder, _scene())
    assert recorder.ego_skipped_frame_count == 1
    assert recorder.ego_stream_open is True

    recorder._reset_episode_stream_state()
    assert recorder.ego_stream_open is False, 'the gate must be re-armed for the next episode'
    assert recorder.ego_skipped_frame_count == 0
    assert recorder.ego_noise_skip_count == 0
    assert recorder.ego_post_warmup_discard_count == 0
    assert recorder.ego_settle_timed_out is False


def test_holdback_explains_itself_and_is_cleared_once_frames_flow(gate, tmp_path):
    """An episode ending mid-hold-back must not leave an unexplained empty video.

    Every held-back frame records a transient explanation, so a run that dies
    while the renderer is still settling says why it produced no frames.  Once a
    real frame is admitted the message must be cleared again, otherwise a healthy
    run would ship a stale error file.
    """
    recorder = _make_recorder(gate, tmp_path)
    error_file = tmp_path / 'video_recording_error.txt'

    for i in range(2):
        _tick(recorder, _noisy_scene(2.0, seed=i))
    assert recorder.ego_writer.frames == []
    assert error_file.exists(), 'a hold-back must leave a stated reason behind'
    message = error_file.read_text(encoding='utf-8')
    assert message.startswith(gate.settling_prefix)
    assert '/ego/rgb' in message
    assert 'reason=unconverged_render' in message

    _tick(recorder, _scene())
    assert len(recorder.ego_writer.frames) == 1
    assert not error_file.exists(), (
        'the transient hold-back message must be cleared once frames are flowing'
    )


def test_settle_timeout_error_is_not_treated_as_transient(gate, tmp_path):
    """The fail-open finding is a real result and must survive later writes."""
    recorder = _make_recorder(gate, tmp_path)
    recorder.current_episode_info['started_at_wall_time'] = time.time() - 30.0
    _tick(recorder, _noisy_scene(4.0))
    error_file = tmp_path / 'video_recording_error.txt'
    assert 'never settled' in error_file.read_text(encoding='utf-8')

    # the frame was admitted, so _clear_transient_error ran on this very tick
    assert len(recorder.ego_writer.frames) == 1
    assert error_file.exists(), 'the settle-timeout error must not be cleared away'
    assert 'never settled' in error_file.read_text(encoding='utf-8')


def test_deprecated_fallback_gradient_still_rejected(gate, tmp_path):
    """The pre-existing ego check must keep working alongside the new gate."""
    recorder = _make_recorder(gate, tmp_path)
    yy, xx = np.mgrid[0:480, 0:640]
    pattern = np.empty((480, 640, 3), dtype=np.uint8)
    pattern[..., 0] = (xx % 256).astype(np.uint8)
    pattern[..., 1] = (yy % 256).astype(np.uint8)
    pattern[..., 2] = 96
    _tick(recorder, pattern)
    assert recorder.ego_writer.frames == []
    assert recorder.ego_skipped_frame_count == 0, (
        'the synthetic-pattern rejection is a separate mechanism and must not be '
        'counted as a render warm-up skip'
    )
    assert 'synthetic fallback gradient' in (
        tmp_path / 'video_recording_error.txt'
    ).read_text(encoding='utf-8')
