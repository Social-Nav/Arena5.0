import ast
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE_PATH = Path(__file__).parents[1] / 'arena_bringup' / 'internnav_eval.py'


def _recorder_source():
    tree = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'))
    start = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_start_eval_video_recorder')
    assignment = next(
        node
        for node in start.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'recorder_code' for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def _recorder_namespace():
    tree = ast.parse(_recorder_source())
    wanted = {
        '_codec_is_h264',
        '_remaining_subprocess_timeout',
        '_probe_video_codec',
        '_probe_video_codec_before_deadline',
        '_transcode_to_h264',
        'VideoWriterWrapper',
        'EvalVideoRecorder',
    }
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id.startswith('VIDEO_') for target in node.targets)
        )
        or (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted)
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    message_type = type('Message', (), {})
    namespace = {
        'json': json,
        'math': __import__('math'),
        'os': os,
        'shutil': shutil,
        'signal': signal,
        'subprocess': subprocess,
        'time': time,
        'Path': Path,
        'np': SimpleNamespace(ndarray=object, uint8=object),
        'Node': object,
        'Int16': message_type,
        'Empty': message_type,
        'Image': message_type,
        'CameraInfo': message_type,
        'Odometry': message_type,
        'PoseStamped': message_type,
        'LaserScan': message_type,
        'BACKEND_NAME': 'cv2',
        'BACKEND_MODULE': None,
    }
    exec(compile(module, '<video-recorder-definitions>', 'exec'), namespace)
    return namespace


def _outer_finalization_namespace():
    tree = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'))
    wanted_functions = {
        '_persist_video_finalization_error',
        '_force_video_recorder_exit_after_signal',
        '_finalize_video_recorder_process',
        '_select_evaluator_returncode',
    }
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id.startswith('VIDEO_RECORDER_') for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in wanted_functions)
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {
        'json': json,
        'os': os,
        'signal': signal,
        'subprocess': subprocess,
        'time': time,
        '_read_json_if_exists': lambda _path: None,
        '_read_text_if_exists': lambda _path: None,
        '_write_text': lambda _path, _data: None,
        '_terminate_process_tree': lambda _proc, grace_period_sec: -9,
    }
    exec(compile(module, '<video-parent-finalization>', 'exec'), namespace)
    return namespace


def _ownership_scope_function(namespace):
    tree = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    start = next(
        index
        for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'video_proc' for target in node.targets)
    )
    ownership_try = next(
        index
        for index, node in enumerate(main.body[start:], start=start)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'active_exception' for target in child.targets)
            for child in node.finalbody
        )
    )
    function = ast.FunctionDef(
        name='_exercise_ownership_scope',
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[*main.body[start : ownership_try + 1], ast.Return(value=ast.Constant(value=None))],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, '<ownership-scope>', 'exec'), namespace)
    return namespace['_exercise_ownership_scope']


class _ClosableStream:
    def __init__(self, name, events, error=None):
        self.name = name
        self.events = events
        self.error = error
        self.closed = False

    def close(self):
        self.events.append(('stream_close', self.name))
        if self.error is not None:
            raise self.error
        self.closed = True


class _OwnedProcess:
    def __init__(self, name, events, *, poll_effects=None, wait_timeout=False):
        self.name = name
        self.pid = 100
        self.events = events
        self.returncode = None
        self.poll_effects = list(poll_effects or [])
        self.wait_timeout = wait_timeout
        self.stdin = _ClosableStream(f'{name}.stdin', events)
        self.stdout = _ClosableStream(f'{name}.stdout', events)
        self.stderr = _ClosableStream(f'{name}.stderr', events)

    def poll(self):
        self.events.append(('poll', self.name))
        if self.poll_effects:
            effect = self.poll_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            if effect is not None:
                self.returncode = effect
            return effect
        return self.returncode

    def send_signal(self, signum):
        self.events.append(('signal', self.name, signum))

    def kill(self):
        self.events.append(('kill', self.name))
        self.returncode = -9

    def wait(self, timeout):
        self.events.append(('wait', self.name, timeout))
        if self.wait_timeout:
            raise subprocess.TimeoutExpired(self.name, timeout)
        self.returncode = 0
        return 0


def _ownership_harness(*, stage=None, cleanup_failure=None, stream_failure=None):
    events = []
    processes = {}
    finalizer_calls = []
    manifest = {'artifacts': {}, 'result': {}}
    args = SimpleNamespace(
        save_eval_video=True,
        task_reset_topic='/task_reset',
        finished_topic='/finished',
        eval_video_fps=10.0,
        eval_video_top_down_size_px=640,
        eval_video_top_down_window_m=10.0,
        dual_vln_status_topic='/status',
        internnav_direct_cmd_vel=True,
        shutdown_grace_period_sec=20.0,
        social_eval=False,
    )

    def make_process(name, *, poll_effects=None):
        process = _OwnedProcess(name, events, poll_effects=poll_effects)
        if stream_failure:
            owner_name, stream_name = stream_failure.split('.', 1)
            if owner_name == name:
                getattr(process, stream_name).error = RuntimeError(f'cleanup:{stream_failure}')
        processes[name] = process
        return process

    def start_video(*_args, **_kwargs):
        if stage == 'pre_recorder':
            raise RuntimeError('primary:pre_recorder')
        return make_process('video')

    def start_watcher(name):
        def start(*_args, **_kwargs):
            if stage == name:
                raise RuntimeError(f'primary:{name}')
            return make_process(name)

        return start

    def popen(*_args, **_kwargs):
        if stage == 'launch_popen':
            raise RuntimeError('primary:launch_popen')
        effects = [RuntimeError('primary:main_wait'), None] if stage == 'main_wait' else [0]
        return make_process('launch', poll_effects=effects)

    def wait_for_file(*_args, **_kwargs):
        if stage == 'start_error_check':
            raise RuntimeError('primary:start_error_check')
        return stage not in {'start_error_text', 'start_error_manifest'}

    def write_text(*_args, **_kwargs):
        if stage == 'start_error_text':
            raise RuntimeError('primary:start_error_text')
        events.append(('write_text', None))

    def write_yaml(*_args, **_kwargs):
        events.append(('write_yaml', None))
        if stage == 'start_error_manifest':
            raise RuntimeError('primary:start_error_manifest')

    def terminate(process, *, grace_period_sec):
        events.append(('terminate', process.name, grace_period_sec))
        if cleanup_failure == process.name:
            raise RuntimeError(f'cleanup:{process.name}')
        process.returncode = 0
        return 0

    outer = _outer_finalization_namespace()
    outer['_read_json_if_exists'] = lambda _path: {'finalization_status': 'complete'}
    outer['_read_text_if_exists'] = lambda _path: None

    def finalize(process, **kwargs):
        finalizer_calls.append((process, kwargs))
        return outer['_finalize_video_recorder_process'](process, **kwargs)

    def monotonic():
        if stage == 'launch_deadline':
            raise RuntimeError('primary:launch_deadline')
        return 0.0

    namespace = {
        'args': args,
        'env': {},
        'output_dir': '/tmp/output',
        'map_yaml_path': '/tmp/map.yaml',
        'robot_scenario_reset_topic': '/scenario_reset',
        'robot_ego_topic': '/ego',
        'robot_depth_topic': '/depth',
        'robot_camera_info_topic': '/camera_info',
        'robot_debug_overlay_topic': '/debug',
        'robot_sim_top_down_topic': '/sim_top',
        'robot_odom_topic': '/odom',
        'robot_goal_topic': '/goal',
        'robot_scan_topic': '/scan',
        'video_index_path': '/tmp/video_index.json',
        'video_error_path': '/tmp/video_error.txt',
        'episode_outcome_topic': '/outcome',
        'episode_outcome_path': '/tmp/outcome.json',
        'dual_vln_status_path': '/tmp/status.json',
        'internnav_trace_path': '/tmp/trace.jsonl',
        'launch_cmd': ['ros2', 'launch'],
        'launch_timeout_sec': 10.0,
        'vln_task_metrics_cmd': ['vln'],
        'social_metrics_cmd': ['social'],
        'artifact_validation_cmd': ['artifact'],
        'manifest': manifest,
        'manifest_path': '/tmp/manifest.yaml',
        '_start_eval_video_recorder': start_video,
        '_wait_for_file': wait_for_file,
        '_write_text': write_text,
        '_write_yaml': write_yaml,
        '_start_finished_watcher': start_watcher('finished_watcher'),
        '_start_episode_outcome_watcher': start_watcher('outcome_watcher'),
        '_start_status_watcher': start_watcher('status_watcher'),
        '_terminate_process_tree': terminate,
        '_finalize_video_recorder_process': finalize,
        'subprocess': SimpleNamespace(Popen=popen, run=lambda *_args, **_kwargs: None),
        'time': SimpleNamespace(monotonic=monotonic, sleep=lambda _seconds: None),
        'os': os,
        'sys': __import__('sys'),
    }
    exercise = _ownership_scope_function(namespace)
    return exercise, SimpleNamespace(
        events=events,
        processes=processes,
        finalizer_calls=finalizer_calls,
        manifest=manifest,
    )


def _install_fake_ffmpeg(monkeypatch, namespace, run):
    monkeypatch.setattr(shutil, 'which', lambda name: f'/fake/{name}')
    monkeypatch.setattr(subprocess, 'run', run)


class _FakeBackendWriter:
    def __init__(self):
        self.release_count = 0

    def release(self):
        self.release_count += 1


class _RecorderWriter:
    def __init__(self, name, order, recorder=None, reentrant=False, succeeds=True):
        self.name = name
        self.order = order
        self.recorder = recorder
        self.reentrant = reentrant
        self.succeeds = succeeds
        self.codec = 'h264' if succeeds else 'mpeg4'
        self.actual_codec = self.codec
        self.transcode_error = None if succeeds else f'{name} transcode interrupted'
        self.reentrant_result = None
        self.finalization_deadline = None

    def close(self, *, finalization_deadline):
        self.finalization_deadline = finalization_deadline
        self.order.append(self.name)
        if self.reentrant:
            self.reentrant_result = self.recorder._close_episode(reason='reentrant')
        return self.succeeds


def _make_recorder(namespace, tmp_path, *, reentrant_index=None, failed_index=None):
    recorder = object.__new__(namespace['EvalVideoRecorder'])
    episode_info = {'episode': 0}
    recorder.current_episode = 0
    recorder.current_episode_info = episode_info
    recorder._episode_finalization_state = 'open'
    recorder.trajectory_world = [(0.0, 0.0)]
    recorder.index = {'finalization_status': 'recording', 'finalization_errors': [], 'episodes': [episode_info]}
    recorder.error_path = tmp_path / 'video_recording_error.txt'
    recorder.index_path = tmp_path / 'video_index.json'
    recorder._index_write_count = 0

    def write_index():
        recorder._index_write_count += 1
        recorder.index_path.write_text(json.dumps(recorder.index), encoding='utf-8')

    recorder._write_index = write_index
    recorder._record_error = lambda message: recorder.error_path.write_text(message, encoding='utf-8')
    recorder._clear_transient_error = lambda: None
    order = []
    writers = [
        _RecorderWriter(
            name,
            order,
            recorder=recorder,
            reentrant=index == reentrant_index,
            succeeds=index != failed_index,
        )
        for index, name in enumerate(('ego', 'top_down', 'debug_overlay', 'sim_top_down'))
    ]
    recorder.ego_writer, recorder.top_writer, recorder.debug_overlay_writer, recorder.sim_top_down_writer = writers
    return recorder, episode_info, writers, order


@pytest.mark.parametrize('reentrant_index', [0, 2])
def test_episode_close_serializes_four_streams_and_ignores_reentrant_close(tmp_path, reentrant_index):
    namespace = _recorder_namespace()
    recorder, episode_info, writers, order = _make_recorder(
        namespace,
        tmp_path,
        reentrant_index=reentrant_index,
    )

    assert recorder._close_episode(reason='finished') is True

    assert order == ['ego', 'top_down', 'debug_overlay', 'sim_top_down']
    assert writers[reentrant_index].reentrant_result is False
    assert recorder._index_write_count == 1
    assert episode_info['close_reason'] == 'finished'
    assert episode_info['video_finalization_status'] == 'complete'
    assert recorder.index['finalization_status'] == 'complete'
    assert recorder._episode_finalization_state == 'closed'
    assert recorder.current_episode_info is None
    assert recorder._close_episode(reason='second_close') is True
    assert recorder._index_write_count == 1
    assert order == ['ego', 'top_down', 'debug_overlay', 'sim_top_down']


def test_episode_close_records_stream_failure_after_all_four_streams(tmp_path):
    namespace = _recorder_namespace()
    recorder, episode_info, _writers, order = _make_recorder(namespace, tmp_path, failed_index=2)

    assert recorder._close_episode(reason='finished') is False

    assert order == ['ego', 'top_down', 'debug_overlay', 'sim_top_down']
    assert recorder._index_write_count == 1
    assert episode_info['close_reason'] == 'finished'
    assert episode_info['debug_overlay_video_finalization_status'] == 'failed'
    assert 'debug_overlay transcode interrupted' in episode_info['debug_overlay_video_finalization_error']
    assert recorder.index['finalization_status'] == 'failed'
    assert 'debug_overlay transcode interrupted' in recorder.error_path.read_text(encoding='utf-8')
    assert recorder._close_episode(reason='second_close') is False
    assert recorder._index_write_count == 1


@pytest.mark.parametrize(
    ('expired_index', 'expired_stream'),
    list(enumerate(('ego', 'top_down', 'debug_overlay', 'sim_top_down'))),
)
def test_shared_deadline_exhaustion_is_truthful_at_each_stream(
    tmp_path,
    monkeypatch,
    expired_index,
    expired_stream,
):
    namespace = _recorder_namespace()
    recorder, episode_info, _writers, _order = _make_recorder(namespace, tmp_path)
    stream_names = ('ego', 'top_down', 'debug_overlay', 'sim_top_down')
    clock_values = iter([100.0] + [100.0] * expired_index + [431.0] * (4 - expired_index))
    namespace['time'] = SimpleNamespace(
        monotonic=lambda: next(clock_values),
        time=lambda: 1000.0,
        time_ns=lambda: 1,
    )
    probe_calls = []

    def run(command, **kwargs):
        probe_calls.append((command, kwargs['timeout']))
        return subprocess.CompletedProcess(command, 0, 'h264\n', '')

    _install_fake_ffmpeg(monkeypatch, namespace, run)
    wrappers = []
    for stream_name in stream_names:
        path = tmp_path / f'{stream_name}.mp4'
        path.write_bytes(b'h264')
        wrapper = namespace['VideoWriterWrapper'](path, 10.0)
        wrapper._writer = _FakeBackendWriter()
        wrappers.append(wrapper)
    recorder.ego_writer, recorder.top_writer, recorder.debug_overlay_writer, recorder.sim_top_down_writer = wrappers

    assert recorder._close_episode(reason='finished') is False

    assert len(probe_calls) == expired_index
    assert episode_info['video_finalization_deadline_sec'] == 330.0
    for index, stream_name in enumerate(stream_names):
        expected = 'complete' if index < expired_index else 'failed'
        assert episode_info[f'{stream_name}_video_finalization_status'] == expected
    assert 'deadline exhausted' in episode_info[f'{expired_stream}_video_finalization_error']
    assert recorder.index['finalization_status'] == 'failed'
    assert recorder._index_write_count == 1


def test_writer_close_is_idempotent_and_requires_positive_h264_probe(tmp_path):
    namespace = _recorder_namespace()
    path = tmp_path / 'direct.mp4'
    path.write_bytes(b'h264')
    backend_writer = _FakeBackendWriter()
    namespace['_probe_video_codec_before_deadline'] = lambda _path, **_kwargs: ('h264', None)
    namespace['_transcode_to_h264'] = lambda _path, **_kwargs: pytest.fail('direct H264 must not transcode')
    wrapper = namespace['VideoWriterWrapper'](path, 10.0)
    wrapper._writer = backend_writer
    deadline = time.monotonic() + 1.0

    assert wrapper.close(finalization_deadline=deadline) is True
    assert wrapper.close(finalization_deadline=deadline) is True
    assert backend_writer.release_count == 1
    assert wrapper.actual_codec == 'h264'
    assert wrapper.finalization_status == 'complete'


def test_hung_initial_ffprobe_is_bounded_and_reported(tmp_path, monkeypatch):
    namespace = _recorder_namespace()
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'original-mpeg4')
    backend_writer = _FakeBackendWriter()

    def run(command, **kwargs):
        assert command[0] == '/fake/ffprobe'
        assert 0.0 < kwargs['timeout'] <= namespace['VIDEO_FFPROBE_TIMEOUT_CAP_SEC']
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])

    _install_fake_ffmpeg(monkeypatch, namespace, run)
    wrapper = namespace['VideoWriterWrapper'](path, 10.0)
    wrapper._writer = backend_writer

    assert wrapper.close(finalization_deadline=time.monotonic() + 5.0) is False
    assert 'ffprobe timed out' in wrapper.transcode_error
    assert wrapper.finalization_status == 'failed'
    assert backend_writer.release_count == 1
    assert path.read_bytes() == b'original-mpeg4'


def test_transcode_probes_temp_before_replace_and_final_after_replace(tmp_path, monkeypatch):
    namespace = _recorder_namespace()
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'original-mpeg4')
    probe_order = []

    def run(command, **kwargs):
        assert kwargs['check'] is False
        assert 0.0 < kwargs['timeout'] <= namespace['VIDEO_FFMPEG_TIMEOUT_CAP_SEC']
        Path(command[-1]).write_bytes(b'transcoded-h264')
        return subprocess.CompletedProcess(command, 0, '', '')

    def probe(candidate, **_kwargs):
        probe_order.append(Path(candidate))
        codec = 'h264' if Path(candidate).read_bytes() == b'transcoded-h264' else 'mpeg4'
        return codec, None

    _install_fake_ffmpeg(monkeypatch, namespace, run)
    namespace['_probe_video_codec_before_deadline'] = probe

    ok, codec = namespace['_transcode_to_h264'](
        path,
        finalization_deadline=time.monotonic() + 5.0,
    )

    assert ok is True
    assert codec == 'h264'
    assert path.read_bytes() == b'transcoded-h264'
    assert probe_order[0] != path
    assert probe_order[-1] == path
    assert list(tmp_path.iterdir()) == [path]


def test_hung_final_ffprobe_restores_original_and_cleans_owned_files(tmp_path, monkeypatch):
    namespace = _recorder_namespace()
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'original-mpeg4')
    probe_count = 0

    def run(command, **kwargs):
        nonlocal probe_count
        if command[0] == '/fake/ffmpeg':
            Path(command[-1]).write_bytes(b'transcoded-h264')
            return subprocess.CompletedProcess(command, 0, '', '')
        assert command[0] == '/fake/ffprobe'
        assert 0.0 < kwargs['timeout'] <= namespace['VIDEO_FFPROBE_TIMEOUT_CAP_SEC']
        probe_count += 1
        if probe_count == 2:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, 0, 'h264\n', '')

    _install_fake_ffmpeg(monkeypatch, namespace, run)

    ok, detail = namespace['_transcode_to_h264'](
        path,
        finalization_deadline=time.monotonic() + 5.0,
    )

    assert ok is False
    assert 'ffprobe timed out' in detail
    assert probe_count == 2
    assert path.read_bytes() == b'original-mpeg4'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('failure_mode', ['returncode', 'timeout'])
def test_transcode_failure_removes_owned_partial_and_retains_original(
    tmp_path,
    monkeypatch,
    failure_mode,
):
    namespace = _recorder_namespace()
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'original-mpeg4')

    def run(command, **kwargs):
        Path(command[-1]).write_bytes(b'partial-h264')
        if failure_mode == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, 17, '', 'ffmpeg interrupted')

    _install_fake_ffmpeg(monkeypatch, namespace, run)
    namespace['_probe_video_codec_before_deadline'] = lambda _path, **_kwargs: pytest.fail(
        'failed ffmpeg output must not be probed'
    )

    ok, detail = namespace['_transcode_to_h264'](
        path,
        finalization_deadline=time.monotonic() + 5.0,
    )

    assert ok is False
    assert 'timed out' in detail if failure_mode == 'timeout' else 'ffmpeg interrupted' in detail
    assert path.read_bytes() == b'original-mpeg4'
    assert list(tmp_path.iterdir()) == [path]


def test_non_h264_temp_never_overwrites_original(tmp_path, monkeypatch):
    namespace = _recorder_namespace()
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'original-mpeg4')

    def run(command, **_kwargs):
        Path(command[-1]).write_bytes(b'not-h264')
        return subprocess.CompletedProcess(command, 0, '', '')

    _install_fake_ffmpeg(monkeypatch, namespace, run)
    namespace['_probe_video_codec_before_deadline'] = lambda _path, **_kwargs: ('mpeg4', None)

    ok, detail = namespace['_transcode_to_h264'](
        path,
        finalization_deadline=time.monotonic() + 5.0,
    )

    assert ok is False
    assert 'expected h264' in detail
    assert path.read_bytes() == b'original-mpeg4'
    assert list(tmp_path.iterdir()) == [path]


def test_failed_post_replace_probe_restores_original(tmp_path, monkeypatch):
    namespace = _recorder_namespace()
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'original-mpeg4')

    def run(command, **_kwargs):
        Path(command[-1]).write_bytes(b'transcoded-h264')
        return subprocess.CompletedProcess(command, 0, '', '')

    def probe(candidate, **_kwargs):
        return ('h264', None) if Path(candidate) != path else ('mpeg4', None)

    _install_fake_ffmpeg(monkeypatch, namespace, run)
    namespace['_probe_video_codec_before_deadline'] = probe

    ok, detail = namespace['_transcode_to_h264'](
        path,
        finalization_deadline=time.monotonic() + 5.0,
    )

    assert ok is False
    assert 'replaced file codec' in detail
    assert path.read_bytes() == b'original-mpeg4'
    assert list(tmp_path.iterdir()) == [path]


def test_finalization_bound_arithmetic_includes_overhead():
    recorder = _recorder_namespace()
    outer = _outer_finalization_namespace()

    assert recorder['VIDEO_RECORDER_FINALIZATION_DEADLINE_SEC'] == 330.0
    assert recorder['VIDEO_FFMPEG_TIMEOUT_CAP_SEC'] == 330.0
    assert recorder['VIDEO_FFPROBE_TIMEOUT_CAP_SEC'] == 15.0
    assert outer['VIDEO_RECORDER_SUBPROCESS_DEADLINE_SEC'] == 330.0
    assert outer['VIDEO_RECORDER_FINALIZATION_OVERHEAD_SEC'] == 30.0
    assert outer['VIDEO_RECORDER_FINALIZATION_TIMEOUT_SEC'] == 390.0
    assert 330.0 + 30.0 <= 390.0
    assert outer['VIDEO_RECORDER_FINALIZATION_TIMEOUT_SEC'] >= 360.0
    recorder['time'] = SimpleNamespace(monotonic=lambda: 100.0)
    assert recorder['_remaining_subprocess_timeout'](105.0, 15.0, 'probe') == (5.0, None)
    timeout, error = recorder['_remaining_subprocess_timeout'](100.0, 15.0, 'probe')
    assert timeout is None
    assert 'deadline exhausted' in error


def test_live_recorder_gets_exactly_one_owner_sigint_then_bounded_wait():
    namespace = _outer_finalization_namespace()

    class Process:
        pid = 123
        returncode = None

        def __init__(self):
            self.signals = []
            self.wait_timeouts = []

        def poll(self):
            return self.returncode

        def send_signal(self, signum):
            self.signals.append(signum)

        def wait(self, timeout):
            self.wait_timeouts.append(timeout)
            self.returncode = 0
            return 0

    process = Process()
    namespace['_read_json_if_exists'] = lambda _path: {'finalization_status': 'complete'}
    namespace['_read_text_if_exists'] = lambda _path: None

    assert namespace['_finalize_video_recorder_process'](
        process,
        index_path='video_index.json',
        error_path='video_recording_error.txt',
    ) == 0
    assert namespace['_finalize_video_recorder_process'](
        process,
        index_path='video_index.json',
        error_path='video_recording_error.txt',
    ) == 0
    assert process.signals == [signal.SIGINT]
    assert process.wait_timeouts == [namespace['VIDEO_RECORDER_FINALIZATION_TIMEOUT_SEC']]


def test_exited_recorder_collects_returncode_without_signal_or_wait():
    namespace = _outer_finalization_namespace()
    persisted = []

    class Process:
        pid = 123
        returncode = 7

        def poll(self):
            return self.returncode

        def send_signal(self, _signum):
            pytest.fail('exited recorder must not be signaled')

        def wait(self, _timeout):
            pytest.fail('exited recorder must not be waited again')

    namespace['_read_json_if_exists'] = lambda _path: {'finalization_status': 'recording'}
    namespace['_read_text_if_exists'] = lambda _path: None
    namespace['_persist_video_finalization_error'] = lambda index, error, message: persisted.append(
        (index, error, message)
    )

    assert namespace['_finalize_video_recorder_process'](
        Process(),
        index_path='video_index.json',
        error_path='video_recording_error.txt',
    ) == 7
    assert persisted[-1][-1] == 'video recorder exited with return code 7'


def test_parent_force_occurs_only_after_owner_signal_and_wait_deadline():
    namespace = _outer_finalization_namespace()
    events = []
    persisted = []

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return None

        def send_signal(self, signum):
            events.append(('signal', signum))

        def wait(self, timeout):
            events.append(('wait', timeout))
            raise subprocess.TimeoutExpired('video-recorder', timeout)

    process = Process()

    def force(proc):
        assert proc is process
        events.append(('force', None))
        return -15

    namespace['_force_video_recorder_exit_after_signal'] = force
    namespace['_read_json_if_exists'] = lambda _path: {'finalization_status': 'recording'}
    namespace['_read_text_if_exists'] = lambda _path: None
    namespace['_persist_video_finalization_error'] = lambda index, error, message: persisted.append(
        (index, error, message)
    )

    assert namespace['_finalize_video_recorder_process'](
        process,
        index_path='video_index.json',
        error_path='video_recording_error.txt',
        timeout_sec=360.0,
    ) == -15
    assert events == [
        ('signal', signal.SIGINT),
        ('wait', 360.0),
        ('force', None),
    ]
    assert persisted[-1][-1] == 'video recorder finalization timed out after 360.0 seconds'


@pytest.mark.parametrize(
    ('lifecycle_rc', 'postprocess_rc', 'video_rc', 'expected'),
    [
        (9, 8, 1, 9),
        (0, 8, 1, 8),
        (0, 0, 1, 86),
        (0, 0, 0, 0),
        (None, None, None, 0),
    ],
)
def test_evaluator_returncode_precedence(lifecycle_rc, postprocess_rc, video_rc, expected):
    namespace = _outer_finalization_namespace()
    assert namespace['_select_evaluator_returncode'](
        lifecycle_returncode=lifecycle_rc,
        postprocess_returncode=postprocess_rc,
        video_recorder_returncode=video_rc,
    ) == expected


def _assert_video_owned_cleanup_once(state):
    assert len(state.finalizer_calls) == 1
    video = state.processes['video']
    assert [event for event in state.events if event[:2] == ('signal', 'video')] == [
        ('signal', 'video', signal.SIGINT)
    ]
    assert len([event for event in state.events if event[:2] == ('wait', 'video')]) == 1
    assert video.stdin.closed is True
    assert video.stdout.closed is True
    assert video.stderr.closed is True


def test_pre_recorder_failure_has_no_owned_cleanup():
    exercise, state = _ownership_harness(stage='pre_recorder')

    with pytest.raises(RuntimeError, match='primary:pre_recorder'):
        exercise()

    assert state.processes == {}
    assert state.finalizer_calls == []
    assert not any(event[0] in {'signal', 'wait', 'terminate', 'kill'} for event in state.events)


@pytest.mark.parametrize(
    'stage',
    [
        'start_error_check',
        'start_error_text',
        'start_error_manifest',
        'finished_watcher',
        'outcome_watcher',
        'status_watcher',
        'launch_popen',
        'launch_deadline',
        'main_wait',
    ],
)
def test_every_post_recorder_setup_failure_preserves_primary_and_cleans_once(stage):
    exercise, state = _ownership_harness(stage=stage)

    with pytest.raises(RuntimeError, match=f'primary:{stage}'):
        exercise()

    _assert_video_owned_cleanup_once(state)
    for name, process in state.processes.items():
        if name == 'video':
            continue
        assert process.returncode is not None
        assert process.stdin.closed is True
        assert process.stdout.closed is True
        assert process.stderr.closed is True


@pytest.mark.parametrize('owner_name', ['launch', 'finished_watcher', 'outcome_watcher', 'status_watcher'])
def test_each_non_recorder_cleanup_failure_is_recorded_without_masking_primary(owner_name):
    exercise, state = _ownership_harness(stage='main_wait', cleanup_failure=owner_name)

    with pytest.raises(RuntimeError, match='primary:main_wait'):
        exercise()

    _assert_video_owned_cleanup_once(state)
    assert ('kill', owner_name) in state.events
    assert state.processes[owner_name].returncode is not None
    errors = state.manifest['artifacts']['process_cleanup_errors']
    assert any(f'{owner_name} process cleanup failed' in error for error in errors)


def test_stream_close_failure_is_recorded_and_other_streams_still_close():
    exercise, state = _ownership_harness(
        stage='status_watcher',
        stream_failure='finished_watcher.stdout',
    )

    with pytest.raises(RuntimeError, match='primary:status_watcher'):
        exercise()

    _assert_video_owned_cleanup_once(state)
    finished = state.processes['finished_watcher']
    assert finished.stdin.closed is True
    assert finished.stdout.closed is False
    assert finished.stderr.closed is True
    errors = state.manifest['artifacts']['process_cleanup_errors']
    assert any('finished_watcher stdout close failed' in error for error in errors)


def test_cleanup_only_failure_remains_nonzero_after_all_owners_are_cleaned():
    exercise, state = _ownership_harness(cleanup_failure='finished_watcher')

    with pytest.raises(RuntimeError, match='cleanup:finished_watcher'):
        exercise()

    _assert_video_owned_cleanup_once(state)
    assert state.processes['finished_watcher'].returncode is not None
    assert any(
        'finished_watcher process cleanup failed' in error
        for error in state.manifest['artifacts']['process_cleanup_errors']
    )


def test_main_integrates_video_returncode_into_manifest_artifact_and_return():
    source = SOURCE_PATH.read_text(encoding='utf-8')
    assert 'video_returncode = _finalize_video_recorder_process(' in source
    assert "manifest['artifacts']['video_recorder_failed'] = video_returncode not in (None, 0)" in source
    assert "video_artifact_issues.append(f'video_recorder_returncode_{video_returncode}')" in source
    assert source.count('evaluator_returncode = _select_evaluator_returncode(') == 2
    assert "manifest['result']['evaluator_returncode'] = evaluator_returncode" in source


def test_parent_persists_finalization_failure_in_index_and_error_file(tmp_path):
    namespace = _outer_finalization_namespace()
    index_path = tmp_path / 'video_index.json'
    error_path = tmp_path / 'video_recording_error.txt'
    namespace['_read_json_if_exists'] = lambda path: (
        json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else None
    )
    namespace['_read_text_if_exists'] = lambda path: (
        Path(path).read_text(encoding='utf-8') if Path(path).exists() else None
    )
    namespace['_write_text'] = lambda path, data: Path(path).write_text(data, encoding='utf-8')

    namespace['_persist_video_finalization_error'](
        str(index_path),
        str(error_path),
        'video recorder exited with return code 1',
    )

    index = json.loads(index_path.read_text(encoding='utf-8'))
    assert index['finalization_status'] == 'failed'
    assert index['finalization_errors'] == ['video recorder exited with return code 1']
    assert index['episodes'] == []
    assert error_path.read_text(encoding='utf-8') == 'video recorder exited with return code 1\n'


def test_signal_handler_defers_close_and_failed_close_sets_recorder_exit():
    recorder_source = _recorder_source()
    recorder_tree = ast.parse(recorder_source)
    shutdown = next(node for node in recorder_tree.body if isinstance(node, ast.FunctionDef) and node.name == '_shutdown')
    shutdown_calls = [node for node in ast.walk(shutdown) if isinstance(node, ast.Call)]

    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == '_close_episode'
        for call in shutdown_calls
    )
    assert "if not node._close_episode(reason=close_reason):\n            exit_code = 1" in recorder_source
    assert 'raise SystemExit(exit_code)' in recorder_source
