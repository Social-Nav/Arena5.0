"""CPU-only negative tests for the Prox diag v2 timebase audit.

No ROS graph, no simulator, no video encoding: the renderer helpers are loaded
directly from source and exercised with synthetic saved-data rows.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / 'arena_bringup' / 'render_prox_diag_video.py'


def _load_renderer():
    spec = importlib.util.spec_from_file_location('prox_diag_render_under_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def renderer():
    return _load_renderer()


def _write_jpg(path, color=(0, 0, 0)):
    image = np.full((48, 64, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_ratio_is_none_for_zero_or_nonfinite_denominators(renderer):
    assert renderer._ratio(0.0, 0.0) is None          # first PTS = 0
    assert renderer._ratio(1.0, 0.0) is None          # zero wall time
    assert renderer._ratio(float('inf'), 1.0) is None
    assert renderer._ratio(float('nan'), 1.0) is None
    assert renderer._ratio(None, 1.0) is None
    assert renderer._ratio(1.0, 0.2) == pytest.approx(5.0)


def test_xN_stats_never_reports_huge_or_nonfinite(renderer):
    rows = [
        {'xN': None}, {'xN': 1.0}, {'xN': 1.0},
        {'xN': 1.0}, {'xN': 1.0},
    ]
    stats = renderer._xN_stats(rows)
    assert stats['null_or_na_frames'] == 1
    assert stats['huge_frames'] == 0
    assert stats['nonfinite_frames'] == 0
    assert stats['max'] == pytest.approx(1.0)


def test_frame_lookup_marks_exact_and_mismatch(tmp_path, renderer):
    for stamp in (100, 200, 300):
        _write_jpg(tmp_path / f'{stamp}.jpg')
    exact = renderer._frame_lookup(tmp_path, 200)
    assert exact['stamp_ns'] == 200 and exact['exact'] is True
    mismatch = renderer._frame_lookup(tmp_path, 250)
    assert mismatch['exact'] is False
    assert mismatch['stamp_ns'] in (200, 300)


def test_exact_source_lookup_never_falls_back_to_nearest(tmp_path, renderer):
    ego = tmp_path / 'ego_frames'
    ego.mkdir()
    _write_jpg(ego / '100.jpg')
    _write_jpg(ego / '300.jpg')
    assert renderer._exact_frame_lookup(ego, 100)['path'].endswith('100.jpg')
    # 250 has nearby frames but no exact request/source capture: fail closed.
    assert renderer._exact_frame_lookup(ego, 250) is None
    assert renderer._exact_frame_lookup(ego, None) is None


def test_panel_wraps_without_horizontal_clipping(renderer):
    canvas = np.full((renderer.H, renderer.W, 3), 30, dtype=np.uint8)
    lines = [
        'request session-abc-123',
        'update UNKNOWN',
        'units normalized action accepted',
        'typed no_valid_scan_range_for_bearing',
        'source stamp 22216667825',
        'source age 6.123s OUTDATED',
        'latency 8054.8ms',
        'goal kind pixel_goal',
        'confidence 0.95',
        'grounding accepted: scan_grounded',
        'reason_codes []',
        'source frame exact=True',
        'source path 22216667825.jpg',
        'pixel DRAWN',
        'main frame 23000000000',
        'model Qwen/Qwen3-VL-8B-Instruct-FP8',
    ]
    layout = renderer._draw_panel(canvas, lines)
    assert layout['clipped'] is False
    assert layout['max_line_width_px'] <= layout['panel_width_px'] - 12
    assert layout['drawn_lines'] >= layout['requested_lines']


def test_ego_rows_keep_first_pts_na_and_finite_xN(tmp_path, renderer):
    ego = tmp_path / 'ego_frames'
    ego.mkdir()
    for stamp in (10_000_000_000, 11_000_000_000):
        _write_jpg(ego / f'{stamp}.jpg')
    traces = [
        {'event': 'response', 'response_sim_ns': 10_000_000_000,
         'source_stamp_ns': 10_000_000_000, 'source_age_ns': 6_000_000_000,
         'outdated': True, 'latency_ms': 6000.0, 'request_id': 'r1',
         'update_id': 'u1', 'raw_goal': {'kind': 'pixel_goal', 'units': 'normalized',
                                          'value': {'x': 0.5, 'y': 0.5}, 'confidence': 0.9},
         'grounding': {'status': 'accepted', 'reason': 'scan_grounded'},
         'decision': 'accepted', 'reason_codes': [], 'model': 'qwen'},
        {'event': 'response', 'response_sim_ns': 11_000_000_000,
         'source_stamp_ns': 10_500_000_000, 'source_age_ns': 500_000_000,
         'outdated': True, 'latency_ms': 6000.0, 'request_id': 'r2',
         'update_id': 'u2', 'raw_goal': {'kind': 'pixel_goal', 'units': 'normalized',
                                          'value': {'x': 0.5, 'y': 0.5}, 'confidence': 0.9},
         'grounding': {'status': 'accepted', 'reason': 'scan_grounded'},
         'decision': 'accepted', 'reason_codes': [], 'model': 'qwen'},
    ]
    frames, rows, _hashes, _layout, _t0, _t1 = renderer._build_ego_rows(
        str(tmp_path), traces, [], fps=5.0)
    assert frames
    assert rows[0]['video_pts_sec'] == 0.0
    assert rows[0]['xN'] is None
    assert all(row['xN'] is None or math.isfinite(row['xN']) for row in rows)
    assert max(row['xN'] for row in rows if row['xN'] is not None) == pytest.approx(1.0)
    assert rows[0]['wall_time_sec'] is None
    assert rows[0]['outdated'] is True
    assert rows[0]['source_age_sec'] > 5.0
    # Second trace points at 10.5s, which has no captured frame: fail closed.
    second = next(row for row in rows if row['request_id'] == 'r2')
    assert second['source_frame_exact'] is False
    assert second['source_frame_path'] is None
    assert second['pixel_drawn'] is False
    assert second['pixel_invalid_reason'] == 'SOURCE_FRAME_UNRECORDED'


def test_main_frame_never_newer_than_presentation(tmp_path, renderer):
    ego = tmp_path / 'ego_frames'
    ego.mkdir()
    _write_jpg(ego / '100.jpg')
    _write_jpg(ego / '300.jpg')
    before = renderer._frame_lookup_at_or_before(ego, 250)
    assert before['stamp_ns'] == 100
    assert before['exact'] is False
    assert renderer._frame_lookup_at_or_before(ego, 50) is None
    assert renderer._frame_lookup_at_or_before(ego, 300)['stamp_ns'] == 300


def test_pixel_units_scale_from_source_dimensions(tmp_path, renderer):
    ego = tmp_path / 'ego_frames'
    ego.mkdir()
    _write_jpg(ego / '10000000000.jpg')
    trace = {
        'event': 'response', 'response_sim_ns': 10_000_000_000,
        'source_stamp_ns': 10_000_000_000, 'source_age_ns': 0,
        'outdated': False, 'latency_ms': 10.0, 'request_id': 'r1',
        'update_id': 'u1', 'image_width': 640, 'image_height': 480,
        'raw_goal': {'kind': 'pixel_goal', 'units': 'pixel',
                     'value': {'x': 320, 'y': 240}, 'confidence': 0.9},
        'grounding': {'status': 'accepted', 'reason': 'scan_grounded'},
        'decision': 'accepted', 'reason_codes': [], 'model': 'qwen',
    }
    _frames, rows, _hashes, _layout, _t0, _t1 = renderer._build_ego_rows(
        str(tmp_path), [trace], [], fps=5.0)
    row = rows[0]
    assert row['pixel_units'] == 'pixel'
    assert row['pixel_drawn'] is True
    sx, sy = row['pixel_xy_small']
    # 320/640 and 240/480 both map to the centre of the 240x180 window.
    assert sx == pytest.approx((240 - 1) / 2.0)
    assert sy == pytest.approx((180 - 1) / 2.0)


def test_unsupported_units_fail_closed(tmp_path, renderer):
    ego = tmp_path / 'ego_frames'
    ego.mkdir()
    _write_jpg(ego / '10000000000.jpg')
    trace = {
        'event': 'response', 'response_sim_ns': 10_000_000_000,
        'source_stamp_ns': 10_000_000_000, 'source_age_ns': 0,
        'outdated': False, 'latency_ms': 10.0, 'request_id': 'r1',
        'update_id': 'u1', 'image_width': 640, 'image_height': 480,
        'raw_goal': {'kind': 'pixel_goal', 'units': 'meters',
                     'value': {'x': 0.5, 'y': 0.5}, 'confidence': 0.9},
        'grounding': {'status': 'accepted', 'reason': 'scan_grounded'},
        'decision': 'accepted', 'reason_codes': [], 'model': 'qwen',
    }
    _frames, rows, _hashes, _layout, _t0, _t1 = renderer._build_ego_rows(
        str(tmp_path), [trace], [], fps=5.0)
    assert rows[0]['pixel_drawn'] is False
    assert rows[0]['pixel_invalid_reason'] == 'UNSUPPORTED_UNITS'


def test_marker_header_zero_is_unknown(renderer):
    assert renderer._marker_text(0) == 'UNKNOWN'
    assert renderer._marker_text(None) == 'UNKNOWN'
    assert renderer._marker_text(123) == '123'


def test_tf_availability_marks_unrecorded_source_transform(tmp_path, renderer):
    diag = tmp_path / 'diag'
    diag.mkdir()
    (diag / 'PROX_DIAG.jsonl').write_text('')
    assert renderer._tf_availability(str(diag))['status'] == 'TF_UNRECORDED'
    rows = [
        {'event': 'grounded_goal', 'source_stamp_ns': 123,
         'odom': None, 'tf_error': "AttributeError('PoseStamped')"},
    ]
    (diag / 'PROX_DIAG.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in rows) + '\n')
    availability = renderer._tf_availability(str(diag))
    assert availability['status'] == 'TF_UNRECORDED'
    assert availability['source_time_lookup'] is False


def test_scan_publisher_ownership_stays_unknown_without_graph(tmp_path, renderer):
    diag = tmp_path / 'diag'
    diag.mkdir()
    meta = {'topics': {'scan': {'topic': '/task_generator_node/Ai2_Bot2/laser_scan',
                                'publisher_count': 2}}}
    (diag / 'PROX_DIAG_META.json').write_text(json.dumps(meta))
    ownership = renderer._scan_publisher_ownership(str(diag))
    assert ownership['status'] == 'UNKNOWN'
    assert ownership['publisher_count'] == 2
    assert 'no live ROS graph' in ownership['reason']
