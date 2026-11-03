#!/usr/bin/env python3
"""Offline renderer for the ProxMPC A-track diagnostic videos.

Consumes only recorded real ROS data and the sidecar's allowlisted VLM trace.
No prompt/key/reasoning is present in the inputs; no point is synthesized when a
native marker/range/transform is missing.

v2 (``--name-suffix v2``) writes new files and never overwrites the v1 MP4s.
It recomputes presentation time from frame index / fps (mux PTS), separates
wall latency from sim source age, and emits a timebase audit with unique/held
frame accounting.
"""
import argparse
import bisect
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

MAP_W, MAP_H = 720, 540
PANEL_W = 240
W, H = MAP_W + PANEL_W, MAP_H
COLORS = {
    'free': (245, 245, 245), 'occupied': (40, 40, 220),
    'unknown': (120, 120, 120), 'path': (30, 200, 30),
    'planner': (20, 120, 200), 'goal': (0, 220, 255),
    'tracker': (0, 0, 255), 'marker': (255, 80, 80),
    'footprint': (255, 0, 255), 'odom': (255, 255, 0),
}


def _load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _nearest(rows, key, t):
    times = [r.get(key) for r in rows if r.get(key) is not None]
    if not times:
        return None
    idx = bisect.bisect_right(times, t) - 1
    return rows[idx] if idx >= 0 else None


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ratio(numerator, denominator):
    """Finite ratio or None.  Never divide by 1e-9 to fake a huge xN."""
    top = _finite_or_none(numerator)
    bottom = _finite_or_none(denominator)
    if top is None or bottom is None or bottom <= 0.0:
        return None
    ratio = top / bottom
    return ratio if math.isfinite(ratio) else None


def _frame_lookup(ego_dir, stamp_ns):
    """Nearest real ego frame, with an exact-stamp flag."""
    files = sorted(Path(ego_dir).glob('*.jpg'), key=lambda p: int(p.stem))
    if not files:
        return None
    stamps = [int(p.stem) for p in files]
    requested = int(stamp_ns)
    idx = bisect.bisect_left(stamps, requested)
    if idx <= 0:
        chosen = 0
    elif idx >= len(files):
        chosen = len(files) - 1
    else:
        before = idx - 1
        chosen = before if abs(stamps[before] - requested) <= abs(stamps[idx] - requested) else idx
    return {
        'path': str(files[chosen]),
        'stamp_ns': stamps[chosen],
        'exact': stamps[chosen] == requested,
        'file_count': len(files),
    }


def _exact_frame_lookup(ego_dir, stamp_ns):
    """Exact request/source RGB lookup.  Never nearest, never guess-match."""
    if stamp_ns is None:
        return None
    path = Path(ego_dir) / f'{int(stamp_ns)}.jpg'
    if not path.exists():
        return None
    return {'path': str(path), 'stamp_ns': int(stamp_ns), 'exact': True}


def _frame_lookup_at_or_before(ego_dir, stamp_ns):
    """Main-view capture must not be newer than the presentation time."""
    if stamp_ns is None:
        return None
    files = sorted(Path(ego_dir).glob('*.jpg'), key=lambda p: int(p.stem))
    if not files:
        return None
    stamps = [int(p.stem) for p in files]
    index = bisect.bisect_right(stamps, int(stamp_ns)) - 1
    if index < 0:
        return None
    return {'path': str(files[index]), 'stamp_ns': stamps[index],
            'exact': stamps[index] == int(stamp_ns)}


def _marker_text(value):
    return 'UNKNOWN' if value in (None, 0, '0') else str(value)


def _frame_index(ego_dir, stamp_ns):
    lookup = _frame_lookup(ego_dir, stamp_ns)
    return Path(lookup['path']) if lookup else None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _show(value, fallback='UNKNOWN'):
    if value is None or value == '':
        return fallback
    return str(value)


def _age_text(value):
    if value is None:
        return 'N/A'
    text = f'{value:.3f}s'
    return text + (' CLOCK_AHEAD' if value < 0 else '')


def _wrap_line(text, max_width, font_scale, thickness):
    """Character wrap so every drawn panel line fits inside the panel."""
    lines = []
    current = ''
    for char in str(text):
        candidate = current + char
        (width, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, thickness)
        if width <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = char
    if current:
        lines.append(current)
    return lines or ['']


def _draw_panel(canvas, lines):
    x0 = MAP_W
    cv2.rectangle(canvas, (x0, 0), (W, H), (20, 20, 20), -1)
    margin = 6
    max_width = PANEL_W - 2 * margin
    scale = 0.36
    line_height = 15
    wrapped = []
    for text in lines:
        wrapped.extend(_wrap_line(text, max_width, scale, 1))
    if len(wrapped) * line_height > H - 12:
        scale = 0.30
        line_height = 13
        wrapped = []
        for text in lines:
            wrapped.extend(_wrap_line(text, max_width, scale, 1))
    if len(wrapped) * line_height > H - 12:
        scale = 0.26
        line_height = 12
        wrapped = []
        for text in lines:
            wrapped.extend(_wrap_line(text, max_width, scale, 1))
    y = line_height
    max_px = 0
    for text in wrapped[: int((H - 6) // line_height)]:
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        max_px = max(max_px, width)
        cv2.putText(canvas, text, (x0 + margin, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (230, 230, 230), 1, cv2.LINE_AA)
        y += line_height
    return {
        'font_scale': scale,
        'line_height': line_height,
        'max_line_width_px': max_px,
        'panel_width_px': PANEL_W,
        'clipped': max_px > max_width,
        'drawn_lines': min(len(wrapped), int((H - 6) // line_height)),
        'requested_lines': len(lines),
    }


def _draw_costmap(canvas, costmap, center, window_m=12.0):
    if not costmap:
        cv2.putText(canvas, 'COSTMAP UNKNOWN', (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        return
    info = costmap['info']
    data = costmap['data']
    res = float(info['resolution'])
    width, height = int(info['width']), int(info['height'])
    ox, oy = float(info['origin']['x']), float(info['origin']['y'])
    scale = MAP_H / window_m
    img = np.full((MAP_H, MAP_W, 3), 245, dtype=np.uint8)
    cx, cy = center
    for row in range(height):
        my = oy + (row + 0.5) * res
        py = int(MAP_H / 2 - (my - cy) * scale)
        if py < 0 or py >= MAP_H:
            continue
        base = row * width
        for col in range(width):
            mx = ox + (col + 0.5) * res
            px = int(MAP_W / 2 + (mx - cx) * scale)
            if px < 0 or px >= MAP_W:
                continue
            value = data[base + col]
            if value < 0:
                img[py, px] = COLORS['unknown']
            elif value > 0:
                img[py, px] = COLORS['occupied']
    canvas[:, :MAP_W] = img


def _draw_path(canvas, poses, color, center, window_m=12.0, width=2):
    if not poses:
        return
    scale = MAP_H / window_m
    cx, cy = center
    pts = []
    for p in poses:
        pts.append((int(MAP_W / 2 + (float(p['x']) - cx) * scale),
                    int(MAP_H / 2 - (float(p['y']) - cy) * scale)))
    for a, b in zip(pts, pts[1:]):
        cv2.line(canvas, a, b, color, width, cv2.LINE_AA)
    if pts:
        cv2.circle(canvas, pts[-1], 4, color, -1)


def _draw_markers(canvas, markers, center, window_m=12.0):
    if not markers:
        return
    scale = MAP_H / window_m
    cx, cy = center
    for m in markers:
        for p in m.get('points', [])[:64]:
            px = int(MAP_W / 2 + (float(p['x']) - cx) * scale)
            py = int(MAP_H / 2 - (float(p['y']) - cy) * scale)
            if 0 <= px < MAP_W and 0 <= py < MAP_H:
                cv2.circle(canvas, (px, py), 2, COLORS['marker'], -1)


def _draw_tracker(canvas, tracker, center, window_m=12.0):
    if not tracker:
        return
    scale = MAP_H / window_m
    cx, cy = center
    for o in tracker.get('obstacles', []):
        px = int(MAP_W / 2 + (float(o['x']) - cx) * scale)
        py = int(MAP_H / 2 - (float(o['y']) - cy) * scale)
        if not (0 <= px < MAP_W and 0 <= py < MAP_H):
            continue
        cv2.circle(canvas, (px, py), 7, COLORS['tracker'], 2)
        cv2.putText(canvas, f'{o["id"]}', (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, COLORS['tracker'], 1, cv2.LINE_AA)


def _draw_goal(canvas, goal, center, window_m=12.0):
    """Draw only a real source-time TF result, never a map-frame guess."""
    odom = goal.get('odom') if goal else None
    if not odom or str(odom.get('frame', '')) != 'Ai2_Bot2/odom':
        return
    scale = MAP_H / window_m
    cx, cy = center
    gx, gy = float(odom['x']), float(odom['y'])
    px = int(MAP_W / 2 + (gx - cx) * scale)
    py = int(MAP_H / 2 - (gy - cy) * scale)
    if 0 <= px < MAP_W and 0 <= py < MAP_H:
        cv2.drawMarker(canvas, (px, py), COLORS['goal'], cv2.MARKER_STAR, 18, 2)
        cv2.putText(canvas, 'Qwen goal', (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, COLORS['goal'], 1, cv2.LINE_AA)
        return
    # The goal is real but outside the current 12 m presentation window.
    # Keep it visible as an edge indicator; do not invent an in-window map cell.
    distance = math.hypot(gx - cx, gy - cy)
    edge_x = min(max(px, 12), MAP_W - 12)
    edge_y = min(max(py, 12), MAP_H - 12)
    cv2.drawMarker(canvas, (edge_x, edge_y), COLORS['goal'],
                   cv2.MARKER_TRIANGLE_UP, 14, 2)
    cv2.putText(canvas, f'Qwen goal {distance:.1f}m', (edge_x + 8, edge_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['goal'], 1, cv2.LINE_AA)


def _draw_footprint(canvas, footprint, center, window_m=12.0):
    if not footprint:
        return
    scale = MAP_H / window_m
    cx, cy = center
    pts = [(int(MAP_W / 2 + (float(p['x']) - cx) * scale),
            int(MAP_H / 2 - (float(p['y']) - cy) * scale)) for p in footprint.get('points', [])]
    if len(pts) >= 3:
        cv2.polylines(canvas, [np.array(pts)], True, COLORS['footprint'], 2)


def _draw_odom_trace(canvas, states, center, window_m=12.0):
    scale = MAP_H / window_m
    cx, cy = center
    pts = []
    for s in states[-200:]:
        p = s.get('robot_pose')
        if p:
            pts.append((int(MAP_W / 2 + (float(p['x']) - cx) * scale),
                        int(MAP_H / 2 - (float(p['y']) - cy) * scale)))
    for a, b in zip(pts, pts[1:]):
        cv2.line(canvas, a, b, COLORS['odom'], 1, cv2.LINE_AA)


def _write_video(frames, path, fps):
    if not frames:
        return 0
    writer = None
    for frame in frames:
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(path), fourcc, float(fps), (W, H))
        writer.write(frame)
    if writer is not None:
        writer.release()
    return len(frames)


def _names(suffix):
    tag = f'_{suffix}' if suffix else ''
    return {
        'ego_video': f'ego_vlm_debug{tag}.mp4',
        'ego_pts': f'video_frame_pts_ego{tag}.json',
        'costmap_video': f'local_costmap_prox_debug{tag}.mp4',
        'costmap_pts': f'video_frame_pts_costmap{tag}.json',
        'audit': f'TIMEBASE_AUDIT{tag}.json',
        'hashes': f'FRAME_HASHES{tag}.json',
        'decode': f'DECODE{tag}.json',
        'receipt': f'PROX_DIAG_RENDER_RECEIPT{tag}.json',
        'source_match': f'SOURCE_PATH_MATCH{tag}.json',
        'ego_panel_png': f'PANEL_CHECK_EGO{tag}.png',
        'costmap_panel_png': f'PANEL_CHECK_COSTMAP{tag}.png',
    }


def _probe_video(path):
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames',
        '-show_entries', 'format=duration',
        '-of', 'json', str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 - decode audit is best effort
        return {'error': repr(exc), 'path': str(path)}
    if result.returncode != 0:
        return {'error': (result.stderr or '').strip()[-300:], 'path': str(path),
                'returncode': result.returncode}
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        return {'error': f'ffprobe json: {exc}', 'path': str(path)}
    payload['path'] = str(path)
    return payload


def _summarize(rows, key):
    values = [r.get(key) for r in rows if r.get(key) is not None]
    unique = len(set(values))
    return {
        'rendered_frames': len(rows),
        'unique_values': unique,
        'held_frames': len(values) - unique,
        'value_key': key,
    }


def _xN_stats(rows):
    values = [r['xN'] for r in rows if r.get('xN') is not None]
    nonfinite = sum(1 for r in rows
                    if r.get('xN') is not None and not math.isfinite(r['xN']))
    return {
        'count': len(values),
        'null_or_na_frames': len(rows) - len(values),
        'min': min(values) if values else None,
        'max': max(values) if values else None,
        'mean': sum(values) / len(values) if values else None,
        'huge_frames': sum(1 for v in values if abs(v) > 1000.0),
        'nonfinite_frames': nonfinite,
    }


def _source_path_matches(diag_dir, traces):
    """Frozen request -> exact source RGB path/hash mapping."""
    ego_dir = os.path.join(diag_dir, 'ego_frames')
    rows = []
    for trace in traces:
        stamp = trace.get('source_stamp_ns')
        lookup = _exact_frame_lookup(ego_dir, stamp)
        rows.append({
            'request_id': trace.get('request_id'),
            'update_id': trace.get('update_id'),
            'session_id': trace.get('session_id'),
            'episode': trace.get('episode'),
            'source_stamp_ns': stamp,
            'source_path': lookup['path'] if lookup else None,
            'source_file_sha256': _sha256_file(lookup['path']) if lookup else None,
            'image_sha256': trace.get('image_sha256'),
            'image_width': trace.get('image_width'),
            'image_height': trace.get('image_height'),
            'status': 'MATCH' if lookup else 'SOURCE_FRAME_UNRECORDED',
        })
    return {'schema': 'arena.prox_diag_source_path_match.v2', 'requests': rows}


def _build_ego_rows(diag_dir, traces, states, fps):
    ego_dir = os.path.join(diag_dir, 'ego_frames')
    rows = []
    frames = []
    hashes = []
    layout_metrics = []
    t0 = traces[0].get('response_sim_ns') / 1e9
    t1 = traces[-1].get('response_sim_ns') / 1e9
    prev_main = None
    prev_source = None
    for i in range(int((t1 - t0) * fps) + 1):
        t = t0 + i / fps
        row = None
        for candidate in traces:
            if candidate.get('response_sim_ns', 0) / 1e9 <= t:
                row = candidate
        if row is None:
            continue
        video_pts = i / fps
        sim_elapsed = i / fps
        state = _nearest(states, 'sim_now', t)
        wall_time = state.get('wall_now') if state else None
        # Main presentation capture must be at or before the response time,
        # never a future frame.
        main = _frame_lookup_at_or_before(ego_dir, row.get('response_sim_ns', 0))
        # Source RGB must be the exact request/source stamp.  Nearest/guess
        # matching is forbidden: a missing frame is a typed rejection.
        src = _exact_frame_lookup(ego_dir, row.get('source_stamp_ns'))
        main_stamp = main['stamp_ns'] if main else None
        src_stamp = src['stamp_ns'] if src else None
        response_sim = row.get('response_sim_ns')
        response_sec = response_sim / 1e9 if response_sim is not None else None
        if row.get('source_age_ns') is not None:
            base_age = row['source_age_ns'] / 1e9
        elif response_sec is not None and row.get('source_stamp_ns') is not None:
            base_age = response_sec - row['source_stamp_ns'] / 1e9
        else:
            base_age = None
        source_age = (base_age + (t - response_sec)
                      if base_age is not None and response_sec is not None else None)
        meta = {
            'frame_index': i,
            'video_pts_sec': video_pts,
            'sim_time_sec': t,
            'sim_elapsed_sec': sim_elapsed,
            'xN': _ratio(sim_elapsed, video_pts),
            'wall_time_sec': wall_time,
            'wall_time_source': 'nearest_state_wall_now' if wall_time is not None else 'UNAVAILABLE',
            'request_latency_ms': row.get('latency_ms'),
            'request_id': row.get('request_id'),
            'update_id': row.get('update_id'),
            'source_stamp_ns': row.get('source_stamp_ns'),
            'source_age_sec': source_age,
            'outdated': bool(row.get('outdated')),
            'main_frame_stamp_ns': main_stamp,
            'main_frame_exact': bool(main['exact']) if main else False,
            'held_main': main_stamp is not None and main_stamp == prev_main,
            'source_frame_stamp_ns': src_stamp,
            'source_frame_exact': bool(src['exact']) if src else False,
            'source_frame_path': src['path'] if src else None,
            'source_image_sha256': row.get('image_sha256'),
            'held_source': src_stamp is not None and src_stamp == prev_source,
        }
        prev_main, prev_source = main_stamp, src_stamp

        canvas = np.full((H, W, 3), 30, dtype=np.uint8)
        if main:
            img = cv2.imread(main['path'])
            if img is not None:
                canvas[:, :MAP_W] = cv2.resize(img, (MAP_W, MAP_H))
        pixel_drawn = False
        pixel_invalid_reason = None
        if src:
            img = cv2.imread(src['path'])
            if img is not None:
                small = cv2.resize(img, (240, 180))
                canvas[MAP_H - 190:MAP_H - 10, MAP_W - 250:MAP_W - 10] = small
                raw = row.get('raw_goal') or {}
                value = raw.get('value') if isinstance(raw.get('value'), dict) else {}
                units = raw.get('units')
                image_w = float(row.get('image_width') or 640)
                image_h = float(row.get('image_height') or 480)
                if value.get('x') is None or value.get('y') is None:
                    pixel_invalid_reason = 'PIXEL_MISSING'
                elif units not in ('normalized', 'pixel'):
                    pixel_invalid_reason = 'UNSUPPORTED_UNITS'
                else:
                    if units == 'normalized':
                        sx = float(value['x']) * (small.shape[1] - 1)
                        sy = float(value['y']) * (small.shape[0] - 1)
                    else:
                        sx = (float(value['x']) / image_w) * (small.shape[1] - 1)
                        sy = (float(value['y']) / image_h) * (small.shape[0] - 1)
                    meta['pixel_xy_small'] = (sx, sy)
                    if 0 <= sx < small.shape[1] and 0 <= sy < small.shape[0]:
                        px = MAP_W - 250 + int(sx)
                        py = MAP_H - 190 + int(sy)
                        cv2.drawMarker(canvas, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
                        pixel_drawn = True
                    else:
                        pixel_invalid_reason = 'PIXEL_OUT_OF_BOUNDS'
        else:
            pixel_invalid_reason = 'SOURCE_FRAME_UNRECORDED'
        meta['pixel_drawn'] = pixel_drawn
        meta['pixel_invalid_reason'] = pixel_invalid_reason
        meta['pixel_units'] = (row.get('raw_goal') or {}).get('units')
        grounding = row.get('grounding') or {}
        raw = row.get('raw_goal') or {}
        typed_reason = row.get('typed_rejection')
        if not typed_reason and grounding.get('status') == 'rejected':
            typed_reason = grounding.get('reason')
        lines = [
            f'request={_show(row.get("request_id"))}',
            f'update={_show(row.get("update_id"))}',
            f'action={_show(row.get("decision"))}',
            f'units={_show(raw.get("units"))}',
            f'typed={_show(typed_reason, "none")}',
            f'source_stamp={_show(row.get("source_stamp_ns"))}',
            f'source_age={_age_text(source_age)}'
            + (' OUTDATED' if row.get('outdated') else ''),
            f'latency={_show(row.get("latency_ms"))}ms',
            f'goal_kind={_show(raw.get("kind"))}',
            f'confidence={_show(raw.get("confidence"))}',
            f'grounding={_show(grounding.get("status"))}:{_show(grounding.get("reason"))}',
            f'reason_codes={_show(row.get("reason_codes"), "none")}',
            f'source_frame_exact={meta["source_frame_exact"]}',
            f'source_path={os.path.basename(src["path"]) if src else "UNRECORDED"}',
            f'pixel={("DRAWN" if pixel_drawn else "INVALID")}'
            + (f' {pixel_invalid_reason}' if pixel_invalid_reason else ''),
            f'main_frame={_show(main_stamp)} exact={meta["main_frame_exact"]}',
            f'model={_show(row.get("model"))}',
            'schema=v2',
        ]
        layout = _draw_panel(canvas, lines)
        layout_metrics.append(layout)
        frames.append(canvas)
        rows.append(meta)
        hashes.append({
            'frame_index': i,
            'main_frame_stamp_ns': main_stamp,
            'main_frame_sha256': _sha256_file(main['path']) if main else None,
            'source_frame_stamp_ns': src_stamp,
            'source_frame_sha256': _sha256_file(src['path']) if src else None,
            'pixel_drawn': pixel_drawn,
            'pixel_invalid_reason': pixel_invalid_reason,
        })
    return frames, rows, hashes, layout_metrics, t0, t1


def render_ego(diag_dir, out_dir, fps, names):
    traces = [r for r in _load_jsonl(os.path.join(diag_dir, 'VLM_TRACE.jsonl'))
              if r.get('event') == 'response']
    states = [r for r in _load_jsonl(os.path.join(diag_dir, 'PROX_DIAG.jsonl'))
              if r.get('event') == 'state']
    if not traces:
        return None
    traces.sort(key=lambda r: r.get('response_sim_ns') or 0)
    frames, rows, hashes, layout_metrics, t0, t1 = _build_ego_rows(
        diag_dir, traces, states, fps)
    path = os.path.join(out_dir, names['ego_video'])
    count = _write_video(frames, path, fps)
    Path(os.path.join(out_dir, names['ego_pts'])).write_text(
        json.dumps(rows, indent=2) + '\n')
    if frames:
        cv2.imwrite(os.path.join(out_dir, names['ego_panel_png']), frames[0])
    ego_dir = os.path.join(diag_dir, 'ego_frames')
    captured = len(list(Path(ego_dir).glob('*.jpg')))
    panel_layout = {
        'screenshots': [names['ego_panel_png']],
        'frames_checked': len(layout_metrics),
        'clipped_frames': sum(1 for m in layout_metrics if m.get('clipped')),
        'max_line_width_px': max((m.get('max_line_width_px', 0) for m in layout_metrics),
                                 default=0),
        'panel_width_px': PANEL_W,
        'min_font_scale': min((m.get('font_scale', 0) for m in layout_metrics), default=None),
        'dropped_lines_frames': sum(1 for m in layout_metrics
                                    if m.get('drawn_lines', 0) < m.get('requested_lines', 0)),
    }
    audit = {
        'source': 'VLM_TRACE response frames + recorded ego_frames',
        'rendered_frames': count,
        'ego_captured_frames': captured,
        'video_pts_duration_sec': (count - 1) / fps if count else 0.0,
        'sim_duration_sec': (t1 - t0) if frames else 0.0,
        'unique_main_frames': len({r['main_frame_stamp_ns'] for r in rows
                                   if r['main_frame_stamp_ns'] is not None}),
        'held_main_frames': sum(1 for r in rows if r['held_main']),
        'unique_source_window_frames': len({r['source_frame_stamp_ns'] for r in rows
                                            if r['source_frame_stamp_ns'] is not None}),
        'held_source_window_frames': sum(1 for r in rows if r['held_source']),
        'source_frame_mismatch_frames': sum(1 for r in rows if not r['source_frame_exact']),
        'main_frame_mismatch_frames': sum(1 for r in rows if not r['main_frame_exact']),
        'outdated_frames': sum(1 for r in rows if r['outdated']),
        'pixel_invalid_frames': sum(1 for r in rows if not r['pixel_drawn']),
        'xN': _xN_stats(rows),
        'first_pts_sec': rows[0]['video_pts_sec'] if rows else None,
        'wall_time_null_frames': sum(1 for r in rows if r['wall_time_sec'] is None),
        'panel_layout': panel_layout,
    }
    source_matches = _source_path_matches(diag_dir, traces)
    Path(os.path.join(out_dir, names['source_match'])).write_text(
        json.dumps(source_matches, indent=2, sort_keys=True) + '\n')
    return {'path': path, 'frames': count, 'duration_sec': (t1 - t0) if frames else 0,
            'pts_file': os.path.join(out_dir, names['ego_pts']),
            'source_path_match': os.path.join(out_dir, names['source_match']),
            'audit': audit, 'hashes': hashes}


def _build_costmap_rows(diag_dir, fps):
    events = _load_jsonl(os.path.join(diag_dir, 'PROX_DIAG.jsonl'))
    traces = [r for r in _load_jsonl(os.path.join(diag_dir, 'VLM_TRACE.jsonl'))
              if r.get('event') == 'response']
    traces.sort(key=lambda r: r.get('response_sim_ns') or 0)
    states = [r for r in events if r.get('event') == 'state']
    costmaps = [r for r in events if r.get('event') == 'costmap']
    plans = [r for r in events if r.get('event') == 'local_plan']
    markers = [r for r in events if r.get('event') == 'predicted_markers']
    trackers = [r for r in events if r.get('event') == 'tracker']
    goals = [r for r in events if r.get('event') == 'grounded_goal']
    planner_paths = [r for r in events if r.get('event') == 'planner_path']
    diags = [r for r in events if r.get('event') == 'diagnostics']
    collisions = [r for r in events if r.get('event') == 'collision_monitor']
    scans = [r for r in events if r.get('event') == 'scan']
    if not states:
        return None
    t0 = states[0]['sim_now']
    t1 = states[-1]['sim_now']
    rows = []
    frames = []
    layout_metrics = []
    prev = {}
    for i in range(int((t1 - t0) * fps) + 1):
        t = t0 + i / fps
        state = _nearest(states, 'sim_now', t) or states[-1]
        pose = state.get('robot_pose') or {'x': 0.0, 'y': 0.0}
        center = (float(pose.get('x', 0.0)), float(pose.get('y', 0.0)))
        video_pts = i / fps
        sim_elapsed = i / fps
        cm = _nearest(costmaps, 'sim_now', t)
        lp = _nearest(plans, 'sim_now', t)
        mk = _nearest(markers, 'sim_now', t)
        tr = _nearest(trackers, 'sim_now', t)
        goal = _nearest(goals, 'sim_now', t)
        pp = _nearest(planner_paths, 'sim_now', t)
        dg = _nearest(diags, 'sim_now', t)
        coll = _nearest(collisions, 'sim_now', t)
        scan = _nearest(scans, 'sim_now', t)
        stamps = {
            'costmap': cm.get('source_stamp_ns') if cm else None,
            'local_plan': lp.get('source_stamp_ns') if lp else None,
            'marker': mk.get('source_stamp_ns') if mk else None,
            'tracker': tr.get('source_stamp_ns') if tr else None,
            'planner_path': pp.get('source_stamp_ns') if pp else None,
            'scan': scan.get('source_stamp_ns') if scan else None,
        }
        meta = {
            'frame_index': i,
            'video_pts_sec': video_pts,
            'sim_time_sec': t,
            'sim_elapsed_sec': sim_elapsed,
            'xN': _ratio(sim_elapsed, video_pts),
            'wall_time_sec': state.get('wall_now'),
            'costmap_stamp_ns': stamps['costmap'],
            'scan_stamp_ns': stamps['scan'],
            'local_plan_stamp_ns': stamps['local_plan'],
            'marker_stamp_ns': stamps['marker'],
            'tracker_stamp_ns': stamps['tracker'],
            'planner_path_stamp_ns': stamps['planner_path'],
        }
        for key, stamp in stamps.items():
            meta[f'{key}_age_sec'] = (t - stamp / 1e9) if stamp is not None else None
            meta[f'held_{key}'] = (stamp is not None and stamp == prev.get(key))
            prev[key] = stamp
        goal_tf = 'TF_UNRECORDED'
        goal_tf_reason = None
        goal_req = None
        goal_update = None
        goal_frame = None
        goal_stamp = None
        if goal:
            goal_frame = goal.get('frame') or goal.get('frame_id')
            goal_stamp = (goal.get('stamp') if goal.get('stamp') is not None
                          else goal.get('source_stamp_ns'))
            goal_tf = goal.get('tf_status') or ('OK' if goal.get('odom') else 'TF_UNRECORDED')
            goal_tf_reason = goal.get('tf_error')
            for trace in traces:
                map_pose = (trace.get('grounding') or {}).get('map_pose') or {}
                if map_pose.get('stamp_ns') == goal_stamp:
                    goal_req = trace.get('request_id')
                    goal_update = trace.get('update_id')
                    break
        meta['goal_tf_status'] = goal_tf
        meta['goal_tf_reason'] = goal_tf_reason
        meta['goal_request_id'] = goal_req
        meta['goal_update_id'] = goal_update
        meta['goal_map'] = ({'x': goal.get('x'), 'y': goal.get('y'),
                             'frame': goal_frame, 'stamp_ns': goal_stamp}
                            if goal else None)
        meta['goal_odom'] = goal.get('odom') if goal else None

        canvas = np.full((H, W, 3), 245, dtype=np.uint8)
        _draw_costmap(canvas, cm, center)
        if pp:
            _draw_path(canvas, pp.get('odom_poses') or pp.get('poses'),
                       COLORS['planner'], center)
        if lp:
            _draw_path(canvas, lp.get('poses'), COLORS['path'], center)
        if mk:
            _draw_markers(canvas, mk.get('markers'), center)
        if tr:
            _draw_tracker(canvas, tr, center)
        if goal and goal.get('odom'):
            _draw_goal(canvas, goal, center)
        fp = state.get('footprint')
        if fp:
            _draw_footprint(canvas, fp, center)
        _draw_odom_trace(canvas, states, center)
        vlm = _nearest(traces, 'response_sim_ns', int(t * 1e9)) if traces else None
        goal_map_text = ('UNAVAILABLE (no grounded_goal at this t)'
                         if not goal else
                         f'{goal.get("x"):.2f},{goal.get("y"):.2f} '
                         f'frame={_show(goal_frame)} stamp={_show(goal_stamp)}')
        goal_tf_text = goal_tf
        if goal_tf_reason:
            goal_tf_text = f'{goal_tf} ({goal_tf_reason})'
        lines = [
            f't={t - t0:.1f}s pts={video_pts:.1f}s '
            f'xN={meta["xN"] if meta["xN"] is not None else "N/A"}',
            f'VLM={_show((vlm or {}).get("decision"), "NONE")}',
            f'VLM_request={_show((vlm or {}).get("request_id"))}',
            f'VLM_update={_show((vlm or {}).get("update_id"))}',
            f'VLM_model={_show((vlm or {}).get("model"))}',
            f'VLM_typed={_show((vlm or {}).get("typed_rejection"), "none")}',
            f'grounding={_show((vlm or {}).get("grounding", {}).get("status"))}',
            f'goal_map={goal_map_text}',
            f'goal_request={_show(goal_req)}',
            f'goal_update={_show(goal_update)}',
            f'goal_TF={goal_tf_text}',
            f'costmap_age={_age_text(meta["costmap_age_sec"])}',
            f'scan_age={_age_text(meta["scan_age_sec"])}',
            f'plan_age={_age_text(meta["local_plan_age_sec"])}',
            f'marker_age={_age_text(meta["marker_age_sec"])}',
            f'solve={_show((dg or {}).get("solve_time_ms"))}ms',
            f'slack={_show((dg or {}).get("max_obstacle_slack"))}',
            f'deadline_missed={_show((dg or {}).get("deadline_missed"))}',
            f'active={_show((dg or {}).get("num_active_obstacles"))}',
            'static_dynamic=UNKNOWN',
            f'tracker={_show((tr or {}).get("count"))}',
            f'control_age={_show((dg or {}).get("control_source_age_s"))}',
            'fallback=UNKNOWN',
            f'safety={_show((coll or {}).get("action_type"))}',
            f'cmd={_show(state.get("cmd"))}',
            f'marker_stamp={_marker_text(state.get("marker_stamp"))}',
            'schema=v2',
        ]
        layout = _draw_panel(canvas, lines)
        layout_metrics.append(layout)
        frames.append(canvas)
        rows.append(meta)
    return frames, rows, layout_metrics, t0, t1, {
        'costmap_messages': len(costmaps), 'scan_messages': len(scans),
        'local_plan_messages': len(plans), 'marker_messages': len(markers),
        'tracker_messages': len(trackers), 'diagnostics_messages': len(diags),
        'collision_messages': len(collisions), 'goal_messages': len(goals),
        'planner_path_messages': len(planner_paths),
    }


def render_costmap(diag_dir, out_dir, fps, names):
    built = _build_costmap_rows(diag_dir, fps)
    if not built:
        return None
    frames, rows, layout_metrics, t0, t1, counts = built
    path = os.path.join(out_dir, names['costmap_video'])
    count = _write_video(frames, path, fps)
    Path(os.path.join(out_dir, names['costmap_pts'])).write_text(
        json.dumps(rows, indent=2) + '\n')
    if frames:
        goal_frames = [r['frame_index'] for r in rows
                       if r.get('goal_map') is not None]
        goal_index = goal_frames[-1] if goal_frames else len(frames) // 2
        cv2.imwrite(os.path.join(out_dir, names['costmap_panel_png']),
                    frames[min(goal_index, len(frames) - 1)])
    panel_layout = {
        'screenshots': [names['costmap_panel_png']],
        'frames_checked': len(layout_metrics),
        'clipped_frames': sum(1 for m in layout_metrics if m.get('clipped')),
        'max_line_width_px': max((m.get('max_line_width_px', 0) for m in layout_metrics),
                                 default=0),
        'panel_width_px': PANEL_W,
        'min_font_scale': min((m.get('font_scale', 0) for m in layout_metrics), default=None),
        'dropped_lines_frames': sum(1 for m in layout_metrics
                                    if m.get('drawn_lines', 0) < m.get('requested_lines', 0)),
    }
    goal_chain = []
    seen_goals = set()
    for r in rows:
        goal_map = r.get('goal_map')
        if goal_map is None:
            continue
        key = (goal_map.get('stamp_ns'), goal_map.get('x'), goal_map.get('y'))
        if key in seen_goals:
            continue
        seen_goals.add(key)
        goal_chain.append({
            'first_frame_index': r['frame_index'],
            'goal_map': goal_map,
            'goal_tf_status': r.get('goal_tf_status'),
            'goal_tf_reason': r.get('goal_tf_reason'),
            'goal_odom': r.get('goal_odom'),
            'goal_request_id': r.get('goal_request_id'),
            'goal_update_id': r.get('goal_update_id'),
        })
    audit = {
        'source': 'PROX_DIAG state timeline + native costmap/scan/plan/marker',
        'rendered_frames': count,
        'video_pts_duration_sec': (count - 1) / fps if count else 0.0,
        'sim_duration_sec': (t1 - t0) if frames else 0.0,
        'unique_costmap_stamps': len({r['costmap_stamp_ns'] for r in rows
                                      if r['costmap_stamp_ns'] is not None}),
        'held_costmap_frames': sum(1 for r in rows if r['held_costmap']),
        'unique_scan_stamps': len({r['scan_stamp_ns'] for r in rows
                                   if r['scan_stamp_ns'] is not None}),
        'held_scan_frames': sum(1 for r in rows if r['held_scan']),
        'unique_local_plan_stamps': len({r['local_plan_stamp_ns'] for r in rows
                                         if r['local_plan_stamp_ns'] is not None}),
        'held_local_plan_frames': sum(1 for r in rows if r['held_local_plan']),
        'unique_marker_stamps': len({r['marker_stamp_ns'] for r in rows
                                     if r['marker_stamp_ns'] is not None}),
        'held_marker_frames': sum(1 for r in rows if r['held_marker']),
        'goal_tf_statuses': sorted({r['goal_tf_status'] for r in rows}),
        'goal_chain': goal_chain,
        'panel_layout': panel_layout,
        'xN': _xN_stats(rows),
        'first_pts_sec': rows[0]['video_pts_sec'] if rows else None,
        'wall_time_null_frames': sum(1 for r in rows if r['wall_time_sec'] is None),
        'sparse_note': (
            f'rendered {count} presentation frames at {fps} fps; native '
            f'costmap messages={counts["costmap_messages"]}, '
            f'scan messages={counts["scan_messages"]}; non-new geometry is '
            'held presentation, not unique captures'),
    }
    return {'path': path, 'frames': count, 'duration_sec': (t1 - t0) if frames else 0,
            'pts_file': os.path.join(out_dir, names['costmap_pts']),
            'audit': audit, 'native_counts': counts}


def _tf_availability(diag_dir):
    goals = [r for r in _load_jsonl(os.path.join(diag_dir, 'PROX_DIAG.jsonl'))
             if r.get('event') == 'grounded_goal']
    if not goals:
        return {'status': 'TF_UNRECORDED', 'goal_messages': 0,
                'reason': 'no grounded_goal event in saved data'}
    latest = goals[-1]
    if latest.get('odom'):
        return {'status': 'OK', 'goal_messages': len(goals),
                'source_time_lookup': True,
                'result_frame': latest['odom'].get('frame'),
                'result_stamp_ns': latest['odom'].get('stamp_ns'),
                'result_type': latest['odom'].get('result_type')}
    return {'status': latest.get('tf_status') or 'TF_UNRECORDED',
            'goal_messages': len(goals),
            'tf_error': latest.get('tf_error'),
            'source_time_lookup': False,
            'reason': 'saved goal event has no source-time map->odom transform'}


def _scan_publisher_ownership(diag_dir):
    meta_path = os.path.join(diag_dir, 'PROX_DIAG_META.json')
    try:
        meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return {'status': 'UNKNOWN', 'reason': 'no readable PROX_DIAG_META.json'}
    scan = (meta.get('topics') or {}).get('scan') or {}
    return {
        'status': 'UNKNOWN',
        'publisher_count': scan.get('publisher_count'),
        'topic': scan.get('topic'),
        'reason': ('saved metadata records the count only; no live ROS graph '
                   'node/GID/owner attribution was captured'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag-dir', required=True)
    parser.add_argument('--out-dir', default='')
    parser.add_argument('--fps', type=float, default=5.0)
    parser.add_argument('--name-suffix', default='',
                        help="output filename suffix, e.g. 'v2'; empty keeps v1 names")
    args = parser.parse_args()
    out_dir = args.out_dir or args.diag_dir
    os.makedirs(out_dir, exist_ok=True)
    names = _names(args.name_suffix)
    ego = render_ego(args.diag_dir, out_dir, args.fps, names)
    costmap = render_costmap(args.diag_dir, out_dir, args.fps, names)
    tf = _tf_availability(args.diag_dir)
    ownership = _scan_publisher_ownership(args.diag_dir)
    audit = {
        'schema': 'arena.prox_diag_timebase_audit.v2',
        'diag_dir': args.diag_dir,
        'fps': args.fps,
        'name_suffix': args.name_suffix,
        'ego': (ego or {}).get('audit'),
        'costmap': (costmap or {}).get('audit'),
        'tf_availability': tf,
        'scan_publisher_ownership': ownership,
        'negative_checks': {
            'huge_xN_frames': ((ego or {}).get('audit', {}).get('xN', {}).get('huge_frames', 0)
                               + (costmap or {}).get('audit', {}).get('xN', {}).get('huge_frames', 0)),
            'nonfinite_xN_frames': ((ego or {}).get('audit', {}).get('xN', {}).get('nonfinite_frames', 0)
                                    + (costmap or {}).get('audit', {}).get('xN', {}).get('nonfinite_frames', 0)),
        },
    }
    Path(os.path.join(out_dir, names['audit'])).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + '\n')
    hashes = {
        'schema': 'arena.prox_diag_frame_hashes.v2',
        'ego': (ego or {}).get('hashes', []),
        'ego_video_sha256': _sha256_file(ego['path']) if ego else None,
        'costmap_video_sha256': _sha256_file(costmap['path']) if costmap else None,
    }
    Path(os.path.join(out_dir, names['hashes'])).write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    decode = {
        'schema': 'arena.prox_diag_decode.v2',
        'ego': _probe_video(ego['path']) if ego else None,
        'costmap': _probe_video(costmap['path']) if costmap else None,
    }
    Path(os.path.join(out_dir, names['decode'])).write_text(
        json.dumps(decode, indent=2, sort_keys=True) + '\n')
    renderer_path = str(Path(__file__).resolve())
    source_lines = Path(__file__).read_text(encoding='utf-8').splitlines()
    exact_line = next((index + 1 for index, line in enumerate(source_lines)
                       if line.strip().startswith('def _exact_frame_lookup')), None)
    receipt = {
        'schema': 'arena.prox_diag_render.v2',
        'diag_dir': args.diag_dir,
        'fps': args.fps,
        'name_suffix': args.name_suffix,
        'ego': ego,
        'costmap': costmap,
        'timebase_audit': os.path.join(out_dir, names['audit']),
        'frame_hashes': os.path.join(out_dir, names['hashes']),
        'decode': os.path.join(out_dir, names['decode']),
        'source_path_match': os.path.join(out_dir, names['source_match']),
        'source_lookup': {
            'function': '_exact_frame_lookup',
            'path': renderer_path,
            'line': exact_line,
            'mode': 'exact request/source stamp; missing => SOURCE_FRAME_UNRECORDED',
            'negative_test': ('arena_bringup/test/test_prox_diag_timebase.py::'
                              'test_exact_source_lookup_never_falls_back_to_nearest'),
        },
        'tests': [
            'arena_bringup/test/test_prox_diag_timebase.py',
            'arena_bringup/test/test_prox_diag_projection.py',
            'arena_bringup/test/test_lisn_scan_grounding.py',
        ],
        'tf_availability': tf,
        'scan_publisher_ownership': ownership,
        'panel_checks': {
            'ego': os.path.join(out_dir, names['ego_panel_png']),
            'costmap': os.path.join(out_dir, names['costmap_panel_png']),
        },
    }
    Path(os.path.join(out_dir, names['receipt'])).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
