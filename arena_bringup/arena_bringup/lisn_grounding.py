"""Minimal live-grounding math for the LISN slow loop.

Pure functions only: no ROS, no network, no model. A VLM point is accepted
only when its kind/units/frame/confidence are explicit, the depth sample is
finite, the CameraInfo intrinsics are finite, and an exact-stamp map<-camera
transform is supplied. Unknown map coordinates are never trusted.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

POINT_KINDS = frozenset({'pixel_goal', 'point'})
POINT_UNITS = frozenset({'pixel', 'normalized'})


class GroundingError(ValueError):
    """Typed fail-closed grounding error."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalize_point(goal: Mapping[str, Any], *, width: int, height: int,
                    min_confidence: float = 0.5) -> dict[str, Any]:
    """Validate a VLM point and convert it to integer pixel coordinates."""
    if not isinstance(goal, Mapping):
        raise GroundingError('goal_not_object')
    kind = goal.get('kind')
    if kind not in POINT_KINDS:
        raise GroundingError(f'unsupported_point_kind:{kind}')
    units = goal.get('units')
    if units not in POINT_UNITS:
        raise GroundingError(f'missing_or_invalid_units:{units}')
    frame_id = goal.get('frame_id')
    if not isinstance(frame_id, str) or not frame_id or frame_id == 'map':
        raise GroundingError(f'invalid_camera_frame:{frame_id}')
    value = goal.get('value')
    if not isinstance(value, Mapping):
        raise GroundingError('goal_value_not_object')
    x, y = value.get('x'), value.get('y')
    if not _finite(x) or not _finite(y):
        raise GroundingError('point_not_finite')
    confidence = goal.get('confidence')
    if not _finite(confidence) or not (0.0 <= float(confidence) <= 1.0):
        raise GroundingError('missing_or_invalid_confidence')
    if float(confidence) < float(min_confidence):
        raise GroundingError(f'confidence_below_min:{confidence}<{min_confidence}')
    if width <= 0 or height <= 0:
        raise GroundingError('invalid_image_size')
    if units == 'normalized':
        u = float(x) * (width - 1)
        v = float(y) * (height - 1)
    else:
        u = float(x)
        v = float(y)
    if not (0 <= u <= width - 1 and 0 <= v <= height - 1):
        raise GroundingError(f'point_out_of_image:{u},{v}')
    return {
        'u': int(round(u)),
        'v': int(round(v)),
        'units': units,
        'frame_id': frame_id,
        'confidence': float(confidence),
        'kind': kind,
    }


def project_pixel_to_map(*, u: int, v: int, depth_m: float,
                         camera_info_k: Sequence[float],
                         tf_map_camera: Sequence[Sequence[float]]) -> dict[str, float]:
    """Pinhole projection + rigid transform map<-camera."""
    if not _finite(depth_m) or float(depth_m) <= 0.0:
        raise GroundingError('invalid_depth')
    if len(camera_info_k) != 9 or not all(_finite(x) for x in camera_info_k):
        raise GroundingError('invalid_camera_info_k')
    fx, fy = float(camera_info_k[0]), float(camera_info_k[4])
    cx, cy = float(camera_info_k[2]), float(camera_info_k[5])
    if fx <= 0 or fy <= 0:
        raise GroundingError('invalid_focal_length')
    z = float(depth_m)
    x_cam = (float(u) - cx) * z / fx
    y_cam = (float(v) - cy) * z / fy
    tf = [[float(x) for x in row] for row in tf_map_camera]
    if len(tf) != 4 or any(len(row) != 4 for row in tf) or not all(_finite(x) for row in tf for x in row):
        raise GroundingError('invalid_tf_matrix')
    x_map = tf[0][0] * x_cam + tf[0][1] * y_cam + tf[0][2] * z + tf[0][3]
    y_map = tf[1][0] * x_cam + tf[1][1] * y_cam + tf[1][2] * z + tf[1][3]
    z_map = tf[2][0] * x_cam + tf[2][1] * y_cam + tf[2][2] * z + tf[2][3]
    if not (_finite(x_map) and _finite(y_map) and _finite(z_map)):
        raise GroundingError('projected_point_not_finite')
    return {'x': x_map, 'y': y_map, 'z': z_map}


def _mat4_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(float(a[i][k]) * float(b[k][j]) for k in range(4)) for j in range(4)] for i in range(4)]


def _mat4_vec(a: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [sum(float(a[i][k]) * float(v[k]) for k in range(3)) + float(a[i][3]) for i in range(3)]


def _mat3_vec(a: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [sum(float(a[i][k]) * float(v[k]) for k in range(3)) for i in range(3)]


def _transpose3(a: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(a[j][i]) for j in range(3)] for i in range(3)]


def _validate_tf(tf: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if len(tf) != 4 or any(len(row) != 4 for row in tf):
        raise GroundingError(f'invalid_{name}')
    out = [[float(x) for x in row] for row in tf]
    if not all(_finite(x) for row in out for x in row):
        raise GroundingError(f'{name}_not_finite')
    return out


def _camera_ray_scan_bearing(*, u: int, v: int, camera_info_k: Sequence[float],
                             tf_map_camera: Sequence[Sequence[float]],
                             tf_map_scan: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Return (bearing_rad_in_scan_frame, vertical_component_in_scan_frame)."""
    if len(camera_info_k) != 9 or not all(_finite(x) for x in camera_info_k):
        raise GroundingError('invalid_camera_info_k')
    fx, fy = float(camera_info_k[0]), float(camera_info_k[4])
    cx, cy = float(camera_info_k[2]), float(camera_info_k[5])
    if fx <= 0 or fy <= 0:
        raise GroundingError('invalid_focal_length')
    d_cam = [(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0]
    norm = math.sqrt(sum(x * x for x in d_cam))
    if not math.isfinite(norm) or norm <= 0.0:
        raise GroundingError('invalid_camera_ray')
    d_cam = [x / norm for x in d_cam]
    r_map_cam = _transpose3([row[:3] for row in tf_map_camera])
    r_map_scan = _transpose3([row[:3] for row in tf_map_scan])
    # R_scan_cam = R_map_scan^T * R_map_camera
    r_scan_cam = [[sum(r_map_scan[k][i] * r_map_cam[k][j] for k in range(3))
                   for j in range(3)] for i in range(3)]
    d_scan = _mat3_vec(r_scan_cam, d_cam)
    bearing = math.atan2(d_scan[1], d_scan[0])
    if not math.isfinite(bearing):
        raise GroundingError('scan_bearing_not_finite')
    return bearing, d_scan[2]


def resolve_scan_grounded_goal(*, goal: Mapping[str, Any], width: int, height: int,
                               camera_info_k: Sequence[float],
                               tf_map_camera: Sequence[Sequence[float]],
                               scan_ranges: Sequence[float],
                               scan_angle_min: float, scan_angle_increment: float,
                               scan_range_min: float, scan_range_max: float,
                               tf_map_scan: Sequence[Sequence[float]],
                               expected_camera_frame: str,
                               expected_scan_frame: str,
                               source_stamp_ns: int, scan_stamp_ns: int,
                               scan_age_ns: int, max_scan_age_ns: int,
                               min_confidence: float = 0.5,
                               max_scan_range_m: float = 7.0,
                               beam_search_radius: int = 12) -> dict[str, Any]:
    """Resolve a VLM pixel/bearing with a same-source real 2D LiDAR range.

    This is the A-track resolver: it deliberately does **not** use a depth image
    or CameraInfo depth.  The pixel ray is rotated into the scan frame using the
    frozen source-time TF, the nearest finite beam within a bounded search
    window supplies the metric range, and the measured point is transformed to
    map.  A missing/invalid range fails closed; no map point is invented.
    """
    point = normalize_point(goal, width=width, height=height,
                            min_confidence=min_confidence)
    if point['frame_id'] != expected_camera_frame:
        raise GroundingError(f'camera_frame_mismatch:{point["frame_id"]}!={expected_camera_frame}')
    if not isinstance(source_stamp_ns, int) or source_stamp_ns <= 0:
        raise GroundingError('invalid_source_stamp')
    if not isinstance(scan_stamp_ns, int) or scan_stamp_ns <= 0:
        raise GroundingError('invalid_scan_stamp')
    if not isinstance(scan_age_ns, int) or scan_age_ns < 0:
        raise GroundingError('invalid_scan_age')
    if scan_age_ns > int(max_scan_age_ns):
        raise GroundingError(f'scan_age_exceeds_bound:{scan_age_ns}>{max_scan_age_ns}')
    if not isinstance(expected_scan_frame, str) or not expected_scan_frame:
        raise GroundingError('invalid_expected_scan_frame')
    ranges = [float(r) for r in scan_ranges]
    if not ranges:
        raise GroundingError('empty_scan')
    if not _finite(scan_angle_increment) or float(scan_angle_increment) <= 0.0:
        raise GroundingError('invalid_scan_angle_increment')
    if not (_finite(scan_angle_min) and _finite(scan_range_min) and _finite(scan_range_max)):
        raise GroundingError('invalid_scan_geometry')
    if float(scan_range_min) < 0.0 or float(scan_range_max) <= float(scan_range_min):
        raise GroundingError('invalid_scan_range_window')
    if not _finite(max_scan_range_m) or float(max_scan_range_m) <= 0.0:
        raise GroundingError('invalid_max_scan_range')
    tf_cam = _validate_tf(tf_map_camera, 'tf_map_camera')
    tf_scan = _validate_tf(tf_map_scan, 'tf_map_scan')
    bearing, _vertical = _camera_ray_scan_bearing(
        u=point['u'], v=point['v'], camera_info_k=camera_info_k,
        tf_map_camera=tf_cam, tf_map_scan=tf_scan,
    )
    inc = float(scan_angle_increment)
    amin = float(scan_angle_min)
    n = len(ranges)
    angle_max = amin + inc * (n - 1)
    full_turn = abs((angle_max - amin) - 2.0 * math.pi) <= 1e-3
    index = int(round((bearing - amin) / inc))
    candidates = [index]
    for delta in range(1, max(0, int(beam_search_radius)) + 1):
        candidates += [index - delta, index + delta]
    chosen = None
    for raw_index in candidates:
        idx = raw_index
        if full_turn:
            idx %= n
        elif idx < 0 or idx >= n:
            continue
        r = ranges[idx]
        if not math.isfinite(r):
            continue
        if r < float(scan_range_min) or r > float(scan_range_max):
            continue
        if r > float(max_scan_range_m):
            continue
        chosen = (idx, r)
        break
    if chosen is None:
        raise GroundingError('no_valid_scan_range_for_bearing')
    idx, r = chosen
    beam_angle = amin + inc * idx
    angular_offset = math.atan2(
        math.sin(beam_angle - bearing), math.cos(beam_angle - bearing)
    )
    p_scan = [r * math.cos(beam_angle), r * math.sin(beam_angle), 0.0]
    p_map = _mat4_vec(tf_scan, p_scan)
    if not all(_finite(x) for x in p_map):
        raise GroundingError('projected_scan_point_not_finite')
    return {
        'trusted': True,
        'source_valid': True,
        'reason': 'scan_grounded',
        'point': point,
        'map_pose': {'frame_id': 'map', 'stamp_ns': int(source_stamp_ns),
                     'x': p_map[0], 'y': p_map[1], 'z': p_map[2]},
        'confidence': point['confidence'],
        'source_stamp_ns': int(source_stamp_ns),
        'scan': {
            'frame_id': expected_scan_frame,
            'stamp_ns': int(scan_stamp_ns),
            'age_ns': int(scan_age_ns),
            'beam_index': int(idx),
            'beam_count': int(n),
            'beam_angle_rad': float(beam_angle),
            'angular_offset_rad': float(angular_offset),
            'range_m': float(r),
            'bearing_rad': float(bearing),
            'angle_min_rad': float(amin),
            'angle_increment_rad': float(inc),
            'range_min_m': float(scan_range_min),
            'range_max_m': float(scan_range_max),
            'max_scan_range_m': float(max_scan_range_m),
        },
    }


def resolve_grounded_goal(*, goal: Mapping[str, Any], width: int, height: int,
                          depth_image: Sequence[Sequence[float]],
                          camera_info_k: Sequence[float],
                          tf_map_camera: Sequence[Sequence[float]],
                          expected_frame_id: str,
                          source_stamp_ns: int,
                          min_confidence: float = 0.5) -> dict[str, Any]:
    """Return a trusted map goal or raise a typed fail-closed error."""
    point = normalize_point(goal, width=width, height=height,
                            min_confidence=min_confidence)
    if point['frame_id'] != expected_frame_id:
        raise GroundingError(f'camera_frame_mismatch:{point["frame_id"]}!=expected')
    if not isinstance(source_stamp_ns, int) or source_stamp_ns <= 0:
        raise GroundingError('invalid_source_stamp')
    depth_rows = len(depth_image)
    depth_cols = len(depth_image[0]) if depth_rows else 0
    if not (0 <= point['v'] < depth_rows and 0 <= point['u'] < depth_cols):
        raise GroundingError('depth_lookup_out_of_bounds')
    depth = depth_image[point['v']][point['u']]
    pose = project_pixel_to_map(u=point['u'], v=point['v'], depth_m=depth,
                                camera_info_k=camera_info_k,
                                tf_map_camera=tf_map_camera)
    return {
        'trusted': True,
        'source_valid': True,
        'reason': 'grounded',
        'point': point,
        'map_pose': {'frame_id': 'map', 'stamp_ns': int(source_stamp_ns),
                     'x': pose['x'], 'y': pose['y'], 'z': pose['z']},
        'confidence': point['confidence'],
        'source_stamp_ns': int(source_stamp_ns),
    }


def people_entry_from_grounded_goal(*, name: str, map_pose: Mapping[str, Any],
                                    confidence: float) -> dict[str, Any]:
    """Build a deterministic people_msgs/People entry for SocialMPC consumption."""
    if map_pose.get('frame_id') != 'map':
        raise GroundingError('people_entry_requires_map_frame')
    if not (_finite(map_pose.get('x')) and _finite(map_pose.get('y'))):
        raise GroundingError('people_entry_not_finite')
    if not _finite(confidence) or not (0.0 <= float(confidence) <= 1.0):
        raise GroundingError('people_entry_invalid_confidence')
    return {
        'name': str(name),
        'position': {'x': float(map_pose['x']), 'y': float(map_pose['y']), 'z': 0.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'confidence': float(confidence),
        'source': 'lisn_grounding_live_v1',
    }
