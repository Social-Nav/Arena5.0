"""Pure LISN semantic-entity projection and cost-grid construction.

The VLM returns semantic entities in the source camera image.  This module
validates them, projects their grounding pixel with the same frozen
depth/CameraInfo/TF bundle used for the goal, and builds a bounded graded cost
grid.  No ROS, no network, no people: people remain the real HuNav tracks.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from arena_bringup.lisn_grounding import GroundingError, normalize_point, project_pixel_to_map

# Classes the VLM may label.  ``person`` is deliberately absent: people costs
# come only from the real HuNav tracks, never from a VLM label.
SEMANTIC_KNOWN_CLASSES = frozenset({
    'table', 'booth', 'chair', 'shelf', 'wall', 'obstacle', 'door', 'plant',
    'counter', 'cabinet', 'trash_bin', 'display',
})
SEMANTIC_REJECTED_CLASSES = frozenset({'person', 'people', 'unknown'})


class SemanticError(ValueError):
    """Typed fail-closed semantic-entity error."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_entity(entity: Any) -> dict[str, Any]:
    """Validate one VLM entity before projection."""
    if not isinstance(entity, Mapping):
        raise SemanticError('entity_not_object')
    class_name = entity.get('class_name')
    if not isinstance(class_name, str) or not class_name.strip():
        raise SemanticError('entity_class_missing')
    class_name = class_name.strip().lower()
    if class_name in SEMANTIC_REJECTED_CLASSES:
        raise SemanticError(f'entity_class_rejected:{class_name}')
    if class_name not in SEMANTIC_KNOWN_CLASSES:
        raise SemanticError(f'entity_class_unknown:{class_name}')
    attribute = entity.get('attribute')
    if not isinstance(attribute, str) or not attribute.strip():
        raise SemanticError('entity_attribute_missing')
    instance_id = entity.get('instance_id')
    # Compact VLM JSON often emits a numeric instance id; the host coerces it
    # to the contract's string field instead of rejecting the entity.
    if isinstance(instance_id, bool) or not isinstance(instance_id, (str, int)):
        raise SemanticError('entity_instance_id_missing')
    instance_id = str(instance_id).strip()
    if not instance_id:
        raise SemanticError('entity_instance_id_missing')
    confidence = entity.get('confidence')
    if not _finite(confidence) or not (0.0 <= float(confidence) <= 1.0):
        raise SemanticError('entity_confidence_invalid')
    cost = entity.get('cost_value')
    if not isinstance(cost, int) or isinstance(cost, bool) or not 0 <= cost <= 254:
        raise SemanticError('entity_cost_invalid')
    radius = entity.get('inflation_radius_m')
    if not _finite(radius) or not (0.0 <= float(radius) <= 5.0):
        raise SemanticError('entity_radius_invalid')
    decay = entity.get('decay_rate')
    if not _finite(decay) or not (0.0 <= float(decay) <= 10.0):
        raise SemanticError('entity_decay_invalid')
    grounding = entity.get('grounding')
    if not isinstance(grounding, Mapping):
        raise SemanticError('entity_grounding_missing')
    kind = grounding.get('kind')
    if kind not in {'pixel', 'mask', 'point'}:
        raise SemanticError(f'entity_grounding_kind_invalid:{kind}')
    pixel = grounding.get('pixel')
    if not isinstance(pixel, Mapping):
        raise SemanticError('entity_grounding_pixel_missing')
    units = grounding.get('units')
    if units not in {'pixel', 'normalized'}:
        raise SemanticError(f'entity_grounding_units_invalid:{units}')
    return {
        'class_name': class_name,
        'attribute': attribute.strip(),
        'instance_id': instance_id.strip(),
        'confidence': float(confidence),
        'cost_value': int(cost),
        'inflation_radius_m': float(radius),
        'decay_rate': float(decay),
        'grounding': {'kind': kind, 'pixel': {'x': pixel.get('x'), 'y': pixel.get('y')},
                      'units': units},
    }


def project_entity(entity: Mapping[str, Any], *, width: int, height: int,
                   depth_image: Sequence[Sequence[float]],
                   camera_info_k: Sequence[float],
                   tf_map_camera: Sequence[Sequence[float]],
                   expected_frame_id: str, source_stamp_ns: int,
                   min_confidence: float = 0.5) -> dict[str, Any]:
    """Project one validated entity's grounding pixel into the map frame."""
    grounding = entity['grounding']
    pseudo_goal = {
        'kind': 'pixel_goal',
        'frame_id': expected_frame_id,  # host-owned source frame, never VLM-supplied
        'units': grounding['units'],
        'value': {'x': grounding['pixel']['x'], 'y': grounding['pixel']['y']},
        'confidence': entity['confidence'],
    }
    try:
        point = normalize_point(pseudo_goal, width=width, height=height,
                                min_confidence=min_confidence)
        depth_rows = len(depth_image)
        depth_cols = len(depth_image[0]) if depth_rows else 0
        if not (0 <= point['v'] < depth_rows and 0 <= point['u'] < depth_cols):
            raise SemanticError('entity_depth_lookup_out_of_bounds')
        depth = depth_image[point['v']][point['u']]
        pose = project_pixel_to_map(u=point['u'], v=point['v'], depth_m=depth,
                                    camera_info_k=camera_info_k,
                                    tf_map_camera=tf_map_camera)
    except GroundingError as exc:
        raise SemanticError(f'entity_projection_failed:{exc}') from None
    return {
        **entity,
        'map_x': float(pose['x']),
        'map_y': float(pose['y']),
        'source_frame_id': expected_frame_id,
        'source_stamp_ns': int(source_stamp_ns),
        'geometry_ref': (
            f'{entity["class_name"]}:{entity["instance_id"]}:'
            f'{pose["x"]:.3f},{pose["y"]:.3f}'
        ),
    }


def contract_entity(entity: Mapping[str, Any]) -> dict[str, Any]:
    """Strip host-only fields to the hierarchical-modulation entity schema."""
    return {
        'class_name': str(entity['class_name']),
        'attribute': str(entity['attribute']),
        'instance_id': str(entity['instance_id']),
        'frame_id': 'map',
        'stamp_ns': int(entity['source_stamp_ns']),
        'confidence': float(entity['confidence']),
        'geometry_ref': str(entity['geometry_ref']),
        'cost_value': int(entity['cost_value']),
        'inflation_radius_m': float(entity['inflation_radius_m']),
        'decay_rate': float(entity['decay_rate']),
    }


def entity_cost(cost_value: int, decay_rate: float, distance_m: float) -> int:
    """Monotonic decreasing cost; 0..100 grid value (layer scales to 0..254)."""
    base = min(100.0, float(cost_value) * 100.0 / 254.0)
    value = base * math.exp(-float(decay_rate) * max(0.0, float(distance_m)))
    return int(max(0, min(100, round(value))))


def build_semantic_grid(entities: Sequence[Mapping[str, Any]], *, frame_id: str,
                        stamp_ns: int, resolution: float = 0.1,
                        max_cells: int = 400 * 400) -> dict[str, Any] | None:
    """Build a bounded graded grid in ``frame_id``; overlap uses max cost."""
    if not entities:
        return None
    if resolution <= 0.0 or not math.isfinite(resolution):
        raise SemanticError('invalid_grid_resolution')
    min_x = min(float(e['map_x']) - float(e['inflation_radius_m']) for e in entities)
    min_y = min(float(e['map_y']) - float(e['inflation_radius_m']) for e in entities)
    max_x = max(float(e['map_x']) + float(e['inflation_radius_m']) for e in entities)
    max_y = max(float(e['map_y']) + float(e['inflation_radius_m']) for e in entities)
    width = int(math.ceil((max_x - min_x) / resolution)) + 1
    height = int(math.ceil((max_y - min_y) / resolution)) + 1
    if width <= 0 or height <= 0:
        raise SemanticError('invalid_grid_size')
    if width * height > int(max_cells):
        raise SemanticError(f'grid_too_large:{width}x{height}>{max_cells}')
    data = [0] * (width * height)
    for entity in entities:
        cx, cy = float(entity['map_x']), float(entity['map_y'])
        radius = float(entity['inflation_radius_m'])
        cost_value = int(entity['cost_value'])
        decay = float(entity['decay_rate'])
        for iy in range(height):
            wy = min_y + (iy + 0.5) * resolution
            if abs(wy - cy) > radius:
                continue
            for ix in range(width):
                wx = min_x + (ix + 0.5) * resolution
                distance = math.hypot(wx - cx, wy - cy)
                if distance > radius:
                    continue
                value = entity_cost(cost_value, decay, distance)
                if value <= 0:
                    continue
                index = iy * width + ix
                data[index] = max(data[index], value)
    nonzero = sum(1 for value in data if value > 0)
    if nonzero == 0:
        return None
    return {
        'frame_id': str(frame_id),
        'stamp_ns': int(stamp_ns),
        'resolution': float(resolution),
        'width': width,
        'height': height,
        'origin': {'x': float(min_x), 'y': float(min_y)},
        'data': data,
        'nonzero_cells': int(nonzero),
        'max_cost': int(max(data)),
    }


def grid_is_fresh(stamp_ns: int, now_ns: int, ttl_ns: int) -> bool:
    """Fail-closed freshness check for the consumer TTL."""
    if int(ttl_ns) <= 0:
        return True
    return int(now_ns) - int(stamp_ns) <= int(ttl_ns)
