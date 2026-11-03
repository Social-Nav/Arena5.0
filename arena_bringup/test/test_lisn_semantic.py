"""Focused pure tests for the LISN semantic projection/cost chain."""
import numpy as np
import pytest

from arena_bringup.lisn_grounding import resolve_grounded_goal
from arena_bringup.lisn_semantic import (
    SemanticError,
    build_semantic_grid,
    contract_entity,
    entity_cost,
    grid_is_fresh,
    project_entity,
    validate_entity,
)

SOURCE_FRAME = 'Ai2_Bot2/base_footprint/head_camera'
SOURCE_STAMP_NS = 36_516_668_571
WIDTH, HEIGHT = 640, 480
K = (387.8787834656402, 0.0, 320.0, 0.0, 387.8787834656402, 240.0, 0.0, 0.0, 1.0)
TF = (
    (0.016337081777036966, 0.49785697951568975, -0.86710525757057, 0.29083386022908697),
    (0.9998665409599974, -0.008139180712819494, 0.014165239356042247, -6.206130815284729),
    (-5.263108486863732e-06, -0.8672209532090642, -0.4979235064620484, 1.6460177781223873),
    (0.0, 0.0, 0.0, 1.0),
)


def _depth(value=2.0):
    return np.full((HEIGHT, WIDTH), value, dtype=np.float32)


def _entity(**overrides):
    entity = {
        'class_name': 'table',
        'attribute': 'wood',
        'instance_id': 'table-1',
        'confidence': 0.9,
        'cost_value': 200,
        'inflation_radius_m': 1.0,
        'decay_rate': 1.0,
        'grounding': {'kind': 'pixel', 'units': 'normalized',
                      'pixel': {'x': 0.55, 'y': 0.62}},
    }
    entity.update(overrides)
    return entity


def _project(entity):
    return project_entity(
        validate_entity(entity), width=WIDTH, height=HEIGHT,
        depth_image=_depth(), camera_info_k=K, tf_map_camera=TF,
        expected_frame_id=SOURCE_FRAME, source_stamp_ns=SOURCE_STAMP_NS,
    )


def test_numeric_instance_id_is_coerced_to_contract_string():
    projected = _project(_entity(instance_id=1))
    assert projected['instance_id'] == '1'
    assert contract_entity(projected)['instance_id'] == '1'


def test_known_entity_projects_with_source_frame_and_stamp():
    projected = _project(_entity())
    assert projected['source_frame_id'] == SOURCE_FRAME
    assert projected['source_stamp_ns'] == SOURCE_STAMP_NS
    assert np.isfinite(projected['map_x'])
    assert np.isfinite(projected['map_y'])
    assert projected['geometry_ref'].startswith('table:table-1:')
    contract = contract_entity(projected)
    assert contract['frame_id'] == 'map'
    assert contract['stamp_ns'] == SOURCE_STAMP_NS
    assert contract['attribute'] == 'wood'
    assert 'map_x' not in contract and 'grounding' not in contract


@pytest.mark.parametrize('class_name', ['person', 'people', 'unknown', 'car'])
def test_unknown_or_person_class_fails_closed(class_name):
    with pytest.raises(SemanticError, match='entity_class'):
        validate_entity(_entity(class_name=class_name))


@pytest.mark.parametrize('field,value,match', [
    ('cost_value', 300, 'entity_cost_invalid'),
    ('inflation_radius_m', -0.1, 'entity_radius_invalid'),
    ('decay_rate', 11.0, 'entity_decay_invalid'),
    ('confidence', 1.2, 'entity_confidence_invalid'),
])
def test_out_of_range_entity_fields_fail_closed(field, value, match):
    with pytest.raises(SemanticError, match=match):
        validate_entity(_entity(**{field: value}))


def test_missing_attribute_and_grounding_fail_closed():
    entity = _entity()
    entity.pop('attribute')
    with pytest.raises(SemanticError, match='entity_attribute_missing'):
        validate_entity(entity)
    entity = _entity()
    entity.pop('grounding')
    with pytest.raises(SemanticError, match='entity_grounding_missing'):
        validate_entity(entity)


def test_cost_is_monotonic_and_bounded():
    values = [entity_cost(200, 1.0, d) for d in (0.0, 0.5, 1.0, 2.0)]
    assert values[0] >= values[1] >= values[2] >= values[3]
    assert all(0 <= value <= 100 for value in values)
    assert entity_cost(254, 0.0, 0.0) == 100


def test_grid_overlap_uses_max_and_reports_nonzero():
    one = _project(_entity(cost_value=100, inflation_radius_m=1.0, decay_rate=0.0))
    two = dict(one)
    two['map_x'] = one['map_x'] + 0.3
    two['cost_value'] = 220
    grid = build_semantic_grid([one, two], frame_id='Ai2_Bot2/odom',
                               stamp_ns=SOURCE_STAMP_NS)
    assert grid is not None
    assert grid['nonzero_cells'] > 0
    assert 1 <= grid['max_cost'] <= 100
    assert grid['stamp_ns'] == SOURCE_STAMP_NS
    assert grid['frame_id'] == 'Ai2_Bot2/odom'


def test_no_entities_is_a_noop():
    assert build_semantic_grid([], frame_id='Ai2_Bot2/odom', stamp_ns=1) is None


def test_grid_too_large_fails_closed():
    entity = _project(_entity(inflation_radius_m=5.0))
    with pytest.raises(SemanticError, match='grid_too_large'):
        build_semantic_grid([entity], frame_id='map', stamp_ns=1, resolution=0.01,
                            max_cells=100)


def test_grid_freshness_ttl():
    assert grid_is_fresh(10, 20, 20)
    assert not grid_is_fresh(10, 40, 20)
    assert grid_is_fresh(10, 40, 0)


def test_goal_chain_is_independent_of_semantic_projection():
    goal = {
        'kind': 'pixel_goal', 'frame_id': SOURCE_FRAME, 'units': 'normalized',
        'value': {'x': 0.55, 'y': 0.62}, 'confidence': 0.9,
    }
    resolved = resolve_grounded_goal(
        goal=goal, width=WIDTH, height=HEIGHT, depth_image=_depth(),
        camera_info_k=K, tf_map_camera=TF, expected_frame_id=SOURCE_FRAME,
        source_stamp_ns=SOURCE_STAMP_NS,
    )
    assert resolved['map_pose']['frame_id'] == 'map'
    assert resolved['map_pose']['stamp_ns'] == SOURCE_STAMP_NS
    projected = _project(_entity())
    assert projected['source_stamp_ns'] == resolved['map_pose']['stamp_ns']
