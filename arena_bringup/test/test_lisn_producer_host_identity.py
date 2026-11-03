"""Focused CPU tests for host-owned identity/provenance and goal fail-closed."""
import pytest

from arena_bringup.lisn_slow_loop_ros import (
    LISN_GROUNDING_PROMPT,
    LISN_PROPOSAL_PROMPT,
    ProducerConfig,
    ProducerUnavailableError,
    apply_host_provenance,
    assess_goal_trust,
    build_snapshot,
    validate_model_identity,
)


def _proposal(*, goal, valid=True, goal_trust=None, provenance=None):
    doc = {
        'controller_mode': 'goal',
        'controller_parameters': {'max_linear_velocity_mps': 0.3},
        'goal': goal,
        'semantic_entities': [],
        'validity': {'valid': valid, 'reason': 'test'},
        'provenance': provenance or {
            'model_version': 'vlm-forged',
            'prompt_version': 'vlm-forged',
            'tool_version': 'vlm-forged',
            'reasoning_latency_ms': -1,
        },
    }
    if goal_trust is not None:
        doc['goal_trust'] = goal_trust
    return doc


def test_vlm_provenance_is_overwritten_by_host():
    proposal = _proposal(goal={'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1,
                               'value': {'x': 0.0, 'y': 0.0, 'yaw': 0.0}})
    out = apply_host_provenance(
        proposal, model='Qwen/Qwen3-VL-8B-Instruct-FP8',
        model_revision='9cdc6310', model_image_id='img123',
        prompt_version='prompt-sha', tool_version='tool-v1', latency_ms=1234.5,
    )
    assert out['provenance'] == {
        'model_version': 'Qwen/Qwen3-VL-8B-Instruct-FP8@9cdc6310;image=img123',
        'prompt_version': 'prompt-sha',
        'tool_version': 'tool-v1',
        'reasoning_latency_ms': 1234.5,
    }


def test_grounding_prompt_has_no_provenance_and_keeps_old_prompt():
    assert 'provenance {' not in LISN_GROUNDING_PROMPT
    assert 'model_version' not in LISN_GROUNDING_PROMPT
    assert 'units' in LISN_GROUNDING_PROMPT
    assert 'confidence' in LISN_GROUNDING_PROMPT
    assert 'provenance {' in LISN_PROPOSAL_PROMPT


def test_goal_untrusted_without_projection_or_semantic_match():
    goal = {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1,
            'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}}
    trust = assess_goal_trust(goal, target_frame='map')
    assert trust['trusted'] is False
    assert trust['source_invalid'] is True
    assert 'no_known_camera_to_map_projection' in trust['reason']
    trust_proj = assess_goal_trust(goal, target_frame='map', known_projection=True)
    assert trust_proj['trusted'] is True
    trust_sem = assess_goal_trust(goal, target_frame='map', semantic_match=True)
    assert trust_sem['trusted'] is True


def test_free_form_goal_kind_is_untrusted():
    goal = {'kind': 'pose', 'frame_id': 'map', 'stamp_ns': 1742345678901234567,
            'value': {'x': 2.1, 'y': -1.8, 'yaw': 0.0}}
    trust = assess_goal_trust(goal, target_frame='map', known_projection=True)
    assert trust['trusted'] is False
    assert 'unsupported_goal_kind' in trust['reason']


def test_model_alias_and_identity_mismatch_rejected():
    alias = validate_model_identity(request_model='gpt-4o',
                                    served_model='Qwen/Qwen3-VL-8B-Instruct-FP8',
                                    response_model='Qwen/Qwen3-VL-8B-Instruct-FP8')
    assert alias['ok'] is False
    assert alias['reason'] == 'model_alias_forbidden'
    mismatch = validate_model_identity(
        request_model='Qwen/Qwen3-VL-8B-Instruct-FP8',
        served_model='Qwen/Qwen3-VL-8B-Instruct-FP8',
        response_model='gpt-4o')
    assert mismatch['ok'] is False
    assert mismatch['reason'] == 'model_identity_mismatch'
    match = validate_model_identity(
        request_model='Qwen/Qwen3-VL-8B-Instruct-FP8',
        served_model='Qwen/Qwen3-VL-8B-Instruct-FP8',
        response_model='Qwen/Qwen3-VL-8B-Instruct-FP8')
    assert match['ok'] is True


def test_build_snapshot_rejects_untrusted_goal():
    proposal = _proposal(
        goal={'kind': 'pose', 'frame_id': 'map', 'stamp_ns': 7,
              'value': {'x': 2.1, 'y': -1.8, 'yaw': 0.0}},
        valid=False,
        goal_trust={'trusted': False, 'source_invalid': True,
                    'reason': 'no_known_camera_to_map_projection_or_semantic_match'},
    )
    with pytest.raises(ProducerUnavailableError) as exc:
        build_snapshot(
            current_parameters={'max_linear_velocity_mps': 0.3,
                                'max_angular_velocity_rps': 0.5,
                                'goal_weight': 0.8, 'obstacle_weight': 0.2,
                                'people_weight': 0.1},
            proposal=proposal, observation_id='obs-1', frame_id='cam',
            stamp_ns=999, target_frame='map', now_ns=1000, max_age_ns=500,
        )
    assert exc.value.code == 'goal_untrusted'


def test_build_snapshot_uses_host_stamp_not_vlm_stamp():
    proposal = _proposal(
        goal={'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1742345678901234567,
              'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
        goal_trust={'trusted': True, 'source_invalid': False, 'reason': 'trusted'},
    )
    snap = build_snapshot(
        current_parameters={'max_linear_velocity_mps': 0.3,
                            'max_angular_velocity_rps': 0.5,
                            'goal_weight': 0.8, 'obstacle_weight': 0.2,
                            'people_weight': 0.1},
        proposal=proposal, observation_id='obs-1', frame_id='cam',
        stamp_ns=999, target_frame='map', now_ns=1000, max_age_ns=500,
    )
    assert snap['goal']['stamp_ns'] == 999
    assert snap['source_observation']['stamp_ns'] == 999
    assert snap['generated_at_ns'] == 1000
    assert snap['valid_until_ns'] == 1500
    assert snap['provenance']['model_version'] == 'vlm-forged'
