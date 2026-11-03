"""Offline tests for the Seed-Evolving Responses adapter (no network)."""
import json

import pytest

from arena_bringup.lisn_responses_adapter import (
    ARK_RESPONSES_URL,
    SEED_MODEL,
    build_responses_request,
    extract_responses_text,
    parse_responses_proposal,
    responses_failure_receipt,
)

TOKEN = 'DUMMY-SEED-TOKEN-NOT-REAL'
IMAGE = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
PROMPT = 'Return ONLY a JSON object with the LISN proposal fields.'


def test_build_responses_request_diagnostics_no_token_or_header():
    payload, diagnostics = build_responses_request(
        api_token=TOKEN, image_bytes=IMAGE, mime='image/jpeg', prompt=PROMPT,
    )
    assert diagnostics['url'] == ARK_RESPONSES_URL
    assert diagnostics['model'] == SEED_MODEL
    assert diagnostics['auth_method'] == 'bearer'
    assert diagnostics['query_key_present'] is False
    assert diagnostics['input_types'] == ['input_image', 'input_text']
    assert diagnostics['mime'] == 'image/jpeg'
    assert diagnostics['max_output_tokens'] == 4095
    text = json.dumps(diagnostics)
    assert TOKEN not in text
    assert 'Bearer' not in text
    assert 'Authorization' not in text
    assert 'key=' not in text
    content = payload['input'][0]['content']
    assert content[0]['type'] == 'input_image'
    assert content[0]['image_url'].startswith('data:image/jpeg;base64,')
    assert content[1]['type'] == 'input_text'
    assert content[1]['text'] == PROMPT


def test_build_responses_request_requires_token():
    with pytest.raises(ValueError):
        build_responses_request(api_token='', image_bytes=IMAGE, mime='image/jpeg', prompt=PROMPT)


def test_extract_responses_text_output_text_and_output_array():
    assert extract_responses_text({'output_text': 'hello'}) == 'hello'
    data = {'output': [
        {'type': 'reasoning', 'content': [{'type': 'output_text', 'text': 'think'}]},
        {'type': 'message', 'content': [
            {'type': 'output_text', 'text': '{"a":1}'},
            {'type': 'output_text', 'text': ' tail'},
        ]},
    ]}
    assert extract_responses_text(data) == '{"a":1} tail'
    assert extract_responses_text({'output': []}) == ''


def test_parse_responses_proposal_valid_and_invalid():
    valid = {
        'controller_mode': 'follow',
        'controller_parameters': {'max_linear_velocity_mps': 0.3},
        'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1,
                 'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
        'semantic_entities': [],
        'validity': {'valid': True, 'reason': ''},
        'provenance': {'model_version': 'dummy', 'prompt_version': 'v1',
                       'tool_version': 'v1', 'reasoning_latency_ms': 1},
    }
    _, fields, ok = parse_responses_proposal(json.dumps(valid))
    assert ok is True and all(fields.values())
    _, fields, ok = parse_responses_proposal('{"controller_mode":"follow"}')
    assert ok is False and not all(fields.values())
    with pytest.raises(ValueError):
        parse_responses_proposal('not-json')


@pytest.mark.parametrize('code,category,status', [
    ('http_401', 'auth_denied', 401),
    ('http_403', 'auth_denied', 403),
    ('timeout', 'network_tls', 'timeout'),
])
def test_responses_failure_receipt_safe_fields(code, category, status):
    receipt = responses_failure_receipt(
        error_code=code, error_category=category, status=status,
        t0_ns=0, t_headers_ns=500_000_000, t1_ns=1_000_000_000,
        body_read_bytes=0, request_count=1, retry=0,
        container_name='arena-arena_jazzy_ws-arena-1', container_id='deadbeef',
    )
    assert receipt['error_code'] == code
    assert receipt['error_category'] == category
    assert receipt['status'] == status
    assert receipt['request_count'] == 1
    assert receipt['retry'] == 0
    assert receipt['t0_after_serialization'] is True
    assert receipt['headers_seconds'] == 0.5
    assert receipt['elapsed_seconds'] == 1.0
    assert receipt['container_name'] == 'arena-arena_jazzy_ws-arena-1'
    assert receipt['container_id'] == 'deadbeef'
    assert receipt['url'] == ARK_RESPONSES_URL
    assert receipt['model'] == SEED_MODEL
    assert receipt['auth_method'] == 'bearer'
    assert receipt['query_key_present'] is False
    assert receipt['secret_cleared'] is True
    text = json.dumps(receipt)
    assert TOKEN not in text
    assert 'Bearer' not in text
    assert 'Authorization' not in text
    assert 'key=' not in text
