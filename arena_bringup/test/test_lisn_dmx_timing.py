"""Offline tests for the DMX native Gemini timing helpers (no network)."""
import json

import pytest

from arena_bringup.lisn_dmx_timing import (
    DMX_BASE,
    DMX_MODEL,
    build_request,
    failure_receipt,
    parse_proposal,
    parse_sse,
)

TOKEN = 'DUMMY-DMX-TOKEN-NOT-REAL'
IMAGE = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
PROMPT = 'Return ONLY a JSON object with the LISN proposal fields.'


def test_build_request_diagnostics_have_no_token_or_keyed_url():
    payload, diagnostics = build_request(
        api_token=TOKEN, image_bytes=IMAGE, mime='image/jpeg', prompt=PROMPT,
    )
    assert diagnostics['base'] == DMX_BASE
    assert diagnostics['model'] == DMX_MODEL
    assert diagnostics['path'] == f'/v1beta/models/{DMX_MODEL}:streamGenerateContent'
    assert diagnostics['query_key_present'] is True
    assert diagnostics['auth_method'] == 'query_key'
    assert diagnostics['mime'] == 'image/jpeg'
    assert diagnostics['max_output_tokens'] == 4095
    assert diagnostics['response_mime_type'] == 'application/json'
    assert diagnostics['temperature'] == 0
    text = json.dumps(diagnostics)
    assert TOKEN not in text
    assert 'key=' not in text
    assert 'Bearer' not in text
    part = payload['contents'][0]['parts'][0]['inline_data']
    assert part['mime_type'] == 'image/jpeg'
    assert part['data']
    assert payload['contents'][0]['parts'][1]['text'] == PROMPT
    assert payload['generationConfig']['responseMimeType'] == 'application/json'


def test_build_request_requires_token():
    with pytest.raises(ValueError):
        build_request(api_token='', image_bytes=IMAGE, mime='image/jpeg', prompt=PROMPT)


@pytest.mark.parametrize('code,category,status', [
    ('http_401', 'auth_denied', 401),
    ('http_403', 'auth_denied', 403),
    ('timeout', 'network_tls', 'timeout'),
])
def test_failure_receipt_has_timing_identity_and_no_secret(code, category, status):
    receipt = failure_receipt(
        error_code=code, error_category=category, status=status,
        t0_ns=0, t_headers_ns=500_000_000, t1_ns=1_000_000_000,
        body_read_bytes=0, request_count=1, retry=0,
        container_name='arena-arena_jazzy_ws-arena-1', container_id='deadbeef',
    )
    assert receipt['error_code'] == code
    assert receipt['error_category'] == category
    assert receipt['status'] == status
    assert receipt['t0_after_serialization'] is True
    assert receipt['headers_seconds'] == 0.5
    assert receipt['elapsed_seconds'] == 1.0
    assert receipt['body_read_bytes'] == 0
    assert receipt['request_count'] == 1
    assert receipt['retry'] == 0
    assert receipt['container_name'] == 'arena-arena_jazzy_ws-arena-1'
    assert receipt['container_id'] == 'deadbeef'
    assert receipt['secret_cleared'] is True
    text = json.dumps(receipt)
    assert TOKEN not in text
    assert 'key=' not in text
    assert 'raw_body_content' not in text


def test_failure_receipt_unknown_code_is_normalized():
    receipt = failure_receipt(
        error_code='DUMMY_SECRET_CODE', error_category='unknown', status=500,
        t0_ns=0, t1_ns=1_000_000,
    )
    assert receipt['error_code'] == 'unknown'
    assert 'DUMMY_SECRET_CODE' not in json.dumps(receipt)


def _sse_chunks():
    proposal = {
        'controller_mode': 'follow',
        'controller_parameters': {'max_linear_velocity_mps': 0.3},
        'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1,
                 'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
        'semantic_entities': [],
        'validity': {'valid': True, 'reason': ''},
        'provenance': {'model_version': 'dummy', 'prompt_version': 'v1',
                       'tool_version': 'v1', 'reasoning_latency_ms': 1},
    }
    text = json.dumps(proposal)
    third = len(text) // 3
    return [
        {'modelVersion': DMX_MODEL, 'candidates': [{'content': {'parts': [
            {'thought': True, 'text': 'thinking'}]}}]},
        {'candidates': [{'content': {'parts': [{'text': text[:third]}]}}]},
        {'candidates': [{'content': {'parts': [{'text': text[third:2 * third]}]}}]},
        {'candidates': [{'content': {'parts': [{'text': text[2 * third:]}]},
                         'finishReason': 'STOP'}]},
        '[DONE]',
    ]


def test_parse_sse_filters_thought_multi_chunk_done_finish_and_schema():
    lines = [f'data: {json.dumps(c) if not isinstance(c, str) else c}' for c in _sse_chunks()]
    parsed = parse_sse(lines)
    assert parsed['model_version'] == DMX_MODEL
    assert parsed['finish_reason'] == 'STOP'
    assert parsed['first_text_index'] == 1
    assert parsed['errors'] == 0
    assert parsed['done'] is True
    assert 'thinking' not in parsed['text']
    data, fields, valid = parse_proposal(parsed['text'])
    assert valid is True
    assert all(fields.values())


def test_parse_sse_counts_bad_chunk_without_raw_body():
    parsed = parse_sse(['data: not-json', 'data: [DONE]'])
    assert parsed['errors'] == 1
    assert parsed['done'] is True
    assert parsed['text'] == ''
    assert 'not-json' not in json.dumps(parsed)


def test_parse_proposal_invalid_json_and_missing_fields():
    with pytest.raises(ValueError):
        parse_proposal('not-json')
    _, fields, valid = parse_proposal('{"controller_mode":"follow"}')
    assert valid is False
    assert not all(fields.values())
