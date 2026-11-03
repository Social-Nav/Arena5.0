"""Dummy-only tests for the Seed-2.1 ChatCompletions thin adapter and entry."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from arena_bringup.lisn_seed21_adapter import (
    build_chat_request,
    chat_failure_receipt,
    endpoint_from_base,
    extract_chat_text,
    parse_chat_proposal,
)

DUMMY_KEY = 'dummy-seed21-not-a-real-key'
BASE = 'https://ark.example.test/api/v3'
MODEL = 'ep-dummy-000'
PROPOSAL = {
    'controller_mode': 'cautious',
    'controller_parameters': {'max_linear_velocity_mps': 0.30},
    'goal': {'kind': 'map_pose', 'frame_id': 'map', 'stamp_ns': 1_400_000_000,
             'value': {'x': 1.0, 'y': 2.0, 'yaw': 0.0}},
    'semantic_entities': [],
    'validity': {'valid': True, 'reason': 'ok'},
    'provenance': {'model_version': 'dummy', 'prompt_version': 'p',
                   'tool_version': 't', 'reasoning_latency_ms': 1.0},
}


def test_endpoint_from_base_appends_chat_completions():
    assert endpoint_from_base(BASE) == BASE + '/chat/completions'
    assert endpoint_from_base(BASE + '/') == BASE + '/chat/completions'
    assert endpoint_from_base(BASE + '/chat/completions') == BASE + '/chat/completions'


def test_endpoint_requires_base():
    with pytest.raises(ValueError, match='missing_base_url'):
        endpoint_from_base('')


def test_build_request_image_url_and_no_secret_in_diagnostics():
    payload, diagnostics = build_chat_request(
        api_token=DUMMY_KEY, base=BASE, model=MODEL,
        image_bytes=b'\xff\xd8jpeg', mime='image/jpeg', prompt='go',
    )
    content = payload['messages'][0]['content']
    assert content[0]['type'] == 'image_url'
    assert content[0]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert content[1] == {'type': 'text', 'text': 'go'}
    assert payload['model'] == MODEL
    # Minimal payload matches the successful documented curl: no max_tokens,
    # no temperature, no stream, no thinking.
    assert set(payload) == {'model', 'messages'}
    assert 'max_tokens' not in payload
    assert 'max_completion_tokens' not in payload
    assert 'temperature' not in payload
    assert 'stream' not in payload
    assert 'thinking' not in payload
    dumped = json.dumps(diagnostics)
    assert DUMMY_KEY not in dumped
    assert 'Authorization' not in dumped
    assert diagnostics['query_key_present'] is False
    assert diagnostics['max_tokens_present'] is False
    assert diagnostics['temperature_present'] is False


def test_build_request_explicit_generation_fields_are_opt_in():
    payload, diagnostics = build_chat_request(
        api_token=DUMMY_KEY, base=BASE, model=MODEL,
        image_bytes=b'\xff\xd8jpeg', mime='image/jpeg', prompt='go',
        max_output_tokens=512, temperature=0.0,
    )
    assert payload['max_tokens'] == 512
    assert payload['temperature'] == 0.0
    assert diagnostics['max_tokens_present'] is True
    assert diagnostics['temperature_present'] is True


def test_build_request_requires_token_and_model():
    with pytest.raises(ValueError, match='missing_api_token'):
        build_chat_request(api_token='', base=BASE, model=MODEL,
                           image_bytes=b'x', mime='image/jpeg', prompt='go')
    with pytest.raises(ValueError, match='missing_model'):
        build_chat_request(api_token=DUMMY_KEY, base=BASE, model='',
                           image_bytes=b'x', mime='image/jpeg', prompt='go')


def test_extract_and_parse_chat_proposal():
    body = {'choices': [{'message': {'content': json.dumps(PROPOSAL)}}]}
    text = extract_chat_text(body)
    data, fields, valid = parse_chat_proposal(text)
    assert data['controller_mode'] == 'cautious'
    assert all(fields.values()) and valid


def test_extract_chat_text_handles_parts_and_empty():
    assert extract_chat_text({'choices': [{'message': {'content': [
        {'type': 'text', 'text': '{"a":1}'}]}}]}) == '{"a":1}'
    assert extract_chat_text({}) == ''
    assert extract_chat_text({'choices': []}) == ''


def test_failure_receipt_whitelists_and_redacts():
    receipt = chat_failure_receipt(
        error_code=DUMMY_KEY, error_category='auth_denied', status=401,
        t0_ns=1_000_000_000, t_headers_ns=1_500_000_000, t1_ns=2_000_000_000,
        endpoint_label='ark.example.test', model=MODEL,
    )
    assert receipt['error_code'] == 'unknown'
    assert receipt['headers_seconds'] == 0.5
    assert receipt['elapsed_seconds'] == 1.0
    assert DUMMY_KEY not in json.dumps(receipt)
    assert receipt['key_or_url_emitted'] is False


class _Handler(BaseHTTPRequestHandler):
    status = 200
    body = b'{}'
    delay = 0.0
    seen = []

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length)
        type(self).seen.append({
            'path': self.path,
            'authorization': self.headers.get('Authorization'),
            'body': raw,
        })
        if self.delay:
            time.sleep(self.delay)
        self.send_response(self.status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        return


@pytest.fixture()
def server():
    _Handler.status = 200
    _Handler.body = json.dumps({'choices': [{'message': {'content': json.dumps(PROPOSAL)}}]}).encode()
    _Handler.delay = 0.0
    _Handler.seen = []
    httpd = HTTPServer(('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, f'http://127.0.0.1:{httpd.server_port}/api/v3'
    finally:
        httpd.shutdown()
        httpd.server_close()


def _run_entry(tmp_path, base, timeout_sec=5):
    import subprocess
    import sys
    from pathlib import Path
    key_file = tmp_path / 'api_key'
    key_file.write_text(DUMMY_KEY + '\n', encoding='utf-8')
    entry = Path(__file__).resolve().parents[1] / 'tools' / 'timing_seed21.py'
    return subprocess.run(
        [sys.executable, str(entry),
         '--key-file', str(key_file), '--base', base, '--model', MODEL,
         '--timeout-sec', str(timeout_sec)],
        capture_output=True, text=True, timeout=30,
    )


def test_entry_success_is_one_request_and_redacted(tmp_path, server):
    httpd, base = server
    proc = _run_entry(tmp_path, base)
    receipt = json.loads(proc.stdout)
    assert receipt['status'] == 200
    assert receipt['schema_valid'] is True
    assert receipt['request_count'] == 1 and receipt['retry'] == 0
    assert receipt['model_requested'] == MODEL
    assert len(_Handler.seen) == 1
    assert _Handler.seen[0]['authorization'] == f'Bearer {DUMMY_KEY}'
    assert DUMMY_KEY not in proc.stdout
    assert 'Authorization' not in proc.stdout
    assert receipt['secret_cleared'] is True


@pytest.mark.parametrize('status', [401, 403])
def test_entry_auth_failure_one_request_no_retry(tmp_path, server, status):
    _Handler.status = status
    _Handler.body = b'{"error":"denied"}'
    _, base = server
    proc = _run_entry(tmp_path, base)
    receipt = json.loads(proc.stdout)
    assert receipt['error_code'] == f'http_{status}'
    assert receipt['error_category'] == 'auth_denied'
    assert receipt['request_count'] == 1 and receipt['retry'] == 0
    assert len(_Handler.seen) == 1
    assert DUMMY_KEY not in proc.stdout
    assert '"error":"denied"' not in proc.stdout


def test_entry_rate_limit_one_request_no_retry(tmp_path, server):
    _Handler.status = 429
    _Handler.body = b'{"error":"rate"}'
    _, base = server
    proc = _run_entry(tmp_path, base)
    receipt = json.loads(proc.stdout)
    assert receipt['error_code'] == 'http_429'
    assert receipt['error_category'] == 'rate_limited'
    assert receipt['request_count'] == 1 and receipt['retry'] == 0
    assert len(_Handler.seen) == 1
    assert DUMMY_KEY not in proc.stdout
    assert '"error":"rate"' not in proc.stdout


def test_entry_timeout_records_lower_bound(tmp_path, server):
    _Handler.delay = 2.0
    _, base = server
    proc = _run_entry(tmp_path, base, timeout_sec=0.5)
    receipt = json.loads(proc.stdout)
    assert receipt['error_code'] == 'timeout'
    assert receipt['request_count'] == 1 and receipt['retry'] == 0
    assert len(_Handler.seen) == 1
    assert receipt['worker_alive_after_cleanup'] is False
    assert DUMMY_KEY not in proc.stdout
