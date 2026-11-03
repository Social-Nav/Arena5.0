"""Entry-layer tests for the Seed-Evolving Responses timing wrapper (loopback)."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[1] / 'tools' / 'timing_seed_evolving.py'
WORKTREE = Path(__file__).resolve().parents[2]
DUMMY_KEY = 'DUMMY-SEED-KEY-NOT-REAL'
DUMMY_BODY = 'DUMMY_SEED_SECRET_BODY_NOT_REAL'


class _Handler(BaseHTTPRequestHandler):
    status = 401
    requests = 0
    authorization_present = False
    body_received = 0
    body = json.dumps({'error': DUMMY_BODY}).encode('utf-8')

    def do_POST(self):  # noqa: N802
        type(self).requests += 1
        type(self).authorization_present = bool(self.headers.get('Authorization'))
        length = int(self.headers.get('Content-Length') or 0)
        payload = self.rfile.read(length)
        type(self).body_received = len(payload)
        self.send_response(type(self).status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('x-request-id', 'req-seed-123')
        self.send_header('Content-Length', str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *args):
        return


def _server(status):
    handler = type('Handler', (_Handler,), {
        'status': status, 'requests': 0, 'authorization_present': False,
        'body_received': 0,
    })
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler, thread


@pytest.mark.parametrize('status,code', [(401, 'http_401'), (403, 'http_403')])
def test_seed_entry_401_403_safe_sample(tmp_path, status, code):
    server, handler, thread = _server(status)
    key_file = tmp_path / 'dummy_key'
    key_file.write_text(DUMMY_KEY)
    key_file.chmod(0o600)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(WORKTREE / 'arena_bringup') + ':' + env.get('PYTHONPATH', '')
    try:
        result = subprocess.run(
            [
                sys.executable, str(ENTRY),
                '--key-file', str(key_file),
                '--samples', '1',
                '--base', f'http://127.0.0.1:{server.server_port}/api/v3/responses',
                '--model', 'ep-20261026170644-bvqsc',
                '--container-name', 'test-arena',
                '--container-id', 'test-container-id',
            ],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 1, result.stderr
        summary = json.loads(result.stdout)
        assert summary['requests_made'] == 1
        assert summary['retry'] == 0
        assert summary['stopped_reason'] == code
        assert summary['secret_cleared'] is True
        assert summary['stats'] == {}
        sample = summary['samples'][0]
        assert sample['status'] == status
        assert sample['error_code'] == code
        assert sample['error_category'] == 'auth_denied'
        assert sample['request_count'] == 1
        assert sample['retry'] == 0
        assert sample['t0_after_serialization'] is True
        assert sample['headers_seconds'] is not None
        assert sample['elapsed_seconds'] is not None
        assert sample['body_read_bytes'] > 0
        assert sample['container_name'] == 'test-arena'
        assert sample['container_id'] == 'test-container-id'
        assert sample['query_key_present'] is False
        assert sample['auth_method'] == 'bearer'
        assert sample['request_identifier'] == 'req-seed-123'
        assert sample['secret_cleared'] is True
        assert sample['token_emitted'] is False
        assert sample['authorization_header_emitted'] is False
        text = result.stdout
        assert DUMMY_KEY not in text
        assert DUMMY_BODY not in text
        assert 'Bearer' not in text
        assert 'Authorization' not in text
        assert 'key=' not in text
        assert handler.requests == 1
        assert handler.authorization_present is True
        assert handler.body_received > 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
