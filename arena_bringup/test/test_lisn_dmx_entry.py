"""Entry-layer tests for the versioned DMX timing wrapper (loopback only)."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[1] / 'tools' / 'timing_dmx_gemini31.py'
WORKTREE = Path(__file__).resolve().parents[2]
DUMMY_KEY = 'DUMMY-KEY-NOT-REAL'
DUMMY_BODY = 'DUMMY_SECRET_BODY_NOT_REAL'


class _Handler(BaseHTTPRequestHandler):
    status = 401
    requests = 0
    query_has_key = False
    body = json.dumps({'error': DUMMY_BODY}).encode('utf-8')

    def do_POST(self):  # noqa: N802
        type(self).requests += 1
        type(self).query_has_key = 'key=' in self.path
        length = int(self.headers.get('Content-Length') or 0)
        self.rfile.read(length)
        self.send_response(type(self).status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('x-request-id', 'req-dummy-123')
        self.send_header('Content-Length', str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *args):
        return


def _server(status):
    handler = type('Handler', (_Handler,), {
        'status': status, 'requests': 0, 'query_has_key': False,
    })
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler, thread


@pytest.mark.parametrize('status,code', [(401, 'http_401'), (403, 'http_403')])
def test_entry_401_403_writes_safe_failure_receipt(tmp_path, status, code):
    server, handler, thread = _server(status)
    key_file = tmp_path / 'dummy_key'
    key_file.write_text(DUMMY_KEY)
    key_file.chmod(0o600)
    env = dict(os.environ)
    env['PYTHONPATH'] = (
        str(WORKTREE / 'arena_bringup') + ':' + env.get('PYTHONPATH', '')
    )
    try:
        result = subprocess.run(
            [
                sys.executable, str(ENTRY),
                '--key-file', str(key_file),
                '--base', f'http://127.0.0.1:{server.server_port}/v1beta',
                '--model', 'gemini-3.1-pro-preview',
                '--container-name', 'test-arena',
                '--container-id', 'test-container-id',
            ],
            capture_output=True, text=True, timeout=30, env=env,
        )
        assert result.returncode == 1, result.stderr
        receipt = json.loads(result.stdout)
        assert receipt['status'] == status
        assert receipt['error_code'] == code
        assert receipt['error_category'] == 'auth_denied'
        assert receipt['request_count'] == 1
        assert receipt['retry'] == 0
        assert receipt['t0_after_serialization'] is True
        assert receipt['headers_seconds'] is not None
        assert receipt['elapsed_seconds'] is not None
        assert receipt['body_read_bytes'] > 0
        assert receipt['container_name'] == 'test-arena'
        assert receipt['container_id'] == 'test-container-id'
        assert receipt['query_key_present'] is True
        assert receipt['auth_method'] == 'query_key'
        assert receipt['request_identifier'] == 'req-dummy-123'
        assert receipt['secret_cleared'] is True
        assert receipt['raw_body_saved'] is False
        assert receipt['key_or_url_emitted'] is False
        text = result.stdout
        assert DUMMY_KEY not in text
        assert DUMMY_BODY not in text
        assert 'key=' not in text
        assert handler.requests == 1
        assert handler.query_has_key is True
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
