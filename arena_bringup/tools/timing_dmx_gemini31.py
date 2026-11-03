#!/usr/bin/env python3
"""Versioned DMX native Gemini 3.1 Pro timing entry.

Parent mode reads a key file into memory, starts one owned child with
start_new_session=True and supervises it with a 180 s global deadline.  Child
mode performs exactly one native streaming POST and writes a safe receipt.
The repo helper owns request construction, SSE parsing, proposal validation and
failure receipts.  No key is written to argv, files, stdout or a receipt.
"""
import json
import os
import re
import signal
import ssl
import subprocess
import sys
import time
import http.client
from pathlib import Path
from urllib.parse import urlparse

from arena_bringup.lisn_dmx_timing import (
    DMX_BASE,
    DMX_MODEL,
    build_request,
    failure_receipt,
    parse_proposal,
    parse_sse,
)
from arena_bringup.lisn_slow_loop_ros import LISN_PROPOSAL_PROMPT

TIMEOUT_SEC = 180.0
IMAGE_PATH = Path('/home/ubuntu/arena_jazzy_ws/tmp/arena_frame.jpg')
INSTRUCTION = (
    'Walk straight forward down the aisle. Keep the booths with the black and white '
    'patterned cushions on your left. Pass the first table and stop just past the '
    'second table on your left.'
)
REQUEST_ID_NAMES = ('x-request-id', 'x-tt-logid', 'x-dmx-request-id')
REQUEST_ID_RE = re.compile(r'^[A-Za-z0-9._:-]{1,128}$')


def _request_identifier(response) -> str:
    for name in REQUEST_ID_NAMES:
        value = response.getheader(name)
        if value and REQUEST_ID_RE.fullmatch(str(value).strip()):
            return str(value).strip()
    return 'UNKNOWN'


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def _image_and_prompt():
    image = IMAGE_PATH.read_bytes()
    prompt = f'{LISN_PROPOSAL_PROMPT}\nInstruction: {INSTRUCTION}'
    return image, prompt


def _base_diagnostics(diagnostics, model, api_host):
    return {
        'source': 'DMXAPI native Gemini',
        'api_host': api_host,
        'model': model,
        'base': diagnostics['base'],
        'path': diagnostics['path'],
        'query_key_present': diagnostics['query_key_present'],
        'auth_method': diagnostics['auth_method'],
        'mime': diagnostics['mime'],
        'generation': {
            'temperature': diagnostics['temperature'],
            'max_output_tokens': diagnostics['max_output_tokens'],
            'response_mime_type': diagnostics['response_mime_type'],
            'response_mime_type_incremental': True,
        },
        'image': {
            'path': str(IMAGE_PATH),
            'bytes': diagnostics['image_bytes'],
            'sha256': diagnostics['image_sha256'],
            'mime': diagnostics['mime'],
        },
        'prompt': {
            'version': 'lisn_proposal_prompt_v1',
            'chars': len(f'{LISN_PROPOSAL_PROMPT}\nInstruction: {INSTRUCTION}'),
        },
        'request_budget': {'max_requests': 1, 'retry': 0, 'timeout_sec': TIMEOUT_SEC},
        'controller_writes': 0,
        'gpu': False,
    }


def _child():
    secret = json.loads(sys.stdin.read())
    token = str(secret.get('api_token') or '')
    base = str(secret.get('base') or DMX_BASE)
    model = str(secret.get('model') or DMX_MODEL)
    container_name = str(secret.get('container_name') or '')
    container_id = str(secret.get('container_id') or '')
    receipt = None
    conn = None
    if not token:
        receipt = failure_receipt(
            error_code='unknown', error_category='auth', status='error',
            t0_ns=time.monotonic_ns(), t1_ns=time.monotonic_ns(),
            request_count=0, retry=0, container_name=container_name,
            container_id=container_id,
        )
        receipt['error_type'] = 'missing_api_token'
        token = ''
        receipt['secret_cleared'] = True
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    image, prompt = _image_and_prompt()
    payload, diagnostics = build_request(
        api_token=token, image_bytes=image, mime='image/jpeg', prompt=prompt,
    )
    parsed = urlparse(base)
    path = parsed.path.rstrip('/') + f'/models/{model}:streamGenerateContent'
    query = f'key={token}&alt=sse'
    t0 = time.monotonic_ns()
    try:
        if parsed.scheme == 'https':
            conn = http.client.HTTPSConnection(
                parsed.hostname, parsed.port or 443,
                timeout=TIMEOUT_SEC, context=ssl.create_default_context(),
            )
        else:
            conn = http.client.HTTPConnection(
                parsed.hostname, parsed.port or 80, timeout=TIMEOUT_SEC,
            )
        conn.request(
            'POST', path + '?' + query,
            body=json.dumps(payload, separators=(',', ':')).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        response = conn.getresponse()
        t_headers = time.monotonic_ns()
        status = int(response.status)
        if status != 200:
            body = response.read(4096)
            t1 = time.monotonic_ns()
            code = f'http_{status}'
            category = 'auth_denied' if status in (401, 403) else 'http_error'
            receipt = failure_receipt(
                error_code=code, error_category=category, status=status,
                t0_ns=t0, t_headers_ns=t_headers, t1_ns=t1,
                body_read_bytes=len(body), request_count=1, retry=0,
                container_name=container_name, container_id=container_id,
            )
            receipt['request_identifier'] = _request_identifier(response)
            receipt.update(_base_diagnostics(diagnostics, model, parsed.hostname))
            receipt['secret_cleared'] = True
            return 1
        lines = list(response)
        t1 = time.monotonic_ns()
        sse = parse_sse(lines)
        receipt = {
            'schema': 'arena.dmx.gemini31.single_request_timing.v1',
            'status': status,
            'request_count': 1,
            'retry': 0,
            't0_after_serialization': True,
            'headers_seconds': round((t_headers - t0) / 1e9, 6),
            'total_seconds': round((t1 - t0) / 1e9, 6),
            'stream_parse_seconds': round(sse['stream_parse_seconds'], 6),
            'model_version': sse['model_version'],
            'finish_reason': sse['finish_reason'],
            'first_text_chunk_index': sse['first_text_index'],
            'sse_errors': sse['errors'],
            'done': sse['done'],
            'container_name': container_name,
            'container_id': container_id,
        }
        receipt.update(_base_diagnostics(diagnostics, model, parsed.hostname))
        try:
            parse_start = time.monotonic_ns()
            _, fields, valid = parse_proposal(sse['text'])
            parse_end = time.monotonic_ns()
            receipt['six_fields_present'] = fields
            receipt['proposal_valid'] = bool(valid)
            receipt['parse_seconds'] = round((parse_end - parse_start) / 1e9, 6)
            receipt['verdict'] = 'PROPOSAL_VALID' if valid else 'PROPOSAL_INVALID'
            return 0 if valid else 1
        except Exception as exc:
            receipt['error_code'] = 'proposal_rejected'
            receipt['error_category'] = 'parse'
            receipt['verdict'] = 'PARSE_ERROR'
            receipt['parse_error_type'] = type(exc).__name__
            return 1
    except (TimeoutError, OSError) as exc:
        t1 = time.monotonic_ns()
        receipt = failure_receipt(
            error_code='timeout', error_category='network_tls', status='timeout',
            t0_ns=t0, t1_ns=t1, body_read_bytes=0, request_count=1, retry=0,
            container_name=container_name, container_id=container_id,
        )
        receipt['error_type'] = type(exc).__name__
        receipt.update(_base_diagnostics(diagnostics, model, parsed.hostname))
        return 2
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        token = ''
        if receipt is not None:
            receipt['secret_cleared'] = not bool(token)
            print(json.dumps(receipt, indent=2, sort_keys=True))
            sys.stdout.flush()


def _main():
    if '--child' in sys.argv:
        return _child()
    key_file = _arg('--key-file')
    if not key_file:
        print(json.dumps({'schema': 'arena.dmx.gemini31.entry.v1', 'error': 'missing_key_file'}))
        return 2
    token = Path(key_file).read_text(encoding='utf-8').strip()
    secret = json.dumps({
        'api_token': token,
        'base': _arg('--base', DMX_BASE),
        'model': _arg('--model', DMX_MODEL),
        'container_name': _arg('--container-name', ''),
        'container_id': _arg('--container-id', ''),
    })
    token = ''
    t_parent = time.monotonic_ns()
    proc = subprocess.Popen(
        [sys.executable, __file__, '--child'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(secret.encode('utf-8'), timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        receipt = failure_receipt(
            error_code='timeout', error_category='network_tls', status='timeout',
            t0_ns=t_parent, t1_ns=time.monotonic_ns(), body_read_bytes=0,
            request_count=1, retry=0,
            container_name=_arg('--container-name', ''),
            container_id=_arg('--container-id', ''),
        )
        receipt['worker_alive_after_cleanup'] = bool(proc.poll() is None)
        receipt['secret_cleared'] = True
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    finally:
        secret = ''
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=1.0)
    sys.stdout.buffer.write(out)
    return proc.returncode if proc.returncode is not None else 1


if __name__ == '__main__':
    raise SystemExit(_main())
