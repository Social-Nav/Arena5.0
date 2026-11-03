#!/usr/bin/env python3
"""Seed-2.1 Pro ChatCompletions single-request timing entry.

Parent reads the key file into memory and passes it to one owned child over
stdin.  The child performs exactly one non-stream POST (retry=0) to the
runtime-configured Ark ChatCompletions endpoint and writes a redacted receipt.

No token, Authorization header, keyed URL or raw response body is ever printed
or persisted.
"""
import hashlib
import http.client
import json
import os
import signal
import ssl
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from arena_bringup.lisn_seed21_adapter import (
    build_chat_request,
    chat_failure_receipt,
    extract_chat_text,
    parse_chat_proposal,
    proposal_fields,
    proposal_valid,
)
from arena_bringup.lisn_slow_loop_ros import LISN_PROPOSAL_PROMPT

DEFAULT_TIMEOUT_SEC = 60.0
CONTROL_WINDOW_SEC = 10.0
IMAGE_PATH = Path('/opt/arena_ws/tmp/arena_frame.jpg')
HOST_IMAGE_PATH = Path('/home/ubuntu/arena_jazzy_ws/tmp/arena_frame.jpg')
INSTRUCTION = (
    'Walk straight forward down the aisle. Keep the booths with the black and white '
    'patterned cushions on your left. Pass the first table and stop just past the '
    'second table on your left.'
)
PROMPT_VERSION = 'lisn_proposal_prompt_v1'
MAX_OUTPUT_TOKENS = 4095


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def _image_path():
    if IMAGE_PATH.is_file():
        return IMAGE_PATH
    return HOST_IMAGE_PATH


def _child():
    secret = json.loads(sys.stdin.read())
    token = str(secret.get('api_token') or '')
    base = str(secret.get('base') or '')
    model = str(secret.get('model') or '')
    timeout = float(secret.get('timeout_sec') or DEFAULT_TIMEOUT_SEC)
    container_name = str(secret.get('container_name') or '')
    container_id = str(secret.get('container_id') or '')
    receipt = None
    conn = None
    image_path = _image_path()
    if not token or not base or not model:
        receipt = chat_failure_receipt(
            error_code='unknown', error_category='configuration', status='error',
            t0_ns=time.monotonic_ns(), t1_ns=time.monotonic_ns(),
            request_count=0, retry=0, container_name=container_name,
            container_id=container_id, model=model,
        )
        receipt['error_type'] = 'missing_runtime_configuration'
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    image = image_path.read_bytes()
    prompt = f'{LISN_PROPOSAL_PROMPT}\nInstruction: {INSTRUCTION}'
    payload, diagnostics = build_chat_request(
        api_token=token, base=base, model=model, image_bytes=image,
        mime='image/jpeg', prompt=prompt,
    )
    body_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    parsed = urlparse(base.rstrip('/') + '/chat/completions')
    path = parsed.path or '/chat/completions'
    t0 = time.monotonic_ns()
    try:
        if parsed.scheme == 'https':
            conn = http.client.HTTPSConnection(
                parsed.hostname, parsed.port or 443, timeout=timeout,
                context=ssl.create_default_context(),
            )
        else:
            conn = http.client.HTTPConnection(
                parsed.hostname, parsed.port or 80, timeout=timeout,
            )
        conn.request(
            'POST', path, body=body_bytes,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {token}',
            },
        )
        response = conn.getresponse()
        t_headers = time.monotonic_ns()
        status = int(response.status)
        body = response.read()
        t1 = time.monotonic_ns()
        if status != 200:
            code = f'http_{status}'
            if status in (401, 403):
                category = 'auth_denied'
            elif status == 429:
                category = 'rate_limited'
            else:
                category = 'http_error'
            receipt = chat_failure_receipt(
                error_code=code, error_category=category, status=status,
                t0_ns=t0, t_headers_ns=t_headers, t1_ns=t1,
                body_read_bytes=len(body), request_count=1, retry=0,
                container_name=container_name, container_id=container_id,
                endpoint_label=diagnostics['endpoint_label'], model=model,
            )
            return 1
        parse_start = time.monotonic_ns()
        data = json.loads(body)
        text = extract_chat_text(data)
        parse_end = time.monotonic_ns()
        validate_start = time.monotonic_ns()
        try:
            _, fields, valid = parse_chat_proposal(text)
            validate_error = ''
        except Exception as exc:
            fields, valid = {}, False
            validate_error = type(exc).__name__
        validate_end = time.monotonic_ns()
        choice = (data.get('choices') or [{}])[0] if isinstance(data, dict) else {}
        usage = data.get('usage') if isinstance(data, dict) else None
        receipt = {
            'schema': 'arena.seed21.chat.sample.v1',
            'source': 'Seed-2.1 Pro ChatCompletions API',
            'endpoint_label': diagnostics['endpoint_label'],
            'path': diagnostics['path'],
            'status': status,
            'model_requested': model,
            'model_returned': str(data.get('model') or '') if isinstance(data, dict) else '',
            'finish_reason': str(choice.get('finish_reason') or '') if isinstance(choice, dict) else '',
            'usage': usage if isinstance(usage, dict) else {},
            'response_text_len': len(text),
            'response_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'six_fields_present': fields,
            'schema_valid': bool(valid),
            'request_count': 1,
            'retry': 0,
            't0_after_serialization': True,
            'headers_seconds': round((t_headers - t0) / 1e9, 6),
            'body_read_seconds': round((t1 - t_headers) / 1e9, 6),
            'total_seconds': round((t1 - t0) / 1e9, 6),
            'parse_seconds': round((parse_end - parse_start) / 1e9, 6),
            'validate_seconds': round((validate_end - validate_start) / 1e9, 6),
            'timeout_sec': timeout,
            'control_window_sec': CONTROL_WINDOW_SEC,
            'container_name': container_name,
            'container_id': container_id,
            'image': {'path': str(image_path), 'bytes': diagnostics['image_bytes'],
                      'sha256': diagnostics['image_sha256'], 'mime': diagnostics['mime']},
            'prompt': {'version': PROMPT_VERSION, 'chars': len(prompt)},
            'generation': {
                'payload_mode': 'minimal_curl_match',
                'max_tokens_present': diagnostics.get('max_tokens_present'),
                'temperature_present': diagnostics.get('temperature_present'),
            },
            'request_budget': {'max_requests': 1, 'retry': 0, 'timeout_sec': timeout},
            'token_emitted': False,
            'authorization_header_emitted': False,
            'query_key_present': False,
        }
        if validate_error:
            receipt['parse_error_type'] = validate_error
        return 0 if valid else 1
    except (TimeoutError, OSError) as exc:
        t1 = time.monotonic_ns()
        receipt = chat_failure_receipt(
            error_code='timeout', error_category='network_tls', status='timeout',
            t0_ns=t0, t1_ns=t1, body_read_bytes=0, request_count=1, retry=0,
            container_name=container_name, container_id=container_id,
            endpoint_label=diagnostics['endpoint_label'], model=model,
        )
        receipt['error_type'] = type(exc).__name__
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
    base = _arg('--base', os.environ.get('ARENA_LISN_BASE_URL', ''))
    model = _arg('--model', os.environ.get('ARENA_LISN_MODEL', ''))
    timeout = float(_arg('--timeout-sec', str(DEFAULT_TIMEOUT_SEC)))
    container_name = _arg('--container-name', '')
    container_id = _arg('--container-id', '')
    if not key_file:
        print(json.dumps({'schema': 'arena.seed21.chat.timing.v1', 'error': 'missing_key_file'}))
        return 2
    if not base or not model:
        print(json.dumps({'schema': 'arena.seed21.chat.timing.v1',
                          'error': 'missing_runtime_configuration',
                          'base_present': bool(base), 'model_present': bool(model)}))
        return 2
    token = Path(key_file).read_text(encoding='utf-8').strip()
    secret = json.dumps({
        'api_token': token, 'base': base, 'model': model, 'timeout_sec': timeout,
        'container_name': container_name, 'container_id': container_id,
    })
    token = ''
    started = time.monotonic_ns()
    proc = subprocess.Popen(
        [sys.executable, __file__, '--child'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(secret.encode('utf-8'), timeout=timeout + 5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=2.0)
        sample = chat_failure_receipt(
            error_code='timeout', error_category='network_tls', status='timeout',
            t0_ns=started, t1_ns=time.monotonic_ns(), body_read_bytes=0,
            request_count=1, retry=0, container_name=container_name,
            container_id=container_id, model=model,
        )
        sample['worker_alive_after_cleanup'] = bool(proc.poll() is None)
        sample['timeout_sec'] = timeout
        sample['control_window_sec'] = CONTROL_WINDOW_SEC
        print(json.dumps(sample, indent=2, sort_keys=True))
        return 2
    finally:
        secret = ''
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=2.0)
    try:
        sample = json.loads(out)
    except ValueError:
        sample = {'schema': 'arena.seed21.chat.sample.v1', 'status': 'error',
                  'error_code': 'unknown', 'error_category': 'parse',
                  'request_count': 1, 'retry': 0, 'secret_cleared': True}
    sample['worker_alive_after_cleanup'] = bool(proc.poll() is None)
    sample['wall_elapsed_seconds'] = round((time.monotonic_ns() - started) / 1e9, 6)
    print(json.dumps(sample, indent=2, sort_keys=True))
    return int(proc.returncode or 0)


if __name__ == '__main__':
    raise SystemExit(_main())
