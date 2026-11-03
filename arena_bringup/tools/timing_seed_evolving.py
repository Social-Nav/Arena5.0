#!/usr/bin/env python3
"""Seed-Evolving Responses API timing entry (parent/child, sequential samples).

Parent reads the key file into memory, runs up to N sequential owned children
(each 60 s hard deadline, total <=180 s) and stops on auth/closed/timeout/
invalid.  Child performs exactly one non-stream POST and writes a safe receipt.
"""
import json
import os
import re
import signal
import ssl
import statistics
import subprocess
import sys
import time
import http.client
from pathlib import Path
from urllib.parse import urlparse

from arena_bringup.lisn_responses_adapter import (
    ARK_RESPONSES_URL,
    SEED_MODEL,
    build_responses_request,
    extract_responses_text,
    parse_responses_proposal,
    responses_failure_receipt,
)
from arena_bringup.lisn_slow_loop_ros import LISN_PROPOSAL_PROMPT

PER_SAMPLE_SEC = 60.0
TOTAL_BUDGET_SEC = 180.0
IMAGE_PATH = Path('/home/ubuntu/arena_jazzy_ws/tmp/arena_frame.jpg')
INSTRUCTION = (
    'Walk straight forward down the aisle. Keep the booths with the black and white '
    'patterned cushions on your left. Pass the first table and stop just past the '
    'second table on your left.'
)
REQUEST_ID_NAMES = ('x-request-id', 'x-tt-logid', 'x-ark-request-id')
REQUEST_ID_RE = re.compile(r'^[A-Za-z0-9._:-]{1,128}$')


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def _request_identifier(response):
    for name in REQUEST_ID_NAMES:
        value = response.getheader(name)
        if value and REQUEST_ID_RE.fullmatch(str(value).strip()):
            return str(value).strip()
    return 'UNKNOWN'


def _child():
    secret = json.loads(sys.stdin.read())
    token = str(secret.get('api_token') or '')
    base = str(secret.get('base') or ARK_RESPONSES_URL)
    model = str(secret.get('model') or SEED_MODEL)
    container_name = str(secret.get('container_name') or '')
    container_id = str(secret.get('container_id') or '')
    receipt = None
    conn = None
    if not token:
        receipt = responses_failure_receipt(
            error_code='unknown', error_category='auth', status='error',
            t0_ns=time.monotonic_ns(), t1_ns=time.monotonic_ns(),
            request_count=0, retry=0, container_name=container_name,
            container_id=container_id, model=model,
        )
        receipt['error_type'] = 'missing_api_token'
        receipt['secret_cleared'] = True
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    image = IMAGE_PATH.read_bytes()
    prompt = f'{LISN_PROPOSAL_PROMPT}\nInstruction: {INSTRUCTION}'
    payload, diagnostics = build_responses_request(
        api_token=token, image_bytes=image, mime='image/jpeg', prompt=prompt,
    )
    parsed = urlparse(base)
    path = parsed.path or '/api/v3/responses'
    t0 = time.monotonic_ns()
    try:
        if parsed.scheme == 'https':
            conn = http.client.HTTPSConnection(
                parsed.hostname, parsed.port or 443,
                timeout=PER_SAMPLE_SEC, context=ssl.create_default_context(),
            )
        else:
            conn = http.client.HTTPConnection(
                parsed.hostname, parsed.port or 80, timeout=PER_SAMPLE_SEC,
            )
        conn.request(
            'POST', path,
            body=json.dumps(payload, separators=(',', ':')).encode('utf-8'),
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
            category = 'auth_denied' if status in (401, 403) else 'http_error'
            receipt = responses_failure_receipt(
                error_code=code, error_category=category, status=status,
                t0_ns=t0, t_headers_ns=t_headers, t1_ns=t1,
                body_read_bytes=len(body), request_count=1, retry=0,
                container_name=container_name, container_id=container_id,
                model=model,
            )
            receipt['request_identifier'] = _request_identifier(response)
            receipt.update({
                'image': {'path': str(IMAGE_PATH), 'bytes': diagnostics['image_bytes'],
                          'sha256': diagnostics['image_sha256'], 'mime': diagnostics['mime']},
                'prompt': {'version': 'lisn_proposal_prompt_v1', 'chars': len(prompt)},
                'generation': {'max_output_tokens': diagnostics['max_output_tokens']},
                'request_budget': {'max_requests': 1, 'retry': 0, 'timeout_sec': PER_SAMPLE_SEC},
            })
            return 1
        parse_start = time.monotonic_ns()
        data = json.loads(body)
        text = extract_responses_text(data)
        parse_end = time.monotonic_ns()
        validate_start = time.monotonic_ns()
        try:
            _, fields, valid = parse_responses_proposal(text)
            validate_error = ''
        except Exception as exc:
            fields, valid = {}, False
            validate_error = type(exc).__name__
        validate_end = time.monotonic_ns()
        usage = data.get('usage') if isinstance(data, dict) else None
        receipt = {
            'schema': 'arena.seed_evolving.sample.v1',
            'source': 'Seed-Evolving Responses API',
            'url': ARK_RESPONSES_URL,
            'status': status,
            'model': str(data.get('model') or model) if isinstance(data, dict) else model,
            'finish_reason': str(data.get('status') or '') if isinstance(data, dict) else '',
            'usage': usage if isinstance(usage, dict) else {},
            'response_text_len': len(text),
            'response_text_sha256': __import__('hashlib').sha256(text.encode('utf-8')).hexdigest(),
            'six_fields_present': fields,
            'schema_valid': bool(valid),
            'request_count': 1,
            'retry': 0,
            't0_after_serialization': True,
            'headers_seconds': round((t_headers - t0) / 1e9, 6),
            'total_seconds': round((t1 - t0) / 1e9, 6),
            'parse_seconds': round((parse_end - parse_start) / 1e9, 6),
            'validate_seconds': round((validate_end - validate_start) / 1e9, 6),
            'container_name': container_name,
            'container_id': container_id,
            'image': {'path': str(IMAGE_PATH), 'bytes': diagnostics['image_bytes'],
                      'sha256': diagnostics['image_sha256'], 'mime': diagnostics['mime']},
            'prompt': {'version': 'lisn_proposal_prompt_v1', 'chars': len(prompt)},
            'generation': {'max_output_tokens': diagnostics['max_output_tokens']},
            'request_budget': {'max_requests': 1, 'retry': 0, 'timeout_sec': PER_SAMPLE_SEC},
            'token_emitted': False,
            'authorization_header_emitted': False,
        }
        if validate_error:
            receipt['parse_error_type'] = validate_error
        return 0 if valid else 1
    except (TimeoutError, OSError) as exc:
        t1 = time.monotonic_ns()
        receipt = responses_failure_receipt(
            error_code='timeout', error_category='network_tls', status='timeout',
            t0_ns=t0, t1_ns=t1, body_read_bytes=0, request_count=1, retry=0,
            container_name=container_name, container_id=container_id, model=model,
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


def _stats(samples):
    keys = ('total_seconds', 'headers_seconds', 'parse_seconds', 'validate_seconds')
    stats = {}
    for key in keys:
        values = [float(s[key]) for s in samples if isinstance(s.get(key), (int, float))]
        if values:
            stats[key] = {
                'mean': round(statistics.mean(values), 6),
                'median': round(statistics.median(values), 6),
                'min': round(min(values), 6),
                'max': round(max(values), 6),
                'n': len(values),
            }
    return stats


def _main():
    if '--child' in sys.argv:
        return _child()
    key_file = _arg('--key-file')
    if not key_file:
        print(json.dumps({'schema': 'arena.seed_evolving.timing.v1', 'error': 'missing_key_file'}))
        return 2
    samples_requested = int(_arg('--samples', '3'))
    base = _arg('--base', ARK_RESPONSES_URL)
    model = _arg('--model', SEED_MODEL)
    container_name = _arg('--container-name', '')
    container_id = _arg('--container-id', '')
    token = Path(key_file).read_text(encoding='utf-8').strip()
    total_start = time.monotonic_ns()
    samples = []
    stopped_reason = ''
    requests_made = 0
    for _ in range(max(1, min(samples_requested, 3))):
        elapsed = (time.monotonic_ns() - total_start) / 1e9
        if elapsed + PER_SAMPLE_SEC > TOTAL_BUDGET_SEC:
            stopped_reason = 'total_budget_exhausted'
            break
        secret = json.dumps({
            'api_token': token, 'base': base, 'model': model,
            'container_name': container_name, 'container_id': container_id,
        })
        proc = subprocess.Popen(
            [sys.executable, __file__, '--child'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(secret.encode('utf-8'), timeout=PER_SAMPLE_SEC)
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
            requests_made += 1
            sample = responses_failure_receipt(
                error_code='timeout', error_category='network_tls', status='timeout',
                t0_ns=total_start, t1_ns=time.monotonic_ns(), body_read_bytes=0,
                request_count=1, retry=0, container_name=container_name,
                container_id=container_id, model=model,
            )
            sample['worker_alive_after_cleanup'] = bool(proc.poll() is None)
            sample['secret_cleared'] = True
            samples.append(sample)
            stopped_reason = 'timeout'
            break
        finally:
            secret = ''
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=1.0)
        requests_made += 1
        try:
            sample = json.loads(out)
        except ValueError:
            sample = {
                'schema': 'arena.seed_evolving.sample.v1',
                'status': 'error',
                'error_code': 'unknown',
                'error_category': 'parse',
                'request_count': 1,
                'retry': 0,
                'secret_cleared': True,
            }
        samples.append(sample)
        if sample.get('status') != 200 or not sample.get('schema_valid'):
            stopped_reason = sample.get('error_code') or 'invalid_or_incomplete'
            break
    token = ''
    summary = {
        'schema': 'arena.seed_evolving.timing.v1',
        'source': 'Seed-Evolving Responses API',
        'url': ARK_RESPONSES_URL,
        'model': model,
        'samples_requested': samples_requested,
        'requests_made': requests_made,
        'retry': 0,
        'samples': samples,
        'stats': _stats([s for s in samples if s.get('status') == 200 and s.get('schema_valid')]),
        'stopped_reason': stopped_reason,
        'total_elapsed_seconds': round((time.monotonic_ns() - total_start) / 1e9, 6),
        'api_budget_seconds': TOTAL_BUDGET_SEC,
        'per_sample_deadline_seconds': PER_SAMPLE_SEC,
        'secret_cleared': True,
        'container_name': container_name,
        'container_id': container_id,
        'controller_writes': 0,
        'gpu': False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary['stats'] else 1


if __name__ == '__main__':
    raise SystemExit(_main())
