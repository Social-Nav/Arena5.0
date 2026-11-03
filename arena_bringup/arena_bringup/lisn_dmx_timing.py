"""Pure DMX native Gemini request/response helpers for offline validation.

No network, no credential printing.  The caller keeps the api_token in memory
and builds the URL; these helpers return only diagnostics that are safe to
persist (query-key presence boolean, auth method, mime, model, base and
generation fields).
"""
import base64
import hashlib
import json
import time
from typing import Any, Iterable, Mapping

DMX_BASE = 'https://jkwl.dmxapi.cn/v1beta'
DMX_MODEL = 'gemini-3.1-pro-preview'
MAX_OUTPUT_TOKENS = 4095
RESPONSE_MIME_TYPE = 'application/json'
REQUIRED_FIELDS = (
    'controller_mode', 'controller_parameters', 'goal',
    'semantic_entities', 'validity', 'provenance',
)
FAILURE_CODES = frozenset({
    'http_401', 'http_403', 'timeout', 'transport',
    'sse_json', 'proposal_rejected', 'unknown',
})


def build_request(*, api_token: str, image_bytes: bytes, mime: str, prompt: str,
                  max_output_tokens: int = MAX_OUTPUT_TOKENS,
                  response_mime_type: str = RESPONSE_MIME_TYPE):
    """Build the native Gemini payload and safe diagnostics.

    The returned diagnostics never contain the token or a keyed URL.  The
    caller builds ``{base}/models/{model}:streamGenerateContent?key=<secret>
    &alt=sse`` in memory.
    """
    if not api_token:
        raise ValueError('missing_api_token')
    encoded = base64.b64encode(bytes(image_bytes)).decode('ascii')
    generation = {
        'temperature': 0,
        'maxOutputTokens': int(max_output_tokens),
    }
    if response_mime_type:
        generation['responseMimeType'] = str(response_mime_type)
    payload = {
        'contents': [{
            'parts': [
                {'inline_data': {'mime_type': str(mime), 'data': encoded}},
                {'text': str(prompt)},
            ],
        }],
        'generationConfig': generation,
    }
    diagnostics = {
        'base': DMX_BASE,
        'model': DMX_MODEL,
        'path': f'/v1beta/models/{DMX_MODEL}:streamGenerateContent',
        'query_key_present': True,
        'auth_method': 'query_key',
        'mime': str(mime),
        'image_bytes': len(bytes(image_bytes)),
        'image_sha256': hashlib.sha256(bytes(image_bytes)).hexdigest(),
        'max_output_tokens': int(max_output_tokens),
        'response_mime_type': str(response_mime_type or ''),
        'temperature': 0,
        'token_emitted': False,
        'keyed_url_emitted': False,
    }
    return payload, diagnostics


def failure_receipt(*, error_code: str, error_category: str, status,
                    t0_ns: int, t_headers_ns: int | None = None,
                    t1_ns: int | None = None, body_read_bytes: int = 0,
                    request_count: int = 1, retry: int = 0,
                    container_name: str = '', container_id: str = '') -> dict:
    """Build a safe failure receipt with timing and Docker identity.

    The elapsed is measured from ``t0_ns`` recorded after payload
    serialization.  The receipt never contains the key, a keyed URL, headers
    or a raw body.
    """
    code = str(error_code) if str(error_code) in FAILURE_CODES else 'unknown'
    end = int(t1_ns) if t1_ns is not None else time.monotonic_ns()
    headers = None
    if t_headers_ns is not None:
        headers = round((int(t_headers_ns) - int(t0_ns)) / 1e9, 6)
    return {
        'schema': 'arena.dmx.gemini31.failure.v1',
        'error_code': code,
        'error_category': str(error_category),
        'status': status,
        't0_after_serialization': True,
        'headers_seconds': headers,
        'elapsed_seconds': round((end - int(t0_ns)) / 1e9, 6),
        'body_read_bytes': int(body_read_bytes),
        'request_count': int(request_count),
        'retry': int(retry),
        'container_name': str(container_name),
        'container_id': str(container_id),
        'secret_cleared': True,
        'raw_body_saved': False,
        'key_or_url_emitted': False,
    }


def parse_sse(lines: Iterable) -> dict:
    """Parse SSE data lines; filter thought parts and collect finish/schema."""
    texts = []
    model_version = ''
    finish_reason = ''
    first_index = None
    errors = 0
    done = False
    parse_ns = 0
    for index, raw in enumerate(lines):
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', 'replace')
        line = str(raw).strip()
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if payload == '[DONE]':
            done = True
            break
        start = time.monotonic_ns()
        try:
            chunk = json.loads(payload)
        except ValueError:
            errors += 1
            parse_ns += time.monotonic_ns() - start
            continue
        parse_ns += time.monotonic_ns() - start
        if not isinstance(chunk, dict):
            continue
        model_version = str(chunk.get('modelVersion') or chunk.get('model') or model_version)
        candidates = chunk.get('candidates')
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            finish_reason = str(candidate.get('finishReason') or finish_reason)
            content = candidate.get('content')
            parts = content.get('parts') if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict) or part.get('thought') is True:
                    continue
                text = part.get('text')
                if isinstance(text, str) and text:
                    texts.append(text)
                    if first_index is None:
                        first_index = index
    return {
        'text': ''.join(texts),
        'model_version': model_version,
        'finish_reason': finish_reason,
        'first_text_index': first_index,
        'errors': errors,
        'done': done,
        'stream_parse_seconds': parse_ns / 1e9,
    }


def parse_proposal(text: str):
    """Parse a proposal JSON and validate the six required fields."""
    cleaned = str(text or '').strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        if cleaned.startswith('json'):
            cleaned = cleaned[4:].lstrip()
    data = json.loads(cleaned)
    fields = {name: name in data for name in REQUIRED_FIELDS}
    parameters = data.get('controller_parameters')
    valid = bool(all(fields.values()) and isinstance(parameters, dict)
                 and 'max_linear_velocity_mps' in parameters)
    return data, fields, valid
