"""Thin Seed-Evolving Responses API adapter for LISN proposals (pure).

The caller keeps the Bearer token in memory; diagnostics never contain the
token or an Authorization header.  This module builds the Responses payload,
extracts output text, validates the six-field proposal and reuses the safe
failure-receipt helper.
"""
import base64
import hashlib
from typing import Any, Mapping

from arena_bringup.lisn_dmx_timing import failure_receipt, parse_proposal

ARK_RESPONSES_URL = 'https://ark-cn-beijing.bytedance.net/api/v3/responses'
SEED_MODEL = 'ep-20261026170644-bvqsc'
MAX_OUTPUT_TOKENS = 4095


def build_responses_request(*, api_token: str, image_bytes: bytes, mime: str,
                            prompt: str, max_output_tokens: int = MAX_OUTPUT_TOKENS):
    """Build the Responses payload and safe diagnostics (no token/header)."""
    if not api_token:
        raise ValueError('missing_api_token')
    encoded = base64.b64encode(bytes(image_bytes)).decode('ascii')
    payload = {
        'model': SEED_MODEL,
        'input': [{
            'role': 'user',
            'content': [
                {'type': 'input_image', 'image_url': f'data:{mime};base64,{encoded}'},
                {'type': 'input_text', 'text': str(prompt)},
            ],
        }],
        'max_output_tokens': int(max_output_tokens),
    }
    diagnostics = {
        'url': ARK_RESPONSES_URL,
        'model': SEED_MODEL,
        'auth_method': 'bearer',
        'query_key_present': False,
        'mime': str(mime),
        'image_bytes': len(bytes(image_bytes)),
        'image_sha256': hashlib.sha256(bytes(image_bytes)).hexdigest(),
        'max_output_tokens': int(max_output_tokens),
        'input_types': ['input_image', 'input_text'],
        'token_emitted': False,
        'authorization_header_emitted': False,
    }
    return payload, diagnostics


def extract_responses_text(data: Any) -> str:
    """Extract output text; ignore reasoning items and non-text parts."""
    if not isinstance(data, Mapping):
        return ''
    direct = data.get('output_text')
    if isinstance(direct, str) and direct:
        return direct
    output = data.get('output')
    if not isinstance(output, list):
        return ''
    texts = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        if str(item.get('type') or '') in {'reasoning', 'reasoning_summary'}:
            continue
        content = item.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if str(part.get('type') or '') not in {'output_text', 'text'}:
                continue
            text = part.get('text')
            if isinstance(text, str) and text:
                texts.append(text)
    return ''.join(texts)


def parse_responses_proposal(text: str):
    return parse_proposal(text)


def responses_failure_receipt(*, error_code: str, error_category: str, status,
                              t0_ns: int, t_headers_ns: int | None = None,
                              t1_ns: int | None = None, body_read_bytes: int = 0,
                              request_count: int = 1, retry: int = 0,
                              container_name: str = '', container_id: str = '',
                              model: str = SEED_MODEL) -> dict:
    receipt = failure_receipt(
        error_code=error_code, error_category=error_category, status=status,
        t0_ns=t0_ns, t_headers_ns=t_headers_ns, t1_ns=t1_ns,
        body_read_bytes=body_read_bytes, request_count=request_count,
        retry=retry, container_name=container_name, container_id=container_id,
    )
    receipt.update({
        'source': 'Seed-Evolving Responses API',
        'url': ARK_RESPONSES_URL,
        'model': model,
        'auth_method': 'bearer',
        'query_key_present': False,
        'token_emitted': False,
        'authorization_header_emitted': False,
    })
    return receipt
