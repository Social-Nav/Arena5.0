"""Thin Seed-2.1 Pro ChatCompletions adapter for LISN proposals (pure).

The Ark endpoint is OpenAI-compatible at ``{base}/chat/completions``.  The
caller keeps the Bearer token in memory; these helpers never receive or emit
the token, an Authorization header, a keyed URL or a raw response body.

Base URL and model are runtime configuration (``ARENA_LISN_BASE_URL`` and
``ARENA_LISN_MODEL``); this module deliberately defines no default endpoint or
model and performs no network I/O.
"""
import base64
import hashlib
import json
from typing import Any, Mapping

from arena_bringup.lisn_dmx_timing import REQUIRED_FIELDS, parse_proposal

MAX_OUTPUT_TOKENS = 4095
CHAT_PATH = '/chat/completions'
FAILURE_CODES = frozenset({
    'http_401', 'http_403', 'http_429', 'timeout', 'transport',
    'response_json', 'proposal_rejected', 'unknown',
})


def endpoint_from_base(base: str) -> str:
    """Return the ChatCompletions endpoint for an API base.

    The documented base already contains the API version (``/api/v3``); only
    the trailing slash is normalised.  An endpoint that already ends with
    ``/chat/completions`` is accepted unchanged.
    """
    value = str(base or '').strip()
    if not value:
        raise ValueError('missing_base_url')
    value = value.rstrip('/')
    if value.endswith(CHAT_PATH):
        return value
    return value + CHAT_PATH


def build_chat_request(*, api_token: str, base: str, model: str,
                       image_bytes: bytes, mime: str, prompt: str,
                       max_output_tokens: int | None = None,
                       temperature: float | None = None):
    """Build the ChatCompletions payload and safe diagnostics.

    The minimal payload matches the successful documented curl: only
    ``model`` and ``messages`` are required.  ``max_tokens`` and
    ``temperature`` are optional and omitted unless the caller supplies a
    positive/explicit value.  Diagnostics contain the endpoint host label and
    booleans only; the token, Authorization header and keyed URL are never
    included.
    """
    if not api_token:
        raise ValueError('missing_api_token')
    if not str(model or '').strip():
        raise ValueError('missing_model')
    endpoint = endpoint_from_base(base)
    encoded = base64.b64encode(bytes(image_bytes)).decode('ascii')
    payload = {
        'model': str(model).strip(),
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}},
                {'type': 'text', 'text': str(prompt)},
            ],
        }],
    }
    if max_output_tokens is not None and int(max_output_tokens) > 0:
        payload['max_tokens'] = int(max_output_tokens)
    if temperature is not None:
        payload['temperature'] = float(temperature)
    diagnostics = {
        'endpoint_label': endpoint.split('://', 1)[-1].split('/', 1)[0],
        'path': CHAT_PATH,
        'model': str(model).strip(),
        'auth_method': 'bearer',
        'query_key_present': False,
        'mime': str(mime),
        'image_bytes': len(bytes(image_bytes)),
        'image_sha256': hashlib.sha256(bytes(image_bytes)).hexdigest(),
        'max_output_tokens': int(max_output_tokens) if max_output_tokens else None,
        'max_tokens_present': 'max_tokens' in payload,
        'temperature': float(temperature) if temperature is not None else None,
        'temperature_present': 'temperature' in payload,
        'content_types': ['image_url', 'text'],
        'token_emitted': False,
        'authorization_header_emitted': False,
    }
    return payload, diagnostics


def extract_chat_text(data: Any) -> str:
    """Extract assistant text from a ChatCompletions response."""
    if not isinstance(data, Mapping):
        return ''
    choices = data.get('choices')
    if not isinstance(choices, list) or not choices:
        return ''
    first = choices[0]
    if not isinstance(first, Mapping):
        return ''
    message = first.get('message')
    if not isinstance(message, Mapping):
        return ''
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get('text'), str):
                parts.append(part['text'])
        return ''.join(parts)
    return ''


def parse_chat_proposal(text: str):
    return parse_proposal(text)


def chat_failure_receipt(*, error_code: str, error_category: str, status,
                         t0_ns: int, t_headers_ns: int | None = None,
                         t1_ns: int | None = None, body_read_bytes: int = 0,
                         request_count: int = 1, retry: int = 0,
                         container_name: str = '', container_id: str = '',
                         endpoint_label: str = '', model: str = '') -> dict:
    """Build a safe failure receipt (no key, URL, header or raw body)."""
    code = str(error_code) if str(error_code) in FAILURE_CODES else 'unknown'
    end = int(t1_ns) if t1_ns is not None else t0_ns
    headers = None
    if t_headers_ns is not None:
        headers = round((int(t_headers_ns) - int(t0_ns)) / 1e9, 6)
    return {
        'schema': 'arena.seed21.chat.failure.v1',
        'error_code': code,
        'error_category': str(error_category),
        'status': status,
        't0_after_serialization': True,
        'headers_seconds': headers,
        'elapsed_seconds': round((end - int(t0_ns)) / 1e9, 6),
        'body_read_bytes': int(body_read_bytes),
        'request_count': int(request_count),
        'retry': int(retry),
        'endpoint_label': str(endpoint_label),
        'model': str(model),
        'auth_method': 'bearer',
        'query_key_present': False,
        'container_name': str(container_name),
        'container_id': str(container_id),
        'secret_cleared': True,
        'raw_body_saved': False,
        'key_or_url_emitted': False,
    }


def proposal_fields(proposal: Mapping[str, Any]) -> dict[str, bool]:
    return {name: name in proposal for name in REQUIRED_FIELDS}


def proposal_valid(proposal: Mapping[str, Any]) -> bool:
    parameters = proposal.get('controller_parameters')
    return bool(
        all(proposal_fields(proposal).values())
        and isinstance(parameters, dict)
        and 'max_linear_velocity_mps' in parameters
    )


def dump_receipt(receipt: Mapping[str, Any]) -> str:
    return json.dumps(receipt, indent=2, sort_keys=True)
