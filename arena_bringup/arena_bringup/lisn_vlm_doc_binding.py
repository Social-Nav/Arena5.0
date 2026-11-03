"""Provider-scoped binding for the authorized VLM document.

The document has one section per provider.  Each section binds its own
``base_url`` / model / credential; a document-wide "first api_key" rule mixes
providers and produces HTTP 401.  This module resolves exactly one provider
section and refuses ambiguous or incomplete bindings.

It is pure: it takes the document text as a string, never fetches it, and never
logs a credential.  Callers keep the credential in memory and inject it into a
single process environment.
"""
from __future__ import annotations

import re

HEADING_RE = re.compile(r"<(h[1-6])[^>]*>(.*?)</\1>", re.S)
CODE_RE = re.compile(r"<code[^>]*>(.*?)</code>", re.S)
LINE_SPLIT_RE = re.compile(r"<br\s*/?>|\n")
TAG_RE = re.compile(r"<[^>]+>")
FIELD_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*[\"']?([^\"'\n]*?)[\"']?\s*$"
)

PROVIDER_ALIASES = {
    'gemini-2.5-pro': ('gemini2.5pro', 'gemini 2.5 pro', 'gemini-2.5-pro'),
    'gpt-5': ('gpt5', 'gpt-5', 'gpt 5'),
    'seed-1.6-vl-think': ('seed 1.6 vl think', 'seed-1.6-vl-think', 'seed1.6vlthink'),
    'seed-1.8-vl': ('seed 1.8 vl', 'seed-1.8-vl', 'seed1.8vl'),
}

PROVIDER_CREDENTIAL_ENV = {
    'gemini-2.5-pro': 'ARENA_LISN_GEMINI_AK',
    'gpt-5': 'ARENA_LISN_GPT5_AK',
    'seed-1.6-vl-think': 'ARK_API_KEY',
    'seed-1.8-vl': 'ARK_API_KEY',
}

DEFAULT_PROVIDER_RULES = {
    'gemini-2.5-pro': {
        'model_allow': ('gemini-2.5-pro',),
        'model_deny_prefixes': ('ep-',),
        'endpoint_host_allow': ('search.bytedance.net', 'aidp.bytedance.net'),
        'endpoint_path_contains': '/openai/deployments/gpt_openapi',
    },
    'gpt-5': {
        'model_allow': ('gpt-5-2025-08-07',),
        'model_deny_prefixes': ('ep-',),
        'endpoint_host_allow': ('search.bytedance.net', 'aidp.bytedance.net'),
        'endpoint_path_contains': '/openai/deployments/gpt_openapi',
    },
    'seed-1.6-vl-think': {
        'model_prefix_allow': ('ep-',),
        'endpoint_host_allow': ('ark-cn-beijing.bytedance.net',
                                'ark.cn-beijing.volces.com'),
        'endpoint_path_contains': '/api/v3',
    },
    'seed-1.8-vl': {
        'model_prefix_allow': ('ep-',),
        'endpoint_host_allow': ('ark-cn-beijing.bytedance.net',
                                'ark.cn-beijing.volces.com'),
        'endpoint_path_contains': '/api/v3',
    },
}

ENDPOINT_HOST_ALLOWLIST = frozenset({
    'search.bytedance.net',
    'aidp.bytedance.net',
    'ark.cn-beijing.volces.com',
    'ark-cn-beijing.bytedance.net',
})


def safe_endpoint_label(endpoint: str) -> str:
    """Return a hostname-only label; never userinfo, path, query or fragment."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(str(endpoint or ''))
    except Exception:
        return 'unparsed'
    host = str(parsed.hostname or '').strip().lower()
    if not host:
        return 'empty'
    if host in {'127.0.0.1', 'localhost', '::1'}:
        return 'loopback'
    if host in ENDPOINT_HOST_ALLOWLIST:
        return host
    return 'unlisted'


class AmbiguousProviderBinding(RuntimeError):
    """The document does not contain exactly one unambiguous provider binding."""


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text)).strip().lower()


def split_provider_sections(content: str):
    """Split DocxXML content into ``(heading, body)`` pairs by heading tags."""
    matches = list(HEADING_RE.finditer(str(content or '')))
    sections = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        heading = TAG_RE.sub('', match.group(2)).strip()
        sections.append((heading, content[start:end]))
    return sections


def _parse_code_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in LINE_SPLIT_RE.split(block):
        line = TAG_RE.sub('', raw_line).strip().rstrip(',').rstrip(')').strip()
        match = FIELD_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key in fields and fields[key] != value:
            existing = fields[key]
            value_is_variable = bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value))
            existing_is_variable = bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', existing))
            if value_is_variable and not existing_is_variable:
                # ``api_version = api_version`` is the SDK alias, not a
                # conflicting literal; keep the literal value.
                continue
            if existing_is_variable and not value_is_variable:
                fields[key] = value
                continue
            raise AmbiguousProviderBinding('duplicate field with conflicting values')
        fields[key] = value
    return fields


def _resolve_credential(fields: dict[str, str]) -> tuple[str, str]:
    ak = str(fields.get('ak', '') or '').strip()
    api_key = str(fields.get('api_key', '') or '').strip()
    if ak:
        # ``api_key = ak`` is the SDK alias, not a second credential.
        if api_key and api_key not in {'ak', ak}:
            raise AmbiguousProviderBinding('ak and api_key carry different credentials')
        return ak, 'ak'
    if api_key and api_key != 'ak':
        return api_key, 'api_key'
    raise AmbiguousProviderBinding('provider section has no credential')


def _validate_rule(model: str, base_url: str, rule: dict) -> None:
    """Reject a provider section that does not match its explicit rule.

    Error messages are static categories: a malformed synthetic config can
    place a dummy secret in the model name or heading, and that value must not
    be echoed.
    """
    from urllib.parse import urlparse

    deny = tuple(rule.get('model_deny_prefixes') or ())
    if any(model.startswith(prefix) for prefix in deny):
        raise AmbiguousProviderBinding('provider model matches a denied prefix')
    allow = tuple(rule.get('model_allow') or ())
    prefix_allow = tuple(rule.get('model_prefix_allow') or ())
    if not allow and not prefix_allow:
        raise AmbiguousProviderBinding('provider rule has no model allow')
    if allow and model not in allow:
        raise AmbiguousProviderBinding('provider model is not in the allowlist')
    if prefix_allow and not any(model.startswith(prefix) for prefix in prefix_allow):
        raise AmbiguousProviderBinding('provider model does not match an allowed prefix')
    hosts = tuple(rule.get('endpoint_host_allow') or ())
    if not hosts:
        raise AmbiguousProviderBinding('provider rule has no endpoint host allow')
    parsed = urlparse(base_url)
    host = str(parsed.hostname or '').strip().lower()
    if host not in hosts:
        raise AmbiguousProviderBinding('provider endpoint host is not in the allowlist')
    loopback = host in {'127.0.0.1', 'localhost', '::1'}
    if parsed.scheme != 'https' and not (loopback and parsed.scheme == 'http'):
        raise AmbiguousProviderBinding('provider endpoint scheme is not allowed')
    if parsed.username or parsed.password:
        raise AmbiguousProviderBinding('provider endpoint userinfo is not allowed')
    if parsed.query or parsed.fragment:
        raise AmbiguousProviderBinding('provider endpoint query or fragment is not allowed')
    exact = str(rule.get('endpoint_path_exact') or '')
    contains = str(rule.get('endpoint_path_contains') or '')
    if not exact and not contains:
        raise AmbiguousProviderBinding('provider rule has no endpoint path constraint')
    path = str(parsed.path or '')
    if exact and path != exact:
        raise AmbiguousProviderBinding('provider endpoint path does not match the exact rule')
    if contains and contains not in path:
        raise AmbiguousProviderBinding('provider endpoint path does not match the allowed pattern')


def resolve_provider_binding(content: str, provider: str = 'gemini-2.5-pro', *,
                             rules: dict | None) -> dict:
    """Resolve one provider's section to ``{model, base_url, credential, ...}``.

    ``rules`` is required: a provider section is accepted only when an explicit
    ``model_allow``/``model_prefix_allow`` and ``endpoint_host_allow`` rule
    matches it.  Heading text plus a non-empty model/base is not enough.

    Raises ``AmbiguousProviderBinding`` with a static category when the section
    is missing, duplicated, has more than one code block, carries conflicting
    credentials, lacks a model/base URL, or violates the provider rule.  It
    never falls back to another provider's credential and never echoes the
    input values.
    """
    if rules is None:
        raise AmbiguousProviderBinding('explicit provider rules are required')
    key = str(provider or '').strip().lower()
    aliases = PROVIDER_ALIASES.get(key)
    if aliases is None:
        raise AmbiguousProviderBinding('unknown provider')
    rule = rules.get(key)
    if not isinstance(rule, dict):
        raise AmbiguousProviderBinding('no explicit rule for provider')
    matched = [
        (heading, body)
        for heading, body in split_provider_sections(content)
        if _normalize(heading) in aliases
    ]
    if len(matched) != 1:
        raise AmbiguousProviderBinding('provider section missing or duplicated')
    heading, body = matched[0]
    blocks = CODE_RE.findall(body)
    if len(blocks) != 1:
        raise AmbiguousProviderBinding('provider section has an unexpected code block count')
    fields = _parse_code_fields(blocks[0])
    credential, credential_field = _resolve_credential(fields)
    model = str(fields.get('model_name') or fields.get('model') or '').strip()
    base_url = str(fields.get('base_url') or '').strip()
    if not model:
        raise AmbiguousProviderBinding('provider section has no model')
    if not base_url:
        raise AmbiguousProviderBinding('provider section has no base_url')
    _validate_rule(model, base_url, rule)
    return {
        'provider': key,
        'heading': heading,
        'model': model,
        'base_url': base_url,
        'api_version': str(fields.get('api_version') or '').strip(),
        'credential': credential,
        'credential_field': credential_field,
        'credential_env': PROVIDER_CREDENTIAL_ENV.get(key, ''),
        'max_tokens': str(fields.get('max_tokens') or '').strip(),
    }


def redacted_binding(binding: dict) -> dict:
    """Public-safe view: provider/model/endpoint label and field names only."""
    return {
        'provider': binding.get('provider'),
        'heading': binding.get('heading'),
        'model': binding.get('model'),
        'endpoint_label': safe_endpoint_label(binding.get('base_url') or ''),
        'api_version_present': bool(binding.get('api_version')),
        'credential_field': binding.get('credential_field'),
        'credential_env': binding.get('credential_env'),
        'credential_present': bool(binding.get('credential')),
    }
