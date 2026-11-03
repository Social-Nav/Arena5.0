"""Provider-section binding tests with synthetic documents only.

No real document is fetched and no real credential is used.  The dummy values
below exist only to prove section scoping, explicit allow rules and redaction.
"""
import json

import pytest

from arena_bringup.lisn_vlm_doc_binding import (
    AmbiguousProviderBinding,
    redacted_binding,
    resolve_provider_binding,
    safe_endpoint_label,
)

GEMINI_AK = 'DUMMY-GEMINI-AK-NOT-A-SECRET'
SEED_KEY = 'DUMMY-SEED-API-KEY-NOT-A-SECRET'

RULES = {
    'gemini-2.5-pro': {
        'model_allow': ('gemini-2.5-pro',),
        'model_deny_prefixes': ('ep-',),
        'endpoint_host_allow': ('search.example.invalid',),
        'endpoint_path_contains': '/openai/deployments/gpt_openapi',
    },
    'seed-1.6-vl-think': {
        'model_prefix_allow': ('ep-',),
        'endpoint_host_allow': ('ark.example.invalid',),
        'endpoint_path_contains': '/api/v3',
    },
}

DOC = f"""
<h3>Seed 1.6 VL Think</h3>
<p>ark</p>
<code>model_name = "ep-seed-dummy"
base_url = "https://ark.example.invalid/api/v3"
client = OpenAI(
api_key = "{SEED_KEY}",
)</code>
<h3>GPT5</h3>
<code>base_url = "https://search.example.invalid/openai/deployments/gpt_openapi"
api_version = "2024-03-01-preview"
ak = "DUMMY-GPT5-AK"
model_name = "gpt-5-dummy"
api_key = ak</code>
<h3>Gemini2.5Pro</h3>
<code>base_url = "https://search.example.invalid/openai/deployments/gpt_openapi"
api_version = "2024-03-01-preview"
ak = "{GEMINI_AK}"
model_name = "gemini-2.5-pro"
client = openai.AzureOpenAI(
api_key = ak</code>
"""


def _resolve(content, provider='gemini-2.5-pro', rules=RULES):
    return resolve_provider_binding(content, provider, rules=rules)


def _mutate_gemini(doc, old, new):
    prefix, gemini = doc.split('<h3>Gemini2.5Pro</h3>')
    return prefix + '<h3>Gemini2.5Pro</h3>' + gemini.replace(old, new, 1)


def test_gemini_section_binds_its_own_ak():
    binding = _resolve(DOC)
    assert binding['model'] == 'gemini-2.5-pro'
    assert binding['base_url'].endswith('/openai/deployments/gpt_openapi')
    assert binding['credential'] == GEMINI_AK
    assert binding['credential_field'] == 'ak'
    assert binding['credential_env'] == 'ARENA_LISN_GEMINI_AK'


def test_seed_section_does_not_leak_into_gemini():
    seed = _resolve(DOC, 'seed-1.6-vl-think')
    assert seed['credential'] == SEED_KEY
    assert seed['credential_field'] == 'api_key'
    assert seed['credential_env'] == 'ARK_API_KEY'


def test_missing_provider_section_fails_closed():
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(DOC, 'unknown-provider')
    only_seed = DOC.split('<h3>GPT5</h3>')[0]
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(only_seed)


def test_duplicate_provider_section_is_ambiguous():
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(DOC + DOC)


def test_multiple_code_blocks_in_section_are_ambiguous():
    doc = _mutate_gemini(
        DOC,
        'api_key = ak</code>',
        'api_key = ak</code><code>ak = "SECOND"</code>',
    )
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(doc)


def test_conflicting_ak_and_api_key_are_rejected():
    doc = _mutate_gemini(
        DOC,
        'api_key = ak</code>',
        'api_key = "DIFFERENT-LITERAL"</code>',
    )
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(doc)


def test_missing_model_or_base_is_rejected():
    doc = DOC.replace('model_name = "gemini-2.5-pro"\n', '', 1)
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(doc)
    doc = DOC.replace(
        'base_url = "https://search.example.invalid/openai/deployments/gpt_openapi"\n'
        'api_version = "2024-03-01-preview"\n'
        f'ak = "{GEMINI_AK}"',
        'api_version = "2024-03-01-preview"\n' + f'ak = "{GEMINI_AK}"',
        1,
    )
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(doc)


def test_gemini_section_with_seed_model_is_rejected():
    doc = _mutate_gemini(DOC, 'model_name = "gemini-2.5-pro"', 'model_name = "ep-seed-dummy"')
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(doc)


def test_gemini_section_with_ark_endpoint_is_rejected():
    doc = _mutate_gemini(
        DOC,
        'base_url = "https://search.example.invalid/openai/deployments/gpt_openapi"',
        'base_url = "https://ark.example.invalid/api/v3"',
    )
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(doc)


def test_rules_are_required_and_provider_must_have_a_rule():
    with pytest.raises(AmbiguousProviderBinding):
        resolve_provider_binding(DOC, 'gemini-2.5-pro', rules=None)
    with pytest.raises(AmbiguousProviderBinding):
        resolve_provider_binding(DOC, 'gemini-2.5-pro', rules={})


def test_redacted_binding_never_contains_the_credential_or_endpoint_path():
    binding = _resolve(DOC)
    public = redacted_binding(binding)
    text = json.dumps(public)
    assert GEMINI_AK not in text
    assert SEED_KEY not in text
    assert '/openai/' not in text
    assert 'endpoint_path' not in public
    assert 'endpoint_host' not in public
    assert public['credential_present'] is True
    assert public['credential_field'] == 'ak'


def test_safe_endpoint_label_strips_userinfo_path_query():
    label = safe_endpoint_label(
        'https://user:DUMMY_SECRET@search.bytedance.net/openai/deployments/x?token=DUMMY_SECRET#frag'
    )
    assert label == 'search.bytedance.net'
    assert 'DUMMY_SECRET' not in label
    assert safe_endpoint_label('http://127.0.0.1:8080/v1?key=DUMMY_SECRET') == 'loopback'
    assert safe_endpoint_label('https://unknown.invalid/path?q=DUMMY_SECRET') == 'unlisted'


def test_http_scheme_is_rejected_for_non_loopback():
    doc = _mutate_gemini(
        DOC,
        'base_url = "https://search.example.invalid/openai/deployments/gpt_openapi"',
        'base_url = "http://search.example.invalid/openai/deployments/gpt_openapi"',
    )
    with pytest.raises(AmbiguousProviderBinding, match='scheme'):
        _resolve(doc)


def test_userinfo_query_fragment_are_rejected():
    for suffix in (
        'https://user:DUMMY_SECRET@search.example.invalid/openai/deployments/gpt_openapi',
        'https://search.example.invalid/openai/deployments/gpt_openapi?token=DUMMY_SECRET',
        'https://search.example.invalid/openai/deployments/gpt_openapi#DUMMY_SECRET',
    ):
        doc = _mutate_gemini(
            DOC,
            'base_url = "https://search.example.invalid/openai/deployments/gpt_openapi"',
            f'base_url = "{suffix}"',
        )
        with pytest.raises(AmbiguousProviderBinding):
            _resolve(doc)


def test_loopback_http_is_allowed_with_explicit_path_rule():
    rules = {
        'gemini-2.5-pro': {
            'model_allow': ('gemini-2.5-pro',),
            'endpoint_host_allow': ('127.0.0.1',),
            'endpoint_path_contains': '/v1/chat/completions',
        },
    }
    doc = _mutate_gemini(
        DOC,
        'base_url = "https://search.example.invalid/openai/deployments/gpt_openapi"',
        'base_url = "http://127.0.0.1:8080/v1/chat/completions"',
    )
    binding = resolve_provider_binding(doc, 'gemini-2.5-pro', rules=rules)
    assert binding['model'] == 'gemini-2.5-pro'


def test_rule_without_path_constraint_is_rejected():
    rules = {
        'gemini-2.5-pro': {
            'model_allow': ('gemini-2.5-pro',),
            'endpoint_host_allow': ('search.example.invalid',),
        },
    }
    with pytest.raises(AmbiguousProviderBinding, match='path constraint'):
        resolve_provider_binding(DOC, 'gemini-2.5-pro', rules=rules)


def test_exact_path_rule_mismatch_is_rejected():
    rules = {
        'gemini-2.5-pro': {
            'model_allow': ('gemini-2.5-pro',),
            'endpoint_host_allow': ('search.example.invalid',),
            'endpoint_path_exact': '/different/path',
        },
    }
    with pytest.raises(AmbiguousProviderBinding, match='exact'):
        resolve_provider_binding(DOC, 'gemini-2.5-pro', rules=rules)


def test_malformed_values_are_not_echoed():
    secret = 'DUMMY_SECRET_IN_MODEL_NAME'
    doc = _mutate_gemini(DOC, 'model_name = "gemini-2.5-pro"', f'model_name = "{secret}"')
    try:
        _resolve(doc)
    except AmbiguousProviderBinding as exc:
        rendered = str(exc)
    else:
        raise AssertionError('expected a rule rejection')
    assert secret not in rendered


def test_sdk_variable_alias_does_not_conflict_with_literal():
    doc = _mutate_gemini(
        DOC,
        'api_version = "2024-03-01-preview"',
        'api_version = "2024-03-01-preview"\napi_version = api_version',
    )
    binding = _resolve(doc)
    assert binding['api_version'] == '2024-03-01-preview'


def test_two_conflicting_literals_are_still_rejected():
    doc = _mutate_gemini(
        DOC,
        'api_version = "2024-03-01-preview"',
        'api_version = "2024-03-01-preview"\napi_version = "2099-01-01"',
    )
    with pytest.raises(AmbiguousProviderBinding):
        _resolve(doc)
