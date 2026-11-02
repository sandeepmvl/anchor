import copy
import json
import math
from unittest.mock import patch

import httpx
import pytest

from anchor_ai.embeddings import FIXTURE_PATH, cosine, dot, load_config
from anchor_ai.llm_client import LlmClient

FIXTURE = json.loads(FIXTURE_PATH.read_text())
TEXTS = [record['text'] for record in FIXTURE['records']]


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('real network transport is forbidden')
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', forbidden)
    monkeypatch.delenv('ANCHOR_EMBED_PROVIDER', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('VOYAGE_API_KEY', raising=False)


def test_offline_default_preserves_order_duplicates_and_metadata():
    client = LlmClient(provider='offline')
    texts = [TEXTS[2], TEXTS[0], TEXTS[2], TEXTS[1]]
    with patch('anchor_ai.embeddings.httpx.Client', side_effect=AssertionError('no HTTP client')):
        batch = client.embed(texts)
        again = client.embed(texts, input_type='query')
    expected = [FIXTURE['records'][i]['vector'] for i in [2, 0, 2, 1]]
    assert batch.vectors == again.vectors == expected
    assert batch.provider == 'offline'
    assert batch.model == FIXTURE['model']
    assert batch.revision == FIXTURE['revision']
    assert batch.dimensions == FIXTURE['dimensions']
    assert batch.input_type == 'document'
    assert again.input_type == 'query'
    assert callable(client.complete) and callable(client.stream_tokens)


@pytest.mark.parametrize('provider,dimensions', [('openai', 3072), ('voyage', 1536), ('offline', 8)])
def test_incompatible_dimensions_fail_before_http_client(provider, dimensions):
    config = copy.deepcopy(load_config())
    config['providers'][provider]['dimensions'] = dimensions
    with patch('anchor_ai.embeddings.httpx.Client', side_effect=AssertionError('no HTTP client')):
        with pytest.raises(ValueError, match='incompatible'):
            LlmClient(provider='offline').embed(TEXTS, provider=provider, config=config)


@pytest.mark.parametrize('provider', ['openai', 'voyage'])
@pytest.mark.parametrize('mode', ['query', 'document'])
def test_mocked_adapters_share_contract(provider, mode):
    settings = load_config()['providers'][provider]
    dimensions = settings['dimensions']
    first = [1.0] + [0.0] * (dimensions - 1)
    second = [0.0, 1.0] + [0.0] * (dimensions - 2)
    requests = []

    def respond(request):
        requests.append(request)
        payload = json.loads(request.content)
        assert payload['input'] == TEXTS[:2]
        assert payload['model'] == settings['model']
        if provider == 'openai':
            assert request.url.host == 'api.openai.com'
            assert payload['dimensions'] == dimensions
            assert payload['encoding_format'] == 'float'
            assert 'input_type' not in payload
        else:
            assert request.url.host == 'api.voyageai.com'
            assert payload['input_type'] == mode
            assert payload['truncation'] is False
        return httpx.Response(200, json={'model': settings['model'], 'data': [
            {'index': 1, 'embedding': second}, {'index': 0, 'embedding': first}]})

    batch = LlmClient(provider='offline').embed(
        TEXTS[:2], mode, provider=provider,
        transport=httpx.MockTransport(respond), api_key='mock-only-not-a-credential')
    assert len(requests) == 1
    assert batch.vectors == [first, second]
    assert (batch.model, batch.revision, batch.dimensions, batch.input_type) == (
        settings['model'], settings['revision'], dimensions, mode)


@pytest.mark.parametrize('provider', ['openai', 'voyage'])
def test_bad_response_dimensions_rejected_with_mock_transport(provider):
    def respond(request):
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1.0, 0.0]}]})
    with pytest.raises(ValueError, match='dimension mismatch'):
        LlmClient(provider='offline').embed(
            TEXTS[:1], provider=provider, api_key='mock-only',
            transport=httpx.MockTransport(respond))


@pytest.mark.parametrize('data', [
    [{'index': 0, 'embedding': [1.0]}, {'index': 0, 'embedding': [1.0]}],
    [{'index': 1, 'embedding': [1.0]}],
])
def test_invalid_indices_rejected(data):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={'data': data}))
    with pytest.raises(ValueError, match='indices'):
        LlmClient(provider='offline').embed(
            TEXTS[:2], provider='openai', api_key='mock-only', transport=transport)


@pytest.mark.parametrize('texts,mode', [([], 'query'), ([''], 'document'), ([1], 'query'), (TEXTS, 'invalid')])
def test_invalid_inputs(texts, mode):
    with pytest.raises(ValueError):
        LlmClient(provider='offline').embed(texts, mode)


def test_unknown_recording_and_missing_live_key_fail_closed():
    client = LlmClient(provider='offline')
    with pytest.raises(ValueError, match='recording'):
        client.embed(['not a recorded sentence'])
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        client.embed(TEXTS, provider='openai')


def test_cosine_and_unit_dot_product():
    a, b = [3.0, 4.0], [4.0, 3.0]
    unit_a, unit_b = [x / 5 for x in a], [x / 5 for x in b]
    assert math.isclose(cosine(a, b), dot(unit_a, unit_b))
    assert math.isclose(cosine(a, b), cosine([7 * x for x in a], b))
    with pytest.raises(ValueError, match='zero'):
        cosine([0.0, 0.0], b)
    with pytest.raises(ValueError, match='dimensions'):
        cosine([1.0], b)
