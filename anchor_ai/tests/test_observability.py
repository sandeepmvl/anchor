import json

import httpx
import pytest
from fastapi.testclient import TestClient

from anchor_ai.app import app
from anchor_ai import observability as obs
from anchor_ai.llm_client import LlmClient

TRACE = '1234567890abcdef1234567890abcdef'
PARENT = '1234567890abcdef'
MESSAGE = 'PRIVATE_MESSAGE_SENTINEL'
SECRET = 'SERVICE_SECRET_SENTINEL'
KEY = 'PROVIDER_KEY_SENTINEL'


@pytest.fixture
def captured(monkeypatch):
    records = []
    monkeypatch.setattr(obs, 'SINK', lambda record: records.append(json.loads(json.dumps(record))))
    monkeypatch.setenv('ANCHOR_AI_SECRET', SECRET)
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.setenv('ANCHOR_AI_PROVIDER', 'offline')
    monkeypatch.setenv('ANCHOR_LLM_FIXTURE', '200')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o-mini')
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    return records


def test_replay_requests_are_joinable_and_content_free(captured):
    with TestClient(app) as client:
        for number in range(3):
            response = client.post('/v1/infer', headers={
                'X-Anchor-Secret': SECRET, 'X-Request-Id': f'ticket-{number}',
                'traceparent': f'00-{TRACE}-{PARENT}-01'}, json={
                    'version': 'v1', 'tenant_id': 1, 'conversation_id': 1, 'message': MESSAGE})
            assert response.status_code == 200
    assert len(captured) == 6
    for number in range(3):
        records = [r for r in captured if r['request_id'] == f'ticket-{number}']
        infer = next(r for r in records if r['name'] == 'python.infer')
        llm = next(r for r in records if r['name'] == 'llm.call')
        assert infer['trace_id'] == llm['trace_id'] == TRACE
        assert infer['parent_span_id'] == PARENT
        assert llm['parent_span_id'] == infer['span_id']
        assert llm['tokens']['input'] == 24
        assert llm['tokens']['output'] == 14
        assert llm['cost_basis'] == 'offline_replay_estimate'
        assert llm['latency_ms'] >= 0 and llm['status'] == 'ok'
        # A fixture without a returned model is deliberately not priced.
        assert llm['cost_estimate'] == (pytest.approx(0.000012) if llm['response_model'] == 'gpt-4o-mini' else None)
        print(json.dumps(llm, sort_keys=True))
    text = json.dumps(captured)
    for sentinel in (MESSAGE, SECRET, KEY, 'Authorization', 'Bearer'):
        assert sentinel not in text


def test_rejected_provider_has_null_usage(captured):
    class RejectedProvider:
        def key(self):
            return KEY

        def client(self):
            return httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
                httpx.Response(400, headers={'x-request-id': 'req_rejected'}, json={
                    'error': {'message': MESSAGE + KEY, 'type': 'invalid_request_error',
                              'code': 'invalid_request_error'}})))

    import asyncio
    from anchor_ai.provider import ProviderError
    with obs.span('python.infer', obs.context({'x-request-id': 'rejected',
                  'traceparent': f'00-{TRACE}-{PARENT}-01'})):
        with pytest.raises(ProviderError):
            asyncio.run(LlmClient(provider=RejectedProvider()).complete_result(MESSAGE))
    llm = next(r for r in captured if r['name'] == 'llm.call')
    assert llm['status'] == 'error'
    assert llm['tokens'] is None and llm['cost_estimate'] is None
    assert llm['provider_request_id'] == 'req_rejected'
    assert llm['trace_id'] == TRACE and llm['status_code'] == 400
    assert llm['parent_span_id'] == captured[-1]['span_id']
    for sentinel in (MESSAGE, SECRET, KEY, 'Authorization', 'Bearer'):
        assert sentinel not in json.dumps(captured)


def test_category_mapping_and_cost():
    tokens = obs.tokens_from_usage({'prompt_tokens': 1200, 'completion_tokens': 300,
        'prompt_tokens_details': {'cached_tokens': 800},
        'completion_tokens_details': {'reasoning_tokens': 100}})
    assert tokens == dict(input=400, cached_input=800, cache_write_input=0, output=300, reasoning=100)
    prices = {'example': dict(input='2', cached_input='0.5', cache_write_input='2.5', output='8')}
    assert obs.estimate('example', tokens, prices) == pytest.approx(0.0036)
    assert obs.estimate('unknown', tokens) is None
    assert obs.context({'traceparent': '00-' + '0' * 32 + '-' + PARENT + '-01'})['parent_span_id'] is None
