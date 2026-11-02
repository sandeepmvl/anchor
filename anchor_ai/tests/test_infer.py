import secrets

import pytest
from fastapi.testclient import TestClient

from anchor_ai.app import app
from anchor_ai.llm_client import LlmClient

PAYLOAD = {
    'version': 'v1', 'conversation_id': 1,
    'tenant_id': 1, 'message': 'Can I return my order?',
}


@pytest.fixture
def boundary(monkeypatch):
    secret = secrets.token_hex(24)
    monkeypatch.setenv('ANCHOR_AI_SECRET', secret)
    monkeypatch.setenv('ANCHOR_AI_PROVIDER', 'offline')
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.setenv('ANCHOR_LLM_FIXTURE', '200')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with TestClient(app) as client:
        yield client, {'X-Anchor-Secret': secret}


@pytest.mark.parametrize('path', ['/infer', '/v1/infer'])
def test_reply(boundary, path):
    client, headers = boundary
    response = client.post(path, json=PAYLOAD, headers=headers)
    assert response.status_code == 200
    assert response.json() == {
        'reply': LlmClient().complete(PAYLOAD['message']), 'tokens': 38,
        'usage': {'prompt_tokens': 24, 'completion_tokens': 14, 'total_tokens': 38},
        'finish_reason': 'stop', 'attempts': 1,
    }


@pytest.mark.parametrize('headers', [{}, {'X-Anchor-Secret': 'wrong'}])
def test_auth_rejected(boundary, headers):
    client, _ = boundary
    response = client.post('/infer', json=PAYLOAD, headers=headers)
    assert response.status_code == 401
    assert response.json() == {'detail': 'service_auth_failed'}


@pytest.mark.parametrize('change', [
    {'tenant_id': '1'}, {'conversation_id': True}, {'message': ' '},
    {'message': 'x' * 4001}, {'version': 'v2'}, {'extra': 1},
])
def test_invalid_contract(boundary, change):
    client, headers = boundary
    response = client.post('/infer', json={**PAYLOAD, **change}, headers=headers)
    assert response.status_code == 422


def test_missing_field_and_malformed_json(boundary):
    client, headers = boundary
    assert client.post('/infer', json={}, headers=headers).status_code == 422
    response = client.post(
        '/infer', content='not json',
        headers={**headers, 'Content-Type': 'application/json'},
    )
    assert response.status_code == 422


def test_server_requires_configuration(boundary, monkeypatch):
    client, headers = boundary
    monkeypatch.delenv('ANCHOR_AI_SECRET')
    response = client.post('/infer', json=PAYLOAD, headers=headers)
    assert response.status_code == 503


def test_validation_failure_is_correlated_without_provider(boundary, monkeypatch):
    import json
    from anchor_ai import observability as obs
    records = []
    monkeypatch.setattr(obs, 'SINK', records.append)

    async def forbidden(*args, **kwargs):
        pytest.fail('validation must not invoke the provider')

    monkeypatch.setattr(LlmClient, 'complete_result', forbidden)
    client, headers = boundary
    response = client.post('/v1/infer', json={**PAYLOAD, 'tenant_id': 'invalid'},
                           headers={**headers, 'X-Request-Id': 'validation-3660',
                                    'traceparent': '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01'})
    assert response.status_code == 422
    assert len(records) == 1
    item = records[0]
    assert item['request_id'] == 'validation-3660' and item['trace_id'] == 'a' * 32
    assert item['parent_span_id'] == 'b' * 16
    assert item['name'] == 'python.infer' and item['stage'] == 'validation'
    assert item['error_type'] == 'validation_failed' and item['status'] == 'error'
    assert item['tokens'] is None and item['cost_estimate'] is None
    assert PAYLOAD['message'] not in json.dumps(records)
    print(json.dumps(item, sort_keys=True))
