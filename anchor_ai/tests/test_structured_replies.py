import asyncio
import json
from pathlib import Path

import httpx
import pytest

from anchor_ai.app import app
from anchor_ai.contracts import TriageLabels
from anchor_ai.provider import LiveProvider, ReplayProvider
from anchor_ai.schemas import TRIAGE_FORMAT

RECORDS = json.loads(Path(__file__).with_name('fixtures').joinpath('structured_responses.json').read_text())


def post_ticket(ticket, secret='test-boundary-secret'):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://runtime') as client:
            return await client.post('/v1/structured_replies',
                                     headers={'X-Anchor-Secret': secret},
                                     json={'version': 'v1', 'tenant_id': 1,
                                           'conversation_id': 1, 'message': ticket})
    return asyncio.run(run())


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'test-boundary-secret')
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o-mini')


def test_five_ticket_replay():
    schema = TRIAGE_FORMAT['json_schema']['schema']
    assert TRIAGE_FORMAT['json_schema']['strict'] is True
    assert schema['additionalProperties'] is False
    assert set(schema['required']) == {'category', 'urgency'}
    assert set(schema['properties']['category']['enum']) == {'billing', 'technical', 'account', 'order'}
    assert set(schema['properties']['urgency']['enum']) == {'P1', 'P2', 'P3'}
    for record in RECORDS:
        response = post_ticket(record['ticket'])
        if record['expected'] is not None:
            assert response.status_code == 200, response.text
            assert TriageLabels.model_validate(response.json()).model_dump() == record['expected']
        else:
            assert response.status_code == 422
            assert response.json() == {'detail': {'code': 'invalid_model_output', 'reason': 'invalid_json'}}
        print(record['id'], response.status_code, json.dumps(response.json(), sort_keys=True))
    print('Anchor: four validated triage replies; malformed model JSON refused.')


@pytest.mark.parametrize('raw,reason,location,error_type', [
    ('{"category":"technical"}', 'schema_violation', 'urgency', 'missing'),
    ('{"category":"other","urgency":"P1"}', 'schema_violation', 'category', 'literal_error'),
    ('{"category":"technical","urgency":"urgent"}', 'schema_violation', 'urgency', 'literal_error'),
    ('{"category":"technical","urgency":"P1","private_customer_data":true}', 'schema_violation', '$', 'extra_forbidden'),
    ('{"category":', 'invalid_json', None, None),
])
def test_local_validation_uses_real_endpoint(monkeypatch, raw, reason, location, error_type):
    provider = ReplayProvider(task='structured', fixture_id='checkout')
    provider.sequence[0]['body']['choices'][0]['message']['content'] = raw
    monkeypatch.setattr('anchor_ai.llm_client.configured_provider', lambda: LiveProvider(transport=httpx.MockTransport(provider.send)))
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-key')
    response = post_ticket(RECORDS[0]['ticket'])
    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail['reason'] == reason
    if location is None:
        assert 'errors' not in detail
    else:
        assert {'loc': [location], 'type': error_type} in detail['errors']
    assert 'private_customer_data' not in response.text
    assert 'input' not in response.text
    assert 'raw' not in response.text


@pytest.mark.parametrize('refusal,content,finish,reason', [
    ('Refused private details', None, 'stop', 'provider_refusal'),
    (None, None, 'length', 'incomplete_output'),
    (None, '{"category":"technical","urgency":"P1"}', 'length', 'incomplete_output'),
])
def test_refused_or_incomplete_never_returns_triage(monkeypatch, refusal, content, finish, reason):
    provider = ReplayProvider(task='structured', fixture_id='checkout')
    choice = provider.sequence[0]['body']['choices'][0]
    choice['message'] = {'content': content, 'refusal': refusal}
    choice['finish_reason'] = finish
    monkeypatch.setattr('anchor_ai.llm_client.configured_provider', lambda: LiveProvider(transport=httpx.MockTransport(provider.send)))
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-key')
    response = post_ticket(RECORDS[0]['ticket'])
    assert response.status_code == 422
    assert response.json() == {'detail': {'code': 'invalid_model_output', 'reason': reason}}


def test_service_auth_and_request_validation():
    assert post_ticket(RECORDS[0]['ticket'], secret='wrong').status_code == 401
    assert post_ticket(' ').json() == {'detail': 'validation_failed'}
