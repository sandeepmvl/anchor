import asyncio
import json

import pytest

from anchor_ai import observability as obs
from anchor_ai.llm_client import LlmClient
from anchor_ai.provider import ReplayProvider, REASONING_RESPONSES
from anchor_ai.reasoning import REPLAY_MODEL, ReasoningError, request_fields


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv('OPENAI_MODEL', REPLAY_MODEL)
    records = []
    monkeypatch.setattr(obs, 'SINK', records.append)
    ticket = json.loads(REASONING_RESPONSES.read_text())['tickets'][0]
    return ticket, records


@pytest.mark.parametrize('effort,mapped', [('off', 'none'), ('low', 'low'), ('high', 'high')])
def test_effort_reaches_request(setup, effort, mapped):
    ticket, records = setup
    provider = ReplayProvider(task='reasoning', fixture_id=ticket['id'], effort=effort)
    result = asyncio.run(LlmClient(provider).complete_result(ticket['ticket'], max_tokens=4000, effort=effort))
    payload = json.loads(provider.requests[0].content)
    assert payload['reasoning_effort'] == mapped
    assert payload['max_completion_tokens'] == 4000
    assert 'max_tokens' not in payload
    assert result.finish_reason == 'stop'
    assert records[-1]['reasoning_effort'] == mapped


@pytest.mark.parametrize('effort', ['medium', '', 1, False, []])
def test_invalid_effort_rejected_before_send(setup, effort):
    ticket, records = setup
    provider = ReplayProvider(task='reasoning', fixture_id='B1', effort='high')
    with pytest.raises(ReasoningError, match='invalid_reasoning_effort'):
        asyncio.run(LlmClient(provider).complete_result(ticket['ticket'], effort=effort))
    assert provider.requests == []
    assert records[-1]['error_type'] == 'invalid_reasoning_effort'


@pytest.mark.parametrize('limit', [0, -1, 32769, True, 1.5])
def test_invalid_limit(limit):
    with pytest.raises(ReasoningError, match='invalid_reasoning_limit'):
        request_fields(REPLAY_MODEL, 'high', limit, offline=True)


def test_unsupported_capabilities_fail_explicitly(setup, monkeypatch):
    ticket, records = setup
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o-mini')
    provider = ReplayProvider(task='reasoning', fixture_id='B1', effort='high')
    with pytest.raises(ReasoningError, match='unsupported_reasoning_model'):
        asyncio.run(LlmClient(provider).complete_result(ticket['ticket'], effort='high'))
    assert provider.requests == []
    with pytest.raises(ReasoningError, match='unsupported_reasoning_model'):
        request_fields(REPLAY_MODEL, 'high', 4000, offline=False)


def test_no_effort_preserves_original_payload(setup, monkeypatch):
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o-mini')
    provider = ReplayProvider()
    asyncio.run(LlmClient(provider).complete_result('hello'))
    payload = json.loads(provider.requests[0].content)
    assert payload == {'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 256}


def test_truncation_returns_once_and_is_logged(setup):
    ticket, records = setup
    provider = ReplayProvider(task='reasoning', fixture_id='B1', effort='high', max_tokens=300)
    result = asyncio.run(LlmClient(provider).complete_result(ticket['ticket'], max_tokens=300, effort='high'))
    assert result.reply == '' and result.finish_reason == 'length'
    assert result.attempts == len(provider.requests) == 1
    record = records[-1]
    assert record['status'] == 'ok'
    assert record['completion_status'] == 'incomplete'
    assert record['finish_reason'] == 'length'
    assert record['tokens']['reasoning'] == record['tokens']['output'] == 300
    assert record['max_completion_tokens'] == 300
