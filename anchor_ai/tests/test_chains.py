import asyncio
import json
import logging

import httpx
import pytest
from langchain_core.exceptions import OutputParserException

from anchor_ai.chains import TicketResponse, run_ticket_chain
from anchor_ai.llm_client import ChainReplayProvider, LlmClient
from anchor_ai.prompts import RegistryLookupError, resolve_reply_prompt

GOOD = json.dumps({'reply': 'I can help check the return policy.', 'status': 'needs_review'})
BAD = json.dumps({'reply': 'private invalid candidate', 'status': 'refund_sent'})
TICKET = {'tenant_id': 1, 'message': 'private customer ticket order 42'}
CONTRACT = {
    'version': 'chat:v3',
    'template': 'Registry v3. Ticket: {{ticket}}. Return JSON: {{format_instructions}}',
    'model_id': 'gpt-4.1-2025-04-14', 'temperature': 0.0, 'max_tokens': 256}


class RecordingProvider(ChainReplayProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.payloads = []

    def key(self):
        return 'offline-recording'

    def client(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def connection():
            def respond(request):
                payload = json.loads(request.content)
                self.payloads.append(payload)
                text = self.responses[min(self.calls, len(self.responses) - 1)]
                self.calls += 1
                return httpx.Response(200, json={
                    'model': payload['model'],
                    'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}})
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                yield client
        return connection()


def exercise(monkeypatch, responses, *, retry=False, fallback=None):
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'test-internal-secret')
    monkeypatch.setenv('ANCHOR_PROMPT_REGISTRY_URL', 'http://registry.test/api/prompt_versions/active')
    requests = []
    provider = RecordingProvider(responses)

    async def run():
        def registry(request):
            requests.append(request)
            assert request.headers['X-Anchor-Secret'] == 'test-internal-secret'
            assert request.url.path == '/api/prompt_versions/active'
            return httpx.Response(200, json=CONTRACT)
        async with httpx.AsyncClient(transport=httpx.MockTransport(registry)) as http:
            async def resolve():
                return await resolve_reply_prompt(http_client=http)
            result = await run_ticket_chain(
                TICKET, client=LlmClient(provider), resolve_prompt=resolve,
                retry=retry, fallback_client=LlmClient(fallback) if fallback else None,
                trace={'trace_id': 'test-trace', 'request_id': 'test-request',
                       'message': TICKET['message']})
            return result
    return run, provider, requests


def test_registry_v3_offline_pipeline_and_redacted_callback(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='anchor.ticket_chain')
    run, provider, requests = exercise(monkeypatch, [GOOD])
    result = asyncio.run(run())
    assert isinstance(result, TicketResponse)
    assert result.status == 'needs_review'
    assert len(requests) == 1
    payload = provider.payloads[0]
    assert payload['model'] == CONTRACT['model_id']
    assert payload['temperature'] == 0.0
    assert payload['max_tokens'] == 256
    assert 'Registry v3.' in payload['messages'][0]['content']
    assert TICKET['message'] in payload['messages'][0]['content']
    assert '{{ticket}}' not in payload['messages'][0]['content']
    records = [json.loads(row.message) for row in caplog.records
               if row.name == 'anchor.ticket_chain']
    assert records and all(row['prompt_version'] == 'v3' for row in records)
    assert all(row['registry_version'] == CONTRACT['version'] for row in records)
    assert any(row['stage'] == 'parser' and row['outcome'] == 'parse_success'
               for row in records)
    assert all(row['trace_id'] == 'test-trace' for row in records)
    serialized = json.dumps(records)
    assert TICKET['message'] not in serialized
    assert 'I can help check' not in serialized
    assert 'test-internal-secret' not in serialized


def test_invalid_output_is_rejected_without_logging_candidate(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='anchor.ticket_chain')
    run, provider, requests = exercise(monkeypatch, [BAD])
    with pytest.raises(OutputParserException):
        asyncio.run(run())
    assert provider.calls == 1
    assert len(requests) == 1
    assert 'parse_failure' in caplog.text
    assert 'private invalid candidate' not in caplog.text
    assert TICKET['message'] not in caplog.text


def test_retry_regenerates_with_one_registry_resolution(monkeypatch):
    run, provider, requests = exercise(monkeypatch, [BAD, GOOD], retry=True)
    assert asyncio.run(run()).status == 'needs_review'
    assert provider.calls == 2
    assert len(requests) == 1
    assert provider.payloads[0] == provider.payloads[1]


def test_retry_is_bounded(monkeypatch):
    run, provider, requests = exercise(monkeypatch, [BAD], retry=True)
    with pytest.raises(OutputParserException):
        asyncio.run(run())
    assert provider.calls == 2
    assert len(requests) == 1


def test_fallback_includes_validation(monkeypatch):
    secondary = ChainReplayProvider([GOOD])
    run, primary, requests = exercise(monkeypatch, [BAD], fallback=secondary)
    assert asyncio.run(run()).status == 'needs_review'
    assert primary.calls == secondary.calls == 1
    assert len(requests) == 1


def test_unauthorized_registry_response_is_not_a_prompt(monkeypatch):
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'test-secret')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(401, json={'detail': 'service_auth_failed'}))) as http:
            with pytest.raises(RegistryLookupError, match='registry_lookup_failed'):
                await resolve_reply_prompt(http_client=http)
    asyncio.run(run())
