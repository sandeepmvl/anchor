import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from anchor_ai.contracts import InferRequest, PromptContract
from anchor_ai.llm_client import LlmClient
from anchor_ai.prompts import render_chat, contract_fingerprint
from anchor_ai import observability


TICKET = 'Can I get a refund?'
V1 = dict(version='chat:v1', template='Ticket: {{ticket}}. Reply plainly.',
          model_id='gpt-4.1-2025-04-14', temperature=0.0, max_tokens=120)
V2 = dict(V1, version='chat:v2', template='Ticket: {{ticket}}. Begin with Support:.')


def request(contract):
    return InferRequest.model_validate(dict(version='v1', tenant_id=1,
        conversation_id=1, message=TICKET, prompt_contract=contract))


class RecordingProvider:
    def __init__(self):
        self.payloads = []

    def key(self):
        return 'offline-test-key'

    @asynccontextmanager
    async def client(self):
        def respond(req):
            payload = json.loads(req.content)
            self.payloads.append(payload)
            reply = 'Support: fixture reply' if 'Support:' in payload['messages'][0]['content'] else 'fixture reply'
            if payload.get('stream'):
                event = dict(model=payload['model'], choices=[dict(delta={'content': reply}, finish_reason=None)])
                return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                    content='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n')
            return httpx.Response(200, json=dict(model=payload['model'],
                choices=[dict(message={'content': reply}, finish_reason='stop')],
                usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5)))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client


async def collect(client, message):
    return ''.join([part async for part in client.stream_tokens(message)])


@pytest.mark.parametrize('values', [V1, V2])
def test_contract_controls_buffered_and_streaming(values, monkeypatch):
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    monkeypatch.setenv('OPENAI_MODEL', 'must-not-use-floating-alias')
    contract = PromptContract.model_validate(values)
    message = request(values).message
    provider = RecordingProvider()
    client = LlmClient(provider)
    buffered = asyncio.run(client.complete_result(message)).reply
    streamed = asyncio.run(collect(client, message))
    assert buffered == streamed
    assert buffered.startswith('Support:') == (contract.version == 'chat:v2')
    for payload in provider.payloads:
        assert payload['messages'] == [{'role': 'user', 'content': render_chat(contract, TICKET)}]
        assert payload['model'] == contract.model_id
        assert payload['temperature'] == contract.temperature
        assert payload['max_tokens'] == contract.max_tokens
    calls = [record for record in records if record['name'] == 'llm.call']
    assert len(calls) == 2
    for record in calls:
        assert record['prompt_version'] == contract.version
        assert record['prompt_fingerprint'] == contract_fingerprint(contract)
        assert record['request_model'] == contract.model_id
        assert record['temperature'] == contract.temperature
        assert record['max_tokens'] == contract.max_tokens
        assert TICKET not in json.dumps(record)


def test_bad_contract_and_legacy_input():
    with pytest.raises(ValidationError):
        request(dict(V1, model_id='gpt-4o'))
    with pytest.raises(ValidationError):
        request(dict(V1, template='missing placeholder'))
    legacy = InferRequest(version='v1', tenant_id=1, conversation_id=1, message=TICKET)
    assert type(legacy.message) is str
    assert legacy.prompt_contract is None


def test_existing_replay_provider_remains_offline(monkeypatch):
    monkeypatch.setattr(observability, 'SINK', lambda record: None)
    client = LlmClient('offline')
    first = asyncio.run(client.complete_result(request(V1).message)).reply
    assert first == asyncio.run(collect(LlmClient('offline'), request(V1).message))


def test_registry_export_demo(monkeypatch):
    source = os.environ.get('ANCHOR_PROMPT_EXPORT')
    if not source:
        return
    contracts = json.loads(Path(source).read_text())
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    for values in contracts:
        contract = PromptContract.model_validate(values)
        provider = RecordingProvider()
        client = LlmClient(provider)
        message = request(values).message
        reply = asyncio.run(client.complete_result(message)).reply
        assert reply == asyncio.run(collect(client, message))
        trace = records[-1]
        print(json.dumps(dict(ticket=TICKET, reply=reply,
            selected_template=contract.template,
            prompt_version=trace['prompt_version'],
            fingerprint=trace['prompt_fingerprint'],
            model_id=trace['request_model'], temperature=trace['temperature'],
            max_tokens=trace['max_tokens'])))
    assert [item['version'] for item in contracts] == ['chat:v1', 'chat:v2']
    print('PASS: same ticket, two pinned contracts; buffered and streaming settings match')
