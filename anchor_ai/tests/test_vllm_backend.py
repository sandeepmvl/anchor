import asyncio
import json

import httpx
import pytest

from anchor_ai.llm_client import LlmClient
from anchor_ai.provider import RoutingProvider, configured_provider
from anchor_ai.vllm_backend import (
    FIXTURE_REPLY, VllmBackend, endpoint, release_gate, validate_deployment,
)


def run(awaitable):
    return asyncio.run(awaitable)


def payload(**extra):
    return dict(model='hosted-logical-model',
                messages=[{'role': 'user', 'content': 'Refund timing?'}],
                max_tokens=64, **extra)


async def post(backend, body):
    async with backend.client() as client:
        return await client.post('https://api.openai.com/v1/chat/completions', json=body)


def test_completion_usage_and_default_transport(monkeypatch):
    monkeypatch.setenv('LLM_BACKEND', 'self_hosted')
    monkeypatch.delenv('ANCHOR_VLLM_TRANSPORT', raising=False)
    backend = configured_provider()
    result = run(LlmClient(backend).complete_result('Refund timing?'))
    assert result.reply == FIXTURE_REPLY
    assert result.usage.total_tokens == 36
    assert result.usage.prompt_tokens + result.usage.completion_tokens == 36
    assert backend.server.calls == 1
    assert backend.server.last_payload['model'] == 'tiny-llama'
    assert backend.mode == 'offline'


def test_hosted_selection_unchanged(monkeypatch):
    from anchor_ai.provider import ReplayProvider
    monkeypatch.setenv('LLM_BACKEND', 'hosted')
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.delenv('ANCHOR_MODEL_RELEASE_STATE', raising=False)
    assert isinstance(configured_provider(), ReplayProvider)


def test_structured_reply_uses_existing_validator():
    result = run(LlmClient(VllmBackend()).complete_structured('Refund timing?'))
    assert result.category == 'billing'
    assert result.urgency == 'P3'


def test_tool_calls_are_not_flattened():
    backend = VllmBackend()
    tools = [{'type': 'function', 'function': {'name': 'lookup_order',
              'parameters': {'type': 'object', 'properties': {}}}}]
    response = run(post(backend, payload(tools=tools, tool_choice='auto')))
    choice = response.json()['choices'][0]
    assert choice['finish_reason'] == 'tool_calls'
    assert choice['message']['content'] is None
    call = choice['message']['tool_calls'][0]
    assert call['id'] == 'offline_call_1'
    assert json.loads(call['function']['arguments']) == {'order_id': 'R-42'}
    assert backend.server.last_payload['tools'] == tools


def test_stream_contract_including_usage():
    async def collect():
        backend = VllmBackend()
        async with backend.client() as client:
            async with client.stream('POST', 'https://api.openai.com/v1/chat/completions',
                    json=payload(stream=True, stream_options={'include_usage': True})) as response:
                lines = [line async for line in response.aiter_lines() if line.startswith('data:')]
        assert lines[-1] == 'data: [DONE]'
        events = [json.loads(line[5:]) for line in lines[:-1]]
        assert events[0]['choices'][0]['delta']['content'] == FIXTURE_REPLY
        assert events[1]['choices'][0]['finish_reason'] == 'stop'
        assert events[2]['usage']['total_tokens'] == 36
    run(collect())


def test_bounded_timeout_cancels_without_retry():
    backend = VllmBackend(timeout=0.01)
    backend.server.block = True
    with pytest.raises(httpx.ReadTimeout):
        run(post(backend, payload()))
    assert backend.server.calls == 1
    assert backend.server.cancelled


def test_redacted_diagnostics():
    backend = VllmBackend()
    secret_text = 'customer@example.org secret-token-123'
    body = payload()
    body['messages'][0]['content'] = secret_text
    run(post(backend, body))
    encoded = json.dumps(backend.events)
    assert secret_text not in encoded
    assert 'offline-vllm-key' not in encoded
    assert backend.events[0]['usage']['total_tokens'] == 36
    assert backend.events[0]['backend'] == 'self_hosted'


def test_router_keeps_tenant_approval():
    config = {'order': ['local'], 'sdk_retries': 0, 'threshold': 2,
              'deployments': {'local': {'backend': 'self_hosted', 'model': 'tiny-llama',
                  'region': 'private', 'evaluated': True, 'capabilities': ['chat'],
                  'context_bytes': 10000}},
              'tenants': {'allowed': {'approved': ['local'], 'regions': ['private']}}}
    router = RoutingProvider({}, object(), config=config)
    assert isinstance(router.adapters['local'], VllmBackend)
    assert router.eligible('local', 'allowed', payload())
    assert not router.eligible('local', 'other', payload())
    assert router.adapters['local'].server.calls == 0


@pytest.mark.parametrize('mutation', ['sequence', 'tokens', 'profile', 'release'])
def test_deployment_rejects_batching_drift(tmp_path, mutation):
    from pathlib import Path
    import yaml
    spec = yaml.safe_load(Path('ops/deploy/vllm_service.yml').read_text())
    if mutation in {'sequence', 'tokens'}:
        flag = '--max-num-seqs' if mutation == 'sequence' else '--max-num-batched-tokens'
        index = spec['command'].index(flag) + 1
        spec['command'][index] = str(int(spec['command'][index]) + 1)
    elif mutation == 'profile':
        spec['batching_profile'] = 'unapproved'
    else:
        spec['release']['real_evidence_required'] = False
    path = tmp_path / 'deployment.yml'
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='invalid_vllm_deployment'):
        validate_deployment(path)


def test_deployment_and_release_refusal():
    assert validate_deployment()['artifact']['format'] == 'awq'
    with pytest.raises(ValueError, match='release_evidence_missing'):
        release_gate({'offline_contract_passed': True})
    with pytest.raises(ValueError, match='release_evidence_missing'):
        VllmBackend(production=True)
    with pytest.raises(ValueError):
        endpoint('http://untrusted.example/v1')
