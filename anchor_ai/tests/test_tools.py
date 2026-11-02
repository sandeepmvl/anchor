import asyncio
import json
from copy import deepcopy

import pytest

from anchor_ai import observability as telemetry
from anchor_ai.contracts import ToolCall, ToolScope
from anchor_ai.tools import (
    MemoryOrderBackend, ToolReplayClient, definitions, dispatch_batch,
    normalize, registry_for, result_messages, run_conversation,
)


SCOPE = ToolScope(tenant_id=1, conversation_id=7)


def call(call_id, name='lookup_order', arguments=None):
    return ToolCall(call_id=call_id, name=name,
                    arguments={'order_id': 'A100'} if arguments is None else arguments)


def first_calls():
    return [call('lookup_1'), call('check_1', 'check_refund')]


def test_concurrent_reverse_correlated_redacted(monkeypatch):
    records = []
    monkeypatch.setattr(telemetry, 'SINK', records.append)
    backend = MemoryOrderBackend(reverse=True)

    async def execute():
        with telemetry.span('test.parent') as parent:
            results = await dispatch_batch(first_calls(), registry_for(backend), SCOPE)
            return results, dict(parent.record)

    results, parent = asyncio.run(execute())
    assert backend.completed == ['check_refund', 'lookup_order']
    assert backend.max_active == 2
    by_id = {r['call_id']: r['payload'] for r in results}
    assert by_id['lookup_1'] == {'ok': True, 'data': {
        'order_id': 'A100', 'payment_id': 'pay_A100', 'status': 'delivered',
        'customer_contact': '[REDACTED]'}}
    assert by_id['check_1'] == {'ok': True, 'data': {
        'order_id': 'A100', 'eligible': True, 'reason': 'within_window'}}
    children = [r for r in records if r['name'] == 'python.tool']
    assert len(children) == 2
    assert all(r['trace_id'] == parent['trace_id'] for r in children)
    assert all(r['parent_span_id'] == parent['span_id'] for r in children)
    assert len({r['span_id'] for r in children}) == 2
    assert len({r['tool_call_fingerprint'] for r in children}) == 2
    serialized = json.dumps([results, records])
    assert 'alex@example.test' not in serialized
    assert 'customer_contact' not in json.dumps(records)
    print('Reverse completion: check_refund then lookup_order; both original IDs correct and redacted.')


@pytest.mark.parametrize('name,arguments,code', [
    ('unknown@example.test', '{}', 'unknown_tool'),
    ('lookup_order', '{', 'invalid_json'),
    ('lookup_order', '{"order_id":7}', 'invalid_arguments'),
    ('lookup_order', '{"order_id":" "}', 'invalid_arguments'),
    ('lookup_order', '[]', 'invalid_arguments'),
    ('lookup_order', '{"order_id":"A100","tenant_id":2}', 'invalid_arguments'),
    ('lookup_order', '{"order_id":"A100","scope":{"tenant_id":2}}', 'invalid_arguments'),
])
def test_rejected_before_backend(name, arguments, code, monkeypatch):
    records = []
    monkeypatch.setattr(telemetry, 'SINK', records.append)
    backend = MemoryOrderBackend()
    result = asyncio.run(dispatch_batch(
        [call('bad_1', name, arguments)], registry_for(backend), SCOPE))
    assert result == [{'call_id': 'bad_1', 'payload': {'ok': False, 'error': code}}]
    assert backend.invocations == []
    assert records[0]['error_type'] == code
    assert 'unknown@example.test' not in json.dumps(records)
    assert 'arguments' not in records[0]
    print(code, 'rejected; backend invocations = 0')


def test_tenant_isolation_and_phone_redaction():
    backend = MemoryOrderBackend()
    results = asyncio.run(dispatch_batch(first_calls(), registry_for(backend),
                         ToolScope(tenant_id=2, conversation_id=8)))
    assert results[0]['payload']['data']['payment_id'] == 'pay_A200'
    assert results[0]['payload']['data']['customer_contact'] == '[REDACTED]'
    assert results[1]['payload']['data']['eligible'] is False
    assert '+1 202 555 0199' not in json.dumps(results)
    assert all(invocation[1] == 2 for invocation in backend.invocations)


def test_exception_is_correlated_without_details(monkeypatch):
    records = []
    monkeypatch.setattr(telemetry, 'SINK', records.append)
    backend = MemoryOrderBackend()

    async def broken(**kwargs):
        raise RuntimeError('secret@example.test database credentials')

    backend.lookup_order = broken
    result = asyncio.run(dispatch_batch([call('exception_1')], registry_for(backend), SCOPE))
    assert result == [{'call_id': 'exception_1',
                       'payload': {'ok': False, 'error': 'execution_error'}}]
    assert 'secret@example.test' not in json.dumps([records, result])
    assert 'credentials' not in json.dumps([records, result])


def test_invalid_backend_result_is_not_exposed():
    backend = MemoryOrderBackend()

    async def invalid(**kwargs):
        return {'order_id': 'A100', 'payment_id': 'pay_A100', 'status': 'delivered',
                'customer_contact': 'alex@example.test', 'extra_secret': 'private'}

    backend.lookup_order = invalid
    result = asyncio.run(dispatch_batch([call('result_1')], registry_for(backend), SCOPE))
    assert result[0]['payload'] == {'ok': False, 'error': 'execution_error'}


def test_three_provider_envelopes():
    arguments = '{"order_id":"A100"}'
    envelopes = {
        'chat': {'tool_calls': [{'id': 'lookup_1', 'type': 'function',
                 'function': {'name': 'lookup_order', 'arguments': arguments}}]},
        'responses': {'output': [{'type': 'reasoning', 'summary': []},
                      {'type': 'function_call', 'call_id': 'lookup_1',
                       'name': 'lookup_order', 'arguments': arguments}]},
        'anthropic': {'content': [{'type': 'text', 'text': 'Checking.'},
                      {'type': 'tool_use', 'id': 'lookup_1', 'name': 'lookup_order',
                       'input': json.loads(arguments)}]},
    }
    for api, envelope in envelopes.items():
        normalized = normalize(api, envelope)
        assert normalized[0].call_id == 'lookup_1'
        assert normalized[0].name == 'lookup_order'
        raw = normalized[0].arguments
        assert (json.loads(raw) if isinstance(raw, str) else raw) == json.loads(arguments)
        registry = registry_for(MemoryOrderBackend())
        results = asyncio.run(dispatch_batch(normalized, registry, SCOPE))
        messages = result_messages(api, results)
        if api == 'chat':
            assert messages[0]['tool_call_id'] == 'lookup_1'
        elif api == 'responses':
            assert messages[0]['call_id'] == 'lookup_1'
        else:
            assert messages[0]['content'][0]['tool_use_id'] == 'lookup_1'
        assert len(definitions(registry, api)) == 2


def test_history_chain_errors_and_correction(monkeypatch):
    monkeypatch.setenv('ANCHOR_LLM_PROVIDER', 'fake')
    client = ToolReplayClient()
    backend = MemoryOrderBackend(reverse=True)
    history, batches = asyncio.run(run_conversation(client, registry_for(backend), SCOPE))
    assert client.position == 5
    assert [len(batch) for batch in batches] == [2, 1, 5, 1]
    assert batches[1][0]['call_id'] == 'chain_1'
    assert batches[1][0]['payload']['data']['order_id'] == batches[0][0]['payload']['data']['order_id']
    assert [r['payload']['error'] for r in batches[2]] == [
        'unknown_tool', 'invalid_json', 'invalid_arguments', 'invalid_arguments', 'execution_error']
    assert batches[3][0]['payload']['ok'] is True
    assert len(backend.invocations) == 5
    assert 'eligible' in history[-1]['content']
    assert 'alex@example.test' not in json.dumps(history)
    assert '[REDACTED]' in json.dumps(history)
    assert history[1]['tool_calls'][0]['id'] == 'lookup_1'


def test_missing_result_rejected_by_fake(monkeypatch):
    monkeypatch.setenv('ANCHOR_LLM_PROVIDER', 'fake')
    client = ToolReplayClient()
    registry = registry_for(MemoryOrderBackend())
    history = [{'role': 'user', 'content': 'Can I refund order A100?'}]
    message = client.complete(history, system=definitions(registry))
    history.append(deepcopy(message))
    results = asyncio.run(dispatch_batch(normalize('chat', message), registry, SCOPE))
    history.extend(result_messages('chat', results)[:1])
    with pytest.raises(AssertionError, match='incomplete batch'):
        client.complete(history, system=definitions(registry))
