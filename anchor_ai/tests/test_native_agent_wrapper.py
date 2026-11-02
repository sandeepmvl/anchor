import asyncio
import json

import pytest

from anchor_ai.contracts import ToolScope
from anchor_ai.native_agent_wrapper import NativeAgent
from anchor_ai.orchestration import run_native_support_turn
from anchor_ai.provider import NativeReplayProvider, configured_native_provider
from anchor_ai.storage import NativeSessionStore
from anchor_ai.tools import MemoryOrderBackend, registry_for
import anchor_ai.native_agent_wrapper as wrapper


FIRST = 'For ticket 456, look up order A100.'
FOLLOWUP = 'Which order did we discuss, and what was its status?'


@pytest.fixture
def setup(tmp_path):
    provider = NativeReplayProvider()
    backend = MemoryOrderBackend()
    store = NativeSessionStore(tmp_path / 'sessions.sqlite')
    scope = ToolScope(tenant_id=1, conversation_id=456)
    agent = NativeAgent(provider, registry_for(backend), store)
    yield provider, backend, store, scope, agent
    store.close()


def test_two_turns_use_opaque_provider_state_without_resending_history(setup):
    provider, backend, store, scope, agent = setup

    async def scenario():
        first = await run_native_support_turn(agent, scope, FIRST, enabled=True)
        second = await run_native_support_turn(
            agent, scope, FOLLOWUP, enabled=True,
            previous_response_id=first['response_id'])
        assert second['response_id'] != first['response_id']
        assert 'A100' in second['answer'] and 'delivered' in second['answer']
        assert provider.requests[-1]['previous_response_id'] == first['response_id']
        assert provider.requests[-1]['input'] == [{'role': 'user', 'content': FOLLOWUP}]
        assert 'messages' not in provider.requests[-1]
        assert provider.requests[1]['input'][0]['type'] == 'function_call_output'
        result = json.loads(provider.requests[1]['input'][0]['output'])
        assert result['ok'] is True
        assert result['data']['customer_contact'] == '[REDACTED]'
        assert 'alex@example.test' not in json.dumps(provider.requests)
        assert {t['name'] for t in provider.requests[0]['tools']} == {
            'lookup_order', 'check_refund', 'issue_refund'}
        assert all(t['type'] == 'function' for t in provider.requests[0]['tools'])
        assert len(backend.invocations) == 1
        history = provider.responses[second['response_id']]['history']
        assert any(i.get('type') == 'function_call_output' for i in history)
        columns = [r[1] for r in store.db.execute('PRAGMA table_info(native_response_bindings)')]
        assert columns == ['provider', 'response_id', 'tenant_id', 'conversation_id']
        assert any(e['event'] == 'native.guarded_tool' and e['ok'] for e in agent.events)
        assert all(e['tenant_id'] == 1 and e['conversation_id'] == 456 for e in agent.events)

    asyncio.run(scenario())


def test_cross_tenant_and_cross_conversation_reuse_rejected_before_provider(setup):
    provider, backend, store, scope, agent = setup

    async def scenario():
        first = await agent.turn(scope, FIRST)
        count = len(provider.requests)
        for other in (ToolScope(tenant_id=2, conversation_id=456),
                      ToolScope(tenant_id=1, conversation_id=457)):
            with pytest.raises(ValueError, match='native_session_scope_mismatch'):
                await agent.turn(other, FOLLOWUP, previous_response_id=first['response_id'])
        assert len(provider.requests) == count
        with pytest.raises(ValueError, match='native_session_scope_mismatch'):
            await agent.turn(scope, FOLLOWUP, previous_response_id='untrusted_id')
        assert len(provider.requests) == count

    asyncio.run(scenario())


def test_duplicate_effect_rejected_even_with_new_call_id_and_reopened_store(setup, tmp_path, monkeypatch):
    provider, backend, store, scope, agent = setup
    call = provider.records['initial_output'][0]
    call.update(name='issue_refund', arguments='{"order_id":"A100","amount_cents":100}')
    effects = []

    async def approved_dispatch(calls, registry, authenticated_scope, *, run=None):
        # A test double for an already approved local dispatcher, not a production bypass.
        effects.append((authenticated_scope.tenant_id, calls[0].name))
        return [{'call_id': calls[0].call_id, 'payload': {'ok': True, 'data': {}}}]

    monkeypatch.setattr(wrapper, 'dispatch_batch', approved_dispatch)

    async def scenario():
        await agent.turn(scope, FIRST)
        reopened = NativeSessionStore(tmp_path / 'sessions.sqlite')
        try:
            provider.records['initial_output'][0]['call_id'] = 'different_provider_call_id'
            another = NativeAgent(provider, registry_for(backend), reopened)
            with pytest.raises(ValueError, match='native_duplicate_effect'):
                await another.turn(scope, FIRST)
        finally:
            reopened.close()
        assert effects == [(1, 'issue_refund')]

    asyncio.run(scenario())


def test_real_guard_rejects_unapproved_refund(setup):
    provider, backend, store, scope, agent = setup
    provider.records['initial_output'][0].update(
        name='issue_refund', arguments='{"order_id":"A100","amount_cents":100}')
    result = asyncio.run(agent.turn(scope, FIRST))
    assert result['blocked'] is True
    assert backend.refunds == []
    assert not any(i[0] == 'issue_refund' for i in backend.invocations)
    assert len(provider.requests) == 1
    assert agent.events[-1]['event'] == 'native.guarded_tool'
    assert agent.events[-1]['ok'] is False


def test_unsupported_capabilities_and_provider_managed_tools_rejected(setup):
    provider, backend, store, scope, agent = setup
    with pytest.raises(ValueError, match='unsupported_native_capability'):
        NativeAgent(provider, registry_for(backend), store, capabilities={'web_search'})
    provider.capabilities = frozenset({'responses'})
    with pytest.raises(ValueError, match='unsupported_native_capability'):
        NativeAgent(provider, registry_for(backend), store)
    provider.records['initial_output'] = [{'type': 'web_search_call', 'id': 'search_1'}]
    with pytest.raises(ValueError, match='unsupported_native_capability'):
        asyncio.run(agent.turn(scope, FIRST))
    assert backend.invocations == []


def test_deleted_provider_state_is_not_invented(setup):
    provider, backend, store, scope, agent = setup

    async def scenario():
        first = await agent.turn(scope, FIRST)
        provider.responses.clear()
        with pytest.raises(ValueError, match='native_response_unavailable'):
            await agent.turn(scope, FOLLOWUP, previous_response_id=first['response_id'])
        assert len(backend.invocations) == 1

    asyncio.run(scenario())


def test_native_path_requires_opt_in_and_defaults_offline(setup, monkeypatch):
    provider, backend, store, scope, agent = setup
    monkeypatch.delenv('PROVIDER', raising=False)
    assert isinstance(configured_native_provider(), NativeReplayProvider)
    with pytest.raises(ValueError, match='native_agent_requires_opt_in'):
        asyncio.run(run_native_support_turn(agent, scope, FIRST))
    assert provider.requests == []
