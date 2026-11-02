import asyncio
import json

import pytest

from anchor_ai import observability
from anchor_ai.autonomy import ExecutionRun, load_policy
from anchor_ai.contracts import ToolCall, ToolScope
from anchor_ai.tools import MemoryOrderBackend, dispatch, dispatch_batch, registry_for


def proposal(name='lookup_order', call_id='one', **arguments):
    return ToolCall(call_id=call_id, name=name,
                    arguments=arguments or {'order_id': 'A100'})


def test_committed_policy_distinguishes_authority():
    policy = load_policy()
    assert policy['tools']['lookup_order']['mode'] == 'read'
    assert policy['tools']['issue_refund']['mode'] == 'approval'
    assert 'delete_account' not in policy['tools']
    assert 'issue_refund' not in policy['tenants'][2]['roles']['support_supervisor']
    print('POLICY: lookup_order=read; issue_refund=approval; delete_account=forbidden')


def test_out_of_allowlist_is_redacted_before_backend(monkeypatch):
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    async def scenario():
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=1, conversation_id=7)
        state = ExecutionRun(scope)
        call = proposal('issue_refund', order_id='A100', amount_cents=100,
                        private_note='secret@example.test')
        result = await dispatch(call, registry_for(backend), scope, run=state)
        assert result['payload'] == {'ok': False, 'error': 'policy_denied'}
        assert backend.invocations == []
        assert backend.refunds == []
        assert state.events[-1]['event'] == 'policy_denied'
        assert 'secret@example.test' not in json.dumps(records + state.events + [result])
        print('Can I refund order A100? Anchor: I cannot issue this refund under your role policy.')
        print(json.dumps(state.events[-1]), 'backend_invocations=0')
    asyncio.run(scenario())


@pytest.mark.parametrize('calls,cost,expected,reason', [
    (2, 16, 2, 'call_budget'),
    (8, 1, 1, 'cost_budget'),
])
def test_parallel_atomic_reservations(calls, cost, expected, reason):
    async def scenario():
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=1, conversation_id=7)
        state = ExecutionRun(scope, max_calls=calls, max_cost_units=cost)
        batch = [proposal(call_id=str(i)) for i in range(6)]
        results = await dispatch_batch(batch, registry_for(backend), scope, run=state)
        assert sum(r['payload']['ok'] for r in results) == expected
        assert len(backend.invocations) == expected
        assert state.cost == expected
        assert state.cost <= state.max_cost
        assert state.calls <= state.max_calls
        assert any(r['payload'].get('error') == reason for r in results)
        before = len(backend.invocations)
        denied = await dispatch(proposal(call_id='later'), registry_for(backend), scope)
        assert denied['payload']['error'] == 'run_halted'
        assert len(backend.invocations) == before
        print(reason, 'executed=', before, 'reserved_cost_units=', state.cost,
              'next_function_not_called=True')
    asyncio.run(scenario())


def test_expired_run_cannot_dispatch():
    async def scenario():
        now = [0.0]
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=1, conversation_id=7)
        state = ExecutionRun(scope, clock=lambda: now[0])
        now[0] = 31.0
        results = await dispatch_batch([proposal(call_id=str(i)) for i in range(3)],
                                       registry_for(backend), scope, run=state)
        assert results[0]['payload']['error'] == 'deadline'
        assert all(not r['payload']['ok'] for r in results)
        assert backend.invocations == []
        assert state.cost == state.calls == 0
    asyncio.run(scenario())


def test_scripted_loop_stops_before_next_function():
    async def scenario():
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=1, conversation_id=7)
        state = ExecutionRun(scope, max_calls=2)
        for i in range(6):
            assert await state.step(scope) is None
            result = await dispatch(proposal(call_id=str(i)), registry_for(backend), scope, run=state)
            if not result['payload']['ok']:
                break
        assert i == 2
        assert len(backend.invocations) == 2
        assert state.events[-1]['reason'] == 'call_budget'
        print('SCRIPTED LOOP: stopped before function 3; backend_invocations=2')
    asyncio.run(scenario())


@pytest.mark.parametrize('tenant,order,amount', [(1, 'B999', 100), (1, 'A100', 501)])
def test_ownership_and_amount_rejected_before_backend(tenant, order, amount):
    async def scenario():
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=tenant, conversation_id=7, role='support_supervisor')
        result = await dispatch(proposal('issue_refund', order_id=order, amount_cents=amount),
                                registry_for(backend), scope)
        assert result['payload']['error'] == 'argument_policy'
        assert backend.invocations == []
    asyncio.run(scenario())


def test_scope_cannot_replace_shared_state():
    async def scenario():
        scope = ToolScope(tenant_id=1, conversation_id=7)
        backend = MemoryOrderBackend()
        await dispatch(proposal(), registry_for(backend), scope)
        with pytest.raises(ValueError, match='replacement_forbidden'):
            await dispatch(proposal(), registry_for(backend), scope, run=ExecutionRun(scope))
    asyncio.run(scenario())
