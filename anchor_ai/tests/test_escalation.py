import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from anchor_ai.autonomy import ExecutionRun, guarded_dispatch, load_policy
from anchor_ai.contracts import RefundArguments, ReviewIdentity, ToolCall, ToolScope
from anchor_ai.escalation import EscalationService, context_bundle, risk_rule


class OfflineRefundBackend:
    """Offline adapter with durable provider-style key deduplication."""
    def __init__(self, path):
        self.path = path
        self.invocations = []
        with sqlite3.connect(path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS fake_refunds
                (action_key TEXT PRIMARY KEY, tenant_id INTEGER NOT NULL, arguments TEXT NOT NULL)''')

    async def execute(self, call, registry, scope, *, idempotency_key):
        assert idempotency_key
        self.invocations.append(idempotency_key)
        arguments = RefundArguments.model_validate(call.arguments).model_dump()
        encoded = json.dumps(arguments, sort_keys=True)
        with sqlite3.connect(self.path) as db:
            db.execute('INSERT OR IGNORE INTO fake_refunds VALUES(?,?,?)',
                       (idempotency_key, scope.tenant_id, encoded))
            saved = db.execute('SELECT tenant_id,arguments FROM fake_refunds WHERE action_key=?',
                               (idempotency_key,)).fetchone()
            assert saved == (scope.tenant_id, encoded)
        return {'call_id': call.call_id, 'payload': {'ok': True, 'data': {
            **arguments, 'status': 'refunded'}}}

    def count(self):
        with sqlite3.connect(self.path) as db:
            return db.execute('SELECT COUNT(*) FROM fake_refunds').fetchone()[0]


def scenario(path, conversation_id=1):
    # Reuse the existing configured approval tool, role, and tenant limit.
    policy = load_policy()
    selected = None
    for tenant_id, tenant in policy['tenants'].items():
        for role, names in tenant.get('roles', {}).items():
            for name in names:
                if policy['tools'][name]['mode'] == 'approval' and 'refund' in name:
                    selected = (tenant_id, tenant, role, name)
                    break
            if selected:
                break
        if selected:
            break
    assert selected is not None, 'existing refund policy must be present'
    tenant_id, tenant, role, name = selected
    scope = ToolScope(tenant_id=tenant_id, conversation_id=conversation_id, role=role)
    service = EscalationService(path)
    backend = OfflineRefundBackend(path)
    registry = {name: SimpleNamespace(arguments=RefundArguments, consequential=True,
                authorize=lambda trusted, args: trusted.tenant_id == tenant_id and args['order_id'] == 'A001')}
    call = ToolCall(call_id='refund-' + str(conversation_id), name=name, arguments={
        'order_id': 'A001', 'amount_cents': tenant['refund_ceiling_cents'] + 1})
    state = ExecutionRun(scope)
    state.escalation = service
    state.review_context = context_bundle(
        [{'role': 'user', 'content': 'Refund order A001. Contact alice@example.com.'},
         {'role': 'assistant', 'tool_calls': [{'id': call.call_id, 'name': name, 'arguments': call.arguments}]}],
        [{'id': 'refund-policy-v1', 'text': 'Eligible order; amount exceeds automatic authority.'}],
        {'strategy': 'tool_loop', 'iteration': 2, 'next_step': 'await_tool_result'},
        {'calls': state.max_calls, 'cost_units': state.max_cost, 'steps': state.max_steps},
        'Policy matched; review required for amount.')
    reviewer = ReviewIdentity(tenant_id=tenant_id, reviewer_id='reviewer-1', role='refund_reviewer')
    return service, backend, registry, call, scope, state, reviewer


async def submit(parts):
    service, backend, registry, call, scope, state, reviewer = parts
    result = await guarded_dispatch(call, registry, scope, backend.execute, state)
    assert result['payload']['error'] == 'review_required'
    return result['payload']['checkpoint_id']


def test_over_limit_persists_redacted_full_context_then_pauses(tmp_path):
    parts = scenario(tmp_path / 'review.sqlite3')
    service, backend, registry, call, scope, state, reviewer = parts
    checkpoint = asyncio.run(submit(parts))
    record = service.inspect(checkpoint, reviewer)
    context = record['context']
    assert record['status'] == 'pending'
    assert state.paused == checkpoint and not state.halted
    assert backend.count() == 0 and backend.invocations == []
    assert context['pending_tool_call']['arguments'] == call.arguments
    assert context['pending_tool_call']['call_id'] == call.call_id
    assert context['risk']['policy_reason'] == 'over_automatic_limit'
    assert context['risk']['score'] == 2 and context['risk']['threshold'] == 1
    assert context['loop']['evidence'][0]['id'] == 'refund-policy-v1'
    assert context['loop']['plan_position']['next_step'] == 'await_tool_result'
    assert context['remaining_budget']['calls'] == state.max_calls
    assert context['run']['steps'] == state.steps
    with sqlite3.connect(service.path) as db:
        stored = db.execute('SELECT context_json FROM escalations').fetchone()[0]
    assert 'alice@example.com' not in stored
    assert 'REDACTED' in stored
    again = asyncio.run(guarded_dispatch(call, registry, scope, backend.execute, state))
    assert again['payload']['error'] == 'review_required'
    assert backend.count() == 0


def test_only_authorized_same_tenant_reviewer_can_resolve_and_rejection_never_executes(tmp_path):
    parts = scenario(tmp_path / 'review.sqlite3')
    service, backend, registry, call, scope, state, reviewer = parts
    checkpoint = asyncio.run(submit(parts))
    for identity in (
        reviewer.model_copy(update={'tenant_id': reviewer.tenant_id + 1000}),
        reviewer.model_copy(update={'role': 'support_reader'}),
    ):
        with pytest.raises(PermissionError):
            asyncio.run(service.resolve(checkpoint, identity, 'approve', registry=registry, execute=backend.execute))
        with pytest.raises(PermissionError):
            service.inspect(checkpoint, identity)
        assert service.inspect(checkpoint, reviewer)['status'] == 'pending'
    result = asyncio.run(service.resolve(checkpoint, reviewer, 'reject', reason='Proof of purchase is missing.'))
    assert result['status'] == 'rejected'
    assert result['result']['do_not_retry']
    assert result['outcome']['human_handoff']
    assert result['outcome']['next_step'] == 'done'
    assert result['outcome']['pending_tool_call'] is None
    assert result['outcome']['messages'][-2]['tool_call_id'] == call.call_id
    assert 'not issued' in result['outcome']['reply']
    repeated = asyncio.run(service.resolve(checkpoint, reviewer, 'approve', registry=registry, execute=backend.execute))
    assert repeated['deduplicated'] and repeated['status'] == 'rejected'
    assert backend.count() == 0 and backend.invocations == []


def test_duplicate_approval_resumes_once_with_original_idempotency_key(tmp_path):
    parts = scenario(tmp_path / 'review.sqlite3')
    service, backend, registry, call, scope, state, reviewer = parts
    checkpoint = asyncio.run(submit(parts))
    key = service.inspect(checkpoint, reviewer)['action_key']
    # A new service object proves resume does not depend on the paused run object.
    resumed = EscalationService(service.path)
    first = asyncio.run(resumed.resolve(checkpoint, reviewer, 'approve', registry=registry, execute=backend.execute))
    second = asyncio.run(resumed.resolve(checkpoint, reviewer, 'approve', registry=registry, execute=backend.execute))
    assert first['status'] == second['status'] == 'completed'
    assert first['outcome'] == second['outcome']
    assert first['result']['ok']
    assert not first['deduplicated'] and second['deduplicated']
    assert backend.invocations == [key]
    assert backend.count() == 1


def test_rule_tiers_and_expired_approval_do_not_grant_budget(tmp_path):
    assert risk_rule(100, 100, True)['score'] == 0
    assert risk_rule(101, 100, True)['score'] == 2
    assert risk_rule(100, 100, False)['score'] == 2
    assert risk_rule(101, 100, False)['score'] == 4
    parts = scenario(tmp_path / 'review.sqlite3')
    service, backend, registry, call, scope, state, reviewer = parts
    checkpoint = asyncio.run(submit(parts))
    with sqlite3.connect(service.path) as db:
        raw = json.loads(db.execute('SELECT context_json FROM escalations WHERE id=?', (checkpoint,)).fetchone()[0])
        raw['run']['expires_at'] = 0
        db.execute('UPDATE escalations SET context_json=? WHERE id=?', (json.dumps(raw), checkpoint))
    result = asyncio.run(service.resolve(checkpoint, reviewer, 'approve', registry=registry, execute=backend.execute))
    assert result['result']['error'] == 'deadline'
    assert backend.count() == 0 and backend.invocations == []
