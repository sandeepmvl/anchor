import asyncio
import copy

import pytest
import yaml

from anchor_ai.contracts import VerifiedExecution
from anchor_ai.runtime import ACTIONS, BoundedRuntime


class OfflineModel:
    reservation = 8

    def __init__(self, bad=False):
        self.bad = bad
        self.plans = []

    async def propose(self, strategy, plan, memory, max_tokens):
        self.plans.append(copy.deepcopy(plan))
        return ('issue_refund' if self.bad else plan[len(memory)]), 2


class OfflineBackend:
    def __init__(self):
        self.refunds = []

    async def call(self, scope, action, arguments, key):
        result = dict(arguments)
        if action == 'check_refund':
            result['eligible'] = True
        if action == 'issue_refund':
            self.refunds.append(key)
            result.update(status='refunded', idempotency_key=key)
        return result


def runtime(tmp_path, *, bad=False, **limits):
    caps = dict(max_steps=8, max_tokens=100, max_calls=8,
                max_cost_units=8, deadline_seconds=60)
    caps.update(limits)
    policy = dict(
        run=caps,
        bounded=dict(strategies={'single-lookup': 'direct',
                                 'multi-step refund': 'planned'},
                     call_timeout_seconds=1),
        tools={action: {'cost_units': 1} for action in ACTIONS},
        tenants={1: dict(refund_ceiling_cents=1000,
                         roles={'agent': ACTIONS, 'refund_reviewer': ACTIONS})})
    policy_path = tmp_path / 'policy.yml'
    policy_path.write_text(yaml.safe_dump(policy))

    async def retrieve(scope, message):
        return dict(tenant_id=scope.tenant_id,
                    hits=[dict(tenant_id=scope.tenant_id, text='Refund policy')])

    def verify(token):
        role = 'refund_reviewer' if token == 'reviewer' else 'agent'
        return VerifiedExecution(1, 42, token, role)

    return BoundedRuntime(
        tmp_path / 'runs.sqlite3', verify=verify, model=OfflineModel(bad),
        retrieve=retrieve, backend=OfflineBackend(), policy_path=policy_path)


def execute(rt, ticket_type='multi-step refund', operation='start', token='agent'):
    return asyncio.run(rt.execute(token, dict(
        run_id='R42', ticket_type=ticket_type, operation=operation,
        message='Disputed charge after outage', order_id='A104', amount_cents=300)))


def assignments(state):
    return [event['worker'] for event in state['events']
            if event['subsystem'] == 'worker_assignment']


def test_refund_routes_to_workers_and_human_with_full_context(tmp_path):
    rt = runtime(tmp_path)
    state = execute(rt)
    assert state['orchestration']['pattern'] == 'supervisor-worker'
    assert assignments(state) == ['technical', 'billing', 'refund', 'refund']
    assert rt.model.plans == [ACTIONS[:i] for i in range(1, 5)]
    assert state['status'] == 'awaiting_approval'
    assert rt.backend.refunds == []
    assert state['caps'] == rt.policy['run']
    handoff = state['escalation']
    for field in ('tenant_id', 'conversation_id', 'memory', 'attempts',
                  'plan', 'phase', 'caps', 'cost_units', 'tokens', 'steps', 'calls',
                  'orchestration'):
        assert handoff[field] == state[field]
    assert handoff['proposed_action'] == state['arguments']
    assert handoff['idempotency_key'] == state['key']
    assert rt.load(rt.identity('agent'), 'R42')['escalation'] == handoff
    execute(rt, operation='approve', token='reviewer')
    completed = execute(rt, operation='resume')
    assert completed['status'] == 'completed'
    assert completed['steps'] == 4
    assert completed['cost_units'] == 4
    assert len(rt.model.plans) == 4
    assert rt.backend.refunds == [state['key']]
    print('Refund escalation -> supervisor-worker -> technical/billing/refund -> human approval')


def test_faq_keeps_existing_single_agent_path(tmp_path):
    rt = runtime(tmp_path)
    state = execute(rt, 'single-lookup')
    assert state['orchestration'] == {'pattern': 'single-agent', 'workers': {}}
    assert state['status'] == 'completed'
    assert state['plan'] == ['retrieve']
    assert assignments(state) == []
    assert rt.model.plans == [['retrieve']]
    assert state['caps'] == rt.policy['run']
    assert state['steps'] == state['calls'] == state['cost_units'] == 1
    print('FAQ -> single-agent; every selected flow inherits shared step and cost budgets')


@pytest.mark.parametrize('ticket_type', ['single-lookup', 'multi-step refund'])
@pytest.mark.parametrize('limit,reason', [
    ('max_steps', 'step_or_token_budget'),
    ('max_cost_units', 'call_or_cost_budget')])
def test_every_flow_enforces_inherited_limits(tmp_path, ticket_type, limit, reason):
    rt = runtime(tmp_path, **{limit: 0})
    state = execute(rt, ticket_type)
    assert state['status'] == 'exhausted'
    assert state['escalation']['reason'] == reason
    assert state['caps'] == rt.policy['run']
    assert state['calls'] == state['cost_units'] == 0
    if limit == 'max_steps':
        assert rt.model.plans == []


@pytest.mark.parametrize('limit,reason,steps', [
    ('max_steps', 'step_or_token_budget', 2),
    ('max_cost_units', 'call_or_cost_budget', 3)])
def test_worker_transition_cannot_refresh_budget(tmp_path, limit, reason, steps):
    rt = runtime(tmp_path, **{limit: 2})
    state = execute(rt)
    assert state['status'] == 'exhausted'
    assert state['escalation']['reason'] == reason
    assert assignments(state)[:2] == ['technical', 'billing']
    assert state['steps'] == steps
    assert state['calls'] == state['cost_units'] == 2
    resumed = execute(rt, operation='resume')
    assert resumed['steps'] == steps
    assert resumed['caps'] == rt.policy['run']


def test_worker_cannot_skip_supervisor_plan(tmp_path):
    rt = runtime(tmp_path, bad=True)
    state = execute(rt)
    assert state['status'] == 'denied'
    assert state['escalation']['reason'] == 'unauthorized_action'
    assert assignments(state) == ['technical']
    assert state['calls'] == 0
    assert rt.backend.refunds == []
