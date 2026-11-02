import asyncio
import copy
from pathlib import Path

import pytest
import yaml

from anchor_ai import observability
from anchor_ai.agent_interop import InteropAdapter, agent_card
from anchor_ai.contracts import VerifiedExecution
from anchor_ai.harness import HarnessAdapter
from anchor_ai.runtime import ACTIONS, BoundedRuntime, ContextError


class RecordedModel:
    reservation = 8

    async def propose(self, strategy, plan, memory, max_tokens):
        return plan[len(memory)], 2


class OfflineBackend:
    """No money moves. Production financial backends must durably deduplicate."""
    def __init__(self):
        self.calls = []
        self.receipts = {}

    async def call(self, scope, action, arguments, key):
        self.calls.append(action)
        result = dict(arguments)
        if action == 'check_refund':
            result['eligible'] = True
        if action == 'issue_refund':
            result.update(status='refunded', idempotency_key=key)
            result = self.receipts.setdefault(key, result)
        return result


async def offline_retrieve(scope, message):
    return {'tenant_id': scope.tenant_id, 'hits': [
        {'tenant_id': scope.tenant_id, 'text': 'Refund policy evidence'}]}


def make_runtime(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    policy = {
        'tenants': {1: {'roles': {'support_reader': ACTIONS,
                                'refund_reviewer': ACTIONS},
                        'refund_ceiling_cents': 10000}},
        'tools': {name: {'cost_units': 1} for name in ACTIONS},
        'run': {'deadline_seconds': 30, 'max_steps': 8, 'max_tokens': 64,
                'max_calls': 4, 'max_cost_units': 4},
        'bounded': {'call_timeout_seconds': 1, 'strategies': {
            'single-lookup': 'recorded', 'multi-step refund': 'recorded'}}}
    policy_path = directory / 'offline_policy.yml'
    policy_path.write_text(yaml.safe_dump(policy))

    def verify(token):
        identities = {
            'reader': VerifiedExecution(1, 42, 'agent', 'support_reader'),
            'reviewer': VerifiedExecution(1, 42, 'human', 'refund_reviewer')}
        if token not in identities:
            raise ContextError('unverified_execution_context')
        return identities[token]

    return BoundedRuntime(directory / 'runs.sqlite3', verify=verify,
                          model=RecordedModel(), retrieve=offline_retrieve,
                          backend=OfflineBackend(), policy_path=policy_path)


def request(run_id='refund42'):
    return {'run_id': run_id, 'message': 'Refund order A104',
            'order_id': 'A104', 'amount_cents': 300}


def task(run_id='refund42'):
    return {'protocol': 'anchor-a2a-loopback-v1', 'task_id': run_id,
            'tenant_id': 1, 'conversation_id': 42,
            'skill': 'refund-resolution', 'request': request(run_id)}


def run(awaitable):
    return asyncio.run(awaitable)


def test_local_loop_approval_and_trace(tmp_path, monkeypatch):
    spans = []
    monkeypatch.setattr(observability, 'SINK', spans.append)
    runtime = make_runtime(tmp_path)
    adapter = HarnessAdapter(runtime)
    state = run(adapter.run('reader', request()))
    assert state['status'] == 'awaiting_approval'
    assert runtime.backend.calls == ['lookup_order', 'check_refund']
    assert len(state['escalation']['memory']) == 3
    assert state['escalation']['proposed_action']['amount_cents'] == 300
    run(adapter.run('reviewer', {'run_id': 'refund42', 'operation': 'approve'}))
    state = run(adapter.run('reader', {'run_id': 'refund42', 'operation': 'resume'}))
    assert state['status'] == 'completed'
    assert (state['steps'], state['tokens'], state['calls'], state['cost_units']) == (4, 8, 4, 4)
    allowed = set()
    for event in state['events']:
        if event['subsystem'] == 'permission_decision' and event['decision'] == 'allow':
            allowed.add(event['action'])
        if event['subsystem'] == 'tool_dispatch':
            assert event['action'] in allowed
    model_spans = [s for s in spans if s['name'] == 'python.bounded_agent.model']
    tool_spans = [s for s in spans if s['name'] == 'python.bounded_agent.tool']
    root = next(s for s in spans if s['name'] == 'python.bounded_agent')
    assert model_spans and tool_spans
    roots = {s['span_id']: s for s in spans if s['name'] == 'python.bounded_agent'}
    for child in model_spans + tool_spans:
        assert child['parent_span_id'] in roots
        assert child['trace_id'] == roots[child['parent_span_id']]['trace_id']
    assert runtime.backend.calls.count('issue_refund') == 1


@pytest.mark.parametrize('fault', ['exception', 'timeout', 'truthy'])
def test_hook_fails_closed(tmp_path, fault):
    runtime = make_runtime(tmp_path)

    async def hook(*args):
        if fault == 'exception':
            raise RuntimeError('unavailable')
        if fault == 'timeout':
            await asyncio.sleep(1)
        return 'allow'

    state = run(HarnessAdapter(runtime, hook=hook, timeout=0.01).run('reader', request()))
    assert state['status'] == 'denied'
    assert state['calls'] == 0 and runtime.backend.calls == []
    permission = next(e for e in state['events'] if e['subsystem'] == 'permission_decision')
    assert permission['decision'] == 'deny'
    assert permission['reason'] == ('policy_denied' if fault == 'truthy' else 'policy_error')


def test_shared_budget_and_authorization(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.policy['run']['max_calls'] = 1
    state = run(HarnessAdapter(runtime).run('reader', request()))
    assert state['status'] == 'exhausted' and state['calls'] == 1
    assert runtime.backend.calls == []
    runtime = make_runtime(tmp_path / 'denied')
    runtime.policy['tenants'][1]['roles']['support_reader'] = []
    state = run(HarnessAdapter(runtime, hook=lambda *args: True).run('reader', request()))
    assert state['status'] == 'denied' and state['calls'] == 0


def test_interop_scope_and_protocol(tmp_path):
    runtime = make_runtime(tmp_path)
    interop = InteropAdapter(HarnessAdapter(runtime))
    for token in ('', 'unknown'):
        with pytest.raises(ContextError):
            run(interop.delegate(token, task()))
    for field in ('tenant_id', 'conversation_id'):
        envelope = task()
        envelope[field] = 99
        with pytest.raises(ContextError, match='delegation_scope_mismatch'):
            run(interop.delegate('reader', envelope))
    with pytest.raises(ValueError, match='mcp_result_is_not_a2a_task'):
        run(interop.delegate('reader', {'content': [{'type': 'text', 'text': 'done'}],
                                      'isError': False}))
    assert runtime.backend.calls == []
    result = run(interop.delegate('reader', task()))
    assert (result['tenant_id'], result['conversation_id'], result['task_id']) == (1, 42, 'refund42')
    assert result['status'] == 'awaiting_approval'
    assert result['trace_id'] == runtime.load(runtime.identity('reader'), 'refund42')['trace_id']
    envelope = copy.deepcopy(task('review_attempt'))
    envelope['request']['operation'] = 'approve'
    with pytest.raises(ValueError, match='delegation_cannot_review'):
        run(interop.delegate('reader', envelope))


def test_card_is_explicit_stub():
    card = agent_card('http://127.0.0.1:8000')
    assert card['skills'][0]['id'] == 'refund-resolution'
    assert card['capabilities'] == {}
    assert 'not implemented' in card['description']
