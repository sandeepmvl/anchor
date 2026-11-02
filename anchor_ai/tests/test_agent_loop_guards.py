import asyncio
import json
from pathlib import Path

import pytest

from anchor_ai import observability
from anchor_ai.agentic_rag import ABSTENTION, AgenticRuntime, RecordedRetrievalBoundary
from anchor_ai.schemas import DenseRagResponse, normalize_retrieval_query

FIXTURE = Path(__file__).parent / 'fixtures/repeated_retrieval_responses.json'


class Policy:
    def __init__(self, queries, final=None):
        self.queries = queries
        self.final = final
        self.calls = 0

    async def decide(self, question, evidence):
        index = self.calls
        self.calls += 1
        if index >= len(self.queries) and self.final is not None:
            return {'kind': 'sufficient', 'answer': self.final}
        query = self.queries[index % len(self.queries)]
        return {'kind': 'retrieve', 'tool': 'retrieve', 'arguments': {'query': query}}


@pytest.fixture
def recording():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(autouse=True)
def spans(monkeypatch):
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    return records


def run(recording, scenario, *, final=None, history=(), **limits):
    retriever = RecordedRetrievalBoundary(recording)
    policy = Policy(recording['trajectories'][scenario], final)
    events = []
    runtime = AgenticRuntime(retriever, policy=policy, emit=events.append,
                            generation_id=recording['generation_id'], **limits)
    answer = asyncio.run(runtime.answer(recording['question'], request_id='guard-test',
                                       trace_id='a' * 32,
                                       conversation_seen_ids=history))
    DenseRagResponse.model_validate(answer)
    return runtime, retriever, events, answer


def assert_safe(answer):
    assert answer == {'reply': ABSTENTION, 'citations': [], 'abstained': True}


def test_normalized_duplicate_query_stops_before_retrieval(recording):
    runtime, retriever, events, answer = run(recording, 'duplicate_query')
    assert normalize_retrieval_query('  REFUND   POLICY  ') == 'refund policy'
    assert len(retriever.calls) == 1 < runtime.max_tool_calls
    assert events[-1]['exit'] == 'duplicate_query'
    assert runtime.last_trajectory[-1]['duplicate_query'] is True
    assert_safe(answer)


def test_rephrasing_with_unchanged_evidence_stops_early(recording):
    runtime, retriever, events, answer = run(recording, 'repeated_evidence')
    assert len(retriever.calls) == 2 < runtime.max_tool_calls
    assert events[-1]['exit'] == 'no_new_evidence'
    assert runtime.last_trajectory[-1]['new_chunk_ids'] == []
    assert events[-1]['seen_chunk_ids'] == [101, 102]
    assert_safe(answer)


def test_distinct_productive_searches_can_answer(recording):
    expected = {'reply': 'Support hours appear on the contact page.',
                'citations': [103], 'abstained': False}
    runtime, retriever, events, answer = run(recording, 'productive', final=expected)
    assert len(retriever.calls) == 4
    assert all(row['new_chunk_ids'] for row in runtime.last_trajectory[:-1])
    assert events[-1]['exit'] == 'evidence_sufficient'
    assert answer == expected


@pytest.mark.parametrize('limits, expected_calls, reason', [
    ({'max_hops': 2}, 2, 'max_hops'),
    ({'max_tool_calls': 2}, 2, 'tool_budget'),
    ({'max_tool_calls': 0}, 0, 'tool_budget'),
    ({'max_iterations': 2}, 2, 'iteration_budget'),
    ({}, 4, 'tool_budget'),
])
def test_hard_budgets_still_stop_productive_searches(recording, limits, expected_calls, reason):
    _, retriever, events, answer = run(recording, 'productive', **limits)
    assert len(retriever.calls) == expected_calls
    assert events[-1]['exit'] == reason
    assert_safe(answer)


def test_alternating_evidence_uses_cumulative_seen_ids(recording):
    recording['trajectories']['alternating'][-1] = 'reworded contact search'
    recording['searches']['reworded contact search'] = [101]
    _, retriever, events, answer = run(recording, 'alternating')
    assert len(retriever.calls) == 3
    assert events[-1]['exit'] == 'no_new_evidence'
    assert_safe(answer)


def test_trusted_prior_conversation_ids_short_circuit(recording):
    _, retriever, events, answer = run(recording, 'repeated_evidence', history=(101, 102))
    assert len(retriever.calls) == 1
    assert events[-1]['exit'] == 'no_new_evidence'
    assert_safe(answer)


def test_empty_retrieval_stops(recording):
    _, retriever, events, answer = run(recording, 'empty')
    assert len(retriever.calls) == 1
    assert events[-1]['exit'] == 'no_new_evidence'
    assert_safe(answer)


def test_seen_ids_do_not_depend_on_evidence_budget(recording):
    _, retriever, events, answer = run(recording, 'repeated_evidence', evidence_budget=1)
    assert len(retriever.calls) == 2
    assert events[-1]['evidence_chunk_ids'] == []
    assert events[-1]['seen_chunk_ids'] == [101, 102]
    assert events[-1]['exit'] == 'no_new_evidence'
    assert_safe(answer)


def test_spans_link_model_tool_and_retrieval_without_raw_queries(recording, spans):
    run(recording, 'repeated_evidence')
    root = next(row for row in spans if row['name'] == 'python.agent')
    tools = [row for row in spans if row['name'] == 'python.agent.tool']
    retrievals = [row for row in spans if row['name'] == 'python.agent.retrieve']
    models = [row for row in spans if row['name'] == 'python.agent.model']
    assert len(tools) == len(retrievals) == len(models) == 2
    assert all(row['trace_id'] == 'a' * 32 for row in spans)
    assert all(row['parent_span_id'] == root['span_id'] for row in tools + models)
    assert {row['parent_span_id'] for row in retrievals} == {row['span_id'] for row in tools}
    assert tools[-1]['new_chunk_ids'] == []
    assert root['agent_exit'] == 'no_new_evidence'
    assert len(tools[0]['query_fingerprint']) == 64
    assert 'refund policy' not in json.dumps(spans)


from anchor_ai.autonomy import ExecutionRun
from anchor_ai.contracts import ToolCall, ToolScope
from anchor_ai.tools import MemoryOrderBackend, dispatch, dispatch_batch, registry_for


def refund_call(amount=100, call_id='refund'):
    return ToolCall(call_id=call_id, name='issue_refund',
                    arguments={'order_id': 'A100', 'amount_cents': amount})


def approve_refund(state, scope, call, key='refund-intent-001'):
    token = state.confirm(scope, call, call.arguments)
    state.approve(scope, call, call.arguments, confirmation=token, idempotency_key=key)
    return token


def test_consequential_tool_requires_confirmation():
    async def scenario():
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=1, conversation_id=7, role='support_supervisor')
        result = await dispatch(refund_call(), registry_for(backend), scope)
        assert result['payload']['error'] == 'confirmation_required'
        assert backend.invocations == backend.refunds == []
    asyncio.run(scenario())


def test_confirmation_is_argument_bound_and_key_required():
    async def scenario():
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=1, conversation_id=7, role='support_supervisor')
        state = ExecutionRun(scope)
        call = refund_call()
        token = state.confirm(scope, call, call.arguments)
        with pytest.raises(ValueError, match='invalid_confirmation'):
            state.approve(scope, call, call.arguments, confirmation=token, idempotency_key='')
        with pytest.raises(ValueError, match='invalid_confirmation'):
            changed = refund_call(101)
            state.approve(scope, changed, changed.arguments,
                          confirmation=token, idempotency_key='refund-intent-001')
        approve_refund(state, scope, call)
        result = await dispatch(refund_call(101), registry_for(backend), scope, run=state)
        assert result['payload']['error'] == 'confirmation_required'
        assert backend.invocations == backend.refunds == []
    asyncio.run(scenario())


def test_confirmed_parallel_retries_are_idempotent():
    async def scenario():
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=1, conversation_id=7, role='support_supervisor')
        state = ExecutionRun(scope)
        approve_refund(state, scope, refund_call())
        results = await dispatch_batch([refund_call(call_id='a'), refund_call(call_id='b')],
                                       registry_for(backend), scope, run=state)
        assert all(r['payload']['ok'] for r in results)
        assert len(backend.refunds) == len(backend.invocations) == 1
        assert state.cost == 6
    asyncio.run(scenario())


def test_refund_dry_run_has_no_mutation():
    async def scenario():
        backend = MemoryOrderBackend()
        scope = ToolScope(tenant_id=1, conversation_id=7, role='support_supervisor')
        state = ExecutionRun(scope, dry_run=True)
        result = await dispatch(refund_call(), registry_for(backend), scope, run=state)
        assert result['payload']['ok'] is True
        assert result['payload']['data'] == {
            'dry_run': True, 'proposed_action': 'issue_refund',
            'arguments': {'order_id': 'A100', 'amount_cents': 100}}
        assert backend.invocations == backend.refunds == []
        assert backend.idempotency == {}
        print('Can I refund order A100? Anchor: Proposed refund of 100 cents; dry-run only.')
        print(json.dumps(result['payload']), 'financial_mutations=0')
    asyncio.run(scenario())


def test_retrieval_uses_shared_execution_budget(recording):
    async def scenario():
        retriever = RecordedRetrievalBoundary(recording)
        scope = ToolScope(tenant_id=recording['tenant_id'], conversation_id=7)
        state = ExecutionRun(scope, max_calls=1)
        events = []
        runtime = AgenticRuntime(retriever,
                                policy=Policy(recording['trajectories']['productive']),
                                emit=events.append,
                                generation_id=recording['generation_id'])
        answer = await runtime.answer(recording['question'], request_id='shared-budget',
                                      trace_id='b' * 32, execution_scope=scope,
                                      execution_run=state)
        assert len(retriever.calls) == 1
        assert events[-1]['exit'] == 'call_budget'
        assert state.halted
        assert_safe(answer)
    asyncio.run(scenario())
