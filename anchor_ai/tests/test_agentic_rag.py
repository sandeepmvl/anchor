import asyncio
import copy
import json

import pytest

from anchor_ai import observability
from anchor_ai.agentic_rag import (
    AgenticRuntime, LivePolicy, RecordedPolicy, RecordedRetrievalBoundary,
    RetrievalTool, fixture_data, select_policy,
)
from anchor_ai.rag import RagRuntime, published_agentic_rag_runtime
from anchor_ai.schemas import RagRequest


def run(runtime, question):
    return asyncio.run(runtime.answer(question, request_id='agent-test', trace_id='b' * 32))


class ConstantPolicy:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    async def decide(self, question, evidence):
        self.calls += 1
        return self.decision


def search_decision(query='Cedar annual'):
    return {'kind': 'retrieve', 'tool': 'retrieve', 'arguments': {'query': query}}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.setattr(observability, 'SINK', lambda record: None)


def test_recorded_trajectories_have_different_counts_and_exits():
    data = fixture_data()
    expected = {'supplied-policy': (0, 'final_answer'),
                'known-policy': (1, 'evidence_sufficient'),
                'compound-policy': (3, 'evidence_sufficient')}
    for name, (count, reason) in expected.items():
        events = []
        backend = RecordedRetrievalBoundary(data)
        runtime = AgenticRuntime(backend, RecordedPolicy(data), emit=events.append)
        result = run(runtime, data['tickets'][name]['question'])
        assert len(backend.calls) == count
        assert events[-1]['exit'] == reason
        assert result == data['tickets'][name]['answer']
        assert all(c[:2] == (1, 'agentic-fixture-v1') for c in backend.calls)
    assert [r['passage_ids'] for r in runtime.last_trajectory] == [[101], [102], [103], []]


@pytest.mark.parametrize('decision', [
    {'kind': 'retrieve', 'tool': 'delete', 'arguments': {'query': 'x'}},
    {'kind': 'retrieve', 'tool': 'retrieve', 'arguments': {'query': 42}},
    {'kind': 'retrieve', 'tool': 'retrieve', 'arguments': {'query': ' '}},
    {'kind': 'retrieve', 'tool': 'retrieve', 'arguments': {'query': 'x', 'tenant_id': 2}},
    {'kind': 'retrieve', 'tool': 'retrieve', 'arguments': {'query': 'x', 'generation_id': 'other'}},
    {'kind': 'retrieve', 'tool': 'retrieve', 'arguments': {'query': 'é' * 151}},
    {'kind': 'unknown'},
])
def test_invalid_tools_are_rejected_before_execution(decision):
    backend = RecordedRetrievalBoundary()
    events = []
    result = run(AgenticRuntime(backend, ConstantPolicy(decision), emit=events.append), 'ticket')
    assert result['abstained'] is True
    assert events[-1]['exit'] == 'invalid_decision'
    assert backend.calls == []


def test_tool_wrapper_revalidates_arguments():
    backend = RecordedRetrievalBoundary()
    tool = RetrievalTool(backend, 1)
    with pytest.raises(ValueError):
        asyncio.run(tool({'query': 'x', 'tenant_id': 2}, 3))
    assert backend.calls == []


@pytest.mark.parametrize('limits,expected_count,exit_reason', [
    ({'max_tool_calls': 1}, 1, 'tool_budget'),
    ({'max_tool_calls': 0}, 0, 'tool_budget'),
    ({'max_iterations': 2}, 2, 'iteration_budget'),
    ({'max_iterations': 1000, 'max_tool_calls': 1000}, 4, 'tool_budget'),
])
def test_hard_budgets_terminate_repeated_calls(limits, expected_count, exit_reason):
    backend = RecordedRetrievalBoundary()
    events = []
    policy = ConstantPolicy(search_decision())
    result = run(AgenticRuntime(backend, policy, emit=events.append, **limits), 'ticket')
    assert len(backend.calls) == expected_count
    assert events[-1]['exit'] == exit_reason
    assert result['abstained'] and result['citations'] == []
    assert policy.calls <= 6


@pytest.mark.parametrize('change', ['tenant', 'generation', 'hit'])
def test_scope_mismatch_never_becomes_evidence(change):
    class CorruptBoundary(RecordedRetrievalBoundary):
        async def retrieve(self, query, top_k=3):
            result = await super().retrieve(query, top_k)
            if change == 'tenant':
                result['tenant_id'] = 2
            elif change == 'generation':
                result['generation_id'] = 'unpublished-generation'
            else:
                result['hits'][0]['tenant_id'] = 2
            return result
    events = []
    runtime = AgenticRuntime(CorruptBoundary(), ConstantPolicy(search_decision()),
                             generation_id='agentic-fixture-v1', emit=events.append)
    assert run(runtime, 'ticket')['abstained']
    assert events[-1]['exit'] == 'tool_error'
    assert events[-1]['evidence_chunk_ids'] == []


def test_generation_cannot_change_between_calls():
    class ChangingBoundary(RecordedRetrievalBoundary):
        async def retrieve(self, query, top_k=3):
            result = await super().retrieve(query, top_k)
            if len(self.calls) > 1:
                result['generation_id'] = 'new-generation'
            return result
    events = []
    runtime = AgenticRuntime(ChangingBoundary(), ConstantPolicy(search_decision()), emit=events.append)
    assert run(runtime, 'ticket')['abstained']
    assert events[-1]['exit'] == 'tool_error'
    assert events[-1]['evidence_chunk_ids'] == [101]


def test_published_factory_pins_the_existing_snapshot(monkeypatch):
    from anchor_ai import rag
    acquired = []
    snapshot = {'generation_id': 'ready-7'}
    def acquire(tenant):
        acquired.append(tenant)
        return snapshot
    def hybrid(tenant, acquire_snapshot, dense, sparse, **settings):
        assert acquire_snapshot(tenant) is snapshot
        assert acquire_snapshot(tenant) is snapshot
        with pytest.raises(ValueError):
            acquire_snapshot(2)
        return RecordedRetrievalBoundary()
    monkeypatch.setattr(rag, 'published_hybrid_runtime', hybrid)
    runtime = published_agentic_rag_runtime(1, acquire, None, None,
        policy=RecordedPolicy(), client=object())
    assert acquired == [1]
    assert runtime.tool.tenant_id == 1


def test_final_answer_requires_known_citations():
    events = []
    policy = ConstantPolicy({'kind': 'final', 'answer': {
        'reply': 'Invented policy.', 'citations': [999], 'abstained': False}})
    result = run(AgenticRuntime(RecordedRetrievalBoundary(), policy, emit=events.append), 'ticket')
    assert result['abstained']
    assert events[-1]['exit'] == 'invalid_answer'


def test_evidence_cap_does_not_fabricate_coverage():
    data = fixture_data()
    events = []
    runtime = AgenticRuntime(RecordedRetrievalBoundary(), RecordedPolicy(),
                             evidence_budget=1, emit=events.append)
    result = run(runtime, data['tickets']['compound-policy']['question'])
    assert result['abstained']
    assert events[-1]['exit'] == 'tool_budget'
    assert events[-1]['evidence_chunk_ids'] == []


def test_trace_is_correlated_and_sanitized(monkeypatch):
    spans = []
    monkeypatch.setattr(observability, 'SINK', spans.append)
    secret = 'customer-private-marker@example.test'
    run(AgenticRuntime(RecordedRetrievalBoundary(), ConstantPolicy(search_decision()),
                       max_tool_calls=1), secret)
    root = next(s for s in spans if s['name'] == 'python.agent')
    model = next(s for s in spans if s['name'] == 'python.agent.model')
    tool = next(s for s in spans if s['name'] == 'python.agent.tool')
    retrieval = next(s for s in spans if s['name'] == 'python.agent.retrieve')
    assert model['parent_span_id'] == root['span_id']
    assert tool['parent_span_id'] == root['span_id']
    assert retrieval['parent_span_id'] == tool['span_id']
    assert {s['trace_id'] for s in spans} == {'b' * 32}
    assert {s['request_id'] for s in spans} == {'agent-test'}
    encoded = json.dumps(spans)
    assert secret not in encoded
    assert 'Cedar annual' not in encoded
    assert 'Cedar annual uses the Flex tier.' not in encoded
    assert model['decision_summary'] == 'Request another targeted search.'


def test_live_adapter_uses_same_controller_without_network():
    class RecordedClient:
        async def complete(self, prompt, system=None, max_tokens=256):
            payload = json.loads(prompt)
            assert payload['tool']['name'] == 'retrieve'
            if not payload['passages']:
                return json.dumps(search_decision('R7 refund window'))
            return json.dumps({'kind': 'final', 'answer': {
                'reply': 'R7 permits refunds within 14 days.',
                'citations': [103], 'abstained': False}})
    backend = RecordedRetrievalBoundary()
    events = []
    result = run(AgenticRuntime(backend, LivePolicy(RecordedClient()), emit=events.append), 'R7?')
    assert result['citations'] == [103]
    assert len(backend.calls) == 1
    assert events[-1]['exit'] == 'final_answer'


def test_mode_switch_and_rag_dispatch(monkeypatch):
    assert isinstance(select_policy()[1], RecordedPolicy)
    monkeypatch.setattr(LivePolicy, '__init__', lambda self: None)
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'live')
    assert isinstance(select_policy()[1], LivePolicy)
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    request = RagRequest(tenant_id=1, conversation_id=10, message='R7?', mode='agent')
    assert request.mode == 'agent'
    data = fixture_data()
    runtime = RagRuntime(RecordedRetrievalBoundary(), None, evidence_budget=2400)
    result = asyncio.run(runtime.answer(data['tickets']['compound-policy']['question'],
        request_id='dispatch', trace_id='c' * 32, mode='agent'))
    assert result['citations'] == [101, 102, 103]
    assert len(runtime.last_trajectory) == 4
