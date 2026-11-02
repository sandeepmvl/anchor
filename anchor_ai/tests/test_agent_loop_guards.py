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
