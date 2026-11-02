import asyncio
import json

from anchor_ai.rag import (
    ABSTENTION, RagRuntime, RecordedGenerationClient, fixture_runtime, pack_evidence)
from anchor_ai.dense_retriever import DenseRetriever, fixture_dependencies

QUERY = 'How do I get a refund?'


def run(runtime, question=QUERY):
    return asyncio.run(runtime.answer(question, request_id='request-3672', trace_id='a' * 32))


def test_grounded_reply_budgeted_untrusted_evidence_and_trace():
    traces = []
    runtime = fixture_runtime(emit=traces.append)
    result = run(runtime)
    assert not result['abstained']
    assert result['citations'] == [101]
    assert 'Refunds require review.' in result['reply']
    messages = runtime.client.calls[0]
    evidence = json.loads(messages[-1]['content'])['evidence']
    assert evidence['label'] == 'UNTRUSTED_EVIDENCE'
    assert len(json.dumps(evidence, ensure_ascii=False).encode('utf-8')) <= 600
    assert set(result['citations']) <= {s['id'] for s in evidence['sources']}
    trace = traces[0]
    assert trace['request_id'] == 'request-3672'
    assert trace['trace_id'] == 'a' * 32
    assert trace['chunk_ids'] == [101, 102, 103]
    assert all(name in trace for name in ('embedding_ms', 'search_ms', 'fallback_ms', 'generation_ms'))
    serialized = json.dumps(trace)
    assert QUERY not in serialized
    assert 'Refunds require review.' not in serialized


def test_no_support_and_no_retrieval_abstain():
    for runtime, question in [(fixture_runtime(), 'Do you cover lunar shipping?'),
                              (fixture_runtime(999), QUERY)]:
        result = run(runtime, question)
        assert result == dict(reply=ABSTENTION, citations=[], abstained=True)


def test_mutating_supplied_evidence_changes_offline_answer():
    generation, embeddings = fixture_dependencies()
    generation['chunks'][-3]['text'] = 'Refunds require review. Provide the violet receipt code.'
    runtime = RagRuntime(DenseRetriever(generation, 1, embeddings), RecordedGenerationClient())
    assert 'violet receipt code' in run(runtime)['reply']


def test_unknown_citation_and_unbudgeted_citation_are_rejected():
    class InvalidReferenceClient:
        async def complete(self, messages):
            return json.dumps(dict(reply='Invented policy', citations=[9999], abstained=False))
    runtime = fixture_runtime()
    runtime.client = InvalidReferenceClient()
    assert run(runtime)['abstained']
    runtime = fixture_runtime()
    runtime.evidence_budget = 60
    assert run(runtime)['abstained']
    assert runtime.client.calls == []


def test_whole_sources_are_packed_without_dangling_references():
    hits = [dict(id=1, text='x' * 10000), dict(id=2, text='Short supporting policy.')]
    evidence = pack_evidence(hits, 160)
    assert [s['id'] for s in evidence['sources']] == [2]
    assert len(json.dumps(evidence, ensure_ascii=False).encode('utf-8')) <= 160
