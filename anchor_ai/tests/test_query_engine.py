import asyncio
from functools import wraps
import json
import multiprocessing
import os
import traceback

import pytest

from anchor_ai.hybrid_retriever import HybridRuntime, Snapshot
from anchor_ai.rag import RecordedEvidenceClient, fixture_runtime


def execute_isolated_test(name, writer):
    """Spawn imports this module afresh, without inherited native thread locks."""
    try:
        globals()[name].__wrapped__()
        writer.send(None)
    except BaseException:
        writer.send(traceback.format_exc())
    finally:
        writer.close()


def isolated_framework_test(test):
    """Release framework imports and isolate numerical-library thread pools."""
    @wraps(test)
    def run():
        context = multiprocessing.get_context('spawn')
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=execute_isolated_test, args=(test.__name__, writer))
        thread_variables = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
        previous = {key: os.environ.get(key) for key in thread_variables}
        try:
            for key in thread_variables:
                os.environ[key] = '1'
            process.start()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        writer.close()
        try:
            assert reader.poll(30), 'Query-engine test child timed out'
            try:
                failure = reader.recv()
            except EOFError:
                pytest.fail('Query-engine child exited before reporting its result')
            process.join(15)
            assert failure is None, failure
            assert process.exitcode == 0
        finally:
            reader.close()
            if process.is_alive():
                process.terminate()
                process.join()
    return run


def QueryEngineTool(*args, **kwargs):
    # Keep framework imports out of the parent regression-suite process.
    from anchor_ai.query_engine import QueryEngineTool as implementation
    return implementation(*args, **kwargs)


QUERY = 'refund policy for EU customers'
FACTS = (
    'EU customers may request a refund within 14 days.',
    'EU refund requests require an order number.',
    'Approved EU refunds return to the original payment method.',
)


def seeded_runtime():
    snapshot = Snapshot(1, 'eu-policy-g1', tuple(
        dict(id=101 + index, tenant_id=1, text=text)
        for index, text in enumerate(FACTS)))
    calls = []

    def acquire(tenant_id):
        calls.append(('snapshot', tenant_id))
        assert tenant_id == 1
        return snapshot

    def search(query, pinned, depth):
        calls.append(('search', pinned.generation_id))
        return [dict(id=row['id'], tenant_id=1, generation_id=pinned.generation_id)
                for row in pinned.chunks[:depth]]

    return HybridRuntime(1, acquire, search, search), calls


@isolated_framework_test
def test_three_citations_reuse_existing_hybrid_and_model_adapter():
    runtime, calls = seeded_runtime()
    model = RecordedEvidenceClient()
    result = asyncio.run(QueryEngineTool(runtime, 1, model)(QUERY))
    assert result['citations'] == [101, 102, 103]
    assert not result['abstained']
    assert len(result['source_nodes']) == 3
    assert [node['text'] for node in result['source_nodes']] == list(FACTS)
    assert result['generation_id'] == 'eu-policy-g1'
    assert calls.count(('snapshot', 1)) == 1
    assert len([call for call in calls if call[0] == 'search']) == 2
    assert len(model.calls) == 1
    assert all(f'[{chunk_id}]' in result['answer'] for chunk_id in result['citations'])


@isolated_framework_test
def test_invented_citation_fails_closed():
    class InventedClient:
        async def complete(self, messages):
            return json.dumps(dict(reply='Invented policy [999]', citations=[999], abstained=False))

    runtime, _ = seeded_runtime()
    result = asyncio.run(QueryEngineTool(runtime, 1, InventedClient())(QUERY))
    assert result['abstained']
    assert result['citations'] == []
    assert 'Invented policy' not in result['answer']


@isolated_framework_test
def test_foreign_envelope_is_rejected_before_synthesis():
    class ForeignRuntime:
        async def retrieve(self, query, top_k):
            return dict(tenant_id=2, generation_id='foreign', hits=[])

    model = RecordedEvidenceClient()
    with pytest.raises(ValueError, match='scope_mismatch'):
        asyncio.run(QueryEngineTool(ForeignRuntime(), 1, model)(QUERY))
    assert model.calls == []


@isolated_framework_test
def test_existing_dense_offline_embedding_and_generation_adapters():
    runtime = fixture_runtime(tenant_id=1)
    result = asyncio.run(QueryEngineTool(runtime.retriever, 1, runtime.client)(
        'How do I get a refund?'))
    assert 'source_nodes' in result
    assert result['source_nodes']
    assert all(node['tenant_id'] == 1 for node in result['source_nodes'])
    trace = runtime.retriever.last_trace
    assert result['generation_id'] == trace['generation_id']
    assert {node['document_chunk_id'] for node in result['source_nodes']}.issubset(
        set(trace['chunk_ids']))


@isolated_framework_test
def test_attach_does_not_call_embedding_or_model():
    class Boundary:
        embedding_calls = 0

        async def retrieve(self, query, top_k):
            raise AssertionError('Attachment must not retrieve')

        def embed(self, texts):
            self.embedding_calls += 1
            raise AssertionError('Attachment must not embed')

    boundary = Boundary()
    model = RecordedEvidenceClient()
    QueryEngineTool(boundary, 1, model)
    assert boundary.embedding_calls == 0
    assert model.calls == []


@isolated_framework_test
def test_tool_has_no_model_supplied_tenant_argument():
    runtime, _ = seeded_runtime()
    tool = QueryEngineTool(runtime, 1, RecordedEvidenceClient())
    with pytest.raises(TypeError):
        asyncio.run(tool(QUERY, tenant_id=2))


@isolated_framework_test
def test_dense_explicit_foreign_hit_is_rejected_before_synthesis():
    class ForeignDense:
        def search(self, query, top_k):
            self.last_trace = dict(tenant_id=1, generation_id='dense-g1', chunk_ids=[101])
            return [dict(id=101, tenant_id=2, text='Foreign policy')]

    model = RecordedEvidenceClient()
    with pytest.raises(ValueError, match='hit_mismatch'):
        asyncio.run(QueryEngineTool(ForeignDense(), 1, model)(QUERY))
    assert model.calls == []
