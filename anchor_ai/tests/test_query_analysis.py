import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anchor_ai.query_analyzer import QueryAnalyzer, RecordedQueryClient, identifiers, sanitize
from anchor_ai.rag import fixture_analyzed_runtime
from anchor_ai.hybrid_retriever import HybridRuntime, Snapshot
from eval.retrieval_gate import load_dataset, evaluate, enforce

ROOT = Path(__file__).resolve().parents[2]


def run(awaitable):
    return asyncio.run(awaitable)


class OutputClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    async def complete(self, ticket):
        self.calls.append(ticket)
        return self.output


@pytest.mark.parametrize('case', load_dataset()['cases'])
def test_five_recorded_rewrites(case):
    client = RecordedQueryClient()
    plan = run(QueryAnalyzer(client).analyze(case['ticket']))
    assert not plan.fallback
    assert plan.queries == client.responses[case['ticket']]['queries']
    assert identifiers(case['ticket']) <= identifiers(' '.join(plan.queries))
    assert 1 <= len(plan.queries) <= 3
    assert client.calls == [sanitize(case['ticket'])]


@pytest.mark.parametrize('output', [
    'not json', '[]', '{"queries":[]}', '{"queries":[7]}',
    '{"queries":[" "]}', '{"queries":["a","b","c","d"]}',
    '{"queries":["refund"],"tenant_id":2}',
    '{"queries":["refund"],"generation_id":"foreign"}',
    json.dumps({'queries': ['x' * 301]}),
])
def test_invalid_output_falls_back_once(output):
    client = OutputClient(output)
    ticket = 'refund? email person@example.test or 555-123-4567'
    plan = run(QueryAnalyzer(client).analyze(ticket))
    assert plan.fallback
    assert plan.queries == [sanitize(ticket)]
    assert 'person@example.test' not in client.calls[0]
    assert '555-123-4567' not in client.calls[0]
    assert len(client.calls) == 1


def test_identifier_loss_and_numeric_order_preservation():
    ticket = 'ORDER-42 and order 4821; warranty SKU-7'
    bad = OutputClient('{"queries":["warranty SKU-7"]}')
    assert run(QueryAnalyzer(bad).analyze(ticket)).queries == [ticket]
    good = OutputClient('{"queries":["warranty SKU-7 ORDER-42 4821"]}')
    assert not run(QueryAnalyzer(good).analyze(ticket)).fallback
    assert len(bad.calls) == len(good.calls) == 1


def test_identifier_bypass_and_configured_cap():
    client = OutputClient('{"queries":["refund","shipping"]}')
    assert run(QueryAnalyzer(client).analyze('SKU-7')).reason == 'identifier_bypass'
    assert not client.calls
    plan = run(QueryAnalyzer(client, max_queries=1).analyze('two questions'))
    assert plan.fallback and plan.queries == ['two questions']
    assert len(client.calls) == 1


def test_timeout_and_client_error_do_not_retry():
    class Slow:
        calls = 0

        async def complete(self, ticket):
            self.calls += 1
            await asyncio.sleep(1)

    client = Slow()
    plan = run(QueryAnalyzer(client, deadline_seconds=0.001).analyze('refund please'))
    assert plan.fallback and plan.reason == 'timeout' and client.calls == 1

    class Broken:
        calls = 0

        async def complete(self, ticket):
            self.calls += 1
            raise RuntimeError('private provider detail')

    client = Broken()
    plan = run(QueryAnalyzer(client).analyze('refund please'))
    assert plan.reason == 'client_error' and client.calls == 1


def test_one_snapshot_scope_deduplication_and_safe_trace(monkeypatch):
    import anchor_ai.observability as telemetry
    records = []
    monkeypatch.setattr(telemetry, 'SINK', records.append)
    runtime = fixture_analyzed_runtime(1)
    ticket = load_dataset()['cases'][0]['ticket']
    result = run(runtime.retrieve(ticket))
    client = runtime.fixture_client
    assert client.acquisitions == [1]
    assert len(client.calls) == 4
    assert {(call[2], call[3]) for call in client.calls} == {(1, 'query-analysis-g1')}
    ids = [hit['id'] for hit in result['hits']]
    assert {201, 202} <= set(ids)
    assert len(ids) == len(set(ids))
    assert all(hit['tenant_id'] == 1 for hit in result['hits'])
    serialized = json.dumps(records)
    assert 'ORDER-42' not in serialized and ticket not in serialized
    names = {record['name'] for record in records}
    assert {'python.query_analysis', 'python.analyzed_retrieval',
            'python.retrieval.dense', 'python.retrieval.sparse'} <= names
    assert len({record['trace_id'] for record in records}) == 1


@pytest.mark.parametrize('tenant,generation', [(2, 'g1'), (1, 'g2')])
def test_reject_cross_tenant_or_generation_hits(tenant, generation):
    snapshot = Snapshot(1, 'g1', (dict(id=1, tenant_id=1, text='refund'),))

    def malicious(query, snapshot, depth):
        return [dict(id=1, tenant_id=tenant, generation_id=generation)]

    runtime = HybridRuntime(1, lambda tenant: snapshot, malicious, malicious)
    with pytest.raises(ValueError, match='hit_scope_mismatch'):
        run(runtime.retrieve_analyzed('refund', QueryAnalyzer(RecordedQueryClient())))


def test_route_authentication_and_fixture_scope(monkeypatch):
    from anchor_ai.app import app
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'test-secret')
    monkeypatch.setenv('ANCHOR_QUERY_FIXTURE', '1')
    body = dict(tenant_id=1, conversation_id=10,
                query=load_dataset()['cases'][0]['ticket'], top_k=3)
    with TestClient(app) as client:
        assert client.post('/v1/retrieve/analyzed', json=body).status_code == 401
        headers = {'X-Anchor-Secret': 'test-secret'}
        response = client.post('/v1/retrieve/analyzed', json=body, headers=headers)
        assert response.status_code == 200
        assert response.json()['plan']['queries'][0].endswith('ORDER-42')
        assert response.json()['tenant_id'] == 1
        assert client.post('/v1/retrieve/analyzed', json={**body, 'tenant_id': 2},
                           headers=headers).status_code == 503
        monkeypatch.setattr(app.state, 'analyzed_runtime_factory',
                            lambda tenant_id: fixture_analyzed_runtime(2), raising=False)
        assert client.post('/v1/retrieve/analyzed', json=body,
                           headers=headers).status_code == 422
        monkeypatch.delattr(app.state, 'analyzed_runtime_factory')


def test_golden_gate_and_nonzero_exit_on_regression(tmp_path):
    dataset = load_dataset()
    report = run(evaluate(dataset, verbose=False))
    enforce(report, dataset['thresholds'])
    assert report['chunk_recall'] == report['identifier_retention'] == 1.0
    assert report['cross_tenant_results'] == 0
    assert report['ticket_hit_rate'] > report['raw_ticket_hit_rate']
    # Keep a valid label but replace its passage so retrieval cannot recover it.
    dataset['chunks'][0]['text'] = 'zzzzunrelated'
    path = tmp_path / 'regression.json'
    path.write_text(json.dumps(dataset))
    # Bound fresh numerical-library thread pools in the command-line process.
    # Inherit the remaining environment and retain the real gate and deadline.
    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    process = subprocess.run([sys.executable, '-m', 'eval.retrieval_gate', str(path)],
                             cwd=ROOT, capture_output=True, text=True, timeout=8,
                             env=env)
    assert process.returncode != 0
    assert 'retrieval_regression' in process.stderr
