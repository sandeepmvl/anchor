import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from anchor_ai.app import app
from anchor_ai import observability
from anchor_ai.hybrid_retriever import HybridRuntime, RecordedHybridClient, Snapshot, rrf_fuse

QUERY = 'refund for delayed shipment order 4821'
HEADERS = {'X-Anchor-Secret': 'hybrid-test', 'X-Request-Id': 'hybrid-request',
           'traceparent': '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01'}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'hybrid-test')
    monkeypatch.setenv('ANCHOR_HYBRID_FIXTURE', '1')
    monkeypatch.delattr(app.state, 'hybrid_runtime_factory', raising=False)
    with TestClient(app) as http:
        yield http


def post(client, query=QUERY, path='/retrieve/hybrid', **changes):
    body = dict(tenant_id=1, conversation_id=10, query=query, top_k=3)
    body.update(changes)
    return client.post(path, json=body, headers=HEADERS)


def test_rrf_arithmetic():
    rows = rrf_fuse([{'id': 101}, {'id': 101}, {'id': 102}],
                    [{'id': 102}, {'id': 103}])
    assert [row['id'] for row in rows] == [102, 101, 103]
    assert rows[0]['dense'] == pytest.approx(1 / 62)
    assert rows[0]['sparse'] == pytest.approx(1 / 61)
    assert rows[0]['score'] == pytest.approx(1 / 62 + 1 / 61)
    assert rrf_fuse([], []) == []
    assert [r['id'] for r in rrf_fuse([{'id': 2}], [{'id': 1}])] == [1, 2]
    for bad in (0, -1, float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            rrf_fuse([], [], bad)
    print('chunk 102: dense=1/62 sparse=1/61 sum=', rows[0]['score'])


@pytest.mark.parametrize('path', ['/retrieve/hybrid', '/v1/retrieve/hybrid'])
def test_routes(client, path):
    response = post(client, path=path)
    assert response.status_code == 200
    assert response.headers['X-Request-Id'] == 'hybrid-request'
    result = response.json()
    assert result['generation_id'] == 'fixture-g1'
    assert result['tenant_id'] == 1
    assert [r['id'] for r in result['hits']] == [102, 101, 103]


def test_mixed_fixture(client):
    fake = RecordedHybridClient()
    snapshot = fake.acquire_snapshot(1)
    cases = [(QUERY, 102), ('Can I get my money back because my parcel arrived late?', 101)]
    counts = dict(dense=0, sparse=0, hybrid=0)
    for query, label in cases:
        dense = fake.dense_search(query, snapshot, 5)
        sparse = fake.sparse_search(query, snapshot, 5)
        assert label in {row['id'] for row in dense + sparse}
        response = post(client, query)
        assert response.status_code == 200
        counts['dense'] += dense[0]['id'] == label
        counts['sparse'] += sparse[0]['id'] == label
        counts['hybrid'] += response.json()['hits'][0]['id'] == label
    assert counts == dict(dense=1, sparse=1, hybrid=2)
    print('top-1 hits / 2: dense=1, sparse=1, hybrid=2')


def test_auth_validation_and_missing_runtime(client, monkeypatch):
    body = dict(tenant_id=1, conversation_id=10, query=QUERY)
    assert client.post('/retrieve/hybrid', json=body).status_code == 401
    monkeypatch.delenv('ANCHOR_AI_SECRET')
    assert post(client).status_code == 503
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'hybrid-test')
    assert post(client, tenant_id=2).status_code == 503
    assert post(client, history=['not accepted']).status_code == 422
    assert post(client, top_k=0).status_code == 422
    assert post(client, query=' ').status_code == 422
    monkeypatch.delenv('ANCHOR_HYBRID_FIXTURE')
    assert post(client).status_code == 503


def test_overlap_and_publication_pin_one_snapshot(client, monkeypatch):
    fake = RecordedHybridClient()
    original = fake.snapshot
    current = [original]
    barrier = threading.Barrier(2, timeout=1)
    seen = []
    lock = threading.Lock()

    def search(query, snapshot, depth):
        with lock:
            seen.append(snapshot)
        barrier.wait()
        with lock:
            current[0] = Snapshot(1, 'fixture-g2', original.chunks)
        return [dict(id=102, tenant_id=1, generation_id=snapshot.generation_id)]

    runtime = HybridRuntime(1, lambda tenant: current[0], search, search)
    monkeypatch.setattr(app.state, 'hybrid_runtime_factory', lambda **kw: runtime, raising=False)
    response = post(client)
    assert response.status_code == 200
    assert len(seen) == 2 and all(snapshot is original for snapshot in seen)
    assert current[0].generation_id == 'fixture-g2'
    assert response.json()['generation_id'] == 'fixture-g1'
    assert response.json()['hits'][0]['dense'] > 0
    assert response.json()['hits'][0]['sparse'] > 0


@pytest.mark.parametrize('field,value', [('tenant_id', 2), ('generation_id', 'fixture-g2'), ('id', 999)])
def test_mixed_hits_fail_closed(client, monkeypatch, field, value):
    fake = RecordedHybridClient()

    def bad(query, snapshot, depth):
        hit = dict(id=102, tenant_id=1, generation_id=snapshot.generation_id)
        hit[field] = value
        return [hit]

    runtime = HybridRuntime(1, fake.acquire_snapshot, fake.dense_search, bad)
    monkeypatch.setattr(app.state, 'hybrid_runtime_factory', lambda **kw: runtime, raising=False)
    response = post(client)
    assert response.status_code == 422
    assert response.json() == {'detail': 'retrieval_failed'}


def test_wrong_snapshot_rejected(client, monkeypatch):
    fake = RecordedHybridClient()
    wrong = Snapshot(2, 'other-g1', (dict(id=201, tenant_id=2, text='private'),))
    runtime = HybridRuntime(1, lambda tenant: wrong, fake.dense_search, fake.sparse_search)
    monkeypatch.setattr(app.state, 'hybrid_runtime_factory', lambda **kw: runtime, raising=False)
    assert post(client).status_code == 422


def test_deadline(client, monkeypatch):
    fake = RecordedHybridClient()

    def slow(query, snapshot, depth):
        time.sleep(0.05)
        return []

    runtime = HybridRuntime(1, fake.acquire_snapshot, slow, slow, deadline_seconds=0.005)
    monkeypatch.setattr(app.state, 'hybrid_runtime_factory', lambda **kw: runtime, raising=False)
    response = post(client)
    assert response.status_code == 504
    assert response.json() == {'detail': 'retrieval_deadline'}


def test_private_distributed_trace(client, monkeypatch):
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    assert post(client).status_code == 200
    names = {'python.retrieval.dense', 'python.retrieval.sparse',
             'python.retrieval.fusion', 'python.hybrid_retrieval'}
    stages = [row for row in records if row['name'] in names]
    assert {row['name'] for row in stages} == names
    assert all(row['trace_id'] == 'a' * 32 for row in stages)
    assert all(row['request_id'] == 'hybrid-request' for row in stages)
    summary = next(row for row in stages if row['name'] == 'python.hybrid_retrieval')
    assert all(summary[key] >= 0 for key in ('dense_ms', 'sparse_ms', 'fusion_ms', 'total_ms'))
    assert all(row['parent_span_id'] == summary['span_id'] for row in stages if row is not summary)
    serialized = json.dumps(records)
    assert QUERY not in serialized
    assert 'Synthetic excerpt' not in serialized
    assert 'Refunds require review.' not in serialized
    assert 'query' not in summary and 'text' not in summary
