import asyncio
import importlib.util
import json
from pathlib import Path
import pytest

from anchor_ai.graph_index import graph_snapshot
from anchor_ai.graph_retriever import GraphRetriever, retrieve_snapshot
from anchor_ai.index_maintenance import (
    active, start_migration, backfill, activate, delete_document, maintain_document)
from anchor_ai.rag import published_graph_rag_runtime
from anchor_ai.schemas import DenseRagResponse

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('graph_demo',
    ROOT / 'labs/graph-retrieval/graph_retrieval_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture
def data():
    db, fixture = demo.build_database()
    yield db, fixture
    db.close()


def retrieve(db, fixture, index=0, **limits):
    q = fixture['questions'][index]
    return GraphRetriever(db, 1, 'support', **limits).retrieve(q['question'], q['terms'])


def test_scoped_cited_product_plan_exception_path(data):
    db, fixture = data
    report = retrieve(db, fixture)
    assert report['status'] == 'supported'
    path = report['paths'][0]
    assert [path[0]['source']] + [e['target'] for e in path] == fixture['expected_nodes']
    assert [e['source_doc'] for e in path] == ['plans.md', 'policies.md', 'exceptions.md']
    assert all(e['tenant_id'] == 1 and e['scope'] == 'support' for e in path)
    assert all(e['generation_id'] == active(db, 1)['id'] for e in path)
    for edge in path:
        chunk = db.execute('SELECT * FROM document_chunks WHERE id=?', (edge['citation'],)).fetchone()
        assert edge['passage'] in chunk['text']
        assert chunk['source_hash'] == edge['source_hash']
    assert retrieve(db, fixture, 1)['status'] == 'unsupported'
    assert retrieve(db, fixture, 2)['status'] == 'unlinked'
    assert GraphRetriever(db, 1, 'other').retrieve('Pro?', ['Pro', 'renewal exception'])['status'] == 'unlinked'


def test_scope_cannot_supply_missing_edge(data):
    db, fixture = data
    record = fixture['records'][3]
    maintain_document(db, 1, 'private.md', record['text'],
        {'graph': {'nodes': record['nodes'], 'edges': record['edges']}},
        'sentence-v1', 'private')
    assert retrieve(db, fixture, 1)['status'] == 'unsupported'
    assert all(e['scope'] == 'support' for e in graph_snapshot(db, 1, 'support')['edges'])


def test_depth_cycle_and_expansion_limits(data):
    db, fixture = data
    assert retrieve(db, fixture, max_depth=2)['status'] == 'unsupported'
    assert retrieve(db, fixture, max_expansions=1)['truncated']
    row = db.execute("SELECT * FROM documents WHERE tenant_id=1 AND external_id='plans.md'").fetchone()
    metadata = json.loads(row['metadata'])
    metadata['graph']['edges'].append(dict(source='plan:growth', relation='includes_product',
        target='product:pro', passage='Pro belongs to Growth.'))
    maintain_document(db, 1, row['external_id'], row['text'], metadata, 'sentence-v1', 'support')
    report = retrieve(db, fixture)
    assert report['status'] == 'supported'
    assert len(report['paths'][0]) == 3
    assert report['expansions'] <= 4
    assert retrieve(db, fixture, 1)['status'] == 'unsupported'


def test_generation_backfill_activation_and_rollback(data):
    db, fixture = data
    old = active(db, 1)['id']
    pending = start_migration(db, 1)
    backfill(db, 1, pending, batch_size=100)
    assert retrieve(db, fixture)['generation_id'] == old
    activate(db, 1, pending)
    report = retrieve(db, fixture)
    assert report['generation_id'] == pending
    assert all(e['generation_id'] == pending for e in report['paths'][0])
    activate(db, 1, old, rollback=True)
    assert retrieve(db, fixture)['generation_id'] == old


def test_deleted_source_is_excluded_even_if_stale_edge_survives(data):
    db, fixture = data
    edge = dict(retrieve(db, fixture)['paths'][0][-1])
    delete_document(db, 1, 'exceptions.md')
    assert not db.execute('SELECT 1 FROM graph_edges WHERE document_id=?', (edge['document_id'],)).fetchone()
    columns = list(edge)
    db.execute('INSERT INTO graph_edges (' + ','.join(columns) + ') VALUES (' +
               ','.join('?' for _ in columns) + ')', tuple(edge[c] for c in columns))
    db.commit()
    snapshot = graph_snapshot(db, 1, 'support')
    assert all(e['document_id'] != edge['document_id'] for e in snapshot['edges'])
    assert retrieve(db, fixture)['paths'] == []


def test_replacement_failure_and_bad_provenance_roll_back(data):
    db, fixture = data
    before = graph_snapshot(db, 1, 'support')
    row = db.execute("SELECT * FROM documents WHERE tenant_id=1 AND external_id='plans.md'").fetchone()
    metadata = json.loads(row['metadata'])
    with pytest.raises(RuntimeError):
        maintain_document(db, 1, 'plans.md', 'Replacement.', metadata,
                          'sentence-v1', 'support', fail=True)
    assert graph_snapshot(db, 1, 'support') == before
    metadata['graph']['edges'][0]['passage'] = 'Invented relationship.'
    with pytest.raises(ValueError, match='graph_passage_without_chunk'):
        maintain_document(db, 1, 'plans.md', row['text'], metadata, 'sentence-v1', 'support')
    assert graph_snapshot(db, 1, 'support') == before


def test_filter_removal_exposes_known_foreign_witness(data):
    db, fixture = data
    q = fixture['questions'][1]
    snapshot = graph_snapshot(db, 1, 'support')
    assert retrieve_snapshot(snapshot, q['question'], q['terms'])['status'] == 'unsupported'
    foreign = graph_snapshot(db, 2, 'support')
    snapshot['nodes'] += foreign['nodes']
    snapshot['edges'] += foreign['edges']
    report = retrieve_snapshot(snapshot, q['question'], q['terms'])
    assert report['status'] == 'supported'
    assert any(e['tenant_id'] == 2 for e in report['paths'][0])


def test_graph_rag_preserves_answer_contract(data):
    db, fixture = data
    runtime = published_graph_rag_runtime(db, 1, 'support')
    q = fixture['questions'][0]
    answer = asyncio.run(runtime.answer(dict(question=q['question'], terms=q['terms']),
                         request_id='test', trace_id='test'))
    DenseRagResponse.model_validate(answer)
    assert not answer['abstained']
    assert len(answer['citations']) == 3
    assert runtime.last_report['status'] == 'supported'
