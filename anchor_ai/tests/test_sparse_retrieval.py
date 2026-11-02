import runpy
from pathlib import Path

import pytest

from anchor_ai.index_maintenance import (
    EmbeddingClient, activate, active, backfill, delete_document,
    maintain_document, retrieval_snapshot, start_migration,
)
from anchor_ai.sparse_retriever import SparseRetriever, tokenize

HELPERS = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'bin/bm25_query.py'))


@pytest.fixture
def db():
    connection = HELPERS['fixture_db']()
    yield connection
    connection.close()


def put(db, tenant, key, text, metadata=None, fail=False):
    # Use production sentence chunking here, including during migration.
    return maintain_document(db, tenant, key, text, metadata or {},
                             'sentence-v1', 'sparse-tests',
                             client=EmbeddingClient(), fail=fail)


def seed(db, tenant=1):
    put(db, tenant, 'order', 'Synthetic order 4821 has a shipping delay.')
    put(db, tenant, 'policy', 'Shipping delay refund policy requires review.')
    put(db, tenant, 'login', 'Password reset authentication.')
    put(db, tenant, 'hours', 'Opening hours holidays.')
    put(db, tenant, 'invoice', 'Invoice tax receipts.')


def test_numeric_codes_punctuation_and_result_contract(db):
    seed(db)
    retriever = SparseRetriever(db)
    assert tokenize('4821? SKU-1234-B! case_91 ref.82') == [
        '4821', 'sku-1234-b', 'case_91', 'ref.82']
    hits = retriever.search('4821?', 1)
    assert hits == retriever.search('4821', 1)
    assert hits and '4821' in tokenize(hits[0]['text'])
    assert all(set(hit) == {'id', 'tenant_id', 'text', 'score'} for hit in hits)
    assert all(hit['tenant_id'] == 1 and isinstance(hit['score'], float) for hit in hits)
    assert hits == sorted(hits, key=lambda hit: (-hit['score'], hit['id']))
    put(db, 1, 'sku', 'Track SKU-1234-B for this repair.')
    assert 'sku-1234-b' in tokenize(retriever.search('SKU-1234-B?', 1)[0]['text'])


def test_other_tenant_cannot_change_candidates_or_statistics(db):
    seed(db)
    retriever = SparseRetriever(db)
    expected = retriever.search('shipping delay 4821', 1)
    assert expected
    for i in range(12):
        put(db, 2, str(i), 'Shipping delay refund policy order 4821 private.')
    assert retriever.search('shipping delay 4821', 1) == expected
    assert SparseRetriever(db).search('shipping delay 4821', 1) == expected
    assert retriever.search('private', 1) == []
    assert retriever.search('private', 2)


def test_empty_unknown_and_invalid_tenants(db):
    retriever = SparseRetriever(db)
    assert retriever.search('4821', 99) == []
    seed(db)
    assert retriever.search('???', 1) == []
    assert retriever.search('nonexistenttoken', 1) == []
    for tenant in (True, 0, -1, '1'):
        with pytest.raises(ValueError, match='invalid_tenant'):
            retriever.search('4821', tenant)


def test_replacement_metadata_noop_failure_and_deletion_refresh(db):
    seed(db)
    retriever = SparseRetriever(db)
    assert retriever.search('4821', 1)
    epoch = active(db, 1)['epoch']
    result = put(db, 1, 'order', 'Synthetic order 4821 has a shipping delay.')
    assert result['skipped'] and active(db, 1)['epoch'] == epoch
    put(db, 1, 'order', 'Synthetic order 4821 has a shipping delay.', {'tag': 'review'})
    assert active(db, 1)['epoch'] > epoch
    assert retriever.search('4821', 1)
    with pytest.raises(RuntimeError, match='injected replacement failure'):
        put(db, 1, 'order', 'Synthetic order 9999 is delivered.', fail=True)
    assert retriever.search('4821', 1)
    assert retriever.search('9999', 1) == []
    put(db, 1, 'order', 'Synthetic order 9999 is delivered.')
    assert retriever.search('4821', 1) == []
    assert retriever.search('9999', 1)
    delete_document(db, 1, 'order')
    assert retriever.search('9999', 1) == []


def test_building_generation_activation_rollback_and_delete(db):
    seed(db)
    retriever = SparseRetriever(db)
    original = active(db, 1)['id']
    old_hits = retriever.search('4821', 1)
    old_ids = {hit['id'] for hit in old_hits}
    pending = start_migration(db, 1)
    while backfill(db, 1, pending, batch_size=2):
        pass
    # A fully populated building generation must not enter the active corpus.
    assert retriever.search('4821', 1) == old_hits
    activate(db, 1, pending)
    new_hits = retriever.search('4821', 1)
    assert new_hits and old_ids.isdisjoint(hit['id'] for hit in new_hits)
    gen, rows = retrieval_snapshot(db, 1)
    assert gen['id'] == pending
    assert all(row['retrieval_generation_id'] == pending for row in rows)
    activate(db, 1, original, rollback=True)
    assert retriever.search('4821', 1) == old_hits
    delete_document(db, 1, 'order')
    assert retriever.search('4821', 1) == []
    activate(db, 1, pending, rollback=True)
    assert retriever.search('4821', 1) == []


def test_snapshot_excludes_stale_hash_and_preserves_callers_transaction(db):
    seed(db)
    db.execute('BEGIN')
    db.execute("UPDATE documents SET content_hash='stale' WHERE external_id='order'")
    _, rows = retrieval_snapshot(db, 1)
    assert db.in_transaction
    assert not any('4821' in tokenize(row['text']) for row in rows)
    db.rollback()
    assert SparseRetriever(db).search('4821', 1)
