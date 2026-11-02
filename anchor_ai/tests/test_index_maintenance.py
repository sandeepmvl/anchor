import pytest
from anchor_ai.storage import initialize_sandbox, connect
from anchor_ai.pii import prepare
from anchor_ai.ingestion import sync_directory
from anchor_ai.index_maintenance import (
    EmbeddingClient, maintain_document, delete_document, active, start_migration,
    backfill, activate, complete, generation,
)
from anchor_ai.vector_index import VectorIndex


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('ANCHOR_EMBEDDING_PROVIDER', 'offline')
    path = tmp_path / 'index.sqlite3'
    initialize_sandbox(path)
    db = connect(path)
    yield db, path
    db.close()


def put(db, tenant=1, text='Refunds are available within thirty days.', **kwargs):
    return maintain_document(db, tenant, 'policy.md', prepare(text),
                             kwargs.pop('metadata', {'title': 'Policy'}),
                             kwargs.pop('version', 'sentence-v1'), None, **kwargs)


def finish(db, identity):
    while backfill(db, 1, identity, batch_size=1):
        pass


def test_sanitized_hash_metadata_and_chunker(store):
    db, _ = store
    client = EmbeddingClient()
    put(db, text='Contact alice@example.com for refunds.', client=client)
    before = [tuple(r) for r in db.execute('SELECT * FROM document_chunks')]
    client.calls = 0
    assert put(db, text='Contact bob@example.com for refunds.', client=client)['skipped']
    assert client.calls == 0
    assert before == [tuple(r) for r in db.execute('SELECT * FROM document_chunks')]
    put(db, text='Contact bob@example.com for refunds.', metadata={'title': 'Current'}, client=client)
    assert client.calls == 0
    ids = [r['id'] for r in db.execute('SELECT id FROM document_chunks')]
    put(db, text='Contact bob@example.com for refunds.', version='sentence-v2', client=client)
    assert client.calls > 0
    assert set(ids).isdisjoint(r['id'] for r in db.execute('SELECT id FROM document_chunks'))
    snapshot = [tuple(r) for r in db.execute('SELECT * FROM document_chunks')]
    with pytest.raises(RuntimeError):
        put(db, text='Changed policy.', fail=True)
    assert snapshot == [tuple(r) for r in db.execute('SELECT * FROM document_chunks')]


def test_inventory_and_tenant_isolation(store, tmp_path):
    db, _ = store
    kb = tmp_path / 'kb'
    kb.mkdir()
    (kb / 'policy.md').write_text('Refunds within thirty days.')
    sync_directory(db, 1, kb)
    put(db, tenant=2, text='Other tenant policy.')
    (kb / 'policy.md').unlink()
    sync_directory(db, 1, kb)
    assert db.execute('SELECT count(*) FROM documents WHERE tenant_id=1').fetchone()[0] == 0
    assert db.execute('SELECT count(*) FROM documents WHERE tenant_id=2').fetchone()[0] == 1
    with pytest.raises(ValueError):
        sync_directory(db, 1, tmp_path / 'missing')


def test_delete_all_generations_and_snapshot_invalidation(store):
    db, _ = store
    put(db)
    put(db, tenant=2, text='Tenant two refund policy.')
    target = start_migration(db, 1)
    finish(db, target)
    index = VectorIndex()
    assert index.search_active(db, 1, 'refund')['hits']
    key = index.snapshot_key
    delete_document(db, 1, 'policy.md')
    assert not index.search_active(db, 1, 'refund')['hits']
    assert key != index.snapshot_key
    assert db.execute('SELECT count(*) FROM document_chunks WHERE tenant_id=1').fetchone()[0] == 0
    assert db.execute('SELECT count(*) FROM document_chunks WHERE tenant_id=2').fetchone()[0] > 0
    assert not db.execute('PRAGMA foreign_key_check').fetchall()


def test_failure_concurrent_edit_activation_and_rollback(store):
    db, path = store
    put(db)
    old = active(db, 1)['id']
    target = start_migration(db, 1)
    with pytest.raises(RuntimeError):
        backfill(db, 1, target, fail=True)
    with pytest.raises(ValueError, match='incomplete'):
        activate(db, 1, target)
    finish(db, target)
    other = connect(path)
    put(other, text='Refunds now require a receipt.')
    other.close()
    with pytest.raises(ValueError, match='incomplete'):
        activate(db, 1, target)
    assert VectorIndex().search_active(db, 1, 'refund')['generation_id'] == old
    finish(db, target)
    competing = start_migration(db, 1, revision='competing', dimensions=8)
    finish(db, competing)
    activate(db, 1, target)
    with pytest.raises(ValueError, match='stale'):
        activate(db, 1, competing)
    assert VectorIndex().search_active(db, 1, 'refund')['dimensions'] == 6
    put(db, text='Refunds now require a receipt within fourteen days.')
    assert complete(db, 1, generation(db, 1, old))
    activate(db, 1, old, rollback=True)
    assert VectorIndex().search_active(db, 1, 'refund')['dimensions'] == 4
