import asyncio
import json
import sqlite3
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI, Request

from anchor_ai.tenant_scope import (
    ScopeError, TenantScopeMiddleware, authenticated_scope, require_tenant,
)
from anchor_ai.storage import initialize_sandbox, migrate_tenant_ownership, TenantStorage
from anchor_ai.dense_retriever import DenseRetriever, RecordedEmbeddingClient
from anchor_ai.sparse_retriever import SparseRetriever
from anchor_ai.semantic_cache import Snapshot, Generated
from anchor_ai.semantic_cache import TenantSemanticCache
from anchor_ai.checkpoints import CheckpointStore

A = 'offline-tenant-a-credential-000000000000'
B = 'offline-tenant-b-credential-000000000000'
CREDENTIALS = {A: 1, B: 2}
QUESTION = 'What is our refund policy?'
SECRET = 'B_ONLY_COPPER_MOON'


@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'isolation.sqlite3'
    initialize_sandbox(path)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    for tenant, vector in ((1, [0.8, 0.6]), (2, [1.0, 0.0])):
        text = 'refund policy review' if tenant == 1 else 'refund policy ' + SECRET
        db.execute('''INSERT INTO documents
            (id,tenant_id,external_id,source_name,content_hash,text,status)
            VALUES (?,?,?,?,?,?,'ready')''',
            (tenant, tenant, 'policy', 'offline', str(tenant), text))
        db.execute('''INSERT INTO retrieval_generations
            (id,tenant_id,model,revision,dimensions,status,active,vector_column)
            VALUES (?,?,'offline','v1',2,'ready',1,'embedding_v2')''', (tenant, tenant))
        db.execute('''INSERT INTO document_chunks
            (id,tenant_id,document_id,chunk_index,is_leaf,text,embedding,
             embedding_model,embedding_revision,embedding_dimensions,
             retrieval_generation_id,embedding_v2)
            VALUES (?,?,?,0,1,?,?,'offline','v1',2,?,?)''',
            (tenant, tenant, tenant, text, json.dumps(vector), tenant, json.dumps(vector)))
    db.execute('''UPDATE document_chunks SET
        source_hash=(SELECT content_hash FROM documents WHERE documents.id=document_chunks.document_id),
        chunker_version='sentence-v1' ''')
    db.commit()
    migrate_tenant_ownership(db)
    yield db
    db.close()


def generation(db):
    return dict(generation_id=1, model='offline', revision='v1', dimensions=2,
                metric='cosine', normalize=True,
                chunks=[dict(id=row['id'], tenant_id=row['tenant_id'], text=row['text'],
                             vector=json.loads(row['embedding']))
                        for row in db.execute('SELECT * FROM document_chunks')])


def embedding_client():
    return RecordedEmbeddingClient({(QUESTION, 'offline', 'v1', 2): [1.0, 0.0]})


def cache_snapshot(tenant):
    return Snapshot(tenant, 'agent', 'public', 'en', 'permissions-v1', 'faq',
                    'kb-v1', 'offline-v1', 'prompt-v1', 'model-v1', 'retrieval-v1',
                    eligibility='public_faq')


def cache():
    return TenantSemanticCache(emit=lambda event: None)


def test_dense_prefilters_before_vector_processing(database):
    gen = generation(database)
    assert gen['chunks'][1]['vector'] == [1.0, 0.0]
    with authenticated_scope(A, CREDENTIALS):
        dense = DenseRetriever(gen, 1, embedding_client())
        assert dense.index.ntotal == 1
        assert [hit['id'] for hit in dense.search(QUESTION, 1)] == [1]
        assert dense.last_trace['candidate_count'] == 1
        assert dense.last_trace['filter_path'] == 'tenant_prefilter'
        gen['chunks'][1]['vector'] = [float('nan')]
        assert DenseRetriever(gen, 1, embedding_client()).search(QUESTION, 1)[0]['id'] == 1
        with pytest.raises(ScopeError):
            DenseRetriever(gen, 2, embedding_client())
    with pytest.raises(ScopeError):
        dense.search(QUESTION, 1)


def test_sparse_partition_is_selected_before_bm25(database, monkeypatch):
    import anchor_ai.sparse_retriever as module
    observed = []
    original = module.BM25Okapi

    def inspect(tokens, **options):
        observed.extend(tokens)
        assert not any(SECRET.lower() in words for words in tokens)
        return original(tokens, **options)

    monkeypatch.setattr(module, 'BM25Okapi', inspect)
    sparse = SparseRetriever(database)
    with authenticated_scope(A, CREDENTIALS):
        hits = sparse.search('refund policy', 1)
        assert hits and {hit['tenant_id'] for hit in hits} == {1}
        assert observed
        with pytest.raises(ScopeError):
            sparse.search('refund', 2)
    # Internal offline adapters retain the earlier trusted-argument contract.
    # Public HTTP and TenantStorage still reject absent authentication.
    assert {hit['tenant_id'] for hit in sparse.search('refund', 1)} == {1}


def test_storage_conversation_and_checkpoint_scope(database):
    store = TenantStorage(database)
    with authenticated_scope(B, CREDENTIALS):
        assert store.document(2)['tenant_id'] == 2
        store.write_conversation(7, {'answer': SECRET})
        b_key = store.save_checkpoint(7, 'shared-thread', {'answer': SECRET})
        assert store.restore_checkpoint(7, 'shared-thread')['answer'] == SECRET
    with authenticated_scope(A, CREDENTIALS):
        assert store.document(1)['tenant_id'] == 1
        assert store.document(2) is None
        assert {row['tenant_id'] for row in store.candidates()} == {1}
        assert store.conversation(7) is None
        assert store.restore_checkpoint(7, 'shared-thread') is None
        a_key = store.save_checkpoint(7, 'shared-thread', {'answer': 'A local'})
        assert a_key != b_key
        assert json.loads(a_key)[0] == 1
        assert store.restore_checkpoint(7, 'shared-thread')['answer'] == 'A local'
        with pytest.raises(ScopeError):
            store.write_conversation(7, {'answer': 'forged'}, tenant_id=2)
        with pytest.raises(ScopeError):
            store.save_checkpoint(7, 'shared-thread', {}, tenant_id=2)
    statements = []
    database.set_trace_callback(statements.append)
    with pytest.raises(ScopeError):
        store.candidates()
    with pytest.raises(ScopeError):
        store.conversation(7)
    assert statements == []
    database.set_trace_callback(None)


def test_semantic_cache_authenticates_snapshot_and_key():
    shared = cache()
    with authenticated_scope(B, CREDENTIALS):
        assert shared.store(cache_snapshot(2), QUESTION, [1, 0], Generated(SECRET, public_faq=True))
        assert shared.lookup(cache_snapshot(2), QUESTION, [1, 0]) == SECRET
        b_key = shared.partition_key(cache_snapshot(2))
    with authenticated_scope(A, CREDENTIALS):
        assert shared.lookup(cache_snapshot(1), QUESTION, [1, 0]) is None
        assert shared.partition_key(cache_snapshot(1)) != b_key
        with pytest.raises(ScopeError):
            shared.lookup(cache_snapshot(2), QUESTION, [1, 0])
        with pytest.raises(ScopeError):
            shared.store(cache_snapshot(2), QUESTION, [1, 0], Generated('forged', public_faq=True))
        assert shared.store(cache_snapshot(1), QUESTION, [1, 0], Generated('A local', public_faq=True))
        assert shared.lookup(cache_snapshot(1), QUESTION, [1, 0]) == 'A local'
    with pytest.raises(ScopeError):
        shared.lookup(cache_snapshot(1), QUESTION, [1, 0])


def test_migration_is_atomic_and_idempotent(database):
    migrate_tenant_ownership(database)
    with pytest.raises(sqlite3.IntegrityError):
        with database:
            database.execute('UPDATE document_chunks SET tenant_id=2 WHERE id=1')
    assert database.execute('SELECT tenant_id FROM document_chunks WHERE id=1').fetchone()[0] == 1
    legacy = sqlite3.connect(':memory:')
    legacy.executescript('''CREATE TABLE documents(id INTEGER PRIMARY KEY);
        CREATE TABLE document_chunks(id INTEGER PRIMARY KEY,document_id INTEGER,parent_id INTEGER);
        INSERT INTO documents VALUES(1); INSERT INTO document_chunks VALUES(1,1,NULL);''')
    with pytest.raises(ValueError):
        migrate_tenant_ownership(legacy)
    assert 'tenant_id' not in {r[1] for r in legacy.execute('PRAGMA table_info(documents)')}
    migrate_tenant_ownership(legacy, {1: 1})
    migrate_tenant_ownership(legacy, {1: 1})
    assert legacy.execute('SELECT tenant_id FROM document_chunks').fetchone()[0] == 1
    with pytest.raises(ValueError):
        migrate_tenant_ownership(legacy, {1: 2})
    legacy.close()


def test_http_identity_is_verified_and_reset():
    inner = FastAPI()
    calls = []

    @inner.post('/retrieve')
    async def retrieve(request: Request):
        calls.append(require_tenant())
        return {'tenant_id': require_tenant()}

    wrapped = TenantScopeMiddleware(inner, CREDENTIALS)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapped), base_url='http://offline') as client:
            missing = await client.post('/retrieve', json={'tenant_id': 1})
            assert missing.status_code == 403
            forged = await client.post('/retrieve', headers={'X-Anchor-Tenant-Credential': A}, json={'tenant_id': 2})
            assert forged.status_code == 403
            valid = await client.post('/retrieve', headers={'X-Anchor-Tenant-Credential': A}, json={'tenant_id': 1})
            assert valid.json() == {'tenant_id': 1}
        assert calls == [1]
        with pytest.raises(ScopeError):
            require_tenant()
    asyncio.run(run())


def test_production_missing_credentials_fails_closed(monkeypatch):
    from anchor_ai.app import app
    monkeypatch.setenv('ANCHOR_ENV', 'production')
    monkeypatch.delenv('ANCHOR_TENANT_CREDENTIALS', raising=False)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://offline') as client:
            response = await client.post('/v1/retrieve/hybrid', json={'tenant_id': 1, 'query': QUESTION})
            assert response.status_code == 403
            assert response.json()['detail'] == 'tenant_scope_rejected'
    asyncio.run(run())


def test_existing_cache_class_checks_authenticated_scope():
    from anchor_ai.semantic_cache import SemanticCache
    shared = SemanticCache(emit=lambda event: None)
    with authenticated_scope(A, CREDENTIALS):
        with pytest.raises(ScopeError):
            shared.lookup(cache_snapshot(2), QUESTION, [1, 0])
        with pytest.raises(ScopeError):
            shared.store(cache_snapshot(2), QUESTION, [1, 0], Generated(SECRET, public_faq=True))


def test_product_checkpointer_restores_only_local_state(database):
    path = database.execute('PRAGMA database_list').fetchone()[2]
    checkpoints = CheckpointStore(path)
    with authenticated_scope(B, CREDENTIALS):
        checkpoints.save_state(7, 'shared', {'answer': SECRET})
        assert checkpoints.restore_state(7, 'shared')['answer'] == SECRET
    with authenticated_scope(A, CREDENTIALS):
        assert checkpoints.restore_state(7, 'shared') is None
        with pytest.raises(ScopeError):
            checkpoints.save_state(7, 'shared', {}, tenant_id=2)
        checkpoints.save_state(7, 'shared', {'answer': 'A local'})
        assert checkpoints.restore_state(7, 'shared')['answer'] == 'A local'
    with pytest.raises(ScopeError):
        checkpoints.restore_state(7, 'shared')
