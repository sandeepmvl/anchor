import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.storage import initialize_sandbox, migrate_tenant_ownership, TenantStorage
from anchor_ai.tenant_scope import authenticated_scope, ScopeError
from anchor_ai.dense_retriever import DenseRetriever
from anchor_ai.semantic_cache import Generated
from anchor_ai.tests.test_tenant_isolation import (
    A, B, CREDENTIALS, QUESTION, SECRET, generation, embedding_client, cache, cache_snapshot,
)


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'anchor.sqlite3'
        initialize_sandbox(path)
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        for tenant, vector in ((1, [0.8, 0.6]), (2, [1.0, 0.0])):
            text = 'A policy: review required' if tenant == 1 else 'B policy: ' + SECRET
            db.execute('''INSERT INTO documents
                (id,tenant_id,external_id,source_name,content_hash,text,status)
                VALUES(?,?,?,?,?,?,'ready')''', (tenant, tenant, 'policy', 'offline', str(tenant), text))
            db.execute('''INSERT INTO document_chunks
                (id,tenant_id,document_id,chunk_index,is_leaf,text,embedding)
                VALUES(?,?,?,0,1,?,?)''', (tenant, tenant, tenant, text, json.dumps(vector)))
        db.commit()
        migrate_tenant_ownership(db)
        gen = generation(db)
        global_winner = max(gen['chunks'], key=lambda row: row['vector'][0])
        assert global_winner['tenant_id'] == 2
        print('unsafe global winner: tenant B')
        store = TenantStorage(db)
        shared_cache = cache()
        with authenticated_scope(B, CREDENTIALS):
            store.write_conversation(7, {'answer': SECRET})
            b_key = store.save_checkpoint(7, 'same-thread', {'answer': SECRET})
            shared_cache.store(cache_snapshot(2), QUESTION, [1, 0], Generated(SECRET, public_faq=True))
        with authenticated_scope(A, CREDENTIALS):
            dense = DenseRetriever(gen, 1, embedding_client())
            hits = dense.search(QUESTION, 1)
            assert hits and all(hit['tenant_id'] == 1 and SECRET not in hit['text'] for hit in hits)
            assert store.conversation(7) is None
            assert store.restore_checkpoint(7, 'same-thread') is None
            assert store.checkpoint_key(7, 'same-thread') != b_key
            assert shared_cache.lookup(cache_snapshot(1), QUESTION, [1, 0]) is None
            try:
                store.write_conversation(7, {}, tenant_id=2)
            except ScopeError:
                pass
            else:
                raise AssertionError('mis-scoped write accepted')
            print('tenant A candidates:', dense.index.ntotal)
            print('tenant A answer:', hits[0]['text'])
            print('tenant B chunks: 0; foreign conversation: absent; cache: tenant-local miss; checkpoint: absent')
            print('mis-scoped write: rejected')
        try:
            store.candidates()
        except ScopeError:
            print('missing scope: rejected')
        else:
            raise AssertionError('missing scope accepted')
        db.close()
        print('offline evidence only: production routing, Rails login, and hosted vector infrastructure not verified')


if __name__ == '__main__':
    main()
