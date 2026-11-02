import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchor_ai.index_maintenance import (
    delete_document, digest, maintain_document, retrieval_snapshot,
)
from anchor_ai.pii import prepare
from anchor_ai.sparse_retriever import SparseRetriever, tokenize


class DemoEmbeddingClient:
    """Authored semantic preference, not measured model behavior."""
    def embed(self, text, generation):
        if generation['dimensions'] != 4:
            raise ValueError('demo_requires_four_dimensions')
        if '4821' in text and text.startswith(('Customer:', 'Agent:')):
            return [0.8, 0.6, 0.0, 0.0]
        if text.startswith('Shipping policy:') or (
                '?' in text and not text.startswith(('Customer:', 'Agent:'))):
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]


def fixture_db():
    """An isolated schema fixture for maintenance tests, never an app migration."""
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, tenant_id INTEGER NOT NULL,
            external_id TEXT, source_name TEXT, text TEXT, content_hash TEXT,
            status TEXT, metadata TEXT, metadata_hash TEXT, chunker_version TEXT,
            source_scope TEXT, created_at TEXT, updated_at TEXT,
            UNIQUE(tenant_id, external_id));
        CREATE TABLE retrieval_generations (
            id INTEGER PRIMARY KEY, tenant_id INTEGER NOT NULL, model TEXT,
            revision TEXT, dimensions INTEGER, status TEXT DEFAULT 'building',
            active INTEGER DEFAULT 0, epoch INTEGER DEFAULT 0,
            vector_column TEXT DEFAULT 'embedding', base_generation_id INTEGER);
        CREATE UNIQUE INDEX one_active ON retrieval_generations(tenant_id)
            WHERE active=1;
        CREATE TABLE document_chunks (
            id INTEGER PRIMARY KEY, tenant_id INTEGER NOT NULL,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            retrieval_generation_id INTEGER REFERENCES retrieval_generations(id),
            chunk_index INTEGER, parent_id INTEGER, is_leaf INTEGER,
            text TEXT, window_text TEXT, embedding TEXT, embedding_v2 TEXT,
            embedding_model TEXT, embedding_revision TEXT,
            embedding_dimensions INTEGER, source_hash TEXT, chunker_version TEXT,
            metadata TEXT, created_at TEXT, updated_at TEXT);
    ''')
    return db


def demo_nodes(text):
    # Preserve fixture line boundaries, then prepare every persisted node.
    nodes = [dict(chunk_index=0, parent_index=None, is_leaf=False,
                  text=prepare(text))]
    for i, line in enumerate(text.splitlines(), 1):
        line = prepare(line)
        if line.strip():
            nodes.append(dict(chunk_index=i, parent_index=0, is_leaf=True,
                              text=line, window_text=line))
    return nodes


def put(db, tenant, external_id, text):
    return maintain_document(
        db, tenant, prepare(external_id), prepare(text), {}, 'sentence-v1', 'bm25-demo',
        client=DemoEmbeddingClient(), supplied_nodes=lambda: demo_nodes(text))


def seed_demo(db, tenant):
    transcript = (ROOT / 'anchor_ai/tests/fixtures/kb/transcript.txt').read_text()
    old = transcript.split('Customer: Where is order 4821?')[0]
    put(db, tenant, 'transcript', old)
    old_hash = db.execute('SELECT content_hash FROM documents WHERE tenant_id=?',
                          (tenant,)).fetchone()[0]
    put(db, tenant, 'transcript', transcript)
    new_hash = db.execute('SELECT content_hash FROM documents WHERE tenant_id=?',
                          (tenant,)).fetchone()[0]
    assert old_hash != new_hash and new_hash == digest(prepare(transcript))
    print('transcript hash before:', old_hash)
    print('transcript hash after: ', new_hash)
    put(db, tenant, 'policy',
        'Shipping policy: shipping delay refund policy requires a tracking review.')
    for name, text in [('login', 'Password reset authentication access.'),
                       ('hours', 'Opening hours holidays weekends.'),
                       ('invoice', 'Invoice tax receipts payment.'),
                       ('warranty', 'Warranty repair replacement service.')]:
        put(db, tenant, name, text)
    put(db, tenant + 1, 'private', 'Other tenant secret order 4821.')


def cosine_reference(db, query, tenant):
    gen, rows = retrieval_snapshot(db, tenant)
    if gen is None:
        return []
    q = DemoEmbeddingClient().embed(query, gen)
    hits = []
    for row in rows:
        vector = json.loads(row[gen['vector_column']])
        score = sum(a*b for a, b in zip(q, vector)) / (
            math.sqrt(sum(v*v for v in q)) * math.sqrt(sum(v*v for v in vector)))
        hits.append(dict(id=row['id'], tenant_id=tenant, text=row['text'], score=score))
    return sorted(hits, key=lambda row: (-row['score'], row['id']))[:5]


def show(label, hits):
    print(label, json.dumps(hits, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query')
    parser.add_argument('--tenant', type=int, default=1)
    parser.add_argument('--db', help='Existing Anchor SQLite database; read-only')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    if args.tenant <= 0:
        parser.error('tenant must be positive')
    if args.db and (args.compare or args.delete):
        parser.error('comparison and deletion are isolated-demo options only')
    if args.db:
        uri = Path(args.db).resolve().as_uri() + '?mode=ro'
        db = sqlite3.connect(uri, uri=True)
        db.row_factory = sqlite3.Row
    else:
        db = fixture_db()
        seed_demo(db, args.tenant)
    try:
        sparse = SparseRetriever(db)
        gen, _ = retrieval_snapshot(db, args.tenant)
        print('tenant:', args.tenant, 'snapshot:',
              (gen['id'], gen['epoch']) if gen else None)
        print('query_tokens:', tokenize(args.query))
        hits = sparse.search(args.query, args.tenant)
        show('BM25', hits)
        assert all(hit['tenant_id'] == args.tenant for hit in hits)
        if args.compare:
            show('DENSE offline authored-vector cosine reference',
                 cosine_reference(db, args.query, args.tenant))
            complaint = 'Where is order 4821?'
            lexical = sparse.search(complaint, args.tenant)
            dense = cosine_reference(db, complaint, args.tenant)
            show('same complaint BM25', lexical)
            show('same complaint DENSE offline reference', dense)
            assert lexical and '4821' in tokenize(lexical[0]['text'])
            assert dense and dense[0]['text'].startswith('Shipping policy:')
            print('DEMO: same order-specific complaint; dense offline reference favors '
                  'generic policy, BM25 surfaces synthetic order 4821.')
        if args.delete:
            before = sparse.search('4821', args.tenant)
            assert before
            show('before deletion', before)
            delete_document(db, args.tenant, 'transcript')
            after = sparse.search('4821', args.tenant)
            show('after deletion, refreshed same retriever', after)
            assert after == []
            print('PASS: order 4821 disappeared after document deletion.')
    finally:
        db.close()


if __name__ == '__main__':
    main()
