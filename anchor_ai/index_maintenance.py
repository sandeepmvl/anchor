import hashlib
import json
import math
import os
from contextlib import contextmanager
from .pii import prepare


def valid_tenant(tenant):
    if type(tenant) is not int or tenant <= 0:
        raise ValueError('invalid_tenant')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@contextmanager
def transaction(db):
    db.execute('BEGIN IMMEDIATE')
    try:
        yield
        db.commit()
    except Exception:
        db.rollback()
        raise


class EmbeddingClient:
    def __init__(self):
        self.calls = 0

    def embed(self, text, generation):
        self.calls += 1
        size = generation['dimensions']
        provider = os.environ.get('ANCHOR_EMBEDDING_PROVIDER', 'offline')
        if provider == 'offline':
            seed = (generation['model'] + ':' + generation['revision'] + ':' + text).encode()
            vector = [(int.from_bytes(hashlib.sha256(seed + str(i).encode()).digest()[:4], 'big')
                       / 2**31) - 1 for i in range(size)]
        elif provider == 'openai':
            from openai import OpenAI
            vector = OpenAI().embeddings.create(model=generation['model'], input=text,
                                                dimensions=size).data[0].embedding
        else:
            raise ValueError('unknown_embedding_provider')
        validate_vector(vector, generation)
        return vector


def validate_vector(vector, generation):
    if len(vector) != generation['dimensions'] or not all(math.isfinite(v) for v in vector):
        raise ValueError('invalid_vector')
    if not any(vector):
        raise ValueError('zero_vector')


def active(db, tenant):
    valid_tenant(tenant)
    row = db.execute('SELECT * FROM retrieval_generations WHERE tenant_id=? AND active=1',
                     (tenant,)).fetchone()
    return dict(row) if row else None


def bootstrap(db, tenant):
    current = active(db, tenant)
    if current:
        return current
    legacy = db.execute('''SELECT DISTINCT embedding_model,embedding_revision,embedding_dimensions
        FROM document_chunks WHERE tenant_id=? AND is_leaf=1 AND retrieval_generation_id IS NULL''',
        (tenant,)).fetchall()
    if len(legacy) > 1:
        raise ValueError('mixed_legacy_generation')
    model, revision, dimensions = tuple(legacy[0]) if legacy else ('anchor-recorded-embedding-v1', 'authored-fixture-v1', 4)
    cursor = db.execute('''INSERT INTO retrieval_generations
        (tenant_id,model,revision,dimensions,status,active,vector_column)
        VALUES(?,?,?,?,'ready',1,'embedding')''', (tenant, model, revision, dimensions))
    db.execute('''UPDATE document_chunks SET retrieval_generation_id=?,
        source_hash=(SELECT content_hash FROM documents WHERE id=document_id),
        chunker_version=(SELECT chunker_version FROM documents WHERE id=document_id)
        WHERE tenant_id=? AND retrieval_generation_id IS NULL''', (cursor.lastrowid, tenant))
    return active(db, tenant)


def bump(db, tenant):
    db.execute('UPDATE retrieval_generations SET epoch=epoch+1 WHERE tenant_id=?', (tenant,))


def write_chunks(db, document, generation, client, nodes=None):
    from .ingestion import sentence_nodes
    nodes = nodes if nodes is not None else sentence_nodes(document['text'])
    nodes = [dict(n) if isinstance(n, dict) else vars(n) for n in nodes]
    if not nodes or not any(n['is_leaf'] for n in nodes):
        raise ValueError('empty_nodes')
    parents = {}
    for node in sorted(nodes, key=lambda n: n['is_leaf']):
        text, window = node['text'], node.get('window_text')
        if text != prepare(text) or (window and window != prepare(window)):
            raise ValueError('unredacted_node')
        leaf = node['is_leaf']
        vector = client.embed(text, generation) if leaf else None
        column = generation['vector_column']
        if column not in {'embedding', 'embedding_v2'}:
            raise ValueError('unknown_vector_column')
        cursor = db.execute('''INSERT INTO document_chunks
            (tenant_id,document_id,retrieval_generation_id,chunk_index,parent_id,is_leaf,text,
             window_text,embedding,embedding_v2,embedding_model,embedding_revision,
             embedding_dimensions,source_hash,chunker_version,metadata,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)''',
            (document['tenant_id'], document['id'], generation['id'], node['chunk_index'],
             parents[node['parent_index']] if leaf else None, int(leaf), text, window,
             canonical(vector) if leaf and column == 'embedding' else None,
             canonical(vector) if leaf and column == 'embedding_v2' else None,
             generation['model'] if leaf else None, generation['revision'] if leaf else None,
             generation['dimensions'] if leaf else None, document['content_hash'],
             document['chunker_version'], document['metadata']))
        if not leaf:
            parents[node['chunk_index']] = cursor.lastrowid


def maintain_document(db, tenant, external_id, text, metadata, version, scope,
                      client=None, supplied_nodes=None, fail=False):
    valid_tenant(tenant)
    if text != prepare(text) or any(prepare(v) != v for v in [external_id, canonical(metadata)]):
        raise ValueError('unredacted_payload')
    if version not in {'sentence-v1', 'sentence-v2'} or not text.strip():
        raise ValueError('invalid_document')
    client = client or EmbeddingClient()
    content_hash, metadata_json = digest(text), canonical(metadata)
    metadata_hash = digest(metadata_json)
    with transaction(db):
        bootstrap(db, tenant)
        old = db.execute('SELECT * FROM documents WHERE tenant_id=? AND external_id=?',
                         (tenant, external_id)).fetchone()
        replace = old is None or old['content_hash'] != content_hash or old['chunker_version'] != version
        if old and old['source_scope'] != scope:
            raise ValueError('source_scope_conflict')
        if old and not replace and old['metadata_hash'] == metadata_hash:
            return {'document_id': old['id'], 'status': 'ready', 'skipped': True}
        if old is None:
            db.execute('''INSERT INTO documents
                (tenant_id,external_id,source_name,text,content_hash,status,metadata,metadata_hash,
                 chunker_version,source_scope,created_at,updated_at)
                VALUES(?,?,?,?,?,'ready',?,?,?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)''',
                (tenant, external_id, 'policy-upload', text, content_hash, metadata_json,
                 metadata_hash, version, scope))
        else:
            db.execute('''UPDATE documents SET text=?,content_hash=?,metadata=?,metadata_hash=?,
                chunker_version=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND tenant_id=?''',
                (text, content_hash, metadata_json, metadata_hash, version, old['id'], tenant))
        document = dict(db.execute('SELECT * FROM documents WHERE tenant_id=? AND external_id=?',
                                   (tenant, external_id)).fetchone())
        if replace:
            db.execute('DELETE FROM document_chunks WHERE document_id=? AND tenant_id=?',
                       (document['id'], tenant))
            if fail:
                raise RuntimeError('injected replacement failure')
            nodes = supplied_nodes() if supplied_nodes else None
            for generation in db.execute("SELECT * FROM retrieval_generations WHERE tenant_id=? AND status='ready'", (tenant,)).fetchall():
                write_chunks(db, document, dict(generation), client, nodes)
        else:
            db.execute('UPDATE document_chunks SET metadata=? WHERE tenant_id=? AND document_id=?',
                       (metadata_json, tenant, document['id']))
        bump(db, tenant)
        return {'document_id': document['id'], 'status': 'ready', 'skipped': False,
                'action': 'replace' if replace else 'metadata'}


def delete_document(db, tenant, external_id):
    valid_tenant(tenant)
    with transaction(db):
        # The database cascade reaches every generation, not just the active one.
        db.execute('DELETE FROM documents WHERE tenant_id=? AND external_id=?', (tenant, external_id))
        bump(db, tenant)


def start_migration(db, tenant, model='anchor-offline-v2', revision='v2', dimensions=6):
    valid_tenant(tenant)
    if not model or not revision or type(dimensions) is not int or dimensions < 1:
        raise ValueError('invalid_generation')
    with transaction(db):
        base = bootstrap(db, tenant)
        pending = db.execute('''SELECT id FROM retrieval_generations WHERE tenant_id=? AND
            model=? AND revision=? AND dimensions=? AND status='building' AND base_generation_id=?''',
            (tenant, model, revision, dimensions, base['id'])).fetchone()
        if pending:
            return pending['id']
        return db.execute('''INSERT INTO retrieval_generations
            (tenant_id,model,revision,dimensions,base_generation_id)
            VALUES(?,?,?,?,?)''', (tenant, model, revision, dimensions, base['id'])).lastrowid


def generation(db, tenant, identity):
    row = db.execute('SELECT * FROM retrieval_generations WHERE tenant_id=? AND id=?',
                     (tenant, identity)).fetchone()
    if row is None:
        raise ValueError('unknown_generation')
    return dict(row)


def complete(db, tenant, gen):
    documents = db.execute('SELECT * FROM documents WHERE tenant_id=?', (tenant,)).fetchall()
    for document in documents:
        rows = db.execute('''SELECT * FROM document_chunks WHERE tenant_id=? AND document_id=?
            AND retrieval_generation_id=?''', (tenant, document['id'], gen['id'])).fetchall()
        from .ingestion import sentence_nodes
        expected = sentence_nodes(document['text'])
        if len(rows) != len(expected):
            return False
        expected = {n['chunk_index']: n for n in expected}
        for row in rows:
            node = expected.get(row['chunk_index'])
            if not node or row['source_hash'] != document['content_hash'] or row['chunker_version'] != document['chunker_version']:
                return False
            if row['text'] != node['text'] or row['window_text'] != node.get('window_text'):
                return False
            if row['is_leaf']:
                if (row['embedding_model'], row['embedding_revision'], row['embedding_dimensions']) != (gen['model'], gen['revision'], gen['dimensions']):
                    return False
                validate_vector(json.loads(row[gen['vector_column']]), gen)
    return True


def backfill(db, tenant, identity, batch_size=2, client=None, fail=False):
    if batch_size < 1:
        raise ValueError('invalid_batch_size')
    client = client or EmbeddingClient()
    done = 0
    # One transaction per document; committed documents survive a later failure.
    ids = [r['id'] for r in db.execute('SELECT id FROM documents WHERE tenant_id=? ORDER BY id', (tenant,))]
    for document_id in ids:
        with transaction(db):
            gen = generation(db, tenant, identity)
            if gen['status'] != 'building' or active(db, tenant)['id'] != gen['base_generation_id']:
                raise ValueError('stale_migration')
            document = db.execute('SELECT * FROM documents WHERE id=? AND tenant_id=?', (document_id, tenant)).fetchone()
            if document is None:
                continue
            row = db.execute('''SELECT source_hash,chunker_version FROM document_chunks
                WHERE document_id=? AND retrieval_generation_id=? LIMIT 1''', (document_id, identity)).fetchone()
            if row and row['source_hash'] == document['content_hash'] and row['chunker_version'] == document['chunker_version']:
                continue
            db.execute('DELETE FROM document_chunks WHERE document_id=? AND retrieval_generation_id=?', (document_id, identity))
            write_chunks(db, dict(document), gen, client)
            if fail:
                raise RuntimeError('injected migration failure')
            done += 1
        if done >= batch_size:
            break
    return done


def activate(db, tenant, identity, rollback=False):
    valid_tenant(tenant)
    with transaction(db):
        gen = generation(db, tenant, identity)
        current = active(db, tenant)
        if not rollback and gen['base_generation_id'] != current['id']:
            raise ValueError('stale_migration')
        if rollback and gen['status'] != 'ready':
            raise ValueError('rollback_requires_ready_generation')
        if not complete(db, tenant, gen):
            raise ValueError('incomplete_generation')
        db.execute('UPDATE retrieval_generations SET active=0 WHERE tenant_id=?', (tenant,))
        db.execute("UPDATE retrieval_generations SET active=1,status='ready' WHERE tenant_id=? AND id=?", (tenant, identity))
        bump(db, tenant)
