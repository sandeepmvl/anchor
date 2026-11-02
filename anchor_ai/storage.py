import json
import sqlite3
from pathlib import Path
from .pii import prepare


def connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=rw', uri=True, timeout=2)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('SELECT status FROM documents LIMIT 0')
    return db


def persist(db, tenant_id, external_id, text, content_hash, make_nodes):
    if type(tenant_id) is not int or tenant_id <= 0:
        raise ValueError('invalid_tenant')
    if text != prepare(text):
        raise ValueError('unredacted_payload')
    db.execute('BEGIN IMMEDIATE')
    try:
        previous = db.execute('SELECT content_hash FROM documents WHERE tenant_id=? AND external_id=?',
                              (tenant_id, external_id)).fetchone()
        if previous and previous['content_hash'] != content_hash:
            raise ValueError('external_id_conflict')
        document = db.execute('SELECT * FROM documents WHERE tenant_id=? AND content_hash=?',
                              (tenant_id, content_hash)).fetchone()
        if document and document['status'] == 'ready':
            db.commit()
            return {'document_id': document['id'], 'status': 'ready', 'skipped': True}
        db.execute('''INSERT INTO documents
            (tenant_id,external_id,source_name,content_hash,text,status,created_at,updated_at)
            VALUES(?,?,?,?,?,'processing',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
            ON CONFLICT(tenant_id,content_hash) DO NOTHING''',
            (tenant_id, external_id, 'policy-upload', content_hash, text))
        document = db.execute('SELECT * FROM documents WHERE tenant_id=? AND content_hash=?',
                              (tenant_id, content_hash)).fetchone()
        nodes = make_nodes()
        parents = {}
        sql = '''INSERT INTO document_chunks
          (tenant_id,document_id,chunk_index,parent_id,is_leaf,text,window_text,embedding,
           embedding_model,embedding_revision,embedding_dimensions,created_at,updated_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
          ON CONFLICT(document_id,chunk_index) DO NOTHING'''
        for node in sorted(nodes, key=lambda node: node.is_leaf):
            parent_id = parents[node.parent_index] if node.is_leaf else None
            if node.text != prepare(node.text) or (node.window_text and node.window_text != prepare(node.window_text)):
                raise ValueError('unredacted_node')
            db.execute(sql, (document['tenant_id'], document['id'], node.chunk_index,
                parent_id, int(node.is_leaf), node.text, node.window_text,
                json.dumps(node.embedding) if node.embedding is not None else None,
                node.embedding_model, node.embedding_revision, node.embedding_dimensions))
            if not node.is_leaf:
                row = db.execute('''SELECT id FROM document_chunks
                    WHERE document_id=? AND tenant_id=? AND chunk_index=? AND is_leaf=0''',
                    (document['id'], document['tenant_id'], node.chunk_index)).fetchone()
                if row is None:
                    raise ValueError('parent_identity_conflict')
                parents[node.chunk_index] = row['id']
        if not nodes or not any(node.is_leaf for node in nodes):
            raise ValueError('empty_nodes')
        db.execute("UPDATE documents SET status='ready', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                   (document['id'],))
        db.commit()
        return {'document_id': document['id'], 'status': 'ready', 'skipped': False}
    except Exception:
        db.rollback()
        raise
