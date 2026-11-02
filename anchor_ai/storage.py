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
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='retrieval_generations'").fetchone():
        from .index_maintenance import maintain_document
        return maintain_document(db, tenant_id, external_id, text, {}, 'sentence-v1',
                                 None, supplied_nodes=make_nodes)
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


def initialize_sandbox(path):
    """Create only a new, explicitly named lab database; never reset one."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        db.executescript('''
        PRAGMA foreign_keys=ON;
        CREATE TABLE documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id INTEGER NOT NULL,
          external_id TEXT NOT NULL, source_name TEXT NOT NULL, content_hash TEXT NOT NULL,
          text TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT, updated_at TEXT,
          metadata_hash TEXT NOT NULL DEFAULT '', metadata TEXT NOT NULL DEFAULT '{}',
          chunker_version TEXT NOT NULL DEFAULT 'sentence-v1', source_scope TEXT,
          UNIQUE(tenant_id, external_id), UNIQUE(tenant_id, content_hash));
        CREATE TABLE retrieval_generations (
          id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id INTEGER NOT NULL,
          model TEXT NOT NULL, revision TEXT NOT NULL, dimensions INTEGER NOT NULL CHECK(dimensions>0),
          status TEXT NOT NULL DEFAULT 'building', active INTEGER NOT NULL DEFAULT 0,
          vector_column TEXT NOT NULL DEFAULT 'embedding_v2'
            CHECK(vector_column IN ('embedding','embedding_v2')),
          base_generation_id INTEGER, epoch INTEGER NOT NULL DEFAULT 0,
          created_at TEXT, updated_at TEXT);
        CREATE UNIQUE INDEX one_active_retrieval_generation
          ON retrieval_generations(tenant_id) WHERE active=1;
        CREATE TABLE document_chunks (
          id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id INTEGER NOT NULL,
          document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          chunk_index INTEGER NOT NULL, parent_id INTEGER REFERENCES document_chunks(id) ON DELETE CASCADE,
          is_leaf INTEGER NOT NULL, text TEXT NOT NULL, window_text TEXT,
          embedding TEXT, embedding_model TEXT, embedding_revision TEXT,
          embedding_dimensions INTEGER, created_at TEXT, updated_at TEXT,
          retrieval_generation_id INTEGER REFERENCES retrieval_generations(id) ON DELETE CASCADE,
          embedding_v2 TEXT, source_hash TEXT, chunker_version TEXT,
          metadata TEXT NOT NULL DEFAULT '{}');
        CREATE UNIQUE INDEX unique_generation_chunk
          ON document_chunks(document_id,retrieval_generation_id,chunk_index);
        CREATE UNIQUE INDEX unique_legacy_chunk ON document_chunks(document_id,chunk_index)
          WHERE retrieval_generation_id IS NULL;
        ''')
    finally:
        db.close()


def connect_memory(path):
    """Memory-only connection: no dependency on document or Rails tables."""
    db = sqlite3.connect(path, timeout=2)
    db.row_factory = sqlite3.Row
    try:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS agent_memories (
          tenant_id INTEGER NOT NULL CHECK(tenant_id>0),
          conversation_id INTEGER NOT NULL CHECK(conversation_id>0),
          key TEXT NOT NULL, value TEXT NOT NULL,
          updated_at REAL NOT NULL, expires_at REAL NOT NULL,
          UNIQUE(tenant_id,conversation_id,key));
        CREATE TABLE IF NOT EXISTS episodic_memory_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id INTEGER NOT NULL CHECK(tenant_id>0),
          conversation_id INTEGER NOT NULL CHECK(conversation_id>0),
          summary TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS memory_fact_expiry
          ON agent_memories(tenant_id,conversation_id,expires_at);
        CREATE INDEX IF NOT EXISTS memory_episode_recency
          ON episodic_memory_log(tenant_id,conversation_id,created_at DESC,id DESC);
        CREATE INDEX IF NOT EXISTS memory_episode_expiry
          ON episodic_memory_log(tenant_id,conversation_id,expires_at);
        ''')
        return db
    except Exception:
        db.close()
        raise
