class NativeSessionStore:
    """Local authorization metadata, never provider-held conversation history."""
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=2, isolation_level=None)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS native_response_bindings (
          provider TEXT NOT NULL, response_id TEXT NOT NULL,
          tenant_id INTEGER NOT NULL, conversation_id INTEGER NOT NULL,
          PRIMARY KEY(provider,response_id));
        CREATE TABLE IF NOT EXISTS native_action_claims (
          tenant_id INTEGER NOT NULL, conversation_id INTEGER NOT NULL,
          action_hash TEXT NOT NULL,
          PRIMARY KEY(tenant_id,conversation_id,action_hash));
        ''')

    def authorize(self, provider, response_id, scope):
        row = self.db.execute('''SELECT tenant_id,conversation_id
          FROM native_response_bindings WHERE provider=? AND response_id=?''',
          (provider, response_id)).fetchone()
        if row != (scope.tenant_id, scope.conversation_id):
            raise ValueError('native_session_scope_mismatch')

    def remember(self, provider, response_id, scope):
        if not isinstance(response_id, str) or not response_id:
            raise ValueError('invalid_native_response_id')
        self.db.execute('''INSERT OR IGNORE INTO native_response_bindings
          VALUES (?,?,?,?)''',
          (provider, response_id, scope.tenant_id, scope.conversation_id))
        self.authorize(provider, response_id, scope)

    def claim(self, scope, call, tool):
        import hashlib
        raw = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
        arguments = tool.arguments.model_validate(raw).model_dump(mode='json')
        identity = json.dumps([call.name, arguments], sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(identity.encode()).hexdigest()
        try:
            self.db.execute('INSERT INTO native_action_claims VALUES (?,?,?)',
                            (scope.tenant_id, scope.conversation_id, digest))
        except sqlite3.IntegrityError:
            raise ValueError('native_duplicate_effect') from None

    def close(self):
        self.db.close()


import json
import sqlite3
from pathlib import Path
from .pii import prepare


def recovery_inventory(db):
    """Canonical per-record evidence for every application table, including ledgers."""
    import hashlib
    import base64

    def encode(value):
        return {'base64': base64.b64encode(value).decode()} if isinstance(value, bytes) else value

    result = {}
    names = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for name in names:
        quoted = '"' + name.replace('"', '""') + '"'
        columns = [r[1] for r in db.execute('PRAGMA table_info(' + quoted + ')')]
        records = []
        for row in db.execute('SELECT * FROM ' + quoted):
            raw = json.dumps([encode(v) for v in row], ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False)
            records.append(hashlib.sha256(raw.encode()).hexdigest())
        result[name] = {'columns': columns, 'count': len(records),
                        'record_checksums': sorted(records)}
    return result


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
        return maintain_document(db, tenant_id, external_id, text, {}, 'sentence-v1', None, supplied_nodes=make_nodes)
    db.execute('BEGIN IMMEDIATE')
    try:
        previous = db.execute('SELECT content_hash FROM documents WHERE tenant_id=? AND external_id=?', (tenant_id, external_id)).fetchone()
        if previous and previous['content_hash'] != content_hash:
            raise ValueError('external_id_conflict')
        document = db.execute('SELECT * FROM documents WHERE tenant_id=? AND content_hash=?', (tenant_id, content_hash)).fetchone()
        if document and document['status'] == 'ready':
            db.commit()
            return {'document_id': document['id'], 'status': 'ready', 'skipped': True}
        db.execute('''INSERT INTO documents
            (tenant_id,external_id,source_name,content_hash,text,status,created_at,updated_at)
            VALUES(?,?,?,?,?,'processing',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
            ON CONFLICT(tenant_id,content_hash) DO NOTHING''', (tenant_id, external_id, 'policy-upload', content_hash, text))
        document = db.execute('SELECT * FROM documents WHERE tenant_id=? AND content_hash=?', (tenant_id, content_hash)).fetchone()
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
            db.execute(sql, (document['tenant_id'], document['id'], node.chunk_index, parent_id, int(node.is_leaf), node.text, node.window_text, json.dumps(node.embedding) if node.embedding is not None else None, node.embedding_model, node.embedding_revision, node.embedding_dimensions))
            if not node.is_leaf:
                row = db.execute('''SELECT id FROM document_chunks
                    WHERE document_id=? AND tenant_id=? AND chunk_index=? AND is_leaf=0''', (document['id'], document['tenant_id'], node.chunk_index)).fetchone()
                if row is None:
                    raise ValueError('parent_identity_conflict')
                parents[node.chunk_index] = row['id']
        if not nodes or not any(node.is_leaf for node in nodes):
            raise ValueError('empty_nodes')
        db.execute("UPDATE documents SET status='ready', updated_at=CURRENT_TIMESTAMP WHERE id=?", (document['id'],))
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
          tenant_id INTEGER NOT NULL CHECK(tenant_id>0), conversation_id INTEGER NOT NULL CHECK(conversation_id>0),
          key TEXT NOT NULL, value TEXT NOT NULL, updated_at REAL NOT NULL, expires_at REAL NOT NULL,
          UNIQUE(tenant_id,conversation_id,key));
        CREATE TABLE IF NOT EXISTS episodic_memory_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id INTEGER NOT NULL CHECK(tenant_id>0),
          conversation_id INTEGER NOT NULL CHECK(conversation_id>0), summary TEXT NOT NULL,
          created_at REAL NOT NULL, expires_at REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS memory_fact_expiry ON agent_memories(tenant_id,conversation_id,expires_at);
        CREATE INDEX IF NOT EXISTS memory_episode_recency ON episodic_memory_log(tenant_id,conversation_id,created_at DESC,id DESC);
        CREATE INDEX IF NOT EXISTS memory_episode_expiry ON episodic_memory_log(tenant_id,conversation_id,expires_at);
        ''')
        return db
    except Exception:
        db.close()
        raise


def connect_escalations(path):
    """Production checkpoint storage; independent of document and Rails tables."""
    db = sqlite3.connect(path, timeout=2, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute('''CREATE TABLE IF NOT EXISTS escalations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tenant_id INTEGER NOT NULL CHECK(tenant_id>0),
          conversation_id INTEGER NOT NULL CHECK(conversation_id>0),
          action_key TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN
            ('pending','approved','executing','completed','rejected','outcome_unknown')),
          context_json TEXT NOT NULL, reviewer TEXT, reason TEXT,
          result_json TEXT, final_json TEXT,
          created_at REAL NOT NULL, updated_at REAL NOT NULL,
          UNIQUE(tenant_id,conversation_id,action_key))''')
        return db
    except Exception:
        db.close()
        raise


from .tenant_scope import require_tenant, scoped_key, ScopeError


def migrate_tenant_ownership(db, audited_documents=None):
    """Privileged, transactional backfill. Never infer tenancy from content."""
    audited_documents = {} if audited_documents is None else audited_documents
    db.execute('BEGIN IMMEDIATE')
    try:
        for table in ('documents', 'document_chunks'):
            columns = {row[1] for row in db.execute('PRAGMA table_info(' + table + ')')}
            if not columns:
                raise ValueError('missing_ownership_table')
            if 'tenant_id' not in columns:
                db.execute('ALTER TABLE ' + table + ' ADD COLUMN tenant_id INTEGER')
        for ident, tenant in db.execute('SELECT id,tenant_id FROM documents').fetchall():
            expected = audited_documents.get(ident, tenant)
            if type(expected) is not int or expected <= 0 or tenant not in (None, expected):
                raise ValueError('unresolved_document_ownership')
            db.execute('UPDATE documents SET tenant_id=? WHERE id=?', (expected, ident))
        rows = db.execute('''SELECT c.id,c.tenant_id,d.tenant_id FROM document_chunks c
            LEFT JOIN documents d ON d.id=c.document_id''').fetchall()
        for ident, tenant, expected in rows:
            if expected is None or tenant not in (None, expected):
                raise ValueError('conflicting_chunk_ownership')
            db.execute('UPDATE document_chunks SET tenant_id=? WHERE id=?', (expected, ident))
        # Existing schemas retain their columns and indexes. Triggers cover both
        # old nullable columns and current NOT NULL columns without table rebuilds.
        for operation in ('INSERT', 'UPDATE'):
            suffix = operation.lower()
            db.execute('''CREATE TRIGGER IF NOT EXISTS tenant_documents_''' + suffix + '''
                BEFORE ''' + operation + ''' ON documents
                WHEN NEW.tenant_id IS NULL OR typeof(NEW.tenant_id)!='integer' OR NEW.tenant_id<=0
                BEGIN SELECT RAISE(ABORT,'invalid_document_tenant'); END''')
            db.execute('''CREATE TRIGGER IF NOT EXISTS tenant_chunks_''' + suffix + '''
                BEFORE ''' + operation + ''' ON document_chunks
                WHEN NEW.tenant_id IS NULL OR NOT EXISTS
                  (SELECT 1 FROM documents d WHERE d.id=NEW.document_id AND d.tenant_id=NEW.tenant_id)
                  OR (NEW.parent_id IS NOT NULL AND NOT EXISTS
                    (SELECT 1 FROM document_chunks p WHERE p.id=NEW.parent_id
                     AND p.tenant_id=NEW.tenant_id AND p.document_id=NEW.document_id))
                BEGIN SELECT RAISE(ABORT,'invalid_chunk_tenant'); END''')
        db.execute('''CREATE TRIGGER IF NOT EXISTS tenant_document_reassignment
            BEFORE UPDATE OF tenant_id ON documents
            WHEN OLD.tenant_id!=NEW.tenant_id
            BEGIN SELECT RAISE(ABORT,'document_tenant_immutable'); END''')
        db.execute('''CREATE TRIGGER IF NOT EXISTS tenant_chunk_reassignment
            BEFORE UPDATE OF tenant_id ON document_chunks
            WHEN OLD.tenant_id!=NEW.tenant_id
            BEGIN SELECT RAISE(ABORT,'chunk_tenant_immutable'); END''')
        db.execute('''CREATE TABLE IF NOT EXISTS tenant_conversation_state (
            tenant_id INTEGER NOT NULL CHECK(tenant_id>0),
            conversation_id INTEGER NOT NULL CHECK(conversation_id>0),
            state_json TEXT NOT NULL,
            PRIMARY KEY(tenant_id,conversation_id))''')
        db.commit()
    except BaseException:
        db.rollback()
        raise


class TenantStorage:
    """Scoped facade for retrieval snapshots and durable conversation state.

    This is application enforcement, not SQLite row-level security. Do not
    expose the underlying connection to customer-selected SQL.
    """
    def __init__(self, db):
        self.db = db

    def candidates(self, tenant_id=None):
        tenant = require_tenant(tenant_id)
        rows = self.db.execute('''SELECT c.* FROM document_chunks c
            JOIN documents d ON d.id=c.document_id AND d.tenant_id=c.tenant_id
            WHERE c.tenant_id=? AND d.tenant_id=? AND c.is_leaf=1 AND d.status='ready'
            ORDER BY c.id''', (tenant, tenant))
        return [dict(row) for row in rows]

    def document(self, ident):
        tenant = require_tenant()
        row = self.db.execute('SELECT * FROM documents WHERE tenant_id=? AND id=?',
                              (tenant, ident)).fetchone()
        return dict(row) if row else None

    def write_conversation(self, conversation_id, state, tenant_id=None):
        tenant = require_tenant(tenant_id)
        if type(conversation_id) is not int or conversation_id <= 0:
            raise ValueError('invalid_conversation')
        # The authenticated caller must also authorize this conversation within
        # its tenant. A tenant boundary does not replace user-level RBAC.
        safe = prepare(json.dumps(state, sort_keys=True, allow_nan=False))
        with self.db:
            self.db.execute('''INSERT INTO tenant_conversation_state VALUES (?,?,?)
                ON CONFLICT(tenant_id,conversation_id) DO UPDATE SET state_json=excluded.state_json''',
                (tenant, conversation_id, safe))

    def conversation(self, conversation_id):
        tenant = require_tenant()
        row = self.db.execute('''SELECT state_json FROM tenant_conversation_state
            WHERE tenant_id=? AND conversation_id=?''', (tenant, conversation_id)).fetchone()
        return json.loads(row[0]) if row else None

    def checkpoint_key(self, conversation_id, thread):
        return scoped_key('checkpoint', conversation_id, thread)

    def save_checkpoint(self, conversation_id, thread, state, tenant_id=None):
        tenant = require_tenant(tenant_id)
        key = self.checkpoint_key(conversation_id, thread)
        with self.db:
            self.db.execute('''CREATE TABLE IF NOT EXISTS tenant_state_checkpoints (
                tenant_id INTEGER NOT NULL, checkpoint_key TEXT NOT NULL,
                state_json TEXT NOT NULL, PRIMARY KEY(tenant_id,checkpoint_key))''')
            self.db.execute('''INSERT INTO tenant_state_checkpoints VALUES (?,?,?)
                ON CONFLICT(tenant_id,checkpoint_key) DO UPDATE SET state_json=excluded.state_json''',
                (tenant, key, prepare(json.dumps(state, sort_keys=True, allow_nan=False))))
        return key

    def restore_checkpoint(self, conversation_id, thread):
        tenant = require_tenant()
        key = self.checkpoint_key(conversation_id, thread)
        exists = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='tenant_state_checkpoints'").fetchone()
        if not exists:
            return None
        row = self.db.execute('''SELECT state_json FROM tenant_state_checkpoints
            WHERE tenant_id=? AND checkpoint_key=?''', (tenant, key)).fetchone()
        return json.loads(row[0]) if row else None
