import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
from anchor_ai import ingestion
from anchor_ai.llm_client import LlmClient
from anchor_ai.pii import ROOT, prepare
from anchor_ai.schemas import IngestionNode
from anchor_ai.storage import connect, persist

RAW = 'Refunds require review. Contact pat@example.com for help. Call 415-555-0199 for updates.'
SAFE = 'Refunds require review. Contact [EMAIL] for help. Call [PHONE] for updates.'


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / 'anchor.sqlite3'
    monkeypatch.setenv('ANCHOR_STAGE_ROOT', str(tmp_path / 'uploads'))
    monkeypatch.setenv('ANCHOR_INGESTION_EMBEDDINGS', 'offline')
    script = "require 'active_record'; require 'sqlite3'; require_relative 'db/migrate/20260101000100_create_documents_and_document_chunks'; ActiveRecord::Base.establish_connection(adapter: 'sqlite3', database: ARGV[0]); ActiveRecord::Migration.verbose=false; CreateDocumentsAndDocumentChunks.new.migrate(:up)"
    subprocess.run(['bundle', 'exec', 'ruby', '-e', script, str(path)], cwd=ROOT, check=True, capture_output=True)
    return path


def test_redacted_lineage_and_retry(database, monkeypatch, capsys, caplog):
    seen = []
    original = LlmClient.embed_ingestion
    def spy(self, texts):
        seen.extend(texts)
        assert all('pat@example.com' not in text and '415-555-0199' not in text for text in texts)
        return original(self, texts)
    monkeypatch.setattr(LlmClient, 'embed_ingestion', spy)
    reference = ingestion.stage(101, RAW)
    staged = ingestion.stage_path(reference).read_text()
    assert 'pat@example.com' not in staged and '415-555-0199' not in staged
    assert json.loads(staged)['text'] == SAFE
    first = ingestion.ingest(101, reference, database)
    assert first['status'] == 'ready' and not first['skipped']
    assert ingestion.ingest(101, reference, database)['skipped']
    assert len(seen) == 3
    second = ingestion.stage(202, RAW)
    ingestion.ingest(202, second, database)
    with connect(database) as db:
        documents = db.execute('SELECT * FROM documents ORDER BY tenant_id').fetchall()
        assert len(documents) == 2
        assert all(row['text'] == SAFE for row in documents)
        assert all(row['content_hash'] == hashlib.sha256(SAFE.encode()).hexdigest() for row in documents)
        rows = db.execute('SELECT * FROM document_chunks').fetchall()
        assert len(rows) == 8
        assert all('pat@example.com' not in row['text'] for row in rows)
        assert db.execute('''SELECT COUNT(*) FROM document_chunks l
            LEFT JOIN document_chunks p ON p.id=l.parent_id AND p.is_leaf=0
            AND p.document_id=l.document_id AND p.tenant_id=l.tenant_id
            WHERE l.is_leaf=1 AND p.id IS NULL''').fetchone()[0] == 0
        assert db.execute('''SELECT COUNT(*) FROM document_chunks c JOIN documents d ON d.id=c.document_id
            WHERE c.tenant_id<>d.tenant_id''').fetchone()[0] == 0
        assert all(row['embedding'] is None for row in rows if not row['is_leaf'])
        assert all(len(json.loads(row['embedding'])) == row['embedding_dimensions'] for row in rows if row['is_leaf'])
        assert any('[EMAIL]' in row['window_text'] for row in rows if row['is_leaf'])
    output = capsys.readouterr()
    diagnostics = output.out + output.err + caplog.text
    assert 'pat@example.com' not in diagnostics and '415-555-0199' not in diagnostics


def test_atomic_failure_and_conflict_safe_inserts(database):
    text = prepare(RAW)
    digest = hashlib.sha256(text.encode()).hexdigest()
    parent = IngestionNode(chunk_index=0, parent_index=None, is_leaf=False, text=text)
    bad_leaf = IngestionNode(chunk_index=1, parent_index=99, is_leaf=True, text='Review.',
        window_text='Review.', embedding=[1.0], embedding_model='fake', embedding_revision='v1', embedding_dimensions=1)
    db = connect(database)
    with pytest.raises(KeyError):
        persist(db, 101, 'a' * 32, text, digest, lambda: [parent, bad_leaf])
    assert db.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 0
    assert db.execute('SELECT COUNT(*) FROM document_chunks').fetchone()[0] == 0
    reference = ingestion.stage(101, RAW)
    ingestion.ingest(101, reference, database)
    db.execute("UPDATE documents SET status='processing'")
    db.commit()
    ingestion.ingest(101, reference, database)
    assert db.execute('SELECT COUNT(*) FROM document_chunks').fetchone()[0] == 4
    with pytest.raises(sqlite3.IntegrityError):
        db.execute('UPDATE document_chunks SET tenant_id=202')
    db.rollback()
    db.close()


def test_tenant_boundary_and_safe_diagnostics(database, monkeypatch):
    reference = ingestion.stage(101, RAW)
    with pytest.raises(ValueError, match='upload_tenant_mismatch'):
        ingestion.ingest(202, reference, database)
    monkeypatch.setenv('ANCHOR_DB_PATH', str(database))
    result = subprocess.run([sys.executable, '-m', 'anchor_ai.ingestion', 'ingest', '--tenant', '202',
        '--reference', reference], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 1
    assert result.stderr.strip() == 'ingestion_failed'
    assert 'pat@example.com' not in result.stdout + result.stderr
