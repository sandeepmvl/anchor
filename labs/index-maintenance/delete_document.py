from incremental_reindex import setup, DB, KB
from anchor_ai.storage import connect
from anchor_ai.ingestion import sync_directory
from anchor_ai.index_maintenance import delete_document
from anchor_ai.vector_index import VectorIndex

setup()
db = connect(DB)
try:
    sync_directory(db, 1, KB)
    document = db.execute("SELECT id FROM documents WHERE tenant_id=1 AND external_id='faq.txt'").fetchone()['id']
    index = VectorIndex()
    index.search_active(db, 1, 'shipping')
    old_key = index.snapshot_key
    (KB / 'faq.txt').unlink()
    delete_document(db, 1, 'faq.txt')
    index.search_active(db, 1, 'shipping')
    remaining = db.execute('SELECT count(*) FROM document_chunks WHERE document_id=?', (document,)).fetchone()[0]
    orphans = db.execute('''SELECT count(*) FROM document_chunks c
        LEFT JOIN documents d ON d.id=c.document_id WHERE d.id IS NULL''').fetchone()[0]
    assert remaining == orphans == 0
    assert old_key != index.snapshot_key
    print({'deleted_document_chunks_all_generations': remaining, 'orphans': orphans,
           'cached_snapshot_invalidated': True})
finally:
    db.close()
