from incremental_reindex import setup, DB, KB
from anchor_ai.storage import connect
from anchor_ai.ingestion import sync_directory
from anchor_ai.index_maintenance import active, start_migration, backfill, activate
from anchor_ai.vector_index import VectorIndex

setup()
db = connect(DB)
try:
    sync_directory(db, 1, KB)
    index = VectorIndex()
    query = 'What is the refund policy?'
    before = index.search_active(db, 1, query)
    target = start_migration(db, 1)
    try:
        backfill(db, 1, target, fail=True)
    except RuntimeError:
        print('Injected failure: previous complete generation still serves')
    during = index.search_active(db, 1, query)
    assert during['generation_id'] == before['generation_id']
    backfill(db, 1, target, batch_size=1)
    shadow = db.execute('SELECT count(*) FROM document_chunks WHERE retrieval_generation_id=?', (target,)).fetchone()[0]
    assert active(db, 1)['id'] == before['generation_id']
    print({'shadow_generation': target, 'shadow_chunks': shadow,
           'serving_generation': during['generation_id']})
    while backfill(db, 1, target, batch_size=1):
        pass
    activate(db, 1, target)
    after = index.search_active(db, 1, query)
    policy = KB / 'policy.md'
    policy.write_text(policy.read_text() + ' Contact support before returning an item.')
    sync_directory(db, 1, KB)
    activate(db, 1, before['generation_id'], rollback=True)
    rolled_back = index.search_active(db, 1, query)
    assert after['dimensions'] == 6
    assert rolled_back['dimensions'] == before['dimensions'] == 4
    assert rolled_back['generation_id'] == before['generation_id']
    assert all(result['hits'] for result in [before, during, after, rolled_back])
    print({'query': query,
           'before': (before['generation_id'], before['dimensions']),
           'during': (during['generation_id'], during['dimensions']),
           'after': (after['generation_id'], after['dimensions']),
           'rollback': (rolled_back['generation_id'], rolled_back['dimensions']),
           'outcome': 'Anchor served one complete generation throughout; rollback required no application-code change'})
finally:
    db.close()
