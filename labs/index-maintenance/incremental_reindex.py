import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['ANCHOR_EMBEDDING_PROVIDER'] = 'offline'
from anchor_ai.storage import initialize_sandbox, connect
from anchor_ai.ingestion import sync_directory
from anchor_ai.index_maintenance import EmbeddingClient

SANDBOX = Path(__file__).resolve().parent / 'sandbox'
DB = SANDBOX / 'index.sqlite3'
KB = SANDBOX / 'kb'


def setup():
    initialize_sandbox(DB)
    KB.mkdir(exist_ok=True)
    for name, text in [('policy.md', 'Refunds are available within thirty days.'),
                       ('faq.txt', 'Contact support for shipping assistance.')]:
        path = KB / name
        if not path.exists():
            path.write_text(text)


def run(edit=False):
    setup()
    db = connect(DB)
    try:
        sync_directory(db, 1, KB)
        before = [r['id'] for r in db.execute('''SELECT c.id FROM document_chunks c
            JOIN documents d ON d.id=c.document_id WHERE d.external_id='faq.txt' ''')]
        if edit:
            path = KB / 'policy.md'
            path.write_text(path.read_text() + ' A receipt is required.')
        client = EmbeddingClient()
        result = sync_directory(db, 1, KB, client=client)
        after = [r['id'] for r in db.execute('''SELECT c.id FROM document_chunks c
            JOIN documents d ON d.id=c.document_id WHERE d.external_id='faq.txt' ''')]
        assert before == after
        assert client.calls > 0 if edit else client.calls == 0
        print({'edited_policy_only': edit, 'embedding_calls': client.calls, 'documents': result})
    finally:
        db.close()


if __name__ == '__main__':
    run('--edit' in sys.argv)
