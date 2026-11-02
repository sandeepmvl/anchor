import argparse
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anchor_ai.ingestion import ingest_graph_records
from anchor_ai.graph_index import graph_snapshot
from anchor_ai.graph_retriever import retrieve_snapshot
from anchor_ai.rag import published_graph_rag_runtime

FIXTURE = ROOT / 'anchor_ai/tests/fixtures/graph_relationships.json'


def build_database():
    # Explicit offline harness; never alter the Rails development database.
    os.environ['ANCHOR_EMBEDDING_PROVIDER'] = 'offline'
    fixture = json.loads(FIXTURE.read_text())
    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    for sql in fixture['schema']:
        db.execute(sql)
    with tempfile.TemporaryDirectory() as directory:
        for tenant in (1, 2):
            path = Path(directory) / f'{tenant}.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in fixture['records']
                                    if r['tenant_id'] == tenant))
            ingest_graph_records(db, tenant, path)
    return db, fixture


class RecordedTextClient:
    """Rules supply follow-up queries; SQL LIKE searches text, never edges."""
    def __init__(self, steps):
        self.steps = steps

    def followup(self, query, passage):
        for step in self.steps:
            if step['query'] == query and step['passage'] in passage:
                return step['next_query']
        return None


def text_multihop(db, fixture):
    from anchor_ai.index_maintenance import retrieval_snapshot
    generation, chunks = retrieval_snapshot(db, 1)
    chunks = [c for c in chunks if db.execute(
        'SELECT source_scope FROM documents WHERE id=?', (c['document_id'],)
        ).fetchone()[0] == 'support']
    client = RecordedTextClient(fixture['text_steps'])
    query, trajectory, seen = 'Pro', [], set()
    for _ in range(3):
        matches = [c for c in chunks if query.casefold() in c['text'].casefold()
                   and c['id'] not in seen]
        if not matches:
            break
        # Deterministic fixture tie-breaker; no graph access.
        hit = matches[0]
        seen.add(hit['id'])
        trajectory.append(dict(query=query, citation=hit['id'], passage=hit['text']))
        query = client.followup(query, hit['text'])
        if query is None:
            break
    return dict(mode='multi', question=fixture['questions'][0]['question'],
                generation_id=generation['id'], trajectory=trajectory,
                mechanism='iterative text retrieval with recorded follow-up queries')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['graph', 'multi', 'compare'], default='graph')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--no-tenant-filter', action='store_true')
    args = parser.parse_args()
    db, fixture = build_database()
    try:
        snapshot = graph_snapshot(db, 1, 'support')
        if args.inspect:
            print(json.dumps(snapshot, indent=2))
        if args.no_tenant_filter:
            # Deliberately violate the snapshot authorization boundary, here only.
            # Each owner's active generation remains intact; only ownership is removed.
            other = graph_snapshot(db, 2, 'support')
            snapshot['nodes'] += other['nodes']
            snapshot['edges'] += other['edges']
        if args.mode in ('graph', 'compare'):
            foreign = 0
            for question in fixture['questions']:
                report = retrieve_snapshot(snapshot, question['question'], question['terms'], args.max_depth)
                report['id'] = question['id']
                foreign += sum(e['tenant_id'] != 1 for p in report['paths'] for e in p)
                print(json.dumps(report))
                if args.max_depth == 3 and not args.no_tenant_filter:
                    assert report['status'] == question['expected']
                if args.no_tenant_filter and question['id'] == 'delayed':
                    assert report['status'] == 'supported'
            print(f'foreign-tenant edges: {foreign}')
            if args.no_tenant_filter:
                assert foreign > 0
                print('Offline leak demonstration verified; production has no bypass')
            else:
                assert foreign == 0
                runtime = published_graph_rag_runtime(db, 1, 'support')
                q = fixture['questions'][0]
                answer = asyncio.run(runtime.answer(dict(question=q['question'], terms=q['terms']),
                                     request_id='graph-demo', trace_id='graph-demo'))
                print(json.dumps(dict(mode='graph', answer=answer)))
        if args.mode in ('multi', 'compare'):
            result = text_multihop(db, fixture)
            assert len(result['trajectory']) == 3
            print(json.dumps(result))
    finally:
        db.close()


if __name__ == '__main__':
    main()
