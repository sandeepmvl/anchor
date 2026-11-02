import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from anchor_ai.app import app
from anchor_ai import observability
from anchor_ai.hybrid_retriever import RecordedHybridClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query')
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    os.environ['ANCHOR_HYBRID_FIXTURE'] = '1'
    os.environ['ANCHOR_AI_SECRET'] = 'offline-hybrid-cli-only'
    records = []
    previous_sink = observability.SINK
    observability.SINK = records.append
    try:
        with TestClient(app) as client:
            response = client.post('/retrieve/hybrid',
                json=dict(tenant_id=1, conversation_id=10, query=args.query, top_k=3),
                headers={'X-Anchor-Secret': os.environ['ANCHOR_AI_SECRET'],
                         'X-Request-Id': 'hybrid-cli',
                         'traceparent': '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01'})
            response.raise_for_status()
            result = response.json()
    finally:
        observability.SINK = previous_sink
    fake = RecordedHybridClient()
    snapshot = fake.acquire_snapshot(1)
    dense = fake.dense_search(args.query, snapshot, 5)
    dense_order = [row['id'] for row in dense]
    fused_order = [row['id'] for row in result['hits']]
    print(json.dumps(dict(dense_order=dense_order, fused_order=fused_order)))
    print(json.dumps(result, indent=2))
    if '4821' in args.query:
        assert fused_order.index(102) < fused_order.index(101)
        print('Anchor: exact-order chunk 102 moves above generic policy 101.')
    if args.trace:
        for record in records:
            print(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    main()
