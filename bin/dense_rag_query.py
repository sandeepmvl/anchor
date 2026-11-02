import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.rag import fixture_runtime


async def main():
    parser = argparse.ArgumentParser(description='Anchor offline dense-RAG acceptance harness')
    parser.add_argument('query', nargs='?', default='How do I get a refund?')
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--http-payload')
    parser.add_argument('--request-id')
    parser.add_argument('--traceparent')
    args = parser.parse_args()
    if args.http_payload:
        from fastapi.testclient import TestClient
        from anchor_ai.app import app
        from anchor_ai import observability
        os.environ['ANCHOR_DENSE_FIXTURE'] = '1'
        os.environ['ANCHOR_AI_SECRET'] = 'offline-dense-test-only'
        observability.SINK = lambda record: print(json.dumps(record), file=sys.stderr)
        with TestClient(app) as client:
            response = client.post('/v1/dense_rag', json=json.loads(args.http_payload), headers={
                'X-Anchor-Secret': os.environ['ANCHOR_AI_SECRET'],
                'X-Request-Id': args.request_id,
                'traceparent': args.traceparent})
            if response.status_code != 200:
                raise SystemExit('offline HTTP request failed')
            print(json.dumps(response.json()))
        return
    state = Path('tmp/dense-rag-fixture.json')
    if args.seed:
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({'tenant_id': 1, 'generation_id': 17}))
        print('Anchor fixture tenant 1: published generation 17 is ready.')
        return
    if not state.exists():
        raise SystemExit('Run this command with --seed first.')
    seeded = json.loads(state.read_text())
    if seeded != {'tenant_id': 1, 'generation_id': 17}:
        raise SystemExit('Invalid fixture seed.')
    traces = []
    runtime = fixture_runtime(tenant_id=seeded['tenant_id'], emit=traces.append)
    request_id = uuid.uuid4().hex
    trace_id = uuid.uuid4().hex
    result = await runtime.answer(args.query, request_id=request_id, trace_id=trace_id)
    trace = traces[-1]
    print(json.dumps({'course_project': 'Anchor', 'mode': 'offline_acceptance',
                      'ranked_chunk_ids': trace['chunk_ids'], 'scores': trace['scores'],
                      'reply': result['reply'], 'citations': result['citations'],
                      'abstained': result['abstained']}, indent=2))
    print(json.dumps(trace, separators=(',', ':')), file=sys.stderr)


if __name__ == '__main__':
    asyncio.run(main())
