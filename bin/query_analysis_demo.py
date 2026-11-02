import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anchor_ai.query_analyzer import identifiers, sanitize
from anchor_ai.rag import fixture_analyzed_runtime
from eval.retrieval_gate import load_dataset


async def main():
    dataset = load_dataset()
    print('Anchor: offline query analysis; synthetic evidence, no network')
    for case in dataset['cases']:
        runtime = fixture_analyzed_runtime(case['tenant_id'])
        client = runtime.fixture_client
        snapshot = client.acquire_snapshot(case['tenant_id'])
        raw = await runtime.hybrid.retrieve(sanitize(case['ticket']), dataset['top_k'],
                                            snapshot=snapshot)
        client.acquisitions.clear()
        client.calls.clear()
        analyzed = await runtime.retrieve(case['ticket'], dataset['top_k'])
        assert client.acquisitions == [case['tenant_id']]
        assert {hit['id'] for hit in analyzed['hits']} >= set(case['required'])
        assert identifiers(case['ticket']) <= identifiers(' '.join(analyzed['plan']['queries']))
        print(json.dumps(dict(case=case['id'], ticket=case['ticket'],
            raw_ids=[hit['id'] for hit in raw['hits']],
            plan=analyzed['plan'], subqueries=len(analyzed['plan']['queries']),
            tenant_id=analyzed['tenant_id'], generation_id=analyzed['generation_id'],
            identifiers=sorted(identifiers(' '.join(analyzed['plan']['queries']))),
            evidence=[dict(id=hit['id'], text=hit['text']) for hit in analyzed['hits']]),
            ensure_ascii=False))
    print('PASS: same misspelled delivery complaint -> targeted refund + address evidence; '
          'ORDER-42 retained; one tenant and generation; bounded traceable plan')


if __name__ == '__main__':
    asyncio.run(main())
