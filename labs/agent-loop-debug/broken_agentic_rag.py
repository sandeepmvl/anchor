import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / 'anchor_ai/tests/fixtures/repeated_retrieval_responses.json'


def data():
    return json.loads(FIXTURE.read_text())


class RepeatingPolicy:
    def __init__(self, queries):
        self.queries = queries
        self.calls = 0

    async def decide(self, question, evidence):
        query = self.queries[self.calls % len(self.queries)]
        self.calls += 1
        return {'kind': 'retrieve', 'tool': 'retrieve', 'arguments': {'query': query}}


def broken(recording):
    import hashlib
    queries = recording['trajectories']['duplicate_query']
    seen, query_keys = set(), set()
    for hop in range(1, 13):
        query = queries[(hop - 1) % len(queries)]
        normalized = ' '.join(query.casefold().split())
        key = hashlib.sha256(normalized.encode()).hexdigest()
        ids = recording['searches'][query]
        new_ids = set(ids) - seen
        print(json.dumps({'event': 'synthetic_retrieval', 'hop': hop,
                          'query_fingerprint': key, 'duplicate_query': key in query_keys,
                          'chunk_ids': ids, 'new_chunk_ids': sorted(new_ids),
                          'fixture_latency_ms': 1050,
                          'cumulative_fixture_latency_ms': hop * 1050,
                          'cumulative_fixture_cost_usd': round(hop * 0.001, 6)}))
        seen.update(ids)
        query_keys.add(key)
    print(json.dumps({'event': 'harness_exit', 'reason': 'external_ceiling', 'calls': 12}))


async def product(recording):
    from anchor_ai.agentic_rag import AgenticRuntime, RecordedRetrievalBoundary
    policy = RepeatingPolicy(recording['trajectories']['repeated_evidence'])
    retriever = RecordedRetrievalBoundary(recording)
    events = []
    runtime = AgenticRuntime(retriever, policy=policy, emit=events.append)
    result = await runtime.answer(recording['question'], request_id='refund-debug',
                                  trace_id='1' * 32)
    print(json.dumps({'event': 'product_result', 'result': result,
                      'exit': events[-1]['exit'], 'calls': len(retriever.calls)}))
    assert events[-1]['exit'] == 'no_new_evidence'
    assert len(retriever.calls) == 2
    assert result['abstained'] and result['citations'] == []
    print('Anchor: repeated evidence stopped after 2 retrievals, before the 4-call budget; insufficient evidence, no refund claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--product', action='store_true')
    args = parser.parse_args()
    if args.product:
        asyncio.run(product(data()))
    else:
        broken(data())
