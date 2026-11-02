import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchor_ai.hybrid_retriever import HybridRuntime, QueryFixtureClient
from anchor_ai.query_analyzer import QueryAnalyzer, RecordedQueryClient, identifiers, sanitize


def load_dataset(path=None):
    return json.loads(Path(path or ROOT / 'eval/datasets/retrieval_queries.json').read_text())


async def evaluate(dataset, verbose=True):
    if not dataset['cases']:
        raise ValueError('empty_dataset')
    client = QueryFixtureClient(dataset)
    analyzer = QueryAnalyzer(RecordedQueryClient())
    raw_hits = analyzed_hits = recalled = required_total = retained = identified = 0
    canonical = {(row['tenant_id'], row['id']) for row in dataset['chunks']}
    for case in dataset['cases']:
        tenant = case['tenant_id']
        required = set(case['required'])
        if not required or any((tenant, i) not in canonical for i in required):
            raise ValueError('invalid_labels')
        snapshot = client.acquire_snapshot(tenant)
        # An offline gate measures retrieval quality, not production latency.
        runtime = HybridRuntime(tenant, client.acquire_snapshot,
                                client.dense_search, client.sparse_search, deadline_seconds=10)
        raw = await runtime.retrieve(sanitize(case['ticket']), dataset['top_k'], snapshot=snapshot)
        analyzed = await runtime.retrieve_analyzed(case['ticket'], analyzer, dataset['top_k'])
        for result in (raw, analyzed):
            if result['tenant_id'] != tenant or result['generation_id'] != snapshot.generation_id:
                raise ValueError('scope_regression')
            if any(hit['tenant_id'] != tenant or (tenant, hit['id']) not in canonical
                   for hit in result['hits']):
                raise ValueError('cross_tenant_result')
        raw_ids = {hit['id'] for hit in raw['hits']}
        ids = {hit['id'] for hit in analyzed['hits']}
        source_ids = identifiers(sanitize(case['ticket']))
        plan_ids = identifiers(' '.join(analyzed['plan']['queries']))
        retained += len(source_ids & plan_ids)
        identified += len(source_ids)
        recalled += len(required & ids)
        required_total += len(required)
        raw_hits += required <= raw_ids
        analyzed_hits += required <= ids
        if verbose:
            print(json.dumps(dict(case=case['id'], raw_ids=sorted(raw_ids),
                analyzed_ids=sorted(ids), required=sorted(required),
                plan=analyzed['plan'], generation_id=snapshot.generation_id)))
    report = dict(dataset_version=dataset['version'], generation_id=dataset['generation_id'],
        tickets=len(dataset['cases']), raw_ticket_hit_rate=raw_hits / len(dataset['cases']),
        ticket_hit_rate=analyzed_hits / len(dataset['cases']),
        chunk_recall=recalled / required_total,
        identifier_retention=retained / identified if identified else 1.0,
        cross_tenant_results=0)
    return report


def enforce(report, thresholds):
    for metric, minimum in thresholds.items():
        if report[metric] < minimum:
            raise ValueError('retrieval_regression:' + metric)
    if report['cross_tenant_results']:
        raise ValueError('cross_tenant_result')


def main():
    dataset = load_dataset(sys.argv[1] if len(sys.argv) > 1 else None)
    report = asyncio.run(evaluate(dataset))
    print(json.dumps(report, sort_keys=True))
    enforce(report, dataset['thresholds'])
    print('PASS: offline retrieval gate; synthetic corpus, not production quality')
    from eval.retrieval_eval import evaluate as evaluate_document_quality
    evaluate_document_quality()


if __name__ == '__main__':
    main()
