import argparse
import asyncio

from anchor_ai.hybrid_retriever import Snapshot
from anchor_ai.rag import (RecordedGenerationClient, published_hybrid_runtime,
                           published_hybrid_rag_runtime)
from eval.retrieval_metrics import mean_scores, score


QUERIES = ('refund damaged order', 'damaged order refund policy')
JUDGMENTS = {doc: int(101 <= doc <= 105) for doc in range(86, 106)}


class RefundFixture:
    """Authored branch rankings, intentionally not a provider quality test."""

    def __init__(self):
        self.snapshot = Snapshot(1, 'rerank-g1', tuple(
            dict(id=i, tenant_id=1, document_id=85 + i,
                 metadata={'section': 'policy' if i >= 16 else 'faq'},
                 text=('Refunds require review. Damaged order refund policy: '
                       'contact support with your receipt.' if i >= 16 else
                       'Refund FAQ: general account and payment information.'))
            for i in range(1, 21)))
        self.depths = []

    def acquire_snapshot(self, tenant_id):
        if tenant_id != 1:
            raise ValueError('fixture_scope_mismatch')
        return self.snapshot

    def search(self, query, snapshot, depth):
        self.depths.append(depth)
        return [dict(id=row['id'], tenant_id=snapshot.tenant_id,
                     generation_id=snapshot.generation_id)
                for row in snapshot.chunks[:depth]]

    def runtime(self, **settings):
        return published_hybrid_runtime(1, self.acquire_snapshot,
                                       self.search, self.search, **settings)


def document_ranking(hits, snapshot):
    documents = {row['id']: row['document_id'] for row in snapshot.chunks}
    return [documents[hit['id']] for hit in hits]


async def run(mode):
    fixture = RefundFixture()
    # Equal candidate coverage isolates the effect of reordering.
    baseline = fixture.runtime(candidate_depth=20, reranker=None)
    selected = fixture.runtime(candidate_depth=20)
    before_rows, after_rows = [], []
    for query in QUERIES:
        before = (await baseline.retrieve(query, 20))['hits']
        print('query:', query)
        print('fused top-20:', [hit['id'] for hit in before])
        print('before top-five:', [hit['id'] for hit in before[:5]])
        before_metric = score(document_ranking(before, fixture.snapshot), JUDGMENTS)
        before_rows.append(before_metric)
        print('before precision@5:', f"{before_metric['precision@5']:.3f}")
        if mode != 'before':
            after = (await selected.retrieve(query, 20))['hits']
            after_metric = score(document_ranking(after, fixture.snapshot), JUDGMENTS)
            after_rows.append(after_metric)
            print('after top-five:', [hit['id'] for hit in after[:5]])
            print('after precision@5:', f"{after_metric['precision@5']:.3f}")
    before_mean = mean_scores(before_rows)['precision@5']
    print('hybrid mean precision@5:', f'{before_mean:.3f}')
    if after_rows:
        after_mean = mean_scores(after_rows)['precision@5']
        print('selected mean precision@5:', f'{after_mean:.3f}')
        print('absolute lift:', f'{after_mean - before_mean:+.3f}')
        reader = RecordedGenerationClient()
        rag = published_hybrid_rag_runtime(1, fixture.acquire_snapshot,
            fixture.search, fixture.search, client=reader, candidate_depth=20,
            evidence_budget=2000)
        answer = await rag.answer(QUERIES[0], k=5,
                                  request_id='rerank-demo', trace_id='rerank-demo')
        print('Anchor answer:', answer)
    print('Offline fixture: validates wiring and metric arithmetic; '
          'does not prove learned-model relevance or production latency.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('before', 'compare'), default='compare')
    args = parser.parse_args()
    asyncio.run(run(args.mode))


if __name__ == '__main__':
    main()
