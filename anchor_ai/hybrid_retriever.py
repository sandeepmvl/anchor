import asyncio
import math
import time
from dataclasses import dataclass
from types import MappingProxyType

from anchor_ai.observability import span


def rrf_fuse(dense, sparse, k=60):
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k <= 0:
        raise ValueError('invalid_rrf_k')
    merged = {}
    for name, hits in (('dense', dense), ('sparse', sparse)):
        seen = set()
        rank = 0
        for hit in hits:
            chunk_id = hit['id']
            if type(chunk_id) is not int or chunk_id <= 0:
                raise ValueError('invalid_chunk_id')
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            row = merged.setdefault(chunk_id, dict(id=chunk_id, score=0.0,
                                                   dense=0.0, sparse=0.0))
            contribution = 1.0 / (k + rank)
            row[name] = contribution
            row['score'] += contribution
    return sorted(merged.values(), key=lambda row: (-row['score'], row['id']))


@dataclass(frozen=True)
class Snapshot:
    tenant_id: int
    generation_id: str
    chunks: tuple

    def __post_init__(self):
        rows = tuple(MappingProxyType(dict(row)) for row in self.chunks)
        if type(self.tenant_id) is not int or self.tenant_id <= 0 or not self.generation_id:
            raise ValueError('invalid_snapshot')
        ids = [row['id'] for row in rows]
        if len(set(ids)) != len(ids) or any(type(i) is not int or i <= 0 for i in ids):
            raise ValueError('invalid_snapshot')
        if any(row['tenant_id'] != self.tenant_id or not isinstance(row['text'], str)
               for row in rows):
            raise ValueError('invalid_snapshot')
        object.__setattr__(self, 'chunks', rows)


class HybridRuntime:
    def __init__(self, tenant_id, acquire_snapshot, dense_search, sparse_search,
                 candidate_depth=5, rrf_k=60, deadline_seconds=2):
        if type(candidate_depth) is not int or not 1 <= candidate_depth <= 100:
            raise ValueError('invalid_candidate_depth')
        rrf_fuse([], [], rrf_k)
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError('invalid_deadline')
        self.tenant_id = tenant_id
        self.acquire_snapshot = acquire_snapshot
        self.branches = (dense_search, sparse_search)
        self.candidate_depth = candidate_depth
        self.rrf_k = rrf_k
        self.deadline_seconds = deadline_seconds

    async def retrieve(self, query, top_k=3, *, snapshot=None):
        if not query.strip() or len(query.encode('utf-8')) > 1800:
            raise ValueError('invalid_query')
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('invalid_top_k')
        started = time.monotonic()
        with span('python.hybrid_retrieval', tenant_id=self.tenant_id, stage='retrieval') as summary:
            if snapshot is None:
                snapshot = self.acquire_snapshot(self.tenant_id)
            if not isinstance(snapshot, Snapshot) or snapshot.tenant_id != self.tenant_id:
                raise ValueError('snapshot_scope_mismatch')
            summary.set(generation_id=snapshot.generation_id)
            canonical = {row['id']: row for row in snapshot.chunks}

            def run(name, search):
                begin = time.monotonic()
                with span('python.retrieval.' + name, tenant_id=self.tenant_id,
                          generation_id=snapshot.generation_id, stage='retrieval'):
                    # The closure retains this snapshot even if the caller times out.
                    hits = list(search(query, snapshot, self.candidate_depth))
                    for hit in hits:
                        if (hit.get('tenant_id') != self.tenant_id
                                or hit.get('generation_id') != snapshot.generation_id
                                or hit.get('id') not in canonical):
                            raise ValueError('hit_scope_mismatch')
                    return hits, (time.monotonic() - begin) * 1000

            tasks = [asyncio.create_task(asyncio.to_thread(run, name, search))
                     for name, search in zip(('dense', 'sparse'), self.branches)]
            try:
                dense, sparse = await asyncio.wait_for(asyncio.gather(*tasks),
                                                       self.deadline_seconds)
            except BaseException:
                for task in tasks:
                    task.cancel()
                raise
            begin = time.monotonic()
            with span('python.retrieval.fusion', tenant_id=self.tenant_id,
                      generation_id=snapshot.generation_id, stage='retrieval'):
                fused = rrf_fuse(dense[0], sparse[0], self.rrf_k)[:top_k]
                hits = [dict(row, tenant_id=self.tenant_id,
                             text=canonical[row['id']]['text']) for row in fused]
            summary.set(dense_ms=dense[1], sparse_ms=sparse[1],
                        fusion_ms=(time.monotonic() - begin) * 1000,
                        total_ms=(time.monotonic() - started) * 1000,
                        chunk_ids=[row['id'] for row in hits])
            return dict(tenant_id=self.tenant_id, generation_id=snapshot.generation_id,
                        hits=hits)

    async def retrieve_analyzed(self, query, analyzer, top_k=3):
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('invalid_top_k')
        with span('python.analyzed_retrieval', tenant_id=self.tenant_id,
                  stage='retrieval') as item:
            plan = await analyzer.analyze(query)
            snapshot = self.acquire_snapshot(self.tenant_id)
            if not isinstance(snapshot, Snapshot) or snapshot.tenant_id != self.tenant_id:
                raise ValueError('snapshot_scope_mismatch')

            async def execute():
                lists = []
                for subquery in plan.queries:
                    result = await self.retrieve(subquery, top_k, snapshot=snapshot)
                    lists.append(result['hits'])
                merged, seen = [], set()
                for rank in range(top_k):
                    for hits in lists:
                        if rank < len(hits) and hits[rank]['id'] not in seen:
                            hit = hits[rank]
                            seen.add(hit['id'])
                            merged.append(hit)
                            if len(merged) == top_k:
                                return merged
                return merged

            hits = await asyncio.wait_for(execute(), self.deadline_seconds)
            item.set(generation_id=snapshot.generation_id,
                     analysis_version=plan.version, subquery_count=len(plan.queries),
                     analysis_fallback=plan.fallback, analysis_reason=plan.reason,
                     chunk_ids=[hit['id'] for hit in hits])
            return dict(tenant_id=self.tenant_id, generation_id=snapshot.generation_id,
                        plan=plan.model_dump(), hits=hits)


class RecordedHybridClient:
    """Authored rankings for orchestration regression, not a relevance benchmark."""

    def __init__(self):
        self.snapshot = Snapshot(1, 'fixture-g1', (
            dict(id=101, tenant_id=1, text='Refunds require review. Contact support with your receipt.'),
            dict(id=102, tenant_id=1, text='Synthetic excerpt: delayed shipment order 4821; request refund review.'),
            dict(id=103, tenant_id=1, text='Order-code help: include your order number when contacting support.'),
        ))

    def acquire_snapshot(self, tenant_id):
        if tenant_id != 1:
            raise ValueError('fixture_scope_mismatch')
        return self.snapshot

    def search(self, branch, query, snapshot, depth):
        if branch == 'dense':
            ids = [101, 102]
        else:
            ids = [102, 103] if '4821' in query else [103, 101]
        return [dict(id=i, tenant_id=snapshot.tenant_id,
                     generation_id=snapshot.generation_id) for i in ids[:depth]]

    def dense_search(self, query, snapshot, depth):
        return self.search('dense', query, snapshot, depth)

    def sparse_search(self, query, snapshot, depth):
        return self.search('sparse', query, snapshot, depth)


class QueryFixtureClient:
    """Synthetic corpus ranking, not a learned embedding relevance benchmark."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.acquisitions = []
        self.calls = []

    def acquire_snapshot(self, tenant_id):
        self.acquisitions.append(tenant_id)
        rows = tuple(row for row in self.dataset['chunks'] if row['tenant_id'] == tenant_id)
        if not rows:
            raise ValueError('fixture_scope_mismatch')
        return Snapshot(tenant_id, self.dataset['generation_id'], rows)

    def search(self, branch, query, snapshot, depth):
        import re
        import numpy as np
        from sklearn.feature_extraction.text import HashingVectorizer
        from rank_bm25 import BM25Okapi
        stop = {'a', 'an', 'the', 'for', 'and', 'i', 'my', 'it', 'to', 'your', 'me', 'of', 'can'}

        def tokens(text):
            return [t for t in re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text.lower())
                    if t not in stop]

        self.calls.append((branch, query, snapshot.tenant_id, snapshot.generation_id))
        documents = [row['text'] for row in snapshot.chunks]
        if branch == 'sparse':
            scores = BM25Okapi([tokens(text) for text in documents]).get_scores(tokens(query))
        else:
            vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False,
                analyzer=tokens, norm='l2')
            vectors = vectorizer.transform([query] + documents)
            scores = np.asarray((vectors[1:] @ vectors[0].T).toarray()).ravel()
        ranked = sorted(zip(snapshot.chunks, scores), key=lambda pair: (-float(pair[1]), pair[0]['id']))
        return [dict(id=row['id'], tenant_id=snapshot.tenant_id,
                     generation_id=snapshot.generation_id)
                for row, score in ranked if score > 0][:depth]

    def dense_search(self, query, snapshot, depth):
        return self.search('dense', query, snapshot, depth)

    def sparse_search(self, query, snapshot, depth):
        return self.search('sparse', query, snapshot, depth)
