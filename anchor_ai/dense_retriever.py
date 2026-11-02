import copy
import math
import time

import faiss
import numpy as np


class RetrievalError(ValueError):
    pass


class RecordedEmbeddingClient:
    """Offline default. A missing recording is an error, never a guessed vector."""

    def __init__(self, records):
        self.records = records
        self.calls = []

    def embed(self, query, *, model, revision, dimensions):
        self.calls.append((query, model, revision, dimensions))
        key = (query, model, revision, dimensions)
        if key not in self.records:
            raise RetrievalError('embedding_replay_miss')
        return self.records[key]


class DenseRetriever:
    def __init__(self, generation, tenant_id, embedding_client, index=None):
        if type(tenant_id) is not int or tenant_id <= 0:
            raise RetrievalError('trusted_tenant_required')
        self.generation = copy.deepcopy(generation)
        self.tenant_id = tenant_id
        self.embedding_client = embedding_client
        self.chunks = self.generation['chunks']
        self.dimensions = self.generation['dimensions']
        self.metric = self.generation['metric']
        self.normalize = self.generation['normalize']
        if self.metric not in ('cosine', 'ip', 'l2'):
            raise RetrievalError('unsupported_metric')
        if self.metric == 'cosine' and not self.normalize:
            raise RetrievalError('cosine_requires_normalization')
        ids = [c['id'] for c in self.chunks]
        if len(set(ids)) != len(ids) or any(type(i) is not int or i <= 0 for i in ids):
            raise RetrievalError('invalid_chunk_ids')
        self.by_id = {c['id']: c for c in self.chunks}
        vectors = [self._vector(c['vector']) for c in self.chunks]
        self.vectors = np.asarray(vectors, dtype='float32').reshape(-1, self.dimensions)
        self.rows = {c['id']: row for row, c in enumerate(self.chunks)}
        self.tenant_ids = [c['id'] for c in self.chunks if c['tenant_id'] == tenant_id]
        if index is None:
            base = (faiss.IndexFlatL2(self.dimensions) if self.metric == 'l2'
                    else faiss.IndexFlatIP(self.dimensions))
            index = faiss.IndexIDMap2(base)
            if ids:
                index.add_with_ids(self.vectors, np.asarray(ids, dtype='int64'))
        self.index = index
        self.last_trace = None

    def _vector(self, value):
        vector = np.asarray(value, dtype='float32').copy()
        if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
            raise RetrievalError('invalid_vector')
        if self.normalize:
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                raise RetrievalError('zero_vector')
            vector /= norm
        return vector

    def _flat(self, query):
        scored = []
        for chunk_id in self.tenant_ids:
            vector = self.vectors[self.rows[chunk_id]]
            score = (-float(np.sum((query - vector) ** 2)) if self.metric == 'l2'
                     else float(query @ vector))
            scored.append((chunk_id, score))
        return scored

    def search(self, query, k):
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError('query_required')
        if type(k) is not int or not 1 <= k <= 20:
            raise RetrievalError('invalid_k')
        started = time.monotonic()
        embedded = self.embedding_client.embed(
            query, model=self.generation['model'], revision=self.generation['revision'],
            dimensions=self.dimensions)
        vector = self._vector(embedded)
        embedded_at = time.monotonic()
        native_scores, result_ids = self.index.search(vector.reshape(1, -1), 10 * k)
        survivors = {}
        for chunk_id, native in zip(result_ids[0], native_scores[0]):
            chunk_id = int(chunk_id)
            if chunk_id == -1:
                continue
            chunk = self.by_id.get(chunk_id)
            if chunk is None:
                raise RetrievalError('index_generation_mismatch')
            if chunk['tenant_id'] != self.tenant_id:
                continue
            score = -float(native) if self.metric == 'l2' else float(native)
            if not math.isfinite(score):
                raise RetrievalError('invalid_score')
            survivors[chunk_id] = score
        searched_at = time.monotonic()
        count = len(survivors)
        target = min(k, len(self.tenant_ids))
        fallback = count < target
        ranked = self._flat(vector) if fallback else list(survivors.items())
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        selected = ranked[:target]
        ended = time.monotonic()
        self.last_trace = {
            'generation_id': self.generation['generation_id'],
            'tenant_id': self.tenant_id,
            'embedding_model': self.generation['model'],
            'embedding_revision': self.generation['revision'],
            'metric': self.metric,
            'k': k,
            'overfetch': 10,
            'post_filter_count': count,
            'filter_path': 'tenant_flat' if fallback else 'post_filter',
            'chunk_ids': [i for i, _ in selected],
            'scores': [score for _, score in selected],
            'embedding_ms': round((embedded_at - started) * 1000, 3),
            'search_ms': round((searched_at - embedded_at) * 1000, 3),
            'fallback_ms': round((ended - searched_at) * 1000, 3)
        }
        return [dict(id=i, score=score, tenant_id=self.tenant_id,
                     text=self.by_id[i]['text']) for i, score in selected]


def fixture_dependencies():
    """A published snapshot injected only by the acceptance harness."""
    policy = 'Refunds require review. Contact support with your receipt to request a refund.'
    chunks = [dict(id=i, tenant_id=2, text='Other tenant policy.', vector=[1.0, 0.0])
              for i in range(1, 41)]
    chunks += [dict(id=101, tenant_id=1, text=policy, vector=[0.8, 0.6]),
               dict(id=102, tenant_id=1, text='Keep your receipt.', vector=[0.7, 0.7]),
               dict(id=103, tenant_id=1, text='Support reviews requests.', vector=[0.6, 0.8])]
    generation = dict(generation_id=17, model='anchor-recorded-embedding-v1',
                      revision='authored-fixture-v1', dimensions=2,
                      metric='cosine', normalize=True, chunks=chunks)
    records = {(query, generation['model'], generation['revision'], 2): [1.0, 0.0]
               for query in ('How do I get a refund?', 'Do you cover lunar shipping?')}
    return generation, RecordedEmbeddingClient(records)
