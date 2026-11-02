"""Process-local shared FAQ cache. Similarity never supplies authority."""
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from anchor_ai.pii import prepare
from anchor_ai.vector_index import semantic_ann_candidates


@dataclass(frozen=True)
class Snapshot:
    tenant_id: int
    role: str
    audience: str
    locale: str
    permission_version: str
    answer_path: str
    kb: str
    embedding_version: str
    prompt_version: str
    model_version: str
    retrieval_version: str
    eligibility: str = 'unclassified'

    def __post_init__(self):
        if type(self.tenant_id) is not int or self.tenant_id <= 0:
            raise ValueError('invalid_cache_tenant')
        if any(not isinstance(v, str) or not v for v in self.scope[1:]):
            raise ValueError('incomplete_cache_scope')
        if self.eligibility not in REASONS:
            raise ValueError('invalid_cache_eligibility')

    @property
    def scope(self):
        return (self.tenant_id, self.role, self.audience, self.locale,
                self.permission_version, self.answer_path, self.kb,
                self.embedding_version, self.prompt_version, self.model_version,
                self.retrieval_version)


REASONS = {'public_faq', 'unclassified', 'personalized', 'action_bearing',
           'permission_filtered', 'pii'}


@dataclass(frozen=True)
class Generated:
    text: str
    public_faq: bool = False
    personalized: bool = False
    action_bearing: bool = False
    permission_filtered: bool = False

    @property
    def safe(self):
        return (self.public_faq is True and self.personalized is False
                and self.action_bearing is False and self.permission_filtered is False
                and isinstance(self.text, str) and bool(self.text.strip())
                and prepare(self.text) == self.text)


def configuration(path='config/semantic_cache.yml'):
    c = yaml.safe_load(Path(path).read_text())
    if (set(c) != {'threshold', 'ttl_seconds', 'ann_tenant_entries',
                   'candidate_count', 'max_tenant_entries', 'max_entries', 'record_content'}
            or type(c['threshold']) not in (int, float)
            or not 0 < c['threshold'] <= 1 or c['record_content'] is not False):
        raise ValueError('unsafe_semantic_cache_configuration')
    for name in ('ttl_seconds', 'ann_tenant_entries', 'candidate_count',
                 'max_tenant_entries', 'max_entries'):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError('unsafe_semantic_cache_limit')
    if not c['ann_tenant_entries'] < c['max_tenant_entries'] <= c['max_entries']:
        raise ValueError('inconsistent_semantic_cache_limits')
    return c


def normalized(vector):
    v = np.asarray(vector, dtype=np.float64)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
        raise ValueError('invalid_cache_vector')
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length == 0:
        raise ValueError('invalid_cache_vector')
    return v / length


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


class SemanticCache:
    def __init__(self, *, config=None, clock=time.monotonic, emit=None):
        self.config = configuration() if config is None else dict(config)
        self.clock = clock
        if emit is None:
            from anchor_ai.observability import emit_semantic_cache
            emit = emit_semantic_cache
        self.emit = emit
        self.rows = []

    def prune(self):
        now = self.clock()
        self.rows[:] = [r for r in self.rows
                        if now - r['created'] < self.config['ttl_seconds']]

    def event(self, outcome, reason, mode, count, started):
        self.emit(dict(cache_outcome=outcome, cache_reason=reason,
                       cache_lookup_mode=mode, cache_candidates=count,
                       cache_lookup_ms=max(0, (self.clock() - started) * 1000)))

    def eligible(self, snapshot, query):
        if prepare(query) != query:
            return 'pii'
        return snapshot.eligibility

    def lookup(self, snapshot, query, vector=None):
        started = self.clock()
        self.prune()
        reason = self.eligible(snapshot, query)
        if reason != 'public_faq':
            self.event('bypass', reason, 'none', 0, started)
            return None
        q = normalized(vector)
        partition = [r for r in self.rows if r['scope'] == snapshot.scope]
        tenant_count = sum(r['scope'][0] == snapshot.tenant_id for r in self.rows)
        mode = 'ann' if tenant_count > self.config['ann_tenant_entries'] else 'linear'
        # Candidate search receives only this complete authorized partition.
        if partition and mode == 'ann':
            ids = semantic_ann_candidates([r['vector'] for r in partition], q,
                                         self.config['candidate_count'])
            candidates = [partition[i] for i in ids]
        else:
            candidates = partition
        exact = fingerprint(query)
        best, score = None, -1.0
        for row in candidates:
            if row['vector'].shape != q.shape:
                raise ValueError('cache_embedding_version_mismatch')
            similarity = float(row['vector'] @ q)
            if similarity >= self.config['threshold'] and similarity > score:
                best, score = row, similarity
        if best is None:
            self.event('miss', 'no_match', mode, len(candidates), started)
            return None
        self.event('hit', 'exact' if best['query_hash'] == exact else 'semantic',
                   mode, len(candidates), started)
        return best['answer']

    def store(self, snapshot, query, vector, generated):
        if self.eligible(snapshot, query) != 'public_faq' or not generated.safe:
            return False
        self.prune()
        row = dict(scope=snapshot.scope, query_hash=fingerprint(query),
                   vector=normalized(vector), answer=generated.text, created=self.clock())
        self.rows.append(row)
        tenant = snapshot.tenant_id
        while sum(r['scope'][0] == tenant for r in self.rows) > self.config['max_tenant_entries']:
            del self.rows[next(i for i, r in enumerate(self.rows) if r['scope'][0] == tenant)]
        while len(self.rows) > self.config['max_entries']:
            del self.rows[0]
        return True


async def with_answer_cache(cache, snapshot, query, vector, generate):
    """Generate returns (Generated, provider_report); eligibility stays explicit."""
    answer = cache.lookup(snapshot, query, vector)
    if answer is not None:
        return answer, dict(answer_cache_hit=True, provider_called=False,
                            provider_cache_status='not_called',
                            cached_input_tokens=None, cache_write_tokens=None,
                            estimated_input_saving_usd=None,
                            evidence='anchor_answer_cache')
    generated, evidence = await generate()
    cache.store(snapshot, query, vector, generated)
    return generated.text, evidence
