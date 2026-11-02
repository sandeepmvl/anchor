import asyncio
import json

import pytest

from anchor_ai.hybrid_retriever import HybridRuntime, Snapshot, rrf_fuse
from anchor_ai.rag import (RecordedGenerationClient, published_hybrid_rag_runtime)
from anchor_ai.reranking import OfflinePairScorer, Reranker, make_reranker
from eval.rerank_bench import RefundFixture, JUDGMENTS, document_ranking
from eval.retrieval_metrics import score


def run(awaitable):
    return asyncio.run(awaitable)


def test_factory(monkeypatch):
    monkeypatch.delenv('ANCHOR_RERANKER', raising=False)
    assert make_reranker() is None
    monkeypatch.setenv('ANCHOR_RERANKER', 'offline')
    assert isinstance(make_reranker().scorer, OfflinePairScorer)
    fixture = RefundFixture()
    assert fixture.runtime().reranker is not None
    assert fixture.runtime(reranker=None).reranker is None
    monkeypatch.setenv('ANCHOR_RERANKER', 'unknown')
    with pytest.raises(ValueError, match='unknown_reranker'):
        make_reranker()


def test_pair_cap_metadata_and_stable_ties():
    rows = [dict(id=i, text='refund damaged' if i == 20 else 'refund',
                 tenant_id=1, generation_id='g1', score=1 / i,
                 metadata={'source': i}) for i in range(1, 26)]
    original = [dict(row) for row in rows]

    class Spy(OfflinePairScorer):
        def score_pairs(self, pairs):
            self.pairs = pairs
            return super().score_pairs(pairs)

    scorer = Spy()
    reranker = Reranker(scorer)
    result = reranker.rerank('refund damaged', rows)
    assert len(scorer.pairs) == 20
    assert [row['id'] for row in result] == [20] + list(range(1, 20))
    assert reranker.rerank('refund damaged', rows) == result
    assert rows == original
    for row in result:
        assert {key: value for key, value in row.items()
                if key != 'rerank_score'} == original[row['id'] - 1]


@pytest.mark.parametrize('values', [[], [float('nan')], [float('inf')], [True]])
def test_invalid_scores(values):
    class BadScorer:
        def score_pairs(self, pairs):
            return values
    with pytest.raises(ValueError, match='invalid_pair_scores'):
        Reranker(BadScorer()).rerank('refund', [dict(id=1, text='refund')])


def test_disabled_preserves_complete_legacy_hits():
    fixture = RefundFixture()
    runtime = fixture.runtime(reranker=None)
    actual = run(runtime.retrieve('refund damaged order', 3))
    branch = fixture.search('refund damaged order', fixture.snapshot, 5)
    expected = [dict(row, tenant_id=1, text=fixture.snapshot.chunks[row['id'] - 1]['text'])
                for row in rrf_fuse(branch, branch)[:3]]
    assert actual == dict(tenant_id=1, generation_id='rerank-g1', hits=expected)
    assert all(depth == 5 for depth in fixture.depths)


def test_enabled_scores_before_final_selection():
    fixture = RefundFixture()
    runtime = fixture.runtime(reranker=Reranker(OfflinePairScorer()))
    result = run(runtime.retrieve('refund damaged order', 5))
    assert [hit['id'] for hit in result['hits']] == list(range(16, 21))
    assert fixture.depths == [20, 20]
    assert result['hits'][0]['metadata'] == {'section': 'policy'}
    assert score(document_ranking(result['hits'], fixture.snapshot),
                 JUDGMENTS)['precision@5'] == 1


@pytest.mark.parametrize('bad_field,bad_value', [
    ('tenant_id', 2), ('generation_id', 'other'), ('id', 999)])
def test_scope_checked_before_scoring(bad_field, bad_value):
    fixture = RefundFixture()

    class Never:
        def score_pairs(self, pairs):
            pytest.fail('unauthorized text reached scorer')

    def bad_search(query, snapshot, depth):
        hit = dict(id=1, tenant_id=1, generation_id=snapshot.generation_id)
        hit[bad_field] = bad_value
        return [hit]

    runtime = HybridRuntime(1, fixture.acquire_snapshot, bad_search, fixture.search,
                            reranker=Reranker(Never()))
    with pytest.raises(ValueError, match='hit_scope_mismatch'):
        run(runtime.retrieve('refund'))


def test_snapshot_scope_checked_before_scoring():
    fixture = RefundFixture()
    runtime = fixture.runtime(reranker=Reranker(OfflinePairScorer()))
    alien = Snapshot(2, 'alien', (dict(id=1, tenant_id=2, text='private'),))
    with pytest.raises(ValueError, match='snapshot_scope_mismatch'):
        run(runtime.retrieve('refund', snapshot=alien))
    assert fixture.depths == []


def test_generation_trajectory(monkeypatch):
    monkeypatch.setenv('ANCHOR_RERANKER', 'offline')
    fixture = RefundFixture()
    reader = RecordedGenerationClient()
    runtime = published_hybrid_rag_runtime(1, fixture.acquire_snapshot,
        fixture.search, fixture.search, client=reader, evidence_budget=2000)
    result = run(runtime.answer('refund damaged order', k=5,
                                request_id='test', trace_id='test'))
    assert len(reader.calls) == 1
    payload = json.loads(reader.calls[0][-1]['content'])
    sources = payload['evidence']['sources']
    assert sources[0]['id'] == 16
    assert {source['id'] for source in sources} == set(range(16, 21))
    assert result['citations'] == [16]
    assert result['abstained'] is False
    assert 'Damaged order refund policy' in result['reply']


@pytest.mark.parametrize('ranked,expected', [([], 0), ([1], .2),
    ([1, 2], .4), ([1, 1, 2], .4), ([1, 2, 3], .4)])
def test_precision_fixed_denominator(ranked, expected):
    assert score(ranked, {1: 1, 2: 2, 3: 0})['precision@5'] == expected


def test_precision_nondefault_cutoff():
    assert score([1], {1: 1}, k=2)['precision@2'] == .5
