import copy

import numpy as np
import pytest

from anchor_ai.dense_retriever import (
    DenseRetriever, RecordedEmbeddingClient, RetrievalError, fixture_dependencies)

QUERY = 'How do I get a refund?'


@pytest.mark.parametrize('metric,normalize', [('cosine', True), ('ip', False), ('l2', False)])
def test_shortage_recovers_tenant_count_with_active_model(metric, normalize):
    generation, embeddings = fixture_dependencies()
    generation.update(metric=metric, normalize=normalize)
    retriever = DenseRetriever(generation, 1, embeddings)
    hits = retriever.search(QUERY, 3)
    assert [h['id'] for h in hits] == [101, 102, 103]
    assert all(h['tenant_id'] == 1 for h in hits)
    assert retriever.last_trace['post_filter_count'] == 0
    assert retriever.last_trace['filter_path'] == 'tenant_flat'
    assert embeddings.calls == [(QUERY, generation['model'], generation['revision'], 2)]
    assert [h['score'] for h in hits] == sorted([h['score'] for h in hits], reverse=True)
    assert len(retriever.search(QUERY, 5)) == 3
    assert retriever.search(QUERY, 3) == hits


def test_unknown_tenant_and_missing_trusted_scope():
    generation, embeddings = fixture_dependencies()
    assert DenseRetriever(generation, 999, embeddings).search(QUERY, 3) == []
    for tenant in (None, '', True, 0):
        with pytest.raises(RetrievalError, match='trusted_tenant_required'):
            DenseRetriever(generation, tenant, embeddings)


def test_approximate_selector_shortage_still_needs_exact_fallback():
    class RestrictedVisitedLists:
        def search(self, query, limit):
            # An eligible-ID filter cannot visit unsearched lists.
            return (np.array([[0.8] + [0.0] * (limit - 1)], dtype='float32'),
                    np.array([[101] + [-1] * (limit - 1)], dtype='int64'))
    generation, embeddings = fixture_dependencies()
    retriever = DenseRetriever(generation, 1, embeddings, index=RestrictedVisitedLists())
    assert len(retriever.search(QUERY, 3)) == 3
    assert retriever.last_trace['post_filter_count'] == 1
    assert retriever.last_trace['filter_path'] == 'tenant_flat'


def test_stable_ids_ties_and_model_revision_are_not_dimension_checks():
    generation, embeddings = fixture_dependencies()
    generation['chunks'] = [dict(id=i, tenant_id=1, text='Policy', vector=[1.0, 0.0])
                            for i in (103, 101, 102)]
    retriever = DenseRetriever(generation, 1, embeddings)
    assert [h['id'] for h in retriever.search(QUERY, 3)] == [101, 102, 103]
    changed = copy.deepcopy(generation)
    changed['revision'] = 'different-revision'
    with pytest.raises(RetrievalError, match='embedding_replay_miss'):
        DenseRetriever(changed, 1, embeddings).search(QUERY, 3)


def test_bad_vector_and_index_metadata_mismatch_fail_closed():
    generation, embeddings = fixture_dependencies()
    bad = RecordedEmbeddingClient({(QUERY, generation['model'], generation['revision'], 2): [1.0]})
    with pytest.raises(RetrievalError, match='invalid_vector'):
        DenseRetriever(generation, 1, bad).search(QUERY, 3)
