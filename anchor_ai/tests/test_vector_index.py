import numpy as np
import pytest

from anchor_ai.vector_index import Generation, VectorIndex


G1 = Generation('policy-model', 'v1', 2)
G2 = Generation('policy-model', 'v2', 2)


def row(tenant, chunk, vector, generation=G1):
    return dict(tenant_id=tenant, chunk_id=chunk, vector=vector,
                model=generation.model, revision=generation.revision,
                dimensions=generation.dimensions)


def test_exact_deterministic_tenant_and_generation_partitions():
    rows = [row('a', 'policy-b', [1, 0]),
            row('a', 'policy-a', [1, 0]),
            row('a', 'other', [0, 1]),
            row('b', 'foreign', [1, 0]),
            row('a', 'new-generation', [1, 0], G2)]
    index = VectorIndex()
    assert index.load('a', G1, rows) == 3
    index.load('b', G1, rows)
    index.load('a', G2, rows)
    expected = ['policy-a', 'policy-b', 'other']
    for _ in range(2):
        hits = index.search('a', G1, [1, 0])
        assert [hit['chunk_id'] for hit in hits] == expected
        assert [hit['score'] for hit in hits] == [1, 1, 0]
    assert index.search('b', G1, [1, 0])[0]['chunk_id'] == 'foreign'
    assert index.search('a', G2, [1, 0])[0]['chunk_id'] == 'new-generation'
    assert index.search('missing', G1, [1, 0]) == []
    assert index.search('a', Generation('other-model', 'v1', 2), [1, 0]) == []


def test_load_rejects_dimensions_without_replacing_partition():
    index = VectorIndex()
    index.load('a', G1, [row('a', 'original', [1, 0])])
    with pytest.raises(ValueError, match='dimension'):
        index.load('a', G1, [row('a', 'bad-vector', [1, 0, 0])])
    bad_metadata = row('a', 'bad-metadata', [1, 0])
    bad_metadata['dimensions'] = 3
    with pytest.raises(ValueError, match='dimension'):
        index.load('a', G1, [bad_metadata])
    with pytest.raises(ValueError, match='dimension'):
        index.search('a', G1, [1, 0, 0])
    assert index.search('a', G1, [1, 0])[0]['chunk_id'] == 'original'


@pytest.mark.parametrize('vector', [[0, 0], [np.nan, 1], [np.inf, 1]])
def test_rejects_invalid_cosine_vectors(vector):
    with pytest.raises(ValueError):
        VectorIndex().load('a', G1, [row('a', 'bad', vector)])


def test_metrics_and_copy_on_load():
    rows = [row('a', 'unit', [1, 0]), row('a', 'large', [2, 1])]
    cosine = VectorIndex()
    cosine.load('a', G1, rows)
    inner = VectorIndex('inner_product', False)
    inner.load('a', G1, rows)
    l2 = VectorIndex('l2', False)
    l2.load('a', G1, rows)
    rows[0]['vector'][0] = -10
    assert cosine.search('a', G1, [1, 0], 1)[0]['chunk_id'] == 'unit'
    assert inner.search('a', G1, [1, 0], 1)[0]['chunk_id'] == 'large'
    assert l2.search('a', G1, [1, 0], 1)[0]['score'] == 0


def test_configured_product_baseline():
    index = VectorIndex.from_config()
    assert index.metric == 'cosine'
    assert index.normalize is True


def test_duplicate_ids_are_not_silently_merged():
    with pytest.raises(ValueError, match='duplicate'):
        VectorIndex().load('a', G1, [row('a', 'x', [1, 0]),
                                    row('a', 'x', [0, 1])])
