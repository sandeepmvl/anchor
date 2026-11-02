from math import log2

import pytest

from eval.retrieval_eval import evaluate
from eval.retrieval_metrics import score


def test_binary_hand_calculation():
    labels = dict(A=1, B=1, C=0, D=0, E=0, F=0)
    result = score(['C', 'A', 'D', 'E', 'F', 'B'], labels)
    assert result['recall@5'] == 0.5
    assert result['rr'] == 0.5
    assert result['nDCG@5'] == pytest.approx((1 / log2(3)) / (1 + 1 / log2(3)))


def test_graded_hand_calculation():
    labels = dict(A=3, B=2, C=1, X=0)
    result = score(['B', 'X', 'A'], labels)
    ideal = 7 + 3 / log2(3) + 1 / log2(4)
    assert result['recall@5'] == pytest.approx(2 / 3)
    assert result['rr'] == 1
    assert result['nDCG@5'] == pytest.approx((3 + 7 / log2(4)) / ideal)


def test_duplicate_document_hits_are_collapsed_before_cutoff():
    labels = dict(A=3, B=2, C=0, D=0, E=0, F=0)
    repeated = score(['C', 'A', 'A', 'A', 'D', 'E', 'B', 'F'], labels)
    unique = score(['C', 'A', 'D', 'E', 'B', 'F'], labels)
    assert repeated == unique
    assert repeated['recall@5'] == 1
    assert repeated['rr'] == 0.5


def test_missing_results_and_short_lists():
    labels = dict(A=3, B=2, X=0)
    assert score([], labels) == {'recall@5': 0, 'rr': 0, 'nDCG@5': 0}
    assert score(['X'], labels) == score([], labels)
    partial = score(['A'], labels)
    assert partial['recall@5'] == 0.5
    assert partial['nDCG@5'] == pytest.approx(7 / (7 + 3 / log2(3)))


def test_mrr_uses_full_supplied_document_list():
    labels = dict(A=1, B=0, C=0, D=0, E=0, F=0)
    result = score(['B', 'C', 'D', 'E', 'F', 'A'], labels)
    assert result['recall@5'] == result['nDCG@5'] == 0
    assert result['rr'] == pytest.approx(1 / 6)


def test_invalid_labels_and_unjudged_results_fail_closed():
    for labels in ({'A': 0}, {'A': -1}, {'A': 4}):
        with pytest.raises(ValueError):
            score(['A'], labels)
    with pytest.raises(ValueError, match='unjudged_document'):
        score(['unknown'], {'A': 1})
    with pytest.raises(ValueError, match='invalid_cutoff'):
        score(['A'], {'A': 1}, k=0)


def test_versioned_fixture_comparison(monkeypatch):
    monkeypatch.setenv('ANCHOR_EMBEDDING_PROVIDER', 'offline')
    report = evaluate(verbose=False)
    assert report['queries'] == 24
    assert report['dataset_version'] == 'advanced-queries-v1'
    for metric in ('recall@5', 'nDCG@5'):
        assert report['means']['hybrid'][metric] > report['means']['dense'][metric]
    for row in report['rows']:
        assert len(row['rankings']['dense']) == len(row['rankings']['hybrid']) == 10
