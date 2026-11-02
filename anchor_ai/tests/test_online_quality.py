from unittest.mock import patch
import pytest
from anchor_ai import online_quality as q
from eval import groundedness_scorer as scorer


def test_seeded_sampling():
    p = dict(q.policy(), sample_rate=0.05)
    a = [q.selected(str(i), p) for i in range(1000)]
    assert a == [q.selected(str(i), p) for i in range(1000)]
    assert 0 < sum(a) < 1000
    assert all(q.selected(str(i), dict(p, sample_rate=1)) for i in range(10))
    assert not q.selected('x', dict(p, sample_rate=0))


def test_existing_prompt_and_scorer_are_reused():
    payload = {'answer': 'unsupported', 'evidence': [], 'citations': []}
    judge = object()
    with patch.object(scorer, 'score_payload', return_value={'status': 'scored', 'score': 0.0}) as shared:
        assert q.score(payload, judge) == {'status': 'scored', 'score': 0.0}
        shared.assert_called_once_with(payload, judge)
    with patch.object(q, 'score_online', return_value={'status': 'no_claims', 'score': None}):
        assert q.score(payload)['status'] == 'insufficient'
    with patch.object(q, 'score_online', side_effect=RuntimeError('judge unavailable')):
        with pytest.raises(RuntimeError):
            q.score(payload)


def test_sustained_drop_and_insufficient_windows():
    p = q.policy()
    group = ['cedar', 'refund', 'v2', p['evaluator_version']]
    values = [[1]*5, [0]*5, [0]*5, [None]*5, [0]*5, [0]*4, [0]*5, [0]*5]
    rows = [{'group': group, 'window': i, 'scores': v} for i, v in enumerate(values)]
    result = q.windows(rows, p)
    assert [r['drift'] for r in result] == [False, False, True, False, False, False, False, True]
    assert result[3]['mean_groundedness'] is None
    assert result[5]['status'] == 'insufficient'
    assert not q.windows([rows[1], rows[4]], p)[-1]['drift']
    other = dict(rows[2], group=['birch', 'refund', 'v2', p['evaluator_version']])
    assert not q.windows([other], p)[0]['drift']
    with pytest.raises(ValueError):
        q.windows([rows[0], rows[0]], p)
