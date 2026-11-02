from pathlib import Path
from unittest.mock import patch
import pytest
import yaml
from anchor_ai.customization import ROOT, score


def features():
    path = ROOT / 'labs/customization-decision/features.yml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))['features']


def test_worked_recommendations():
    refund, tone, policy = [score(f) for f in features()]
    assert refund['totals'] == {'prompt': 9, 'RAG': 0, 'fine-tune': 2}
    assert tone['totals'] == {'prompt': 4, 'RAG': 1, 'fine-tune': 13}
    assert policy['totals'] == {'prompt': 2, 'RAG': 9, 'fine-tune': 2}
    assert [r['recommendation'] for r in (refund, tone, policy)] == ['prompt', 'fine-tune', 'RAG']
    assert tone['fine_tune_eligible'] and tone['experiment_required']
    assert 'Gated experiment only' in tone['explanation']
    assert 'not memorized weights' in policy['explanation']


def test_prompt_fix_overrides_training_total():
    result = score(dict(features()[1], residual_failure=False))
    assert result['totals']['fine-tune'] > result['totals']['prompt']
    assert not result['fine_tune_eligible']
    assert result['recommendation'] == 'prompt'


def test_volatility_strengthens_rag_margin():
    candidate = features()[2]
    before = score(candidate)
    after = score(dict(candidate, knowledge_volatility=3))
    assert after['recommendation'] == 'RAG'
    assert before['totals']['RAG'] == 9
    assert after['totals']['RAG'] == 17
    assert after['totals']['prompt'] == before['totals']['prompt']
    assert not after['fine_tune_eligible']


@pytest.mark.parametrize('changes', [{'labels': 'probably'}, {'knowledge_volatility': 6}])
def test_invalid_evidence(changes):
    with pytest.raises(ValueError):
        score(dict(features()[0], **changes))


def test_recommendations_have_no_execution_side_effects():
    candidates = features()
    config_paths = list((ROOT / 'config').glob('*.yml'))
    before = {path: path.read_bytes() for path in config_paths}
    with patch('builtins.open', side_effect=AssertionError('No file writes or opens allowed')):
        with patch.object(Path, 'write_text', side_effect=AssertionError('No writes allowed')):
            from anchor_ai.customization import load_policy
            # Load before scoring; Path.read_text uses io.open, not builtins.open.
            policy = load_policy()
            results = [score(f, policy) for f in candidates]
    assert {path: path.read_bytes() for path in config_paths} == before
    for result in results:
        assert result['advisory_only'] is True
        assert result['refund_authorized'] is False
        assert result['model_override'] is None
