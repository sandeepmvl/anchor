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


def test_model_release_pinning_promotion_and_rollback(tmp_path):
    import runpy
    from eval.release_gate import manifest, offline_bundle, save
    from ops.deploy import model_release

    real_transition = model_release.transition

    def transition_with_evidence(candidate=None, rollback=False,
                                 path=model_release.STATE,
                                 manifest_path=model_release.MANIFEST,
                                 evidence=None):
        # Supply fixtures only at the test boundary; retain the real release gate.
        if evidence is None:
            if rollback:
                target = model_release.current_state(path, manifest_path)['previous']
                assert target is not None
            else:
                target = model_release.approved_entry(candidate, manifest_path)
            expected = manifest('model', target)
            evidence = save(tmp_path / 'customization-evaluation.json',
                            offline_bundle(expected))
        return real_transition(candidate, rollback, path, manifest_path,
                               evidence=evidence)

    helper = Path(__file__).resolve().parents[2] / 'labs/adapter-merging/release_checks.py'
    with patch.object(model_release, 'transition', side_effect=transition_with_evidence):
        runpy.run_path(str(helper))['check_selection'](tmp_path)


def test_dpo_safety_is_not_overridden_by_win_rate():
    import copy
    from anchor_ai.customization import dpo_candidate_decision
    manifest = {'candidate': 'dpo-recorded', 'experiment_id': 'bound-run'}
    analysis = {'source': 'constructed_recording_not_measured',
                'preference': [{'step': 1, 'win_rate': .5},
                               {'step': 100, 'win_rate': .875}],
                'reward_hacking_flag': False}
    safety = {'candidate': 'dpo-recorded', 'experiment_id': 'bound-run',
              'source': 'constructed_recording_not_measured',
              'checks': dict.fromkeys(('grounding', 'privacy', 'injection',
                                       'escalation', 'tool_safety'), True)}
    assert dpo_candidate_decision(manifest, analysis, safety)['offline_approved']
    for check in safety['checks']:
        unsafe = copy.deepcopy(safety)
        unsafe['checks'][check] = False
        decision = dpo_candidate_decision(manifest, analysis, unsafe)
        assert not decision['offline_approved']
        assert check in decision['blockers']
        assert decision['model_override'] is None
