import copy
from pathlib import Path
import pytest

from anchor_ai.dataset_quality import read_rows, has_pii, audit_preferences
from anchor_ai.preferences import build_pairs, trainer_fields, analyze
from anchor_ai.training import dpo_experiment
from eval.model_gate import replay_dpo

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'eval/finetune/fixtures'


def transcripts():
    return read_rows(FIXTURES / 'preference_transcripts.jsonl')


def test_necessary_handoff_beats_unauthorized_resolution():
    pair = build_pairs(transcripts(), 'demo')[0]
    assert pair['dispositions'] == {'chosen': 'escalated', 'rejected': 'resolved'}
    assert 'human review' in pair['chosen']
    assert 'refund is completed' in pair['rejected']
    assert pair['review']['rationale']
    assert pair['tenant_id'] == 'demo'
    fields = trainer_fields([pair])[0]
    assert set(fields) == {'prompt', 'chosen', 'rejected'}
    assert 'No refund authorization' in fields['prompt']


@pytest.mark.parametrize('mutation', ['tenant', 'context', 'rationale', 'review'])
def test_pair_requires_shared_context_and_real_review(mutation):
    rows = transcripts()
    if mutation == 'tenant':
        rows[0]['replies'][1]['tenant_id'] = 'other'
    elif mutation == 'context':
        rows[0]['replies'][1]['context_id'] = 'different-tool-results'
    elif mutation == 'rationale':
        rows[0]['review']['rationale'] = ''
    else:
        del rows[0]['review']
    with pytest.raises(ValueError):
        build_pairs(rows, 'demo')


def test_dispositions_cannot_choose_a_reply():
    rows = transcripts()
    rows[0]['review']['chosen_id'] = None
    with pytest.raises(ValueError, match='identify_chosen'):
        build_pairs(rows, 'demo')


def test_redaction_happens_before_export():
    rows = transcripts()
    rows[0]['customer'] += ' Contact example.person@example.test.'
    pairs = build_pairs(rows, 'demo')
    assert not has_pii(pairs)
    assert '[REDACTED_EMAIL]' in pairs[0]['prompt']
    assert pairs[0]['pii_redacted'] is True


def test_related_conversations_cannot_cross_splits():
    pairs = build_pairs(transcripts(), 'demo')
    pairs[-1]['family'] = pairs[0]['family']
    with pytest.raises(ValueError, match='cross_split_family'):
        audit_preferences(pairs)


def test_recorded_metrics_are_recomputed_not_trusted():
    logs = read_rows(FIXTURES / 'dpo_training_log.jsonl')
    replies = read_rows(FIXTURES / 'dpo_heldout_replies.jsonl')
    manifest = dpo_experiment()
    report = analyze(logs, replies, manifest['heldout_ids'])
    assert [r['win_rate'] for r in report['preference']] == [0, .5, .75, .875]
    assert report['preference'][0]['ties'] == 8
    assert report['resolved_ticket_eval']['before']['appropriate_rate'] == 1
    assert report['resolved_ticket_eval']['after']['appropriate_rate'] == .25
    assert report['reward_hacking_flag']
    bad = copy.deepcopy(logs)
    bad[-1]['rewards/accuracies'] = 1
    with pytest.raises(ValueError, match='aggregate_mismatch'):
        analyze(bad, replies, manifest['heldout_ids'])


def test_missing_heldout_pair_is_not_silently_accepted():
    logs = read_rows(FIXTURES / 'dpo_training_log.jsonl')
    replies = read_rows(FIXTURES / 'dpo_heldout_replies.jsonl')
    logs[-1]['pairs'].pop()
    with pytest.raises(ValueError, match='coverage_mismatch'):
        analyze(logs, replies, dpo_experiment()['heldout_ids'])


def test_higher_win_rate_unsafe_candidate_is_blocked():
    report = replay_dpo()
    assert report['evaluation']['preference'][-1]['win_rate'] > .7
    assert not report['decision']['offline_approved']
    for name in ('privacy', 'injection', 'grounding', 'escalation', 'tool_safety'):
        assert name in report['decision']['blockers']
    assert report['decision']['model_override'] is None
    assert not report['deployment_approved']
