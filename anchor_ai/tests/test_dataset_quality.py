import asyncio
from copy import deepcopy
import json

import pytest
from anchor_ai.customization import load_policy, synthetic_batch_decision
from anchor_ai.dataset_quality import generate, filter_rows, sanitize, unique_words, write_rows


def sources():
    policy = load_policy()
    seed = [{'id': f'seed-{i}', 'tenant_id': 'demo', 'source': 'real',
             'split': 'train', 'training_permitted': True,
             'input': 'Reviewed refund request ' + unique_words(f'seed-{i}'),
             'output': policy['synthetic']['approved_outputs']['refund_review'],
             'label': 'refund_review'} for i in range(100)]
    registry = {row['id']: row for row in seed}
    for split in ('validation', 'held_out'):
        registry[split] = dict(seed[0], id=split, split=split)
    return policy, seed, registry


def test_offline_mixed_quality_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    policy, seed, registry = sources()
    raw = asyncio.run(generate(seed, 200, registry, policy))
    assert raw == asyncio.run(generate(seed, 200, registry, policy))
    clean, rejected, summary = filter_rows(raw, seed, registry, policy)
    assert (len(raw), len(clean), len(rejected)) == (200, 80, 120)
    reasons = {reason for row in rejected for reason in row['quality']['reasons']}
    assert {'pii_detected', 'near_duplicate', 'unsupported_policy_claim',
            'inconsistent_labels', 'held_out_lineage', 'tenant_boundary'} <= reasons
    assert all(not row['quality']['reasons'] for row in clean)
    assert summary['reject_rate'] == 0.6
    assert summary['synthetic_real_ratio'] == 0.8
    assert summary['decision'] == 'eligible_for_review'
    assert summary['merged'] is False
    path = tmp_path / 'raw.jsonl'
    write_rows(path, raw)
    assert 'invented.person@example.test' not in path.read_text()
    assert '[REDACTED_EMAIL]' in path.read_text()
    assert sanitize('Call +1 202 555 0199') == 'Call [REDACTED_PHONE]'


@pytest.mark.parametrize('split', ['validation', 'held_out'])
def test_recursive_nontraining_ancestry_cannot_be_laundered(split, monkeypatch):
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    policy, seed, registry = sources()
    raw = asyncio.run(generate(seed, 1, registry, policy))
    registry['middle'] = dict(seed[1], id='middle', source='synthetic',
                              parent_ids=[split], split='train')
    raw[0]['parent_ids'] = ['middle']
    raw[0]['split'] = 'train'
    clean, rejected, _ = filter_rows(raw, seed, registry, policy)
    assert not clean
    assert 'held_out_lineage' in rejected[0]['quality']['reasons']


def test_duplicate_between_candidates_and_untrusted_metadata(monkeypatch):
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    policy, seed, registry = sources()
    first = asyncio.run(generate(seed, 1, registry, policy))[0]
    duplicate = deepcopy(first)
    duplicate['id'] = 'another-id'
    clean, rejected, _ = filter_rows([first, duplicate], seed, registry, policy)
    assert len(clean) == 1
    assert 'near_duplicate' in rejected[0]['quality']['reasons']
    for changes, reason in [({'parent_ids': ['missing']}, 'unknown_lineage'),
                            ({'training_permitted': False}, 'training_not_permitted'),
                            ({'tenant_id': 'elsewhere'}, 'tenant_boundary')]:
        candidate = dict(first, **changes)
        assert reason in filter_rows([candidate], seed, registry, policy)[1][0]['quality']['reasons']


def test_batch_gate_counts_existing_synthetic_and_holds_empty_batches():
    policy = load_policy()
    assert synthetic_batch_decision(200, 120, 100, 0, 80, policy)['within_bands']
    assert not synthetic_batch_decision(200, 120, 100, 30, 80, policy)['within_bands']
    assert synthetic_batch_decision(0, 0, 100, 0, 0, policy)['decision'] == 'hold'
    assert synthetic_batch_decision(10, 0, 100, 0, 10, policy)['decision'] == 'hold'
    with pytest.raises(ValueError):
        synthetic_batch_decision(10, 2, 100, 0, 9, policy)
