from copy import deepcopy

import pytest

from anchor_ai.customization import load_policy, sft_activation_decision
from anchor_ai.training import ROOT, analyze_loss, experiment, read_jsonl, recorded_evaluation
from eval.model_gate import replay


def recording(epochs):
    return read_jsonl(ROOT / f'eval/finetune/fixtures/sft_epochs{epochs}.jsonl')


def test_short_recording_is_inconclusive():
    rows = recording(3)
    assert all(b['val_loss'] < a['val_loss'] for a, b in zip(rows, rows[1:]))
    result = analyze_loss(rows)
    assert result['best_epoch'] == 3
    assert result['first_upturn'] is None
    assert result['interpretation'] == 'still_falling_short_run_inconclusive'


def test_eight_epochs_select_validation_not_final_training_loss():
    rows = recording(8)
    assert rows[:3] == recording(3)
    assert min(rows, key=lambda r: r['train_loss'])['epoch'] == 8
    result = analyze_loss(rows)
    assert result['best_epoch'] == 3
    assert result['first_upturn'] == 4
    assert result['selected_checkpoint'] == 'checkpoint-epoch3'
    assert result['interpretation'] == 'overfitting_after_validation_minimum'


def test_manifest_is_reproducible_and_not_a_training_release():
    manifest = experiment()
    assert manifest == experiment()
    assert manifest['train_sources'] == {'curated': 1, 'synthetic': 1}
    assert manifest['parameter_state_bytes'] == 8000000000
    assert manifest['activation_budget_is_additional']
    assert not manifest['training_launched']
    assert not manifest['checkpoint_created']
    assert not manifest['training_release']
    policy = load_policy()
    policy['sft']['recorded_rows'][0]['output'] += ' Changed.'
    assert experiment(policy)['experiment_id'] != manifest['experiment_id']


def test_seed_families_cannot_cross_splits():
    policy = load_policy()
    policy['sft']['recorded_rows'][1]['split'] = 'validation'
    with pytest.raises(ValueError, match='cross_split_family'):
        experiment(policy)


def test_unaccepted_synthetic_is_not_training_input():
    policy = load_policy()
    policy['sft']['recorded_rows'][1]['status'] = 'rejected'
    with pytest.raises(ValueError, match='unaccepted_or_unlinked_synthetic'):
        experiment(policy)


def test_recorded_quality_pass_never_activates():
    report = replay('sft-recorded')
    assert report['evaluation']['passed']
    assert report['evaluation']['quality'] == 1
    assert report['evaluation']['critical_failures'] == 0
    assert len(report['evaluation']['paired_replies']) == 2
    assert not report['decision']['active']
    assert report['decision']['model_override'] is None
    assert 'measured_held_out_evidence' in report['decision']['blockers']
    assert 'existing_retrieval_regression_gate' in report['decision']['blockers']


def measured_contract_objects():
    # Unit-test objects only: no measured model or retrieval result is produced.
    manifest = experiment()
    evidence = recorded_evaluation(manifest)
    manifest.update(training_release=True, evidence_status='measured')
    evidence['source'] = 'measured'
    retrieval = {'gate': 'existing_retrieval_regression', 'passed': True,
                 'candidate': manifest['candidate'],
                 'experiment_id': manifest['experiment_id'],
                 'checkpoint': manifest['selection']['selected_checkpoint'],
                 'artifact_digest': 'unit-test-artifact-not-a-real-gate-result'}
    return manifest, evidence, retrieval


def test_activation_contract_requires_all_three_evidence_groups():
    manifest, evidence, retrieval = measured_contract_objects()
    assert sft_activation_decision(manifest, evidence, retrieval)['active']
    assert not sft_activation_decision(manifest, None, retrieval)['active']
    assert not sft_activation_decision(manifest, evidence, None)['active']
    changed = deepcopy(evidence)
    changed['lineage']['train'] = 'different-dataset'
    assert not sft_activation_decision(manifest, changed, retrieval)['active']
    changed = deepcopy(evidence)
    changed['heldout_digest'] = 'different-heldout'
    assert not sft_activation_decision(manifest, changed, retrieval)['active']
    changed = deepcopy(evidence)
    changed['passed'] = False
    assert not sft_activation_decision(manifest, changed, retrieval)['active']
    changed_gate = dict(retrieval, passed=False)
    assert not sft_activation_decision(manifest, evidence, changed_gate)['active']
    changed_gate = dict(retrieval, checkpoint='checkpoint-epoch8')
    assert not sft_activation_decision(manifest, evidence, changed_gate)['active']
    final = sft_activation_decision(manifest, evidence, retrieval, 'checkpoint-epoch8')
    assert not final['active']
    assert 'validation_selected_checkpoint' in final['blockers']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True])
def test_invalid_losses_are_rejected(value):
    with pytest.raises(ValueError, match='invalid_loss'):
        analyze_loss([{'epoch': 1, 'train_loss': value, 'val_loss': 1.0}])


def test_dpo_manifest_has_frozen_reference_and_offline_lineage():
    from anchor_ai.training import dpo_experiment
    manifest = dpo_experiment()
    contract = manifest['contract']
    assert contract['policy']['id'] != contract['reference']['id']
    assert contract['reference']['frozen'] is True
    assert contract['policy']['initial_checkpoint'] == contract['reference']['checkpoint']
    assert len(set(manifest['lineage'].values())) == 3
    assert manifest['metrics_scope'] == 'offline_evidence_only'
    assert manifest['evidence_status'] == 'constructed_recording_not_measured'
    assert manifest['training_launched'] is False
    assert manifest['checkpoint_created'] is False
    policy = deepcopy(load_policy())
    policy['dpo']['reference']['frozen'] = False
    with pytest.raises(ValueError, match='invalid_reference'):
        dpo_experiment(policy)
