import json
import shutil

import numpy as np
import pytest

from anchor_ai.adapters import (
    ROOT, artifact_contract, comparisons, load_slice, numeric_report,
    precision_evidence, read_json,
)

FIXTURE = ROOT / 'eval/finetune/fixtures/qlora-adapter'
BASE = ('anchor-offline-intent-base', 'constructed-slice-v1')


def test_fold_matches_unmerged_floating_outputs():
    model = load_slice(FIXTURE, *BASE)
    x = np.random.default_rng(3687).normal(size=(4, 9)).astype(np.float32)
    np.testing.assert_allclose(model.merged() @ x, model.forward(x), rtol=1e-5, atol=1e-6)
    report = numeric_report(FIXTURE)
    assert report['additional_nf4_merge_error'] > 0
    assert report['wrong_alpha_self_output_error'] < 1e-6


def test_dimensions_and_base_identity(tmp_path):
    with pytest.raises(ValueError, match='base_identity'):
        load_slice(FIXTURE, 'different-base', BASE[1])
    with pytest.raises(ValueError, match='base_identity'):
        load_slice(FIXTURE, BASE[0], 'different-revision')
    target = tmp_path / 'adapter'
    shutil.copytree(FIXTURE, target)
    (target / 'b.json').write_text('[[1, 2]]')
    with pytest.raises(ValueError, match='shapes'):
        load_slice(target, *BASE)


def test_scaling_is_captured_at_load():
    config = read_json(FIXTURE / 'adapter_config.json')
    correct = load_slice(FIXTURE, *BASE, config=config)
    config['lora_alpha'] = 1
    assert correct.scale == 2
    wrong = load_slice(FIXTURE, *BASE, config=config)
    assert wrong.scale == .5
    np.testing.assert_allclose(wrong.merged(), wrong.forward(np.eye(4)), atol=1e-6)
    assert not np.allclose(correct.merged(), wrong.merged())


def test_recorded_paired_regressions_use_independent_reference():
    _, record, hashes = artifact_contract()
    assert len(hashes) == 5
    rows = comparisons(record)
    assert rows['fp_reload']['regression'] is False
    assert rows['wrong_alpha']['regression'] is True
    assert rows['nf4_merge']['regression'] is True
    assert rows['wrong_alpha']['difference'] < rows['nf4_merge']['difference'] < 0
    assert record['scores']['wrong_alpha'] == record['scores']['wrong_alpha_unmerged']


def test_quantized_artifact_requires_separate_evidence():
    precision_evidence('floating', {})
    with pytest.raises(ValueError, match='separate_quantized'):
        precision_evidence('quantized', {'artifact_kind': 'floating', 'passed': True})
    with pytest.raises(ValueError, match='separate_quantized'):
        precision_evidence('quantized', {'artifact_kind': 'quantized', 'passed': True})
    precision_evidence('quantized', {
        'artifact_kind': 'quantized', 'artifact_hash': 'separately-evaluated-example', 'passed': True})


def test_request_client_retains_release(tmp_path, monkeypatch):
    from anchor_ai.provider import configured_provider
    from ops.deploy.model_release import transition
    state = tmp_path / 'request-state.json'
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.setenv('ANCHOR_MODEL_RELEASE_STATE', str(state))
    before = configured_provider()
    transition('adapter-recorded', path=state)
    after = configured_provider()
    ticket = 'Please refund my order.'
    assert before.selection.complete(ticket) != after.selection.complete(ticket)
    transition(rollback=True, path=state)
    assert after.selection.candidate == 'adapter-recorded'
    assert configured_provider().selection.complete(ticket) == before.selection.complete(ticket)
