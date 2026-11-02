from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from anchor_ai.training import estimate_training_memory, experiment, quantized_training_preflight

ROOT = Path(__file__).resolve().parents[2]


def policy():
    return yaml.safe_load((ROOT / 'config/fine_tuning.yml').read_text())


def test_components_and_total():
    report = estimate_training_memory(16384, 100, 'NF4', 2000, 3000, 100000)
    assert report['components_bytes'] == {
        'base_weights': 8192,
        'quantization_metadata': 260,
        'adapters': 200,
        'gradients': 200,
        'optimizer_state': 800,
        'activation_allowance': 2000,
        'runtime_overhead': 3000,
    }
    assert report['estimated_total_bytes'] == 14652
    assert report['accepted'] is True
    assert report['evidence_status'] == 'estimate_not_gpu_run'


def test_formats_and_double_quantization():
    reports = {m: estimate_training_memory(16384, 100, m, 2000, 3000, 100000)
               for m in ('fp16', '8-bit', 'NF4')}
    assert [reports[m]['components_bytes']['base_weights']
            for m in reports] == [32768, 16384, 8192]
    assert [reports[m]['components_bytes']['quantization_metadata']
            for m in reports] == [0, 1024, 260]
    ordinary = estimate_training_memory(16384, 100, 'NF4', 2000, 3000, 100000, False)
    assert ordinary['components_bytes']['quantization_metadata'] == 1024
    for key in ('adapters', 'gradients', 'optimizer_state', 'activation_allowance', 'runtime_overhead'):
        assert ordinary['components_bytes'][key] == reports['NF4']['components_bytes'][key]


def test_budget_boundary_and_weight_only_false_positive():
    exact = estimate_training_memory(16384, 100, 'NF4', 2000, 3000, 14652)
    rejected = estimate_training_memory(16384, 100, 'NF4', 2000, 3000, 14651)
    assert exact['accepted'] is True
    assert rejected['components_bytes']['base_weights'] < rejected['budget_bytes']
    assert rejected['accepted'] is False
    assert rejected['decision'] == 'rejected_memory_budget'
    assert rejected['warning'] == 'Estimated total exceeds selected hardware budget'
    assert rejected['training_launched'] is False
    assert rejected['checkpoint_created'] is False


@pytest.mark.parametrize('field', range(5))
def test_invalid_memory_assumptions(field):
    arguments = [16384, 100, 2000, 3000, 100000]
    arguments[field] = -1
    with pytest.raises(ValueError, match='invalid_memory_assumption'):
        estimate_training_memory(arguments[0], arguments[1], 'NF4', *arguments[2:])


def test_anchor_manifest_rejects_same_request_under_smaller_budget():
    original = policy()
    accepted = experiment(original)
    report = accepted['qlora_preflight']
    assert report['accepted'] is True
    assert report['adapter'] == accepted['contract']['adapter']
    smaller = deepcopy(original)
    smaller['sft']['quantized_training']['gpu_budget_bytes'] = 1_000_000_000
    rejected = experiment(smaller)
    result = rejected['qlora_preflight']
    assert result['components_bytes'] == report['components_bytes']
    assert result['components_bytes']['base_weights'] < result['budget_bytes']
    assert result['accepted'] is False
    assert result['decision'] == 'rejected_memory_budget'
    assert rejected['training_launched'] is False
    assert rejected['checkpoint_created'] is False
    assert rejected['training_release'] is False
    assert accepted['experiment_id'] != rejected['experiment_id']
    assert accepted['selection'] == rejected['selection']
    assert accepted['evidence_status'] == 'constructed_recording_not_measured'


def test_optional_configuration_preserves_previous_experiment_contract():
    previous = policy()
    del previous['sft']['quantized_training']
    manifest = experiment(previous)
    assert 'qlora_preflight' not in manifest
    assert 'proposed_quantized_training' not in manifest['contract']
    assert manifest['parameter_state_bytes'] == 8_000_000_000
    assert manifest['training_launched'] is False


def test_quantized_training_requires_an_adapter():
    spec = deepcopy(policy()['sft'])
    del spec['adapter']
    with pytest.raises(ValueError, match='quantized_training_requires_adapter'):
        quantized_training_preflight(spec)
