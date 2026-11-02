from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from anchor_ai import training

ROOT = Path(__file__).resolve().parents[2]


def policy():
    return yaml.safe_load((ROOT / 'config/fine_tuning.yml').read_text())


def test_expected_budget_and_scaling():
    spec = policy()['sft']
    before = deepcopy(spec)
    result = training.validate_adapter(spec)
    # Per block: 8 * (1792 + 1024 + 1024 + 1792 + 5760 * 3).
    assert result['trainable_parameters'] == 4399104
    assert result['effective_scale'] == 2
    assert result['trainable_percent_of_declared_base'] == pytest.approx(0.8798208)
    assert result['base_weights_frozen'] is True
    assert result['use_rslora'] is False
    assert spec == before
    assert len(result['modules']) == 7


@pytest.mark.parametrize('rank', [0, -1, True, 1.5, '8', 129])
def test_invalid_ranks(rank):
    spec = policy()['sft']
    spec['adapter']['rank'] = rank
    with pytest.raises(ValueError, match='invalid_adapter_rank'):
        training.validate_adapter(spec)


@pytest.mark.parametrize('targets', [[], ['missing_proj'], ['q_proj', 'q_proj'],
                                     'q_proj', [None]])
def test_invalid_targets(targets):
    spec = policy()['sft']
    spec['adapter']['target_modules'] = targets
    with pytest.raises(ValueError, match='invalid_adapter_targets'):
        training.validate_adapter(spec)


@pytest.mark.parametrize('alpha', [0, -1, True, float('nan'), float('inf')])
def test_invalid_alpha(alpha):
    spec = policy()['sft']
    spec['adapter']['alpha'] = alpha
    with pytest.raises(ValueError, match='invalid_adapter_alpha'):
        training.validate_adapter(spec)


def test_budget_and_coverage():
    spec = policy()['sft']
    spec['adapter']['max_trainable_parameters'] = 4399103
    with pytest.raises(ValueError, match='adapter_parameter_budget_exceeded'):
        training.validate_adapter(spec)
    spec['adapter']['target_modules'] = ['q_proj', 'v_proj']
    result = training.validate_adapter(spec)
    assert result['trainable_parameters'] == 540672


def test_forward_scale_merge_and_frozen_base():
    weight = np.arange(12, dtype=float).reshape(3, 4)
    original = weight.copy()
    a = np.ones((2, 4))
    b = np.ones((3, 2))
    x = np.array([1., 2., 3., 4.])
    output = training.lora_forward(weight, a, b, x, alpha=4)
    np.testing.assert_allclose(output, weight @ x + 2 * b @ a @ x)
    np.testing.assert_allclose(output, (weight + 2 * b @ a) @ x)
    np.testing.assert_array_equal(weight, original)


def test_invalid_adapter_fails_before_loading_loss_log(monkeypatch):
    request = policy()
    request['sft']['adapter']['rank'] = 0
    def forbidden(*args):
        pytest.fail('invalid adapter reached loss-log loading')
    monkeypatch.setattr(training, 'read_jsonl', forbidden)
    with pytest.raises(ValueError, match='invalid_adapter_rank'):
        training.experiment(request)


def test_manifest_identity_includes_adapter():
    request = policy()
    first = training.experiment(request)
    request['sft']['adapter']['alpha'] = 8
    second = training.experiment(request)
    assert first['contract']['adapter']['effective_scale'] == 2
    assert second['contract']['adapter']['effective_scale'] == 1
    assert first['experiment_id'] != second['experiment_id']
    assert first['training_launched'] is False
    assert first['parameter_state_bytes'] == 8000000000
