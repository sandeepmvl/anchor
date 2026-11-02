import copy
import json
import subprocess
import sys

import pytest

from anchor_ai.serving_artifacts import (
    ROOT, compare, numerical_experiment, policy, selection,
)


def artifact(tmp_path, name, fmt='gptq', simulated=True, scenario='clean'):
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps({'format': fmt, 'bits': 4 if fmt != 'fp16' else 16,
                               'simulated': simulated, 'fixture_scenario': scenario,
                               'estimated_weight_bytes': 550000000}))
    return path


@pytest.mark.parametrize('fmt', ['gptq', 'awq', 'gguf'])
def test_simulated_formats_are_never_weights(tmp_path, fmt):
    base = artifact(tmp_path, 'base', 'fp16')
    cand = artifact(tmp_path, 'candidate', fmt)
    runtime = 'llama_cpp_cpu' if fmt == 'gguf' else 'vllm_gpu'
    report = compare(base, cand, runtime)
    decision = selection(cand, runtime, report, base)
    assert decision['selectable'] is False
    assert 'simulated_manifest_not_quantized_weights' in decision['blockers']
    assert report['sizes']['candidate']['manifest_bytes'] == cand.stat().st_size
    assert report['sizes']['candidate']['weight_bytes'] is None
    assert report['quality_loss'] == 0
    assert report['candidate']['completed_req_s'] > 0
    assert 'fixture' in report['latency_mode']


def test_runtime_and_quality_are_independent_blockers(tmp_path):
    base = artifact(tmp_path, 'base', 'fp16')
    gguf = artifact(tmp_path, 'gguf', 'gguf')
    assert 'incompatible_runtime' in selection(gguf, 'vllm_gpu')['blockers']
    assert 'incompatible_runtime' not in selection(gguf, 'llama_cpp_cpu')['blockers']
    awq = artifact(tmp_path, 'awq', 'awq', scenario='refund_regression')
    report = compare(base, awq, 'vllm_gpu')
    assert report['quality_loss'] == pytest.approx(1 / 3)
    assert [r['id'] for r in report['baseline']['rows']] == [r['id'] for r in report['candidate']['rows']]
    assert 'quality_threshold_failed' in selection(awq, 'vllm_gpu', report, base)['blockers']


def test_unmeasured_and_forged_real_evidence_fail_closed(tmp_path):
    base = artifact(tmp_path, 'base', 'fp16')
    cand = artifact(tmp_path, 'real_claim', simulated=False)
    assert not selection(cand, 'vllm_gpu')['selectable']
    report = compare(base, cand, 'vllm_gpu')
    report['latency_mode'] = 'measured_real_runtime'
    report['deployment_approved'] = True
    decision = selection(cand, 'vllm_gpu', report, base)
    assert not decision['deployment_approved']
    assert 'trusted_measured_runtime_evidence_required' in decision['blockers']
    assert 'missing_or_invalid_comparison' in decision['blockers']


def test_report_binding_and_scores(tmp_path):
    base = artifact(tmp_path, 'base', 'fp16')
    cand = artifact(tmp_path, 'cand')
    report = compare(base, cand, 'vllm_gpu')
    for field, value in [('candidate_sha256', 'wrong'), ('dataset_sha256', 'wrong'),
                         ('quality_loss', float('nan'))]:
        changed = copy.deepcopy(report)
        changed[field] = value
        assert 'missing_or_invalid_comparison' in selection(cand, 'vllm_gpu', changed, base)['blockers']
    cand.write_text(cand.read_text() + ' ')
    assert 'missing_or_invalid_comparison' in selection(cand, 'vllm_gpu', report, base)['blockers']


def test_rounding_error_is_measured_not_format_evidence():
    low, high = numerical_experiment(4), numerical_experiment(8)
    assert low['mse'] > high['mse'] > 0
    assert 'not_format_quantization' in low['method']


def test_deployment_specification():
    rules = policy()
    launch = rules['vllm_launch']
    assert rules['min_pass_rate'] == 1.0
    assert rules['max_quality_loss'] == 0.0
    assert launch['executable'] == 'vllm' and launch['subcommand'] == 'serve'
    assert launch['host'] == '127.0.0.1' and launch['port'] == 8765
    assert launch['served_model_name'] == 'tiny-llama'
    assert launch['model_path'] == '/srv/anchor/models/approved'
    assert launch['approval_required'] is True
    assert set(launch['supported_formats']) == {'fp16', 'gptq', 'awq'}
    assert rules['compatible_runtimes']['gguf'] == ['llama_cpp_cpu']


def test_gate_expected_rejection_cli(tmp_path):
    cand = artifact(tmp_path, 'gguf', 'gguf')
    command = [sys.executable, str(ROOT / 'eval/model_gate.py'), '--offline',
               '--candidate', 'quantized', '--artifact', str(cand), '--runtime', 'vllm_gpu']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'incompatible_runtime' in result.stdout
    expected = subprocess.run(command + ['--expect-rejected'], capture_output=True, text=True)
    assert expected.returncode == 0
