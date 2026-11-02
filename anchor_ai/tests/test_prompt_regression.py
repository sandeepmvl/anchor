import copy
import subprocess
import sys

import pytest

from eval.regression_provider import fixture_input, digest
from eval.regression_gate import run, gate


def test_same_twenty_examples_and_per_example_results():
    report = run(fixture_input())
    assert report['provider_calls'] == 120
    assert report['cache'] is False
    assert report['cli_integration_verified'] is False
    groups = {tag: {(r['id'], r['repeat']) for r in report['rows'] if r['tag'] == tag}
              for tag in ('chat:v1', 'chat:v2')}
    assert groups['chat:v1'] == groups['chat:v2']
    assert len(groups['chat:v1']) == 60
    for row in report['rows']:
        assert set(row['checks']) == {'contains', 'regex', 'semantic', 'safety'}
        assert row['rubric_source'] == 'constructed_control_not_live_judge'
    assert gate(report)['passed']
    assert not gate(report, release=True)['passed']
    assert report['deployment_approved'] is False


def test_refund_safety_overrides_high_similarity_and_aggregate_rate():
    report = run(fixture_input(), regressed=True)
    bad = [r for r in report['rows'] if r['id'] == 'refund-01' and r['tag'] == 'chat:v2']
    assert all(r['similarity'] > 0.9 for r in bad)
    for row in bad:
        row['similarity'] = 1.0
        row['checks'] = dict.fromkeys(row['checks'], True)
    decision = gate(report)
    assert not decision['passed']
    assert decision['unsafe_ids'] == ['refund-01']
    assert decision['regressed_ids'] == ['refund-01']
    assert 'non_negotiable_safety' in decision['blockers']
    assert decision['assertion_pass_rate'] > 0.95


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'blank', 'manifest', 'hash'])
def test_incomplete_or_changed_evidence_fails(mutation):
    report = copy.deepcopy(run(fixture_input()))
    if mutation == 'missing':
        report['rows'].pop()
    elif mutation == 'duplicate':
        report['rows'][-1] = report['rows'][0]
    elif mutation == 'blank':
        report['rows'][0]['output'] = ''
    elif mutation == 'manifest':
        report['source']['manifest']['expected'] = 'Changed policy'
        report['input_hash'] = digest(report['source'])
    else:
        report['manifest_hash'] = 'wrong'
    with pytest.raises(ValueError):
        gate(report)


def test_cli_gate_really_exits_nonzero(tmp_path):
    path = tmp_path / 'report.json'
    command = [sys.executable, 'eval/regression_gate.py', '--run',
               '--regressed', '--report', str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'refund-01' in result.stdout


def test_hosted_selection_does_not_fall_back(monkeypatch):
    monkeypatch.setenv('ANCHOR_REGRESSION_PROVIDER', 'hosted')
    with pytest.raises(RuntimeError, match='hosted_adapter_not_implemented'):
        run(fixture_input())
