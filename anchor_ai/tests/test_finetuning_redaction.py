import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from anchor_ai import pii
from anchor_ai.customization import sft_examples

ROOT = Path(__file__).resolve().parents[2]


def example():
    return {'input': 'Policy: returns within 30 days. Email customer@example.test or call 212-555-0199.',
            'output': 'Please contact support@example.test or 212-555-0188 for eligibility review.',
            'label': 'refund_review'}


def cli_module():
    spec = importlib.util.spec_from_file_location('redact_cli', ROOT / 'bin/redact_finetuning.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inputs_and_outputs_are_redacted_before_serialization():
    source = example()
    safe = pii.redact_finetuning_batch([source])
    encoded = json.dumps(safe)
    for value in ('customer@example.test', 'support@example.test', '212-555-0199', '212-555-0188'):
        assert value not in encoded
    assert source == example(), 'Do not mutate the raw source'
    for field in ('input', 'output'):
        assert '[REDACTED_EMAIL]' in safe[0][field]
        assert '[REDACTED_PHONE]' in safe[0][field]
        assert not pii.finetuning_detected(safe[0][field])


def test_cli_never_prints_raw_examples():
    result = subprocess.run([sys.executable, str(ROOT / 'bin/redact_finetuning.py')],
                            input=json.dumps([example()]), text=True, capture_output=True, check=True)
    assert result.stderr == ''
    assert 'customer@example.test' not in result.stdout
    assert 'support@example.test' not in result.stdout
    assert json.loads(result.stdout)[0]['label'] == 'refund_review'


def test_failed_redaction_returns_no_partial_output_or_sensitive_error(monkeypatch):
    module = cli_module()
    monkeypatch.setattr(pii, 'redact_finetuning_text', lambda value: value)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps([example()])))
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = module.main()
    assert status == 1
    assert stdout.getvalue() == ''
    assert stderr.getvalue() == 'finetuning_redaction_failed\n'
    with pytest.raises(ValueError, match='^finetuning_redaction_failed$'):
        pii.redact_finetuning_batch([example()])


def test_broken_redactor_aborts_product_export(tmp_path):
    wrapper = tmp_path / 'broken-python'
    wrapper.write_text(
        '#!/bin/sh\nexec "' + sys.executable + '" -c '
        "\"import runpy; from anchor_ai import pii; "
        "pii.redact_finetuning_text=lambda value:value; "
        "runpy.run_path('bin/redact_finetuning.py', run_name='__main__')\"\n")
    wrapper.chmod(0o700)
    out = tmp_path / 'export' / 'train.jsonl'
    code = "require_relative 'app/services/finetuning/dataset_curator'; " + (
        "begin; Finetuning::DatasetCurator.new(tenant: 'demo', since: '30d', "
        "as_of: '2026-01-31', source: 'eval/finetune/fixtures/transcripts.jsonl', "
        "out: ARGV[0], python: ARGV[1]).call; rescue; warn 'export_aborted'; exit 1; end")
    result = subprocess.run(['ruby', '-e', code, str(out), str(wrapper)], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert result.stdout == ''
    assert result.stderr == 'export_aborted\n'
    assert not out.parent.exists()


def test_sft_mapping_uses_only_response_text_as_completion():
    safe = pii.redact_finetuning_batch([example()])
    converted = sft_examples(safe)
    assert converted == [{'prompt': [{'role': 'user', 'content': safe[0]['input']}],
                          'completion': [{'role': 'assistant', 'content': safe[0]['output']}]}]
    assert 'label' not in converted[0]


def test_billing_fixture_tokens_are_stable_and_absent_from_export(monkeypatch):
    monkeypatch.setenv('ANCHOR_PII_HMAC_KEY', 'b9' * 32)
    fixture = json.loads((Path(__file__).parent / 'fixtures/pii_test_fixtures.json').read_text())
    rows = [{'input': 'Policy: returns within 30 days. ' + fixture['message'],
             'output': fixture['response'], 'label': 'refund_review'}]
    safe = pii.redact_finetuning_batch(rows)
    assert pii.redact_finetuning_batch(safe) == safe
    assert '[REDACTED_CARD_' in safe[0]['input']
    result = subprocess.run([sys.executable, str(ROOT / 'bin/redact_finetuning.py')],
                            input=json.dumps(rows), capture_output=True, text=True, check=True)
    assert result.stderr == ''
    exported = json.loads(result.stdout)
    assert exported == safe
    for identifier in fixture['identifiers']:
        assert identifier not in result.stdout
    assert pii.redact_finetuning_batch(exported) == exported
