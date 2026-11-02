import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from eval import ragas_eval, trulens_eval
from eval.faithfulness import groundedness_dimension
from eval.triad import (DIMENSIONS, ROOT, RecordedTriadClient, compare_reports,
                        load_rows, normalize)

DATASET = ROOT / 'eval/datasets/anchor_eval_20.jsonl'
ADAPTERS = (ragas_eval.evaluate_batch, trulens_eval.evaluate_batch)


@pytest.fixture
def rows():
    return load_rows(DATASET)


@pytest.mark.parametrize('adapter', ADAPTERS)
def test_reproducible_shared_schema(adapter, rows, monkeypatch):
    monkeypatch.delenv('ANCHOR_EVAL_PROVIDER', raising=False)
    report = adapter(rows)
    assert report == adapter(rows)
    assert len(report['rows']) == 20
    assert report['mode'] == 'offline'
    assert 'synthetic' in report['provenance']
    assert set(report['aggregates']) == set(DIMENSIONS)
    for source, result in zip(rows, report['rows']):
        assert result['id'] == source['id']
        assert result['tenant_id'] == source['tenant_id']
        assert result['generation_id'] == source['generation_id']
        assert set(result['scores']) == set(DIMENSIONS)
    low = {r['id'] for r in report['rows'] if r['scores']['context_relevance'] < 0.5}
    assert low == {'refund-07', 'refund-18'}


def test_exact_metric_mapping_and_aggregates(rows):
    ragas = ragas_eval.evaluate_batch(rows)
    trulens = trulens_eval.evaluate_batch(rows)
    assert ragas['rows'][0]['scores'] == {
        'context_relevance': 0.9, 'groundedness': 0.95, 'answer_relevance': 0.9}
    assert trulens['rows'][0]['scores'] == {
        'context_relevance': 0.91, 'groundedness': 0.94, 'answer_relevance': 0.92}
    assert ragas['aggregates'] == pytest.approx({
        'context_relevance': 0.825, 'groundedness': 0.865, 'answer_relevance': 0.9})
    assert trulens['aggregates'] == pytest.approx({
        'context_relevance': 0.837, 'groundedness': 0.858, 'answer_relevance': 0.92})


@pytest.mark.parametrize('adapter', ADAPTERS)
def test_unknown_replay_key_and_stale_inputs_rejected(adapter, rows):
    unknown = copy.deepcopy(rows)
    unknown[0]['id'] = 'unknown-ticket'
    with pytest.raises(ValueError, match='unknown_replay_key'):
        adapter(unknown)
    for field, value in [('response', 'Changed answer'), ('tenant_id', 'other-tenant'),
                         ('generation_id', 'other-index'), ('category', 'other-category'),
                         ('user_input', 'Changed question'), ('retrieved_contexts', ['Changed'])]:
        changed = copy.deepcopy(rows)
        changed[0][field] = value
        with pytest.raises(ValueError, match='stale_replay_input'):
            adapter(changed)
    with pytest.raises(ValueError, match='duplicate'):
        adapter(rows + [rows[0]])


@pytest.mark.parametrize('framework', ['ragas', 'trulens'])
def test_unknown_metric_keys_rejected(framework, rows):
    client = RecordedTriadClient()
    raw = client.scores(framework, rows[0])
    raw['unknown_metric'] = 0.9
    with pytest.raises(ValueError, match='metric_key'):
        normalize(framework, raw)


@pytest.mark.parametrize('value', [True, -0.1, 1.1, float('nan'), float('inf'), '0.9'])
def test_invalid_scores_rejected(value):
    with pytest.raises(ValueError, match='invalid_triad_score'):
        normalize('trulens', dict(context_relevance=value, groundedness=0.8, answer_relevance=0.9))


def test_comparison_checks_identity(rows, capsys):
    ragas = ragas_eval.evaluate_batch(rows)
    trulens = trulens_eval.evaluate_batch(list(reversed(rows)))
    compare_reports(ragas, trulens)
    assert 'Comparable triad reports' in capsys.readouterr().out
    trulens['dataset_fingerprint'] = 'different'
    with pytest.raises(ValueError, match='incomparable_reports'):
        compare_reports(ragas, trulens)


def test_custom_faithfulness_bridge():
    assert groundedness_dimension({'status': 'scored', 'score': 0.75}) == {'groundedness': 0.75}
    for status in ('no_claims', 'evaluation_error'):
        assert groundedness_dimension({'status': status, 'score': None}) == {'groundedness': None}


def test_live_mode_is_explicitly_unsupported(monkeypatch):
    monkeypatch.setenv('ANCHOR_EVAL_PROVIDER', 'live')
    with pytest.raises(ValueError, match='not_implemented'):
        RecordedTriadClient()


def test_no_framework_imports_network_or_keys():
    script = '''
import importlib.abc
import socket
import sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ragas', 'trulens', 'trulens_eval'}:
            raise AssertionError('optional SDK import: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
def blocked(*args, **kwargs):
    raise AssertionError('network access')
socket.socket = blocked
socket.create_connection = blocked
from eval.ragas_eval import evaluate_batch as ragas
from eval.trulens_eval import evaluate_batch as trulens
from eval.triad import load_rows
rows = load_rows('eval/datasets/anchor_eval_20.jsonl')
assert len(ragas(rows)['rows']) == len(trulens(rows)['rows']) == 20
print('20-case adapters run without optional SDKs, network, or keys')
'''
    env = {key: value for key, value in os.environ.items()
           if 'API_KEY' not in key and key != 'ANCHOR_EVAL_PROVIDER'}
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert 'without optional SDKs' in result.stdout


@pytest.mark.parametrize('module', ['eval.ragas_eval', 'eval.trulens_eval'])
def test_cli(module, tmp_path):
    output = tmp_path / 'report.json'
    result = subprocess.run(
        [sys.executable, '-m', module, '--eval-set', str(DATASET), '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert '20 cases' in result.stdout
    assert len(json.loads(output.read_text())['rows']) == 20
