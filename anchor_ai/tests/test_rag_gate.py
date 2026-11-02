import asyncio
import copy
import json
import subprocess
from pathlib import Path

import pytest

from anchor_ai.rag import RecordedGenerationClient
from eval.heuristic_judge import json_bytes, score_capture
from eval.rag_gate import capture, enforce, load_dataset, run_gate
from eval.triad import RecordedTriadClient, fingerprint


class UnsupportedRefundClient(RecordedGenerationClient):
    """Committed generation fault: keep the citation, invent a refund window."""
    async def complete(self, messages):
        result = json.loads(await super().complete(messages))
        if not result['abstained']:
            result['reply'] = result['reply'].replace('30 days', '300 days')
        return json.dumps(result)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    import eval.rag_gate as gate

    monkeypatch.setenv('ANCHOR_LLM_PROVIDER', 'offline')
    monkeypatch.setenv('ANCHOR_EVAL_PROVIDER', 'offline')
    repository = tmp_path / 'source'
    repository.mkdir()
    subprocess.run(['git', 'init', '-q', str(repository)], check=True)
    subprocess.run([
        'git', '-C', str(repository), '-c', 'user.name=Anchor Test',
        '-c', 'user.email=test@anchor.invalid', 'commit', '-q',
        '--allow-empty', '-m', 'Evaluation test source'
    ], check=True)
    # DATASET remains the absolute path to the real golden dataset.
    # Only Git commands use this isolated, genuinely committed source tree.
    monkeypatch.setattr(gate, 'ROOT', Path(repository))


def current_capture():
    dataset = load_dataset()
    provenance = dict(run_id='test-run', commit_sha='test-commit',
        config_hash=fingerprint(dataset['configuration']),
        dataset_hash=fingerprint(dataset), flag=True)
    captured = asyncio.run(capture(dataset, provenance))
    return dataset, provenance, captured, score_capture(captured)


def test_fresh_gate_never_reads_replay(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('recorded scores cannot authorize release')
    monkeypatch.setattr(RecordedTriadClient, 'scores', forbidden)
    report = run_gate(tmp_path)
    assert report['passed'], report
    saved = json.loads((tmp_path / 'scores.json').read_text())
    assert saved['provenance']['run_id'] == report['run_id']
    assert saved['provenance']['config_hash'] == report['config_hash']
    assert saved['provenance']['commit_sha'] == report['commit_sha']
    assert saved['provenance']['flag'] is True
    import eval.rag_gate as gate
    expected_sha = subprocess.check_output(
        ['git', '-C', str(gate.ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert report['commit_sha'] == expected_sha


def test_deterministic_scores_depend_on_answer():
    _, _, captured, scores = current_capture()
    assert json_bytes(scores) == json_bytes(score_capture(captured))
    changed = copy.deepcopy(captured)
    changed['rows'][0]['answer'] = 'Refund window policy is 300 days.'
    revised = score_capture(changed)
    assert revised['aggregates']['groundedness'] < scores['aggregates']['groundedness']
    assert revised['capture_hash'] != scores['capture_hash']


def test_unsupported_cited_answer_blocks_release(tmp_path):
    report = run_gate(tmp_path, client=UnsupportedRefundClient())
    assert not report['passed']
    assert report['error'] == 'metric_regression'
    assert report['metrics']['answer_relevance']['passed']
    assert not report['metrics']['groundedness']['passed']
    assert not report['metrics']['citation_support']['passed']
    print('Anchor blocks the degraded advanced-RAG build for the same golden tickets; '
          'replay scores cannot approve it.')


def test_degraded_retriever_blocks_release(tmp_path):
    report = run_gate(tmp_path, degraded=True)
    assert not report['passed']
    assert not report['metrics']['context_relevance']['passed']
    assert (tmp_path / 'report.json').exists()


@pytest.mark.parametrize('fault', ['missing', 'partial', 'duplicate', 'nan',
                                   'boolean', 'stale', 'aggregate', 'capture'])
def test_missing_invalid_stale_or_partial_scores_fail_closed(fault):
    dataset, provenance, captured, scores = current_capture()
    if fault == 'missing':
        del scores['rows'][0]['scores']['groundedness']
    elif fault == 'partial':
        scores['rows'].pop()
    elif fault == 'duplicate':
        scores['rows'][1] = copy.deepcopy(scores['rows'][0])
    elif fault == 'nan':
        scores['rows'][0]['scores']['groundedness'] = float('nan')
    elif fault == 'boolean':
        scores['rows'][0]['scores']['groundedness'] = True
    elif fault == 'stale':
        scores['provenance']['run_id'] = 'yesterday'
    elif fault == 'aggregate':
        scores['aggregates']['groundedness'] = 0.99
    else:
        captured['rows'][0]['answer'] = 'An unsupported changed answer.'
    with pytest.raises((ValueError, KeyError, TypeError)):
        enforce(scores, captured, dataset, provenance)


def test_missing_score_file_is_not_success(tmp_path, monkeypatch):
    import eval.rag_gate as gate
    original = gate.write_json
    def lose_scores(path, value):
        if path.name != 'scores.json':
            original(path, value)
    monkeypatch.setattr(gate, 'write_json', lose_scores)
    report = gate.run_gate(tmp_path)
    assert not report['passed']
    assert report['error'].startswith('FileNotFoundError:')


def test_flag_off_cannot_authorize_advanced_release(tmp_path):
    dataset = load_dataset()
    dataset['configuration']['advanced_rag'] = False
    report = run_gate(tmp_path, dataset=dataset)
    assert not report['passed']
    assert 'advanced_flag_required' in report['error']
