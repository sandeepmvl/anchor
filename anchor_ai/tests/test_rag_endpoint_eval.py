import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from anchor_ai.app import app
from eval.eval_endpoint import ReleaseRuntime, evaluate, main, request_for
from eval.rag_gate import git, load_dataset


def test_endpoint_preserves_complete_capture_and_scores(tmp_path, monkeypatch):
    monkeypatch.setenv('ANCHOR_LLM_PROVIDER', 'offline')
    report = evaluate(tmp_path)
    assert report['passed']
    assert report['local_hash'] == report['endpoint_hash']
    assert report['provenance']['commit_sha'] == git('rev-parse', 'HEAD')
    assert report['provenance']['flag'] is True
    local = json.loads((tmp_path / 'local-capture.json').read_text())
    endpoint = json.loads((tmp_path / 'endpoint-capture.json').read_text())
    assert endpoint == local
    for left, right in zip(local['rows'], endpoint['rows'], strict=True):
        assert [c['id'] for c in right['evidence']] == [c['id'] for c in left['evidence']]
        assert all(type(c['id']) is int for c in right['evidence'])
    assert json.loads((tmp_path / 'local-scores.json').read_text()) == json.loads(
        (tmp_path / 'endpoint-scores.json').read_text())


def test_endpoint_auth_validation_and_scope(monkeypatch):
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'test-secret')
    monkeypatch.setenv('ANCHOR_LLM_PROVIDER', 'offline')
    dataset = load_dataset()
    body = request_for(dataset['tickets'][0])
    factory = lambda: ReleaseRuntime(dataset)
    with patch.object(app.state, 'rag_release_runtime_factory', factory, create=True):
        with TestClient(app) as client:
            assert client.get('/healthz').json() == {'ready': True}
            assert client.post('/rag', json=body).status_code == 401
            headers = {'X-Anchor-Secret': 'test-secret'}
            assert client.post('/rag', json={**body, 'conversation_id': 0},
                headers=headers).status_code == 422
            assert client.post('/rag', json={**body, 'tenant_id': body['tenant_id'] + 1000},
                headers=headers).status_code == 422
            assert client.post('/rag', json={**body, 'generation_id': 'wrong-generation'},
                headers=headers).status_code == 422
            response = client.post('/rag', json=body, headers=headers)
            assert response.status_code == 200
            assert response.json()['scope']['conversation_id'] == body['conversation_id']
    assert not hasattr(app.state, 'rag_release_runtime')


def test_endpoint_evaluation_failure_has_nonzero_exit(tmp_path):
    with patch('eval.eval_endpoint.evaluate', side_effect=ValueError('boundary changed')):
        assert main(['--testclient', '--output', str(tmp_path)]) == 1


def test_endpoint_mismatch_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv('ANCHOR_LLM_PROVIDER', 'offline')
    from eval.eval_endpoint import collect
    def changed(*args, **kwargs):
        rows = collect(*args, **kwargs)
        rows[0]['citations'] = []
        rows[0]['question'] = 'transport changed this question'
        return rows
    with patch('eval.eval_endpoint.collect', side_effect=changed):
        with pytest.raises(ValueError, match='endpoint_parity_failed'):
            evaluate(tmp_path)
