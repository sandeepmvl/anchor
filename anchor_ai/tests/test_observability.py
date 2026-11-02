import json

import httpx
import pytest
from fastapi.testclient import TestClient

from anchor_ai.app import app
from anchor_ai import observability as obs
from anchor_ai.llm_client import LlmClient

TRACE = '1234567890abcdef1234567890abcdef'
PARENT = '1234567890abcdef'
MESSAGE = 'PRIVATE_MESSAGE_SENTINEL'
SECRET = 'SERVICE_SECRET_SENTINEL'
KEY = 'PROVIDER_KEY_SENTINEL'


@pytest.fixture
def captured(monkeypatch):
    records = []
    monkeypatch.setattr(obs, 'SINK', lambda record: records.append(json.loads(json.dumps(record))))
    monkeypatch.setenv('ANCHOR_AI_SECRET', SECRET)
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.setenv('ANCHOR_AI_PROVIDER', 'offline')
    monkeypatch.setenv('ANCHOR_LLM_FIXTURE', '200')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-4o-mini')
    monkeypatch.setenv('OPENAI_API_KEY', KEY)
    return records


def test_replay_requests_are_joinable_and_content_free(captured):
    with TestClient(app) as client:
        for number in range(3):
            response = client.post('/v1/infer', headers={
                'X-Anchor-Secret': SECRET, 'X-Request-Id': f'ticket-{number}',
                'traceparent': f'00-{TRACE}-{PARENT}-01'}, json={
                    'version': 'v1', 'tenant_id': 1, 'conversation_id': 1, 'message': MESSAGE})
            assert response.status_code == 200
    assert len(captured) == 6
    for number in range(3):
        records = [r for r in captured if r['request_id'] == f'ticket-{number}']
        infer = next(r for r in records if r['name'] == 'python.infer')
        llm = next(r for r in records if r['name'] == 'llm.call')
        assert infer['trace_id'] == llm['trace_id'] == TRACE
        assert infer['parent_span_id'] == PARENT
        assert llm['parent_span_id'] == infer['span_id']
        assert llm['tokens']['input'] == 24
        assert llm['tokens']['output'] == 14
        assert llm['cost_basis'] == 'offline_replay_estimate'
        assert llm['latency_ms'] >= 0 and llm['status'] == 'ok'
        # A fixture without a returned model is deliberately not priced.
        assert llm['cost_estimate'] == (pytest.approx(0.000012) if llm['response_model'] == 'gpt-4o-mini' else None)
        print(json.dumps(llm, sort_keys=True))
    text = json.dumps(captured)
    for sentinel in (MESSAGE, SECRET, KEY, 'Authorization', 'Bearer'):
        assert sentinel not in text


def test_rejected_provider_has_null_usage(captured):
    class RejectedProvider:
        def key(self):
            return KEY

        def client(self):
            return httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
                httpx.Response(400, headers={'x-request-id': 'req_rejected'}, json={
                    'error': {'message': MESSAGE + KEY, 'type': 'invalid_request_error',
                              'code': 'invalid_request_error'}})))

    import asyncio
    from anchor_ai.provider import ProviderError
    with obs.span('python.infer', obs.context({'x-request-id': 'rejected',
                  'traceparent': f'00-{TRACE}-{PARENT}-01'})):
        with pytest.raises(ProviderError):
            asyncio.run(LlmClient(provider=RejectedProvider()).complete_result(MESSAGE))
    llm = next(r for r in captured if r['name'] == 'llm.call')
    assert llm['status'] == 'error'
    assert llm['tokens'] is None and llm['cost_estimate'] is None
    assert llm['provider_request_id'] == 'req_rejected'
    assert llm['trace_id'] == TRACE and llm['status_code'] == 400
    assert llm['parent_span_id'] == captured[-1]['span_id']
    for sentinel in (MESSAGE, SECRET, KEY, 'Authorization', 'Bearer'):
        assert sentinel not in json.dumps(captured)


def test_category_mapping_and_cost():
    tokens = obs.tokens_from_usage({'prompt_tokens': 1200, 'completion_tokens': 300,
        'prompt_tokens_details': {'cached_tokens': 800},
        'completion_tokens_details': {'reasoning_tokens': 100}})
    assert tokens == dict(input=400, cached_input=800, cache_write_input=0, output=300, reasoning=100)
    prices = {'example': dict(input='2', cached_input='0.5', cache_write_input='2.5', output='8')}
    assert obs.estimate('example', tokens, prices) == pytest.approx(0.0036)
    assert obs.estimate('unknown', tokens) is None
    assert obs.context({'traceparent': '00-' + '0' * 32 + '-' + PARENT + '-01'})['parent_span_id'] is None


def test_reasoning_logged_without_double_charge(captured, monkeypatch):
    import asyncio
    from anchor_ai.provider import ReplayProvider, REASONING_RESPONSES
    from anchor_ai.reasoning import REPLAY_MODEL
    monkeypatch.setenv('OPENAI_MODEL', REPLAY_MODEL)
    ticket = json.loads(REASONING_RESPONSES.read_text())['tickets'][0]
    provider = ReplayProvider(task='reasoning', fixture_id='B1', effort='high')
    asyncio.run(LlmClient(provider).complete_result(ticket['ticket'], max_tokens=4000, effort='high'))
    record = captured[-1]
    assert record['tokens']['reasoning'] == 1200
    assert record['tokens']['output'] == 1224
    assert record['cost_estimate'] == pytest.approx(0.01244)
    assert record['price_version'] == 'fixture-illustrative-v1'
    assert record['cost_basis'] == 'illustrative_fixture_estimate'
    assert record['fixture_latency_ms'] == 3300
    assert record['latency_ms'] >= 0
    assert ticket['ticket'] not in json.dumps(captured)


def test_unknown_reasoning_is_not_zero():
    tokens = obs.tokens_from_usage({'prompt_tokens': 10, 'completion_tokens': 20})
    assert tokens['reasoning'] is None
    assert obs.estimate('gpt-4o-mini', tokens) is not None
    assert obs.tokens_from_usage(None) is None
    assert obs.estimate('gpt-4o-mini', None) is None


def test_missing_usage_stays_unknown_in_error_log(captured):
    import asyncio
    from anchor_ai.provider import ProviderError

    class MissingUsageProvider:
        def key(self):
            return KEY

        def client(self):
            body = {'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}]}
            return httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)))

    with pytest.raises(ProviderError, match='invalid_provider_response'):
        asyncio.run(LlmClient(MissingUsageProvider()).complete_result(MESSAGE))
    assert captured[-1]['tokens'] is None
    assert captured[-1]['cost_estimate'] is None
    assert captured[-1]['error_type'] == 'invalid_provider_response'


@pytest.mark.parametrize('delegated', [False, True])
def test_trajectory_projection_preserves_scope_and_span_tree(captured, delegated):
    from eval.trajectory_eval import replay, normalize
    trace = replay(delegated=delegated)
    steps = normalize(trace)
    by_id = {r['span_id']: r for r in captured}
    for step in steps:
        child = by_id[step['span_id']]
        parent = by_id[step['parent_span_id']]
        assert child['tenant_id'] == parent['tenant_id'] == step['tenant_id'] == 1
        assert child['trace_id'] == parent['trace_id'] == step['trace_id']
        assert parent['name'] == ('python.fixture.worker' if delegated else 'python.fixture.bounded')
    text = json.dumps(trace) + json.dumps(captured)
    for forbidden in ('A17', 'amount_cents', MESSAGE, SECRET, KEY, 'chain_of_thought'):
        assert forbidden not in text


def test_production_metrics_have_bounded_labels(captured):
    exported = []
    previous = obs.EXPORT_SINK
    obs.EXPORT_SINK = exported.append
    try:
        with obs.span('llm.call', stage='llm', tenant_id=1, conversation_id=7) as item:
            item.set(tokens={'input': 24, 'output': 14}, cost_estimate=.000012)
        projected = exported[0]
        assert projected['attributes']['tenant_id'] == 1
        assert projected['attributes']['conversation_id'] == 7
        points = obs.metric_points(projected)
        assert points
        for _, _, labels in points:
            assert set(labels) <= {'stage', 'status', 'direction'}
            assert 'tenant_id' not in labels and 'conversation_id' not in labels
        traces, metrics = obs.otlp_payloads(projected)
        assert 'tenant_id' in json.dumps(traces)
        assert 'tenant_id' not in json.dumps(metrics)
        assert 'conversation_id' not in json.dumps(metrics)
    finally:
        obs.EXPORT_SINK = previous


def test_observability_release_refuses_offline_evidence():
    for evidence in (None, {}, {'execution_mode': 'offline',
            'rails_http_trace': True, 'collector_delivery': True,
            'dashboard_queries': True, 'pii_export_scan': True}):
        with pytest.raises(ValueError, match='observability_release_evidence_missing'):
            obs.require_observability_evidence(evidence)


def test_decision_projection_rejects_unrestricted_text(captured):
    from eval.trajectory_eval import fingerprint
    with obs.span('root', tenant_id=1):
        with obs.span('decision'):
            with pytest.raises(ValueError, match='unsafe_decision_evidence'):
                obs.trajectory_decision(kind='tool', tool='lookup_order',
                    expected_tool='lookup_order', argument_fingerprint=fingerprint({}),
                    expected_argument_fingerprint=fingerprint({}), evidence=[MESSAGE],
                    claim='evidence_available')
    assert MESSAGE not in json.dumps(captured)
