import asyncio
import json

import pytest
from anchor_ai.observability import (
    ACTIVE, PARENT, Profile, distribution, configuration, LatencyBoundary)
from ops.load.profile_tickets import profile_ticket, boundary_probe


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_nested_parallel_inclusive_and_exclusive_without_double_counting():
    clock = Clock()
    profile = Profile(clock=clock)
    with profile.activate():
        with profile.span('request'):
            with profile.span('retrieval'):
                with profile.span('branch'):
                    with profile.span('embedding'):
                        clock.value = .020
                    with profile.span('search'):
                        clock.value = .060
                    with profile.span('rerank'):
                        clock.value = .080
                # Simulate a second concurrently observed branch on this clock.
                clock.value = .010
                with profile.span('branch'):
                    clock.value = .075
                clock.value = .080
            with profile.span('model'):
                clock.value = .180
            with profile.span('tool'):
                clock.value = .195
            clock.value = .200
    report = profile.report()
    rows = report['spans']
    retrieval = next(r for r in rows if r['name'] == 'retrieval')
    assert retrieval['inclusive_ms'] == pytest.approx(80)
    assert retrieval['exclusive_ms'] == pytest.approx(0)
    assert report['total_ms'] == pytest.approx(200)
    assert report['application_remainder_ms'] == pytest.approx(5)
    parents = [r for r in rows if r['parent'] == 0]
    assert sum(r['inclusive_ms'] for r in parents) == pytest.approx(195)
    assert sum(r['exclusive_ms'] for r in rows) != pytest.approx(200)
    assert sum(r['inclusive_ms'] for r in rows if r['name'] == 'branch') > 80


def test_error_restores_context_and_closes_span():
    profile = Profile()
    with pytest.raises(RuntimeError):
        with profile.activate():
            with profile.span('request'):
                raise RuntimeError('private ticket content')
    assert ACTIVE.get() is None
    assert PARENT.get() is None
    report = profile.report()
    assert report['spans'][0]['status'] == 'error'
    assert 'private ticket' not in json.dumps(report)


def test_unknown_names_and_unfinished_spans_are_rejected():
    profile = Profile()
    with pytest.raises(ValueError, match='unsafe_span_name'):
        with profile.span('alice@example.com'):
            pass
    with profile.activate():
        with profile.span('request'):
            with pytest.raises(ValueError, match='unfinished_span'):
                profile.report()


def test_synthetic_stream_and_call_count():
    report = asyncio.run(profile_ticket('rag', 3))
    assert report['round_trips'] == 3
    models = [r for r in report['spans'] if r['name'] == 'model']
    assert len(models) == 3
    for model in models:
        assert model['ttft_ms'] > 0
        assert len(model['text_event_gaps_ms']) == 2
        assert model['inter_token_span_ms'] > 0
        assert model['cached_tokens'] == 0
        assert model['injected_inference_ms'] == 12
    assert all(r['exclusive_ms'] >= 0 for r in report['spans'])


def test_boundary_trace_and_redacted_schema():
    trace = 'a' * 32
    for mode in ('chat', 'rag'):
        report = asyncio.run(boundary_probe('00-' + trace + '-' + 'b' * 16 + '-01', mode))
        assert report['trace_id'] == trace
        assert report['response_completed'] is True
        assert all(row['trace_id'] == trace for row in report['spans'])
        assert not {'message', 'prompt', 'reply', 'email'} & set(report)


def test_nearest_rank_and_configuration_fail_closed(tmp_path):
    assert distribution(list(range(1, 21))) == {'p50': 10, 'p95': 19, 'p99': 20}
    assert configuration()['record_content'] is False
    path = tmp_path / 'unsafe.yml'
    path.write_text('profiling: {enabled: true, record_content: true}')
    with pytest.raises(ValueError, match='unsafe_performance_configuration'):
        configuration(path)


def test_boundary_supplies_correlation_without_incoming_trace():
    from anchor_ai.observability import context
    records, observed = [], []

    async def application(scope, receive, send):
        headers = {k.decode(): v.decode() for k, v in scope['headers']}
        observed.append(context(headers)['trace_id'])
        await send({'type': 'http.response.body', 'body': b''})

    async def noop(*args):
        pass

    asyncio.run(LatencyBoundary(application, enabled=True, sink=records.append)(
        {'type': 'http', 'headers': []}, noop, noop))
    assert observed == [records[0]['trace_id']]


def test_application_registers_profile_boundary():
    from anchor_ai.app import app
    assert any(m.cls is LatencyBoundary for m in app.user_middleware)


def test_runtime_uses_shared_model_wrapper():
    from anchor_ai import runtime, observability
    assert runtime.model_call is observability.model_call


def test_starlette_builds_and_executes_profile_middleware():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    records = []
    application = FastAPI()
    application.add_middleware(LatencyBoundary, enabled=True, sink=records.append)

    @application.get('/latency-smoke')
    async def smoke():
        return {'ok': True}

    with TestClient(application) as client:
        response = client.get('/latency-smoke', headers={
            'traceparent': '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01'})
    assert response.status_code == 200
    assert response.json() == {'ok': True}
    assert len(records) == 1
    assert records[0]['trace_id'] == 'a' * 32
    assert records[0]['response_completed'] is True
    assert records[0]['total_ms'] >= 0


def test_worker_finishing_after_parent_is_clipped_not_rejected():
    clock = Clock()
    profile = Profile(clock=clock)
    with profile.activate():
        with profile.span('request'):
            with profile.span('retrieval'):
                # A copied context models the context inherited by to_thread.
                import contextvars
                worker_context = contextvars.copy_context()
                worker = profile.span('branch')
                worker_context.run(worker.__enter__)
                clock.value = .005
            clock.value = .006
        first = profile.report()
        assert first['incomplete'] is True
        clock.value = .050
        worker_context.run(worker.__exit__, None, None, None)
    second = profile.report()
    assert second['incomplete'] is True
    for report in (first, second):
        assert report['total_ms'] == pytest.approx(6)
        branch = next(r for r in report['spans'] if r['name'] == 'branch')
        assert branch['inclusive_ms'] == pytest.approx(5)
        assert branch['interval_clipped'] is True
        assert all(r['exclusive_ms'] >= 0 for r in report['spans'])


def test_boundary_preserves_timeout_exception():
    records = []

    async def application(scope, receive, send):
        raise TimeoutError('retrieval_deadline')

    async def noop(*args):
        pass

    with pytest.raises(TimeoutError, match='retrieval_deadline'):
        asyncio.run(LatencyBoundary(application, enabled=True, sink=records.append)(
            {'type': 'http', 'headers': []}, noop, noop))
    assert len(records) == 1
    assert records[0]['response_completed'] is False
    assert records[0]['spans'][0]['status'] == 'error'


def test_profile_storage_is_bounded_and_closed_profiles_do_not_grow():
    from anchor_ai.observability import MAX_SPANS, MAX_TEXT_EVENTS
    profile = Profile()
    with profile.activate():
        with profile.span('request'):
            with profile.span('model') as node:
                for _ in range(MAX_TEXT_EVENTS + 2):
                    profile.text(node)
            for _ in range(MAX_SPANS + 2):
                with profile.span('tool'):
                    pass
        count = len(profile.nodes)
        with profile.span('tool'):
            pass
    report = profile.report()
    assert count == len(profile.nodes) == MAX_SPANS
    assert len(node['text_events']) == MAX_TEXT_EVENTS
    assert report['truncated'] is True
    assert report['spans'][1]['text_events_truncated'] is True
    assert len(report['spans']) == MAX_SPANS
