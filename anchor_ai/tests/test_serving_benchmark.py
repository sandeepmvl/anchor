import asyncio
import json
import socket

import pytest

from anchor_ai import observability as obs
from anchor_ai import provider
from anchor_ai import serving_benchmark as bench


@pytest.fixture(autouse=True)
def no_external_connections(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('offline execution attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def test_targets_share_contract_workload_and_mode(monkeypatch):
    seen = []
    async def fake(message, *, mode):
        seen.append((str(message), mode, message.context_max_tokens))
        await asyncio.sleep(.001)
        yield provider.Delta(text='Refund ')
        yield provider.Delta(text='policy.')
        yield provider.Delta(stop='stop')
    monkeypatch.setattr(bench, 'serving_benchmark_stream', fake)
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'live')
    records = []
    for target in ('local', 'hosted'):
        for concurrency in (1, 8):
            result = asyncio.run(bench.run(target, concurrency, samples=8, sink=records.append))
            assert result['completed'] == 8 and result['failures'] == 0
            assert result['execution_mode'] == 'offline'
            assert result['latency_p99_ms'] >= result['latency_p50_ms'] > 0
            assert result['ttft_p99_ms'] >= result['ttft_p50_ms'] > 0
            assert result['completed_req_s'] > 0
            assert result['output_tokens_s'] is None
            assert result['memory']['fits_device'] is None
            assert result['memory']['basis'] == 'simulated_not_hardware_evidence'
            assert bench.WORKLOAD not in json.dumps(result)
    assert len(records) == 4
    assert set(seen) == {(bench.WORKLOAD, 'offline', 32)}


def test_actual_existing_replay_wrapper_is_offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('live provider constructed')
    monkeypatch.setattr(provider, 'LiveProvider', forbidden)
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'live')
    async def collect():
        return [d async for d in provider.serving_benchmark_stream(bench.WORKLOAD)]
    events = asyncio.run(collect())
    assert ''.join(d.text for d in events) == 'Refunds take five days.'
    assert events[-1].stop == 'stop'
    with pytest.raises(ValueError):
        provider.serving_benchmark_stream(bench.WORKLOAD, mode='automatic')


def test_first_text_not_metadata_and_incomplete_is_failure(monkeypatch):
    async def events():
        yield {'metadata': 'headers_are_not_tokens'}
        await asyncio.sleep(.01)
        yield {'text': ''}
        yield {'text': 'first'}
        await asyncio.sleep(.01)
        yield {'done': True}
    row = asyncio.run(bench.measure(events()))
    assert row['ttft_ms'] >= 5
    assert row['latency_ms'] > row['ttft_ms']
    async def incomplete(mode):
        yield {'text': 'partial'}
    monkeypatch.setattr(bench, 'hosted_events', incomplete)
    row = asyncio.run(bench.run('hosted', 8, samples=2, sink=lambda r: None))
    assert row['completed'] == 0 and row['failures'] == 2
    assert row['completed_req_s'] == 0 and row['latency_p99_ms'] is None


def test_single_request_execution():
    active = 0
    peak = 0
    async def generate(message):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(.001)
            yield provider.Delta(text='answer')
            yield provider.Delta(stop='stop')
        finally:
            active -= 1
    async def exercise():
        server = bench.LocalServing(generate)
        async def consume():
            return [e async for e in server.events(bench.WORKLOAD)]
        return await asyncio.gather(*(consume() for _ in range(8)))
    rows = asyncio.run(exercise())
    assert peak == 1
    assert all(r[-1] == {'done': True} for r in rows)
    assert max(r[1]['queue_ms'] for r in rows) > 0


def test_trace_inheritance_and_content_rejection(monkeypatch):
    records = []
    monkeypatch.setattr(obs, 'SINK', records.append)
    safe = dict(target='local', execution_mode='offline', concurrency=1,
                completed=1, failures=0, completed_req_s=2,
                latency_p50_ms=500, latency_p99_ms=500,
                ttft_p50_ms=100, ttft_p99_ms=100)
    with obs.span('test.parent') as parent:
        obs.emit_serving_benchmark(safe)
        with pytest.raises(ValueError):
            obs.emit_serving_benchmark(dict(safe, prompt='private ticket'))
    child = records[0]
    assert child['trace_id'] == parent.record['trace_id']
    assert child['parent_span_id'] == parent.record['span_id']
    assert 'private ticket' not in json.dumps(records)


def test_configuration_is_fail_closed(monkeypatch):
    original = bench.yaml.safe_load
    def unsafe(text):
        c = original(text)
        c['host'] = '0.0.0.0'
        return c
    monkeypatch.setattr(bench.yaml, 'safe_load', unsafe)
    with pytest.raises(ValueError, match='unsafe_serving_configuration'):
        bench.configuration()
