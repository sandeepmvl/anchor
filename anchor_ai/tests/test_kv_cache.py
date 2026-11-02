import asyncio
import copy

import httpx
import pytest

@pytest.fixture(autouse=True)
def isolated_kv_dependencies(monkeypatch):
    """Keep offline test configuration out of other product tests."""
    import importlib
    import os

    # Import application dependencies inside the fixture, not at collection.
    # Environment writes made by an offline dependency stay test-local.
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    accounting = importlib.import_module('anchor_ai.kv_cache')
    backend = importlib.import_module('anchor_ai.vllm_backend')
    dashboard = importlib.import_module('ops.load.kv_cache_dashboard')
    for module, names in (
        (accounting, ('Admission', 'budget', 'configuration', 'metric_view', 'sample')),
        (backend, ('VllmBackend', 'release_gate', 'validate_deployment')),
        (dashboard, ('simulate',)),
    ):
        for name in names:
            monkeypatch.setitem(globals(), name, getattr(module, name))


def test_estimate_and_rounding():
    c = configuration()
    b = budget(c)
    assert b['bytes_per_token'] == 22528
    assert b['request_bytes'] == 11534336
    assert b['pool_bytes'] <= 700000000
    assert budget(c, 513, 3)['batch_bytes'] == 528 * 22528 * 3
    assert budget(c, 513)['useful_bytes'] == 513 * 22528
    c['block_size'] = 32
    assert budget(c, 513)['request_bytes'] == 544 * 22528


@pytest.mark.parametrize('heads,tp,local', [(32, 1, 32), (4, 2, 2), (1, 1, 1), (1, 4, 1), (4, 8, 1)])
def test_gqa_mqa_and_replication(heads, tp, local):
    c = configuration()
    c['model']['num_key_value_heads'] = heads
    c['tensor_parallel_size'] = tp
    assert budget(c)['bytes_per_token'] == 2 * 22 * local * 64 * 2


@pytest.mark.parametrize('field,value', [('layers', 44), ('head_dimension', 128), ('dtype', 'float32')])
def test_dimensions_and_dtype(field, value):
    c = configuration()
    c['model'][field] = value
    assert budget(c)['bytes_per_token'] == 45056


def test_invalid_candidates():
    for key, value in [('block_size', 7), ('gpu_memory_utilization', float('nan')),
                       ('max_num_seqs', True), ('tensor_parallel_size', 3)]:
        c = configuration()
        c[key] = value
        with pytest.raises(ValueError):
            budget(c)


def test_missing_stale_future_and_pressure_are_bounded():
    a = Admission(configuration(), clock=lambda: 100)
    for raw in (None, {}, sample(90), sample(101), sample(100, .99)):
        a.update(raw)
        for _ in range(100):
            assert a.acquire()['decision'] != 'admitted'
        assert a.active == 0
    assert metric_view({}, 100, 5)['cache_utilization'] is None
    a.update(sample(100))
    for _ in range(4):
        assert a.acquire()['decision'] == 'admitted'
    assert a.acquire()['decision'] == 'capacity'
    for _ in range(4):
        a.release()
    assert a.active == 0


def test_pool_limit_independent_of_sequence_limit():
    c = configuration()
    c.update(gpu_memory_utilization=.70, max_num_seqs=48)
    a = Admission(c, clock=lambda: 100)
    a.update(sample(100))
    accepted = sum(a.acquire()['decision'] == 'admitted' for _ in range(100))
    assert accepted == budget(c)['sequence_capacity'] < 48


def test_counter_intervals_and_reset():
    view = metric_view(sample(101, .99, 20, 40, 110), 102, 5,
                       sample(100, .93, 8, 12, 100))
    assert view['preemptions_delta'] == 28
    assert view['tokens_per_second'] == 10
    assert metric_view(sample(102), 102, 5, sample(101, tokens=100))['tokens_per_second'] is None


def test_surge_recovers_and_rejects_oversize():
    unsafe, bounded = simulate('unsafe'), simulate('bounded')
    assert unsafe['preemptions'] > 0
    assert bounded['preemptions'] == 0
    assert bounded['completed'] == 4 and bounded['rejected'] == 44
    assert bounded['output_tokens_s'] > unsafe['output_tokens_s']
    c = configuration()
    c['max_model_len'] = 256
    assert simulate(config=c)['rejected'] == 48


def test_adapter_cancellation_and_rejection():
    async def scenario():
        b = VllmBackend()
        c = configuration()
        c['max_num_seqs'] = 1
        b.kv_admission = Admission(c)
        b.server.block = True
        request = httpx.Request('POST', 'http://localhost/v1/chat/completions')
        body = {'messages': [{'role': 'user', 'content': 'fixture'}]}
        task = asyncio.create_task(b.send(request, body))
        while not b.server.calls:
            await asyncio.sleep(0)
        for _ in range(20):
            assert (await b.send(request, body)).status_code == 503
        assert b.server.calls == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert b.kv_admission.active == 0 and b.server.cancelled
        b.mode = 'live'
        b.kv_admission.update(None)
        assert (await b.send(request, body)).status_code == 503
    asyncio.run(scenario())


def test_timeout_cleanup_and_release_gate():
    async def scenario():
        b = VllmBackend(timeout=.01)
        b.server.block = True
        with pytest.raises(httpx.ReadTimeout):
            await b.send(httpx.Request('POST', 'http://localhost/v1/chat/completions'), {})
        assert b.kv_admission.active == 0 and b.server.calls == 1
    asyncio.run(scenario())
    assert validate_deployment()['release']['real_evidence_required'] is True
    with pytest.raises(ValueError, match='release_evidence_missing'):
        release_gate({'offline_contract_passed': True})
    with pytest.raises(ValueError, match='release_evidence_missing'):
        VllmBackend(production=True)
