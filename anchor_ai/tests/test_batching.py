import asyncio
import hashlib
import sqlite3
import time
from unittest.mock import patch

import httpx
import pytest
from fastapi.responses import StreamingResponse

from anchor_ai.batching import (BatchEnvelope, BatchItem, BatchRuntime,
                               OfflineBatchProvider, Overloaded, Scheduler,
                               configuration)
from ops.load.batch_throughput import stream_payload
from anchor_ai.provider import offline_batch_provider
from anchor_ai.runtime import create_batch_runtime
from anchor_ai.observability import batch_measurement


def item(n=0, **changes):
    values = dict(request_id=str(n), tenant_id=1, operation='embedding',
                  model='offline-hash', revision='v1',
                  settings={'input_type': 'document'}, text=f'document {n}')
    values.update(changes)
    return BatchItem(**values)


def config(**changes):
    return dict(configuration(), **changes)


def authorize(token, items):
    # Explicit fixture authority. Production injects its signed-context verifier.
    return token == 'fixture-worker' and all(
        p.tenant_id in {1, 2} and p.conversation_id in {0, 7} for p in items)


def runtime(tmp_path, scheduler):
    return BatchRuntime(scheduler, verify=authorize, path=tmp_path / 'jobs.sqlite3')


def test_compatible_partitioning_and_result_identity():
    async def run():
        provider = OfflineBatchProvider()
        scheduler = Scheduler(provider, config(size=2, window_ms=5))
        try:
            inputs = [item(0), item(1), item(2, tenant_id=2),
                      item(3, model='different'),
                      item(4, settings={'input_type': 'query'}),
                      item(5, revision='v2'), item(6, conversation_id=7),
                      item(7, operation='inference', conversation_id=7)]
            results = await asyncio.gather(*(scheduler.submit(p) for p in inputs))
            for p, result in zip(inputs, results):
                if p.operation == 'embedding':
                    assert result['embedding'] == list(hashlib.sha256(p.text.encode()).digest()[:8])
                else:
                    assert result['reply'] == 'Offline reply: ' + p.text
            assert provider.calls == 7
            assert sorted(size for _, size in provider.groups) == [1, 1, 1, 1, 1, 1, 2]
        finally:
            await scheduler.close()
    asyncio.run(run())


def test_deadline_flush_and_aggregate_cap():
    async def run():
        called = asyncio.Event()
        class Provider(OfflineBatchProvider):
            async def call(self, key, records):
                called.set()
                return await super().call(key, records)
        provider = Provider()
        scheduler = Scheduler(provider, config(window_ms=20, unit_cap=3))
        try:
            task = asyncio.create_task(scheduler.submit(item()))
            await asyncio.sleep(.005)
            assert not called.is_set()
            await asyncio.wait_for(called.wait(), .5)
            assert 'embedding' in await task
            called.clear()
            await asyncio.gather(scheduler.submit(item(1)), scheduler.submit(item(2)))
            assert [n for _, n in provider.groups] == [1, 1, 1]
            with pytest.raises(ValueError):
                await scheduler.submit(item(text='one two three four'))
        finally:
            await scheduler.close()
    asyncio.run(run())


def test_overload_counts_inflight_and_cancellation_is_local():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        class Provider(OfflineBatchProvider):
            async def call(self, key, records):
                started.set()
                await release.wait()
                return await super().call(key, records)
        provider = Provider()
        scheduler = Scheduler(provider, config(size=2, max_pending=2))
        try:
            first = asyncio.create_task(scheduler.submit(item(0)))
            second = asyncio.create_task(scheduler.submit(item(1)))
            await asyncio.wait_for(started.wait(), .5)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            with pytest.raises(Overloaded):
                await scheduler.submit(item(2))
            assert scheduler.pending == 2
            release.set()
            assert 'embedding' in await second
            assert provider.calls == 1
            queued = asyncio.create_task(scheduler.submit(item(3)))
            await asyncio.sleep(0)
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
            await asyncio.sleep(.01)
            assert scheduler.pending == 0
        finally:
            release.set()
            await scheduler.close()
    asyncio.run(run())


def test_partial_and_whole_call_failures():
    async def run():
        scheduler = Scheduler(OfflineBatchProvider(), config(size=2))
        try:
            good, bad = await asyncio.gather(
                scheduler.submit(item()),
                scheduler.submit(item(1, text='!partial_failure')))
            assert 'embedding' in good
            assert bad == {'error': 'provider_item_failed'}
        finally:
            await scheduler.close()
        class Broken:
            async def call(self, key, records):
                raise RuntimeError('injected transport failure')
        scheduler = Scheduler(Broken(), config(size=2))
        try:
            results = await asyncio.gather(scheduler.submit(item()), scheduler.submit(item(1)))
            assert results == [{'error': 'provider_call_failed'}] * 2
        finally:
            await scheduler.close()
    asyncio.run(run())


def test_async_identifiers_persist_and_cannot_cross_scope(tmp_path):
    async def run():
        scheduler = Scheduler(OfflineBatchProvider(), config())
        boundary = runtime(tmp_path, scheduler)
        try:
            result = await boundary.execute('fixture-worker', BatchEnvelope(
                items=[item()], asynchronous=True))
            assert result['status'] == 'submitted_not_completed'
            job = result['jobs'][0]
            assert job['provider_job_id'].startswith('fake_batch_')
            assert scheduler.adapter.calls == 0
            # Reopen the runtime/database, as a later polling worker would.
            reopened = runtime(tmp_path, scheduler)
            assert reopened.polling_identifier('fixture-worker', item(), job['job_id']) == job['provider_job_id']
            with pytest.raises(PermissionError):
                reopened.polling_identifier('fixture-worker', item(tenant_id=2), job['job_id'])
            with pytest.raises(PermissionError):
                await boundary.execute('untrusted', BatchEnvelope(items=[item()]))
        finally:
            await boundary.close()
    asyncio.run(run())


def test_production_gate_refuses_missing_real_evidence(tmp_path):
    scheduler = Scheduler(OfflineBatchProvider(), config())
    with pytest.raises(ValueError, match='batch_release_evidence_missing'):
        BatchRuntime(scheduler, verify=authorize, path=tmp_path / 'jobs.sqlite3',
                     production=True, tokenizer=lambda text, key: 1,
                     evidence={'tenant_authorization': True, 'provider_limits': True,
                               'load_test': True, 'interactive_capacity': True,
                               'real_tokenizer': True})
    class LiveShape:
        fixture = False
    scheduler = Scheduler(LiveShape(), config())
    with pytest.raises(ValueError, match='batch_release_evidence_missing'):
        BatchRuntime(scheduler, verify=authorize, path=tmp_path / 'jobs.sqlite3', production=True)


def test_endpoint_auth_mixed_tenants_and_stream_bypass(tmp_path, monkeypatch):
    async def run():
        from anchor_ai.app import app
        import anchor_ai.streaming
        monkeypatch.setenv('ANCHOR_AI_SECRET', 'fixture-service-secret')
        provider = OfflineBatchProvider()
        scheduler = Scheduler(provider, config(size=2, max_pending=2, window_ms=1000))
        boundary = runtime(tmp_path, scheduler)
        monkeypatch.setattr(app.state, 'batch_runtime', boundary, raising=False)
        headers = {'X-Anchor-Secret': 'fixture-service-secret',
                   'X-Anchor-Context': 'fixture-worker'}
        stream_calls = []
        def immediate(body, request):
            stream_calls.append(body.message)
            async def events():
                yield 'data: ticket reply\n\n'
            return StreamingResponse(events(), media_type='text/event-stream')
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url='http://anchor') as client:
                payload = {'items': [item().model_dump()]}
                assert (await client.post('/v1/internal/batch', json=payload)).status_code == 401
                rejected = await client.post('/v1/internal/batch', json=payload,
                                             headers={**headers, 'X-Anchor-Context': 'wrong'})
                assert rejected.status_code == 403
                mixed = await client.post('/v1/internal/batch', headers=headers,
                    json={'items': [item().model_dump(), item(1, tenant_id=2).model_dump()]})
                assert mixed.status_code == 200
                assert [p['tenant_id'] for p in mixed.json()['items']] == [1, 2]
                assert [key[0] for key, _ in provider.groups] == [1, 2]
                # Fill admission while all work waits for its collection deadline.
                scheduler.config['size'] = 8
                waiting = [asyncio.create_task(scheduler.submit(item(i))) for i in (4, 5)]
                await asyncio.sleep(.005)
                overloaded = await client.post('/v1/internal/batch', json=payload, headers=headers)
                assert overloaded.json()['items'][0]['error'] == 'batch_overloaded'
                with patch.object(anchor_ai.streaming, 'response_for', immediate):
                    response = await client.post('/v1/infer/stream', headers=headers,
                                                 json=stream_payload('Help with my ticket'))
                assert response.status_code == 200
                assert 'ticket reply' in response.text
                assert stream_calls == ['Help with my ticket']
                assert all(not task.done() for task in waiting)
                assert scheduler.pending == 2
                for task in waiting:
                    task.cancel()
                await asyncio.gather(*waiting, return_exceptions=True)
        finally:
            await boundary.close()
    asyncio.run(run())


def test_batch_composition_and_safe_measurements(tmp_path):
    async def run():
        provider = offline_batch_provider()
        events = []
        boundary = create_batch_runtime(
            adapter=provider, verify=authorize, path=tmp_path / 'composed.sqlite3',
            emit=lambda record: events.append(batch_measurement(record)))
        try:
            result = await boundary.execute('fixture-worker', BatchEnvelope(items=[item()]))
            assert 'embedding' in result['items'][0]
            assert provider.calls == 1
            assert events[0]['batch_size'] == 1
            assert events[0]['outstanding'] == 1
            assert events[0]['queue_ms'][0] >= 0
        finally:
            await boundary.close()
    asyncio.run(run())
    with pytest.raises(ValueError, match='unsafe_batch_telemetry'):
        batch_measurement({'batch_size': 1, 'queue_ms': [1], 'outstanding': 1,
                           'text': 'must not be retained'})
    with pytest.raises(ValueError, match='unsafe_batch_telemetry'):
        batch_measurement({'batch_size': 1, 'queue_ms': [float('nan')], 'outstanding': 1})
    with pytest.raises(ValueError, match='batch_release_evidence_missing'):
        create_batch_runtime(adapter=offline_batch_provider(), verify=authorize,
                             path=tmp_path / 'production.sqlite3', production=True)


def test_stream_fixture_obeys_existing_request_contract():
    from anchor_ai.contracts import InferRequest
    payload = stream_payload('Interactive ticket')
    request = InferRequest.model_validate(payload)
    assert request.message == 'Interactive ticket'
