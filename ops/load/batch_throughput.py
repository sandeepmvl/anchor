import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from fastapi.responses import StreamingResponse
from anchor_ai.batching import (BatchEnvelope, BatchItem, BatchRuntime,
                               OfflineBatchProvider, Overloaded, Scheduler,
                               configuration)
from anchor_ai.provider import offline_batch_provider
from anchor_ai.observability import batch_measurement, distribution


def stream_payload(message):
    """Build and validate a fixture using the existing HTTP contract's JSON types.

    In particular, identity fields must not be assumed to be integers merely
    because the separate bulk endpoint uses integer tenant identifiers.
    """
    from anchor_ai.contracts import InferRequest
    schema = InferRequest.model_json_schema()
    proposed = {'tenant_id': 1, 'conversation_id': 7, 'message': message,
                'request_id': '11111111-1111-4111-8111-111111111111',
                'trace_id': '11111111111111111111111111111111'}
    payload = {}
    for name, field in schema['properties'].items():
        if name not in proposed and name not in schema.get('required', []):
            continue
        if '$ref' in field:
            field = schema['$defs'][field['$ref'].split('/')[-1]]
        if 'anyOf' in field:
            field = next(part for part in field['anyOf'] if part.get('type') != 'null')
        value = proposed.get(name, 'fixture')
        if 'const' in field:
            value = field['const']
        elif 'enum' in field:
            value = field['enum'][0]
        elif field.get('type') == 'string':
            value = str(value)
        elif field.get('type') == 'integer':
            value = value if type(value) is int else max(1, field.get('minimum', 1))
        elif field.get('type') == 'boolean':
            value = False
        payload[name] = value
    # Validation is not patched or bypassed. An invalid fixture fails here,
    # before an HTTP 422 could be mistaken for a batching regression.
    return InferRequest.model_validate(payload).model_dump(mode='json', exclude_none=True)


def item(n, tenant=1):
    return BatchItem(request_id=f'doc-{n}', tenant_id=tenant,
                     operation='embedding', model='offline-hash', revision='v1',
                     settings={'input_type': 'document'}, text=f'policy document {n}')


def authorize(token, items):
    return token == 'offline-scope' and all(
        p.tenant_id in {1, 2} and p.conversation_id == 0 for p in items)


async def measure(grouped):
    c = configuration()
    if not grouped:
        c['size'] = 1
    provider, events = offline_batch_provider(), []
    scheduler = Scheduler(provider, c, emit=lambda record: events.append(batch_measurement(record)))
    callers = asyncio.Semaphore(32)
    latencies = []
    async def send(n):
        async with callers:
            submitted = time.perf_counter()
            p = item(n)
            result = await scheduler.submit(p)
            assert result['embedding'] == list(hashlib.sha256(p.text.encode()).digest()[:8])
            latencies.append((time.perf_counter() - submitted) * 1000)
    start = time.perf_counter()
    try:
        outcomes = await asyncio.gather(*(send(n) for n in range(64)), return_exceptions=True)
        failures = sum(isinstance(value, BaseException) for value in outcomes)
        elapsed = time.perf_counter() - start
        waits = sorted(v for event in events for v in event['queue_ms'])
        return {'mode': 'microbatched' if grouped else 'unbatched',
                'inputs': 64, 'caller_concurrency': 32, 'provider_concurrency': 2,
                'provider_calls': provider.calls, 'seconds': elapsed,
                'inputs_per_second': (64 - failures) / elapsed,
                'completed': 64 - failures, 'failures': failures,
                'successful_latency_ms': distribution(latencies),
                'p95_queue_ms': waits[math.ceil(.95 * len(waits)) - 1] if waits else None,
                'input_test_units': 64 * 3,
                'evidence': 'synthetic_12ms_per_call_not_provider_benchmark'}
    finally:
        await scheduler.close()


async def product(mode):
    from anchor_ai.app import app
    import anchor_ai.streaming
    os.environ['ANCHOR_AI_SECRET'] = 'offline-service'
    c = configuration()
    if mode == 'overload':
        c.update(max_pending=2, size=2, window_ms=1000)
    provider = offline_batch_provider()
    scheduler = Scheduler(provider, c)
    with tempfile.TemporaryDirectory() as directory:
        runtime = BatchRuntime(scheduler, verify=authorize,
                               path=Path(directory) / 'jobs.sqlite3')
        app.state.batch_runtime = runtime
        headers = {'X-Anchor-Secret': 'offline-service', 'X-Anchor-Context': 'offline-scope'}
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url='http://anchor') as client:
                if mode == 'mixed':
                    response = await client.post('/v1/internal/batch', headers=headers,
                        json={'items': [item(n, 1 + n % 2).model_dump() for n in range(16)]})
                    response.raise_for_status()
                    outcomes = response.json()['items']
                    assert all(p['tenant_id'] == 1 + n % 2 for n, p in enumerate(outcomes))
                    assert sorted((key[0], size) for key, size in provider.groups) == [(1, 8), (2, 8)]
                    return {'tenant_batches': [[key[0], size] for key, size in provider.groups],
                            'identity_mapping': 'passed'}
                if mode == 'async':
                    response = await client.post('/v1/internal/batch', headers=headers,
                        json={'items': [item(0).model_dump()], 'asynchronous': True})
                    response.raise_for_status()
                    result = response.json()
                    job = result['jobs'][0]
                    assert runtime.polling_identifier('offline-scope', item(0), job['job_id']) == job['provider_job_id']
                    return dict(result, persistence='reopened_SQLite_lookup_passed',
                                inference_calls=provider.calls)
                scheduler.config['size'] = 8
                waiting = [asyncio.create_task(scheduler.submit(item(n))) for n in (0, 1)]
                await asyncio.sleep(.005)
                response = await client.post('/v1/internal/batch', headers=headers,
                    json={'items': [item(2).model_dump()]})
                assert response.json()['items'][0]['error'] == 'batch_overloaded'
                def stream(body, request):
                    async def events():
                        yield 'data: offline ticket reply\n\n'
                    return StreamingResponse(events(), media_type='text/event-stream')
                with patch.object(anchor_ai.streaming, 'response_for', stream):
                    response = await client.post('/v1/infer/stream', headers=headers,
                                                 json=stream_payload('Interactive ticket'))
                assert response.status_code == 200 and 'ticket reply' in response.text
                assert all(not task.done() for task in waiting)
                for task in waiting:
                    task.cancel()
                await asyncio.gather(*waiting, return_exceptions=True)
                return {'backpressure': 'batch_overloaded',
                        'interactive_stream': 'bypassed_batch_wait',
                        'waiting_bulk_items_during_stream': 2}
        finally:
            await runtime.close()
            del app.state.batch_runtime


async def spike_batch_report():
    """Use the existing scheduler and HTTP batch endpoint with offline inference."""
    before, after = await measure(False), await measure(True)
    endpoint = await product('mixed')
    return dict(unbatched=before, batched=after, endpoint=endpoint,
                not_verified=['GPU throughput', 'production signed context',
                              'real tokenizer limits', 'customer network latency'])


async def main(mode):
    if mode in {'baseline', 'micro'}:
        result = await measure(mode == 'micro')
    elif mode == 'demo':
        before, after = await measure(False), await measure(True)
        assert before['provider_calls'] == 64
        assert after['provider_calls'] < before['provider_calls']
        result = {'before': before, 'after': after, 'ticket': await product('overload')}
    else:
        result = await product(mode)
    result['not_verified'] = ['real_provider_throughput', 'real_tokenizer_limits',
                              'production_signed_context', 'customer_network_TTFT']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'micro', 'mixed', 'async', 'overload', 'demo'], default='demo')
    asyncio.run(main(parser.parse_args().mode))
