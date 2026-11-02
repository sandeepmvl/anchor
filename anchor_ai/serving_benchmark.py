import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

import httpx
import yaml

from anchor_ai.provider import serving_benchmark_stream
from anchor_ai.observability import emit_serving_benchmark

ROOT = Path(__file__).resolve().parents[1]
# A synthetic redacted ticket, shared byte-for-byte by both targets.
WORKLOAD = 'Customer [REDACTED]: What is the refund window?'


class RequestText(str):
    context_max_tokens = 32
    context_temperature = 0


def configuration():
    c = yaml.safe_load((ROOT / 'config/self_hosted.yml').read_text())
    if (c.get('version') != 1 or c.get('record_content') is not False
            or c.get('host') != '127.0.0.1' or c.get('port') != 8765
            or c.get('model_alias') != 'tiny-llama'
            or c.get('samples') != 8 or c.get('max_new_tokens') != 32
            or c.get('batching') != 'single_request'):
        raise ValueError('unsafe_serving_configuration')
    a = c['assumptions']
    if a.get('basis') != 'simulated_not_hardware_evidence':
        raise ValueError('unsafe_memory_basis')
    for key in ('parameters', 'bytes_per_weight', 'layers', 'kv_heads',
                'head_dimension', 'bytes_per_kv_element', 'context_tokens',
                'memory_budget_bytes'):
        if type(a.get(key)) is not int or a[key] <= 0:
            raise ValueError('invalid_memory_assumption')
    return c


def assumptions(c):
    a = c['assumptions']
    weights = a['parameters'] * a['bytes_per_weight']
    kv = (2 * a['layers'] * a['kv_heads'] * a['head_dimension']
          * a['bytes_per_kv_element'] * a['context_tokens'])
    return dict(a, weights_bytes=weights, kv_bytes_per_active_request=kv,
                active_requests=1, runtime_overhead_bytes=None,
                fits_device=None, runtime='transformers_generate_assumption',
                hardware='unspecified_not_measured', batching=c['batching'],
                warning='Weights + KV excludes temporary tensors and runtime allocations.')


def frame(value):
    return ('data: ' + json.dumps(value) + '\n\n').encode()


class LocalServing:
    """One execution slot, held through generation and stream delivery."""
    def __init__(self, generate=None):
        self.lock = asyncio.Lock()
        self.generate = generate or (lambda message: serving_benchmark_stream(message, mode='offline'))

    async def events(self, message):
        if message != WORKLOAD:
            raise ValueError('benchmark_workload_required')
        yield {'metadata': 'single_request'}
        queued = time.perf_counter()
        async with self.lock:
            yield {'queue_ms': (time.perf_counter() - queued) * 1000}
            stopped = False
            async for delta in self.generate(RequestText(message)):
                if delta.text:
                    yield {'text': delta.text}
                if delta.stop:
                    stopped = True
            if not stopped:
                raise ValueError('incomplete_generation')
            yield {'done': True}


class EventBytes(httpx.AsyncByteStream):
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        async for event in self.events:
            yield frame(event)

    async def aclose(self):
        await self.events.aclose()


def local_transport(server):
    async def dispatch(request):
        payload = json.loads(request.content)
        if payload != {'prompt': WORKLOAD, 'max_new_tokens': 32}:
            return httpx.Response(400)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              stream=EventBytes(server.events(payload['prompt'])))
    return httpx.MockTransport(dispatch)


async def local_events(client):
    async with client.stream('POST', 'http://127.0.0.1:8765/generate',
                             json={'prompt': WORKLOAD, 'max_new_tokens': 32}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                yield json.loads(line[6:])


async def hosted_events(mode):
    stopped = False
    async for delta in serving_benchmark_stream(RequestText(WORKLOAD), mode=mode):
        if delta.text:
            yield {'text': delta.text}
        if delta.stop:
            stopped = True
    if stopped:
        yield {'done': True}


async def measure(events):
    start = time.perf_counter()
    first = None
    count = 0
    done = False
    queue_ms = None
    async with asyncio.timeout(8):
        async for event in events:
            if 'queue_ms' in event:
                queue_ms = event['queue_ms']
            if event.get('text'):
                first = first if first is not None else time.perf_counter()
                count += 1
            done = done or event.get('done', False)
    end = time.perf_counter()
    if first is None or not done:
        raise ValueError('incomplete_stream')
    return dict(latency_ms=(end - start) * 1000, ttft_ms=(first - start) * 1000,
                output_events_s=count / max(end - first, 1e-9), queue_ms=queue_ms)


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1] if values else None


async def run(target, concurrency, *, mode='offline', samples=None, sink=emit_serving_benchmark):
    c = configuration()
    if target not in {'local', 'hosted'} or concurrency not in {1, 8} or mode not in {'offline', 'live'}:
        raise ValueError('invalid_benchmark_selection')
    n = c['samples'] if samples is None else samples
    if type(n) is not int or not 1 <= n <= 1000:
        raise ValueError('invalid_sample_count')
    server = LocalServing()
    transport = local_transport(server) if mode == 'offline' else httpx.AsyncHTTPTransport(retries=0)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=8) as client:
        async def one():
            async with semaphore:
                try:
                    events = local_events(client) if target == 'local' else hosted_events(mode)
                    return await measure(events)
                except Exception:
                    # Error text may contain provider data. Retain only a failure count.
                    return None
        start = time.perf_counter()
        rows = await asyncio.gather(*(one() for _ in range(n)))
        elapsed = time.perf_counter() - start
    good = [r for r in rows if r is not None]
    report = dict(target=target, execution_mode=mode, concurrency=concurrency,
                  completed=len(good), failures=n - len(good), completed_req_s=len(good) / elapsed)
    for metric in ('latency', 'ttft'):
        for label, fraction in [('p50', .5), ('p99', .99)]:
            report[metric + '_' + label + '_ms'] = percentile([r[metric + '_ms'] for r in good], fraction)
    sink(report)
    return dict(report, samples=n, warmups=0, startup_included=False,
                workload_sha256=hashlib.sha256(WORKLOAD.encode()).hexdigest(),
                output_cap=32, output_tokens_s=None,
                output_tokens_basis='not_available_from_text_chunks',
                output_events_s_p50=percentile([r['output_events_s'] for r in good], .5),
                queue_ms_p99=percentile([r['queue_ms'] for r in good if r['queue_ms'] is not None], .99),
                timing_basis='measured_in_memory_harness_simulated_generation' if mode == 'offline'
                             else 'measured_streaming_client',
                memory=assumptions(c),
                evidence_warning='Offline timings do not verify weights, hardware, model speed, or hosted service capacity.')
