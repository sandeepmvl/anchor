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


def compare_artifacts(baseline, candidate, runtime):
    from anchor_ai.serving_artifacts import compare, selection
    report = compare(baseline, candidate, runtime)
    report['decision'] = selection(candidate, runtime, report, baseline)
    return report


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


# Surge experiments are separate from the inherited single-request benchmark.
def surge_profile(name=None):
    c = configuration()
    profiles = c['batching_profiles']
    name = name or profiles['selected']
    if profiles['version'] != 1 or name not in ('default', 'tuned'):
        raise ValueError('invalid_batching_profile')
    p = dict(profiles[name])
    seqs, tokens = p['max_num_seqs'], p['max_num_batched_tokens']
    if (type(seqs) is not int or type(tokens) is not int
            or not 1 <= seqs <= 64 or not seqs <= tokens <= 4096):
        raise ValueError('invalid_batching_bounds')
    a = c['assumptions']
    kv = assumptions(c)['kv_bytes_per_active_request'] * seqs
    if a['parameters'] * a['bytes_per_weight'] + kv > a['memory_budget_bytes']:
        raise ValueError('assumed_kv_budget_exceeded')
    policy = dict(profiles['admission'])
    for key in ('tenant_inflight_limit', 'tenant_token_quota', 'request_deadline_s'):
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError('invalid_admission_policy')
    return name, p, policy, kv


def surge_launch(name=None):
    name, p, _, _ = surge_profile(name)
    spec = yaml.safe_load((ROOT / 'ops/deploy/vllm_service.yml').read_text())
    command = list(spec['command'])
    _, committed, _, _ = surge_profile(spec['batching_profile'])
    for key, value in committed.items():
        flag = '--' + key.replace('_', '-')
        if command.count(flag) != 1 or command[command.index(flag) + 1] != str(value):
            raise ValueError('deployment_profile_drift')
    if ('--enable-chunked-prefill' not in command
            or command[command.index('--scheduling-policy') + 1] != 'fcfs'
            or spec['release']['real_evidence_required'] is not True):
        raise ValueError('unsafe_batching_deployment')
    for key, value in p.items():
        command[command.index('--' + key.replace('_', '-')) + 1] = str(value)
    return command


def require_surge_promotion(report):
    # This lesson does not implement the independently reviewed live validator.
    raise ValueError('real_hardware_measurements_and_reviewed_validator_required')


class SurgeProtocolServer:
    """OpenAI-compatible fixture responses, not model inference or a tokenizer."""
    def __call__(self, request):
        body = json.loads(request.content)
        if (request.url.path != '/v1/chat/completions'
                or body.get('model') != 'tiny-llama'
                or body.get('messages') != [{'role': 'user', 'content': WORKLOAD}]
                or body.get('max_tokens') not in (4, 40)):
            return httpx.Response(400)
        n = body['max_tokens']
        return httpx.Response(200, json={
            'id': 'offline-fixture', 'object': 'chat.completion',
            'model': 'tiny-llama',
            'choices': [{'index': 0, 'message': {'role': 'assistant',
                         'content': 'Synthetic support reply.'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 32, 'completion_tokens': n,
                      'total_tokens': 32 + n}})


def surge(concurrency=50, profile=None, scheduling='continuous', *, samples=200,
          admission=None):
    if (type(concurrency) is not int or not 1 <= concurrency <= 200
            or type(samples) is not int or not 1 <= samples <= 1000
            or scheduling not in ('continuous', 'static', 'window')):
        raise ValueError('invalid_surge_selection')
    name, p, policy, kv = surge_profile(profile)
    launch = surge_launch(name)
    if admission is not None:
        if set(admission) != set(policy) or any(type(v) is not int or v <= 0 for v in admission.values()):
            raise ValueError('invalid_admission_override')
        policy = dict(admission)
    # Fixture calls validate the production-facing request shape in-process.
    descriptors = []
    with httpx.Client(transport=httpx.MockTransport(SurgeProtocolServer()), trust_env=False) as client:
        for i in range(samples):
            response = client.post('http://offline/v1/chat/completions', json={
                'model': 'tiny-llama', 'messages': [{'role': 'user', 'content': WORKLOAD}],
                'max_tokens': 40 if i % 4 == 0 else 4})
            response.raise_for_status()
            descriptors.append(response.json()['usage'])
    tenants = {t: dict(offered=0, accepted=0, rejected=0, completed=0,
                       timeouts=0, inflight=0, reserved_tokens=0, peak_inflight=0)
               for t in ('tenant-a', 'tenant-b')}
    records, waiting, active, trace = [], [], [], []
    now, submitted, terminal, window = 0.0, 0, 0, None

    def submit():
        nonlocal submitted, terminal
        i = submitted
        submitted += 1
        tenant = 'tenant-a' if i % 2 == 0 else 'tenant-b'
        t, usage = tenants[tenant], descriptors[i]
        t['offered'] += 1
        r = dict(id=i, tenant=tenant, arrival=now, admission=None, first=None,
                 finish=None, status=None, prefill=32, remaining=usage['completion_tokens'],
                 short=usage['completion_tokens'] == 4)
        records.append(r)
        cost = usage['total_tokens']
        if (t['inflight'] >= policy['tenant_inflight_limit']
                or t['reserved_tokens'] + cost > policy['tenant_token_quota']):
            r.update(status='rejected', finish=now)
            t['rejected'] += 1
            terminal += 1
        else:
            t['accepted'] += 1
            t['inflight'] += 1
            t['reserved_tokens'] += cost
            t['peak_inflight'] = max(t['peak_inflight'], t['inflight'])
            waiting.append(r)

    for _ in range(100000):
        while submitted < samples and submitted - terminal < concurrency:
            submit()
        if terminal == samples:
            break
        for r in records:
            if r['status'] is None and now - r['arrival'] >= policy['request_deadline_s']:
                r.update(status='timeouts', finish=now)
                tenants[r['tenant']]['timeouts'] += 1
                tenants[r['tenant']]['inflight'] -= 1
                terminal += 1
        waiting = [r for r in waiting if r['status'] is None]
        if scheduling == 'continuous':
            active = [r for r in active if r['status'] is None]
        elif active and all(r['status'] is not None for r in active):
            active = []
        may_admit = scheduling == 'continuous' or not active
        if scheduling == 'window' and not active and waiting:
            window = now + .03 if window is None else window
            may_admit = now >= window
        if may_admit:
            while waiting and len(active) < p['max_num_seqs']:
                r = waiting.pop(0)
                r['admission'] = now
                active.append(r)
                trace.append(dict(id=r['id'], time_s=now,
                                  members=[x['id'] for x in active]))
            if active:
                window = None
        budget, used, decoded = p['max_num_batched_tokens'], 0, []
        for r in active:
            if r['status'] is None and r['prefill'] == 0 and budget:
                r['remaining'] -= 1
                budget -= 1
                used += 1
                decoded.append(r)
        for r in active:
            if r['status'] is None and r['prefill'] and budget:
                chunk = min(budget, r['prefill'])
                r['prefill'] -= chunk
                budget -= chunk
                used += chunk
        # Explicit assumed service cost in virtual seconds, not a GPU timing model.
        now += .01 + .0005 * len(active) + .00002 * used
        for r in decoded:
            if r['first'] is None:
                r['first'] = now
            if r['remaining'] == 0:
                r.update(status='completed', finish=now)
                tenants[r['tenant']]['completed'] += 1
                tenants[r['tenant']]['inflight'] -= 1
                terminal += 1
    else:
        raise RuntimeError('surge_did_not_converge')
    good = [r for r in records if r['status'] == 'completed']
    waits = [(r['admission'] if r['admission'] is not None else r['finish'])
             - r['arrival'] for r in records]
    return dict(
        execution_mode='offline_protocol_and_virtual_scheduler', profile=name,
        scheduling=scheduling, concurrency=concurrency, samples=samples,
        workload_sha256=hashlib.sha256(json.dumps(descriptors, sort_keys=True).encode()).hexdigest(),
        completed=len(good), failures=samples - len(good),
        timeouts=sum(t['timeouts'] for t in tenants.values()),
        rejected=sum(t['rejected'] for t in tenants.values()),
        duration_virtual_s=now, completed_req_s=len(good) / now,
        latency_p50_s=percentile([r['finish'] - r['arrival'] for r in good], .5),
        latency_p99_s=percentile([r['finish'] - r['arrival'] for r in good], .99),
        short_first_p99_s=percentile([r['first'] - r['arrival'] for r in good if r['short']], .99),
        queue_wait_p99_s=percentile(waits, .99), tenants=tenants,
        profile_parameters=p, launch_arguments=launch, assumed_peak_kv_bytes=kv,
        kv_warning='Full-context KV estimate excludes runtime allocations; not device evidence.',
        production_promotion_allowed=False,
        evidence_warning='Real hardware measurements and a reviewed live validator are required before production promotion.',
        trace=trace)

