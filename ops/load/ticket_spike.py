"""Anchor surge reporting. Default execution never contacts a network service."""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import httpx
import yaml
from anchor_ai.observability import distribution, spike_measurement
from ops.load.profile_tickets import spike_profile


NOT_VERIFIED = [
    'production bounded-agent and RAG HTTP contracts', 'Rails and database capacity',
    'production authentication and cost reservations', 'real provider limits',
    'GPU inference saturation', 'customer network latency',
]


def configuration(path='config/performance.yml'):
    c = yaml.safe_load(Path(path).read_text())['ticket_spike']
    for key in ('workers', 'tenant_pending', 'tenant_requests', 'tenant_cost_units'):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError('invalid_spike_limits')
    for key in ('timeout_s', 'p99_objective_ms', 'queue_objective_ms'):
        if type(c.get(key)) not in (int, float) or not 0 < c[key] < 10000:
            raise ValueError('invalid_spike_objective')
    if c.get('scope') != 'benchmark_only' or c.get('record_content') is not False:
        raise ValueError('unsafe_spike_configuration')
    return c


class Admission:
    """Per-tenant pending and fixed-run request/cost budgets; no shared FIFO."""
    def __init__(self, config):
        self.c = config
        self.pending = Counter()
        self.requests = Counter()
        self.cost = Counter()

    def enter(self, tenant):
        if self.pending[tenant] >= self.c['tenant_pending']:
            return 'admission_rejected'
        if self.requests[tenant] >= self.c['tenant_requests']:
            return 'rate_rejected'
        # One unit reserves the fixture's bounded model work, not a USD amount.
        if self.cost[tenant] + 1 > self.c['tenant_cost_units']:
            return 'cost_rejected'
        self.pending[tenant] += 1
        self.requests[tenant] += 1
        self.cost[tenant] += 1
        return None

    def leave(self, tenant):
        self.pending[tenant] -= 1


class LocalTicketAdapter:
    """Test HTTP seam, not a replacement for anchor_ai.app or its auth guards."""
    def __init__(self, workers, every_429=0):
        self.workers = asyncio.Semaphore(workers)
        self.every_429 = every_429
        self.sequence = 0

    async def __call__(self, scope, receive, send):
        headers = dict(scope['headers'])
        trace = headers[b'traceparent'].decode().split('-')[1]
        path = scope['path']
        queued = time.perf_counter()
        async with self.workers:
            queue_ms = (time.perf_counter() - queued) * 1000
            self.sequence += 1
            throttled = self.every_429 and self.sequence % self.every_429 == 0
            if throttled:
                body = dict(outcome='provider_429', queue_ms=queue_ms,
                            generation_ms=0.0, trace_id=trace,
                            provider_error='rate_limit_exceeded')
                status = 429
            else:
                profile = await spike_profile(path, trace)
                body = dict(outcome='success', queue_ms=queue_ms,
                            generation_ms=profile['generation_ms'], trace_id=trace)
                status = 200
        await send(dict(type='http.response.start', status=status,
                        headers=[(b'content-type', b'application/json')]))
        await send(dict(type='http.response.body', body=json.dumps(body).encode()))


def summarize(rows, elapsed_s, concurrency, phase):
    counts = Counter(r['outcome'] for r in rows)
    successful = [r for r in rows if r['outcome'] == 'success']
    unsuccessful = [r['latency_ms'] for r in rows if r['outcome'] != 'success']
    return dict(phase=phase, concurrency=concurrency, scheduled=len(rows),
        started=sum(r['started'] for r in rows), completed=len(successful),
        failures=len(rows) - len(successful), outcomes=dict(counts),
        successful_req_s=len(successful) / elapsed_s,
        successful_latency_ms=distribution([r['latency_ms'] for r in successful]),
        unsuccessful_latency_ms=distribution(unsuccessful),
        worker_queue_ms=distribution([r['queue_ms'] for r in rows if r['queue_ms'] is not None]),
        generation_ms=distribution([r['generation_ms'] for r in successful]),
        elapsed_s=elapsed_s)


async def band(concurrency, config, *, every_429=0, admission=False, timeout_s=None):
    adapter = LocalTicketAdapter(config['workers'], every_429)
    policy = Admission(config)
    rows = []
    callers = asyncio.Semaphore(concurrency)
    timeout = config['timeout_s'] if timeout_s is None else timeout_s
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=adapter),
                                 base_url='http://offline-anchor') as client:
        async def ticket(index):
            async with callers:
                tenant = 2 if index % 5 == 4 else 1
                trace = uuid.uuid4().hex
                start = time.perf_counter()
                row = dict(tenant_id=tenant, trace_id=trace, started=0,
                           queue_ms=None, generation_ms=None)
                rejection = policy.enter(tenant) if admission else None
                if rejection:
                    row['outcome'] = rejection
                else:
                    row['started'] = 1
                    try:
                        path = '/rag' if index % 2 == 0 else '/bounded_agent'
                        response = await asyncio.wait_for(client.post(path,
                            headers={'traceparent': '00-' + trace + '-1111111111111111-01'},
                            json={'message': 'Where is order A17?'}), timeout)
                        payload = response.json()
                        row.update(outcome=payload['outcome'], queue_ms=payload['queue_ms'],
                                   generation_ms=payload['generation_ms'])
                    except asyncio.TimeoutError:
                        row['outcome'] = 'timeout'
                    except Exception:
                        row['outcome'] = 'transport_or_contract_error'
                    finally:
                        if admission:
                            policy.leave(tenant)
                row['latency_ms'] = (time.perf_counter() - start) * 1000
                rows.append(spike_measurement(row))
        start = time.perf_counter()
        await asyncio.gather(*(ticket(i) for i in range(100)))
        elapsed = time.perf_counter() - start
    phase = 'admission' if admission else 'provider_429' if every_429 else 'ramp'
    result = summarize(rows, elapsed, concurrency, phase)
    result['tenants'] = {str(t): dict(Counter(r['outcome'] for r in rows
                                           if r['tenant_id'] == t)) for t in (1, 2)}
    return result


def first_knee(bands, config):
    """First sampled objective miss with queue evidence; failures stay in bands."""
    for b in bands:
        latency, queue = b['successful_latency_ms'], b['worker_queue_ms']
        if (latency and queue and latency['p99'] > config['p99_objective_ms']
                and queue['p99'] > config['queue_objective_ms']):
            return b['concurrency']
    return None


def require_capacity_evidence(evidence):
    if (not isinstance(evidence, dict) or evidence.get('execution_mode') != 'live'
            or any(evidence.get(k) is not True for k in
                   ('target_adapter_verified', 'provider_budget_approved',
                    'stage_waits_measured', 'failures_accounted', 'recovery_observed'))):
        raise ValueError('live_capacity_evidence_required')


def markdown(report):
    lines = ['# Anchor ticket surge benchmark', '',
        'Execution: offline local HTTP adapter; not production capacity.',
        'Workload: 100 tickets per band, alternating paths, tenant mix 80/20.',
        'Closed concurrency ramp; no claim of an independent arrival rate.',
        'Nearest-rank percentiles; successful tickets only. Failures remain counted.',
        'Timeout stage timings are unknown, not zero. No retries are hidden.', '',
        '| Phase | Concurrency | Answers/s | p50/p95/p99 ms | Queue p99 ms | Failures | Outcomes |',
        '|---|---:|---:|---|---|---:|---|']
    for b in report['bands']:
        lines.append('| ' + ' | '.join(str(x) for x in (b['phase'], b['concurrency'],
            round(b['successful_req_s'], 2), b['successful_latency_ms'],
            b['worker_queue_ms'], b['failures'], b['outcomes'])) + ' |')
    lines += ['', 'First sampled ramp knee: ' + str(report['first_sampled_knee']),
        'Knee rule: successful p99 exceeds the configured objective AND queue p99 exceeds its objective.',
        'Generation timing, unsuccessful latency, and tenant outcomes are in the JSON output.',
        'Batch endpoint evidence: ' + json.dumps(report['batch'], sort_keys=True),
        'Admission evidence: ' + json.dumps(report['bands'][-1]['tenants'], sort_keys=True),
        'Decision: retain conservative benchmark admission; production capacity release refused.',
        'Not verified: ' + ', '.join(NOT_VERIFIED), '']
    return '\n'.join(lines)


async def run():
    from ops.load.batch_throughput import spike_batch_report
    c = configuration()
    ramp = [await band(n, c) for n in (1, 4, 16, 32)]
    throttled = await band(16, c, every_429=4)
    timed = await band(16, c, timeout_s=.000001)
    admitted = await band(32, c, admission=True)
    report = dict(execution_mode='offline', bands=ramp + [throttled, timed, admitted],
                  first_sampled_knee=first_knee(ramp, c), batch=await spike_batch_report(),
                  not_verified=NOT_VERIFIED)
    Path('ops/load/benchmark_report.md').write_text(markdown(report))
    print(json.dumps(report, indent=2))
    print('Anchor surge capacity report: worker queue wait and fake-provider 429s are separate; '
          'batching and tenant admission evidence recorded. Production capacity not verified.')
    return report


# Optional Locust entry point. Importing the offline benchmark needs no Locust install.
# Set ANCHOR_LOCUST=1 only in a separately prepared staging generator environment.
if os.getenv('ANCHOR_LOCUST') == '1':
    from locust import HttpUser, task, between

    class AnchorTicketUser(HttpUser):
        wait_time = between(1, 3)

        @task
        def ticket(self):
            # Operator supplies a verified staging route and its exact JSON body.
            path = os.environ['ANCHOR_LOAD_PATH']
            payload = json.loads(os.environ['ANCHOR_LOAD_BODY'])
            headers = json.loads(os.environ['ANCHOR_LOAD_HEADERS'])
            with self.client.post(path, json=payload, headers=headers,
                                  timeout=10, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure('non_200_ticket')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(run())
