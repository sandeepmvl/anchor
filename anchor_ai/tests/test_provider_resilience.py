import asyncio
import copy
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import httpx
import pytest

from anchor_ai.provider import ProviderError, RoutingProvider, classify
from anchor_ai.resilience import (
    CircuitStore, ResilienceBoundary, attach, circuit_key, configuration,
    delay_for, is_retryable,
)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


class RouterDouble:
    def __init__(self, clock, scenarios=None):
        self.clock, self.scenarios = clock, scenarios or {}
        self.calls, self.served_by = [], None
        self.config = dict(order=['primary', 'secondary'], logical_model='support-answer',
                           deployments={n: dict(model='fixture', region='eu')
                                        for n in ['primary', 'secondary']})

    def eligible(self, name, tenant_id, payload):
        return tenant_id == 1

    async def physical(self, name, request, payload):
        self.calls.append((name, self.clock()))
        self.clock.now += .01
        scenario = self.scenarios.get(name, 'ok')
        if scenario == 'timeout':
            raise ProviderError('provider_timeout')
        if scenario == 'rate':
            return httpx.Response(429, headers={'Retry-After': '.03'},
                                  json={'error': {'code': 'rate_limit_exceeded'}})
        if scenario == 'context':
            return httpx.Response(400, json={'error': {'code': 'context_length_exceeded'}})
        if scenario == 'outage':
            return httpx.Response(503, json={'error': {'code': 'unavailable'}})
        return httpx.Response(200, json={'answer': 'R-42 policy answer'})


def request(**extra):
    return httpx.Request('POST', 'https://offline.invalid', json=dict(
        messages=[dict(role='user', content='Can I return order R-42?')], **extra))


def boundary(tmp_path, scenarios=None, **changes):
    clock = Clock()
    policy = configuration()
    policy.update(changes)
    router = RouterDouble(clock, scenarios)
    store = CircuitStore(tmp_path / 'circuits.db', policy, clock=clock)
    events = []
    guard = ResilienceBoundary(router, store, policy, clock=clock,
                               sleep=clock.sleep, jitter=lambda: .005,
                               emit=events.append)
    return clock, policy, router, store, guard, events


def test_one_attempt_budget_across_retries_and_failover(tmp_path):
    async def run():
        clock, policy, router, store, guard, events = boundary(
            tmp_path, {'primary': 'timeout', 'secondary': 'outage'},
            trip_threshold=20, max_attempts=3)
        with pytest.raises(ProviderError, match='routing_attempt_limit'):
            await guard.route(request(), 1)
        assert [n for n, _ in router.calls] == ['primary', 'primary', 'secondary']
        assert clock.waits == pytest.approx([.015, .025])
        assert clock() <= policy['total_seconds']
        print('GLOBAL BUDGET: primary retry and secondary failover share 3 attempts')
    asyncio.run(run())


def test_absolute_deadline_includes_retry_after_and_jitter(tmp_path):
    async def run():
        clock, policy, router, store, guard, events = boundary(
            tmp_path, {'primary': 'rate'}, total_seconds=.09,
            trip_threshold=20)
        with pytest.raises(ProviderError, match='routing_deadline'):
            await guard.route(request(), 1)
        # .01 elapsed + .035 required wait + .05 complete attempt > .09.
        assert len(router.calls) == 1
        assert clock.waits == []
        assert delay_for(policy, 0, classify(httpx.Response(
            429, headers={'Retry-After': '.03'},
            json={'error': {'code': 'rate_limit_exceeded'}})), lambda: .005) == pytest.approx(.035)
        assert clock() <= policy['total_seconds']
        print('DEADLINE: Retry-After plus jitter cannot create another allowance')
    asyncio.run(run())


def test_open_requests_fail_fast_and_successful_probes_close(tmp_path):
    async def run():
        clock, policy, router, store, guard, events = boundary(
            tmp_path, {'primary': 'timeout', 'secondary': 'timeout'})
        with pytest.raises(ProviderError):
            await guard.route(request(), 1)
        assert len(router.calls) == 4
        before = len(router.calls)
        with pytest.raises(ProviderError, match='routing_unavailable'):
            await guard.route(request(), 1)
        assert len(router.calls) == before
        clock.now += policy['cooldown_seconds'] + .001
        # Cooldown expiry alone does not admit ordinary customer traffic.
        with pytest.raises(ProviderError, match='routing_unavailable'):
            await guard.route(request(), 1)
        assert len(router.calls) == before
        router.scenarios['primary'] = 'ok'
        assert await guard.probe('primary', 1)
        key = circuit_key(router, 'primary')
        assert store.snapshot(key)['state'] == 'half-open'
        assert await guard.probe('primary', 1)
        assert store.snapshot(key)['state'] == 'closed'
        assert (await guard.route(request(), 1)).status_code == 200
        print('DEMO: degraded request stops; subsequent request fails fast with 0 sends')
        print('RECOVERY: cooldown permits probes; two successful probes close the breaker')
    asyncio.run(run())


def test_shared_store_bounds_simultaneous_worker_probes(tmp_path):
    clock, policy, router, store, guard, events = boundary(tmp_path, trip_threshold=1)
    key = circuit_key(router, 'primary')
    store.finish(key, store.acquire(key), 'health_failure')
    clock.now += policy['cooldown_seconds'] + .001
    barrier = Barrier(4)

    def worker(_):
        separate_connection_owner = CircuitStore(store.path, policy, clock=clock)
        barrier.wait()
        return separate_connection_owner.acquire(key, probe=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        permits = list(pool.map(worker, range(4)))
    admitted = [p for p in permits if p is not None]
    assert len(admitted) == policy['half_open_probes'] == 1
    store.finish(key, admitted[0], 'health_failure')
    assert store.snapshot(key)['state'] == 'open'
    # A late result from the failed generation cannot close the new circuit.
    store.finish(key, admitted[0], 'success')
    assert store.snapshot(key)['state'] == 'open'
    print('SHARED PROBES: 4 worker connections, 1 admitted probe')


def test_permanent_errors_are_not_retried_or_health_failures(tmp_path):
    async def run():
        clock, policy, router, store, guard, events = boundary(
            tmp_path, {'primary': 'context'}, trip_threshold=1)
        response = await guard.route(request(), 1)
        assert response.status_code == 400
        assert len(router.calls) == 1
        state = store.snapshot(circuit_key(router, 'primary'))
        assert state['state'] == 'closed' and state['failures'] == 0
        assert not is_retryable(classify(response))
        assert not is_retryable(classify(httpx.Response(
            429, json={'error': {'code': 'insufficient_quota'}})))
    asyncio.run(run())


def test_neutral_probe_releases_permit_without_recovery(tmp_path):
    clock, policy, router, store, guard, events = boundary(tmp_path, trip_threshold=1)
    key = circuit_key(router, 'primary')
    store.finish(key, store.acquire(key), 'health_failure')
    clock.now += policy['cooldown_seconds'] + .001
    permit = store.acquire(key, probe=True)
    store.finish(key, permit, 'neutral')
    state = store.snapshot(key)
    assert state['successes'] == 0 and not state['leases']
    assert store.acquire(key, probe=True) is not None


def test_expired_probe_cannot_supply_recovery_evidence(tmp_path):
    clock, policy, router, store, guard, events = boundary(tmp_path, trip_threshold=1)
    key = circuit_key(router, 'primary')
    store.finish(key, store.acquire(key), 'health_failure')
    clock.now += policy['cooldown_seconds'] + .001
    old = store.acquire(key, probe=True)
    clock.now += policy['probe_lease_seconds'] + .001
    new = store.acquire(key, probe=True)
    assert new is not None
    store.finish(key, old, 'success')
    assert store.snapshot(key)['successes'] == 0
    store.finish(key, new, 'success')
    assert store.snapshot(key)['successes'] == 1


def load_existing_drill():
    path = Path('labs/provider-failover/drill.py').resolve()
    spec = importlib.util.spec_from_file_location('anchor_resilience_existing_drill', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_product_timeout_retains_unknown_billing_reservations(tmp_path):
    async def run():
        client, router, authority, adapters, clock = load_existing_drill().build(
            tmp_path / 'quota.db')
        guard = attach(router, database=tmp_path / 'product-circuits.db')
        adapters['primary'].scenario = 'timeout'
        await client.complete_result('Can I return order R-42?')
        assert adapters['primary'].cancelled
        assert len(router.collector.unresolved) == 2
        snapshot = authority.snapshot()
        # Two unknown primary attempts retain 100 each; the known successful
        # secondary settles to its smaller actual charge.
        assert snapshot['used'] >= 200
        assert len(snapshot['reservations']) == 3
        assert router.served_by == 'secondary'
        print('ACCOUNTING: timed-out sends retain conservative reservations')
    asyncio.run(run())


def test_outage_burst_has_no_internal_wait_queue(tmp_path):
    async def run():
        clock, policy, router, store, guard, events = boundary(tmp_path, max_inflight=2)
        entered = asyncio.Event()
        release = asyncio.Event()
        peak = 0
        active = 0

        async def stalled(name, req, payload):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            entered.set()
            try:
                await release.wait()
                return httpx.Response(503)
            finally:
                active -= 1

        router.physical = stalled
        tasks = [asyncio.create_task(guard.route(request(), 1)) for _ in range(20)]
        await entered.wait()
        await asyncio.sleep(0)
        assert guard.active <= 2
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(r, ProviderError) for r in results)
        assert peak <= 2 and guard.active == 0
        assert sum(e['decision'] == 'capacity_rejected' for e in events) == 18
        print('BURST: 20 arrivals, at most 2 active operations, 18 rejected, no internal queue')
    asyncio.run(run())


def test_production_still_refuses_offline_routing(tmp_path):
    client, router, authority, adapters, clock = load_existing_drill().build(tmp_path / 'quota.db')
    with pytest.raises(ValueError, match='routing_release_evidence_missing'):
        RoutingProvider(adapters, router.collector, production=True)
