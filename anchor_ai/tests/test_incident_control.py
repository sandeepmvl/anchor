import asyncio
import json

import httpx
import pytest

from anchor_ai.resilience import CircuitStore, ResilienceBoundary, circuit_key, configuration
from anchor_ai.provider import ProviderError


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


class Router:
    def __init__(self, allowed=('primary', 'secondary'), fail_all=False):
        self.config = {'order': ['primary', 'secondary'], 'deployments': {
            name: {'model': 'fixture', 'region': 'eu'}
            for name in ('primary', 'secondary')}}
        self.allowed, self.fail_all = allowed, fail_all
        self.calls = []
        self.enabled = True
        self.incident_check = lambda tenant: tenant == 1 and self.enabled

    def eligible(self, name, tenant, payload):
        return tenant == 1 and name in self.allowed

    async def physical(self, name, request, payload):
        self.calls.append(name)
        return httpx.Response(503 if name == 'primary' or self.fail_all else 200,
                              json={'reply': 'Refund proposal requires confirmation.'})


def setup_boundary(tmp_path, router, **overrides):
    policy = configuration()
    policy.update(overrides)
    clock, events = Clock(), []
    store = CircuitStore(tmp_path / 'circuits.sqlite3', policy, clock=clock)
    boundary = ResilienceBoundary(router, store, policy, clock=clock,
                                 sleep=clock.sleep, jitter=lambda: 0,
                                 emit=events.append)
    return boundary, store, clock, events


def request(**extra):
    return httpx.Request('POST', 'https://offline.invalid/chat', json={
        'messages': [{'role': 'user', 'content': 'Refund order A17'}], **extra})


def test_breaker_and_only_permitted_fallback(tmp_path):
    router = Router()
    boundary, store, clock, events = setup_boundary(tmp_path, router)
    assert asyncio.run(boundary.route(request(), 1)).status_code == 200
    assert router.calls == ['primary', 'primary', 'secondary']
    assert store.snapshot(circuit_key(router, 'primary'))['state'] == 'open'
    assert asyncio.run(boundary.route(request(), 1)).status_code == 200
    assert router.calls == ['primary', 'primary', 'secondary', 'secondary']
    router.allowed = ('primary',)
    with pytest.raises(ProviderError):
        asyncio.run(boundary.route(request(), 1))
    assert router.calls[-1] == 'secondary'
    assert len(router.calls) == 4
    assert any(event['decision'] == 'circuit_skip' for event in events)
    print('Primary breaker open; only tenant-permitted fallback used; no tool dispatch.')


def test_one_attempt_budget_and_deadline(tmp_path):
    router = Router(fail_all=True)
    boundary, store, clock, events = setup_boundary(tmp_path, router)
    with pytest.raises(ProviderError) as caught:
        asyncio.run(boundary.route(request(), 1))
    assert caught.value.code == 'routing_attempt_limit'
    assert len(router.calls) == boundary.policy['max_attempts'] == 4
    assert clock.now < boundary.policy['total_seconds']
    for name in router.config['order']:
        assert store.snapshot(circuit_key(router, name))['state'] == 'open'

    other = tmp_path / 'deadline'
    router = Router(fail_all=True)
    boundary, store, clock, events = setup_boundary(
        other, router, total_seconds=.06, base_seconds=.02)
    with pytest.raises(ProviderError) as caught:
        asyncio.run(boundary.route(request(), 1))
    assert caught.value.code == 'routing_deadline'
    assert router.calls == ['primary']
    assert clock.now <= .06
    print('One shared attempt budget and deadline bound retries plus fallback.')


def test_stop_during_backoff_and_no_provider_tools(tmp_path):
    router = Router()
    boundary, store, clock, events = setup_boundary(tmp_path, router)

    async def stop_during_sleep(seconds):
        clock.now += seconds
        router.enabled = False

    boundary.sleep = stop_during_sleep
    with pytest.raises(ProviderError) as caught:
        asyncio.run(boundary.route(request(), 1))
    assert caught.value.code == 'incident_human_only'
    assert router.calls == ['primary']
    with pytest.raises(ProviderError) as caught:
        asyncio.run(boundary.route(request(tools=[{'type': 'function'}]), 1))
    assert caught.value.code == 'routing_tool_unsupported'
    assert router.calls == ['primary']
