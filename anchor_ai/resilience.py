"""Deadline-aware resilience for Anchor's existing buffered routing boundary.

SQLite coordinates workers on one host. It is not a multi-host authority.
No network adapter, provider selection policy, or quota allowance lives here.
"""
import asyncio
import contextlib
import json
import math
import random
import sqlite3
import time
import uuid
from pathlib import Path
import yaml


def configuration(path='config/provider_resilience.yml'):
    c = yaml.safe_load(Path(path).read_text())
    integers = ('max_attempts', 'retries_per_provider', 'trip_threshold',
                'reset_threshold', 'half_open_probes', 'max_inflight')
    numbers = ('total_seconds', 'attempt_seconds', 'base_seconds', 'cap_seconds',
               'jitter_seconds', 'cooldown_seconds', 'probe_lease_seconds')
    if not isinstance(c, dict):
        raise ValueError('invalid_resilience_configuration')
    for key in integers:
        minimum = 0 if key == 'retries_per_provider' else 1
        if type(c.get(key)) is not int or c[key] < minimum:
            raise ValueError('invalid_resilience_configuration')
    for key in numbers:
        value = c.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value <= 0):
            raise ValueError('invalid_resilience_configuration')
    if (c['attempt_seconds'] > c['total_seconds']
            or c['base_seconds'] > c['cap_seconds']
            or c['probe_lease_seconds'] <= c['attempt_seconds']
            or c.get('sdk_retries') != 0
            or c.get('record_content') is not False
            or c.get('retry_owner', 'resilience') != 'resilience'):
        raise ValueError('invalid_resilience_configuration')
    return c


class CircuitStore:
    """Atomic state and leased probe permits shared by local worker processes.

    Keys include deployment/model/region. Use a separate database per credential
    domain. Every participant must use the same monotonic clock domain.
    """
    def __init__(self, path, policy, *, clock=time.monotonic):
        self.path, self.policy, self.clock = str(path), policy, clock
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS provider_circuits ('
                       'key TEXT PRIMARY KEY, value TEXT NOT NULL)')

    def db(self):
        return sqlite3.connect(self.path, timeout=.1)

    @contextlib.contextmanager
    def locked(self, key):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT value FROM provider_circuits WHERE key=?', (key,)).fetchone()
            state = json.loads(row[0]) if row else dict(
                state='closed', generation=0, failures=0, successes=0,
                until=0, leases={})
            yield state
            db.execute('INSERT OR REPLACE INTO provider_circuits VALUES (?,?)', (key, json.dumps(state)))

    def snapshot(self, key):
        with self.locked(key) as s:
            return json.loads(json.dumps(s))

    def acquire(self, key, *, probe=False):
        with self.locked(key) as s:
            now = self.clock()
            if s['state'] == 'closed':
                return (s['generation'], None, False) if not probe else None
            if not probe or now < s['until']:
                return None
            s['state'] = 'half-open'
            s['leases'] = {k: v for k, v in s['leases'].items() if v > now}
            if len(s['leases']) >= self.policy['half_open_probes']:
                return None
            token = uuid.uuid4().hex
            s['leases'][token] = now + self.policy['probe_lease_seconds']
            return s['generation'], token, True

    def finish(self, key, permit, outcome):
        if outcome not in {'success', 'health_failure', 'neutral'}:
            raise ValueError('invalid_circuit_outcome')
        generation, token, probe = permit
        with self.locked(key) as s:
            if generation != s['generation']:
                return
            if probe:
                expiry = s['leases'].pop(token, None)
                if expiry is None or expiry <= self.clock():
                    return
            if outcome == 'neutral':
                return
            if outcome == 'health_failure':
                s['failures'] += 1
                if probe or s['failures'] >= self.policy['trip_threshold']:
                    s.update(state='open', generation=s['generation'] + 1,
                             successes=0, leases={}, until=self.clock() + self.policy['cooldown_seconds'])
                return
            if probe:
                s['successes'] += 1
                if s['successes'] < self.policy['reset_threshold']:
                    return
            s.update(state='closed', failures=0, successes=0, until=0,
                     leases={}, generation=s['generation'] + (1 if probe else 0))


def circuit_key(router, name):
    d = router.config['deployments'][name]
    return json.dumps([name, d['model'], d['region']], separators=(',', ':'))


def is_retryable(error):
    return getattr(error, 'code', None) in {
        'provider_timeout', 'provider_transport_error',
        'provider_unavailable', 'provider_rate_limited'}


def delay_for(policy, index, error, jitter):
    hint = getattr(error, 'retry_after', None)
    if type(hint) not in (int, float) or not math.isfinite(hint) or hint < 0:
        hint = 0
    noise = jitter()
    if (type(noise) not in (int, float) or not math.isfinite(noise)
            or not 0 <= noise <= policy['jitter_seconds']):
        raise ValueError('invalid_retry_jitter')
    exponential = min(policy['cap_seconds'], policy['base_seconds'] * 2 ** min(index, 30))
    return max(exponential, hint) + noise


class ResilienceBoundary:
    """One retry owner; ordinary traffic never becomes a recovery probe.

    physical() must perform exactly one send and reserve before that send.
    asyncio bounds cooperative asynchronous adapters, not blocking SDK code.
    """
    def __init__(self, router, store, policy, *, clock=time.monotonic,
                 sleep=asyncio.sleep, jitter=None, emit=None):
        order = router.config['order']
        if not order or len(order) != len(set(order)):
            raise ValueError('cyclic_routing')
        if policy.get('retry_owner', 'resilience') != 'resilience' or policy.get('sdk_retries', 0) != 0:
            raise ValueError('conflicting_retry_ownership')
        self.router, self.store, self.policy = router, store, policy
        self.clock, self.sleep = clock, sleep
        self.jitter = jitter or (lambda: random.uniform(0, policy['jitter_seconds']))
        self.emit = emit or (lambda record: None)
        self.active = 0

    def event(self, decision, attempts, delay=0):
        record = dict(decision=decision, attempts=attempts, delay_seconds=delay)
        self.emit(record)

    def error(self, code):
        from anchor_ai.provider import ProviderError
        return ProviderError(code, status=503)

    async def send(self, name, request, payload, allowance, permit):
        from anchor_ai.provider import ProviderError, classify
        outcome = 'neutral'
        try:
            async with asyncio.timeout(allowance):
                response = await self.router.physical(name, request, payload)
            if response.status_code != 200:
                error = classify(response)
                outcome = 'health_failure' if is_retryable(error) else 'neutral'
                return response, error
            outcome = 'success'
            return response, None
        except TimeoutError:
            outcome = 'health_failure'
            return None, ProviderError('provider_timeout', retryable=True)
        except ProviderError as error:
            outcome = 'health_failure' if is_retryable(error) else 'neutral'
            return None, error
        finally:
            self.store.finish(circuit_key(self.router, name), permit, outcome)

    async def route(self, request, tenant_id):
        payload = json.loads(request.content)
        if payload.get('stream'):
            raise self.error('routing_stream_unsupported')
        if payload.get('tools') or payload.get('tool_choice'):
            raise self.error('routing_tool_unsupported')
        if self.active >= self.policy['max_inflight']:
            self.event('capacity_rejected', 0)
            raise self.error('routing_capacity')
        self.active += 1
        try:
            return await self.operation(request, tenant_id, payload)
        finally:
            self.active -= 1

    async def operation(self, request, tenant_id, payload):
        deadline = self.clock() + self.policy['total_seconds']
        attempts, retry_index = 0, 0
        async with asyncio.timeout(self.policy['total_seconds']):
            for name in self.router.config['order']:
                if not self.router.eligible(name, tenant_id, payload):
                    continue
                for local in range(self.policy['retries_per_provider'] + 1):
                    if attempts >= self.policy['max_attempts']:
                        self.event('attempt_limit', attempts)
                        raise self.error('routing_attempt_limit')
                    remaining = deadline - self.clock()
                    if remaining < self.policy['attempt_seconds']:
                        self.event('deadline', attempts)
                        raise self.error('routing_deadline')
                    permit = self.store.acquire(circuit_key(self.router, name))
                    if permit is None:
                        self.event('circuit_skip', attempts)
                        break
                    attempts += 1
                    response, error = await self.send(name, request, payload,
                        min(self.policy['attempt_seconds'], remaining), permit)
                    if self.clock() > deadline:
                        raise self.error('routing_deadline')
                    if error is None:
                        self.router.served_by = name
                        self.event('success', attempts)
                        return response
                    if not is_retryable(error):
                        self.event('permanent_error', attempts)
                        if response is not None:
                            return response
                        raise error
                    if attempts >= self.policy['max_attempts']:
                        raise self.error('routing_attempt_limit')
                    delay = delay_for(self.policy, retry_index, error, self.jitter)
                    retry_index += 1
                    if delay + self.policy['attempt_seconds'] > deadline - self.clock():
                        self.event('deadline', attempts)
                        raise self.error('routing_deadline')
                    self.event('backoff', attempts, delay)
                    await self.sleep(delay)
                    if local == self.policy['retries_per_provider']:
                        break
        self.event('unavailable', attempts)
        raise self.error('routing_unavailable')

    async def probe(self, name, tenant_id):
        import httpx
        payload = dict(model=self.router.config['logical_model'], max_tokens=16,
                       messages=[dict(role='user', content='Synthetic recovery probe')])
        if not self.router.eligible(name, tenant_id, payload):
            return False
        if self.active >= self.policy['max_inflight']:
            return False
        permit = self.store.acquire(circuit_key(self.router, name), probe=True)
        if permit is None:
            self.event('probe_rejected', 0)
            return False
        self.active += 1
        try:
            request = httpx.Request('POST', 'https://offline.invalid/probe', json=payload)
            _, error = await self.send(name, request, payload, self.policy['attempt_seconds'], permit)
            self.event('probe_failed' if error else 'probe_success', 1)
            return error is None
        finally:
            self.active -= 1


def attach(router, *, database, policy_path='config/provider_resilience.yml',
           clock=None, sleep=asyncio.sleep, jitter=None, emit=None):
    """Bind once; this does not grant release evidence or a quota allowance."""
    if hasattr(router, 'resilience'):
        raise ValueError('provider_resilience_already_initialized')
    policy = configuration(policy_path)
    clock = clock or time.monotonic
    if router.config.get('sdk_retries', 0) != 0:
        raise ValueError('nested_provider_retries')
    if router.config['attempt_timeout_seconds'] < policy['attempt_seconds']:
        raise ValueError('nested_timeout_smaller_than_resilience_allowance')
    boundary = ResilienceBoundary(router, CircuitStore(database, policy, clock=clock), policy,
        clock=clock, sleep=sleep, jitter=jitter, emit=emit)
    router.route = boundary.route
    router.probe = boundary.probe
    router.resilience = boundary
    return boundary
