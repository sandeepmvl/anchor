def reserve_routing_attempt():
    """Each deployment send and recovery probe needs fresh authority admission."""
    return reserve_attempt()


"""Admission interface. No implicit network, local allowance, or fail-open mode."""
import contextvars
import json
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
import httpx

ACTIVE_QUOTA = contextvars.ContextVar('anchor_quota', default=None)


class QuotaDenied(RuntimeError):
    def __init__(self, code, status=503, retry_after=None):
        self.code, self.status, self.retry_after = code, status, retry_after
        super().__init__(code)


class QuotaClient:
    def __init__(self, *, url, secret, context, transport=None):
        if not url.startswith(('http://127.0.0.1:', 'https://')) or not secret or not context:
            raise QuotaDenied('quota_configuration_required')
        self.url, self.secret, self.context = url.rstrip('/'), secret, context
        self.transport = transport

    def request(self, method, suffix='', body=None):
        try:
            with httpx.Client(transport=self.transport, timeout=2, trust_env=False,
                              follow_redirects=False) as client:
                response = client.request(method, self.url + suffix, json=body,
                    headers={'X-Anchor-Secret': self.secret,
                             'X-Anchor-Quota-Context': self.context})
            payload = response.json()
            if response.status_code != 200:
                raise QuotaDenied(payload.get('error', 'quota_authority_unavailable'),
                                  response.status_code, response.headers.get('Retry-After'))
            return payload
        except QuotaDenied:
            raise
        except Exception:
            raise QuotaDenied('quota_authority_unavailable') from None

    def reserve(self, *, bound, attempt_id):
        return self.request('POST', body={'bound': bound, 'attempt_id': attempt_id})

    def settle(self, reservation_id, actual):
        return self.request('PATCH', '/' + reservation_id, {'actual': actual})

    def snapshot(self):
        return self.request('GET')


class RubyQuotaClient:
    """Offline test bridge to AiQuotaPolicy; no server and no network."""
    def __init__(self, *, database, config, identity, clock=1893456000):
        self.database, self.config, self.identity, self.clock = database, config, identity, clock

    def request(self, operation, **values):
        driver = Path('labs/quotas-rate-limits/quota_burst_test.rb')
        payload = dict(operation=operation, database=str(self.database), config=self.config,
                       identity=self.identity, clock=self.clock, **values)
        try:
            result = subprocess.run(['ruby', str(driver), '--rpc'],
                                    input=json.dumps(payload), text=True,
                                    capture_output=True, timeout=4, check=True)
            answer = json.loads(result.stdout)
        except Exception:
            raise QuotaDenied('quota_authority_unavailable') from None
        if 'error' in answer:
            raise QuotaDenied(answer['error'], answer['status'], answer.get('retry_after'))
        return answer

    def reserve(self, *, bound, attempt_id):
        return self.request('reserve', bound=bound, attempt_id=attempt_id)

    def settle(self, reservation_id, actual):
        return self.request('settle', reservation_id=reservation_id, actual=actual)

    def snapshot(self):
        return self.request('snapshot')


@contextmanager
def quota_scope(client, *, bound):
    if type(bound) is not int or bound <= 0:
        raise QuotaDenied('quota_bound_required')
    token = ACTIVE_QUOTA.set((client, bound))
    try:
        yield
    finally:
        ACTIVE_QUOTA.reset(token)


def reserve_attempt():
    state = ACTIVE_QUOTA.get()
    if state is None:
        raise QuotaDenied('verified_quota_scope_required')
    client, bound = state
    attempt_id = uuid.uuid4().hex
    reservation = client.reserve(bound=bound, attempt_id=attempt_id)
    if (not isinstance(reservation, dict) or reservation.get('reservation_id') != attempt_id
            or type(reservation.get('bound')) is not int or reservation['bound'] < bound):
        raise QuotaDenied('quota_reservation_invalid')
    return client, reservation


async def admitted_call(call, *args, actual=None, **kwargs):
    """One physical attempt; uncertain usage never releases reservations."""
    client, reservation = reserve_attempt()
    result = await call(*args, **kwargs)
    if actual is not None:
        amount = actual(result)
        if type(amount) is not int or not 0 <= amount <= reservation['bound']:
            raise QuotaDenied('quota_bound_exceeded', 409)
        client.settle(reservation['reservation_id'], amount)
    return result


class AdmittedBatchAdapter:
    """Trusted scope reserves the homogeneous batch's maximum charge.

    An adapter that internally retries must admit inside each retry.
    """
    def __init__(self, adapter, *, scope):
        self.adapter, self.scope = adapter, scope
        self.fixture = getattr(adapter, 'fixture', True)

    async def call(self, key, records):
        with self.scope(key, records):
            return await admitted_call(self.adapter.call, key, records)

    async def submit_job(self, key, records):
        submit = getattr(self.adapter, 'submit_job', None)
        if submit is None:
            raise QuotaDenied('async_batch_capability_unavailable')
        with self.scope(key, records):
            return await admitted_call(submit, key, records)


class AdmittedProvider:
    """HTTP hooks admit each send, including application retries.

    Trusted composition supplies a bound covering the enforced payload.
    Streaming retains its reservation until final usage is reconciled.
    """
    def __init__(self, provider):
        self.provider = provider

    def key(self):
        return self.provider.key()

    def client(self):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def opened():
            async def admit(request):
                client, reservation = reserve_attempt()
                request.extensions['anchor_budget_reservation'] = reservation
            async with self.provider.client() as client:
                client.event_hooks['request'].insert(0, admit)
                yield client
        return opened()


def require_quota_release(*, production, fixture, evidence=None):
    required = {'all_paid_paths', 'real_bounds', 'legacy_reconciled', 'shared_authority'}
    if production and (fixture or not evidence or
                       any(evidence.get(k) is not True for k in required)):
        raise QuotaDenied('quota_release_evidence_missing')


def retain_uncertain_attempt(collector, attempt_id):
    """Record uncertainty only; never release the authority's bound."""
    from anchor_ai.observability import safe_id
    if safe_id(attempt_id) is None:
        raise QuotaDenied('quota_reservation_invalid')
    collector.unresolved.append(attempt_id)


class IncidentAuthority:
    """Trusted local replay bridge, not a production authenticated transport."""
    def __init__(self, database, tenant=1):
        self.database, self.tenant = str(database), tenant

    def call(self, operation, **arguments):
        payload = dict(database=self.database, operation=operation,
                       arguments=dict(tenant=self.tenant, **arguments))
        try:
            result = subprocess.run(
                ['ruby', 'labs/failover-budget-incident/incident_replay.rb', '--rpc'],
                input=json.dumps(payload), text=True, capture_output=True, timeout=4,
                check=True)
            answer = json.loads(result.stdout)
            if not isinstance(answer, dict) or 'error' in answer:
                raise ValueError('invalid_authority_response')
            return answer
        except Exception:
            from anchor_ai.provider import ProviderError
            raise ProviderError('quota_authority_unavailable', status=503) from None
