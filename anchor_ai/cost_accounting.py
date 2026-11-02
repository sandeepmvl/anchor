"""Attempt accounting adapter; never store prompts, replies, or HTTP bodies."""
import contextvars
import copy
import json
import re
import uuid
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

DRIVERS = ('uncached_input', 'cached_input', 'visible_output',
           'reasoning_output', 'embedding_calls', 'overhead')
ROLES = {'base_completion', 'first_query_embedding', 'extra_query_embedding', 'rerank'}
ACTIVE_COST = contextvars.ContextVar('anchor_cost_scope', default=None)


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid_cost_integer')
    return value


def configuration(path='config/ai_budgets.yml'):
    # JSON is also YAML; both services read exactly the same values.
    c = json.loads(Path(path).read_text())
    for key in ('price_version', 'budget_version', 'unit'):
        if not isinstance(c[key], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', c[key]):
            raise ValueError('invalid_cost_configuration')
    if type(c['fixture']) is not bool:
        raise ValueError('invalid_cost_configuration')
    for values in c['models'].values():
        for value in values.values():
            integer(value)
    for budget in c['budgets'].values():
        integer(budget['limit'], 1)
        if not 1 <= integer(budget['warning_percent'], 1) <= 100:
            raise ValueError('invalid_warning_percent')
    return c


def normalize(raw, style='chat', embedding=False):
    if not isinstance(raw, dict):
        raise ValueError('usage_unresolved')
    if embedding:
        i = integer(raw.get('prompt_tokens', raw.get('input_tokens')))
        return dict(input=i, cached=0, output=0, reasoning=0)
    if style not in {'chat', 'responses'}:
        raise ValueError('unsupported_usage_style')
    ik, ok = ('prompt_tokens', 'completion_tokens') if style == 'chat' else ('input_tokens', 'output_tokens')
    i, o = integer(raw.get(ik)), integer(raw.get(ok))
    # This adapter's supported API schema defines omitted detail counters as zero.
    # APIs with cache-write premiums need their own normalizer and rate schema.
    details, output_details = raw.get(ik + '_details') or {}, raw.get(ok + '_details') or {}
    if details.get('cache_write_tokens', 0):
        raise ValueError('cache_write_pricing_not_supported')
    c, r = integer(details.get('cached_tokens', 0)), integer(output_details.get('reasoning_tokens', 0))
    if c > i or r > o:
        raise ValueError('invalid_usage_subsets')
    return dict(input=i, cached=c, output=o, reasoning=r)


def price(event, config):
    role, attempt = event['role'], integer(event['attempt'], 1)
    if role not in ROLES or event['feature'] not in config['budgets']:
        raise ValueError('unconfigured_cost_destination')
    u = event['usage']
    if set(u) != {'input', 'cached', 'output', 'reasoning'}:
        raise ValueError('invalid_usage_fields')
    i, c, o, r = (integer(u[k]) for k in ('input', 'cached', 'output', 'reasoning'))
    if c > i or r > o:
        raise ValueError('invalid_usage_subsets')
    rates = config['models'][event['model']]
    embedding = role in {'first_query_embedding', 'extra_query_embedding'}
    if embedding:
        if c or o or r:
            raise ValueError('invalid_embedding_usage')
        total = i * rates['embedding']
    else:
        total = (i-c) * rates['input'] + c * rates['cached'] + o * rates['output']
    costs = dict.fromkeys(DRIVERS, 0)
    if attempt > 1 or role in {'extra_query_embedding', 'rerank'}:
        costs['overhead'] = total
    elif embedding:
        costs['embedding_calls'] = total
    else:
        costs.update(uncached_input=(i-c)*rates['input'], cached_input=c*rates['cached'],
                     visible_output=(o-r)*rates['output'], reasoning_output=r*rates['output'])
    return costs


class Collector:
    def __init__(self, config=None):
        self.config = config or configuration()
        self.events, self.unresolved = [], []

    @contextmanager
    def scope(self, *, tenant_id, feature, role, model, logical_call_id=None, window=None):
        integer(tenant_id, 1)
        if feature not in self.config['budgets'] or role not in ROLES or model not in self.config['models']:
            raise ValueError('unconfigured_cost_scope')
        logical_call_id = logical_call_id or uuid.uuid4().hex
        window = window or datetime.now(timezone.utc).strftime('%Y-%m')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', logical_call_id):
            raise ValueError('invalid_logical_call_id')
        if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', window):
            raise ValueError('invalid_cost_window')
        state = dict(tenant_id=tenant_id, feature=feature, role=role, model=model,
                     logical_call_id=logical_call_id, window=window, attempts=0)
        token = ACTIVE_COST.set(state)
        try:
            yield state
        finally:
            ACTIVE_COST.reset(token)

    def attempt(self):
        state = ACTIVE_COST.get()
        if state is None:
            raise ValueError('verified_cost_scope_required')
        state['attempts'] += 1
        return dict(tenant_id=state['tenant_id'], feature=state['feature'],
                    role=state['role'], model=state['model'], window=state['window'],
                    logical_call_id=state['logical_call_id'], attempt=state['attempts'],
                    attempt_id=uuid.uuid4().hex,
                    price_version=self.config['price_version'],
                    budget_version=self.config['budget_version'])

    def settle(self, attempt, raw, *, style='chat'):
        event = dict(attempt, usage=normalize(raw, style,
            embedding=attempt['role'] in {'first_query_embedding', 'extra_query_embedding'}))
        price(event, self.config)
        self.events.append(copy.deepcopy(event))
        return event

    def require_release(self, *, production=False, evidence=None):
        if self.unresolved:
            raise ValueError('usage_reconciliation_required')
        if production and (self.config['fixture'] or not evidence or
                           evidence.get('provider_invoice_reconciled') is not True):
            raise ValueError('real_cost_evidence_required')


class AccountingProvider:
    """Wrap existing client()/key() providers. No implicit live-provider selection.

    Use only with buffered JSON requests. Streaming needs a final-usage adapter.
    One request hook assigns one attempt, not one event per logical invocation.
    """
    def __init__(self, provider, collector, *, style='chat'):
        self.provider, self.collector, self.style = provider, collector, style

    def key(self):
        return self.provider.key()

    @asynccontextmanager
    async def client(self):
        pending = {}

        async def request_hook(request):
            attempt = self.collector.attempt()
            request.extensions['anchor_cost_attempt'] = attempt
            pending[attempt['attempt_id']] = attempt

        async def response_hook(response):
            attempt = response.request.extensions['anchor_cost_attempt']
            try:
                await response.aread()
                body = response.json()
                self.collector.settle(attempt, body.get('usage'), style=self.style)
            except (ValueError, KeyError, TypeError):
                self.collector.unresolved.append(attempt['attempt_id'])
            finally:
                pending.pop(attempt['attempt_id'], None)

        try:
            async with self.provider.client() as client:
                client.event_hooks['request'].append(request_hook)
                client.event_hooks['response'].append(response_hook)
                yield client
        finally:
            # Transport failures can still incur spend. Never settle them as zero.
            self.collector.unresolved.extend(pending)


async def account_batch(adapter, key, records, collector):
    """Opt-in adapter protocol: (results, aggregate_usage), one event per attempt.

    key and records remain those of Anchor's batching interface. The caller must
    open one verified tenant-feature scope for the whole homogeneous batch.
    An adapter lacking usage refuses this path rather than inventing quantities.
    """
    call = getattr(adapter, 'call_with_usage', None)
    if call is None:
        raise ValueError('batch_usage_capability_required')
    attempt = collector.attempt()
    try:
        results, aggregate = await call(key, records)
        collector.settle(attempt, aggregate)
        return results
    except Exception:
        collector.unresolved.append(attempt['attempt_id'])
        raise


def settle_quota_usage(client, reservation, attempt, raw, config, *, style='chat'):
    """Settle authority first; historical events are not an extra budget debit."""
    from anchor_ai.quota_client import QuotaDenied
    event = dict(attempt, usage=normalize(raw, style,
        embedding=attempt['role'] in {'first_query_embedding', 'extra_query_embedding'}))
    amount = sum(price(event, config).values())
    if amount > reservation['bound']:
        raise QuotaDenied('quota_bound_exceeded', 409)
    client.settle(reservation['reservation_id'], amount)
    return dict(event, reservation_id=reservation['reservation_id'])
