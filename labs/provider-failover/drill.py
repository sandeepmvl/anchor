import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.cost_accounting import Collector
from anchor_ai.provider import ProviderError, OfflineDeployment, RoutingProvider
from anchor_ai.llm_client import RoutedLlmClient
from anchor_ai.quota_client import QuotaDenied


class FixtureAuthority:
    """SQLite fixture, not Rails authentication or the production quota service."""
    def __init__(self, path, limit=1000):
        self.path, self.limit = str(path), limit
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, amount INTEGER)')

    def reserve(self, *, bound, attempt_id):
        with sqlite3.connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            used = db.execute('SELECT COALESCE(SUM(amount),0) FROM reservations').fetchone()[0]
            if used + bound > self.limit:
                raise QuotaDenied('tenant_cost_quota', 429)
            db.execute('INSERT INTO reservations VALUES (?,?)', (attempt_id, bound))
        return {'reservation_id': attempt_id, 'bound': bound}

    def settle(self, reservation_id, actual):
        with sqlite3.connect(self.path) as db:
            old = db.execute('SELECT amount FROM reservations WHERE id=?', (reservation_id,)).fetchone()
            if old is None or type(actual) is not int or not 0 <= actual <= old[0]:
                raise QuotaDenied('invalid_settlement')
            db.execute('UPDATE reservations SET amount=? WHERE id=?', (actual, reservation_id))

    def snapshot(self):
        with sqlite3.connect(self.path) as db:
            rows = db.execute('SELECT id, amount FROM reservations').fetchall()
        return {'used': sum(row[1] for row in rows), 'reservations': rows}


class Clock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        return self.value
    def advance(self):
        self.value += 31


def build(path, *, tenant=1, limit=1000):
    config = {'fixture': True, 'price_version': 'fixture_v1', 'budget_version': 'fixture_v1',
              'unit': 'synthetic', 'budgets': {'chat': {'limit': 1000, 'warning_percent': 80}},
              'models': {'gpt-4o-mini': {'input': 1, 'cached': 1, 'output': 1, 'embedding': 1}}}
    collector = Collector(config)
    adapters = {name: OfflineDeployment() for name in ('primary', 'secondary')}
    clock = Clock()
    router = RoutingProvider(adapters, collector, clock=clock)
    authority = FixtureAuthority(path, limit)
    identity = SimpleNamespace(tenant_id=tenant, role='support')
    def verify(token):
        if token != 'fixture-execution-token':
            raise ValueError('fixture_identity_rejected')
        return identity
    client = RoutedLlmClient(router, verify=verify, token='fixture-execution-token',
                             client_for=lambda identity, feature: authority)
    return client, router, authority, adapters, clock


async def run(mode):
    with tempfile.TemporaryDirectory() as directory:
        client, router, authority, adapters, clock = build(
            Path(directory) / 'budget.sqlite3', tenant=2 if mode == 'policy' else 1,
            limit=110 if mode == 'budget' else 1000)
        ticket = 'Can I return order R-42?'
        first = await client.complete_result(ticket)
        assert router.served_by == 'primary'
        before = authority.snapshot()['used']
        outcome = 'primary'
        if mode != 'primary':
            adapters['primary'].scenario = 'outage'
            try:
                for _ in range(3 if mode in {'outage', 'recovery'} else 1):
                    await client.complete_result(ticket)
                outcome = router.served_by
            except (ProviderError, QuotaDenied) as error:
                outcome = error.code
            if mode == 'recovery':
                clock.advance()
                assert not await client.probe('primary')
                adapters['primary'].scenario = 'ok'
                clock.advance()
                assert await client.probe('primary')
                await client.complete_result(ticket)
                outcome = router.served_by
        after = authority.snapshot()['used']
        result = dict(ticket='R-42', outcome=outcome, served_by=router.served_by,
                      initial_reply=first.reply, budget_before=before, budget_after=after,
                      budget_reset=False, secondary_dispatches=len(adapters['secondary'].requests),
                      reservations=len(authority.snapshot()['reservations']), offline=True,
                      not_verified=['Rails authentication', 'live provider equivalence',
                                    'invoice reconciliation', 'shared worker health'])
        if mode == 'outage':
            assert outcome == 'secondary' and after > before
            assert router.health['primary']['degraded']
        if mode == 'policy':
            assert outcome == 'routing_unavailable' and not adapters['secondary'].requests
        if mode == 'budget':
            assert outcome == 'tenant_cost_quota' and not adapters['secondary'].requests
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'outage'
    if mode not in {'primary', 'outage', 'policy', 'budget', 'recovery'}:
        raise SystemExit('unknown drill scenario')
    asyncio.run(run(mode))
