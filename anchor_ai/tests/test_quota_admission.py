import asyncio
import json
import subprocess
from contextlib import contextmanager

import httpx
import pytest

from anchor_ai.quota_client import (
    AdmittedBatchAdapter, AdmittedProvider, QuotaClient, QuotaDenied,
    RubyQuotaClient, admitted_call, quota_scope, require_quota_release,
)


@pytest.fixture
def authority(tmp_path):
    database = tmp_path / 'application.sqlite3'
    code = "require_relative 'labs/quotas-rate-limits/quota_burst_test'; QuotaLab.prepare(ARGV.fetch(0))"
    subprocess.run(['ruby', '-e', code, str(database)], check=True, capture_output=True)
    config = json.load(open('config/ai_budgets.yml'))
    config['quotas']['daily_limit'] = 5
    config['budgets']['chat']['limit'] = 5
    identity = {'tenant_id': 91001, 'actor': 'customer-1', 'feature': 'chat'}
    return RubyQuotaClient(database=database, config=config, identity=identity)


def test_every_physical_retry_needs_sufficient_reservation(authority):
    calls = []

    async def provider():
        calls.append('sent')
        raise TimeoutError('uncertain provider outcome')

    async def run():
        with quota_scope(authority, bound=3):
            with pytest.raises(TimeoutError):
                await admitted_call(provider)
            with pytest.raises(QuotaDenied, match='budget_exhausted'):
                await admitted_call(provider)
    asyncio.run(run())
    assert calls == ['sent']
    assert authority.snapshot()['reserved'] == 3


def test_successful_settlement_releases_only_unused_bound(authority):
    async def provider():
        return {'actual': 2}

    async def run():
        with quota_scope(authority, bound=3):
            await admitted_call(provider, actual=lambda result: result['actual'])
            await admitted_call(provider, actual=lambda result: result['actual'])
            with pytest.raises(QuotaDenied, match='budget_exhausted'):
                await admitted_call(provider, actual=lambda result: result['actual'])
    asyncio.run(run())
    assert authority.snapshot()['spent'] == 4
    assert authority.snapshot()['reserved'] == 0


def test_unavailable_authority_blocks_background_and_batch_paths():
    calls = []

    def broken(request):
        raise httpx.ConnectError('offline', request=request)

    client = QuotaClient(url='http://127.0.0.1:9999/api/ai_budget_reservations',
                         secret='service', context='signed', transport=httpx.MockTransport(broken))

    class Adapter:
        async def call(self, key, records):
            calls.append('batch')
            return {}

        async def submit_job(self, key, records):
            calls.append('job')
            return 'provider-job'

    @contextmanager
    def scope(key, records):
        with quota_scope(client, bound=3):
            yield

    async def background():
        calls.append('background')

    async def run():
        with quota_scope(client, bound=3):
            with pytest.raises(QuotaDenied, match='quota_authority_unavailable'):
                await admitted_call(background)
        adapter = AdmittedBatchAdapter(Adapter(), scope=scope)
        for operation in (adapter.call, adapter.submit_job):
            with pytest.raises(QuotaDenied, match='quota_authority_unavailable'):
                await operation(('verified-tenant',), [('id', 'text')])
        with pytest.raises(QuotaDenied, match='verified_quota_scope_required'):
            await admitted_call(background)
    asyncio.run(run())
    assert calls == []


def test_http_hook_reserves_each_send(authority):
    calls = []

    class Provider:
        def client(self):
            def send(request):
                calls.append('physical-send')
                return httpx.Response(200, json={'reply': 'fake'})
            return httpx.AsyncClient(transport=httpx.MockTransport(send))

    async def run():
        with quota_scope(authority, bound=3):
            async with AdmittedProvider(Provider()).client() as client:
                await client.post('https://offline.invalid/completion', json={})
                with pytest.raises(QuotaDenied, match='budget_exhausted'):
                    await client.post('https://offline.invalid/completion', json={})
    asyncio.run(run())
    assert calls == ['physical-send']


def test_production_release_requires_real_evidence():
    with pytest.raises(QuotaDenied, match='quota_release_evidence_missing'):
        require_quota_release(production=True, fixture=True, evidence={})
