import asyncio
import importlib.util
from pathlib import Path

import httpx
import pytest
from anchor_ai.provider import ProviderError, RoutingProvider
from anchor_ai.quota_client import QuotaDenied

path = Path(__file__).resolve().parents[2] / 'labs/provider-failover/drill.py'
spec = importlib.util.spec_from_file_location('anchor_failover_drill', path)
drill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drill)


def test_threshold_fresh_reservations_and_bounded_visits(tmp_path):
    async def scenario():
        client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db')
        await client.complete_result('Can I return order R-42?')
        initial = authority.snapshot()['used']
        adapters['primary'].scenario = 'outage'
        for _ in range(2):
            start = len(router.events)
            result = await client.complete_result('Can I return order R-42?')
            visits = [e['provider'] for e in router.events[start:] if e['classification'] == 'dispatch']
            assert visits == ['primary', 'secondary']
            assert result.reply and router.served_by == 'secondary'
        assert router.health['primary']['degraded']
        primary_calls = len(adapters['primary'].requests)
        await client.complete_result('Can I return order R-42?')
        assert len(adapters['primary'].requests) == primary_calls
        rows = authority.snapshot()['reservations']
        assert len(rows) == len({row[0] for row in rows}) == 6
        assert authority.snapshot()['used'] > initial
        events = router.collector.events
        assert any(e['provider'] == 'secondary' and e['attempt'] == 2 for e in events)
        assert len({e['reservation_id'] for e in events}) == len(events)
        assert len(router.collector.unresolved) == 2
    asyncio.run(scenario())


@pytest.mark.parametrize('scenario,code', [
    ('bad', 'provider_bad_request'), ('auth', 'provider_auth_failed'),
    ('forbidden', 'provider_forbidden'), ('quota', 'provider_quota_exhausted'),
    ('unknown429', 'provider_unknown_429'),
])
def test_nonavailability_errors_pass_through(tmp_path, scenario, code):
    async def run():
        client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db')
        adapters['primary'].scenario = scenario
        with pytest.raises(ProviderError) as caught:
            await client.complete_result('R-42')
        assert caught.value.code == code
        assert router.health['primary']['failures'] == 0
        assert len(adapters['primary'].requests) == 1
        assert not adapters['secondary'].requests
    asyncio.run(run())


@pytest.mark.parametrize('tenant', [2, 3])
def test_approval_and_residency_fail_closed(tmp_path, tenant):
    async def run():
        client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db', tenant=tenant)
        adapters['primary'].scenario = 'outage'
        with pytest.raises(ProviderError, match='routing_unavailable'):
            await client.complete_result('R-42')
        assert not adapters['secondary'].requests
        if tenant == 3:
            assert not adapters['primary'].requests
            assert authority.snapshot()['used'] == 0
    asyncio.run(run())


def test_budget_not_reset_before_secondary_dispatch(tmp_path):
    async def run():
        client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db', limit=110)
        await client.complete_result('R-42')
        adapters['primary'].scenario = 'outage'
        with pytest.raises(QuotaDenied, match='tenant_cost_quota'):
            await client.complete_result('R-42')
        assert authority.snapshot()['used'] == 110
        assert not adapters['secondary'].requests
    asyncio.run(run())


def test_stream_and_tool_requests_never_replay(tmp_path):
    async def run():
        client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db')
        emitted = []
        with pytest.raises(ProviderError, match='routing_stream_unsupported'):
            async for text in client.stream_tokens('R-42'):
                emitted.append(text)
        assert emitted == []
        async with client.scoped_client() as (identity, _):
            for extension in ({'stream': True}, {'tools': [{'name': 'issue_refund'}]}):
                request = httpx.Request('POST', 'https://offline.invalid', json={
                    'messages': [{'role': 'user', 'content': 'R-42'}], **extension})
                with pytest.raises(ProviderError):
                    await router.route(request, identity.tenant_id)
        assert all(not a.requests for a in adapters.values())
        assert authority.snapshot()['used'] == 0
    asyncio.run(run())


def test_real_timeout_and_probe_gated_failback(tmp_path):
    async def run():
        client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db')
        adapters['primary'].scenario = 'timeout'
        for _ in range(2):
            await client.complete_result('R-42')
        assert adapters['primary'].cancelled
        assert router.health['primary']['degraded']
        assert not await client.probe('primary')
        clock.advance()
        assert not await client.probe('primary')
        assert router.health['primary']['degraded']
        adapters['primary'].scenario = 'ok'
        assert not await client.probe('primary')
        clock.advance()
        assert await client.probe('primary')
        await client.complete_result('R-42')
        assert router.served_by == 'primary'
    asyncio.run(run())


def test_production_refuses_fixture_routing(tmp_path):
    client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db')
    with pytest.raises(ValueError, match='routing_release_evidence_missing'):
        RoutingProvider(adapters, router.collector, production=True)


def test_routing_verifies_identity_once(tmp_path):
    async def run():
        client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db')
        original, calls = client.verify, []
        def verify(token):
            calls.append(token)
            return original(token)
        client.verify = verify
        await client.complete_result('R-42')
        assert calls == ['fixture-execution-token']
        assert router.collector.events[0]['tenant_id'] == 1
    asyncio.run(run())


def test_unsafe_routing_metadata_is_rejected():
    from anchor_ai.observability import emit_provider_routing
    with pytest.raises(ValueError, match='unsafe_routing_telemetry'):
        emit_provider_routing({'prompt': 'customer text'})


@pytest.mark.parametrize('scenario', ['connection', 'rate'])
def test_transient_errors_fail_over_once(tmp_path, scenario):
    async def run():
        client, router, authority, adapters, clock = drill.build(tmp_path / 'quota.db')
        adapters['primary'].scenario = scenario
        await client.complete_result('R-42')
        assert router.served_by == 'secondary'
        assert router.health['primary']['failures'] == 1
        assert [len(adapters[n].requests) for n in ('primary', 'secondary')] == [1, 1]
    asyncio.run(run())
