"""Run accounting verification without sharing application state with the suite."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHILD = 'ANCHOR_COST_PYTHON_CHILD'


if os.environ.get(CHILD) != '1':
    class CostAccountingProcessTest(unittest.TestCase):
        def test_accounting_in_isolated_process(self):
            env = dict(os.environ, **{CHILD: '1'})
            # Third-party pytest plugins are unnecessary for this synchronous
            # verifier. Do not inherit the parent runner's plugin options.
            env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
            env.pop('PYTEST_ADDOPTS', None)
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pytest', '-q', str(Path(__file__).resolve())],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=25)
            except subprocess.TimeoutExpired as error:
                self.fail('Accounting child exceeded 25 seconds; it was stopped.\n'
                          + str(error.stdout or '') + str(error.stderr or ''))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
else:
    # This branch exists only in the bounded verification child. In particular,
    # collecting the full project suite does not import the profiler here.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import asyncio
    import pytest
    from anchor_ai.cost_accounting import Collector, normalize, price, account_batch
    from ops.load.profile_tickets import perform, replay

    @pytest.fixture(autouse=True)
    def cost_project_directory(monkeypatch):
        monkeypatch.chdir(ROOT)

    def test_subsets_and_retry_have_exclusive_destinations():
        async def run():
            c = Collector()
            await perform(c, feature='rag', role='base_completion', model='standard',
                          quantities=(200, 50, 50, 10), retry=True)
            assert len(c.events) == 2
            assert c.events[0]['attempt_id'] != c.events[1]['attempt_id']
            assert c.events[0]['logical_call_id'] == c.events[1]['logical_call_id']
            assert price(c.events[0], c.config) == dict(uncached_input=300, cached_input=50,
                visible_output=240, reasoning_output=60, embedding_calls=0, overhead=0)
            assert price(c.events[1], c.config) == dict(uncached_input=0, cached_input=0,
                visible_output=0, reasoning_output=0, embedding_calls=0, overhead=650)
            c.require_release()
            with pytest.raises(ValueError, match='real_cost_evidence_required'):
                c.require_release(production=True, evidence={'provider_invoice_reconciled': True})
        asyncio.run(run())

    def test_usage_validation_does_not_charge_aggregate_twice():
        assert normalize({'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30,
            'completion_tokens_details': {'reasoning_tokens': 8}}) == dict(input=20, cached=0, output=10, reasoning=8)
        for raw in (None, {'prompt_tokens': True, 'completion_tokens': 0},
                    {'prompt_tokens': 2, 'completion_tokens': 1,
                     'prompt_tokens_details': {'cached_tokens': 3}}):
            with pytest.raises(ValueError):
                normalize(raw)

    def test_batch_aggregate_is_one_attempt_not_one_per_item():
        class Batch:
            async def call_with_usage(self, key, records):
                return {identity: {'embedding': [0]} for identity, _ in records}, {'prompt_tokens': 20}
        async def run():
            c = Collector()
            with c.scope(tenant_id=1, feature='rag', role='first_query_embedding', model='embedding'):
                result = await account_batch(Batch(), (), [('a', 'fixture'), ('b', 'fixture')], c)
            assert set(result) == {'a', 'b'}
            assert len(c.events) == 1
            assert price(c.events[0], c.config)['embedding_calls'] == 20
            with c.scope(tenant_id=1, feature='rag', role='first_query_embedding', model='embedding'):
                with pytest.raises(ValueError, match='batch_usage_capability_required'):
                    await account_batch(object(), (), [], c)
        asyncio.run(run())

    def test_unknown_usage_blocks_reconciliation():
        async def run():
            import httpx
            from anchor_ai.cost_accounting import AccountingProvider
            class Unknown:
                def key(self):
                    return 'offline'
                def client(self):
                    return httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda request: httpx.Response(503, json={'error': 'fixture'})))
            c = Collector()
            with c.scope(tenant_id=1, feature='chat', role='base_completion', model='standard'):
                async with AccountingProvider(Unknown(), c).client() as client:
                    await client.post('https://offline.invalid')
            assert not c.events and len(c.unresolved) == 1
            with pytest.raises(ValueError, match='usage_reconciliation_required'):
                c.require_release()
        asyncio.run(run())

    def test_thirty_ticket_oracle():
        c = asyncio.run(replay())
        assert len(c.events) == 120
        assert len({e['attempt_id'] for e in c.events}) == 120
        assert all(set(e['usage']) == {'input', 'cached', 'output', 'reasoning'} for e in c.events)

    def test_verified_scope_rejects_missing_identity_before_accounting():
        from anchor_ai.runtime import verified_cost_scope
        c = Collector()
        with pytest.raises(ValueError, match='missing_execution_context'):
            with verified_cost_scope(c, verify=lambda token: None, token='',
                                     feature='chat', role='base_completion', model='standard'):
                pytest.fail('unverified scope opened')
        assert not c.events

    def test_accounted_batch_adapter_and_safe_projection():
        from anchor_ai.batching import AccountedBatchAdapter
        from anchor_ai.observability import cost_measurement
        class Batch:
            fixture = True
            async def call_with_usage(self, key, records):
                return {identity: {} for identity, _ in records}, {'prompt_tokens': 20}
        async def run():
            c = Collector()
            adapter = AccountedBatchAdapter(Batch(), c, scope=lambda key: c.scope(
                tenant_id=key[0], feature='rag', role='first_query_embedding', model=key[3]))
            result = await adapter.call((1, 0, 'embedding', 'embedding'), [('a', 'fixture'), ('b', 'fixture')])
            assert set(result) == {'a', 'b'} and len(c.events) == 1
            diagnostic = cost_measurement(c.events[0], c.config)
            assert diagnostic['total'] == 20
            assert 'usage' not in diagnostic and 'prompt' not in diagnostic
        asyncio.run(run())
