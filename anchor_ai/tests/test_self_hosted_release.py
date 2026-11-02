import asyncio
import json
from pathlib import Path
import tempfile

import pytest

from anchor_ai.resilience import circuit_key
from ops.deploy.self_hosted_release import OfflineHarness, demo, require_production


def run(operation):
    return asyncio.run(operation)


def test_golden_failover_cases(tmp_path):
    for line in Path('eval/datasets/self_hosted_failover_cases.jsonl').read_text().splitlines():
        case = json.loads(line)
        h = OfflineHarness(tmp_path / case['id'])
        h.profiles.select('local-v1')
        h.local.server.stopped = case['stopped']
        result = run(h.chat(case['tenant']))
        assert result.get('provider', result['status']) == case['expected']
        assert h.committed_actions == {'refund-R42'}
        assert len(h.charges) <= 1
        if result['status'] == 'human_escalation':
            assert 'me@example.com' not in json.dumps(result)
            assert result['context']['committed_actions'] == ['refund-R42']
            assert result['context']['evidence'] and result['context']['messages']
            assert result['tool_calls'] == []


def test_policy_change_invalidates_cached_hosted_answer(tmp_path):
    h = OfflineHarness(tmp_path)
    assert run(h.chat())['provider'] == 'primary'
    def restrict(c):
        c['self_hosted_release']['tenants'][1]['chat'].update(
            external_processing=False, approved=[])
    result = run(h.chat(configure=restrict))
    assert result['status'] == 'human_escalation'
    assert len(h.events) == 1


def test_workload_readiness_and_missing_policy(tmp_path):
    h = OfflineHarness(tmp_path)
    h.profiles.select('local-v1')
    assert run(h.chat(workload='kb_embedding'))['human_handoff']
    def unready(c):
        c['deployments']['local']['ready'] = False
    assert run(h.chat(configure=unready))['provider'] == 'primary'
    h.service.cache.clear()
    assert run(h.chat(99))['human_handoff']


def test_stall_has_bounded_fallback_and_cancellation(tmp_path):
    h = OfflineHarness(tmp_path)
    h.profiles.select('local-v1')
    h.local.server.block = True
    result = run(h.chat())
    assert result['provider'] == 'primary'
    assert h.local.server.cancelled
    assert [e['provider'] for e in h.events] == ['local', 'primary']


def test_shared_attempt_limit(tmp_path):
    h = OfflineHarness(tmp_path)
    h.profiles.select('local-v1')
    original = h.factory
    def factory(*args):
        router = original(*args)
        h.boundaries['local-v1'].policy['max_attempts'] = 1
        return router
    h.service.router_factory = factory
    h.local.server.stopped = True
    assert run(h.chat())['human_handoff']
    assert [e['provider'] for e in h.events] == ['local']


def test_circuit_cooldown_and_consecutive_probes(tmp_path):
    async def scenario():
        h = OfflineHarness(tmp_path)
        h.profiles.select('local-v1')
        h.local.server.stopped = True
        for _ in range(2):
            h.service.cache.clear()
            assert (await h.chat())['provider'] == 'primary'
        boundary = h.boundaries['local-v1']
        key = circuit_key(boundary.router, 'local')
        assert boundary.store.snapshot(key)['state'] == 'open'
        h.local.server.stopped = False
        assert not await boundary.probe('local', 1)
        h.now = 2
        assert await boundary.probe('local', 1)
        assert boundary.store.snapshot(key)['state'] == 'half-open'
        assert await boundary.probe('local', 1)
        assert boundary.store.snapshot(key)['state'] == 'closed'
        assert [event['trace'] for event in h.events[-2:]] == [
            'recovery-probe', 'recovery-probe']
        assert 'ticket-42' in h.charges and 'recovery-probe' in h.charges
        h.service.cache.clear()
        assert (await h.chat())['provider'] == 'local'
    run(scenario())


def test_human_only_runs_no_model_or_tool(tmp_path):
    h = OfflineHarness(tmp_path)
    h.service.human_only = True
    result = run(h.chat())
    assert result['human_handoff'] and not h.events and not h.charges
    assert result['context']['committed_actions'] == ['refund-R42']


def test_rollback_pins_inflight_without_replaying_actions(tmp_path):
    async def scenario():
        h = OfflineHarness(tmp_path)
        h.profiles.select('local-v1')
        entered, release = asyncio.Event(), asyncio.Event()
        original = h.local.send
        async def delayed(request, payload):
            entered.set()
            await release.wait()
            return await original(request, payload)
        h.local.send = delayed
        pending = asyncio.create_task(h.chat())
        await entered.wait()
        h.profiles.select('previous')
        new_request = await h.chat()
        assert new_request['profile'] == 'previous'
        release.set()
        old_request = await pending
        assert old_request['profile'] == 'local-v1'
        assert old_request['provider'] == 'local'
        assert h.committed_actions == {'refund-R42'} and len(h.charges) == 1
    run(scenario())


def test_production_refuses_even_claimed_evidence():
    with pytest.raises(ValueError, match='validator_missing'):
        require_production({'status': 'passed', 'gpu': True})


def test_demo():
    result = run(demo())
    assert result['outage']['provider'] == 'primary'
    assert result['local_only']['human_handoff']
