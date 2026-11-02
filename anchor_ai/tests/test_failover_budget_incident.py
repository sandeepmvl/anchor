import asyncio
import json
import pytest
import yaml
from pathlib import Path
from anchor_ai.provider import incident_policy_violations as violations
from anchor_ai.runtime import (
    IncidentRuntime, INCIDENT_POLICY as POLICY, broken_trace,
    corrected_incident_policy as corrected_policy, recovered_trace,
    incident_regression_cases as regression_cases)


def test_rejects_cycle_and_conflicting_owners(tmp_path):
    broken = json.loads(POLICY.read_text())
    assert violations(broken) == ['cyclic_routing', 'conflicting_retry_ownership',
                                  'quota_window_misconfiguration']
    for mutation, expected in [
        ({'order': ['primary', 'secondary', 'primary']}, 'cyclic_routing'),
        ({'retry_owners': ['resilience', 'sdk']}, 'conflicting_retry_ownership'),
        ({'reset': 'per_request'}, 'quota_window_misconfiguration')]:
        policy = corrected_policy()
        policy.update(mutation)
        with pytest.raises(ValueError, match=expected):
            IncidentRuntime(tmp_path, policy)
    assert regression_cases() == 4


def test_diagnosis_rejects_token_growth_and_parallelism():
    rows = broken_trace()
    attempts = [r for r in rows if r['event'] == 'attempt']
    series = [r for r in rows if r['event'] == 'series']
    assert {r['input_tokens'] for r in attempts} == {128}
    assert {r['active_concurrency'] for r in attempts} == {1}
    assert {r['used_before'] for r in attempts} == {0}
    assert series[-1]['spend_units'] > series[0]['spend_units']
    assert series[-1]['max_attempt_latency_s'] > series[0]['max_attempt_latency_s']


def test_recovery_bounds_attempts_and_spend(tmp_path):
    report = asyncio.run(recovered_trace(tmp_path))
    assert report['authority']['used'] <= 12
    assert len(report['calls']) == report['authority']['attempts']
    assert {c['ticket'] for c in report['calls']} == {'A17-' + str(i) for i in range(6)}
    for index in range(6):
        assert sum(c['ticket'] == 'A17-' + str(index) for c in report['calls']) <= 4
    assert any(d['decision'] == 'circuit_skip' for d in report['decisions'])


def test_no_calls_after_human_only_even_after_worker_restart(tmp_path):
    async def scenario():
        first = IncidentRuntime(tmp_path)
        context = {'message': 'A17', 'history': ['customer question'],
                   'evidence': ['order:A17'], 'proposed_action': 'lookup_order'}
        await first.ticket('before', context)
        first.authority.call('queue', ticket='queued', context=context)
        first.authority.call('human_only')
        calls = len(first.router.calls)
        await first.ticket('queued', context)
        assert len(first.router.calls) == calls
        second = IncidentRuntime(tmp_path)
        await second.ticket('new', context)
        assert not second.router.calls
        snapshot = second.authority.call('snapshot')
        required = yaml.safe_load(Path('config/agent_policy.yml').read_text())['incident']['context_fields']
        assert all(set(required) <= set(t['context']) for t in snapshot['tickets'])
        assert all(t['context'] == context for t in snapshot['tickets'])
        assert all(t['state'] == 'human' for t in snapshot['tickets'])
        assert snapshot['used'] == calls
    asyncio.run(scenario())


def test_missing_real_evidence_refuses_production(tmp_path):
    with pytest.raises(ValueError, match='incident_release_evidence_missing'):
        IncidentRuntime(tmp_path, production=True)


def test_policy_sources_agree():
    budgets = json.loads(Path('config/ai_budgets.yml').read_text())['incident_window']
    policy = corrected_policy()
    assert all(policy[k] == budgets[k] for k in ('reset', 'window_seconds', 'cap_units'))


def test_budget_denial_hands_off_without_send(tmp_path):
    async def scenario():
        runtime = IncidentRuntime(tmp_path)
        context = {'message': 'A17', 'history': [], 'evidence': [], 'proposed_action': 'lookup_order'}
        for i in range(12):
            result = runtime.authority.call('reserve', ticket='seed-' + str(i), context=context,
                                            attempt_id='seed-' + str(i), bound=1)
            assert result['allowed']
        result = await runtime.ticket('blocked', context)
        assert result['status'] == 'human'
        assert not runtime.router.calls
        snapshot = runtime.authority.call('snapshot')
        assert snapshot['used'] == 12
        assert next(t for t in snapshot['tickets'] if t['ticket'] == 'blocked')['context'] == context
    asyncio.run(scenario())
