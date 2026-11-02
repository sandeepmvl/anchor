import asyncio
import copy
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from anchor_ai.orchestration import (
    FRAMEWORK_LABELS, framework_policy, framework_ranking, propose,
    run_framework_task, select_flow, select_framework, worker_for,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'config/orchestration.yml'
EXPECTED = {'escalation': 'LangGraph', 'rag': 'LlamaIndex', 'triage': 'LangChain'}
ENVELOPE = {'tenant_id': 7, 'trace_id': 'mixed-ticket-A17',
            'budget': {'max_tokens': 256, 'max_steps': 4}}


@pytest.mark.parametrize('usecase,winner', EXPECTED.items())
def test_committed_matrix_matches_ruby_and_written_priorities(usecase, winner):
    data = framework_policy(CONFIG)
    rows = framework_ranking(data, usecase)
    assert select_framework(usecase, CONFIG) == winner
    command = ['ruby', str(ROOT / 'labs/framework-comparison/framework_selection_matrix.rb'),
               '--usecase', usecase, '--json']
    ruby_rows = json.loads(subprocess.check_output(command, cwd=ROOT, text=True))
    assert [r['framework'] for r in ruby_rows] == [r['framework'] for r in rows]
    for actual, ruby in zip(rows, ruby_rows):
        assert actual['total'] == pytest.approx(ruby['total'])
        assert actual['breakdown'] == pytest.approx(ruby['breakdown'])
    profile = data['usecases'][usecase]
    weights = profile['weights']
    effective = {k: weights[k] for k in ('state', 'debug', 'ecosystem')}
    effective.update({k: weights['lock-in'] * v for k, v in profile['subweights'].items()})
    top = sorted(effective, key=lambda k: (-effective[k], k))[:2]
    text = profile['justification'].lower()
    assert winner.lower() in text
    assert all(FRAMEWORK_LABELS[key].lower() in text for key in top)


def test_mixed_workload_uses_declared_adapters_and_retains_envelopes():
    calls = []

    def adapter(name):
        async def run(payload, envelope):
            calls.append((name, copy.deepcopy(envelope)))
            assert envelope == ENVELOPE
            return {'ticket': payload['ticket'], 'adapter': name}
        return run

    registry = {name: adapter(name) for name in framework_policy(CONFIG)['frameworks']}

    async def workload():
        for usecase, expected in EXPECTED.items():
            result = await run_framework_task(
                usecase, {'ticket': 'A17'}, ENVELOPE, adapters=registry, path=CONFIG)
            assert result['framework'] == expected
            assert result['result'] == {'ticket': 'A17', 'adapter': expected}
            assert result['envelope'] == ENVELOPE
    asyncio.run(workload())
    assert [name for name, _ in calls] == list(EXPECTED.values())
    print('Mixed A17 workload: escalation=LangGraph; rag=LlamaIndex; triage=LangChain; tenant/trace/budget retained')


@pytest.mark.parametrize('field,value', [('tenant_id', 0), ('trace_id', '')])
def test_invalid_scope_never_dispatches(field, value):
    envelope = copy.deepcopy(ENVELOPE)
    envelope[field] = value
    with pytest.raises(ValueError, match='invalid_framework_scope'):
        asyncio.run(run_framework_task('triage', {}, envelope, adapters={}, path=CONFIG))


@pytest.mark.parametrize('key', ['max_tokens', 'max_steps'])
def test_exhausted_budget_never_dispatches(key):
    envelope = copy.deepcopy(ENVELOPE)
    envelope['budget'][key] = 0
    with pytest.raises(ValueError, match='invalid_framework_budget'):
        asyncio.run(run_framework_task('triage', {}, envelope, adapters={}, path=CONFIG))


def test_missing_adapter_does_not_fall_back():
    with pytest.raises(ValueError, match='framework_adapter_unavailable'):
        asyncio.run(run_framework_task('rag', {}, ENVELOPE, adapters={}, path=CONFIG))


def test_adapter_cannot_rewrite_envelope():
    async def corrupt(payload, envelope):
        envelope['budget']['max_tokens'] += 1
    with pytest.raises(ValueError, match='framework_envelope_mutated'):
        asyncio.run(run_framework_task(
            'triage', {}, ENVELOPE, adapters={'LangChain': corrupt}, path=CONFIG))
    assert ENVELOPE['budget']['max_tokens'] == 256


def test_policy_drift_is_rejected(tmp_path):
    data = framework_policy(CONFIG)
    data['usecases']['triage']['selected'] = 'provider-native'
    policy = tmp_path / 'policy.yml'
    policy.write_text(yaml.safe_dump(data))
    config = tmp_path / 'orchestration.yml'
    config.write_text(yaml.safe_dump({'framework_selection': {'policy': str(policy)}}))
    with pytest.raises(ValueError, match='framework_policy_drift'):
        select_framework('triage', config)


def test_query_adapter_preserves_scope_trace_budget_and_rejects_other_tenant():
    from anchor_ai.query_engine import run_policy_query

    class Tool:
        tenant_id = 7

        async def __call__(self, query, *, envelope):
            assert query == 'Refund policy?'
            assert envelope == ENVELOPE
            return {'answer': 'Needs approval', 'citations': [11]}

    result = asyncio.run(run_policy_query({'query': 'Refund policy?'}, ENVELOPE, tool=Tool()))
    assert result['citations'] == [11]
    other = dict(ENVELOPE, tenant_id=8)
    with pytest.raises(ValueError, match='query_engine_scope_mismatch'):
        asyncio.run(run_policy_query({'query': 'Refund policy?'}, other, tool=Tool()))


def test_chain_adapter_passes_trace_to_existing_chain(monkeypatch):
    from anchor_ai import chains

    async def chain(ticket, **options):
        assert ticket == {'message': 'Refund A17'}
        assert options['trace'] == {'trace_id': ENVELOPE['trace_id']}
        return chains.TicketResponse(reply='Review required', status='needs_review')

    monkeypatch.setattr(chains, 'run_ticket_chain', chain)
    result = asyncio.run(chains.run_policy_triage({'message': 'Refund A17'}, ENVELOPE))
    assert result == {'reply': 'Review required', 'status': 'needs_review'}


def test_legacy_single_agent_behavior_remains_supported():
    assert select_flow('single-lookup', CONFIG)['pattern'] == 'single-agent'
    state = {'strategy': 'react', 'plan': ['retrieve', 'lookup_order'],
             'phase': 1, 'memory': []}
    assert worker_for(state) is None

    class Model:
        async def propose(self, strategy, plan, memory, reserve):
            assert plan == state['plan']
            assert memory == []
            return 'legacy-single-agent'

    assert asyncio.run(propose(Model(), state, 256)) == 'legacy-single-agent'
