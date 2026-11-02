import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anchor_ai.runtime import ContextError

spec = importlib.util.spec_from_file_location('bounded_scenario', 'bin/run_agent_scenario.py')
scenario = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scenario)


def ticket(**changes):
    value = json.loads(Path('anchor_ai/tests/fixtures/refund_ticket.json').read_text())
    return dict(value, **changes)


def run(runtime, token='worker', **changes):
    return asyncio.run(runtime.execute(token, ticket(**changes)))


def test_pause_restart_restores_memory_budgets_and_single_refund(tmp_path):
    runtime, backend = scenario.make_runtime(tmp_path)
    paused = run(runtime)
    assert paused['status'] == 'awaiting_approval'
    assert backend.count() == 0
    assert len(paused['memory']) == 3
    assert paused['escalation']['attempts'][-1]['action'] == 'issue_refund'
    restored, backend = scenario.make_runtime(tmp_path)
    restored.review('reviewer', 'refund-104', True)
    done = run(restored, operation='resume')
    assert done['status'] == 'completed'
    assert done['memory'][:3] == paused['memory']
    assert done['key'] == paused['key']
    assert done['steps'] == paused['steps'] == 4
    assert done['tokens'] == paused['tokens'] == 32
    assert done['calls'] == paused['calls'] + 1 == 4
    assert backend.count() == 1
    assert run(restored, operation='resume')['status'] == 'completed'
    assert backend.count() == 1


class ProcessDied(BaseException):
    pass


def test_crash_after_backend_commit_reuses_pending_key_and_budget(tmp_path):
    runtime, backend = scenario.make_runtime(tmp_path)
    run(runtime)
    runtime.review('reviewer', 'refund-104', True)
    original = backend.call

    async def crash(scope, action, args, key):
        await original(scope, action, args, key)
        raise ProcessDied()

    backend.call = crash
    with pytest.raises(ProcessDied):
        run(runtime, operation='resume')
    assert backend.count() == 1
    restarted, backend = scenario.make_runtime(tmp_path)
    done = run(restarted, operation='resume')
    assert done['status'] == 'completed'
    assert done['calls'] == 4
    assert done['tokens'] == 32
    assert backend.count() == 1


@pytest.mark.parametrize('token', [None, '', 'forged'])
def test_context_rejected_before_state_or_adapters(tmp_path, token):
    runtime, backend = scenario.make_runtime(tmp_path)
    runtime.load = lambda *args: pytest.fail('state accessed before identity verification')
    runtime.retrieve = lambda *args: pytest.fail('retrieval accessed before identity verification')
    with pytest.raises(ContextError):
        run(runtime, token=token)
    assert backend.count() == 0


def test_other_tenant_cannot_restore_memory_or_approve(tmp_path):
    runtime, backend = scenario.make_runtime(tmp_path)
    run(runtime)
    with pytest.raises(ContextError):
        run(runtime, token='other', operation='resume')
    with pytest.raises(ContextError):
        runtime.review('reader', 'refund-104', True)
    assert backend.count() == 0


@pytest.mark.parametrize('trigger', ['evidence', 'budget'])
def test_staff_terminal_escalation_contract(tmp_path, trigger):
    # Exercise each configured planner/orchestration selection, not fake path labels.
    probe, _ = scenario.make_runtime(tmp_path / 'probe')
    for index, ticket_type in enumerate(probe.policy['bounded']['strategies']):
        directory = tmp_path / str(index)
        runtime, backend = scenario.make_runtime(directory)
        sends = []
        original = backend.call

        async def tracked(scope, action, arguments, key):
            sends.append(action)
            return await original(scope, action, arguments, key)

        async def empty(scope, message):
            sends.append('retrieve')
            return {'tenant_id': scope.tenant_id, 'hits': []}

        backend.call = tracked
        runtime.retrieve = empty
        if trigger == 'budget':
            runtime.policy['run']['max_steps'] = 0
        result = run(runtime, ticket_type=ticket_type)
        assert result['status'] == 'exhausted'
        handoff = result['escalation']
        assert handoff['human_handoff'] is True
        assert handoff['do_not_retry'] is True
        assert handoff['tool_calls'] == []
        assert handoff['reason'] == ('evidence_exhausted' if trigger == 'evidence'
                                      else 'step_or_token_budget')
        context = handoff['context']
        assert context['tenant_id'] == result['tenant_id']
        assert context['conversation_id'] == result['conversation_id']
        assert context['messages'] == [result['message']]
        assert context['evidence'] == result['memory']
        assert context['attempts'] == result['attempts']
        assert context['plan'] == result['plan']
        assert context['plan_position'] == result['phase']
        assert context['pending_action'] == result['pending']
        assert context['orchestration'] == result['orchestration']
        assert context['trace_id'] == result['trace_id']
        assert set(context['remaining_budget']) == {'steps', 'tokens', 'calls', 'cost_units'}
        assert sends == (['retrieve'] if trigger == 'evidence' else [])
        before = list(sends)
        assert run(runtime, ticket_type=ticket_type) == result
        assert run(runtime, operation='resume', ticket_type=ticket_type) == result
        assert runtime.review('reviewer', result['run_id'], True) == result
        # Direct loop entry models a downstream runtime caller with the same checkpoint.
        assert asyncio.run(runtime.loop(runtime.identity('worker'), result)) == result
        with pytest.raises(ContextError):
            run(runtime, token='other', operation='resume', ticket_type=ticket_type)
        restored, restored_backend = scenario.make_runtime(directory)
        async def forbidden(*args, **kwargs):
            pytest.fail('adapter called after terminal escalation')
        restored.retrieve = forbidden
        restored.backend.call = forbidden
        restored.model.propose = forbidden
        assert run(restored, operation='resume', ticket_type=ticket_type) == result
        assert sends == before
        assert backend.count() == restored_backend.count() == 0
        print(json.dumps({'demo': 'Unresolved ticket stops safely and transfers complete context to a human',
                          'trigger': trigger, 'ticket_type': ticket_type,
                          'trace_id': result['trace_id'], 'events': result['events'],
                          'subsequent_tool_actions': 0}))


def test_http_boundary_rejects_missing_or_unverified_context(tmp_path, monkeypatch):
    from anchor_ai.app import app
    runtime, backend = scenario.make_runtime(tmp_path)
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'test-secret')
    monkeypatch.setattr(app.state, 'bounded_agent_runtime', runtime, raising=False)
    # No lifespan startup is needed to exercise this isolated route.
    with TestClient(app, raise_server_exceptions=True) as client:
        headers = {'X-Anchor-Secret': 'test-secret'}
        assert client.post('/v1/agent', json=ticket(), headers=headers).status_code == 403
        headers['X-Anchor-Context'] = 'forged'
        assert client.post('/v1/agent', json=ticket(), headers=headers).status_code == 403
        headers['X-Anchor-Context'] = 'worker'
        assert client.post('/v1/agent', json=ticket(), headers=headers).status_code == 200
    assert backend.count() == 0
