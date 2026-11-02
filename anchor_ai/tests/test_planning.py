import asyncio
import copy

import pytest

from anchor_ai.agentic_rag import build_agent_runtime
from anchor_ai.planning import (
    OfflineAccountBackend, PlanExecuteRuntime, PlanningRun, RecordedPlanner,
    fixture, policy, validate_plan,
)
from anchor_ai.schemas import AccountExecutionScope


def scope(role='support_supervisor', tenant_id=1):
    return AccountExecutionScope(tenant_id=tenant_id, conversation_id=1,
                                 customer_id='customer-1', role=role)


def run(scenario='failure', data=None, approvals=True, **kwargs):
    planner, backend = RecordedPlanner(data), OfflineAccountBackend(scenario)
    runtime = PlanExecuteRuntime(tenant_id=1, planner=planner, backend=backend, **kwargs)
    answer = asyncio.run(runtime.answer(fixture()['ticket'], request_id='test', trace_id='test',
        execution_scope=scope(),
        approved_changes=(('customer-1', '10 Harbor Road'),) if approvals else ()))
    return runtime, planner, backend, answer


def test_failed_second_step_never_dispatches_third_and_requests_bounded_replacement():
    runtime, planner, backend, answer = run()
    assert [r['tool'] for r in backend.log] == ['verify_identity', 'update_address', 'report_blocked']
    assert len(planner.requests) == 2
    assert planner.requests[1]['reason'] == {'kind': 'failure'}
    assert runtime.last_run.replans == 1
    assert answer['abstained']
    discard = next(r for r in runtime.last_trajectory if r['event'] == 'discard')
    assert discard['actions'] == ['confirm']


def test_replanning_preserves_spent_budgets_and_completed_action_ids():
    runtime, planner, backend, _ = run()
    ledger = planner.requests[1]['ledger']
    assert ledger['steps'] == 4
    assert ledger['calls'] == 2
    assert ledger['cost_units'] == 6
    assert ledger['planning_calls'] == 2
    assert set(ledger['completed']) == {'verify'}
    assert set(ledger['attempted']) == {'verify', 'update'}
    assert runtime.last_run.cost_units == 7
    assert runtime.last_run.calls == 3
    assert set(runtime.last_run.completed) == {'verify', 'blocked'}
    assert sum(r['tool'] == 'update_address' for r in backend.log) == 1


def test_successful_closed_account_invalidates_remaining_plan():
    runtime, planner, backend, _ = run('closed')
    assert [r['tool'] for r in backend.log] == ['verify_identity', 'report_blocked']
    assert planner.requests[1]['reason']['kind'] == 'invalidating_success'
    assert 'verify' in runtime.last_run.completed


def test_normal_plan_executes_without_revision():
    runtime, planner, backend, answer = run('normal')
    assert [r['tool'] for r in backend.log] == ['verify_identity', 'update_address', 'confirm_change']
    assert len(planner.requests) == 1
    assert not answer['abstained']
    assert runtime.last_run.completed['update']['result']['undo_token']


def test_revision_limit_is_enforced():
    runtime, planner, backend, _ = run(max_replans=0)
    assert len(planner.requests) == 1
    assert len(backend.log) == 2
    assert runtime.last_trajectory[-1]['reason'] == 'revision_limit'


@pytest.mark.parametrize('kind', ['cycle', 'unauthorized', 'unknown_dependency', 'duplicate'])
def test_invalid_initial_plan_is_rejected_before_dispatch(kind):
    data = fixture()
    if kind == 'cycle':
        data['initial']['steps'][0]['requires'] = ['confirm']
    elif kind == 'unauthorized':
        data['initial']['steps'][0]['tool'] = 'issue_refund'
    elif kind == 'unknown_dependency':
        data['initial']['steps'][0]['requires'] = ['missing']
    else:
        data['initial']['steps'][1]['action_id'] = 'verify'
    runtime, _, backend, _ = run(data=data)
    assert backend.log == []
    assert runtime.last_trajectory[-1]['reason'] == 'invalid_plan_or_budget'


def test_replacement_cannot_replay_completed_mutation():
    state = PlanningRun(scope())
    state.completed['update'] = {'tool': 'update_address', 'result': {'ok': True}}
    raw = {'steps': [copy.deepcopy(fixture()['initial']['steps'][1])]}
    with pytest.raises(ValueError, match='action_replay'):
        validate_plan(raw, state, policy())


def test_replacement_cannot_retry_failed_action_id():
    data = fixture()
    data['replacement'] = copy.deepcopy(data['initial'])
    runtime, planner, backend, _ = run(data=data)
    assert len(planner.requests) == 2
    assert len(backend.log) == 2
    assert runtime.last_trajectory[-1]['reason'] == 'invalid_plan_or_budget'


def test_missing_approval_stops_before_mutation():
    runtime, _, backend, _ = run(approvals=False)
    assert [r['tool'] for r in backend.log] == ['verify_identity']
    assert runtime.last_trajectory[-1]['reason'] == 'approval_required'


def test_reader_cannot_obtain_update_by_planning_it():
    with pytest.raises(ValueError, match='unauthorized_tool'):
        validate_plan(fixture()['initial'], PlanningRun(scope('support_reader')), policy())


def test_budget_is_not_reset_to_make_room_for_replacement():
    runtime = PlanExecuteRuntime(tenant_id=1, planner=RecordedPlanner())
    state = PlanningRun(scope(), cost_units=policy()['run']['max_cost_units'] - 1)
    asyncio.run(runtime.answer('Change address', request_id='t', trace_id='t',
                              execution_scope=scope(), execution_run=state))
    assert state.cost_units == policy()['run']['max_cost_units']
    assert state.calls == 0
    assert runtime.backend.log == []


def test_unknown_write_outcome_requires_reconciliation():
    class UncertainBackend(OfflineAccountBackend):
        async def dispatch(self, scope, step, operation_id, *, dry_run=False):
            if step.tool == 'update_address':
                raise TimeoutError('response lost')
            return await super().dispatch(scope, step, operation_id, dry_run=dry_run)
    runtime = PlanExecuteRuntime(tenant_id=1, planner=RecordedPlanner(), backend=UncertainBackend())
    asyncio.run(runtime.answer('Change address', request_id='t', trace_id='t',
        execution_scope=scope(), approved_changes=(('customer-1', '10 Harbor Road'),)))
    assert runtime.last_trajectory[-1]['reason'] == 'outcome_unknown_reconcile'
    assert runtime.last_run.replans == 0
    assert 'update' in runtime.last_run.attempted


def test_dry_run_does_not_supply_completion_evidence():
    runtime = PlanExecuteRuntime(tenant_id=1, planner=RecordedPlanner())
    asyncio.run(runtime.answer('Change address', request_id='t', trace_id='t',
                              execution_scope=scope(), dry_run=True))
    assert not runtime.last_run.completed
    assert runtime.last_trajectory[-1]['reason'] == 'dry_run_no_completion'


def test_selector_adds_planning_without_changing_default_contract():
    runtime = build_agent_runtime(None, strategy='plan_execute', tenant_id=1,
                                  planner=RecordedPlanner())
    assert isinstance(runtime, PlanExecuteRuntime)
