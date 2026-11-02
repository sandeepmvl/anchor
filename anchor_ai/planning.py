import argparse
import asyncio
import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import yaml

from anchor_ai.schemas import AccountExecutionScope, AccountObservation, AccountPlan
from anchor_ai.context import agent_memory_prompt

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / 'tests/fixtures/account_change_plan.json'
ACCOUNT_TOOLS = frozenset(('verify_identity', 'update_address', 'confirm_change', 'report_blocked'))
REVERSIBLE_TOOLS = frozenset(('update_address',))
MAX_PLAN_STEPS = 6
MAX_REPLANS = 1
PLANNER_TIMEOUT_SECONDS = 2
BACKEND_TIMEOUT_SECONDS = 2


def policy():
    return yaml.safe_load((ROOT / 'config/agent_policy.yml').read_text())


def fixture():
    return json.loads(FIXTURE.read_text())


@dataclass
class PlanningRun:
    scope: AccountExecutionScope
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started: float = field(default_factory=time.monotonic)
    steps: int = 0
    calls: int = 0
    cost_units: int = 0
    planning_calls: int = 0
    replans: int = 0
    completed: dict = field(default_factory=dict)
    attempted: dict = field(default_factory=dict)
    halted: bool = False

    def snapshot(self):
        return copy.deepcopy({
            'steps': self.steps, 'calls': self.calls, 'cost_units': self.cost_units,
            'planning_calls': self.planning_calls, 'replans': self.replans,
            'completed': self.completed, 'attempted': self.attempted,
        })


class RecordedPlanner:
    def __init__(self, data=None):
        self.data = data or fixture()
        self.requests = []

    async def plan(self, ticket, ledger, reason=None):
        self.requests.append(copy.deepcopy({'ledger': ledger, 'reason': reason}))
        return copy.deepcopy(self.data['replacement' if reason else 'initial'])


class LivePlanner:
    def __init__(self):
        from anchor_ai.llm_client import LlmClient
        self.client = LlmClient()

    async def plan(self, ticket, ledger, reason=None):
        raw = await self.client.complete(json.dumps({
            'ticket': ticket, 'ledger': ledger, 'reason': reason,
            'schema': AccountPlan.model_json_schema(),
            'allowed_tools': sorted(ACCOUNT_TOOLS),
        }), system='Return only an account plan. Never replay attempted actions.',
            max_tokens=1024)
        return json.loads(raw)


class OfflineAccountBackend:
    """Backend double. A real adapter must persist operation IDs and receipts."""
    def __init__(self, scenario='failure'):
        if scenario not in ('failure', 'closed', 'normal'):
            raise ValueError('invalid_scenario')
        self.scenario = scenario
        self.log = []
        self.receipts = {}

    async def dispatch(self, scope, step, operation_id, *, dry_run=False):
        if operation_id in self.receipts and not dry_run:
            return self.receipts[operation_id]
        self.log.append({'tool': step.tool, 'action_id': step.action_id,
                         'tenant_id': scope.tenant_id, 'operation_id': operation_id,
                         'dry_run': dry_run})
        if dry_run:
            return {'ok': True}
        if step.tool == 'verify_identity':
            result = {'ok': True, 'account': 'closed' if self.scenario == 'closed' else 'open'}
        elif step.tool == 'update_address':
            result = ({'ok': False, 'reason': 'write_rejected_no_change'}
                      if self.scenario == 'failure' else
                      {'ok': True, 'receipt': operation_id, 'undo_token': 'undo:' + operation_id})
        else:
            result = {'ok': True}
        self.receipts[operation_id] = result
        return result


def validate_plan(raw, state, config):
    plan = AccountPlan.model_validate(raw)
    if len(plan.steps) > MAX_PLAN_STEPS:
        raise ValueError('plan_size')
    tenant = config['tenants'].get(state.scope.tenant_id, {})
    allowed = tenant.get('roles', {}).get(state.scope.role, [])
    nodes = {s.action_id: s for s in plan.steps}
    if len(nodes) != len(plan.steps):
        raise ValueError('duplicate_action_id')
    for step in plan.steps:
        if step.tool not in ACCOUNT_TOOLS or step.tool not in allowed or step.tool not in config['tools']:
            raise ValueError('unauthorized_tool')
        if step.action_id in state.attempted or step.action_id in state.completed:
            raise ValueError('action_replay')
        if any(dep not in nodes and dep not in state.completed for dep in step.requires):
            raise ValueError('unknown_prerequisite')
    visiting, visited, ordered = set(), set(), []

    def visit(action_id):
        if action_id in visiting:
            raise ValueError('cyclic_plan')
        if action_id in visited or action_id in state.completed:
            return
        visiting.add(action_id)
        for dep in nodes[action_id].requires:
            visit(dep)
        visiting.remove(action_id)
        visited.add(action_id)
        ordered.append(nodes[action_id])

    for action_id in nodes:
        visit(action_id)
    return ordered


class PlanExecuteRuntime:
    def __init__(self, retriever=None, *, planner=None, backend=None,
                 tenant_id=None, max_replans=None, memory=None):
        self.memory = memory
        self.config = policy()
        if planner is None:
            mode = os.getenv('ANCHOR_PLANNER_MODE', 'offline')
            if mode not in ('offline', 'live'):
                raise ValueError('invalid_planner_mode')
            planner = RecordedPlanner() if mode == 'offline' else LivePlanner()
        self.planner = planner
        self.backend = backend or OfflineAccountBackend()
        self.tenant_id = tenant_id if tenant_id is not None else getattr(retriever, 'tenant_id', None)
        if type(self.tenant_id) is not int or self.tenant_id <= 0:
            raise ValueError('invalid_trusted_tenant')
        if max_replans is not None and (type(max_replans) is not int or max_replans < 0):
            raise ValueError('invalid_revision_limit')
        self.max_replans = MAX_REPLANS if max_replans is None else min(MAX_REPLANS, max_replans)
        self.last_trajectory = []
        self.last_run = None

    def reserve(self, state, *, tool=None):
        limits = self.config['run']
        if time.monotonic() - state.started >= limits['deadline_seconds']:
            raise ValueError('deadline')
        cost = self.config['tools'][tool]['cost_units'] if tool else 1
        if state.steps >= limits['max_steps']:
            raise ValueError('step_budget')
        if state.cost_units + cost > limits['max_cost_units']:
            raise ValueError('cost_budget')
        if tool and state.calls >= limits['max_calls']:
            raise ValueError('call_budget')
        state.steps += 1
        state.cost_units += cost
        if tool:
            state.calls += 1
        else:
            state.planning_calls += 1

    async def request_plan(self, ticket, state, reason=None):
        self.reserve(state)
        ticket = agent_memory_prompt(ticket, self.memory, state.scope)
        raw = await asyncio.wait_for(
            self.planner.plan(ticket, state.snapshot(), reason), PLANNER_TIMEOUT_SECONDS)
        return validate_plan(raw, state, self.config)

    @staticmethod
    def evidence(step, state):
        results = [state.completed[d] for d in step.requires if d in state.completed]
        if len(results) != len(step.requires) or any(not r['result']['ok'] for r in results):
            return False
        if step.tool == 'update_address':
            return any(r['tool'] == 'verify_identity' and
                       r['result'].get('account') == 'open' for r in results)
        if step.tool == 'confirm_change':
            return any(r['tool'] == 'update_address' and
                       r['result'].get('receipt') and r['result'].get('undo_token')
                       for r in results)
        return True

    async def answer(self, question, k=3, *, request_id, trace_id,
                     execution_scope, execution_run=None, approved_changes=(),
                     dry_run=False):
        scope = AccountExecutionScope.model_validate(execution_scope)
        if scope.tenant_id != self.tenant_id:
            raise ValueError('agent_scope_mismatch')
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > 1800:
            raise ValueError('current_question_budget_exceeded')
        state = execution_run or PlanningRun(scope)
        if state.scope != scope or state.halted:
            raise ValueError('invalid_execution_run')
        if self.memory is not None:
            self.memory.add_working(scope.tenant_id, scope.conversation_id, question)
        self.last_run = state
        self.last_trajectory = []
        approvals = frozenset(approved_changes)

        def record(event, **fields):
            self.last_trajectory.append({'event': event, 'request_id': request_id,
                                         'trace_id': trace_id, **fields})

        def finish(reason):
            state.halted = True
            if self.memory is not None:
                self.memory.log_episode(scope.tenant_id, scope.conversation_id,
                                        'planning exit=' + reason + ' tool_calls=' + str(state.calls))
            record('exit', reason=reason, ledger=state.snapshot())
            confirmed = any(r['tool'] == 'confirm_change' for r in state.completed.values())
            return {'reply': 'Address changed.' if confirmed else 'Address change blocked or unconfirmed.',
                    'citations': [], 'abstained': not confirmed}

        try:
            pending = await self.request_plan(question, state)
            record('plan', actions=[s.action_id for s in pending])
            while pending:
                step = pending.pop(0)
                reason = None
                if not self.evidence(step, state):
                    reason = 'invalid_prerequisite'
                else:
                    tool_policy = self.config['tools'][step.tool]
                    if tool_policy['mode'] == 'approval':
                        approval = (scope.customer_id, step.address)
                        if not dry_run and approval not in approvals:
                            return finish('approval_required')
                        if step.tool not in REVERSIBLE_TOOLS:
                            return finish('non_reversible_action')
                    operation_id = ':'.join((str(scope.tenant_id), scope.customer_id,
                                             state.run_id, step.action_id))
                    self.reserve(state, tool=step.tool)
                    state.attempted[step.action_id] = {'tool': step.tool, 'operation_id': operation_id}
                    record('dispatch', action_id=step.action_id, tool=step.tool)
                    try:
                        raw = await asyncio.wait_for(self.backend.dispatch(
                            scope, step, operation_id, dry_run=dry_run), BACKEND_TIMEOUT_SECONDS)
                        result = AccountObservation.model_validate(raw).model_dump()
                    except Exception:
                        return finish('outcome_unknown_reconcile')
                    if dry_run:
                        return finish('dry_run_no_completion')
                    record('observation', action_id=step.action_id, result=result)
                    if not result['ok']:
                        reason = 'failure'
                    elif step.tool == 'update_address' and not (result['receipt'] and result['undo_token']):
                        return finish('outcome_unknown_reconcile')
                    else:
                        state.completed[step.action_id] = {'tool': step.tool, 'result': result}
                        if step.tool == 'verify_identity' and result['account'] != 'open':
                            reason = 'invalidating_success'
                if reason:
                    record('discard', reason=reason, actions=[s.action_id for s in pending])
                    pending = []
                    if state.replans >= self.max_replans:
                        return finish('revision_limit')
                    state.replans += 1
                    pending = await self.request_plan(question, state, {'kind': reason})
                    record('replacement', actions=[s.action_id for s in pending], ledger=state.snapshot())
            return finish('complete')
        except (ValueError, TypeError, KeyError, asyncio.TimeoutError):
            return finish('invalid_plan_or_budget')


def demo(stage):
    data = fixture()
    if stage == 'inspect':
        print(json.dumps({'plan': data['initial'],
                          'update_policy': policy()['tools']['update_address'],
                          'reversible_tools': sorted(REVERSIBLE_TOOLS)}, indent=2))
        return
    backend, planner = OfflineAccountBackend(), RecordedPlanner()
    runtime = PlanExecuteRuntime(tenant_id=1, backend=backend, planner=planner,
                                 max_replans=1 if stage == 'revise' else 0)
    scope = AccountExecutionScope(tenant_id=1, conversation_id=1,
                                  customer_id='customer-1', role='support_supervisor')
    answer = asyncio.run(runtime.answer(data['ticket'], request_id='demo', trace_id='demo',
        execution_scope=scope, approved_changes=(('customer-1', '10 Harbor Road'),)))
    assert 'confirm_change' not in [row['tool'] for row in backend.log]
    print('Anchor stopped at the failed prerequisite; dependent confirm_change was never dispatched.')
    print(json.dumps({'dispatch_log': backend.log, 'trajectory': runtime.last_trajectory,
                      'ledger': runtime.last_run.snapshot(), 'answer': answer}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['inspect', 'execute', 'log', 'revise'], default='revise')
    demo(parser.parse_args().stage)
