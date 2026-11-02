import asyncio
import copy
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

from anchor_ai.contracts import AgentRequest, VerifiedExecution
from anchor_ai.observability import span

ACTIONS = ['retrieve', 'lookup_order', 'check_refund', 'issue_refund']
TERMINAL = {'completed', 'denied', 'exhausted', 'rejected', 'failed'}


class ContextError(ValueError):
    pass


class BoundedRuntime:
    """Durable workflow. Adapters do not own authorization or checkpoints.

    model.propose(strategy, plan, memory, max_tokens) -> (action, actual_tokens)
    retrieve(scope, message) -> scoped evidence dictionary
    backend.call(scope, action, arguments, key) -> authoritative observation
    Financial backends MUST durably deduplicate key and bind it to arguments.
    """
    def __init__(self, path, *, verify, model, retrieve, backend,
                 policy_path='config/agent_policy.yml', emit=None):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.verify = verify
        self.model = model
        self.retrieve = retrieve
        self.backend = backend
        self.policy = yaml.safe_load(Path(policy_path).read_text())
        self.emit = emit or (lambda event: None)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS bounded_runs ('
                       'tenant INTEGER, conversation INTEGER, run TEXT, state TEXT, '
                       'PRIMARY KEY(tenant, conversation, run))')

    def db(self):
        return sqlite3.connect(self.path, timeout=1)

    @contextmanager
    def locked(self):
        # A separate lock database permits checkpoint commits while serializing
        # competing workers. A process exit releases this SQLite write lock.
        lock = sqlite3.connect(self.path + '.locks', timeout=1)
        try:
            lock.execute('BEGIN IMMEDIATE')
            yield
        finally:
            lock.rollback()
            lock.close()

    def identity(self, token):
        if not token:
            raise ContextError('missing_execution_context')
        scope = self.verify(token)
        if (not isinstance(scope, VerifiedExecution)
                or type(scope.tenant_id) is not int or scope.tenant_id <= 0
                or type(scope.conversation_id) is not int or scope.conversation_id <= 0
                or not scope.principal_id):
            raise ContextError('unverified_execution_context')
        return scope

    def load(self, scope, run_id):
        with self.db() as db:
            row = db.execute('SELECT state FROM bounded_runs WHERE '
                             'tenant=? AND conversation=? AND run=?',
                             (scope.tenant_id, scope.conversation_id, run_id)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, s):
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO bounded_runs VALUES(?,?,?,?)',
                       (s['tenant_id'], s['conversation_id'], s['run_id'], json.dumps(s)))

    def event(self, s, subsystem, **fields):
        event = dict(sequence=len(s['events']) + 1, subsystem=subsystem,
                     trace_id=s['trace_id'], tenant_id=s['tenant_id'],
                     conversation_id=s['conversation_id'], step=s['steps'], **fields)
        s['events'].append(event)
        self.emit(copy.deepcopy(event))

    def stop(self, s, status, reason):
        s['status'] = status
        s['escalation'] = dict(reason=reason, tenant_id=s['tenant_id'],
                              conversation_id=s['conversation_id'],
                              memory=copy.deepcopy(s['memory']),
                              attempts=copy.deepcopy(s['attempts']),
                              proposed_action=s['arguments'], idempotency_key=s['key'],
                              tokens=s['tokens'], steps=s['steps'], calls=s['calls'])
        self.event(s, 'escalation', reason=reason, persisted=True)
        self.save(s)
        return s

    def allowed(self, scope, action, arguments):
        tenant = self.policy['tenants'].get(scope.tenant_id, {})
        names = tenant.get('roles', {}).get(scope.role, [])
        return (action in names and action in self.policy['tools']
                and (action != 'issue_refund' or
                     arguments['amount_cents'] <= tenant.get('refund_ceiling_cents', 0)))

    def review(self, token, run_id, approved):
        scope = self.identity(token)
        if scope.role != 'refund_reviewer':
            raise ContextError('reviewer_required')
        with self.locked():
            s = self.load(scope, run_id)
            if not s:
                raise ContextError('run_not_in_scope')
            if s['status'] != 'awaiting_approval':
                return s
            self.event(s, 'review', approved=approved, reviewer=scope.principal_id)
            if not approved:
                return self.stop(s, 'rejected', 'approval_rejected')
            s['approval'] = dict(reviewer=scope.principal_id, digest=s['digest'])
            s['status'] = 'running'
            self.save(s)
            return s

    async def execute(self, token, request):
        # Identity is checked before loading any run, retrieval, or tool adapter.
        scope = self.identity(token)
        req = AgentRequest.model_validate(request)
        if req.operation in ('approve', 'reject'):
            return self.review(token, req.run_id, req.operation == 'approve')
        with self.locked():
            s = self.load(scope, req.run_id)
            if s is None:
                if req.operation != 'start':
                    raise ContextError('run_not_in_scope')
                if not req.message.strip():
                    raise ValueError('message_required')
                strategy = self.policy['bounded']['strategies'][req.ticket_type]
                args = {'order_id': req.order_id, 'amount_cents': req.amount_cents}
                key = hashlib.sha256(json.dumps(
                    [scope.tenant_id, scope.conversation_id, req.run_id, args],
                    sort_keys=True).encode()).hexdigest()
                s = dict(run_id=req.run_id, tenant_id=scope.tenant_id,
                         conversation_id=scope.conversation_id, trace_id=key[:32],
                         strategy=strategy, message=req.message, arguments=args,
                         key=key, digest=key, status='running', phase=0,
                         plan=ACTIONS[:] if req.ticket_type == 'multi-step refund' else ['retrieve'],
                         steps=0, calls=0, cost_units=0, tokens=0, memory=[],
                         attempts=[], events=[], pending=None, approval=None,
                         caps=dict(self.policy['run']), started=time.time())
                self.event(s, 'planner_selection', strategy=strategy)
                self.save(s)
            elif req.operation == 'start':
                # Repeating start cannot replace an existing run's arguments.
                if (req.message != s['message'] or req.order_id != s['arguments']['order_id']
                        or req.amount_cents != s['arguments']['amount_cents']):
                    raise ValueError('run_arguments_changed')
            if s['status'] in TERMINAL or s['status'] == 'awaiting_approval':
                return s
            self.event(s, 'memory_read', records=len(s['memory']))
            self.save(s)
            ctx = {'trace_id': s['trace_id'], 'request_id': s['run_id'],
                   'parent_span_id': None}
            with span('python.bounded_agent', ctx=ctx, stage='agent',
                      tenant_id=scope.tenant_id, conversation_id=scope.conversation_id):
                return await self.loop(scope, s)

    async def loop(self, scope, s):
        while s['phase'] < len(s['plan']):
            if time.time() - s['started'] >= s['caps']['deadline_seconds']:
                return self.stop(s, 'exhausted', 'deadline')
            if s['pending'] is None:
                reserve = self.model.reservation
                permitted = (s['steps'] < s['caps']['max_steps']
                             and s['tokens'] + reserve <= s['caps']['max_tokens'])
                self.event(s, 'budget_check', permitted=permitted,
                           steps=s['steps'], tokens=s['tokens'], reserve=reserve)
                if not permitted:
                    return self.stop(s, 'exhausted', 'step_or_token_budget')
                s['steps'] += 1
                s['tokens'] += reserve
                self.save(s)  # Reservation survives provider errors and restarts.
                try:
                    with span('python.bounded_agent.model', stage='llm'):
                        action, used = await asyncio.wait_for(self.model.propose(
                            s['strategy'], s['plan'], copy.deepcopy(s['memory']), reserve),
                            timeout=self.policy['bounded']['call_timeout_seconds'])
                    if type(used) is not int or not 0 <= used <= reserve:
                        raise ValueError('invalid_usage')
                    s['tokens'] -= reserve - used
                    self.event(s, 'planner', action=action, charged=used)
                except Exception:
                    return self.stop(s, 'failed', 'model_error')
                allowed = self.allowed(scope, action, s['arguments'])
                # A planner may not skip evidence, eligibility, or change the plan.
                allowed = allowed and action == s['plan'][s['phase']]
                self.event(s, 'guardrail', action=action, allowed=allowed)
                attempt = dict(action=action, allowed=allowed, result='proposed')
                s['attempts'].append(attempt)
                if not allowed:
                    self.event(s, 'memory_write', outcome='denied')
                    return self.stop(s, 'denied', 'unauthorized_action')
                s['pending'] = dict(action=action, reserved=False)
                self.save(s)
            action = s['pending']['action']
            # Re-authorize on resume; an approval never grants a tool permission.
            if not self.allowed(scope, action, s['arguments']):
                return self.stop(s, 'denied', 'authorization_changed')
            if action == 'issue_refund':
                if not s['approval']:
                    return self.stop(s, 'awaiting_approval', 'approval_required')
                if s['approval']['digest'] != s['digest']:
                    return self.stop(s, 'denied', 'approval_scope_mismatch')
            if not s['pending']['reserved']:
                cost = self.policy['tools'][action]['cost_units']
                permitted = (s['calls'] < s['caps']['max_calls']
                             and s['cost_units'] + cost <= s['caps']['max_cost_units'])
                self.event(s, 'dispatch_budget', permitted=permitted, action=action)
                if not permitted:
                    return self.stop(s, 'exhausted', 'call_or_cost_budget')
                s['calls'] += 1
                s['cost_units'] += cost
                s['pending']['reserved'] = True
                self.save(s)  # Retrying a pending dispatch does not reserve twice.
            try:
                with span('python.bounded_agent.tool', stage='tool', tool_name=action):
                    if action == 'retrieve':
                        observation = await asyncio.wait_for(
                            self.retrieve(scope, s['message']),
                            timeout=self.policy['bounded']['call_timeout_seconds'])
                        if (observation['tenant_id'] != scope.tenant_id
                                or any(h.get('tenant_id', scope.tenant_id) != scope.tenant_id
                                       for h in observation['hits'])):
                            raise ValueError('tenant_memory_leak')
                    else:
                        observation = await asyncio.wait_for(self.backend.call(
                            scope, action, s['arguments'], s['key']),
                            timeout=self.policy['bounded']['call_timeout_seconds'])
                        if observation.get('order_id') != s['arguments']['order_id']:
                            raise ValueError('receipt_scope_mismatch')
                        if action == 'check_refund' and observation.get('eligible') is not True:
                            raise ValueError('refund_ineligible')
                        if action == 'issue_refund' and (
                                observation.get('status') != 'refunded'
                                or observation.get('idempotency_key') != s['key']
                                or observation.get('amount_cents') != s['arguments']['amount_cents']):
                            raise ValueError('invalid_refund_receipt')
            except Exception:
                s['attempts'][-1]['result'] = 'error'
                self.event(s, 'memory_write', outcome='error')
                return self.stop(s, 'failed', 'tool_error')
            self.event(s, 'tool_dispatch', action=action, observation=observation,
                       idempotency_key=s['key'] if action == 'issue_refund' else None)
            s['attempts'][-1]['result'] = 'success'
            s['memory'].append(dict(action=action, observation=observation))
            self.event(s, 'memory_write', outcome='success')
            self.event(s, 'escalation_trigger', reason=None)
            s['phase'] += 1
            s['pending'] = None
            self.save(s)
        s['status'] = 'completed'
        self.event(s, 'complete', calls=s['calls'], tokens=s['tokens'])
        self.save(s)
        return s
