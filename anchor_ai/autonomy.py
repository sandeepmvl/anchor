import asyncio
import hashlib
import json
import secrets
import time
from pathlib import Path

import yaml
from pydantic import ValidationError

from .observability import span


def load_policy():
    path = Path(__file__).resolve().parents[1] / 'config/agent_policy.yml'
    policy = yaml.safe_load(path.read_text())
    if policy['version'] != 1:
        raise ValueError('unsupported_agent_policy')
    return policy


def fingerprint(scope, name, arguments):
    value = [scope.tenant_id, scope.role, name, arguments]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class ExecutionRun:
    """Trusted controller state. Share this object, never a copy of its counters."""
    def __init__(self, scope, *, max_calls=None, max_cost_units=None,
                 max_steps=None, dry_run=False, clock=time.monotonic):
        self.policy = load_policy()
        config = self.policy['run']
        self.identity = (scope.tenant_id, scope.conversation_id, scope.role)
        def limit(value, name):
            if value is None:
                return config[name]
            if type(value) is not int or value < 0:
                raise ValueError('invalid_agent_budget')
            return min(value, config[name])
        self.max_calls = limit(max_calls, 'max_calls')
        self.max_cost = limit(max_cost_units, 'max_cost_units')
        self.max_steps = limit(max_steps, 'max_steps')
        self.clock = clock
        self.deadline = clock() + config['deadline_seconds']
        self.dry_run = dry_run
        self.calls = self.cost = self.steps = 0
        self.halted = False
        self.lock = asyncio.Lock()
        self.confirmations = {}
        self.approvals = {}
        self.events = []

    def matches(self, scope):
        return self.identity == (scope.tenant_id, scope.conversation_id, scope.role)

    def confirm(self, scope, call, arguments):
        """Called by trusted approval handling after showing the exact proposal."""
        if not self.matches(scope) or self.halted:
            raise ValueError('invalid_confirmation_context')
        token = secrets.token_urlsafe(24)
        self.confirmations[token] = fingerprint(scope, call.name, arguments)
        return token

    def approve(self, scope, call, arguments, *, confirmation, idempotency_key):
        """Attach trusted approval, not fields returned by an LLM."""
        if not self.matches(scope):
            raise ValueError('invalid_confirmation_context')
        bound = fingerprint(scope, call.name, arguments)
        if (self.confirmations.get(confirmation) != bound
                or not isinstance(idempotency_key, str)
                or not 8 <= len(idempotency_key) <= 200):
            raise ValueError('invalid_confirmation')
        self.approvals[bound] = (confirmation, idempotency_key)

    def denied(self, scope, reason):
        self.halted = True
        event = {'event': 'policy_denied', 'reason': reason,
                 'tenant_id': scope.tenant_id, 'calls': self.calls,
                 'reserved_cost_units': self.cost}
        self.events.append(event)
        with span('python.autonomy', stage='tool', tenant_id=scope.tenant_id,
                  tool_calls=self.calls, agent_exit=reason) as operation:
            operation.fail(reason)
        return reason, None

    async def step(self, scope):
        async with self.lock:
            if not self.matches(scope):
                return self.denied(scope, 'scope_denied')[0]
            if self.halted:
                return 'run_halted'
            if self.clock() >= self.deadline:
                return self.denied(scope, 'deadline')[0]
            if self.steps >= self.max_steps:
                return self.denied(scope, 'step_budget')[0]
            self.steps += 1
        return None

    async def admit(self, scope, name, arguments, *, owns=True):
        async with self.lock:
            if not self.matches(scope):
                return self.denied(scope, 'scope_denied')
            if self.halted:
                return 'run_halted', None
            if self.clock() >= self.deadline:
                return self.denied(scope, 'deadline')
            if self.calls >= self.max_calls:
                return self.denied(scope, 'call_budget')
            self.calls += 1
            tenant = self.policy['tenants'].get(scope.tenant_id, {})
            allowed = tenant.get('roles', {}).get(scope.role, [])
            spec = self.policy['tools'].get(name)
            if name not in allowed or spec is None:
                return self.denied(scope, 'policy_denied')
            if not owns:
                return self.denied(scope, 'argument_policy')
            key = None
            if spec['mode'] == 'approval':
                amount = arguments.get('amount_cents')
                if type(amount) is not int or not 0 < amount <= tenant['refund_ceiling_cents']:
                    return self.denied(scope, 'argument_policy')
                if not self.dry_run:
                    bound = fingerprint(scope, name, arguments)
                    grant = self.approvals.get(bound)
                    if not grant or self.confirmations.get(grant[0]) != bound:
                        return self.denied(scope, 'confirmation_required')
                    key = grant[1]
            estimated = spec['cost_units']
            if self.cost + estimated > self.max_cost:
                return self.denied(scope, 'cost_budget')
            self.cost += estimated
            self.events.append({'event': 'policy_allowed', 'calls': self.calls,
                                'reserved_cost_units': self.cost})
            with span('python.autonomy', stage='tool',
                      tenant_id=scope.tenant_id, tool_calls=self.calls):
                pass
            return None, key


def run_for(scope, run=None):
    existing = scope._execution.get('run')
    if existing is not None and run is not None and existing is not run:
        raise ValueError('execution_run_replacement_forbidden')
    if existing is None:
        existing = run or ExecutionRun(scope)
        if not existing.matches(scope):
            raise ValueError('execution_scope_mismatch')
        scope._execution['run'] = existing
    return existing


async def guarded_dispatch(call, registry, scope, execute, run=None):
    state = run_for(scope, run)
    tool = registry.get(call.name)
    # Unknown/unlisted names are rejected without parsing or echoing their input.
    tenant = state.policy['tenants'].get(scope.tenant_id, {})
    if call.name not in tenant.get('roles', {}).get(scope.role, []) or tool is None:
        error, _ = await state.admit(scope, call.name, {})
        return {'call_id': call.call_id, 'payload': {'ok': False, 'error': error}}
    try:
        raw = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
        arguments = tool.arguments.model_validate(raw).model_dump()
    except (ValueError, TypeError, ValidationError):
        async with state.lock:
            error, _ = state.denied(scope, 'invalid_arguments')
        return {'call_id': call.call_id, 'payload': {'ok': False, 'error': error}}
    owns = tool.authorize(scope, arguments) if tool.authorize else True
    error, key = await state.admit(scope, call.name, arguments, owns=owns)
    if error:
        return {'call_id': call.call_id, 'payload': {'ok': False, 'error': error}}
    if tool.consequential and state.dry_run:
        return {'call_id': call.call_id, 'payload': {'ok': True, 'data': {
            'dry_run': True, 'proposed_action': call.name, 'arguments': arguments}}}
    result = await execute(call, registry, scope, idempotency_key=key)
    if not result['payload']['ok']:
        state.halted = True
    return result
