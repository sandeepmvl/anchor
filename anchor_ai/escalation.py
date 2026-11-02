import hashlib
import json
import re
import time
from contextlib import contextmanager

from .contracts import ReviewIdentity, ToolCall, ToolScope
from .pii import prepare
from .storage import connect_escalations


def redact(value):
    """Apply existing PII preparation to every text leaf, before serialization."""
    if isinstance(value, str):
        value = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[REDACTED_EMAIL]', value)
        return prepare(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {redact(str(key)): redact(item) for key, item in value.items()}
    if value is None or type(value) in (int, float, bool):
        return value
    raise ValueError('unsupported_checkpoint_value')


def context_bundle(messages, evidence, plan_position, remaining_budget, summary):
    return redact({'messages': messages, 'evidence': evidence,
                   'plan_position': plan_position, 'remaining_budget': remaining_budget,
                   'decision_summary': summary})


def risk_rule(amount_cents, automatic_limit_cents, policy_match):
    if type(amount_cents) is not int or amount_cents <= 0:
        raise ValueError('invalid_amount')
    if type(automatic_limit_cents) is not int or automatic_limit_cents < 0:
        raise ValueError('invalid_limit')
    if type(policy_match) is not bool:
        raise ValueError('invalid_policy_match')
    amount = 2 if amount_cents > automatic_limit_cents else 0
    policy = 0 if policy_match else 2
    return {'amount_tier': amount, 'policy_tier': policy, 'score': amount + policy,
            'threshold': 1, 'rule_version': 'refund-review-v1',
            'tenant_limit_cents': automatic_limit_cents,
            'policy_reason': 'over_automatic_limit' if amount else 'policy_conflict' if policy else 'matched',
            'escalate': amount + policy > 1}


@contextmanager
def transaction(path):
    db = connect_escalations(path)
    try:
        db.execute('BEGIN IMMEDIATE')
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def authorized_row(db, checkpoint_id, identity):
    identity = ReviewIdentity.model_validate(identity)
    if identity.role != 'refund_reviewer':
        raise PermissionError('review_denied')
    row = db.execute('SELECT * FROM escalations WHERE id=? AND tenant_id=?',
                     (checkpoint_id, identity.tenant_id)).fetchone()
    if row is None:
        raise PermissionError('review_denied')
    return row, identity


def public_record(row):
    return {'checkpoint_id': row['id'], 'status': row['status'],
            'action_key': row['action_key'],
            'result': json.loads(row['result_json']) if row['result_json'] else None,
            'outcome': json.loads(row['final_json']) if row['final_json'] else None}


class EscalationService:
    def __init__(self, path):
        self.path = path
        db = connect_escalations(path)
        db.close()

    def inspect(self, checkpoint_id, identity):
        with transaction(self.path) as db:
            row, _ = authorized_row(db, checkpoint_id, identity)
            return {**public_record(row), 'context': json.loads(row['context_json'])}

    def pause(self, scope, state, call, arguments, risk):
        # Configured only by trusted application code, never tool arguments.
        context = state.review_context
        if not context or not context.get('evidence') or not context.get('messages'):
            raise ValueError('incomplete_review_context')
        key_material = [scope.tenant_id, scope.conversation_id, call.call_id,
                        call.name, arguments]
        key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()
        remaining = {'calls': state.max_calls - state.calls,
                     'cost_units': state.max_cost - state.cost,
                     'steps': state.max_steps - state.steps}
        payload = redact({
            'scope': scope.model_dump(),
            'pending_tool_call': {'call_id': call.call_id, 'name': call.name, 'arguments': arguments},
            'risk': risk, 'loop': context, 'remaining_budget': remaining,
            'run': {'max_calls': state.max_calls, 'max_cost_units': state.max_cost,
                    'max_steps': state.max_steps, 'calls': state.calls, 'cost': state.cost,
                    'steps': state.steps, 'dry_run': state.dry_run,
                    'expires_at': time.time() + max(0, state.deadline - state.clock())},
            'idempotency_key': key,
        })
        # Never redact an executable argument into a different action.
        if payload['pending_tool_call']['arguments'] != arguments:
            raise ValueError('sensitive_action_requires_secure_reference')
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with transaction(self.path) as db:
            now = time.time()
            db.execute('''INSERT INTO escalations
                (tenant_id,conversation_id,action_key,status,context_json,created_at,updated_at)
                VALUES(?,?,?,'pending',?,?,?) ON CONFLICT DO NOTHING''',
                (scope.tenant_id, scope.conversation_id, key, encoded, now, now))
            row = db.execute('''SELECT * FROM escalations
                WHERE tenant_id=? AND conversation_id=? AND action_key=?''',
                (scope.tenant_id, scope.conversation_id, key)).fetchone()
            saved = json.loads(row['context_json'])
            if saved['pending_tool_call'] != payload['pending_tool_call']:
                raise ValueError('checkpoint_action_conflict')
            result = public_record(row)
        state.paused = result['checkpoint_id']
        return {'call_id': call.call_id, 'payload': {'ok': False, 'error': 'review_required', **result}}

    async def resolve(self, checkpoint_id, identity, decision, *, reason='', registry=None, execute=None):
        if decision not in ('approve', 'reject'):
            raise ValueError('invalid_review_decision')
        with transaction(self.path) as db:
            row, reviewer = authorized_row(db, checkpoint_id, identity)
            if row['status'] not in ('pending', 'approved'):
                return {**public_record(row), 'deduplicated': True}
            if row['status'] == 'approved' and decision == 'reject':
                return {**public_record(row), 'deduplicated': True}
            payload = json.loads(row['context_json'])
            if decision == 'reject':
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError('rejection_reason_required')
                reason = redact(reason)
                observation = {'ok': False, 'error': 'review_rejected',
                               'reason': reason, 'do_not_retry': True}
                from .react import resume_review_observation
                final = resume_review_observation(payload, observation)
                db.execute('''UPDATE escalations SET status='rejected', reviewer=?, reason=?,
                    result_json=?, final_json=?, updated_at=? WHERE id=? AND status='pending' ''',
                    (reviewer.reviewer_id, reason, json.dumps(observation), json.dumps(final),
                     time.time(), checkpoint_id))
                row = db.execute('SELECT * FROM escalations WHERE id=?', (checkpoint_id,)).fetchone()
                return {**public_record(row), 'deduplicated': False}
            if registry is None or execute is None:
                raise ValueError('resume_dependencies_required')
            db.execute('''UPDATE escalations SET status='approved',reviewer=?,updated_at=?
                WHERE id=? AND status='pending' ''', (reviewer.reviewer_id, time.time(), checkpoint_id))
        # Approval is durable before execution is claimed. No model/tool call under a DB lock.
        with transaction(self.path) as db:
            row, _ = authorized_row(db, checkpoint_id, identity)
            changed = db.execute('''UPDATE escalations SET status='executing',updated_at=?
                WHERE id=? AND status='approved' ''', (time.time(), checkpoint_id)).rowcount
            if changed != 1:
                return {**public_record(row), 'deduplicated': True}
            payload = json.loads(row['context_json'])
        from .autonomy import ExecutionRun, fingerprint, guarded_dispatch
        from .react import resume_review_observation
        scope = ToolScope.model_validate(payload['scope'])
        saved = payload['run']
        state = ExecutionRun(scope, max_calls=saved['max_calls'],
                             max_cost_units=saved['max_cost_units'], max_steps=saved['max_steps'],
                             dry_run=saved['dry_run'])
        state.calls, state.cost, state.steps = saved['calls'], saved['cost'], saved['steps']
        state.deadline = state.clock() + max(0, saved['expires_at'] - time.time())
        call = ToolCall.model_validate(payload['pending_tool_call'])
        arguments = call.arguments
        token = state.confirm(scope, call, arguments)
        state.approve(scope, call, arguments, confirmation=token,
                      idempotency_key=payload['idempotency_key'])
        # A durable review grants only this exact argument fingerprint.
        state.review_grants.add(fingerprint(scope, call.name, arguments))
        try:
            result = await guarded_dispatch(call, registry, scope, execute, state)
            observation = redact(result['payload'])
            final = resume_review_observation(payload, observation)
            status = 'completed'
        except Exception:
            observation = {'ok': False, 'error': 'outcome_unknown_reconcile'}
            final = {'reply': 'Your refund needs further human investigation.',
                     'citations': [], 'abstained': True, 'human_handoff': True}
            status = 'outcome_unknown'
        with transaction(self.path) as db:
            db.execute('''UPDATE escalations SET status=?,result_json=?,final_json=?,updated_at=?
                WHERE id=? AND status='executing' ''',
                (status, json.dumps(observation), json.dumps(redact(final)), time.time(), checkpoint_id))
            row = db.execute('SELECT * FROM escalations WHERE id=?', (checkpoint_id,)).fetchone()
            return {**public_record(row), 'deduplicated': False}


def graph_handoff(ticket, classification):
    from .pii import redact_graph
    from .checkpoints import action_key
    return redact_graph({
        'tenant_id': ticket['tenant_id'],
        'conversation_id': ticket['conversation_id'],
        'provenance': action_key(ticket['tenant_id'], ticket['conversation_id'],
                                 ticket['order_id'], ticket['amount_cents']),
        'order_id': ticket['order_id'], 'amount_cents': ticket['amount_cents'],
        'classification': classification, 'messages': ticket['messages'],
        'evidence': ticket['evidence'], 'plan_position': 'human_approval_checkpoint',
        'remaining_budget': 4,
        'decision_summary': 'Refund exceeds automatic limit; review exact pending action.',
    })


def incomplete_handoff(trusted, error, *, boundary):
    # Only application-held context is eligible for human transfer. Never
    # forward the conflicting candidate or unrestricted provider output.
    from .contracts import HANDOFF_KEYS
    source = trusted if isinstance(trusted, dict) else {}
    context = {key: source[key] for key in HANDOFF_KEYS if key in source}
    return {'status': 'human_escalation', 'human_handoff': True,
            'reason': error.reason, 'invalid_keys': list(error.keys),
            'boundary': boundary, 'tool_calls': [], 'do_not_retry': True,
            'context': redact(context),
            'next_action': 'authenticated human investigation'}


def inference_handoff(context, reason):
    required = {'tenant_id', 'conversation_id', 'messages', 'evidence',
                'plan_position', 'remaining_budget', 'committed_actions', 'trace_id'}
    if not isinstance(context, dict) or not required <= set(context):
        raise ValueError('incomplete_inference_handoff')
    return {'status': 'human_escalation', 'human_handoff': True,
            'reason': reason, 'tool_calls': [], 'do_not_retry': True,
            'context': redact({key: context[key] for key in required})}


def bounded_handoff(state, reason):
    """Snapshot trusted runtime state; a handoff grants no tool authority."""
    from copy import deepcopy
    from .pii import redact_value
    context = dict(
        tenant_id=state['tenant_id'], conversation_id=state['conversation_id'],
        trace_id=state['trace_id'], run_id=state['run_id'],
        messages=[state['message']], evidence=deepcopy(state['memory']),
        plan=deepcopy(state['plan']), plan_position=state['phase'],
        remaining_budget={
            'steps': max(0, state['caps']['max_steps'] - state['steps']),
            'tokens': max(0, state['caps']['max_tokens'] - state['tokens']),
            'calls': max(0, state['caps']['max_calls'] - state['calls']),
            'cost_units': max(0, state['caps']['max_cost_units'] - state['cost_units'])},
        attempts=deepcopy(state['attempts']), pending_action=deepcopy(state['pending']),
        committed_actions=deepcopy(state['memory']),
        orchestration=deepcopy(state.get('orchestration')))
    return redact_value(dict(human_handoff=True, do_not_retry=True,
                             reason=reason, tool_calls=[], context=context))


def realtime_handoff(payload):
    required = {'scope', 'reason', 'transcript', 'heard', 'evidence',
                'pending_action', 'approval_required'}
    if (set(payload) != required or payload['pending_action'] != 'none_executed'
            or payload['approval_required'] is not True):
        raise ValueError('invalid_realtime_handoff')
    return redact(payload)
