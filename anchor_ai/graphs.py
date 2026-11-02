import asyncio
import operator
import re
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .autonomy import ExecutionRun, fingerprint, load_policy
from .checkpoints import CheckpointStore, action_key, actor_claims
from .contracts import ToolCall, ToolScope
from .escalation import graph_handoff
from .orchestration import GraphRoleClient, graph_classification
from .pii import redact_graph
from .tools import DurableOfflineOrderBackend, dispatch, registry_for


class RefundState(TypedDict, total=False):
    ticket: dict
    classification: str
    action_key: str
    context: dict
    approved: bool
    result: dict
    reply: str
    handoff_failure: dict
    audit: Annotated[list[str], operator.add]


def validated_ticket(raw, claims):
    required = {'tenant_id', 'conversation_id', 'order_id', 'amount_cents',
                'limit_cents', 'messages', 'evidence'}
    if set(raw) != required:
        raise ValueError('invalid_ticket_fields')
    for key in ('tenant_id', 'conversation_id', 'amount_cents', 'limit_cents'):
        if type(raw[key]) is not int or raw[key] < (0 if key == 'limit_cents' else 1):
            raise ValueError('invalid_ticket_number')
    if raw['tenant_id'] != claims['tenant_id']:
        raise PermissionError('tenant_mismatch')
    if not isinstance(raw['order_id'], str) or not re.fullmatch(r'[A-Z][A-Z0-9]{1,30}', raw['order_id']):
        raise ValueError('secure_order_reference_required')
    for key in ('messages', 'evidence'):
        if not isinstance(raw[key], list) or not raw[key] or not all(isinstance(v, str) for v in raw[key]):
            raise ValueError('incomplete_review_context')
    # Preserve executable fields exactly; redact only human-readable context.
    safe = dict(raw)
    safe['messages'] = redact_graph(raw['messages'])
    safe['evidence'] = redact_graph(raw['evidence'])
    return safe


@contextmanager
def workflow(path, thread, token, client=None):
    store = CheckpointStore(path)
    claims = actor_claims(token)
    client = client or GraphRoleClient()
    backend = DurableOfflineOrderBackend(path)
    registry = registry_for(backend)

    def triage(state):
        print('ENTER triage', flush=True)
        ticket = state['ticket']
        classification = graph_classification(client, ticket)
        key = action_key(ticket['tenant_id'], ticket['conversation_id'],
                         ticket['order_id'], ticket['amount_cents'])
        context = graph_handoff(ticket, classification)
        store.pending(thread, ticket, key, context)
        return {'classification': classification, 'action_key': key,
                'context': context, 'audit': ['triage']}

    def approval(state):
        reference = interrupt({'node': 'human_approval_checkpoint',
                               'thread_id': thread, 'action_key': state['action_key']})
        if type(reference) is not int:
            raise PermissionError('approval_reference_required')
        store.approval(thread, token, state['action_key'], reference)
        return {'approved': True, 'audit': ['approved']}

    def specialist(state):
        print('ENTER specialist', flush=True)
        trusted = graph_handoff(state['ticket'], state['classification'])
        admission = dispatch_specialist(state.get('context'), trusted,
                                        lambda context: {'admitted': True})
        if admission.get('human_handoff'):
            return {'handoff_failure': admission, 'audit': ['specialist']}
        # Even ordinary refunds use this explicit approval gate in this increment.
        store.approval(thread, token, state['action_key'])
        ticket = state['ticket']
        # This is a trusted service-role selection, not a role from the ticket
        # or the model. Reviewer consent does not confer tool permissions.
        roles = load_policy()['tenants'].get(ticket['tenant_id'], {}).get('roles', {})
        required_tools = {'lookup_order', 'check_refund', 'issue_refund'}
        candidates = [role for role, allowed in roles.items()
                      if required_tools.issubset(set(allowed))]
        if not candidates:
            raise PermissionError('refund_execution_role_required')
        # Prefer the configured role with the fewest tool permissions.
        execution_role = min(candidates, key=lambda role: (len(set(roles[role])), role))
        scope = ToolScope(tenant_id=ticket['tenant_id'],
                          conversation_id=ticket['conversation_id'], role=execution_role)
        arguments = {'order_id': ticket['order_id'], 'amount_cents': ticket['amount_cents']}
        call = ToolCall(call_id='refund_' + state['action_key'][:16],
                        name='issue_refund', arguments=arguments)
        # Use the established execution policy rather than inventing graph budgets.
        run = ExecutionRun(scope, dry_run=False)

        async def execute_approved_action():
            async def guarded_step(tool_call):
                reason = await run.step(scope)
                if reason:
                    raise RuntimeError('refund_execution_step_denied')
                response = await dispatch(tool_call, registry, scope, run=run)
                if not response['payload']['ok']:
                    # Report only the dispatcher's machine-readable error code.
                    code = response['payload'].get('error', 'unknown_error')
                    if not isinstance(code, str) or not re.fullmatch(r'[a-z_]+', code):
                        code = 'unknown_error'
                    raise RuntimeError('refund_dispatch_denied:' + code)
                return response

            # Consent does not replace tenant-scoped reads or eligibility checks.
            await guarded_step(ToolCall(
                call_id='graph_lookup', name='lookup_order',
                arguments={'order_id': ticket['order_id']}))
            eligibility = await guarded_step(ToolCall(
                call_id='graph_eligibility', name='check_refund',
                arguments={'order_id': ticket['order_id']}))
            if eligibility['payload']['data']['eligible'] is not True:
                raise RuntimeError('refund_ineligible')

            reason = await run.step(scope)
            if reason:
                raise RuntimeError('refund_execution_step_denied')
            confirmation = run.confirm(scope, call, arguments)
            run.approve(scope, call, arguments, confirmation=confirmation,
                        idempotency_key=state['action_key'])
            run.review_grants.add(fingerprint(scope, call.name, arguments))
            result = await dispatch(call, registry, scope, run=run)
            if not result['payload']['ok']:
                code = result['payload'].get('error', 'unknown_error')
                if not isinstance(code, str) or not re.fullmatch(r'[a-z_]+', code):
                    code = 'unknown_error'
                raise RuntimeError('refund_dispatch_denied:' + code)
            return result

        result = asyncio.run(execute_approved_action())
        reply = redact_graph(client.complete('refund_specialist', state['context']))
        return {'result': result['payload'], 'reply': reply, 'audit': ['specialist']}

    def handoff(state):
        if state.get('handoff_failure'):
            return {'audit': ['escalation']}
        store.complete(thread, token, state['action_key'])
        return {'audit': ['escalation']}

    graph = StateGraph(RefundState)
    graph.add_node('triage', triage)
    graph.add_node('human_approval_checkpoint', approval)
    graph.add_node('specialist', specialist)
    graph.add_node('escalation', handoff)
    graph.add_edge(START, 'triage')
    graph.add_conditional_edges('triage', lambda s: s['classification'],
        {'large_refund': 'human_approval_checkpoint', 'ordinary_refund': 'human_approval_checkpoint'})
    graph.add_edge('human_approval_checkpoint', 'specialist')
    graph.add_edge('specialist', 'escalation')
    graph.add_edge('escalation', END)
    connection = sqlite3.connect(path, check_same_thread=False)
    try:
        saver = SqliteSaver(connection)
        saver.setup()
        yield graph.compile(checkpointer=saver), store, claims
    finally:
        connection.close()


def start_refund(path, raw, token):
    claims = actor_claims(token)
    ticket = validated_ticket(raw, claims)
    thread = f"t{claims['tenant_id']}-c{ticket['conversation_id']}-{uuid.uuid4().hex}"
    store = CheckpointStore(path)
    with store.lease(thread), workflow(path, thread, token) as (graph, _, _):
        config = {'configurable': {'thread_id': thread}}
        graph.invoke({'ticket': ticket, 'audit': []}, config)
        snapshot = graph.get_state(config)
        assert snapshot.next == ('human_approval_checkpoint',)
    return thread


def resume_refund(path, thread, token):
    store = CheckpointStore(path)
    # Authorize BEFORE asking the saver for executable state.
    row = store.inspect(thread, token)
    approval = store.approval(thread, token, row['action_key'])
    with store.lease(thread), workflow(path, thread, token) as (graph, _, _):
        config = {'configurable': {'thread_id': thread}}
        before = graph.get_state(config)
        if not before.next:
            raise PermissionError('approval_replay_rejected')
        if before.values['ticket']['tenant_id'] != actor_claims(token)['tenant_id']:
            raise PermissionError('tenant_mismatch')
        trusted = graph_handoff(before.values['ticket'], before.values['classification'])
        admission = dispatch_specialist(before.values.get('context'), trusted,
                                        lambda context: {'admitted': True}, boundary='resume')
        if admission.get('human_handoff'):
            return admission
        if before.next == ('human_approval_checkpoint',):
            graph.invoke(Command(resume=approval['id']), config)
        else:
            # Recover a failed specialist/handoff without rerunning completed nodes.
            graph.invoke(None, config)
        after = graph.get_state(config)
        assert not after.next
        assert after.values['audit'] == ['triage', 'approved', 'specialist', 'escalation']
        return after.values


def policy_query_tool(retriever, claims, client=None, **options):
    """Read-only agent capability; claims must come from authenticated code."""
    from anchor_ai.rag import published_query_engine_tool
    return published_query_engine_tool(retriever, claims['tenant_id'],
                                       client=client, **options)


def dispatch_specialist(context, trusted, invoke, *, boundary='dispatch'):
    from .orchestration import dispatch_specialist as guarded
    return guarded(context, trusted, invoke, boundary=boundary)
