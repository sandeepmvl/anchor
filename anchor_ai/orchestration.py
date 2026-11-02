async def run_native_support_turn(agent, scope, message, *, enabled=False,
                                  previous_response_id=None, run=None):
    if not enabled:
        raise ValueError('native_agent_requires_opt_in')
    return await agent.turn(scope, message,
                            previous_response_id=previous_response_id, run=run)


import copy
from pathlib import Path

import yaml

from anchor_ai.contracts import OrchestrationFlow


def select_flow(ticket_type, path='config/orchestration.yml'):
    config = yaml.safe_load(Path(path).read_text())
    pattern = config['routes'][ticket_type]
    workers = config['workers'] if pattern == 'supervisor-worker' else {}
    return OrchestrationFlow(pattern=pattern, workers=workers).model_dump()


def worker_for(state):
    # Runs checkpointed before this lesson keep their original execution path.
    flow = state.get('orchestration', {'pattern': 'single-agent'})
    if flow['pattern'] == 'single-agent':
        return None
    return flow['workers'][state['plan'][state['phase']]]


async def propose(model, state, reserve):
    worker = worker_for(state)
    plan = state['plan']
    if worker is not None:
        # The worker owns only the next action. Keep completed positions so
        # adapters using len(memory) as the plan cursor remain compatible.
        plan = plan[:state['phase'] + 1]
    return await model.propose(
        state['strategy'], copy.deepcopy(plan),
        copy.deepcopy(state['memory']), reserve)


class GraphRoleClient:
    """Two graph roles; provider use remains behind Anchor's client."""
    def complete(self, role, context):
        import os
        if os.getenv('ANCHOR_LLM_PROVIDER', 'fake') not in ('', 'fake', 'offline'):
            from .llm_client import LlmClient
            return LlmClient().complete(str(context), system=role)
        if role == 'triage':
            return 'large_refund' if context['amount_cents'] > context['limit_cents'] else 'ordinary_refund'
        if role == 'refund_specialist':
            return 'Approved refund completed by the offline backend; no money moved.'
        raise ValueError('unknown_graph_role')


def graph_classification(client, ticket):
    # The deterministic risk rule, not a model label, controls approval routing.
    client.complete('triage', ticket)
    from .escalation import risk_rule
    risk = risk_rule(ticket['amount_cents'], ticket['limit_cents'], True)
    return 'large_refund' if risk['escalate'] else 'ordinary_refund'


class ConversationRoleClient:
    """Advisory role text only; no refund tools or approval authority."""
    def complete(self, role, context):
        import os
        if os.getenv('ANCHOR_LLM_PROVIDER', 'fake') not in ('', 'fake', 'offline'):
            from .llm_client import LlmClient
            return LlmClient().complete(
                str(context), system=(role + ': advisory refund eligibility only; '
                'explain manual approval; never claim a refund executed.'))
        if role == 'triage':
            return 'Refund eligibility question. Route to refund_specialist.'
        if role == 'refund_specialist':
            return 'Manual approval required. No refund executed.'
        raise ValueError('unknown_conversation_role')


async def run_support_conversation(ticket, memory, *, enabled=False, **options):
    # Explicit opt-in preserves every existing route and checkpoint contract.
    if not enabled:
        raise ValueError('group_chat_requires_opt_in')
    from .group_chat import run_conversation
    return await run_conversation(ticket, memory, **options)


async def run_harness_support_turn(adapter, token, request, *, enabled=False,
                                   path='config/agent_execution.yml'):
    if not enabled:
        raise ValueError('local_harness_requires_opt_in')
    config = yaml.safe_load(Path(path).read_text())
    mode = config['modes']['local_harness']
    if (mode['loop_owner'] != 'anchor_client'
            or mode['pre_tool_hook'] != 'required'):
        raise ValueError('invalid_local_harness_configuration')
    return await adapter.run(token, request)
