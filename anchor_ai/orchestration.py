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
