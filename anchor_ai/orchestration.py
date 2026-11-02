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


FRAMEWORK_LABELS = {
    'state': 'state-model expressiveness',
    'debug': 'debuggability/observability',
    'ecosystem': 'ecosystem maturity',
    'exit-cost': 'exit-cost',
    'runtime-fit': 'runtime-fit',
}


def framework_policy(path='config/orchestration.yml'):
    import math
    config = yaml.safe_load(Path(path).read_text())
    policy_path = Path(config['framework_selection']['policy'])
    if not policy_path.is_absolute():
        policy_path = Path(path).resolve().parent.parent / policy_path
    data = yaml.safe_load(policy_path.read_text())
    if data['version'] != 1:
        raise ValueError('unsupported_framework_policy')
    candidates = {'LangChain', 'LangGraph', 'LlamaIndex', 'AutoGen', 'provider-native'}
    if set(data['frameworks']) != candidates:
        raise ValueError('invalid_framework_candidates')

    def valid_number(value, low, high):
        return (type(value) in (int, float) and math.isfinite(value)
                and low <= value <= high)

    for scores in data['frameworks'].values():
        if (set(scores) != set(FRAMEWORK_LABELS)
                or not all(valid_number(v, 1, 5) for v in scores.values())):
            raise ValueError('invalid_framework_scores')
    if set(data['usecases']) != {'escalation', 'rag', 'triage'}:
        raise ValueError('invalid_framework_usecases')
    for profile in data['usecases'].values():
        for key, axes in (
            ('weights', {'state', 'debug', 'ecosystem', 'lock-in'}),
            ('subweights', {'exit-cost', 'runtime-fit'}),
        ):
            values = profile[key]
            if (set(values) != axes
                    or not all(valid_number(v, 0, 1) for v in values.values())
                    or abs(sum(values.values()) - 1) >= 1e-9):
                raise ValueError('invalid_framework_weights')
    return data


def framework_ranking(data, usecase):
    profile = data['usecases'][usecase]
    weights = profile['weights']
    effective = {k: weights[k] for k in ('state', 'debug', 'ecosystem')}
    effective.update({k: weights['lock-in'] * v
                      for k, v in profile['subweights'].items()})
    rows = []
    for name, scores in data['frameworks'].items():
        parts = {key: weight * scores[key] for key, weight in effective.items()}
        rows.append(dict(framework=name, total=sum(parts.values()), breakdown=parts))
    return sorted(rows, key=lambda row: (-row['total'], row['framework']))


def select_framework(usecase, path='config/orchestration.yml'):
    data = framework_policy(path)
    rows = framework_ranking(data, usecase)
    if (rows[0]['framework'] != data['usecases'][usecase]['selected']
            or rows[0]['total'] - rows[1]['total'] <= 1e-9):
        raise ValueError('framework_policy_drift')
    return rows[0]['framework']


def validate_framework_envelope(envelope):
    if (not isinstance(envelope, dict)
            or type(envelope.get('tenant_id')) is not int
            or envelope['tenant_id'] <= 0
            or not isinstance(envelope.get('trace_id'), str)
            or not envelope['trace_id'].strip()):
        raise ValueError('invalid_framework_scope')
    budget = envelope.get('budget')
    if (not isinstance(budget, dict)
            or any(type(budget.get(key)) is not int or budget[key] <= 0
                   for key in ('max_tokens', 'max_steps'))):
        raise ValueError('invalid_framework_budget')


async def run_framework_task(usecase, payload, envelope, *, adapters,
                             path='config/orchestration.yml'):
    """Adapters accept (payload, envelope); their runtimes meter the limits."""
    validate_framework_envelope(envelope)
    framework = select_framework(usecase, path)
    adapter = adapters.get(framework)
    if not callable(adapter):
        raise ValueError('framework_adapter_unavailable')
    trusted = copy.deepcopy(envelope)
    supplied = copy.deepcopy(trusted)
    result = await adapter(copy.deepcopy(payload), supplied)
    if supplied != trusted:
        raise ValueError('framework_envelope_mutated')
    return dict(framework=framework, envelope=trusted, result=result)
