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
