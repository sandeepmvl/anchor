import asyncio
import copy
import json
from pathlib import Path

import pytest

from anchor_ai import observability, react
from anchor_ai.agentic_rag import RecordedPolicy, RecordedRetrievalBoundary
from anchor_ai.agentic_rag import RetrievalTool, build_agent_runtime
from anchor_ai.contracts import ToolScope
from anchor_ai.observability import AGENT_SUMMARIES

FIXTURE = Path(__file__).parent / 'fixtures/react_responses.json'


class BudgetSpy:
    def __init__(self, denied=None):
        self.steps = 4
        self.admissions = 2
        self.halted = False
        self.denied = denied
        self.scopes = []

    async def step(self, scope):
        self.steps += 1
        self.scopes.append(scope)
        return None

    async def admit(self, scope, name, arguments):
        assert name == 'retrieve'
        assert set(arguments) == {'query'}
        self.admissions += 1
        self.scopes.append(scope)
        return self.denied, None


class ScriptedClient:
    def __init__(self, queries):
        self.queries = queries
        self.inputs = []

    async def complete(self, messages):
        self.inputs.append(copy.deepcopy(messages))
        turn = len(self.inputs)
        return {
            'role': 'assistant',
            'content': json.dumps({'reasoning': AGENT_SUMMARIES['retrieve']}),
            'tool_calls': [{
                'id': f'call_{turn}', 'type': 'function',
                'function': {'name': 'retrieve', 'arguments':
                             json.dumps({'query': self.queries[turn - 1]})}}],
        }


@pytest.fixture
def setup_run(monkeypatch):
    data = json.loads(FIXTURE.read_text())
    spans, exports, selections = [], [], []
    budget = BudgetSpy()
    scope = ToolScope(tenant_id=data['tenant_id'], conversation_id=1)
    monkeypatch.setattr(observability, 'SINK', spans.append)

    def run_for(received_scope, received_run):
        assert received_scope is scope
        assert received_run is budget
        selections.append(received_run)
        return received_run

    monkeypatch.setattr(react, 'run_for', run_for)
    # Isolate controller limits from deployment configuration. The inherited
    # constructor still validates and clamps every requested limit.
    import anchor_ai.agentic_rag as agentic
    monkeypatch.setattr(agentic, 'settings', lambda: {
        'evidence_budget_bytes': 4096, 'max_iterations': 5,
        'max_tool_calls': 5, 'max_hops': 5, 'policy_timeout_seconds': 1,
        'mode_env': 'ANCHOR_AGENT_MODE',
    })

    def build(**kwargs):
        boundary = RecordedRetrievalBoundary(data)
        runtime = build_agent_runtime(
            boundary, strategy='react', policy=RecordedPolicy(data),
            emit=exports.append, **kwargs)
        assert isinstance(runtime.tool, RetrievalTool)
        return runtime, boundary

    def answer(runtime):
        return asyncio.run(runtime.answer(
            data['tickets']['eligibility']['question'], request_id='react-test',
            trace_id='1' * 32, execution_scope=scope, execution_run=budget))

    return data, spans, exports, selections, budget, scope, build, answer


def test_eligibility_timeline_and_grounded_answer(setup_run):
    data, spans, events, selections, budget, scope, build, answer = setup_run
    runtime, boundary = build()
    result = answer(runtime)
    assert result == data['tickets']['eligibility']['answer']
    assert runtime.last_exit == 'final_answer'
    assert [e['event'] for e in events] == [
        'decision_summary', 'action', 'observation',
        'decision_summary', 'action', 'observation', 'decision_summary', 'exit']
    assert [call[2] for call in boundary.calls] == [
        'A17 eligibility class', 'standard refund window']
    assert events[2]['evidence_chunk_ids'] == [701]
    assert events[5]['evidence_chunk_ids'] == [701, 702]
    assert selections == [budget]
    assert budget.steps == 7 and budget.admissions == 4 and budget.halted
    assert all(s is scope for s in budget.scopes)
    assert {s['trace_id'] for s in spans} == {'1' * 32}
    assert {'llm', 'tool', 'retrieval'} <= {s.get('stage') for s in spans}
    exported = json.dumps(events + spans)
    assert 'customer@example.test' not in exported
    assert 'delivered 10 days ago' not in exported
    assert 'arguments' not in exported
    assert all(e['observation_redacted'] for e in events if e['event'] == 'observation')
    # Every non-root span links to a span in this run.
    ids = {s['span_id'] for s in spans}
    assert all(s['parent_span_id'] in ids for s in spans if s['name'] != 'python.react')
    print('Anchor eligibility: two evidence steps -> grounded final answer; no refund issued.')
    for event in events:
        label = {'decision_summary': 'Thought', 'action': 'Action',
                 'observation': 'Observation', 'exit': 'Stop'}[event['event']]
        print(label + ': ' + json.dumps(event, sort_keys=True))


def test_repeat_reaches_cap_without_resetting_shared_budget(setup_run):
    data, spans, events, selections, budget, scope, build, answer = setup_run
    client = ScriptedClient(data['repeat_queries'])
    runtime, boundary = build(native_client=client, max_iterations=3, detect_cycles=False)
    result = answer(runtime)
    assert result['abstained'] and result['citations'] == []
    assert runtime.last_exit == 'iteration_budget'
    assert len(boundary.calls) == 3
    assert selections == [budget]
    assert budget.steps == 7 and budget.admissions == 5
    assert budget.halted
    assert all(s is scope for s in budget.scopes)
    assert [e['tool_calls'] for e in events if e['event'] == 'observation'] == [1, 2, 3]
    # The second and third invocations receive matched native results.
    for transcript in client.inputs[1:]:
        assistants = [m for m in transcript if m['role'] == 'assistant']
        tools = [m for m in transcript if m['role'] == 'tool']
        assert [m['tool_calls'][0]['id'] for m in assistants] == [m['tool_call_id'] for m in tools]
        assert all(json.loads(m['content'])['passages'] for m in tools)
    print('Repeating read-only action: iteration_budget; shared steps 4 -> 7; admissions 2 -> 5.')


def test_normalized_whole_run_cycle_blocks_without_observation(setup_run):
    data, spans, events, selections, budget, scope, build, answer = setup_run
    runtime, boundary = build(native_client=ScriptedClient(data['cycle_queries']))
    answer(runtime)
    assert runtime.last_exit == 'repeated_action'
    assert len(boundary.calls) == 2
    actions = [e for e in events if e['event'] == 'action']
    assert actions[0]['query_fingerprint'] == actions[2]['query_fingerprint']
    assert [e['event'] for e in events[-3:]] == ['decision_summary', 'action', 'exit']
    assert budget.admissions == 4


def test_shared_policy_denial_prevents_dispatch(setup_run):
    data, spans, events, selections, budget, scope, build, answer = setup_run
    budget.denied = 'tool_budget'
    runtime, boundary = build()
    result = answer(runtime)
    assert runtime.last_exit == 'tool_budget'
    assert result['abstained']
    assert boundary.calls == []
    assert not any(e['event'] == 'observation' for e in events)
    assert selections == [budget]


def test_scope_guard_remains_in_existing_tool(setup_run):
    data, spans, events, selections, budget, scope, build, answer = setup_run
    runtime, boundary = build()
    boundary.tenant_id = 2
    result = answer(runtime)
    assert runtime.last_exit == 'tool_error'
    assert result['abstained']
    assert not any(e['event'] == 'observation' for e in events)
