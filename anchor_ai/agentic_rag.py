import asyncio
import hashlib
import json
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from anchor_ai.autonomy import run_for
from anchor_ai.contracts import ToolScope
from anchor_ai.observability import AGENT_SUMMARIES, span
from anchor_ai.prompts import AGENTIC_RAG_INSTRUCTION
from anchor_ai.schemas import AgentDecision, RetrieveArguments
from anchor_ai.schemas import normalize_retrieval_query

FIXTURE = Path(__file__).parent / 'tests/fixtures/agentic_responses.json'
ABSTENTION = 'I could not find a supporting policy in your documents.'
RETRIEVE_TOOL = {
    'name': 'retrieve',
    'description': 'Search the current tenant knowledge base.',
    'parameters': RetrieveArguments.model_json_schema(),
}


def settings():
    path = Path(__file__).resolve().parents[1] / 'config/retrieval.yml'
    return yaml.safe_load(path.read_text())['agentic']


def fixture_data():
    return json.loads(FIXTURE.read_text())


class RecordedPolicy:
    """Recorded decisions gated by evidence coverage, not semantic judgment."""
    def __init__(self, data=None):
        self.data = data or fixture_data()

    async def decide(self, question, evidence):
        ticket = next((t for t in self.data['tickets'].values()
                       if t['question'] == question), None)
        if ticket is None:
            return {'kind': 'final', 'answer': {
                'reply': ABSTENTION, 'citations': [], 'abstained': True}}
        ids = {p['id'] for p in evidence}
        required = set(ticket['required_ids'])
        if not required:
            return {'kind': 'final', 'answer': ticket['answer']}
        if required <= ids:
            return {'kind': 'sufficient', 'answer': ticket['answer']}
        for decision in ticket['decisions']:
            if not set(decision['requires']) <= ids:
                continue
            returned = set(self.data['searches'][decision['query']])
            if not returned <= ids:
                return {'kind': 'retrieve', 'tool': 'retrieve',
                        'arguments': {'query': decision['query']}}
        return {'kind': 'final', 'answer': {
            'reply': ABSTENTION, 'citations': [], 'abstained': True}}


class LivePolicy:
    """Structured tool decisions through Anchor's existing provider boundary."""
    def __init__(self, client=None):
        if client is None:
            from anchor_ai.llm_client import LlmClient
            client = LlmClient()
        self.client = client

    async def decide(self, question, evidence):
        payload = {'question': question, 'passages': evidence,
                   'tool': RETRIEVE_TOOL,
                   'decision_schema': AgentDecision.model_json_schema()}
        raw = await self.client.complete(json.dumps(payload, ensure_ascii=False),
                                         system=AGENTIC_RAG_INSTRUCTION,
                                         max_tokens=512)
        return json.loads(raw)


def select_policy():
    mode = os.getenv(settings()['mode_env'], 'offline')
    if mode == 'offline':
        return mode, RecordedPolicy()
    if mode == 'live':
        return mode, LivePolicy()
    raise ValueError('invalid_agent_mode')


class RecordedRetrievalBoundary:
    """Fixture observations at the existing retrieve interface, not a search engine."""
    def __init__(self, data=None):
        self.data = data or fixture_data()
        self.tenant_id = self.data['tenant_id']
        self.generation_id = self.data['generation_id']
        self.calls = []

    async def retrieve(self, query, top_k=3):
        self.calls.append((self.tenant_id, self.generation_id, query, top_k))
        ids = self.data['searches'].get(query, [])[:top_k]
        return {'tenant_id': self.tenant_id, 'generation_id': self.generation_id,
                'hits': [dict(self.data['passages'][str(i)],
                              tenant_id=self.tenant_id) for i in ids]}


class RetrievalTool:
    def __init__(self, retriever, tenant_id, generation_id=None):
        if type(tenant_id) is not int or tenant_id <= 0:
            raise ValueError('invalid_trusted_tenant')
        if getattr(retriever, 'tenant_id', tenant_id) != tenant_id:
            raise ValueError('agent_scope_mismatch')
        self.retriever = retriever
        self.tenant_id = tenant_id
        self.generation_id = generation_id

    async def __call__(self, arguments, k):
        args = RetrieveArguments.model_validate(arguments)
        with span('python.agent.retrieve', stage='retrieval',
                  tenant_id=self.tenant_id, k=k) as operation:
            if hasattr(self.retriever, 'retrieve'):
                result = await self.retriever.retrieve(args.query, k)
                hits = result['hits']
                tenant = result['tenant_id']
                generation = result['generation_id']
            else:
                hits = self.retriever.search(args.query, k)
                tenant = self.retriever.last_trace['tenant_id']
                generation = self.retriever.last_trace['generation_id']
            if tenant != self.tenant_id or not generation:
                raise ValueError('agent_scope_mismatch')
            if self.generation_id is None:
                self.generation_id = generation
            if generation != self.generation_id:
                raise ValueError('agent_scope_mismatch')
            if any(h.get('tenant_id', tenant) != tenant for h in hits):
                raise ValueError('agent_scope_mismatch')
            passages = []
            for hit in hits:
                if type(hit['id']) is not int or hit['id'] <= 0 or not isinstance(hit['text'], str):
                    raise ValueError('invalid_retrieval_observation')
                passages.append({'id': hit['id'], 'text': hit['text']})
            operation.set(generation_id=generation,
                          chunk_ids=[p['id'] for p in passages])
            return {'passages': passages}


class AgenticRuntime:
    def __init__(self, retriever, policy=None, emit=None, evidence_budget=None,
                 max_iterations=None, max_tool_calls=None, tenant_id=None,
                 generation_id=None, max_hops=None):
        config = settings()
        self.mode, selected = select_policy() if policy is None else ('injected', policy)
        self.policy = selected
        self.emit = emit or (lambda record: None)
        self.evidence_budget = self._limit(evidence_budget, config['evidence_budget_bytes'], 1)
        self.max_iterations = self._limit(max_iterations, config['max_iterations'], 1)
        self.max_tool_calls = self._limit(max_tool_calls, config['max_tool_calls'], 0)
        self.max_hops = self._limit(max_hops, config['max_hops'], 1)
        self.policy_timeout = config['policy_timeout_seconds']
        trusted_tenant = tenant_id if tenant_id is not None else getattr(retriever, 'tenant_id', None)
        self.tool = RetrievalTool(retriever, trusted_tenant, generation_id)
        self.last_trajectory = []

    @staticmethod
    def _limit(requested, configured, minimum):
        value = configured if requested is None else requested
        if type(value) is not int or value < minimum:
            raise ValueError('invalid_agent_budget')
        return min(value, configured)

    async def answer(self, question, k=3, *, request_id, trace_id,
                     conversation_seen_ids=(), execution_scope=None, execution_run=None):
        if not isinstance(question, str) or not question.strip() or len(question.encode('utf-8')) > 1800:
            raise ValueError('current_question_budget_exceeded')
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError('invalid_top_k')
        # Trusted caller history must belong to this tenant and pinned generation.
        history = tuple(conversation_seen_ids)
        if any(type(i) is not int or i <= 0 for i in history):
            raise ValueError('invalid_conversation_seen_ids')
        if history and self.tool.generation_id is None:
            raise ValueError('agent_scope_mismatch')
        scope = execution_scope or ToolScope(tenant_id=self.tool.tenant_id, conversation_id=1)
        if scope.tenant_id != self.tool.tenant_id:
            raise ValueError('agent_scope_mismatch')
        state = run_for(scope, execution_run)
        seen_ids, seen_queries = set(history), set()
        evidence, calls = {}, 0
        self.last_trajectory = []
        ctx = {'request_id': request_id, 'trace_id': trace_id, 'parent_span_id': None}
        with span('python.agent', ctx=ctx, stage='agent',
                  tenant_id=self.tool.tenant_id, agent_mode=self.mode) as root:
            def finish(reason, answer=None):
                state.halted = True
                result = answer or {'reply': ABSTENTION, 'citations': [], 'abstained': True}
                root.set(agent_exit=reason, tool_calls=calls,
                         evidence_chunk_ids=sorted(evidence), seen_chunk_ids=sorted(seen_ids),
                         citations=result['citations'], abstained=result['abstained'])
                self.emit({'event': 'agent_exit', 'request_id': request_id,
                           'trace_id': trace_id, 'exit': reason, 'tool_calls': calls,
                           'evidence_chunk_ids': sorted(evidence),
                           'seen_chunk_ids': sorted(seen_ids)})
                return result

            for iteration in range(1, self.max_iterations + 1):
                reason = await state.step(scope)
                if reason:
                    return finish(reason)
                with span('python.agent.model', stage='llm', agent_iteration=iteration,
                          iterations_remaining=self.max_iterations - iteration,
                          tool_calls_remaining=self.max_tool_calls - calls,
                          hops_remaining=max(0, self.max_hops - calls)) as model_span:
                    try:
                        raw = await asyncio.wait_for(
                            self.policy.decide(question, list(evidence.values())),
                            timeout=self.policy_timeout)
                        decision = AgentDecision.model_validate(raw)
                    except (ValueError, TypeError, ValidationError):
                        model_span.fail('agent_invalid_decision')
                        return finish('invalid_decision')
                    except Exception:
                        model_span.fail('invalid_provider_response')
                        return finish('policy_error')
                    model_span.set(decision_kind=decision.kind,
                                   decision_summary=AGENT_SUMMARIES[decision.kind])
                record = {'iteration': iteration, 'kind': decision.kind,
                          'summary': AGENT_SUMMARIES[decision.kind],
                          'tool_calls': calls, 'passage_ids': []}
                self.last_trajectory.append(record)
                if decision.kind != 'retrieve':
                    answer = decision.answer.model_dump()
                    citations = answer['citations']
                    valid = (len(citations) == len(set(citations))
                             and all(type(i) is int and i in evidence for i in citations)
                             and (answer['abstained'] or bool(citations)))
                    if not valid:
                        return finish('invalid_answer')
                    reason = 'final_answer' if decision.kind == 'final' else 'evidence_sufficient'
                    return finish(reason, answer)
                if calls >= self.max_tool_calls:
                    return finish('tool_budget')
                if calls >= self.max_hops:
                    return finish('max_hops')
                normalized = normalize_retrieval_query(decision.arguments.query)
                fingerprint = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
                record['query_fingerprint'] = fingerprint
                record['duplicate_query'] = normalized in seen_queries
                root.set(query_fingerprint=fingerprint,
                         duplicate_query=record['duplicate_query'])
                if record['duplicate_query']:
                    return finish('duplicate_query')
                seen_queries.add(normalized)
                reason, _ = await state.admit(scope, 'retrieve', decision.arguments.model_dump())
                if reason:
                    return finish(reason)
                calls += 1
                with span('python.agent.tool', stage='tool', tool_name='retrieve',
                          query_fingerprint=fingerprint, tool_calls=calls) as tool_span:
                    try:
                        observation = await self.tool(decision.arguments.model_dump(), k)
                    except Exception:
                        tool_span.fail('agent_tool_failed')
                        return finish('tool_error')
                    returned_ids = {p['id'] for p in observation['passages']}
                    new_ids = returned_ids - seen_ids
                    seen_ids.update(returned_ids)
                    tool_span.set(new_chunk_ids=sorted(new_ids),
                                  seen_chunk_ids=sorted(seen_ids))
                record['tool_calls'] = calls
                record['passage_ids'] = [p['id'] for p in observation['passages']]
                record['new_chunk_ids'] = sorted(new_ids)
                for passage in observation['passages']:
                    candidate = dict(evidence)
                    candidate[passage['id']] = passage
                    size = len(json.dumps(list(candidate.values()), ensure_ascii=False).encode('utf-8'))
                    if size <= self.evidence_budget:
                        evidence = candidate
                self.emit({'event': 'agent_observation', 'request_id': request_id,
                           'trace_id': trace_id, 'tool': 'retrieve',
                           'iteration': iteration, 'passage_ids': record['passage_ids'],
                           'query_fingerprint': fingerprint, 'new_chunk_ids': sorted(new_ids),
                           'seen_chunk_ids': sorted(seen_ids),
                           'evidence_chunk_ids': sorted(evidence), 'tool_calls': calls})
                if not new_ids:
                    return finish('no_new_evidence')
            return finish('iteration_budget')


def build_agent_runtime(retriever, *, strategy='agentic', **kwargs):
    """Select a controller without changing existing callers' default."""
    if strategy == 'agentic':
        return AgenticRuntime(retriever, **kwargs)
    if strategy == 'react':
        from anchor_ai.react import ReActRuntime
        return ReActRuntime(retriever, **kwargs)
    if strategy == 'plan_execute':
        from anchor_ai.planning import PlanExecuteRuntime
        return PlanExecuteRuntime(retriever, **kwargs)
    raise ValueError('invalid_agent_strategy')
