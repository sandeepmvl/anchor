import json
import os
import re
from pathlib import Path

from anchor_ai.prompts import (
    MULTIHOP_PLAN_INSTRUCTION, MULTIHOP_SYNTHESIS_INSTRUCTION,
)
from anchor_ai.schemas import DenseRagResponse, MultiHopPlan

FIXTURE = Path(__file__).parent / 'tests/fixtures/multihop_responses.json'


def fixture_data():
    return json.loads(FIXTURE.read_text())


def settings():
    import yaml
    path = Path(__file__).resolve().parents[1] / 'config/retrieval.yml'
    return yaml.safe_load(path.read_text())['multihop']


class TaskModelClient:
    """Task recordings at Anchor's existing complete(prompt, system=...) boundary."""

    def __init__(self, mode=None, delegate=None, data=None):
        self.mode = mode or os.environ.get('ANCHOR_LLM_MODE', 'offline')
        if self.mode not in ('offline', 'live'):
            raise ValueError('invalid_llm_mode')
        self.calls = []
        self.data = data if data is not None else fixture_data()
        self.delegate = delegate
        if self.mode == 'live' and delegate is None:
            from anchor_ai.llm_client import LlmClient
            self.delegate = LlmClient()

    async def complete(self, prompt, system=None, max_tokens=256):
        self.calls.append(dict(prompt=prompt, system=system, max_tokens=max_tokens))
        if self.mode == 'live':
            return await self.delegate.complete(prompt, system=system, max_tokens=max_tokens)
        payload = json.loads(prompt)
        if system == MULTIHOP_PLAN_INSTRUCTION:
            return json.dumps(self.data['plan'])
        sources = payload['evidence']['sources']
        by_id = {source['id']: source['text'] for source in sources}
        required = self.data['synthesis']['citations']
        originals = {doc['id']: doc['text'] for doc in self.data['documents']}
        supported = all(i in by_id and by_id[i] == originals[i] for i in required)
        key = 'synthesis' if supported else 'abstention'
        return json.dumps(self.data[key])


def model_client(mode=None):
    from anchor_ai.rag import ExistingClientAdapter
    return ExistingClientAdapter(TaskModelClient(mode=mode))


class FixtureRetriever:
    """Recorded corpus with explicit synthetic ranking, not a search benchmark."""

    def __init__(self, tenant_id=1, data=None):
        self.data = data if data is not None else fixture_data()
        self.tenant_id = tenant_id
        self.generation_id = self.data['generation_id']
        self.calls = []

    async def retrieve(self, query, top_k=1):
        self.calls.append(query)
        docs = [dict(doc) for doc in self.data['documents']
                if doc['tenant_id'] == self.tenant_id]
        lower = query.lower()

        def score(doc):
            text = doc['text'].lower()
            if 'order policy' in lower:
                return 100 if 'eligibility class =' in text else 0
            if lower.startswith('customcare refund eligibility'):
                return 100 if text.startswith('customcare refund eligibility:') else 0
            return 100 if 'general refund faq' in text else 0

        docs.sort(key=score, reverse=True)
        return dict(tenant_id=self.tenant_id, generation_id=self.generation_id,
                    hits=[dict(doc, score=float(score(doc))) for doc in docs[:top_k]])


class MultiHopRuntime:
    def __init__(self, retriever, client, emit=None, evidence_budget=2400,
                 max_hops=2):
        if type(max_hops) is not int or not 1 <= max_hops <= 3:
            raise ValueError('invalid_hop_limit')
        self.retriever = retriever
        self.client = client
        self.emit = emit or (lambda record: None)
        self.evidence_budget = evidence_budget
        self.max_hops = max_hops

    async def answer(self, question, k=1, *, request_id, trace_id):
        from anchor_ai.rag import ABSTENTION, pack_evidence
        result = dict(reply=ABSTENTION, citations=[], abstained=True)
        if len(question.encode('utf-8')) > 1800:
            raise ValueError('current_question_budget_exceeded')
        messages = [dict(role='system', content=MULTIHOP_PLAN_INSTRUCTION),
                    dict(role='user', content=json.dumps(dict(
                        question=question, max_hops=self.max_hops)))]
        try:
            raw = await self.client.complete(messages)
            plan = MultiHopPlan.model_validate_json(raw)
        except (ValueError, TypeError):
            return result
        if len(plan.queries) > self.max_hops:
            return result
        scope = None
        accumulated = {}
        trajectory = []
        required_groups = []
        complete = True
        for number, template in enumerate(plan.queries, 1):
            query = template
            binding_source = None
            if '{eligibility_class}' in template:
                bindings = []
                for source in accumulated.values():
                    match = re.search(r'eligibility class = ([A-Za-z0-9_-]+)\.',
                                      source['text'])
                    if match:
                        bindings.append((match.group(1), source['id']))
                if len({value for value, _ in bindings}) != 1:
                    complete = False
                    break
                value, binding_source = bindings[0]
                query = template.replace('{eligibility_class}', value)
            if len(query.encode('utf-8')) > 300:
                complete = False
                break
            retrieved = await self.retriever.retrieve(query, k)
            current = (retrieved['tenant_id'], retrieved['generation_id'])
            if current[0] != self.retriever.tenant_id:
                raise ValueError('runtime_scope_mismatch')
            if scope is None:
                scope = current
            if current != scope:
                raise ValueError('retrieval_scope_changed')
            hits = retrieved['hits']
            if any(hit['tenant_id'] != scope[0] for hit in hits):
                raise ValueError('foreign_evidence')
            packed = pack_evidence(hits, self.evidence_budget)
            sources = packed['sources']
            trajectory.append(dict(hop=number, query=query,
                                   binding_source_id=binding_source,
                                   passages=sources))
            if not sources:
                complete = False
                break
            required_groups.append({source['id'] for source in sources})
            for source in sources:
                old = accumulated.get(source['id'])
                if old is not None and old['text'] != source['text']:
                    raise ValueError('chunk_content_changed')
                accumulated[source['id']] = source
        evidence = pack_evidence(list(accumulated.values()), self.evidence_budget)
        allowed = {source['id'] for source in evidence['sources']}
        if complete and len(trajectory) == len(plan.queries) and allowed:
            messages = [dict(role='system', content=MULTIHOP_SYNTHESIS_INSTRUCTION),
                        dict(role='user', content=json.dumps(dict(
                            question=question, evidence=evidence)))]
            try:
                raw = await self.client.complete(messages)
                decoded = DenseRagResponse.model_validate_json(raw).model_dump()
                citations = decoded['citations']
                valid = (not decoded['abstained'] and bool(citations)
                         and len(citations) == len(set(citations))
                         and set(citations) <= allowed
                         and all(set(citations) & group for group in required_groups))
                if valid:
                    result = decoded
            except (ValueError, TypeError):
                pass
        self.emit(dict(event='multihop_rag', request_id=request_id,
                       trace_id=trace_id, tenant_id=scope[0] if scope else None,
                       generation_id=scope[1] if scope else None,
                       planned_hops=len(plan.queries), executed_hops=len(trajectory),
                       chunk_ids=sorted(allowed), citations=result['citations'],
                       abstained=result['abstained']))
        # Keep passage text in an explicitly requested local demonstration, not logs.
        self.last_trajectory = trajectory
        return result


def fixture_runtime(tenant_id=1, emit=None):
    from anchor_ai.rag import RagRuntime
    config = settings()
    return RagRuntime(FixtureRetriever(tenant_id), model_client(), emit=emit,
                      evidence_budget=config['evidence_budget_bytes'])
