import json
import time

from anchor_ai.context import pack_retrieved_evidence
from anchor_ai.prompts import DENSE_RAG_INSTRUCTION

ABSTENTION = 'I could not find a supporting policy in your documents.'
INSTRUCTION = DENSE_RAG_INSTRUCTION


def pack_evidence(hits, budget_bytes=600):
    return pack_retrieved_evidence(hits, budget_bytes)


class RecordedGenerationClient:
    """Deterministic evidence reader, not a model-quality evaluation."""

    def __init__(self):
        self.calls = []

    async def complete(self, messages):
        self.calls.append(messages)
        data = json.loads(messages[-1]['content'])
        if 'refund' in data['question'].lower():
            for source in data['evidence']['sources']:
                if 'Refunds require review.' in source['text']:
                    return json.dumps(dict(
                        reply=source['text'], citations=[source['id']], abstained=False))
        return json.dumps(dict(reply=ABSTENTION, citations=[], abstained=True))


class RagRuntime:
    def __init__(self, retriever, client, emit=None, evidence_budget=600):
        self.retriever = retriever
        self.client = client
        self.emit = emit or (lambda record: None)
        self.evidence_budget = evidence_budget

    async def answer(self, question, k=3, *, request_id, trace_id):
        started = time.monotonic()
        if len(question.encode('utf-8')) > 1800:
            raise ValueError('current_question_budget_exceeded')
        hits = self.retriever.search(question, k)
        evidence = pack_evidence(hits, self.evidence_budget)
        messages = [dict(role='system', content=INSTRUCTION),
                    dict(role='user', content=json.dumps(
                        {'question': question, 'evidence': evidence}, ensure_ascii=False))]
        allowed = {source['id'] for source in evidence['sources']}
        result = dict(reply=ABSTENTION, citations=[], abstained=True)
        generation_started = time.monotonic()
        if allowed:
            try:
                raw = await self.client.complete(messages)
                decoded = json.loads(raw)
                valid = (isinstance(decoded, dict)
                         and set(decoded) == {'reply', 'citations', 'abstained'}
                         and isinstance(decoded['reply'], str)
                         and 0 < len(decoded['reply']) <= 4000
                         and type(decoded['abstained']) is bool
                         and isinstance(decoded['citations'], list)
                         and all(type(i) is int and i in allowed
                                 for i in decoded['citations'])
                         and len(set(decoded['citations'])) == len(decoded['citations']))
                if valid and not decoded['abstained'] and decoded['citations']:
                    result = decoded
            except (ValueError, TypeError, KeyError):
                pass
        ended = time.monotonic()
        trace = dict(event='dense_rag', request_id=request_id, trace_id=trace_id,
                     **self.retriever.last_trace,
                     evidence_chunk_ids=sorted(allowed),
                     evidence_bytes=len(json.dumps(evidence, ensure_ascii=False).encode('utf-8')),
                     citations=result['citations'], abstained=result['abstained'],
                     generation_ms=round((ended - generation_started) * 1000, 3),
                     latency_ms=round((ended - started) * 1000, 3))
        self.emit(trace)
        return result


class ExistingClientAdapter:
    """Reuse Anchor's provider client; do not create a second SDK boundary."""

    def __init__(self, client):
        self.client = client

    async def complete(self, messages):
        return await self.client.complete(messages[-1]['content'],
                                          system=messages[0]['content'], max_tokens=256)


def fixture_runtime(tenant_id=1, emit=None):
    from anchor_ai.dense_retriever import DenseRetriever, fixture_dependencies
    generation, embeddings = fixture_dependencies()
    return RagRuntime(DenseRetriever(generation, tenant_id, embeddings),
                      RecordedGenerationClient(), emit=emit)


def published_runtime(snapshot, tenant_id, embedding_client, emit=None):
    from anchor_ai.dense_retriever import DenseRetriever
    from anchor_ai.llm_client import LlmClient
    return RagRuntime(DenseRetriever(snapshot, tenant_id, embedding_client),
                      ExistingClientAdapter(LlmClient()), emit=emit)
