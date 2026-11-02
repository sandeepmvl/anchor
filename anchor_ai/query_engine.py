import json
import math
from numbers import Integral, Real
import uuid

from llama_index.core.base.response.schema import Response
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.schema import NodeWithScore, TextNode

from anchor_ai.rag import RagRuntime, pack_evidence


TOOL_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'query_policy',
        'description': 'Synthesize a policy answer with database chunk citations.',
        'parameters': {
            'type': 'object',
            'properties': {'query': {'type': 'string'}},
            'required': ['query'],
            'additionalProperties': False,
        },
    },
}


class AnchorRetriever(BaseRetriever):
    def __init__(self, runtime, tenant_id, top_k):
        super().__init__()
        self.runtime = runtime
        self.tenant_id = tenant_id
        self.top_k = top_k
        self.result = None

    def _retrieve(self, query_bundle):
        raise RuntimeError('Use the asynchronous Anchor tool')

    async def _aretrieve(self, query_bundle):
        if hasattr(self.runtime, 'retrieve'):
            result = await self.runtime.retrieve(query_bundle.query_str, self.top_k)
        else:
            # Capture dense evidence and its trusted trace without yielding.
            hits = self.runtime.search(query_bundle.query_str, self.top_k)
            trace = dict(self.runtime.last_trace)
            if trace['tenant_id'] != self.tenant_id or not trace['generation_id']:
                raise ValueError('query_engine_scope_mismatch')
            normalized = []
            for hit in hits:
                if (hit.get('tenant_id', trace['tenant_id']) != trace['tenant_id']
                        or hit.get('generation_id', trace['generation_id']) != trace['generation_id']):
                    raise ValueError('query_engine_hit_mismatch')
                normalized.append(dict(hit, tenant_id=trace['tenant_id'],
                                       generation_id=trace['generation_id']))
            result = dict(tenant_id=trace['tenant_id'],
                          generation_id=trace['generation_id'], hits=normalized)
        if result['tenant_id'] != self.tenant_id or not result['generation_id']:
            raise ValueError('query_engine_scope_mismatch')
        seen, nodes, canonical = set(), [], []
        for hit in result['hits']:
            chunk_id = hit['id']
            if (isinstance(chunk_id, bool) or not isinstance(chunk_id, Integral)
                    or chunk_id <= 0 or chunk_id in seen
                    or hit['tenant_id'] != self.tenant_id
                    or hit.get('generation_id', result['generation_id']) != result['generation_id']
                    or not isinstance(hit['text'], str)):
                raise ValueError('query_engine_hit_mismatch')
            chunk_id = int(chunk_id)
            score = hit.get('score')
            if score is not None:
                if isinstance(score, bool) or not isinstance(score, Real):
                    raise ValueError('invalid_query_engine_score')
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError('invalid_query_engine_score')
            seen.add(chunk_id)
            canonical.append(dict(id=chunk_id, text=hit['text'], score=score,
                                  tenant_id=self.tenant_id,
                                  generation_id=result['generation_id']))
            nodes.append(NodeWithScore(node=TextNode(
                id_=str(chunk_id), text=hit['text'], metadata={
                    'document_chunk_id': chunk_id,
                    'tenant_id': self.tenant_id,
                    'generation_id': result['generation_id'],
                }), score=score))
        self.result = dict(tenant_id=self.tenant_id,
                           generation_id=result['generation_id'], hits=canonical)
        return nodes


class PinnedEvidence:
    def __init__(self, result):
        self.result = result

    async def retrieve(self, query, top_k=3):
        return self.result


class AnchorSynthesizer(BaseSynthesizer):
    def __init__(self, adapter, client, evidence_budget, emit, request_id, trace_id):
        # MockLLM supplies framework metadata only. Generation uses Anchor's client.
        super().__init__(llm=MockLLM())
        self.adapter = adapter
        self.client = client
        self.evidence_budget = evidence_budget
        self.emit = emit
        self.request_id = request_id
        self.trace_id = trace_id

    def _get_prompts(self):
        return {}

    def _update_prompts(self, prompts):
        if prompts:
            raise ValueError('Anchor owns the synthesis prompt')

    def get_response(self, query_str, text_chunks, **kwargs):
        raise RuntimeError('Use the asynchronous Anchor tool')

    async def aget_response(self, query_str, text_chunks, **kwargs):
        raise RuntimeError('Use asynthesize with identified nodes')

    async def asynthesize(self, query, nodes, additional_source_nodes=None, **kwargs):
        result = self.adapter.result
        runtime = RagRuntime(PinnedEvidence(result), self.client,
                             emit=self.emit, evidence_budget=self.evidence_budget)
        answer = await runtime.answer(query.query_str, self.adapter.top_k,
            request_id=self.request_id, trace_id=self.trace_id)
        supplied = pack_evidence(result['hits'], self.evidence_budget)['sources']
        allowed = {row['id'] for row in supplied}
        sources = [node for node in nodes
                   if node.node.metadata['document_chunk_id'] in allowed]
        return Response(response=json.dumps(answer), source_nodes=sources)


class QueryEngineTool:
    def __init__(self, retriever, tenant_id, client, top_k=3,
                 evidence_budget=16000, emit=None):
        if type(tenant_id) is not int or tenant_id <= 0:
            raise ValueError('invalid_tenant')
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('invalid_top_k')
        if type(evidence_budget) is not int or not 1 <= evidence_budget <= 16000:
            raise ValueError('invalid_evidence_budget')
        self.retriever = retriever
        self.tenant_id = tenant_id
        self.client = client
        self.top_k = top_k
        self.evidence_budget = evidence_budget
        self.emit = emit

    async def __call__(self, query):
        if not isinstance(query, str) or not query.strip() or len(query.encode()) > 1800:
            raise ValueError('invalid_query')
        adapter = AnchorRetriever(self.retriever, self.tenant_id, self.top_k)
        identifier = uuid.uuid4().hex
        synthesizer = AnchorSynthesizer(adapter, self.client, self.evidence_budget,
                                        self.emit, identifier, identifier)
        engine = RetrieverQueryEngine(adapter, response_synthesizer=synthesizer)
        response = await engine.aquery(query)
        decoded = json.loads(response.response)
        result = adapter.result
        return dict(answer=decoded['reply'], citations=decoded['citations'],
                    abstained=decoded['abstained'], tenant_id=self.tenant_id,
                    generation_id=result['generation_id'], source_nodes=[
                        dict(document_chunk_id=int(item.node.node_id),
                             text=item.node.text, score=item.score,
                             tenant_id=self.tenant_id,
                             generation_id=result['generation_id'])
                        for item in response.source_nodes])
