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

    async def answer(self, question, k=3, *, request_id, trace_id,
                     mode='single', max_hops=None, max_iterations=None,
                     max_tool_calls=None):
        if mode not in ('single', 'multi', 'agent'):
            raise ValueError('invalid_rag_mode')
        if mode == 'agent':
            from anchor_ai.agentic_rag import AgenticRuntime, settings as agent_settings
            runtime = AgenticRuntime(self.retriever, emit=self.emit,
                evidence_budget=min(self.evidence_budget, agent_settings()['evidence_budget_bytes']),
                max_iterations=max_iterations, max_tool_calls=max_tool_calls)
            result = await runtime.answer(question, k, request_id=request_id, trace_id=trace_id)
            self.last_trajectory = runtime.last_trajectory
            return result
        if mode == 'multi':
            from anchor_ai.multihop import MultiHopRuntime, settings
            config = settings()
            limit = config['max_hops']
            if max_hops is not None:
                limit = min(limit, max_hops)
            runtime = MultiHopRuntime(self.retriever, self.client, emit=self.emit,
                                      evidence_budget=self.evidence_budget,
                                      max_hops=limit)
            result = await runtime.answer(question, k, request_id=request_id,
                                          trace_id=trace_id)
            self.last_trajectory = getattr(runtime, 'last_trajectory', [])
            return result
        started = time.monotonic()
        if len(question.encode('utf-8')) > 1800:
            raise ValueError('current_question_budget_exceeded')
        if hasattr(self.retriever, 'retrieve'):
            retrieved = await self.retriever.retrieve(question, k)
            hits = retrieved['hits']
            retrieval_trace = dict(tenant_id=retrieved['tenant_id'],
                                   generation_id=retrieved['generation_id'],
                                   chunk_ids=[hit['id'] for hit in hits])
        else:
            hits = self.retriever.search(question, k)
            retrieval_trace = self.retriever.last_trace
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
                     **retrieval_trace,
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
    from anchor_ai.dense_retriever import DenseRetrieveriever
    from anchor_ai.llm_client import LlmClient
    return RagRuntime(DenseRetrieveriever(snapshot, tenant_id, embedding_client),
                      ExistingClientAdapter(LlmClient()), emit=emit)


def hybrid_settings():
    from pathlib import Path
    import yaml
    path = Path(__file__).resolve().parents[1] / 'config/retrieval.yml'
    return yaml.safe_load(path.read_text())['hybrid']


def fixture_hybrid_runtime(tenant_id):
    from anchor_ai.hybrid_retriever import RecordedHybridClient
    client = RecordedHybridClient()
    client.acquire_snapshot(tenant_id)
    settings = hybrid_settings()
    return published_hybrid_runtime(tenant_id, client.acquire_snapshot,
                                   client.dense_search, client.sparse_search,
                                   **{key: settings[key] for key in
                                      ('candidate_depth', 'rrf_k', 'deadline_seconds')})


def published_hybrid_runtime(tenant_id, acquire_snapshot, dense_search, sparse_search,
                             **settings):
    from anchor_ai.hybrid_retriever import HybridRuntime
    if 'reranker' not in settings:
        from anchor_ai.reranking import make_reranker
        settings['reranker'] = make_reranker(hybrid_settings().get('reranking', {}))
    return HybridRuntime(tenant_id, acquire_snapshot, dense_search, sparse_search, **settings)


class AnalyzedRuntime:
    def __init__(self, hybrid, analyzer):
        self.hybrid = hybrid
        self.analyzer = analyzer
        self.tenant_id = hybrid.tenant_id

    async def retrieve(self, query, top_k=3):
        return await self.hybrid.retrieve_analyzed(query, self.analyzer, top_k)


def published_analyzed_runtime(tenant_id, acquire_snapshot, dense_search, sparse_search,
                               analyzer=None, **settings):
    from anchor_ai.query_analyzer import make_analyzer
    hybrid = published_hybrid_runtime(tenant_id, acquire_snapshot,
                                     dense_search, sparse_search, **settings)
    return AnalyzedRuntime(hybrid, analyzer or make_analyzer())


def fixture_analyzed_runtime(tenant_id):
    from pathlib import Path
    from anchor_ai.hybrid_retriever import QueryFixtureClient
    from anchor_ai.query_analyzer import QueryAnalyzer, RecordedQueryClient
    path = Path(__file__).resolve().parents[1] / 'eval/datasets/retrieval_queries.json'
    client = QueryFixtureClient(json.loads(path.read_text()))
    settings = hybrid_settings()
    # Offline cold imports and thread scheduling must not test production latency.
    settings['deadline_seconds'] = 10
    runtime = published_analyzed_runtime(tenant_id, client.acquire_snapshot,
        client.dense_search, client.sparse_search,
        analyzer=QueryAnalyzer(RecordedQueryClient()),
        **{key: settings[key] for key in ('candidate_depth', 'rrf_k', 'deadline_seconds')})
    runtime.fixture_client = client
    return runtime


def published_hybrid_rag_runtime(tenant_id, acquire_snapshot, dense_search, sparse_search,
                                 client=None, emit=None, evidence_budget=600, **settings):
    from anchor_ai.llm_client import LlmClient
    hybrid = published_hybrid_runtime(tenant_id, acquire_snapshot,
                                     dense_search, sparse_search, **settings)
    generation = client if client is not None else ExistingClientAdapter(LlmClient())
    return RagRuntime(hybrid, generation, emit=emit, evidence_budget=evidence_budget)


def published_bounded_rag_runtime(tenant_id, acquire_snapshot, dense_search,
                                  sparse_search, client=None, emit=None, **settings):
    from anchor_ai.multihop import model_client, settings as multihop_settings
    snapshot = acquire_snapshot(tenant_id)

    def pinned_snapshot(requested_tenant):
        if requested_tenant != tenant_id:
            raise ValueError('runtime_scope_mismatch')
        return snapshot

    hybrid = published_hybrid_runtime(tenant_id, pinned_snapshot,
                                     dense_search, sparse_search, **settings)
    return RagRuntime(hybrid, client if client is not None else model_client(),
                      emit=emit,
                      evidence_budget=multihop_settings()['evidence_budget_bytes'])


class GraphRagRuntime:
    def __init__(self, retriever, emit=None):
        self.retriever = retriever
        self.emit = emit or (lambda record: None)

    async def answer(self, request, *, request_id, trace_id):
        from anchor_ai.graph_retriever import path_answer
        from anchor_ai.schemas import GraphRelationshipRequest
        if not isinstance(request, GraphRelationshipRequest):
            request = GraphRelationshipRequest.model_validate(request)
        started = time.monotonic()
        report = self.retriever.retrieve(request.question, request.terms)
        self.last_report = report
        result = path_answer(report)
        self.emit(dict(event='graph_rag', request_id=request_id, trace_id=trace_id,
            tenant_id=report['tenant_id'], scope=report['scope'],
            generation_id=report['generation_id'], linked_ids=report['linked_ids'],
            status=report['status'], citations=result['citations'],
            latency_ms=round((time.monotonic() - started) * 1000, 3)))
        return result


def published_graph_rag_runtime(db, tenant_id, scope, emit=None):
    from pathlib import Path
    import yaml
    from anchor_ai.graph_retriever import GraphRetriever
    from anchor_ai.schemas import GraphLimits
    path = Path(__file__).resolve().parents[1] / 'config/retrieval.yml'
    limits = GraphLimits.model_validate(yaml.safe_load(path.read_text())['graph'])
    return GraphRagRuntime(GraphRetriever(db, tenant_id, scope,
        **limits.model_dump()), emit=emit)


def published_agentic_rag_runtime(tenant_id, acquire_snapshot, dense_search,
                                  sparse_search, policy=None, emit=None, **settings):
    from anchor_ai.agentic_rag import AgenticRuntime
    bounded = published_bounded_rag_runtime(tenant_id, acquire_snapshot,
        dense_search, sparse_search, emit=emit, **settings)
    return AgenticRuntime(bounded.retriever, policy=policy, emit=emit,
                          tenant_id=tenant_id)


class ReleaseRetriever:
    """Bounded composition shared by the flagged release path and its gate.

    The injected search client performs I/O. Follow-up queries come from retrieved
    evidence, not from the gate's expected trajectory or score labels.
    """
    def __init__(self, ticket, configuration, search):
        self.ticket = ticket
        self.configuration = configuration
        self.search = search
        self.trajectory = []
        self.contexts = []

    async def retrieve(self, question, top_k=3):
        import re
        cfg = self.configuration
        limit = cfg['max_tool_calls'] if cfg['advanced_rag'] else 1
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError('invalid_tool_budget')
        pending, seen, candidates = [question], set(), {}
        calls = 0
        while pending and calls < limit:
            query = pending.pop(0)
            if query in seen:
                continue
            seen.add(query)
            hits = await self.search(query)
            calls += 1
            self.trajectory.append(dict(event='retrieve', query=query,
                                        evidence_ids=[hit['id'] for hit in hits]))
            for hit in hits:
                if (hit['tenant_id'] != self.ticket['tenant_id'] or
                        hit['generation_id'] != self.ticket['generation_id']):
                    raise ValueError('release_scope_mismatch')
                candidates[hit['id']] = hit
                follow_up = hit.get('follow_up')
                if cfg['advanced_rag'] and follow_up and follow_up not in seen:
                    pending.append(follow_up)
        words = set(re.findall(r'[a-z0-9]+', question.lower()))
        ranked = list(candidates.values())
        if cfg['advanced_rag']:
            if cfg['reranker'] != 'offline-term-overlap-v1':
                raise ValueError('unsupported_release_reranker')
            ranked.sort(key=lambda hit: (-len(words & set(re.findall(
                r'[a-z0-9]+', hit['text'].lower()))), hit['id']))
            self.trajectory.append(dict(event='rerank', evidence_ids=[h['id'] for h in ranked]))
        self.contexts = ranked[:top_k]
        self.trajectory.append(dict(event='stop', outcome='budget' if pending else 'complete',
                                    tool_calls=calls))
        return dict(tenant_id=self.ticket['tenant_id'],
                    generation_id=self.ticket['generation_id'], hits=self.contexts)


async def answer_release(ticket, configuration, search, client=None):
    """The gate calls this same flagged entry point; it does not replay answers."""
    import os
    if type(configuration['advanced_rag']) is not bool:
        raise ValueError('invalid_release_flag')
    if client is None:
        if os.getenv('ANCHOR_LLM_PROVIDER', 'offline') == 'offline':
            client = RecordedGenerationClient()
        else:
            from anchor_ai.llm_client import LlmClient
            client = ExistingClientAdapter(LlmClient())
    retriever = ReleaseRetriever(ticket, configuration, search)
    runtime = RagRuntime(retriever, client,
                         evidence_budget=configuration['evidence_budget'])
    result = await runtime.answer(ticket['question'], configuration['top_k'],
                                  request_id=ticket['id'], trace_id=ticket['id'])
    # Capture the packed sources, not candidates discarded by the evidence budget.
    sources = pack_evidence(retriever.contexts, configuration['evidence_budget'])['sources']
    evidence = [dict(source, tenant_id=ticket['tenant_id'],
                     generation_id=ticket['generation_id']) for source in sources]
    return dict(id=ticket['id'], tenant_id=ticket['tenant_id'],
                generation_id=ticket['generation_id'], question=ticket['question'],
                category='release', answer=result['reply'], citations=result['citations'],
                abstained=result['abstained'], evidence=evidence,
                trajectory=retriever.trajectory,
                pipeline='advanced' if configuration['advanced_rag'] else 'legacy')


class RecordedEvidenceClient:
    """Extract supplied evidence; do not represent this as model-quality scoring."""

    def __init__(self):
        self.calls = []

    async def complete(self, messages):
        self.calls.append(messages)
        data = json.loads(messages[-1]['content'])
        sources = data['evidence']['sources']
        if not sources:
            return json.dumps(dict(reply=ABSTENTION, citations=[], abstained=True))
        return json.dumps(dict(
            reply=' '.join(f"{row['text']} [{row['id']}]" for row in sources),
            citations=[row['id'] for row in sources], abstained=False))


def published_query_engine_tool(retriever, tenant_id, client=None, **options):
    from anchor_ai.query_engine import QueryEngineTool
    if client is None:
        from anchor_ai.llm_client import LlmClient
        client = ExistingClientAdapter(LlmClient())
    return QueryEngineTool(retriever, tenant_id, client, **options)
