def transcribe_recorded_call(raw, *, client=None, production=False, policy='silence'):
    """Internal worker seam; ticket callers must use the signed attachment boundary."""
    from anchor_ai.llm_client import transcribe_audio
    return transcribe_audio(raw, client=client, production=production, policy=policy)


def extract_document(raw, *, engine='tesseract', limits=None, vision=None):
    """Internal worker entry point; authorization belongs to the ingestion caller."""
    from anchor_ai.llm_client import extract_document as extract
    return extract(raw, engine=engine, limits=limits, vision=vision)


from anchor_ai.contracts import AgentRequest
from anchor_ai.runtime import ContextError
import hmac
import os
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

try:
    from .contracts import InferRequest, InferResponse
    from .llm_client import LlmClient
    from .provider import ProviderError
    from .schemas import TriageLabels, InvalidModelOutput
    from .observability import span, context
except ImportError:
    from contracts import InferRequest, InferResponse
    from llm_client import LlmClient
    from provider import ProviderError
    from schemas import TriageLabels, InvalidModelOutput
    from observability import span, context

from anchor_ai.injection_filter import screen_customer, REFUSAL
from anchor_ai.jailbreak_guard import evaluate as jailbreak_decision
from anchor_ai.jailbreak_guard import REFUSAL as JAILBREAK_REFUSAL


def require_safe_customer(value):
    if jailbreak_decision(str(value)).refused:
        raise HTTPException(status_code=422, detail='jailbreak_detected',
                            headers={'X-Anchor-Jailbreak': 'refused'})
    safe, flags = screen_customer(str(value))
    if flags:
        raise HTTPException(status_code=422, detail='injection_detected',
                            headers={'X-Anchor-Injection': 'instruction_override'})
    return value if safe == str(value) else safe


@asynccontextmanager
async def lifespan(application):
    if os.environ.get('ANCHOR_DEPLOYMENT_CONFIG'):
        from ops.deploy.validate_manifest import selected_route
        selected_route(require_secret=True)
    from eval.eval_endpoint import ReleaseRuntime
    from anchor_ai.tools import registry_for
    from anchor_ai.mcp import McpClient
    mcp_client = (McpClient(os.environ.get('ANCHOR_MCP_SERVER_ID', 'local_orders'))
                  if os.environ.get('ANCHOR_MCP_ENABLED') == '1' else None)
    application.state.agent_tool_registry_factory = lambda backend: registry_for(backend, mcp_client=mcp_client)
    factory = getattr(application.state, 'rag_release_runtime_factory', ReleaseRuntime)
    application.state.rag_release_runtime = factory()
    batch_factory = getattr(application.state, 'batch_runtime_factory', None)
    if batch_factory is not None:
        application.state.batch_runtime = batch_factory()
    try:
        yield
    finally:
        batch_runtime = getattr(application.state, 'batch_runtime', None)
        if batch_runtime is not None:
            await batch_runtime.close()
            del application.state.batch_runtime
        del application.state.rag_release_runtime
        del application.state.agent_tool_registry_factory


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix='/v1')
from anchor_ai.observability import LatencyBoundary, model_call
app.add_middleware(LatencyBoundary)
from anchor_ai.tenant_scope import TenantScopeMiddleware, ScopeError
app.add_middleware(TenantScopeMiddleware)


@app.exception_handler(ScopeError)
async def tenant_scope_failed(request: Request, error: ScopeError):
    return JSONResponse(status_code=403, content={'detail': 'tenant_scope_rejected'})


@app.post('/v1/agent')
async def bounded_agent(body: AgentRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None,
    x_anchor_context: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth')
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        raise HTTPException(status_code=401, detail='service_auth_failed')
    if not x_anchor_context:
        raise HTTPException(status_code=403, detail='missing_execution_context')
    runtime = getattr(app.state, 'bounded_agent_runtime', None)
    if runtime is None:
        raise HTTPException(status_code=503, detail='agent_not_configured')
    payload = body.model_dump()
    if payload.get('operation', 'start') == 'start':
        decision = jailbreak_decision(payload.get('message'))
        if decision.refused:
            return JSONResponse(content={'status': 'denied', 'reply': JAILBREAK_REFUSAL,
                                         'calls': 0}, headers={'X-Anchor-Jailbreak': 'refused'})
    if isinstance(payload.get('message'), str):
        payload['message'] = require_safe_customer(payload['message']) if payload['message'] else payload['message']
    try:
        return await runtime.execute(x_anchor_context, payload)
    except ContextError:
        raise HTTPException(status_code=403, detail='execution_context_rejected') from None
    except ValueError:
        raise HTTPException(status_code=422, detail='agent_request_rejected') from None


@app.middleware('http')
async def telemetry(request: Request, call_next):
    with span('python.infer', context(request.headers), stage='validation') as item:
        request.state.telemetry = item
        response = await call_next(request)
        item.set(status_code=response.status_code)
        if response.status_code >= 400 and item.record['status'] != 'error':
            item.fail('internal_error', response.status_code)
        response.headers['X-Request-Id'] = item.record['request_id']
        return response


@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, error: RequestValidationError):
    request.state.telemetry.set(stage='validation')
    request.state.telemetry.fail('validation_failed', 422)
    return JSONResponse(status_code=422, content={'detail': 'validation_failed'})


@router.post('/infer', response_model=InferResponse, response_model_exclude_none=True)
async def infer(body: InferRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None) -> InferResponse:
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    supplied = (x_anchor_secret or '').encode('utf-8')
    if not hmac.compare_digest(supplied, expected.encode('utf-8')):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    if jailbreak_decision(str(body.message), body.history or []).refused:
        return JSONResponse(content={'reply': JAILBREAK_REFUSAL}, headers={'X-Anchor-Jailbreak': 'refused'})
    safe, flags = screen_customer(str(body.message))
    if flags:
        return JSONResponse(content={'reply': REFUSAL}, headers={'X-Anchor-Injection': 'instruction_override'})
    message = require_safe_customer(body.message)
    item.set(stage='provider')
    try:
        if os.environ.get('ANCHOR_DEPLOYMENT_CONFIG'):
            from ops.deploy.validate_manifest import inference_client
            client = inference_client()
        else:
            client = LlmClient()
        return await model_call(client.complete_result, message)
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None


@router.post('/structured_replies', response_model=TriageLabels)
async def structured_reply(body: InferRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode('utf-8'), expected.encode('utf-8')):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    message = require_safe_customer(body.message)
    item.set(stage='provider')
    try:
        return await LlmClient().complete_structured(message)
    except InvalidModelOutput as error:
        item.fail('validation_failed', 422)
        return JSONResponse(status_code=422, content={'detail': error.detail()})
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None


@router.post('/infer/stream')
async def infer_stream(body: InferRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    body.message = require_safe_customer(body.message)
    from anchor_ai.streaming import response_for
    return response_for(body, request)


from anchor_ai.schemas import ChunkEmbedRequest, IngestionNode


def chunk_embed(body: ChunkEmbedRequest):
    from anchor_ai.ingestion import sentence_nodes
    from anchor_ai.pii import prepare
    nodes = sentence_nodes(prepare(body.text))
    leaves = [node for node in nodes if node['is_leaf']]
    result = LlmClient(provider='offline').embed_ingestion([node['text'] for node in leaves])
    if len(result['vectors']) != len(leaves):
        raise ValueError('embedding_count_mismatch')
    for node, vector in zip(leaves, result['vectors']):
        node.update(embedding=vector, embedding_model=result['model'],
                    embedding_revision=result['revision'], embedding_dimensions=result['dimensions'])
    return [IngestionNode.model_validate(node) for node in nodes]


@router.post('/chunk_embed', response_model=list[IngestionNode])
def chunk_embed_route(body: ChunkEmbedRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    try:
        return chunk_embed(body)
    except Exception:
        item.fail('ingestion_failed', 422)
        raise HTTPException(status_code=422, detail='ingestion_failed') from None


from anchor_ai.schemas import DenseRagRequest, DenseRagResponse
from anchor_ai.observability import emit_retrieval


@router.post('/dense_rag', response_model=DenseRagResponse)
async def dense_rag(body: DenseRagRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    body.message = require_safe_customer(body.message)
    factory = getattr(app.state, 'retrieval_runtime_factory', None)
    if factory is None:
        if os.environ.get('ANCHOR_DENSE_FIXTURE') != '1' or body.tenant_id != 1:
            item.fail('retrieval_not_configured', 503)
            raise HTTPException(status_code=503, detail='retrieval_not_configured')
        from anchor_ai.rag import fixture_runtime
        factory = fixture_runtime
    item.set(stage='retrieval')
    try:
        runtime = factory(tenant_id=body.tenant_id, emit=emit_retrieval)
        return await runtime.answer(body.message,
            request_id=item.record['request_id'], trace_id=item.record['trace_id'])
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None
    except ValueError:
        item.fail('retrieval_failed', 422)
        raise HTTPException(status_code=422, detail='retrieval_failed') from None


from anchor_ai.schemas import HybridRequest, HybridResponse


@router.post('/retrieve/hybrid', response_model=HybridResponse)
async def hybrid_retrieve(body: HybridRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    body.query = require_safe_customer(body.query)
    factory = getattr(app.state, 'hybrid_runtime_factory', None)
    if factory is None:
        if os.environ.get('ANCHOR_HYBRID_FIXTURE') != '1' or body.tenant_id != 1:
            item.fail('retrieval_not_configured', 503)
            raise HTTPException(status_code=503, detail='retrieval_not_configured')
        from anchor_ai.rag import fixture_hybrid_runtime
        factory = fixture_hybrid_runtime
    item.set(stage='retrieval')
    try:
        runtime = factory(tenant_id=body.tenant_id)
        if runtime.tenant_id != body.tenant_id:
            raise ValueError('runtime_scope_mismatch')
        return await runtime.retrieve(body.query, body.top_k)
    except TimeoutError:
        item.fail('retrieval_deadline', 504)
        raise HTTPException(status_code=504, detail='retrieval_deadline') from None
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None
    except Exception:
        item.fail('retrieval_failed', 422)
        raise HTTPException(status_code=422, detail='retrieval_failed') from None


app.include_router(router)
app.add_api_route('/infer', infer, methods=['POST'], response_model=InferResponse,
                  response_model_exclude_none=True)
app.add_api_route('/retrieve/hybrid', hybrid_retrieve, methods=['POST'], response_model=HybridResponse)
from anchor_ai.schemas import AnalyzedResponse


@app.post('/v1/retrieve/analyzed', response_model=AnalyzedResponse)
async def analyzed_retrieve(body: HybridRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    body.query = require_safe_customer(body.query)
    factory = getattr(app.state, 'analyzed_runtime_factory', None)
    if factory is None:
        if os.environ.get('ANCHOR_QUERY_FIXTURE') != '1' or body.tenant_id != 1:
            item.fail('retrieval_not_configured', 503)
            raise HTTPException(status_code=503, detail='retrieval_not_configured')
        from anchor_ai.rag import fixture_analyzed_runtime
        factory = fixture_analyzed_runtime
    item.set(stage='retrieval')
    try:
        runtime = factory(tenant_id=body.tenant_id)
        if runtime.tenant_id != body.tenant_id:
            raise ValueError('runtime_scope_mismatch')
        return await runtime.retrieve(body.query, body.top_k)
    except TimeoutError:
        item.fail('retrieval_deadline', 504)
        raise HTTPException(status_code=504, detail='retrieval_deadline') from None
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None
    except Exception:
        item.fail('retrieval_failed', 422)
        raise HTTPException(status_code=422, detail='retrieval_failed') from None


from anchor_ai.schemas import RagRequest


@app.post('/v1/rag', response_model=DenseRagResponse)
async def bounded_rag(body: RagRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    body.message = require_safe_customer(body.message)
    factory = getattr(app.state, 'bounded_rag_runtime_factory', None)
    if factory is None:
        if os.environ.get('ANCHOR_MULTIHOP_FIXTURE') != '1' or body.tenant_id != 1:
            item.fail('retrieval_not_configured', 503)
            raise HTTPException(status_code=503, detail='retrieval_not_configured')
        from anchor_ai.multihop import fixture_runtime
        factory = fixture_runtime
    item.set(stage='retrieval')
    try:
        runtime = factory(tenant_id=body.tenant_id)
        if runtime.retriever.tenant_id != body.tenant_id:
            raise ValueError('runtime_scope_mismatch')
        return await runtime.answer(body.message, body.top_k, mode=body.mode,
            request_id=item.record['request_id'], trace_id=item.record['trace_id'])
    except TimeoutError:
        item.fail('retrieval_deadline', 504)
        raise HTTPException(status_code=504, detail='retrieval_deadline') from None
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None
    except ValueError:
        item.fail('retrieval_failed', 422)
        raise HTTPException(status_code=422, detail='retrieval_failed') from None


from anchor_ai.schemas import ReleaseRagRequest, ReleaseRagResponse


@app.get('/healthz')
def healthz():
    # Validate deployment policy before the legacy retrieval-only readiness path.
    if os.environ.get('ANCHOR_DEPLOYMENT_CONFIG'):
        from ops.deploy.validate_manifest import selected_route
        try:
            deployment_route = selected_route(require_secret=True)
        except ValueError:
            raise HTTPException(status_code=503, detail='deployment_not_ready') from None
        if not hasattr(app.state, 'rag_release_runtime'):
            raise HTTPException(status_code=503, detail='retrieval_not_ready')
        return {'ready': True, 'deployment_route': deployment_route['name'],
                'provider': deployment_route['provider']}
    # Deployment readiness must validate routing even when called directly.
    if os.environ.get('ANCHOR_DEPLOYMENT_CONFIG'):
        from ops.deploy.validate_manifest import selected_route
        try:
            route = selected_route(require_secret=True)
        except ValueError:
            raise HTTPException(status_code=503, detail='deployment_not_ready') from None
        if not hasattr(app.state, 'rag_release_runtime'):
            raise HTTPException(status_code=503, detail='retrieval_not_ready')
        return {'ready': True, 'deployment_route': route['name'],
                'provider': route['provider']}
    if not hasattr(app.state, 'rag_release_runtime'):
        raise HTTPException(status_code=503, detail='retrieval_not_ready')
    if os.environ.get('ANCHOR_DEPLOYMENT_CONFIG'):
        from ops.deploy.validate_manifest import selected_route
        try:
            route = selected_route(require_secret=True)
        except ValueError:
            raise HTTPException(status_code=503, detail='deployment_not_ready') from None
        return {'ready': True, 'deployment_route': route['name'],
                'provider': route['provider']}
    return {'ready': True}


@app.post('/rag', response_model=ReleaseRagResponse)
async def release_rag(body: ReleaseRagRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    require_safe_customer(body.question)
    runtime = getattr(app.state, 'rag_release_runtime', None)
    if runtime is None:
        item.fail('retrieval_not_configured', 503)
        raise HTTPException(status_code=503, detail='retrieval_not_configured')
    item.set(stage='retrieval')
    try:
        return await runtime.answer(body)
    except TimeoutError:
        item.fail('retrieval_deadline', 504)
        raise HTTPException(status_code=504, detail='retrieval_deadline') from None
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None
    except ValueError:
        item.fail('retrieval_failed', 422)
        raise HTTPException(status_code=422, detail='retrieval_failed') from None


from anchor_ai.chains import TicketResponse, run_ticket_chain
from anchor_ai.prompts import RegistryLookupError
from langchain_core.exceptions import OutputParserException


@app.post('/v1/chain_replies', response_model=TicketResponse)
async def chain_reply(body: InferRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    from anchor_ai.prompts import RegistryLookupError
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        raise HTTPException(status_code=401, detail='service_auth_failed')
    if os.environ.get('ANCHOR_REPLY_PIPELINE') != 'lcel':
        raise HTTPException(status_code=404, detail='pipeline_not_enabled')
    body.message = require_safe_customer(body.message)
    item.set(stage='chain')
    try:
        return await run_ticket_chain(body.model_dump(), trace={
            'request_id': item.record['request_id'], 'trace_id': item.record['trace_id']})
    except RegistryLookupError:
        item.fail('registry_lookup_failed', 503)
        raise HTTPException(status_code=503, detail='registry_lookup_failed') from None
    except (OutputParserException, InvalidModelOutput, ValueError):
        item.fail('validation_failed', 422)
        raise HTTPException(status_code=422, detail='validation_failed') from None
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None


@app.post('/v1/multi_agent')
async def multi_agent(body: AgentRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None,
    x_anchor_context: Annotated[str | None, Header()] = None):
    from anchor_ai.orchestration import run_milestone, MilestoneUnavailable
    item = request.state.telemetry
    item.set(stage='auth')
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        raise HTTPException(status_code=401, detail='service_auth_failed')
    if not x_anchor_context:
        raise HTTPException(status_code=403, detail='missing_execution_context')
    payload = body.model_dump()
    if isinstance(payload.get('message'), str):
        payload['message'] = require_safe_customer(payload['message']) if payload['message'] else payload['message']
    try:
        with span('python.multi_agent', stage='orchestration'):
            return await run_milestone(getattr(app.state, 'multi_agent_runtime', None), x_anchor_context, payload)
    except MilestoneUnavailable:
        item.fail('orchestration_not_configured', 503)
        raise HTTPException(status_code=503, detail='orchestration_not_configured') from None
    except ContextError:
        raise HTTPException(status_code=403, detail='execution_context_rejected') from None
    except ValueError:
        raise HTTPException(status_code=422, detail='agent_request_rejected') from None


from anchor_ai.schemas import CitedRagEnvelope


@app.post('/v1/rag/cited', response_model=CitedRagEnvelope)
async def cited_rag(body: DenseRagRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        raise HTTPException(status_code=401, detail='service_auth_failed')
    factory = getattr(app.state, 'cited_rag_runtime_factory', None)
    if factory is None:
        raise HTTPException(status_code=503, detail='cited_generation_not_configured')
    body.message = require_safe_customer(body.message)
    try:
        runtime = factory(tenant_id=body.tenant_id)
        if runtime.tenant_id != body.tenant_id:
            raise ValueError('runtime_scope_mismatch')
        item.set(stage='retrieval')
        return await runtime.answer(body.message)
    except ProviderError as error:
        raise HTTPException(status_code=error.status, detail=error.detail()) from None
    except TimeoutError:
        raise HTTPException(status_code=504, detail='retrieval_deadline') from None
    except ValueError:
        raise HTTPException(status_code=422, detail='cited_generation_rejected') from None


from anchor_ai.vision import register_vision
register_vision(app)


def extract_ticket_attachment(raw, *, kind, token, secret, expected_scope, adapter=None, production=False):
    """Internal entry point; Rails authorizes the ticket before signing identity."""
    from anchor_ai.multimodal_context import verify_attachment_context
    from anchor_ai.attachment_evidence import extract
    identity = verify_attachment_context(token, secret, expected_scope)
    return extract(raw, kind=kind, identity=identity,
                   expected_tenant=identity.tenant_id, expected_conversation=identity.conversation_id,
                   adapter=adapter, production=production)


from anchor_ai.realtime import configure_product
configure_product(app)


async def public_faq_answer(token, query, *, path, runtime=None):
    """Internal chat/RAG seam. The runtime owns verified scope, not request JSON."""
    runtime = runtime or getattr(app.state, 'semantic_answer_runtime', None)
    if runtime is None:
        raise HTTPException(status_code=503, detail='semantic_answer_not_configured')
    query = require_safe_customer(query)
    return await runtime.answer(token, query, path=path)


from anchor_ai.batching import register_batch_routes
register_batch_routes(app)


from anchor_ai.pii import RedactionUnavailable, redact_value
from pydantic import BaseModel, ConfigDict


class OnlineQualityRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    payload: dict


@app.post('/v1/quality/score')
def online_quality_route(body: OnlineQualityRequest,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        raise HTTPException(status_code=401, detail='service_auth_failed')
    from anchor_ai.online_quality import score
    try:
        return score(redact_value(body.payload))
    except Exception:
        raise HTTPException(status_code=503, detail='quality_scoring_unavailable') from None


class RedactionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    value: object


@app.exception_handler(RedactionUnavailable)
async def redaction_unavailable(request: Request, error: RedactionUnavailable):
    request.state.telemetry.fail('pii_redaction_unavailable', 503)
    return JSONResponse(status_code=503, content={'detail': 'pii_redaction_unavailable'})


@app.post('/v1/pii/redact')
async def redact_route(body: RedactionRequest, request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None):
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        raise HTTPException(status_code=401, detail='service_auth_failed')
    import json
    if len(json.dumps(body.value, ensure_ascii=True, allow_nan=False)) > 262144:
        raise RedactionUnavailable()
    with span('python.pii', stage='redaction'):
        return {'value': redact_value(body.value)}
