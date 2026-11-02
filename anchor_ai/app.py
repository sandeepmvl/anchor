import hmac
import os
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

app = FastAPI()
router = APIRouter(prefix='/v1')


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
    # Do not serialize error.errors(): it includes the rejected input.
    return JSONResponse(status_code=422, content={'detail': 'validation_failed'})


@router.post('/infer', response_model=InferResponse, response_model_exclude_none=True)
async def infer(
    body: InferRequest,
    request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None,
) -> InferResponse:
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
    item.set(stage='provider')
    try:
        return await LlmClient().complete_result(body.message)
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None


@router.post('/structured_replies', response_model=TriageLabels)
async def structured_reply(
    body: InferRequest,
    request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None,
):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode('utf-8'), expected.encode('utf-8')):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
    item.set(stage='provider')
    try:
        return await LlmClient().complete_structured(body.message)
    except InvalidModelOutput as error:
        item.set(stage='validation')
        item.fail('invalid_model_output', 422)
        return JSONResponse(status_code=422, content={'detail': error.detail()})
    except ProviderError as error:
        item.fail(error.code, error.status)
        raise HTTPException(status_code=error.status, detail=error.detail()) from None


@router.post('/infer/stream')
async def infer_stream(
    body: InferRequest,
    request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None,
):
    item = request.state.telemetry
    item.set(stage='auth', tenant_id=body.tenant_id, conversation_id=body.conversation_id)
    expected = os.environ.get('ANCHOR_AI_SECRET')
    if not expected:
        item.fail('service_not_configured', 503)
        raise HTTPException(status_code=503, detail='service_not_configured')
    if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
        item.fail('service_auth_failed', 401)
        raise HTTPException(status_code=401, detail='service_auth_failed')
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
def chunk_embed_route(
    body: ChunkEmbedRequest,
    request: Request,
    x_anchor_secret: Annotated[str | None, Header()] = None,
):
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


app.include_router(router)
app.add_api_route('/infer', infer, methods=['POST'], response_model=InferResponse,
                  response_model_exclude_none=True)
