import hmac
import os
from typing import Annotated

from fastapi import APIRouter, FastAPI, Header, HTTPException

try:
    from .contracts import InferRequest, InferResponse
    from .llm_client import LlmClient
    from .provider import ProviderError
except ImportError:
    from contracts import InferRequest, InferResponse
    from llm_client import LlmClient
    from provider import ProviderError

app = FastAPI()
router = APIRouter(prefix="/v1")


@router.post("/infer", response_model=InferResponse)
async def infer(
    body: InferRequest,
    x_anchor_secret: Annotated[str | None, Header()] = None,
) -> InferResponse:
    expected = os.environ.get("ANCHOR_AI_SECRET")
    if not expected:
        raise HTTPException(status_code=503, detail="service_not_configured")
    supplied = (x_anchor_secret or "").encode("utf-8")
    if not hmac.compare_digest(supplied, expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="service_auth_failed")
    try:
        return await LlmClient().complete_result(body.message)
    except ProviderError as error:
        raise HTTPException(status_code=error.status, detail=error.detail()) from None


app.include_router(router)
app.add_api_route("/infer", infer, methods=["POST"], response_model=InferResponse)
