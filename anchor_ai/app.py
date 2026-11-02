import hmac
import os
from typing import Annotated

from fastapi import APIRouter, FastAPI, Header, HTTPException

try:
    from .contracts import InferRequest, InferResponse
    from .llm_client import LlmClient
except ImportError:
    from contracts import InferRequest, InferResponse
    from llm_client import LlmClient

app = FastAPI()
router = APIRouter(prefix="/v1")


@router.post("/infer", response_model=InferResponse)
def infer(
    body: InferRequest,
    x_anchor_secret: Annotated[str | None, Header()] = None,
) -> InferResponse:
    expected = os.environ.get("ANCHOR_AI_SECRET")
    if not expected:
        raise HTTPException(status_code=503, detail="service_not_configured")
    supplied = (x_anchor_secret or "").encode("utf-8")
    if not hmac.compare_digest(supplied, expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="service_auth_failed")
    return InferResponse(reply=LlmClient().complete(body.message), tokens=0)


app.include_router(router)
app.add_api_route("/infer", infer, methods=["POST"], response_model=InferResponse)
