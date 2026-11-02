import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from anchor_ai import app as service
from anchor_ai import llm_client as module
from anchor_ai.llm_client import LlmClient
from anchor_ai.provider import (LiveProvider, ProviderError, ReplayProvider,
                                retry_after, safe_retry_after)

PAYLOAD = {"version": "v1", "conversation_id": 1, "tenant_id": 1,
           "message": "Can I return my order?"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    async def denied(*args, **kwargs):
        raise AssertionError("Outbound provider access disabled")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", denied)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANCHOR_LLM_MODE", "offline")


class Clock:
    def __init__(self):
        self.now = 0.0
        self.delays = []

    async def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay


def request(monkeypatch, scenario):
    clock = Clock()
    provider = ReplayProvider(scenario)
    llm = LlmClient(provider, sleep=clock.sleep, clock=lambda: clock.now,
                    jitter=lambda: 0.025)
    monkeypatch.setattr(service, "LlmClient", lambda: llm)
    monkeypatch.setenv("ANCHOR_AI_SECRET", "test-service-secret")
    with TestClient(service.app) as client:
        response = client.post("/infer", json=PAYLOAD,
                               headers={"X-Anchor-Secret": "test-service-secret"})
    return response, clock, provider


def test_recorded_200(monkeypatch):
    response, clock, provider = request(monkeypatch, "200")
    assert response.status_code == 200
    assert response.json() == {
        "reply": "I can help check the return policy for your order.",
        "tokens": 38, "usage": {"prompt_tokens": 24, "completion_tokens": 14,
                               "total_tokens": 38},
        "finish_reason": "stop", "attempts": 1,
    }
    assert len(provider.requests) == 1 and clock.delays == []


@pytest.mark.parametrize("scenario,status,code", [
    ("400", 400, "provider_bad_request"),
    ("401", 401, "provider_auth_failed"),
    ("403", 403, "provider_forbidden"),
    ("404", 404, "provider_not_found"),
    ("429_quota", 429, "provider_quota_exhausted"),
])
def test_terminal_errors(monkeypatch, scenario, status, code):
    response, clock, provider = request(monkeypatch, scenario)
    assert response.status_code == status
    assert response.json()["detail"] == {
        "code": code, "retryable": False, "retry_after": None, "attempts": 1,
    }
    assert clock.delays == [] and len(provider.requests) == 1


def test_429_recovers(monkeypatch):
    response, clock, provider = request(monkeypatch, "429_recover")
    assert response.status_code == 200 and response.json()["attempts"] == 2
    assert clock.delays == [pytest.approx(0.225)]
    assert len(provider.requests) == 2


def test_5xx_exhausts(monkeypatch):
    response, clock, provider = request(monkeypatch, "5xx")
    assert response.status_code == 502
    assert response.json()["detail"] == {
        "code": "provider_unavailable", "retryable": True,
        "retry_after": None, "attempts": 3,
    }
    assert clock.delays == [pytest.approx(0.125), pytest.approx(0.225)]
    assert len(provider.requests) == 3


def test_rate_limit_deadline_and_demo(monkeypatch):
    success, _, _ = request(monkeypatch, "200")
    response, clock, provider = request(monkeypatch, "429_persistent")
    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail == {"code": "provider_rate_limited", "retryable": True,
                      "retry_after": 2, "attempts": 2}
    assert clock.delays == [pytest.approx(2.025)]
    assert len(provider.requests) == 2
    print("Same ticket: " + success.json()["reply"])
    print("Selected rate-limit fixture: " + json.dumps(detail, sort_keys=True))


@pytest.mark.parametrize("scenario", ["timeout", "connection"])
def test_transport_recovers(monkeypatch, scenario):
    response, clock, _ = request(monkeypatch, scenario)
    assert response.status_code == 200 and response.json()["attempts"] == 2
    assert len(clock.delays) == 1


def test_length_is_success(monkeypatch):
    response, clock, provider = request(monkeypatch, "length")
    assert response.status_code == 200
    assert response.json()["finish_reason"] == "length"
    assert len(provider.requests) == 1 and not clock.delays


def test_live_request_and_missing_key(monkeypatch):
    sent = []
    def send(request):
        sent.append(request)
        return ReplayProvider().send(request)
    llm = LlmClient(LiveProvider(httpx.MockTransport(send)))
    with pytest.raises(ProviderError) as failure:
        llm.complete("Hello")
    assert failure.value.code == "missing_provider_key" and sent == []
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")
    assert llm.complete("Hello", system="Be helpful", max_tokens=12)
    request = sent[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer dummy-test-key"
    body = json.loads(request.content)
    assert body["messages"] == [{"role": "system", "content": "Be helpful"},
                                {"role": "user", "content": "Hello"}]
    assert body["max_tokens"] == 12


def test_retry_after_safety():
    assert retry_after(httpx.Headers({"Retry-After": "nan"})) is None
    assert retry_after(httpx.Headers({"Retry-After": "-1"})) is None
    assert retry_after(httpx.Headers({"Retry-After": "bad"})) is None
    assert safe_retry_after(0.2) == 1
    assert safe_retry_after(61) is None


def test_outer_deadline(monkeypatch):
    async def stalled(self, *args):
        await asyncio.sleep(1)
    monkeypatch.setattr(module, "TOTAL_DEADLINE", 0.01)
    monkeypatch.setattr(LlmClient, "_run", stalled)
    with pytest.raises(ProviderError) as failure:
        LlmClient(ReplayProvider()).complete("Hello")
    assert failure.value.retryable
