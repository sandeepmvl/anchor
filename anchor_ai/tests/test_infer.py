import secrets

import pytest
from fastapi.testclient import TestClient

from anchor_ai.app import app
from anchor_ai.llm_client import LlmClient

PAYLOAD = {
    "version": "v1", "conversation_id": 1,
    "tenant_id": 1, "message": "Can I return my order?",
}


@pytest.fixture
def boundary(monkeypatch):
    secret = secrets.token_hex(24)
    monkeypatch.setenv("ANCHOR_AI_SECRET", secret)
    monkeypatch.setenv("ANCHOR_AI_PROVIDER", "offline")
    with TestClient(app) as client:
        yield client, {"X-Anchor-Secret": secret}


@pytest.mark.parametrize("path", ["/infer", "/v1/infer"])
def test_reply(boundary, path):
    client, headers = boundary
    response = client.post(path, json=PAYLOAD, headers=headers)
    assert response.status_code == 200
    assert response.json() == {
        "reply": LlmClient().complete(PAYLOAD["message"]), "tokens": 0,
    }


@pytest.mark.parametrize("headers", [{}, {"X-Anchor-Secret": "wrong"}])
def test_auth_rejected(boundary, headers):
    client, _ = boundary
    response = client.post("/infer", json=PAYLOAD, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "service_auth_failed"}


@pytest.mark.parametrize("change", [
    {"tenant_id": "1"}, {"conversation_id": True}, {"message": " "},
    {"message": "x" * 4001}, {"version": "v2"}, {"extra": 1},
])
def test_invalid_contract(boundary, change):
    client, headers = boundary
    response = client.post("/infer", json={**PAYLOAD, **change}, headers=headers)
    assert response.status_code == 422


def test_missing_field_and_malformed_json(boundary):
    client, headers = boundary
    assert client.post("/infer", json={}, headers=headers).status_code == 422
    response = client.post(
        "/infer", content="not json",
        headers={**headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_server_requires_configuration(boundary, monkeypatch):
    client, headers = boundary
    monkeypatch.delenv("ANCHOR_AI_SECRET")
    response = client.post("/infer", json=PAYLOAD, headers=headers)
    assert response.status_code == 503
