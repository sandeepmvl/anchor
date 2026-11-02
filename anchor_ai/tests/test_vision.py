import asyncio
import base64
import copy
import json
import struct
import zlib

import httpx
import pytest
from pydantic import ValidationError

from anchor_ai.app import app
from anchor_ai.multimodal_context import sign_test_context
from anchor_ai.vision import (IssueSummary, OfflineVisionProvider, ROOT, PNG,
                              settings, prepare_image)

SECRET = 'local-vision-context-secret-32-bytes-minimum'


def chunk(kind, content):
    return (struct.pack('!I', len(content)) + kind + content
            + struct.pack('!I', zlib.crc32(kind + content) & 0xffffffff))


def screenshot(width=160, height=40):
    # A deterministic screenshot-style test image, not a recognition evaluation.
    rows = b''.join(b'\x00' + bytes([240, 240, 240]) * width for _ in range(height))
    header = struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return PNG + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')


def fixture_bytes():
    """A valid RGB transport fixture, not a recognition evaluation."""
    return screenshot(1, 1)


def ensure_fixture():
    """Generate the planned binary fixture; code blocks contain text only."""
    path = ROOT / 'anchor_ai/tests/fixtures/support_screenshot.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture_bytes())
    return path


@pytest.fixture
def setup_vision(monkeypatch):
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'local-service-secret')
    monkeypatch.setenv('ANCHOR_VISION_CONTEXT_SECRET', SECRET)
    monkeypatch.setenv('ANCHOR_MULTIMODAL_PROVIDER', 'offline')
    provider = OfflineVisionProvider()
    monkeypatch.setattr(app.state, 'vision_provider_factory', lambda: provider, raising=False)
    return provider


def headers(**changes):
    result = {'x-anchor-secret': 'local-service-secret',
              'x-anchor-context': sign_test_context(SECRET),
              'x-request-id': 'vision-demo',
              'traceparent': '00-' + 'a' * 32 + '-' + 'b' * 16 + '-01'}
    result.update(changes)
    return result


def body(raw=None):
    if raw is None:
        raw = ensure_fixture().read_bytes()
    return {'image_base64': base64.b64encode(raw).decode(),
            'ticket_text': 'Payment blocked; urgent.'}


def post(payload=None, supplied_headers=None):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://offline') as client:
            return await client.post('/api/v1/vision/analyze',
                                     json=body() if payload is None else payload,
                                     headers=headers() if supplied_headers is None else supplied_headers)
    return asyncio.run(run())


def test_demo(setup_vision):
    cfg = settings()
    assert cfg['provider'] == 'offline' and cfg['external_calls'] is False
    response = post()
    assert response.status_code == 200
    result = IssueSummary.model_validate(response.json())
    assert result.visible_error_code == 'E42'
    assert 0 <= result.confidence <= 1
    assert all(0 <= r.x0 < r.x1 <= 1 and 0 <= r.y0 < r.y1 <= 1 for r in result.evidence_regions)
    assert setup_vision.calls == 1
    print('Anchor screenshot analysis:', json.dumps(result.model_dump(), sort_keys=True))
    print('OFFLINE ONLY: visual recognition, hosted acceptance, and live token cost not verified')


@pytest.mark.parametrize('raw', [b'%PDF-1.7', b'unsupported', b'x' * 2000001, screenshot(769, 1)])
def test_reject_image_before_provider(setup_vision, raw):
    assert post(body(raw)).status_code == 422
    assert setup_vision.calls == 0


def test_invalid_base64_and_body_limit(setup_vision):
    assert post({'image_base64': '!', 'ticket_text': 'test'}).status_code == 422
    assert post({'image_base64': 'A' * 2900001, 'ticket_text': 'test'}).status_code == 422
    assert setup_vision.calls == 0


def test_corrupt_png(setup_vision):
    raw = bytearray(screenshot())
    raw[-1] ^= 1
    assert post(body(bytes(raw))).status_code == 422
    assert setup_vision.calls == 0


@pytest.mark.parametrize('scenario', ['malformed', 'invalid_json', 'refusal', 'truncated'])
def test_invalid_model_output(setup_vision, scenario):
    setup_vision.scenario = scenario
    response = post()
    assert response.status_code == 502
    assert response.json() == {'detail': 'invalid_vision_output'}
    assert setup_vision.calls == 1


def test_provider_rejection(setup_vision):
    setup_vision.scenario = 'rejected'
    response = post()
    assert response.status_code >= 400
    assert response.json()['detail'] == 'vision_provider_rejected'
    assert setup_vision.calls == 1


@pytest.mark.parametrize('token', ['', 'invalid', sign_test_context(SECRET, now=1),
                                    sign_test_context('wrong-secret')])
def test_signed_context_required(setup_vision, token):
    assert post(supplied_headers=headers(**{'x-anchor-context': token})).status_code == 403
    assert setup_vision.calls == 0


def test_tenant_not_accepted_from_body(setup_vision):
    payload = dict(body(), tenant_id=99)
    assert post(payload).status_code == 422
    assert setup_vision.calls == 0


def test_service_auth(setup_vision):
    assert post(supplied_headers=headers(**{'x-anchor-secret': 'wrong'})).status_code == 401
    assert setup_vision.calls == 0


def test_live_mode_fails_closed(setup_vision, monkeypatch):
    monkeypatch.setenv('ANCHOR_MULTIMODAL_PROVIDER', 'live')
    response = post()
    assert response.status_code == 503
    assert response.json() == {'detail': 'live_vision_not_verified'}
    assert setup_vision.calls == 0


def test_schema_boundary():
    records = json.loads((ROOT / 'anchor_ai/tests/fixtures/vision_responses.json').read_text())
    valid = json.loads(records['valid']['content'])
    bad_values = [dict(valid, urgency='banana'), dict(valid, refund_approved=True),
                  dict(valid, confidence=1.1), dict(valid, confidence=float('nan')),
                  dict(valid, evidence_regions=[{'x0': 0.8, 'y0': 0.0, 'x1': 0.2, 'y1': 1.0}]),
                  dict(valid, evidence_regions=[{'x0': 0.0, 'y0': 0.0, 'x1': 1.1, 'y1': 1.0}])]
    for candidate in bad_values:
        with pytest.raises(ValidationError):
            IssueSummary.model_validate(candidate)
    assert IssueSummary.model_validate(dict(valid, urgency='low')).urgency == 'low'


def test_metadata_only_trace(setup_vision, monkeypatch):
    from anchor_ai import observability
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    response = post()
    assert response.status_code == 200
    vision = next(r for r in records if r['name'] == 'python.vision')
    llm = next(r for r in records if r['name'] == 'llm.call')
    root = next(r for r in records if r['name'] == 'python.infer')
    for record in (vision, llm, root):
        assert record['tenant_id'] == 1
        assert record['conversation_id'] == 42
        assert record['request_id'] == 'vision-demo'
        assert record['trace_id'] == 'a' * 32
    assert vision['parent_span_id'] == root['span_id']
    assert llm['parent_span_id'] == vision['span_id']
    assert vision['vision_backend'] == 'offline'
    assert vision['vision_usage_basis'] == 'not_measured_offline'
    serialized = json.dumps(records)
    for secret in (body()['image_base64'], body()['ticket_text'],
                   'E42', 'data:image', 'image_base64', 'visible_error_code',
                   'evidence_regions', 'refund_approved'):
        assert secret not in serialized
    print('Metadata-only vision trace:', json.dumps(records, sort_keys=True))


def test_failure_trace_is_redacted(setup_vision, monkeypatch):
    from anchor_ai import observability
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    setup_vision.scenario = 'malformed'
    assert post().status_code == 502
    serialized = json.dumps(records)
    assert 'banana' not in serialized and 'refund_approved' not in serialized
    assert body()['image_base64'] not in serialized
    assert body()['ticket_text'] not in serialized
    assert any(r['error_type'] == 'invalid_vision_output' for r in records)


def test_generated_fixture_is_valid():
    raw = fixture_bytes()
    decoded, dimensions = prepare_image(base64.b64encode(raw).decode(), settings())
    assert decoded == raw
    assert dimensions == (1, 1)
    assert ensure_fixture().read_bytes() == raw
