import asyncio
import copy
import importlib.util
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'anchor_manifest', ROOT / 'ops/deploy/validate_manifest.py')
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def test_rendered_production_manifest():
    config = manifest.render()
    assert manifest.validate(config) == {'name': 'fake', 'provider': 'offline'}
    assert set(config['services']) == {'rails', 'ai-runtime', 'database', 'worker'}
    for name in ('rails', 'ai-runtime', 'worker'):
        assert config['services'][name]['environment']['ANCHOR_AI_SECRET'].startswith('${')


def test_missing_routing_configuration_is_rejected():
    config = manifest.render()
    del config['services']['ai-runtime']['environment']['ANCHOR_DEPLOYMENT_CONFIG']
    with pytest.raises(ValueError, match='missing routing configuration'):
        manifest.validate(config)
    print('REJECTED: missing routing configuration')


@pytest.mark.parametrize('failure', ['ports', 'secret', 'readiness', 'private', 'worker'])
def test_unsafe_manifest_is_rejected(failure):
    config = copy.deepcopy(manifest.render())
    services = config['services']
    if failure == 'ports':
        services['ai-runtime']['ports'] = ['8000:8000']
    elif failure == 'secret':
        services['rails']['environment']['ANCHOR_AI_SECRET'] = 'embedded-key'
    elif failure == 'readiness':
        del services['database']['healthcheck']
    elif failure == 'private':
        config['networks']['private']['internal'] = False
    else:
        services['worker']['command'] = ['./bin/rails', 'server']
    with pytest.raises(ValueError):
        manifest.validate(config)


@pytest.mark.parametrize('route', ['hosted', 'self_hosted', 'unknown'])
def test_toggle_cannot_authorize_unreleased_provider(route):
    config = manifest.render({'ANCHOR_DEPLOY_ROUTE': route})
    with pytest.raises(ValueError):
        manifest.validate(config)


def test_release_refuses_missing_real_infrastructure():
    with pytest.raises(ValueError, match='container and live-provider evidence missing'):
        manifest.release_gate()


def test_real_inference_endpoint_uses_selected_fake(monkeypatch):
    from fastapi import FastAPI
    from anchor_ai.app import app, healthz, infer, telemetry
    from anchor_ai.contracts import InferResponse

    # Exercise the production handler over ASGI without inventing a signed
    # tenant context. TenantScopeMiddleware has its own authorization tests;
    # this focused test establishes routing and service authentication only.
    inference_app = FastAPI()
    inference_app.middleware('http')(telemetry)
    inference_app.add_api_route('/v1/infer', infer, methods=['POST'],
        response_model=InferResponse, response_model_exclude_none=True)
    inference_client = manifest.inference_client
    selected_route = manifest.selected_route
    monkeypatch.setenv('ANCHOR_DEPLOYMENT_CONFIG', 'config/provider_routing.yml')
    monkeypatch.setenv('ANCHOR_DEPLOY_ROUTE', 'fake')
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'ephemeral-test-only-secret')
    # Even an ambient provider selection cannot override the deployment fake.
    monkeypatch.setenv('ANCHOR_LLM_PROVIDER', 'openai')
    route = selected_route(require_secret=True)
    assert route == {'name': 'fake', 'provider': 'offline'}
    marker = object()
    monkeypatch.setattr(app.state, 'rag_release_runtime', marker, raising=False)
    assert healthz()['deployment_route'] == 'fake'
    message = 'Can I refund order A17?'

    async def exercise():
        expected = await inference_client().complete_result(message)
        transport = httpx.ASGITransport(app=inference_app)
        async with httpx.AsyncClient(transport=transport, base_url='http://anchor') as client:
            response = await client.post('/v1/infer',
                headers={'X-Anchor-Secret': 'ephemeral-test-only-secret'},
                json={'version': 'v1', 'tenant_id': 1, 'conversation_id': 1, 'message': message})
            assert response.status_code == 200, response.text
            # Normalize the existing client result through its response contract.
            from anchor_ai.contracts import InferResponse
            if hasattr(expected, 'model_dump'):
                expected = expected.model_dump(exclude_none=True)
            expected = InferResponse.model_validate(expected).model_dump(exclude_none=True)
            assert response.json() == expected
            assert response.json()['reply']
            denied = await client.post('/v1/infer',
                headers={'X-Anchor-Secret': 'incorrect'},
                json={'version': 'v1', 'tenant_id': 1, 'conversation_id': 1, 'message': message})
            assert denied.status_code == 401
        print('AI readiness: ready; selected route=fake; provider=offline; authenticated fixture answered')
        print('Handler-level rehearsal only: tenant middleware, deployed task completion, and container isolation not verified')

    asyncio.run(exercise())
    monkeypatch.delenv('ANCHOR_AI_SECRET')
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as rejected:
        healthz()
    assert rejected.value.status_code == 503
