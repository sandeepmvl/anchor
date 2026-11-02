import asyncio
import json
import math
import os
import time
from urllib.parse import urlsplit

import httpx


MODEL = 'tiny-llama'
FIXTURE_REPLY = 'Refunds take five days. [kb:refund-policy]'


class OfflineVllmServer:
    """Protocol fixtures, not inference, retrieval or a quality evaluation."""
    def __init__(self):
        self.calls = 0
        self.last_payload = None
        self.block = False
        self.cancelled = False

    async def __call__(self, request):
        self.calls += 1
        if self.block:
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
        if request.method != 'POST' or request.url.path != '/v1/chat/completions':
            return httpx.Response(404, request=request)
        payload = json.loads(request.content)
        self.last_payload = payload
        if payload.get('model') != MODEL or not payload.get('messages'):
            return httpx.Response(400, request=request,
                                  json={'error': {'code': 'bad_request'}})
        usage = {'prompt_tokens': 24, 'completion_tokens': 12, 'total_tokens': 36}
        message = {'role': 'assistant', 'content': FIXTURE_REPLY}
        reason = 'stop'
        if payload.get('tools'):
            name = payload['tools'][0]['function']['name']
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'offline_call_1', 'type': 'function', 'function': {
                    'name': name, 'arguments': '{"order_id":"R-42"}'}}]}
            reason = 'tool_calls'
        elif payload.get('response_format'):
            message['content'] = '{"category":"billing","urgency":"P3"}'
        body = {'id': 'offline-vllm', 'object': 'chat.completion',
                'model': MODEL, 'choices': [{'index': 0, 'message': message,
                                            'finish_reason': reason}], 'usage': usage}
        if not payload.get('stream'):
            return httpx.Response(200, request=request, json=body,
                                  headers={'x-anchor-transport': 'offline'})
        delta = {k: v for k, v in message.items() if k != 'role'}
        events = [dict(model=MODEL, choices=[dict(index=0, delta=delta,
                                                  finish_reason=None)]),
                  dict(model=MODEL, choices=[dict(index=0, delta={},
                                                  finish_reason=reason)]),
                  dict(model=MODEL, choices=[], usage=usage)]
        text = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
        return httpx.Response(200, request=request,
            headers={'content-type': 'text/event-stream',
                     'x-anchor-transport': 'offline'}, text=text + 'data: [DONE]\n\n')


def endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip('/') != '/v1'):
        raise ValueError('invalid_self_hosted_url')
    if parsed.scheme == 'http' and parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('self_hosted_tls_required')
    return value.rstrip('/')


class VllmBackend:
    """Single-send HTTP adapter; retry ownership stays with Anchor's caller."""
    max_retries = 0
    backend_name = 'self_hosted'

    def __init__(self, *, base_url=None, mode=None, model=MODEL, timeout=2.0,
                 transport=None, diagnostic=None, production=False):
        self.mode = mode or os.getenv('ANCHOR_VLLM_TRANSPORT', 'offline')
        if self.mode not in {'offline', 'live'}:
            raise ValueError('invalid_vllm_transport')
        if production:
            raise ValueError('self_hosted_release_evidence_missing')
        if not math.isfinite(timeout) or not 0 < timeout <= 2:
            raise ValueError('invalid_vllm_timeout')
        self.base_url = endpoint(base_url or os.getenv(
            'LLM_BASE_URL', 'http://127.0.0.1:8765/v1'))
        if not isinstance(model, str) or not model.strip():
            raise ValueError('invalid_vllm_model')
        self.model, self.timeout = model, timeout
        self.vision_offline = self.mode == 'offline'
        self.server = OfflineVllmServer() if self.mode == 'offline' else None
        if self.mode == 'offline' and transport is not None:
            raise ValueError('offline_transport_cannot_be_replaced')
        self.transport = transport
        self.diagnostic = diagnostic or (lambda record: None)
        self.events = []

    def key(self):
        if self.mode == 'offline':
            return 'offline-vllm-key'
        key = os.getenv('LLM_API_KEY', '').strip()
        if not key:
            from anchor_ai.provider import ProviderError
            raise ProviderError('missing_provider_key', status=503)
        return key

    def record(self, **fields):
        record = dict(event='anchor.vllm', backend='self_hosted',
                      transport=self.mode, **fields)
        self.events.append(record)
        self.diagnostic(record)

    async def send(self, request, payload):
        """Also implements RoutingProvider's physical-adapter contract."""
        mapped = dict(payload, model=self.model)
        headers = {'content-type': 'application/json',
                   'authorization': 'Bearer ' + self.key()}
        for name in ('traceparent', 'x-request-id'):
            if name in request.headers:
                headers[name] = request.headers[name]
        target = httpx.Request('POST', self.base_url + '/chat/completions',
                               headers=headers, json=mapped)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout):
                if self.server is not None:
                    response = await self.server(target)
                else:
                    transport = self.transport or httpx.AsyncHTTPTransport(retries=0)
                    async with httpx.AsyncClient(transport=transport, trust_env=False,
                            timeout=httpx.Timeout(self.timeout, connect=min(1, self.timeout))) as client:
                        response = await client.send(target)
                        await response.aread()
            usage = None
            if response.status_code == 200 and not mapped.get('stream'):
                usage = response.json().get('usage')
            # Do not log request bodies, output text, URLs, keys or tool arguments.
            self.record(status_code=response.status_code,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        usage=usage)
            return response
        except TimeoutError:
            self.record(status_code=504, error='provider_timeout')
            raise httpx.ReadTimeout('vllm_deadline', request=request) from None

    def client(self):
        async def dispatch(request):
            payload = json.loads(request.content)
            return await self.send(request, payload)
        return httpx.AsyncClient(transport=httpx.MockTransport(dispatch), trust_env=False)


def validate_deployment(path='ops/deploy/vllm_service.yml'):
    from pathlib import Path
    import yaml
    spec = yaml.safe_load(Path(path).read_text())
    command = spec['command']
    # Validate the versioned batching contract while retaining exact validation
    # of the artifact, destination, runtime and all other launch arguments.
    from anchor_ai.serving_benchmark import surge_profile
    try:
        _, profile, _, _ = surge_profile(spec.get('batching_profile'))
        if spec.get('batching_profile') not in {'default', 'tuned'}:
            raise ValueError('invalid_batching_profile')
    except (KeyError, TypeError, ValueError):
        raise ValueError('invalid_vllm_deployment') from None
    expected = ['vllm', 'serve', '/srv/anchor/models/approved',
                '--quantization', 'awq', '--served-model-name', MODEL,
                '--host', '127.0.0.1', '--port', '8765',
                '--max-model-len', '512',
                '--max-num-seqs', str(profile['max_num_seqs']),
                '--max-num-batched-tokens', str(profile['max_num_batched_tokens']),
                '--enable-chunked-prefill', '--scheduling-policy', 'fcfs']
    if command != expected or spec['runtime'] != 'vllm==0.8.5':
        raise ValueError('invalid_vllm_deployment')
    if (spec['artifact']['format'] != 'awq'
            or spec['artifact']['architecture'] != 'LlamaForCausalLM'
            or spec['artifact']['approval_required'] is not True
            or spec['readiness']['path'] != '/v1/models'
            or spec['release']['real_evidence_required'] is not True):
        raise ValueError('invalid_vllm_deployment')
    return spec


def release_gate(evidence=None):
    # Offline protocol checks cannot approve a hardware deployment.
    raise ValueError('self_hosted_release_evidence_missing')
