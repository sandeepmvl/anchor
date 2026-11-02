def incident_policy_violations(policy):
    errors = []
    order = policy.get('order')
    if (not isinstance(order, list) or not order
            or any(not isinstance(n, str) or not n for n in order)
            or len(order) != len(set(order))):
        errors.append('cyclic_routing')
    if policy.get('retry_owners') != ['resilience']:
        errors.append('conflicting_retry_ownership')
    if policy.get('reset') != 'shared_fixed_window' or policy.get('window_seconds') != 60:
        errors.append('quota_window_misconfiguration')
    for key in ('max_attempts', 'cap_units'):
        if type(policy.get(key)) is not int or policy[key] <= 0:
            errors.append('invalid_' + key)
    value = policy.get('total_seconds')
    if type(value) not in (int, float) or not 0 < value < float('inf'):
        errors.append('invalid_total_seconds')
    return errors


def validate_incident_policy(policy):
    errors = incident_policy_violations(policy)
    if errors:
        raise ValueError(','.join(errors))
    return policy


# Buffered routing is opt-in; existing provider selection below is unchanged.
def routing_configuration(path='config/provider_routing.yml'):
    import yaml
    c = yaml.safe_load(Path(path).read_text())
    if type(c.get('sdk_retries')) is not int or c['sdk_retries'] != 0:
        raise ValueError('nested_provider_retries')
    for key in ('threshold', 'max_attempts', 'bound'):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError('invalid_routing_configuration')
    for key in ('cooldown_seconds', 'attempt_timeout_seconds', 'total_timeout_seconds'):
        value = c.get(key)
        if type(value) not in (int, float) or not 0 < value < float('inf'):
            raise ValueError('invalid_routing_configuration')
    order = c['order']
    if not order or len(order) != len(set(order)):
        raise ValueError('invalid_routing_order')
    from anchor_ai.observability import safe_id
    for name in order:
        d = c['deployments'][name]
        if (safe_id(name) is None or not d.get('model') or not d.get('region')
                or type(d.get('context_bytes')) is not int or d['context_bytes'] <= 0
                or not isinstance(d.get('capabilities'), list)
                or type(d.get('evaluated')) is not bool):
            raise ValueError('invalid_routing_deployment')
    if 'incident_policy' in c:
        validate_incident_policy(c['incident_policy'])
    return c


AVAILABILITY = frozenset({'provider_timeout', 'provider_transport_error',
                          'provider_unavailable', 'provider_rate_limited'})


class OfflineDeployment:
    """Single-send deterministic adapter; never opens a network connection."""
    fixture = True
    max_retries = 0

    def __init__(self):
        self.scenario, self.requests, self.cancelled = 'ok', [], False

    async def send(self, request, payload):
        self.requests.append(payload)
        if self.scenario == 'timeout':
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
        if self.scenario == 'connection':
            raise httpx.ConnectError('offline transport failure', request=request)
        status = {'outage': 503, 'bad': 400, 'auth': 401, 'forbidden': 403,
                  'quota': 429, 'unknown429': 429, 'rate': 429}.get(self.scenario, 200)
        if status != 200:
            code = {'quota': 'insufficient_quota', 'rate': 'rate_limit_exceeded'}.get(
                self.scenario, 'fixture_error')
            return httpx.Response(status, json={'error': {'code': code}}, request=request)
        return httpx.Response(200, request=request, json={
            'model': payload['model'],
            'choices': [{'message': {'content': 'I can help check the return policy for R-42.'},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 5, 'total_tokens': 10}})


class RoutingProvider:
    """Buffered router; enable_resilience installs the single retry owner."""
    vision_offline = True

    def __init__(self, adapters, collector, *, config=None, clock=None, production=False):
        import time
        if production:
            raise ValueError('routing_release_evidence_missing')
        self.config = config or routing_configuration()
        if 'incident_policy' in self.config:
            validate_incident_policy(self.config['incident_policy'])
        if self.config.get('sdk_retries', 0) != 0:
            raise ValueError('nested_provider_retries')
        if any(getattr(a, 'max_retries', 0) != 0 for a in adapters.values()):
            raise ValueError('nested_provider_retries')
        adapters = dict(adapters)
        for name, deployment in self.config['deployments'].items():
            if deployment.get('backend') == 'self_hosted' and name not in adapters:
                from anchor_ai.vllm_backend import VllmBackend
                adapters[name] = VllmBackend(model=deployment['model'])
        self.adapters, self.collector, self.clock = adapters, collector, clock or time.monotonic
        self.health = {n: {'failures': 0, 'degraded': False, 'next_probe': 0}
                       for n in self.config['order']}
        self.probe_locks = {n: asyncio.Lock() for n in self.config['order']}
        self.served_by, self.events = None, []

    def enable_resilience(self, *, database, **options):
        from anchor_ai.resilience import attach
        return attach(self, database=database, **options)

    def key(self):
        return 'offline-routing-boundary'

    def event(self, name, reason, attempt_id=None):
        from anchor_ai.observability import emit_provider_routing
        h = self.health[name]
        record = dict(provider=name, classification=reason, failures=h['failures'],
                      degraded=h['degraded'], next_probe=h['next_probe'], attempt_id=attempt_id)
        self.events.append(record)
        emit_provider_routing(record)

    def eligible(self, name, tenant_id, payload):
        policy = self.config['tenants'].get(tenant_id, {})
        d = self.config['deployments'][name]
        return (name in policy.get('approved', []) and d['region'] in policy.get('regions', [])
                and d['evaluated'] is True and 'chat' in d['capabilities']
                and len(json.dumps(payload).encode()) <= d['context_bytes'])

    def failure(self, name):
        h = self.health[name]
        h['failures'] += 1
        if h['failures'] >= self.config['threshold']:
            h['degraded'] = True
            h['next_probe'] = self.clock() + self.config['cooldown_seconds']
        self.event(name, 'availability_failure')

    def success(self, name):
        self.health[name].update(failures=0, degraded=False, next_probe=0)

    async def physical(self, name, request, payload):
        from anchor_ai.quota_client import reserve_routing_attempt
        from anchor_ai.cost_accounting import routing_attempt, settle_quota_usage
        authority, reservation = reserve_routing_attempt()
        attempt = routing_attempt(self.collector, provider=name,
                                  model=self.config['deployments'][name]['model'],
                                  reservation=reservation)
        mapped = dict(payload, model=attempt['model'])
        self.event(name, 'dispatch', attempt['attempt_id'])
        settled = False
        try:
            async with asyncio.timeout(self.config['attempt_timeout_seconds']):
                response = await self.adapters[name].send(request, mapped)
            if response.status_code == 200:
                event = settle_quota_usage(authority, reservation, attempt,
                    response.json().get('usage'), self.collector.config)
                self.collector.events.append(event)
                settled = True
            return response
        except (TimeoutError, httpx.TimeoutException):
            raise ProviderError('provider_timeout') from None
        except httpx.TransportError:
            raise ProviderError('provider_transport_error') from None
        finally:
            if not settled:
                self.collector.unresolved.append(attempt['attempt_id'])

    async def route(self, request, tenant_id):
        payload = json.loads(request.content)
        if payload.get('stream'):
            raise ProviderError('routing_stream_unsupported', status=503)
        if payload.get('tools') or payload.get('tool_choice'):
            raise ProviderError('routing_tool_unsupported', status=503)
        visited = set()
        async with asyncio.timeout(self.config['total_timeout_seconds']):
            for name in self.config['order']:
                if len(visited) >= self.config['max_attempts']:
                    break
                if name in visited or not self.eligible(name, tenant_id, payload):
                    continue
                if self.health[name]['degraded']:
                    self.event(name, 'cooldown_skip')
                    continue
                visited.add(name)
                try:
                    response = await self.physical(name, request, payload)
                    if response.status_code != 200:
                        error = classify(response)
                        if error.code not in AVAILABILITY:
                            self.event(name, 'non_availability_error')
                            return response
                        raise error
                except ProviderError as error:
                    if error.code not in AVAILABILITY:
                        raise
                    self.failure(name)
                    continue
                self.success(name)
                self.served_by = name
                self.event(name, 'success')
                return response
        raise ProviderError('routing_unavailable', status=503)

    def client_for(self, tenant_id):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def opened():
            async def dispatch(request):
                try:
                    return await self.route(request, tenant_id)
                except TimeoutError:
                    raise ProviderError('routing_deadline', status=503) from None
            async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch), trust_env=False) as client:
                yield client
        return opened

    async def probe(self, name, tenant_id):
        async with self.probe_locks[name]:
            h = self.health[name]
            if not h['degraded'] or self.clock() < h['next_probe']:
                return False
            payload = {'model': self.config['logical_model'],
                       'messages': [{'role': 'user', 'content': 'Synthetic recovery probe'}],
                       'max_tokens': 16}
            if not self.eligible(name, tenant_id, payload):
                return False
            request = httpx.Request('POST', 'https://offline.invalid/probe', json=payload)
            try:
                response = await self.physical(name, request, payload)
                if response.status_code != 200:
                    raise classify(response)
            except ProviderError:
                h['next_probe'] = self.clock() + self.config['cooldown_seconds']
                self.event(name, 'probe_failed')
                return False
            self.success(name)
            self.event(name, 'probe_success')
            return True


NATIVE_CAPABILITIES = frozenset({'responses', 'function_tools', 'stored_history'})


class NativeReplayProvider:
    name = 'fake'
    capabilities = NATIVE_CAPABILITIES

    def __init__(self):
        import copy
        self.records = json.loads((Path(__file__).parent / 'tests/fixtures/native_agent_responses.json').read_text())
        self.responses = {}
        self.requests = []
        self.copy = copy.deepcopy

    async def create(self, payload):
        self.requests.append(self.copy(payload))
        if payload.get('store') is not True:
            raise ValueError('native_storage_required')
        previous = payload.get('previous_response_id')
        if previous and previous not in self.responses:
            raise ValueError('native_response_unavailable')
        history = self.copy(self.responses[previous]['history']) if previous else []
        items = payload['input']
        if not previous:
            if items != self.records['initial_input']:
                raise ValueError('native_replay_miss')
            output = self.copy(self.records['initial_output'])
        elif all(item.get('type') == 'function_call_output' for item in items):
            pending = [item['call_id'] for item in history if item.get('type') == 'function_call']
            if [item['call_id'] for item in items] != pending[-len(items):]:
                raise ValueError('native_tool_result_mismatch')
            output = self.copy(self.records['answer_output'])
        else:
            if items != self.records['followup_input']:
                raise ValueError('native_replay_miss')
            if not any(item.get('type') == 'function_call_output' for item in history):
                raise ValueError('native_missing_provider_history')
            output = self.copy(self.records['followup_output'])
        response_id = 'offline_response_' + str(len(self.responses) + 1)
        self.responses[response_id] = {'history': history + self.copy(items) + output}
        return {'id': response_id, 'output': self.copy(output)}


class NativeOpenAIProvider:
    name = 'openai'
    capabilities = NATIVE_CAPABILITIES

    async def create(self, payload):
        provider = LiveProvider()
        try:
            async with provider.client() as client:
                response = await client.post(
                    'https://api.openai.com/v1/responses',
                    headers={'Authorization': 'Bearer ' + provider.key()},
                    json=payload, timeout=httpx.Timeout(30, connect=2))
                if response.status_code != 200:
                    raise classify(response)
                return response.json()
        except httpx.TimeoutException:
            raise ProviderError('provider_timeout') from None
        except httpx.TransportError:
            raise ProviderError('provider_transport_error') from None


def configured_native_provider():
    mode = os.environ.get('PROVIDER', 'fake')
    if mode == 'fake':
        return NativeReplayProvider()
    if mode == 'openai':
        return NativeOpenAIProvider()
    raise ValueError('unsupported_native_provider')


import asyncio
import json
import math
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol
import httpx

try:
    from .prompts import render, render_structured
    from .schemas import TRIAGE_FORMAT
except ImportError:
    from prompts import render, render_structured
    from schemas import TRIAGE_FORMAT
FIXTURES = Path(__file__).parent / 'tests/fixtures/provider_responses.json'
TRIAGE_TICKETS = FIXTURES.with_name('triage_tickets.json')
TRIAGE_RESPONSES = FIXTURES.with_name('triage_responses.json')
REASONING_RESPONSES = FIXTURES.with_name('reasoning_responses.json')
STRUCTURED_RESPONSES = FIXTURES.with_name('structured_responses.json')


def structured_record(ticket):
    try:
        return next(item for item in json.loads(STRUCTURED_RESPONSES.read_text())
                    if item['ticket'] == ticket)
    except (StopIteration, ValueError, KeyError, TypeError):
        raise ProviderError('replay_miss') from None


class ProviderError(Exception):
    def __init__(self, code, status=502, retryable=False, retry_after=None, attempts=0):
        self.code, self.status, self.retryable = code, status, retryable
        self.retry_after, self.attempts = retry_after, attempts
        super().__init__(code)

    def detail(self):
        return {'code': self.code, 'retryable': self.retryable,
                'retry_after': self.retry_after, 'attempts': self.attempts}


class Provider(Protocol):
    def client(self) -> httpx.AsyncClient: ...
    def key(self) -> str: ...


class ReplayProvider:
    def __init__(self, scenario='200', *, task=None, variant=None, fixture_id=None,
                 effort=None, max_tokens=4000):
        self.expected_messages = None
        self.expected_reasoning = None
        self.expected_format = None
        self.fixture_latency_ms = None
        if task == 'synthetic':
            try:
                records = json.loads(FIXTURES.with_name('synthetic_responses.json').read_text())
                record = records[int(fixture_id)]
                if not isinstance(variant, str):
                    raise ValueError('missing_prompt')
                self.expected_messages = [{'role': 'user', 'content': variant}]
                self.sequence = [{'status': 200, 'body': {
                    'model': 'offline-synthetic',
                    'choices': [{'message': {'content': json.dumps(record)}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 40, 'total_tokens': 140}}}]
            except (IndexError, ValueError, TypeError, KeyError):
                raise ProviderError('replay_miss') from None
        elif task == 'structured':
            try:
                record = next(item for item in json.loads(STRUCTURED_RESPONSES.read_text())
                              if item['id'] == fixture_id)
                system, ticket = render_structured(record['ticket'])
                self.expected_messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': ticket}]
                self.expected_format = TRIAGE_FORMAT
                self.sequence = [{'status': 200, 'body': {
                    'model': 'gpt-4o-mini',
                    'choices': [{'message': {'content': record['raw']}, 'finish_reason': record['finish_reason']}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}}]
            except (KeyError, StopIteration, ValueError, TypeError):
                raise ProviderError('replay_miss') from None
        elif task == 'reasoning':
            try:
                data = json.loads(REASONING_RESPONSES.read_text())
                ticket = next(t for t in data['tickets'] if t['id'] == fixture_id)
                run = ticket['runs'][effort]
                limited = max_tokens < run['output']
                if limited:
                    if effort != 'high' or max_tokens != ticket['limited']['cap']:
                        raise KeyError('unrecorded_limit')
                    run = ticket['limited']
                mapped = {'off': 'none', 'low': 'low', 'high': 'high'}[effort]
                self.expected_reasoning = {'model': data['model'], 'reasoning_effort': mapped,
                                           'max_completion_tokens': max_tokens}
                self.expected_messages = [{'role': 'user', 'content': ticket['ticket']}]
                self.fixture_latency_ms = run['fixture_latency_ms']
                body = {'model': data['model'], 'choices': [{'message': {
                    'content': '' if limited else json.dumps(run['labels'])},
                    'finish_reason': 'length' if limited else 'stop'}],
                    'usage': {'prompt_tokens': run['input'], 'completion_tokens': run['output'],
                              'total_tokens': run['input'] + run['output'],
                              'completion_tokens_details': {'reasoning_tokens': run['reasoning']}}}
                self.sequence = [{'status': 200, 'body': body}]
            except (KeyError, StopIteration, ValueError, TypeError):
                raise ProviderError('replay_miss') from None
        elif task is not None:
            try:
                if task != 'triage':
                    raise KeyError(task)
                tickets = json.loads(TRIAGE_TICKETS.read_text())
                ticket = next(item for item in tickets if item['id'] == fixture_id)
                labels = json.loads(TRIAGE_RESPONSES.read_text())[task][variant][fixture_id]
                self.expected_messages = [{'role': 'user', 'content': render(variant, ticket['ticket'])}]
                self.sequence = json.loads(FIXTURES.read_text())['triage']
                self.sequence[0]['body']['choices'][0]['message']['content'] = json.dumps(labels)
            except (KeyError, StopIteration, ValueError, TypeError):
                raise ProviderError('replay_miss') from None
        else:
            self.sequence = json.loads(FIXTURES.read_text())[scenario]
        self.requests = []

    def key(self):
        return 'offline-key'

    def send(self, request):
        if self.expected_messages is not None:
            try:
                payload = json.loads(request.content)
                matches = payload['messages'] == self.expected_messages
                if self.expected_format is not None:
                    matches = matches and payload.get('response_format') == self.expected_format
                if self.expected_reasoning is not None:
                    matches = matches and all(payload.get(k) == v for k, v in self.expected_reasoning.items())
                    matches = matches and 'max_tokens' not in payload
            except (ValueError, KeyError, TypeError):
                matches = False
            if not matches:
                raise ProviderError('replay_miss')
        index = min(len(self.requests), len(self.sequence) - 1)
        self.requests.append(request)
        item = self.sequence[index]
        if item.get('exception') == 'timeout':
            raise httpx.ReadTimeout('recorded timeout', request=request)
        if item.get('exception') == 'connection':
            raise httpx.ConnectError('recorded connection failure', request=request)
        options = {'json': item['body']} if 'body' in item else {'text': item['text']}
        return httpx.Response(item['status'], headers=item.get('headers', {}), **options)

    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.send), trust_env=False)


class LiveProvider:
    def __init__(self, transport=None):
        self.transport = transport

    def key(self):
        key = os.environ.get('OPENAI_API_KEY', '').strip()
        if not key:
            raise ProviderError('missing_provider_key', status=503)
        return key

    def client(self):
        transport = self.transport if self.transport is not None else httpx.AsyncHTTPTransport(retries=0)
        return httpx.AsyncClient(transport=transport, trust_env=False)


def configured_provider():
    backend = os.environ.get('LLM_BACKEND', 'hosted')
    if backend == 'self_hosted':
        from anchor_ai.vllm_backend import VllmBackend
        return VllmBackend()
    if backend != 'hosted':
        raise ProviderError('invalid_provider_mode', status=503)
    mode = os.environ.get('ANCHOR_LLM_MODE', 'offline')
    if mode == 'offline':
        state = os.environ.get('ANCHOR_MODEL_RELEASE_STATE')
        if state:
            try:
                from anchor_ai.customization import model_release_selection
                return ModelReleaseReplayProvider(model_release_selection(state))
            except (ValueError, KeyError, OSError):
                raise ProviderError('invalid_model_release', status=503) from None
        return ReplayProvider(os.environ.get('ANCHOR_LLM_FIXTURE', '200'))
    if mode == 'live':
        return LiveProvider()
    raise ProviderError('invalid_provider_mode', status=503)


def retry_after(headers):
    raw = headers.get('retry-after')
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        try:
            stamp = parsedate_to_datetime(raw)
            value = (stamp - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def safe_retry_after(value):
    if value is None or not math.isfinite(value) or not 0 <= value <= 60:
        return None
    return math.ceil(value)


def classify(response):
    status = response.status_code
    hint = retry_after(response.headers)
    if 500 <= status <= 599:
        return ProviderError('provider_unavailable', retryable=True, retry_after=hint)
    try:
        code = response.json().get('error', {}).get('code')
    except (ValueError, AttributeError):
        code = None
    if status == 429:
        if code == 'insufficient_quota':
            return ProviderError('provider_quota_exhausted', status=429)
        if code in {'rate_limit_exceeded', 'slow_down'}:
            return ProviderError('provider_rate_limited', status=429, retryable=True, retry_after=hint)
        return ProviderError('provider_unknown_429', status=429)
    codes = {400: 'provider_bad_request', 401: 'provider_auth_failed',
             403: 'provider_forbidden', 404: 'provider_not_found'}
    return ProviderError(codes.get(status, 'provider_rejected'), status=status if 400 <= status < 500 else 502)


@dataclass
class Delta:
    text: str = ''
    stop: str | None = None


async def stream_tokens(provider, message):
    if isinstance(provider, ModelReleaseReplayProvider):
        ticket = str(message)
        if ticket != provider.selection.ticket:
            raise ProviderError('replay_miss')
        text = provider.selection.complete(ticket)
        for start in range(0, len(text), 16):
            yield Delta(text=text[start:start + 16])
        yield Delta(stop='stop')
        return
    if isinstance(provider, ContextReplayProvider):
        text = provider.context_reply({'messages': message.context_messages})
        for start in range(0, len(text), 16):
            yield Delta(text=text[start:start + 16])
        yield Delta(stop='stop')
        return
    if isinstance(provider, ReplayProvider):
        for text in ['Refunds ', 'take ', 'five ', 'days.']:
            await asyncio.sleep(0.12)
            yield Delta(text=text)
        yield Delta(stop='stop')
        return
    payload = {
        'model': getattr(message, 'context_model', os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')),
        'messages': getattr(message, 'context_messages', [{'role': 'user', 'content': message}]),
        'max_tokens': getattr(message, 'context_max_tokens', 256), 'stream': True,
    }
    if hasattr(message, 'context_temperature'):
        payload['temperature'] = message.context_temperature
    # Never retry a stream: that could repeat already disclosed text.
    try:
        async with provider.client() as client:
            async with client.stream(
                'POST', 'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': 'Bearer ' + provider.key()},
                json=payload, timeout=httpx.Timeout(30, connect=2),
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise classify(response)
                data = []
                async for line in response.aiter_lines():
                    if line.startswith('data:'):
                        data.append(line[5:].removeprefix(' '))
                    elif line == '' and data:
                        raw, data = '\n'.join(data), []
                        if raw == '[DONE]':
                            return
                        record = json.loads(raw)
                        if 'error' in record:
                            raise ProviderError('provider_transport_error')
                        for choice in record.get('choices', []):
                            if choice.get('index', 0) != 0:
                                continue
                            text = choice.get('delta', {}).get('content')
                            if text:
                                if not isinstance(text, str):
                                    raise ProviderError('invalid_provider_response')
                                yield Delta(text=text)
                            if choice.get('finish_reason'):
                                yield Delta(stop=choice['finish_reason'])
    except httpx.TimeoutException:
        raise ProviderError('provider_timeout') from None
    except httpx.TransportError:
        raise ProviderError('provider_transport_error') from None


class ContextReplayProvider:
    def __init__(self):
        self.records = json.loads(FIXTURES.with_name('context_responses.json').read_text())
        self.requests = []

    def key(self):
        return 'offline-context-key'

    def context_reply(self, payload):
        source = '\n'.join(m['content'] for m in payload['messages'][1:])
        known = ('receipt Friday' in source or 'receipt Friday.' in source)
        summary = not payload['messages'][0]['content'].startswith('<system>')
        if summary:
            key = 'summary_with_commitment' if known else 'summary_without_commitment'
        else:
            key = 'reply_with_commitment' if known else 'reply_without_commitment'
        return self.records[key]

    def send(self, request):
        try:
            from .context import request_count, count
        except ImportError:
            from context import request_count, count
        payload = json.loads(request.content)
        self.requests.append(payload)
        reply = self.context_reply(payload)
        input_tokens, output_tokens = request_count(payload['messages']), count(reply)
        return httpx.Response(200, json={
            'model': payload['model'],
            'choices': [{'message': {'content': reply}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens,
                      'total_tokens': input_tokens + output_tokens}})

    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.send), trust_env=False)


class ModelReleaseReplayProvider(ReplayProvider):
    def __init__(self, selection):
        self.selection = selection
        self.expected_messages = [{'role': 'user', 'content': selection.ticket}]
        self.expected_reasoning = None
        self.expected_format = None
        self.requests = []
        self.sequence = [{'status': 200, 'body': {
            'model': selection.candidate,
            'choices': [{'message': {'content': selection.reply}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 15, 'total_tokens': 20}}}]


class UsageTap:
    """Observe raw usage without changing the existing client's response."""
    def __init__(self, provider):
        self.provider = provider
        self.raw_usage = None
        self.vision_offline = getattr(provider, 'prefix_offline', False)

    def key(self):
        return self.provider.key()

    def client(self):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def opened():
            async def observe(response):
                if response.status_code == 200:
                    await response.aread()
                    self.raw_usage = response.json().get('usage')
            async with self.provider.client() as client:
                client.event_hooks['response'].append(observe)
                yield client
        return opened()


class SyntheticPrefixProvider:
    """HTTP fake; token units and first-text events are synthetic."""
    prefix_offline = True

    def __init__(self, *, ttl_seconds, clock, first_text=None):
        self.ttl_seconds, self.clock = ttl_seconds, clock
        self.first_text = first_text or (lambda: None)
        self.entries, self.requests = {}, []

    def key(self):
        return 'offline-prefix-key'

    def send(self, request):
        from anchor_ai.prompts import digest, canonical
        payload = json.loads(request.content)
        self.requests.append(payload)
        prefix = payload['messages'][0]['content']
        key = digest(canonical({'model': payload['model'], 'prefix': prefix,
                                'temperature': payload.get('temperature')}))
        hit = self.entries.get(key, -1) > self.clock()
        units = max(1, len(prefix.encode('utf-8')) // 4)
        tail = max(1, len(payload['messages'][-1]['content'].encode('utf-8')) // 4)
        self.clock.advance(.001 if hit else .006)
        self.first_text()
        self.entries[key] = self.clock() + self.ttl_seconds
        total = units + tail
        usage = dict(prompt_tokens=total, completion_tokens=12, total_tokens=total + 12,
                     prompt_tokens_details=dict(cached_tokens=units if hit else 0,
                                               cache_write_tokens=0 if hit else units))
        return httpx.Response(200, json={
            'model': payload['model'], 'usage': usage,
            'choices': [{'message': {'content': '{"category":"billing","urgency":"P3"}'},
                         'finish_reason': 'stop'}]})

    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.send), trust_env=False)


def serving_benchmark_stream(message, *, mode='offline'):
    """Explicit benchmark selection; never consult production routing defaults."""
    if mode == 'offline':
        provider = ReplayProvider()
    elif mode == 'live':
        provider = LiveProvider()
    else:
        raise ValueError('invalid_benchmark_mode')
    return stream_tokens(provider, message)


def offline_batch_provider():
    """Explicit offline bulk adapter; never selects a live provider implicitly."""
    from anchor_ai.batching import OfflineBatchProvider
    return OfflineBatchProvider()


def accounted_provider(provider, collector, *, style='chat'):
    """Explicit buffered-JSON accounting seam; streaming is not supported here."""
    from anchor_ai.cost_accounting import AccountingProvider
    return AccountingProvider(provider, collector, style=style)


def admitted_provider(provider):
    """Compose at the HTTP boundary; a verified quota scope is mandatory."""
    from anchor_ai.quota_client import AdmittedProvider
    return AdmittedProvider(provider)


def single_send_openai_client(*, api_key, **options):
    """SDK construction only; never enable retries underneath the router."""
    from openai import AsyncOpenAI
    if options.pop('max_retries', 0) != 0:
        raise ValueError('nested_provider_retries')
    return AsyncOpenAI(api_key=api_key, max_retries=0, **options)
