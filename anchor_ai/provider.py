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


# Existing provider interfaces below remain available to earlier lessons.
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
                    'choices': [{'message': {'content': json.dumps(record)},
                                 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 40,
                              'total_tokens': 140}}}]
            except (IndexError, ValueError, TypeError, KeyError):
                raise ProviderError('replay_miss') from None
        elif task == 'structured':
            try:
                record = next(item for item in json.loads(STRUCTURED_RESPONSES.read_text())
                              if item['id'] == fixture_id)
                system, ticket = render_structured(record['ticket'])
                self.expected_messages = [{'role': 'system', 'content': system},
                                          {'role': 'user', 'content': ticket}]
                self.expected_format = TRIAGE_FORMAT
                self.sequence = [{'status': 200, 'body': {
                    'model': 'gpt-4o-mini',
                    'choices': [{'message': {'content': record['raw']},
                                 'finish_reason': record['finish_reason']}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20,
                              'total_tokens': 120}}}]
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
        return httpx.AsyncClient(transport=self.transport, trust_env=False)


def configured_provider():
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
        self.fixture_latency_ms = None
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
        # Explicit synthetic extension; not a universal OpenAI Chat field.
        usage = dict(prompt_tokens=total, completion_tokens=12, total_tokens=total + 12,
                     prompt_tokens_details=dict(cached_tokens=units if hit else 0,
                                               cache_write_tokens=0 if hit else units))
        return httpx.Response(200, json={
            'model': payload['model'], 'usage': usage,
            'choices': [{'message': {'content': '{"category":"billing","urgency":"P3"}'},
                         'finish_reason': 'stop'}]})

    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.send), trust_env=False)
