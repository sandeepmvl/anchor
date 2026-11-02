import asyncio
import json
import math
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
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
        self.code = code
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = attempts
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
        if task == 'structured':
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
                self.expected_reasoning = {
                    'model': data['model'], 'reasoning_effort': mapped,
                    'max_completion_tokens': max_tokens}
                self.expected_messages = [{'role': 'user', 'content': ticket['ticket']}]
                self.fixture_latency_ms = run['fixture_latency_ms']
                body = {
                    'model': data['model'],
                    'choices': [{'message': {'content': '' if limited else json.dumps(run['labels'])},
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
                self.expected_messages = [
                    {'role': 'user', 'content': render(variant, ticket['ticket'])}
                ]
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
    # Never forward a shorter minimum. Unrepresentable hints are omitted.
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
            return ProviderError('provider_rate_limited', status=429,
                                 retryable=True, retry_after=hint)
        return ProviderError('provider_unknown_429', status=429)
    codes = {400: 'provider_bad_request', 401: 'provider_auth_failed',
             403: 'provider_forbidden', 404: 'provider_not_found'}
    return ProviderError(codes.get(status, 'provider_rejected'), status=status if 400 <= status < 500 else 502)
