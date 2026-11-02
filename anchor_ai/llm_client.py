import asyncio
import os
import random
import time

import httpx

try:
    from .contracts import InferResponse, Usage
    from .provider import (ProviderError, ReplayProvider, LiveProvider,
                           configured_provider, classify, safe_retry_after)
    from .observability import span, tokens_from_usage, estimate, PRICE_VERSION
except ImportError:
    from contracts import InferResponse, Usage
    from provider import (ProviderError, ReplayProvider, LiveProvider,
                          configured_provider, classify, safe_retry_after)
    from observability import span, tokens_from_usage, estimate, PRICE_VERSION

MAX_ATTEMPTS = 3
TOTAL_DEADLINE = 4.0
BACKOFF_CAP = 1.0


class LlmClient:
    def __init__(self, provider=None, *, sleep=asyncio.sleep,
                 clock=time.monotonic, jitter=None):
        if isinstance(provider, str):
            if provider == 'offline':
                provider = ReplayProvider()
            elif provider == 'live':
                provider = LiveProvider()
            else:
                raise ProviderError('invalid_provider_mode', status=503)
        self.provider = provider if provider is not None else configured_provider()
        self.sleep, self.clock = sleep, clock
        self.jitter = jitter or (lambda: random.uniform(0, 0.05))
        self.attempts = 0

    def complete(self, prompt, system=None, max_tokens=256):
        return asyncio.run(self.complete_result(prompt, system, max_tokens)).reply

    async def complete_result(self, prompt, system=None, max_tokens=256):
        self.attempts = 0
        self.last_error = ProviderError('provider_timeout', retryable=True)
        with span('llm.call', stage='provider',
                  request_model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
                  response_model=None, currency='USD', price_version=PRICE_VERSION,
                  cost_basis='offline_replay_estimate' if isinstance(self.provider, ReplayProvider)
                  else 'published_price_estimate') as telemetry:
            self.telemetry = telemetry
            try:
                return await asyncio.wait_for(
                    self._run(prompt, system, max_tokens), timeout=TOTAL_DEADLINE
                )
            except asyncio.TimeoutError:
                raise self._final_error() from None
            finally:
                telemetry.set(attempts=self.attempts)

    def _final_error(self):
        error = self.last_error
        error.attempts = self.attempts
        error.retry_after = safe_retry_after(error.retry_after)
        return error

    async def _run(self, prompt, system, max_tokens):
        key = self.provider.key()
        messages = []
        if system is not None:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        payload = {'model': os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
                   'messages': messages, 'max_tokens': max_tokens}
        end = self.clock() + TOTAL_DEADLINE
        async with self.provider.client() as client:
            while self.attempts < MAX_ATTEMPTS:
                remaining = end - self.clock()
                if remaining <= 0:
                    break
                self.attempts += 1
                try:
                    response = await client.post(
                        'https://api.openai.com/v1/chat/completions',
                        headers={'Authorization': 'Bearer ' + key}, json=payload,
                        timeout=httpx.Timeout(min(2.0, remaining),
                                              connect=min(1.0, remaining)),
                    )
                    self.telemetry.set(provider_request_id=response.headers.get('x-request-id'),
                                       status_code=response.status_code)
                    if response.status_code == 200:
                        try:
                            body = response.json()
                            choice = body['choices'][0]
                            usage = Usage.model_validate(body['usage'])
                            if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
                                raise ValueError('inconsistent usage')
                            result = InferResponse(
                                reply=choice['message']['content'],
                                tokens=usage.total_tokens, usage=usage,
                                finish_reason=choice['finish_reason'], attempts=self.attempts,
                            )
                            tokens = tokens_from_usage(usage)
                            model = body.get('model')
                            self.telemetry.set(response_model=model, tokens=tokens,
                                               cost_estimate=estimate(model, tokens))
                            return result
                        except (ValueError, KeyError, IndexError, TypeError):
                            raise ProviderError('invalid_provider_response', attempts=self.attempts) from None
                    error = classify(response)
                except (httpx.TimeoutException, httpx.ConnectError):
                    error = ProviderError('provider_transport_error', retryable=True)
                error.attempts = self.attempts
                if not error.retryable:
                    raise error
                self.last_error = error
                if self.attempts >= MAX_ATTEMPTS:
                    break
                backoff = min(BACKOFF_CAP, 0.1 * 2 ** (self.attempts - 1))
                delay = max(error.retry_after or 0, backoff) + self.jitter()
                if delay >= end - self.clock():
                    break
                await self.sleep(delay)
        raise self._final_error()
