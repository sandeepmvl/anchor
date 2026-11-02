from contextlib import asynccontextmanager


class RoutedLlmClient:
    """Trusted composition; customers cannot supply verify or client_for."""
    def __init__(self, router, *, verify, token, client_for, feature='chat'):
        self.router, self.verify, self.token = router, verify, token
        self.authority_for, self.feature = client_for, feature

    @asynccontextmanager
    async def scoped_client(self):
        from anchor_ai.runtime import verified_routing_scope
        with verified_routing_scope(self.router, verify=self.verify, token=self.token,
                client_for=self.authority_for, feature=self.feature) as identity:
            router = self.router
            class BoundProvider:
                vision_offline = True
                key = router.key
                client = staticmethod(router.client_for(identity.tenant_id))
            yield identity, LlmClient(BoundProvider())

    async def complete_result(self, prompt, system=None, max_tokens=256):
        async with self.scoped_client() as (_, client):
            return await client.complete_result(prompt, system, max_tokens)

    def complete(self, prompt, system=None, max_tokens=256):
        return asyncio.run(self.complete_result(prompt, system, max_tokens)).reply

    async def stream_tokens(self, prompt, **kwargs):
        raise ProviderError('routing_stream_unsupported', status=503)
        yield

    async def probe(self, name):
        async with self.scoped_client() as (identity, _):
            return await self.router.probe(name, identity.tenant_id)


def transcribe_audio(raw, *, client=None, production=False, policy='silence'):
    """Speech uses an explicit adapter, never the chat provider's live setting."""
    from anchor_ai.audio import transcribe
    return transcribe(raw, client=client, production=production, policy=policy)


def extract_document(raw, *, engine='tesseract', limits=None, vision=None):
    """OCR is separate from chat generation and never inherits its live provider."""
    from anchor_ai.ocr import extract_pdf
    return extract_pdf(raw, engine=engine, limits=limits, vision=vision)


import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager

import httpx

try:
    from .contracts import InferResponse, Usage, TriageLabels, PromptContract
    from .prompts import render, render_structured, render_chat, contract_fingerprint
    from .provider import (ProviderError, ReplayProvider, LiveProvider,
                           configured_provider, classify, safe_retry_after, structured_record)
    from .observability import span, tokens_from_usage, estimate, PRICE_VERSION
    from .reasoning import request_fields
    from .schemas import TRIAGE_FORMAT, InvalidModelOutput, validate_triage
except ImportError:
    from contracts import InferResponse, Usage, TriageLabels, PromptContract
    from prompts import render, render_structured, render_chat, contract_fingerprint
    from provider import (ProviderError, ReplayProvider, LiveProvider,
                          configured_provider, classify, safe_retry_after, structured_record)
    from observability import span, tokens_from_usage, estimate, PRICE_VERSION
    from reasoning import request_fields
    from schemas import TRIAGE_FORMAT, InvalidModelOutput, validate_triage

try:
    from .context import assemble, PreparedPrompt, context_options, request_count
    from .provider import ContextReplayProvider, stream_tokens as provider_stream
    from .prompts import SUMMARY_VERSION
except ImportError:
    from context import assemble, PreparedPrompt, context_options, request_count
    from provider import ContextReplayProvider, stream_tokens as provider_stream
    from prompts import SUMMARY_VERSION

try:
    from .embeddings import embed as embed_texts
except ImportError:
    from embeddings import embed as embed_texts

MAX_ATTEMPTS = 3
TOTAL_DEADLINE = 4.0
BACKOFF_CAP = 1.0


def selected_contract(prompt, contract=None):
    value = contract if contract is not None else getattr(prompt, 'prompt_contract', None)
    return PromptContract.model_validate(value) if value is not None else None


def contract_fields(contract):
    return dict(prompt_version=contract.version,
                prompt_fingerprint=contract_fingerprint(contract),
                request_model=contract.model_id, temperature=contract.temperature,
                max_tokens=contract.max_tokens)


class ChainReplayProvider:
    """Recorded chain candidates, returned through the existing HTTP adapter."""
    def __init__(self, responses=None):
        self.responses = list(responses) if responses is not None else [json.dumps({
            'reply': 'I can help check the return policy for order 42.',
            'status': 'needs_review'})]
        if not self.responses:
            raise ValueError('empty_chain_responses')
        self.calls = 0

    def key(self):
        return 'offline-chain'

    @asynccontextmanager
    async def client(self):
        def respond(request):
            payload = json.loads(request.content)
            candidate = self.responses[min(self.calls, len(self.responses) - 1)]
            self.calls += 1
            return httpx.Response(200, json={
                'model': payload['model'],
                'choices': [{'message': {'content': candidate}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client


class PromptReplayProvider:
    """Authored chat fixtures; no network and no dependence on legacy fixture keys."""
    def key(self):
        return 'offline-prompt-contract'

    @asynccontextmanager
    async def client(self):
        def respond(request):
            payload = json.loads(request.content)
            text = payload['messages'][-1]['content']
            reply = 'I can help with your refund request.'
            if 'Support:' in text:
                reply = 'Support: ' + reply
            return httpx.Response(200, json={
                'model': payload['model'],
                'choices': [{'message': {'content': reply}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30},
            })
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            yield client


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

    async def complete_chain_text(self, text, contract):
        """Consume an already rendered LCEL prompt; do not render_chat again."""
        class ChainPrompt(str):
            pass
        prompt = ChainPrompt(text)
        prompt.context_messages = [{'role': 'user', 'content': text}]
        prompt.context_model = contract.model_id
        prompt.context_temperature = contract.temperature
        provider = self.provider
        if isinstance(provider, (ReplayProvider, PromptReplayProvider)):
            provider = ChainReplayProvider()
        child = LlmClient(provider, sleep=self.sleep, clock=self.clock, jitter=self.jitter)
        try:
            with span('llm.chain', stage='provider', **contract_fields(contract)):
                result = await child.complete_result(prompt, max_tokens=contract.max_tokens)
            if result.finish_reason != 'stop':
                raise InvalidModelOutput('incomplete_output')
            return result.reply
        finally:
            self.attempts = child.attempts

    def embed(self, texts, input_type='document', *, provider=None, config=None,
              transport=None, api_key=None):
        return embed_texts(texts, input_type, provider=provider, config=config,
                          transport=transport, api_key=api_key)

    async def stream_tokens(self, prompt, *, contract=None):
        contract = selected_contract(prompt, contract)
        if context_options(prompt) is not None:
            prepared, assembly, provider = await self._prepare_context(prompt, contract)
            self.context_state, self.context_metrics = assembly.state, assembly.metrics
            with span('llm.call', stage='provider', **contract_fields(contract),
                      **assembly.metrics):
                source = provider_stream(provider, prepared)
                try:
                    async for delta in source:
                        if delta.text:
                            yield delta.text
                finally:
                    await source.aclose()
            return
        if contract is not None:
            source = self._stream_contract(prompt, contract)
        else:
            try:
                from .provider import stream_tokens
            except ImportError:
                from provider import stream_tokens
            source = stream_tokens(self.provider, prompt)
        try:
            async for delta in source:
                yield delta
        finally:
            await source.aclose()

    async def _stream_contract(self, prompt, contract):
        contract = selected_contract(prompt, contract)
        if isinstance(self.provider, (ReplayProvider, PromptReplayProvider)):
            result = await self.complete_result(prompt, contract=contract)
            for start in range(0, len(result.reply), 16):
                yield result.reply[start:start + 16]
            return
        payload = {'model': contract.model_id,
                   'messages': [{'role': 'user', 'content': render_chat(contract, prompt)}],
                   'temperature': contract.temperature, 'max_tokens': contract.max_tokens,
                   'stream': True, 'stream_options': {'include_usage': True}}
        with span('llm.call', stage='provider', **contract_fields(contract)) as telemetry:
            telemetry.set(attempts=1)
            try:
                async with asyncio.timeout(TOTAL_DEADLINE):
                    async with self.provider.client() as client:
                        async with client.stream(
                            'POST', 'https://api.openai.com/v1/chat/completions',
                            headers={'Authorization': 'Bearer ' + self.provider.key()},
                            json=payload, timeout=httpx.Timeout(2.0, connect=1.0),
                        ) as response:
                            telemetry.set(status_code=response.status_code,
                                          provider_request_id=response.headers.get('x-request-id'))
                            if response.status_code != 200:
                                await response.aread()
                                raise classify(response)
                            async for line in response.aiter_lines():
                                if not line.startswith('data:'):
                                    continue
                                data = line[5:].strip()
                                if data == '[DONE]':
                                    return
                                try:
                                    event = json.loads(data)
                                    telemetry.set(response_model=event.get('model'))
                                    if event.get('usage'):
                                        usage = Usage.model_validate(event['usage'])
                                        telemetry.set(tokens=tokens_from_usage(usage))
                                    for choice in event.get('choices', []):
                                        content = choice.get('delta', {}).get('content')
                                        if content is not None:
                                            if not isinstance(content, str):
                                                raise ValueError('invalid delta')
                                            yield content
                                        if choice.get('finish_reason'):
                                            telemetry.set(finish_reason=choice['finish_reason'])
                                except (ValueError, KeyError, IndexError, TypeError):
                                    raise ProviderError('invalid_provider_response') from None
                            raise ProviderError('invalid_provider_response')
            except (TimeoutError, httpx.TimeoutException):
                raise ProviderError('provider_timeout', retryable=True) from None
            except (httpx.ConnectError, httpx.ReadError):
                raise ProviderError('provider_transport_error', retryable=True) from None

    def complete(self, prompt, system=None, max_tokens=256, *, effort=None, contract=None):
        return asyncio.run(self.complete_result(
            prompt, system, max_tokens, effort=effort, contract=contract)).reply

    def complete_triage(self, ticket, *, variant='zero_shot', fixture_id=None):
        prompt = render(variant, ticket)
        original = self.provider
        try:
            if isinstance(original, ReplayProvider):
                self.provider = ReplayProvider(task='triage', variant=variant, fixture_id=fixture_id)
            result = asyncio.run(self.complete_result(prompt))
            if result.finish_reason != 'stop':
                raise ProviderError('invalid_triage_response', attempts=result.attempts)
            return TriageLabels.model_validate_json(result.reply)
        finally:
            self.provider = original

    async def complete_structured(self, ticket):
        provider = self.provider
        if isinstance(provider, ReplayProvider):
            record = structured_record(ticket)
            provider = ReplayProvider(task='structured', fixture_id=record['id'])
        client = LlmClient(provider, sleep=self.sleep, clock=self.clock, jitter=self.jitter)
        system, prompt = render_structured(ticket)
        result = await client.complete_result(prompt, system, response_format=TRIAGE_FORMAT)
        triage = validate_triage(result.reply)
        if result.finish_reason != 'stop':
            raise InvalidModelOutput('incomplete_output')
        return triage

    async def complete_vision(self, raw, ticket, cfg=None):
        from anchor_ai.vision import settings, prepare_image, VisionPrompt, OfflineVisionProvider
        from anchor_ai.schemas import IssueSummary
        from pydantic import ValidationError
        import base64
        cfg = settings() if cfg is None else cfg
        if cfg != settings():
            raise ProviderError('live_vision_not_verified', status=503)
        prepare_image(base64.b64encode(raw).decode(), cfg)
        provider = self.provider
        if isinstance(provider, (ReplayProvider, PromptReplayProvider)):
            provider = OfflineVisionProvider()
        if not isinstance(provider, OfflineVisionProvider):
            raise ProviderError('live_vision_not_verified', status=503)
        child = LlmClient(provider, sleep=self.sleep, clock=self.clock, jitter=self.jitter)
        response_format = {'type': 'json_schema', 'json_schema': {
            'name': 'issue_summary', 'strict': True, 'schema': IssueSummary.model_json_schema()}}
        try:
            result = await child.complete_result(VisionPrompt(raw, ticket, cfg),
                max_tokens=cfg['max_output_tokens'], response_format=response_format)
            if result.finish_reason != 'stop':
                raise InvalidModelOutput('incomplete_output')
            try:
                return IssueSummary.model_validate_json(result.reply)
            except ValidationError:
                raise InvalidModelOutput('schema_violation') from None
        finally:
            self.attempts = child.attempts

    async def complete_result(self, prompt, system=None, max_tokens=256, *, effort=None,
                              response_format=None, contract=None):
        contract = selected_contract(prompt, contract)
        if context_options(prompt) is not None:
            if system is not None or effort is not None or response_format is not None:
                raise ValueError('chat contract cannot be combined with other prompt policies')
            prepared, assembly, provider = await self._prepare_context(prompt, contract)
            child = LlmClient(provider, sleep=self.sleep, clock=self.clock, jitter=self.jitter)
            self.context_state, self.context_metrics = assembly.state, assembly.metrics
            try:
                result = await child.complete_result(prepared, max_tokens=contract.max_tokens)
                result.context_state = assembly.state
                return result
            finally:
                self.attempts = child.attempts
        if contract is not None:
            if system is not None or effort is not None or response_format is not None:
                raise ValueError('chat contract cannot be combined with other prompt policies')
            if isinstance(self.provider, ReplayProvider):
                client = LlmClient(PromptReplayProvider(), sleep=self.sleep,
                                   clock=self.clock, jitter=self.jitter)
                try:
                    return await client.complete_result(prompt, contract=contract)
                finally:
                    self.attempts = client.attempts
            prompt = render_chat(contract, prompt)
            max_tokens = contract.max_tokens
        self.attempts = 0
        self.last_error = ProviderError('provider_timeout', retryable=True)
        model = getattr(prompt, 'context_model',
                        contract.model_id if contract else os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'))
        offline = isinstance(self.provider, (ReplayProvider, PromptReplayProvider, ContextReplayProvider, ChainReplayProvider))
        offline = offline or getattr(self.provider, 'vision_offline', False) is True
        with span('llm.call', stage='provider', request_model=model,
                  response_model=None, currency='USD', price_version=PRICE_VERSION,
                  cost_basis='offline_replay_estimate' if offline else 'published_price_estimate') as telemetry:
            self.telemetry = telemetry
            try:
                if contract is not None:
                    telemetry.set(**contract_fields(contract))
                    fields = {'temperature': contract.temperature}
                else:
                    fields = request_fields(model, effort, max_tokens, offline=offline)
                if hasattr(prompt, 'context_messages'):
                    telemetry.set(**getattr(prompt, 'context_metrics', {}))
                    fields = dict(fields or {}, temperature=prompt.context_temperature)
                if response_format is not None:
                    fields = dict(fields or {})
                    fields['response_format'] = response_format
                if effort is not None:
                    telemetry.set(reasoning_setting=effort,
                                  reasoning_effort=fields['reasoning_effort'],
                                  max_completion_tokens=max_tokens)
                fixture_ms = getattr(self.provider, 'fixture_latency_ms', None)
                if fixture_ms is not None:
                    telemetry.set(fixture_latency_ms=fixture_ms, fixture_latency_basis='authored_example')
                if contract is not None or hasattr(prompt, 'context_messages'):
                    operation = self._run(prompt, system, max_tokens, fields, model=model)
                else:
                    operation = self._run(prompt, system, max_tokens, fields)
                return await asyncio.wait_for(operation, timeout=TOTAL_DEADLINE)
            except asyncio.TimeoutError:
                raise self._final_error() from None
            finally:
                telemetry.set(attempts=self.attempts)

    def _final_error(self):
        error = self.last_error
        error.attempts = self.attempts
        error.retry_after = safe_retry_after(error.retry_after)
        return error

    async def _run(self, prompt, system, max_tokens, reasoning_fields=None, *, model=None):
        key = self.provider.key()
        messages = []
        if system is not None:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        messages = getattr(prompt, 'context_messages', messages)
        payload = {'model': model or os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
                   'messages': messages, 'max_tokens': max_tokens}
        if reasoning_fields:
            if 'max_completion_tokens' in reasoning_fields:
                payload.pop('max_tokens')
            payload.update(reasoning_fields)
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
                        timeout=httpx.Timeout(min(2.0, remaining), connect=min(1.0, remaining)))
                    self.telemetry.set(provider_request_id=response.headers.get('x-request-id'),
                                       status_code=response.status_code)
                    if response.status_code == 200:
                        try:
                            body = response.json()
                            choice = body['choices'][0]
                            if 'response_format' in payload:
                                message = choice['message']
                                if message.get('refusal'):
                                    raise InvalidModelOutput('provider_refusal')
                                if not isinstance(message.get('content'), str):
                                    raise InvalidModelOutput('incomplete_output')
                            usage = Usage.model_validate(body['usage'])
                            if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
                                raise ValueError('inconsistent usage')
                            result = InferResponse(reply=choice['message']['content'],
                                tokens=usage.total_tokens, usage=usage,
                                finish_reason=choice['finish_reason'], attempts=self.attempts)
                            tokens = tokens_from_usage(usage)
                            response_model = body.get('model')
                            self.telemetry.set(response_model=response_model, tokens=tokens,
                                cost_estimate=estimate(response_model, tokens), finish_reason=result.finish_reason,
                                completion_status='incomplete' if result.finish_reason == 'length'
                                else 'completed' if result.finish_reason == 'stop' else 'other')
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

    async def _prepare_context(self, prompt, contract):
        if contract is None:
            raise ValueError('context requires a pinned chat contract')
        provider = self.provider
        if isinstance(provider, (ReplayProvider, PromptReplayProvider)):
            provider = ContextReplayProvider()
        history, state = context_options(prompt)
        async def summarize(messages, limit):
            with span('context.summary', stage='summary', summary_version=SUMMARY_VERSION,
                      summary_input_tokens=request_count(messages), summary_output_cap=limit):
                child = LlmClient(provider, sleep=self.sleep, clock=self.clock, jitter=self.jitter)
                source = PreparedPrompt('', type('Source', (), {'messages': messages})(), contract)
                source.context_max_tokens = limit
                result = await child.complete_result(source, max_tokens=limit)
                if result.finish_reason != 'stop':
                    raise ValueError('incomplete_context_summary')
                return result.reply
        with span('llm.call' if False else 'context.assemble', stage='context', **contract_fields(contract)) as telemetry:
            assembly = await assemble(contract, str(prompt), history, state, summarize)
        prepared = PreparedPrompt(str(prompt), assembly, contract)
        prepared.context_metrics = assembly.metrics
        return prepared, assembly, provider

    def embed_ingestion(self, texts):
        import hashlib
        from pathlib import Path
        import yaml
        from .pii import prepare
        texts = [prepare(text) for text in texts]
        root = Path(__file__).resolve().parents[1]
        provider = os.environ.get('ANCHOR_INGESTION_EMBEDDINGS', 'offline')
        if provider == 'offline':
            fixture = json.loads((root / 'anchor_ai/tests/fixtures/embedding_responses.json').read_text())['ingestion']
            known = {row['text']: row['vector'] for row in fixture['records']}
            dimensions = fixture['dimensions']
            vectors = []
            for text in texts:
                digest = hashlib.sha256(text.encode()).digest()
                vectors.append(known.get(text, [(digest[i] - 127.5) / 127.5 for i in range(dimensions)]))
            return dict(vectors=vectors, model=fixture['model'], revision=fixture['revision'], dimensions=dimensions)
        config = yaml.safe_load((root / 'config/retrieval.yml').read_text())['embeddings']['providers']
        if provider not in config:
            raise ValueError('invalid_ingestion_provider')
        result = self.embed(texts, input_type='document', provider=provider)
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        if isinstance(result, dict):
            vectors = result.get('vectors', result.get('embeddings'))
        else:
            vectors = getattr(result, 'vectors', getattr(result, 'embeddings', result))
        settings = config[provider]
        return dict(vectors=vectors, model=settings['model'], revision=settings['revision'], dimensions=settings['dimensions'])


class CalibrationClient:
    """Recorded calibration by default; never guesses an unseen response."""
    def __init__(self, *, offline=None):
        from pathlib import Path
        self.root = Path(__file__).resolve().parents[1]
        self.mode = 'offline' if offline is True else os.getenv('ANCHOR_CALIBRATION_PROVIDER', 'offline')
        if self.mode not in {'offline', 'live'}:
            raise ValueError('unknown_calibration_provider')
        self.calls = []
        self.records = {}
        if self.mode == 'offline':
            path = self.root / 'eval/fixtures/judge_calibration_responses.jsonl'
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['id'] in self.records:
                    raise ValueError('duplicate_recording')
                self.records[row['id']] = row
            path = self.root / 'labs/llm-judge-calibration/golden_v2.jsonl'
            self.examples = {row['id']: row for row in map(json.loads, path.read_text().splitlines())}

    def complete(self, prompt, system=None, max_tokens=1024):
        data = json.loads(prompt)
        self.calls.append(dict(prompt=prompt, system=system))
        if self.mode == 'live':
            result = asyncio.run(LlmClient('live').complete_result(
                prompt, system=system, max_tokens=max_tokens))
            if result.finish_reason != 'stop':
                raise ValueError('incomplete_judge_response')
            return result.reply
        row = data['example']
        revision = data['revision']
        if revision not in {'baseline', 'clarified'}:
            raise ValueError('unrecorded_rubric')
        expected = self.examples.get(row['id'])
        # Split and tags stay harness-only, as do all reference labels.
        if expected is None or row != {k: v for k, v in expected.items() if k not in {'split', 'tags'}}:
            raise ValueError('unrecorded_evidence')
        record = self.records[row['id']]
        return json.dumps(record.get(revision, record['baseline']))


class PrefixClient:
    """Cache-aware triage through the existing bounded HTTP client."""
    def __init__(self, provider, *, prices, ttl_seconds, synthetic=False):
        self.provider = provider
        self.prices, self.ttl_seconds = prices, ttl_seconds
        self.synthetic = synthetic or getattr(provider, 'prefix_offline', False) is True

    async def complete(self, prompt):
        from anchor_ai.provider import UsageTap
        from anchor_ai.observability import report, emit_provider_prefix
        tap = UsageTap(self.provider)
        client = LlmClient(tap)
        result = await client.complete_result(prompt)
        if result.finish_reason != 'stop':
            raise ValueError('incomplete_triage')
        validate_triage(result.reply)
        evidence = report(tap.raw_usage, style='chat', prices=self.prices,
                          ttl_seconds=self.ttl_seconds, synthetic=self.synthetic)
        emit_provider_prefix(evidence, prompt)
        return result.reply, evidence
