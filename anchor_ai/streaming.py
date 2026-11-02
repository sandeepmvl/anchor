import asyncio
import contextlib
import json
import math
import time

from pydantic import BaseModel, ConfigDict, Field

from .observability import context, span
from .provider import Delta, ProviderError


class AdaptivePacer:
    """Bounded character buffer. All times are local monotonic seconds."""
    def __init__(self, *, target_interval=0.025, chunk_chars=3,
                 max_buffer_chars=96, max_flush_delay=0.12):
        if (type(chunk_chars) is not int or type(max_buffer_chars) is not int
                or not 1 <= chunk_chars <= max_buffer_chars <= 65536
                or type(target_interval) not in (int, float)
                or type(max_flush_delay) not in (int, float)
                or not math.isfinite(target_interval)
                or not math.isfinite(max_flush_delay)
                or not 0 < target_interval <= max_flush_delay <= 1):
            raise ValueError('invalid_stream_pacing')
        self.target = target_interval
        self.chunk_chars = chunk_chars
        self.maximum = max_buffer_chars
        self.delay = max_flush_delay
        self.buffer = ''
        self.oldest = None
        self.last = None
        self.first = True
        self.last_arrival = None
        self.supply_interval = target_interval
        self.peak = 0

    def deadline(self):
        if not self.buffer:
            return None
        # An exponentially weighted arrival interval estimates available supply.
        interval = min(self.target, self.supply_interval)
        interval /= 1 + len(self.buffer) / self.chunk_chars
        return min(self.last + interval, self.oldest + self.delay)

    def _take(self, now):
        text = self.buffer[:self.chunk_chars]
        self.buffer = self.buffer[len(text):]
        self.last = now
        if not self.buffer:
            self.oldest = None
        return text

    def due(self, now):
        if not self.buffer or now + 1e-12 < self.deadline():
            return []
        # At the age limit drain the entire buffer. Never trade the bound for smoothness.
        expired = now + 1e-12 >= self.oldest + self.delay
        result = [self._take(now)]
        while expired and self.buffer:
            result.append(self._take(now))
        return result

    def offer(self, text, now):
        result = self.due(now)
        if not text:
            return result
        if self.last_arrival is not None:
            gap = max(0, now - self.last_arrival)
            self.supply_interval = .8 * self.supply_interval + .2 * gap
        self.last_arrival = now
        if self.first:
            self.first = False
            result.append(text[:self.chunk_chars])
            text = text[self.chunk_chars:]
            self.last = now
        while text:
            room = self.maximum - len(self.buffer)
            if room == 0:
                result.append(self._take(now))
                continue
            piece, text = text[:room], text[room:]
            if not self.buffer:
                self.oldest = now
            self.buffer += piece
            self.peak = max(self.peak, len(self.buffer))
        return result

    def clear(self):
        self.buffer = ''
        self.oldest = None


def pacing_configuration(path='config/performance.yml'):
    from pathlib import Path
    import yaml
    settings = yaml.safe_load(Path(path).read_text())['stream_pacing']
    if set(settings) != {'enabled', 'target_interval', 'chunk_chars',
                         'max_buffer_chars', 'max_flush_delay'}:
        raise ValueError('invalid_stream_pacing')
    if type(settings['enabled']) is not bool:
        raise ValueError('invalid_stream_pacing')
    options = {k: v for k, v in settings.items() if k != 'enabled'}
    AdaptivePacer(**options)  # Validate even when disabled; never silently ignore bad limits.
    return options if settings['enabled'] else None


class ChatReply(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    content: str = Field(min_length=1)
    citations: list[str]


def frame(name, data):
    return ('event: ' + name + '\n' + ''.join(
        'data: ' + line + '\n' for line in data.split('\n')) + '\n').encode()


async def provider_tokens(message):
    from .llm_client import LlmClient
    source = LlmClient().stream_tokens(message)
    try:
        async for delta in source:
            yield delta
    finally:
        await source.aclose()


TOKEN_SOURCE = provider_tokens


async def stream_reply(source, *, ctx=None, interval=0.1, read_timeout=35,
                       pacing=None):
    pending = None
    text = ''
    reason = None
    pacer = AdaptivePacer(**pacing) if pacing is not None else None
    origin = time.monotonic()
    first_text = False
    exhausted = False
    with span('python.stream', ctx, stage='provider') as telemetry:
        try:
            iterator = source.__aiter__()
            started = None
            while True:
                now = time.monotonic()
                if pacer:
                    for piece in pacer.due(now):
                        yield frame('token', piece)
                if exhausted:
                    if not pacer or not pacer.buffer:
                        break
                    await asyncio.sleep(max(0, pacer.deadline() - time.monotonic()))
                    continue
                if pending is None:
                    pending = asyncio.create_task(iterator.__anext__())
                    started = time.monotonic()
                wait = min(interval, max(0, read_timeout - (time.monotonic() - started)))
                if pacer and pacer.buffer:
                    wait = min(wait, max(0, pacer.deadline() - time.monotonic()))
                ready, _ = await asyncio.wait({pending}, timeout=wait)
                if not ready:
                    now = time.monotonic()
                    if now - started >= read_timeout:
                        raise ProviderError('provider_timeout')
                    if pacer and pacer.buffer:
                        for piece in pacer.due(now):
                            yield frame('token', piece)
                    else:
                        yield b': ping\n\n'
                    continue
                task, pending = pending, None
                try:
                    delta = task.result()
                except StopAsyncIteration:
                    exhausted = True
                    continue
                if delta.text:
                    now = time.monotonic()
                    if not first_text:
                        first_text = True
                        telemetry.set(server_ttft_ms=(now - origin) * 1000)
                    text += delta.text
                    pieces = pacer.offer(delta.text, now) if pacer else [delta.text]
                    for piece in pieces:
                        yield frame('token', piece)
                if delta.stop:
                    if reason is not None:
                        raise ProviderError('invalid_provider_response')
                    reason = delta.stop
            if reason not in {'stop', 'length', 'content_filter'}:
                raise ProviderError('invalid_provider_response')
            reply = ChatReply(content=text, citations=[])
            payload = reply.model_dump()
            payload['truncated'] = reason != 'stop'
            telemetry.set(finish_reason=reason, completion_status=(
                'completed' if reason == 'stop' else 'incomplete'),
                stream_total_ms=(time.monotonic() - origin) * 1000,
                pacing_peak_chars=pacer.peak if pacer else 0)
            yield frame('final', json.dumps(payload))
        except (asyncio.CancelledError, GeneratorExit):
            telemetry.fail('client_disconnect')
            raise
        except Exception as error:
            code = error.code if isinstance(error, ProviderError) else 'invalid_provider_response'
            telemetry.fail(code)
            message = ('Buffered provider routing does not replay streams.'
                       if code == 'routing_stream_unsupported'
                       else 'Reply interrupted; please retry.')
            yield frame('error', json.dumps({'code': code, 'message': message}))
        finally:
            if pacer:
                pacer.clear()
            if pending is not None:
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pending
            await source.aclose()


def response_for(body, request):
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        stream_reply(TOKEN_SOURCE(body.message), ctx=context(request.headers),
                     pacing=pacing_configuration()),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


async def cancel_tasks(tasks):
    """Drain all session tasks, including tasks that failed before shutdown."""
    pending = [task for task in tasks if task is not None]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
