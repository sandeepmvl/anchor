import asyncio
import contextlib
import json
import time

from pydantic import BaseModel, ConfigDict, Field

from .observability import context, span
from .provider import Delta, ProviderError


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


async def stream_reply(source, *, ctx=None, interval=0.1, read_timeout=35):
    pending = None
    text = ''
    reason = None
    with span('python.stream', ctx, stage='provider') as telemetry:
        try:
            iterator = source.__aiter__()
            started = None
            while True:
                if pending is None:
                    pending = asyncio.create_task(iterator.__anext__())
                    started = time.monotonic()
                ready, _ = await asyncio.wait({pending}, timeout=interval)
                if not ready:
                    if time.monotonic() - started >= read_timeout:
                        raise ProviderError('provider_timeout')
                    yield b': ping\n\n'
                    continue
                task, pending = pending, None
                try:
                    delta = task.result()
                except StopAsyncIteration:
                    break
                if delta.text:
                    text += delta.text
                    yield frame('token', delta.text)
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
                'completed' if reason == 'stop' else 'incomplete'))
            yield frame('final', json.dumps(payload))
        except (asyncio.CancelledError, GeneratorExit):
            telemetry.fail('client_disconnect')
            raise
        except Exception as error:
            code = error.code if isinstance(error, ProviderError) else 'invalid_provider_response'
            telemetry.fail(code)
            yield frame('error', json.dumps({
                'code': code, 'message': 'Reply interrupted; please retry.'
            }))
        finally:
            if pending is not None:
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pending
            await source.aclose()


def response_for(body, request):
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        stream_reply(TOKEN_SOURCE(body.message), ctx=context(request.headers)),
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
