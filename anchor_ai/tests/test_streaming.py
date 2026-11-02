import asyncio
import json

from anchor_ai.streaming import Delta, stream_reply


def run(coroutine):
    return asyncio.run(coroutine)


def test_closing_iterator_cancels_pending_read_and_closes_resources():
    async def scenario():
        closed = asyncio.Event()
        cancelled = asyncio.Event()

        async def provider():
            try:
                yield Delta(text='Refunds ')
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            finally:
                closed.set()

        response = stream_reply(provider(), interval=0.001)
        packets = [await anext(response), await anext(response)]
        assert packets == [b'event: token\ndata: Refunds \n\n', b': ping\n\n']
        await response.aclose()
        assert cancelled.is_set()
        assert closed.is_set()
        assert not any(b'event: final' in packet for packet in packets)
        try:
            await anext(response)
            assert False, 'closed response yielded another event'
        except StopAsyncIteration:
            pass
    run(scenario())


def test_pings_preserve_read_and_final_reports_truncation():
    async def scenario():
        reads = 0

        async def provider():
            nonlocal reads
            reads += 1
            await asyncio.sleep(0.02)
            yield Delta(text='Refunds ')
            yield Delta(stop='length')

        packets = [packet async for packet in stream_reply(provider(), interval=0.001)]
        assert b': ping\n\n' in packets
        assert reads == 1
        assert sum(p.startswith(b'event: token') for p in packets) == 1
        terminal = packets[-1].decode().split('data: ', 1)[1]
        assert json.loads(terminal) == {
            'content': 'Refunds ', 'citations': [], 'truncated': True}
    run(scenario())


def test_error_and_read_timeout_do_not_emit_final():
    async def scenario(timeout):
        closed = False

        async def provider():
            nonlocal closed
            try:
                yield Delta(text='Refunds ')
                if timeout:
                    await asyncio.sleep(60)
                raise RuntimeError('private provider error')
            finally:
                closed = True

        packets = [p async for p in stream_reply(
            provider(), interval=0.001, read_timeout=0.005)]
        assert closed
        assert packets[-1].startswith(b'event: error')
        assert not any(p.startswith(b'event: final') for p in packets)
        assert b'private provider error' not in b''.join(packets)
        if timeout:
            assert b'provider_timeout' in packets[-1]
    run(scenario(False))
    run(scenario(True))
