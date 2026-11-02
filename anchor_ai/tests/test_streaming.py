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
        assert packets == [b': ping\n\n', b': ping\n\n']
        assert not any(b'event: token' in packet for packet in packets)
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


def test_fake_clock_first_text_order_buffer_and_delay():
    from anchor_ai.streaming import AdaptivePacer
    pacer = AdaptivePacer(chunk_chars=2, max_buffer_chars=6,
                          target_interval=.02, max_flush_delay=.06)
    now = .25
    text = 'Refunds take five days.'
    output = [(now, piece) for piece in pacer.offer(text, now)]
    assert output[0] == (now, 'Re')
    assert len(pacer.buffer) <= 6 and pacer.peak <= 6
    while pacer.buffer:
        now = pacer.deadline()
        output.extend((now, piece) for piece in pacer.due(now))
    assert ''.join(piece for _, piece in output) == text
    assert max(t - .25 for t, _ in output) <= .06 + 1e-10
    assert all(len(piece) <= 2 for _, piece in output)
    assert pacer.deadline() is None


def test_fake_clock_age_limit_wins_over_interval():
    from anchor_ai.streaming import AdaptivePacer
    pacer = AdaptivePacer(chunk_chars=1, max_buffer_chars=20,
                          target_interval=.05, max_flush_delay=.05)
    assert pacer.offer('abcdefghij', 1) == ['a']
    assert pacer.due(1.05) == list('bcdefghij')
    assert not pacer.buffer
    pacer.offer('more', 2)
    pacer.clear()
    assert not pacer.buffer and pacer.deadline() is None


def test_configuration_is_validated_even_when_disabled(tmp_path):
    import pytest
    from anchor_ai.streaming import pacing_configuration
    path = tmp_path / 'performance.yml'
    path.write_text('stream_pacing:\n  enabled: false\n  target_interval: 0.02\n'
                    '  chunk_chars: 2\n  max_buffer_chars: 6\n  max_flush_delay: 0.06\n')
    assert pacing_configuration(path) is None
    path.write_text(path.read_text().replace('chunk_chars: 2', 'chunk_chars: 0'))
    with pytest.raises(ValueError, match='invalid_stream_pacing'):
        pacing_configuration(path)


def test_paced_disconnect_cancels_read_and_clears_buffer(monkeypatch):
    from anchor_ai import streaming
    instances = []
    original = streaming.AdaptivePacer

    class ObservedPacer(original):
        def __init__(self, **options):
            super().__init__(**options)
            instances.append(self)

    monkeypatch.setattr(streaming, 'AdaptivePacer', ObservedPacer)

    async def scenario():
        cancelled, closed = asyncio.Event(), asyncio.Event()

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

        response = stream_reply(provider(), interval=.001,
                                pacing={'chunk_chars': 2})
        # No text reaches the pacer before complete-answer redaction.
        assert await anext(response) == b': ping\n\n'
        assert instances and not instances[0].buffer
        await response.aclose()
        assert cancelled.is_set() and closed.is_set()
        assert instances and not instances[0].buffer
    run(scenario())
