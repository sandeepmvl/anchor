import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from anchor_ai import pii, observability
from anchor_ai.context import message, screen_context_text
from anchor_ai.pii import LocalNER, examples
from anchor_ai.streaming import stream_reply

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/pii_test_fixtures.json').read_text())


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv('ANCHOR_PII_HMAC_KEY', 'a7' * 32)


def assert_no_identifiers(value):
    encoded = json.dumps(value)
    for identifier in FIXTURE['identifiers']:
        assert identifier not in encoded


def test_local_ner_held_out_people_and_addresses():
    ner = LocalNER()
    for row in [('Paulo', 'Santos', 'Any', 'Seattle'),
                ('Nadia', 'Karim', 'Willow', 'Austin')]:
        for text, expected in examples(*row):
            assert [(text[s.start:s.end], s.kind) for s in ner.entities(text)] == expected


def test_structured_detectors_and_hmac_canonicalization():
    source = FIXTURE['message']
    spans = pii.detect(source)
    assert {'NAME', 'EMAIL', 'CARD'} <= {s.kind for s in spans}
    safe = pii.redact(source)
    assert_no_identifiers(safe)
    assert '[REDACTED_NAME]' in safe
    assert '[REDACTED_EMAIL]' in safe
    assert '[REDACTED_CARD_' in safe
    assert pii.redact(safe) == safe
    assert pii.rewrite('4111111111111111') == pii.rewrite('4111 1111 1111 1111')
    assert pii.luhn('4111111111111111')
    assert not pii.luhn(FIXTURE['invalid_card'])
    assert not any(s.kind == 'CARD' for s in pii.detect(FIXTURE['invalid_card']))
    assert any(s.kind == 'SSN' for s in pii.detect(FIXTURE['ssn']))
    assert not any(s.kind == 'SSN' for s in pii.detect('000-00-0000'))
    assert '[REDACTED_PHONE]' in pii.redact(FIXTURE['phone'])
    assert pii.rewrite(source, key=b'x' * 32) != pii.rewrite(source, key=b'y' * 32)


def test_prompt_enrichment_is_rewritten_before_model_or_trace_write():
    incoming = pii.redact(FIXTURE['message'])
    assembled = incoming + '\nContext: ' + FIXTURE['context']
    outbound = message('current', screen_context_text(assembled))
    prompt_trace = json.loads(json.dumps(outbound))
    assert_no_identifiers([outbound, prompt_trace])
    assert '123 Any Street' not in outbound['content']
    assert '[REDACTED_ADDRESS]' in outbound['content']
    assert pii.redact_value({'notes': FIXTURE['notes'], 'message': incoming}) == {'message': incoming}


def test_missing_key_and_detector_outage_fail_closed(monkeypatch):
    monkeypatch.delenv('ANCHOR_PII_HMAC_KEY')
    with pytest.raises(pii.RedactionUnavailable, match='^pii_redaction_unavailable$'):
        pii.redact(FIXTURE['message'])
    class BrokenNER:
        def entities(self, text):
            raise RuntimeError(FIXTURE['message'])
    with pytest.raises(pii.RedactionUnavailable, match='^pii_redaction_unavailable$'):
        pii.rewrite(FIXTURE['message'], ner=BrokenNER())


def test_every_stream_split_is_redacted_before_any_text_event(monkeypatch):
    logs = []
    monkeypatch.setattr(observability, 'SINK', logs.append)
    raw = FIXTURE['response']
    async def run(split):
        async def source():
            yield SimpleNamespace(text=raw[:split], stop=None)
            yield SimpleNamespace(text=raw[split:], stop=None)
            yield SimpleNamespace(text='', stop='stop')
        frames = [frame async for frame in stream_reply(source())]
        wire = b''.join(frames).decode()
        assert_no_identifiers(wire)
        assert '[REDACTED_NAME]' in wire
        assert '[REDACTED_CARD_' in wire
        assert 'event: final' in wire
    for split in range(len(raw) + 1):
        asyncio.run(run(split))
    assert_no_identifiers(logs)


def test_failed_stream_releases_no_partial_text():
    async def run():
        async def source():
            yield SimpleNamespace(text=FIXTURE['response'], stop=None)
            raise RuntimeError('detector-or-provider-outage')
        wire = b''.join([frame async for frame in stream_reply(source())]).decode()
        assert 'event: token' not in wire
        assert 'event: final' not in wire
        assert 'event: error' in wire
        assert_no_identifiers(wire)
    asyncio.run(run())


def test_disconnect_closes_buffered_upstream_before_any_text():
    async def run():
        closed = asyncio.Event()
        waiting = asyncio.Event()
        async def source():
            try:
                yield SimpleNamespace(text=FIXTURE['response'], stop=None)
                waiting.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
        stream = stream_reply(source(), interval=.001)
        first = await asyncio.wait_for(anext(stream), 1)
        assert first == b': ping\n\n'
        await asyncio.wait_for(waiting.wait(), 1)
        await asyncio.wait_for(stream.aclose(), 1)
        assert closed.is_set()
    asyncio.run(run())
