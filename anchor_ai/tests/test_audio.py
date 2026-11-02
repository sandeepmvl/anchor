import pytest
import numpy as np

from anchor_ai.audio import (
    AudioClient, CHUNK_BYTES, CHUNK_SECONDS, MAX_UPLOAD, align, decode,
    generated_call, normalize, reference, transcribe, wer, windows,
)
from anchor_ai.llm_client import transcribe_audio
from anchor_ai.schemas import AudioSegment


@pytest.fixture(scope='module')
def raw():
    return generated_call()


def test_bounded_call_reassembly_and_quality(raw):
    assert len(raw) < MAX_UPLOAD
    client = AudioClient()
    result = transcribe_audio(raw, client=client)
    ledger = result['ledger']
    assert len(ledger) > 1
    assert all(row['bytes'] <= CHUNK_BYTES for row in ledger)
    assert all(row['end'] - row['start'] <= CHUNK_SECONDS for row in ledger)
    assert ledger[0]['start'] == 0 and ledger[-1]['end'] == 720
    assert ledger[0]['prompt_characters'] == 0
    assert ledger[1]['prompt_characters'] > 0
    for left, right in zip(ledger, ledger[1:]):
        assert left['start'] < right['start'] < left['end']
    segments = result['segments']
    assert [row['start'] for row in segments] == [20, 235, 465, 700]
    assert [row['end'] for row in segments] == [24, 239, 469, 704]
    assert all(AudioSegment.model_validate(row).role == 'unknown' for row in segments)
    assert all(row['role'] == row['speaker'] == 'unknown' for row in segments)
    assert result['silence_rejections'] == 1
    assert not result['partial'] and not result['gaps']
    assert 'person@example.com' not in str(result)
    assert 'REDACTED' in segments[1]['text']
    # Evaluate recognition bookkeeping before redaction changes reference tokens.
    responses = client.rows
    ref = ' '.join(turn['text'] for turn in reference()['turns'])
    hyp = ' '.join(row['text'] for row in responses if row['start'] != 100)
    assert 0 < wer(ref, hyp) < 0.1
    assert wer('ONE, two!', 'one two') == 0
    assert wer('one two three four five six seven eight nine ten',
               'one too three four six seven eight nine ten extra') == 0.3


def test_fixed_and_silence_policies_keep_hard_caps(raw):
    samples = decode(raw)
    for policy in ('fixed', 'silence'):
        chunks = list(windows(samples, policy=policy))
        assert chunks[0][0] == 0 and chunks[-1][1] == 720
        assert all(len(data) <= CHUNK_BYTES and end - start <= CHUNK_SECONDS
                   for start, end, data in chunks)
    # A pause before a hard cut can move a silence-aware boundary backwards.
    samples = np.ones(260 * 16000, dtype='<i2') * 5000
    samples[235 * 16000:236 * 16000] = 0
    assert next(windows(samples, policy='silence'))[1] == 236
    assert next(windows(samples, policy='fixed'))[1] == 240


def test_failure_marks_interval_and_resets_prompt(raw):
    client = AudioClient(fail_chunks={1})
    result = transcribe(raw, client=client)
    assert result['partial']
    assert result['gaps'] == [{'start': 230.0, 'end': 470.0, 'reason': 'chunk_failed'}]
    assert client.calls[2]['prompt_characters'] == 0
    # The first request recovers 235–239; the third recovers 465–469.
    # A failed request is not permission to discard successful overlap evidence.
    assert [row['start'] for row in result['segments']] == [20, 235, 465, 700]
    assert [row['end'] for row in result['segments']] == [24, 239, 469, 704]
    assert len(result['segments']) == 4


def test_maximum_overlap_does_not_guess_roles():
    segments = [{'start': 0, 'end': 3, 'text': 'hello'}]
    turns = [{'start': 0, 'end': 2, 'speaker': 'a'},
             {'start': 2, 'end': 3, 'speaker': 'b'}]
    assert align(segments, turns)[0]['speaker'] == 'a'
    assert align(segments, turns)[0]['role'] == 'unknown'
    assert align(segments, turns, {'a': 'customer'})[0]['role'] == 'customer'
    tie = align([{'start': 1, 'end': 3, 'text': 'hello'}], turns)
    assert tie[0]['speaker'] == tie[0]['role'] == 'unknown'


def test_production_refuses_fixture_engine(raw):
    with pytest.raises(ValueError, match='live_audio_not_configured'):
        transcribe(raw, client=AudioClient(), production=True)


def test_unknown_recording_cannot_receive_fixture_transcript(raw):
    changed = bytearray(raw)
    changed[-1] = 1
    with pytest.raises(ValueError, match='unrecorded_audio_fixture'):
        transcribe(bytes(changed))


def test_bad_audio_and_budgets_fail_closed():
    with pytest.raises(ValueError):
        decode(b'not audio')
    with pytest.raises(ValueError):
        list(windows(np.zeros(100), byte_limit=100))
