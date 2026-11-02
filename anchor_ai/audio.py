"""Recorded-call preprocessing and evidence; no automatic live-provider fallback."""
import hashlib
import io
import json
import math
import re
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'anchor_ai/tests/fixtures'
RATE = 16000
MAX_UPLOAD = 24_000_000
MAX_DURATION = 720
CHUNK_SECONDS = 240
CHUNK_BYTES = 8_000_000
OVERLAP = 10


def reference():
    return json.loads((FIXTURES / 'call_reference.json').read_text())


def encode(samples):
    target = io.BytesIO()
    with wave.open(target, 'wb') as stream:
        stream.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        stream.writeframes(np.asarray(samples, dtype='<i2').tobytes())
    return target.getvalue()


def generated_call():
    """Deterministic tones, silence and seeded noise; these are not human speech."""
    samples = np.zeros(MAX_DURATION * RATE, dtype='<i2')
    rng = np.random.default_rng(7)
    for turn in reference()['turns']:
        start, end = turn['start'], turn['end']
        t = np.arange((end - start) * RATE) / RATE
        sound = 5000 * np.sin(2 * np.pi * 220 * t)
        if start == 465:
            sound += rng.normal(0, 2000, len(t))
        samples[start * RATE:end * RATE] = np.clip(sound, -32768, 32767).astype('<i2')
    return encode(samples)


def decode(raw):
    if not isinstance(raw, bytes) or not 44 < len(raw) <= MAX_UPLOAD:
        raise ValueError('audio_size_rejected')
    try:
        with wave.open(io.BytesIO(raw), 'rb') as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                    stream.getcomptype()) != (1, 2, RATE, 'NONE'):
                raise ValueError('audio_format_rejected')
            frames = stream.getnframes()
            if not 0 < frames <= MAX_DURATION * RATE:
                raise ValueError('audio_duration_rejected')
            pcm = stream.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError('audio_truncated')
            return np.frombuffer(pcm, dtype='<i2')
    except (wave.Error, EOFError):
        raise ValueError('audio_format_rejected') from None


def normalize(text):
    return re.findall(r'[a-z0-9]+', text.casefold())


def wer(ref, hyp):
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        raise ValueError('empty_reference')
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        nxt = [i]
        for j, other in enumerate(hyp, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = nxt
    return row[-1] / len(ref)


def rms(samples):
    return float(np.sqrt(np.mean(samples.astype(float) ** 2))) if len(samples) else 0.0


def windows(samples, *, policy='silence', seconds=CHUNK_SECONDS, byte_limit=CHUNK_BYTES):
    if policy not in {'fixed', 'silence'}:
        raise ValueError('split_policy_rejected')
    cap = min(int(seconds * RATE), (byte_limit - 44) // 2)
    overlap = OVERLAP * RATE
    if cap <= overlap + RATE:
        raise ValueError('chunk_budget_rejected')
    start = 0
    while start < len(samples):
        end = min(start + cap, len(samples))
        if policy == 'silence' and end < len(samples):
            # Search backwards, but never move a cut beyond its hard budget.
            for candidate in range(end, max(start + overlap + RATE, end - 20 * RATE), -RATE):
                if rms(samples[candidate - RATE:candidate]) < 100:
                    end = candidate
                    break
        raw = encode(samples[start:end])
        if len(raw) > byte_limit or (end - start) / RATE > seconds:
            raise ValueError('chunk_limit_exceeded')
        yield start / RATE, end / RATE, raw
        if end == len(samples):
            break
        start = end - overlap


class AudioClient:
    """Audio LLM wrapper. Inject a real adapter with the same transcribe signature."""
    offline = True

    def __init__(self, *, fail_chunks=()):
        self.fail_chunks = set(fail_chunks)
        self.calls = []
        self.rows = json.loads((FIXTURES / 'call_transcription_responses.json').read_text())['segments']

    def verify_recording(self, raw):
        if hashlib.sha256(raw).digest() != hashlib.sha256(generated_call()).digest():
            raise ValueError('unrecorded_audio_fixture')

    def transcribe(self, raw, *, start, end, prompt):
        index = len(self.calls)
        self.calls.append({'index': index, 'start': start, 'end': end,
                           'bytes': len(raw), 'prompt_characters': len(prompt)})
        if index in self.fail_chunks:
            raise TimeoutError('fixture_chunk_failure')
        return [dict(start=max(s['start'], start) - start,
                     end=min(s['end'], end) - start, text=s['text'])
                for s in self.rows if s['start'] < end and s['end'] > start]


def align(segments, turns, verified_roles=None):
    """Anonymous acoustic labels are separate from trusted business-role metadata."""
    output = []
    roles = verified_roles or {}
    if any(role not in {'agent', 'customer'} for role in roles.values()):
        raise ValueError('role_metadata_rejected')
    for segment in segments:
        scores = {}
        for turn in turns:
            a, b = turn['start'], turn['end']
            if not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)) or not 0 <= a < b:
                raise ValueError('speaker_interval_rejected')
            speaker = turn['speaker']
            scores[speaker] = scores.get(speaker, 0) + max(
                0, min(segment['end'], b) - max(segment['start'], a))
        best = max(scores.values(), default=0)
        winners = [speaker for speaker, score in scores.items() if score == best and best > 0]
        speaker = winners[0] if len(winners) == 1 else 'unknown'
        output.append(dict(segment, speaker=speaker, role=roles.get(speaker, 'unknown')))
    return output


def transcribe(raw, *, client=None, production=False, policy='silence'):
    from anchor_ai.pii import redact_attachment_span
    client = client or AudioClient()
    if production and getattr(client, 'offline', True):
        raise ValueError('live_audio_not_configured')
    samples = decode(raw)
    if getattr(client, 'offline', True):
        client.verify_recording(raw)
    segments, gaps, reviews = [], [], []
    tail = ''
    previous_end = 0
    rejected = 0
    for start, end, chunk in windows(samples, policy=policy):
        try:
            rows = client.transcribe(chunk, start=start, end=end, prompt=tail)
            if not isinstance(rows, list) or len(rows) > 1000:
                raise ValueError('transcript_shape')
            accepted = []
            for row in rows:
                a, b, text = row['start'], row['end'], row['text']
                if (not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b))
                        or not 0 <= a < b <= end - start or not isinstance(text, str)
                        or not text.strip() or len(text) > 4000):
                    raise ValueError('transcript_segment_rejected')
                a, b = a + start, b + start
                if rms(samples[int(a * RATE):int(b * RATE)]) < 100:
                    rejected += 1
                    continue
                accepted.append({'start': a, 'end': b, 'text': text})
            accepted.sort(key=lambda row: (row['start'], row['end']))
        except Exception:
            # No provider exception text, request audio or partial response escapes.
            gaps.append({'start': start, 'end': end, 'reason': 'chunk_failed'})
            tail = ''
            previous_end = end
            continue
        for row in accepted:
            shared = row['start'] < previous_end and row['end'] > start
            duplicate = shared and any(
                normalize(old['text']) == normalize(row['text'])
                and min(old['end'], row['end'], previous_end) > max(old['start'], row['start'], start)
                for old in segments)
            if duplicate:
                continue
            if shared and any(min(old['end'], row['end']) > max(old['start'], row['start'])
                              for old in segments):
                reviews.append({'start': row['start'], 'end': row['end'], 'reason': 'overlap_review'})
            segments.append(row)
        # Prompts remain process-local and are never written to the chunk ledger.
        tail = ' '.join(' '.join(row['text'] for row in accepted).split()[-40:])
        previous_end = end
    segments.sort(key=lambda row: (row['start'], row['end']))
    # Acoustic annotations in a reference fixture are not recording identity metadata.
    segments = align(segments, [])
    for row in segments:
        row['text'] = redact_attachment_span(row['text'])
    return {'segments': segments, 'gaps': gaps, 'reviews': reviews,
            'silence_rejections': rejected, 'ledger': client.calls,
            'partial': bool(gaps), 'offline_fixture': getattr(client, 'offline', True)}


def extract_voice(raw, identity, *, adapter=None, production=False):
    result = transcribe(raw, client=adapter, production=production)
    evidence = []
    for index, segment in enumerate(result['segments']):
        evidence.append({'tenant_id': identity.tenant_id,
                         'conversation_id': identity.conversation_id,
                         'kind': 'attachment', 'attachment_id': None, 'chunk_id': None,
                         'page': 1, 'region': f'audio_{index}',
                         'span': f"[{segment['start']:.3f}-{segment['end']:.3f}s; role=unknown] {segment['text']}",
                         'confidence': 0.8, 'untrusted': True})
    for index, gap in enumerate(result['gaps']):
        evidence.append({'tenant_id': identity.tenant_id,
                         'conversation_id': identity.conversation_id,
                         'kind': 'attachment', 'attachment_id': None, 'chunk_id': None,
                         'page': 1, 'region': f'audio_gap_{index}',
                         'span': f"[MISSING {gap['start']:.3f}-{gap['end']:.3f}s] Transcription failed; re-upload or ask a human.",
                         'confidence': 0.8, 'untrusted': True})
    readable = bool(result['segments'])
    notice = None
    if result['partial']:
        notice = 'Partial transcript: marked intervals are missing. Re-upload to retry or ask a human.'
    elif result['reviews']:
        notice = 'Transcript overlap needs human review.'
    elif not readable:
        notice = 'No readable speech; re-upload or ask a human.'
    return {'status': 'readable' if readable else 'error', 'evidence': evidence if readable else [],
            'confidence': 0.8 if readable else None, 'adapter': 'audio',
            'gate_fired': not readable or result['partial'], 'tool_requested': False,
            'notice': notice, 'offline_fixture': result['offline_fixture']}


def realtime_frame(raw):
    """Decode one 20 ms PCM16/16 kHz mono frame with its sequence number."""
    import struct
    if not isinstance(raw, bytes) or len(raw) != 648:
        raise ValueError('audio_frame_rejected')
    sequence = struct.unpack('!Q', raw[:8])[0]
    pcm = raw[8:]
    speech = max(abs(v) for v in struct.unpack('<320h', pcm)) > 500
    return sequence, pcm, speech
