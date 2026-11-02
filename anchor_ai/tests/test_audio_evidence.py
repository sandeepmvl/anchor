from types import SimpleNamespace

import pytest

from anchor_ai.audio import AudioClient, extract_voice, generated_call
from anchor_ai.attachment_evidence import attachment_answer


@pytest.fixture(scope='module')
def raw():
    return generated_call()


@pytest.mark.parametrize('failed', [(), (1,)])
def test_voice_evidence_survives_pre_persistence_answer_validation(raw, failed):
    identity = SimpleNamespace(tenant_id=1, conversation_id=7)
    result = extract_voice(raw, identity, adapter=AudioClient(fail_chunks=failed))
    rows = [dict(row, attachment_id=12) for row in result['evidence']]
    answer = attachment_answer(rows, tenant_id=1, conversation_id=7)
    assert answer['citations'] == rows
    assert answer['tool_calls'] == []
    assert '235.000-239.000s; role=unknown' in answer['answer']
    assert 'person@example.com' not in answer['answer']
    assert 'REDACTED' in answer['answer']
    assert ('MISSING 230.000-470.000s' in answer['answer']) == bool(failed)
    assert ('Partial transcript' in (result['notice'] or '')) == bool(failed)


def evidence(span, region='audio_0'):
    return dict(tenant_id=1, conversation_id=7, kind='attachment',
                attachment_id=12, chunk_id=None, page=1, region=region,
                span=span, confidence=0.8, untrusted=True)


def test_audio_prefix_does_not_exempt_transcript_pii():
    row = evidence('[235.000-239.000s; role=unknown] person@example.com')
    with pytest.raises(ValueError, match='unsanitized_evidence'):
        attachment_answer([row], tenant_id=1, conversation_id=7)


@pytest.mark.parametrize('span,region', [
    ('[239.000-235.000s; role=unknown] hello', 'audio_0'),
    ('[700.000-999.000s; role=unknown] hello', 'audio_0'),
    ('[235.000-239.000s; role=agent] hello', 'audio_0'),
    ('[MISSING 230.000-470.000s] retry', 'audio_0'),
    ('[235.000-239.000s; role=unknown] hello', 'audio_gap_0'),
])
def test_invalid_audio_metadata_is_not_a_privacy_exemption(span, region):
    with pytest.raises(ValueError):
        attachment_answer([evidence(span, region)], tenant_id=1, conversation_id=7)


def test_image_evidence_still_rejects_unredacted_text():
    row = evidence('Email person@example.com', region='error')
    with pytest.raises(ValueError, match='unsanitized_evidence'):
        attachment_answer([row], tenant_id=1, conversation_id=7)
