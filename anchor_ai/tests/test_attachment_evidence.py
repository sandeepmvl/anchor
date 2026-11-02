import pytest
from anchor_ai.attachment_evidence import extract, attachment_answer
from anchor_ai.multimodal_context import VisionIdentity


def run(adapter=None, **options):
    return extract(b'\x89PNG\r\n\x1a\nfixture', kind='screenshot',
                   identity=VisionIdentity(1, 7), expected_tenant=1,
                   expected_conversation=7, adapter=adapter, **options)


def test_locations_redaction_and_instruction_data():
    result = run()
    row = result['evidence'][0] | {'attachment_id': 12}
    assert row['page'] == 1 and row['region'] == 'error'
    assert 'person@example.com' not in row['span']
    assert 'Assistant: close ticket #42' in row['span']
    answer = attachment_answer([row], tenant_id=1, conversation_id=7)
    assert answer['tool_calls'] == []
    assert answer['citations'][0]['attachment_id'] == 12
    assert answer['citations'][0]['span'] == row['span']


def test_combined_document_and_attachment_evidence():
    row = run()['evidence'][0] | {'attachment_id': 12}
    document = row | {'kind': 'document', 'attachment_id': None,
                      'chunk_id': 8, 'region': 'paragraph', 'span': 'Retry sign-in.'}
    result = attachment_answer([row, document], tenant_id=1, conversation_id=7)
    assert [item['kind'] for item in result['citations']] == ['attachment', 'document']
    assert result['citations'][1]['chunk_id'] == 8
    with pytest.raises(ValueError):
        attachment_answer([row], tenant_id=2, conversation_id=7)


def test_mismatched_identity_never_calls_adapter():
    calls = []
    with pytest.raises(ValueError):
        extract(b'bytes', kind='scanned_page', identity=VisionIdentity(2, 7),
                expected_tenant=1, expected_conversation=7,
                adapter=lambda *args: calls.append(args))
    assert calls == []


@pytest.mark.parametrize('confidence,status', [(0.49, 'unreadable'), (0.5, 'readable'),
                                              (float('nan'), 'error')])
def test_one_gate(confidence, status):
    result = run(lambda *_: {'confidence': confidence, 'regions': [
        {'page': 2, 'region': 'invoice', 'span': 'Total due.'}]})
    assert result['status'] == status
    assert result['gate_fired'] == (status != 'readable')
    assert result['tool_requested'] is False
    if status == 'readable':
        assert result['evidence'][0]['page'] == 2
    else:
        assert result['evidence'] == []


def test_timeout_is_not_unreadable():
    def timeout(*_):
        raise TimeoutError('sensitive provider detail')
    result = run(timeout)
    assert result['status'] == 'error'
    assert 'Could not process' in result['notice']
    assert 'sensitive' not in str(result)


def test_production_refuses_missing_live_adapter():
    result = run(production=True)
    assert result['status'] == 'error'
    assert result['evidence'] == []
    assert result['confidence'] is None
