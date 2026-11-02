import pytest
from anchor_ai.app import extract_ticket_attachment
from anchor_ai.context import ContextError, pack_attachment_evidence
from anchor_ai.multimodal_context import sign_test_context, VisionContextRejected
from anchor_ai.orchestration import run_attachment_answer
from anchor_ai.pii import redact_attachment_span


def test_signed_scope_is_checked_before_extraction():
    calls = []
    with pytest.raises(VisionContextRejected):
        extract_ticket_attachment(
            b'bytes', kind='screenshot', token=sign_test_context('secret', 2, 7),
            secret='secret', expected_scope={'tenant_id': 1, 'conversation_id': 7},
            adapter=lambda *args: calls.append(args))
    assert calls == []


def test_mixed_evidence_survives_product_seams():
    result = extract_ticket_attachment(
        b'bytes', kind='screenshot', token=sign_test_context('secret', 1, 7),
        secret='secret', expected_scope={'tenant_id': 1, 'conversation_id': 7},
        adapter=lambda *_: {'confidence': 0.9, 'regions': [
            {'page': 1, 'region': 'error', 'span': 'Error E42.'}]})
    attachment = result['evidence'][0] | {'attachment_id': 12}
    document = attachment | {'kind': 'document', 'attachment_id': None,
                             'chunk_id': 8, 'region': 'paragraph',
                             'span': 'Contact support for E42.'}
    rows = [attachment, document]
    answer = run_attachment_answer(rows, tenant_id=1, conversation_id=7)
    assert answer['citations'] == rows
    assert answer['tool_calls'] == []
    with pytest.raises(ContextError):
        pack_attachment_evidence(rows, tenant_id=2, conversation_id=7)
    with pytest.raises(ContextError):
        pack_attachment_evidence(rows, tenant_id=1, conversation_id=7, budget_bytes=10)


def test_attachment_redaction_is_bounded():
    assert 'person@example.com' not in redact_attachment_span('Email person@example.com')
    with pytest.raises(ValueError):
        redact_attachment_span('x' * 4001)


def test_internal_entry_point_refuses_missing_production_adapter():
    result = extract_ticket_attachment(
        b'bytes', kind='scanned_page', token=sign_test_context('secret', 1, 7),
        secret='secret', expected_scope={'tenant_id': 1, 'conversation_id': 7},
        production=True)
    assert result['status'] == 'error' and result['evidence'] == []
