from pathlib import Path
import runpy
import socket
import pytest
from anchor_ai.ocr import Limits, RecordedVision, character_error, extract_pdf
from anchor_ai.pii import prepare


HELPER = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'labs/ocr-pipelines/fixture.py'))
FIXTURES = HELPER['FIXTURES']
invoice_pdf = HELPER['invoice_pdf']
REFERENCE = (FIXTURES / 'scanned_invoice_reference.txt').read_text()
INDEX_REFERENCE = prepare(REFERENCE)


def test_rotated_pixels_meet_configured_fixture_threshold():
    limits = Limits()
    result = extract_pdf(invoice_pdf(), limits=limits)
    assert result.engine == 'fixture' and result.fixture
    assert result.pages[0].page == 1
    assert result.pages[0].quality == 'accepted'
    assert character_error(result.pages[0].text, REFERENCE) <= limits.max_char_error
    assert 'accuracy_not_calibrated' in result.pages[0].warnings


@pytest.mark.parametrize('options,status', [({'blank': True}, 'empty'),
                                           ({'unreadable': True}, 'unreadable')])
def test_non_evidence_pages(options, status):
    result = extract_pdf(invoice_pdf(**options))
    assert result.pages[0].quality == status
    assert result.pages[0].text == ''
    with pytest.raises(ValueError, match='no_accepted'):
        result.require_evidence(allow_fixture=True)


def test_fixture_does_not_authorize_production_release():
    with pytest.raises(ValueError, match='fixture_evidence_not_production_verified'):
        extract_pdf(invoice_pdf()).require_evidence()


def test_recorded_vision_never_calls_network(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError('network prohibited')
    monkeypatch.setattr(socket.socket, 'connect', prohibited)
    raw = invoice_pdf()
    vision = RecordedVision(raw, FIXTURES / 'ocr_responses.json')
    result = extract_pdf(raw, engine='vision_fixture', vision=vision)
    assert character_error(result.pages[0].text, REFERENCE) == 0
    with pytest.raises(ValueError, match='unrecorded'):
        vision.extract(invoice_pdf(blank=True))


@pytest.mark.parametrize('limits,raw,error', [
    (Limits(max_pages=1), invoice_pdf(pages=2), ValueError),
    (Limits(max_pixels=10), invoice_pdf(), ValueError),
    (Limits(max_bytes=10), invoice_pdf(), ValueError),
    (Limits(seconds=1e-12), invoice_pdf(), TimeoutError),
    (Limits(), b'%PDF-broken', ValueError),
])
def test_limits(limits, raw, error):
    with pytest.raises(error):
        extract_pdf(raw, limits=limits)


def test_missing_local_infrastructure_fails_closed(monkeypatch):
    monkeypatch.setattr('anchor_ai.ocr.shutil.which', lambda executable: None)
    with pytest.raises(ValueError, match='infrastructure_missing'):
        extract_pdf(invoice_pdf(), engine='tesseract')


def test_ingestion_preserves_tenant_and_page_provenance(tmp_path, monkeypatch):
    from anchor_ai.ingestion import ingest_scanned_pdf
    calls = []
    def maintain(*args, **kwargs):
        calls.append((args, kwargs))
        return {'indexed': True}
    monkeypatch.setattr('anchor_ai.index_maintenance.maintain_document', maintain)
    path = tmp_path / 'invoice.pdf'
    path.write_bytes(invoice_pdf())
    result = ingest_scanned_pdf(None, 7, path, engine='fixture', allow_fixture=True)
    assert result == {'indexed': True}
    args, kwargs = calls[0]
    assert args[1] == 7
    assert args[3] == INDEX_REFERENCE
    assert args[4]['tenant_id'] == 7
    page = args[4]['pages'][0]
    assert page['page'] == 1 and page['extraction_quality'] == 'accepted'
    assert args[3][page['start']:page['end']] == INDEX_REFERENCE
    assert len(page['source_sha256']) == 64


@pytest.mark.parametrize('raw,limits,allow_fixture', [
    (b'%PDF-broken', Limits(), True),
    (invoice_pdf(pages=2), Limits(max_pages=1), True),
    (invoice_pdf(), Limits(max_pixels=10), True),
    (invoice_pdf(), Limits(seconds=1e-12), True),
    (invoice_pdf(blank=True), Limits(), True),
    (invoice_pdf(), Limits(), False),
])
def test_ingestion_rejects_before_index_write(tmp_path, monkeypatch, raw, limits, allow_fixture):
    from anchor_ai.ingestion import ingest_scanned_pdf
    calls = []
    monkeypatch.setattr('anchor_ai.index_maintenance.maintain_document',
                        lambda *args, **kwargs: calls.append(args))
    path = tmp_path / 'invoice.pdf'
    path.write_bytes(raw)
    with pytest.raises((ValueError, TimeoutError)):
        ingest_scanned_pdf(None, 7, path, engine='fixture', limits=limits,
                           allow_fixture=allow_fixture)
    assert calls == []


def test_application_extraction_uses_shared_contract():
    from anchor_ai.app import extract_document
    from anchor_ai.schemas import OcrExtraction, OcrPage
    from anchor_ai.vision import recorded_document_vision
    raw = invoice_pdf()
    vision = recorded_document_vision(raw, FIXTURES / 'ocr_responses.json')
    result = extract_document(raw, engine='vision_fixture', vision=vision)
    assert isinstance(result, OcrExtraction)
    assert isinstance(result.pages[0], OcrPage)
    assert character_error(result.pages[0].text, REFERENCE) == 0
