import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fixture import FIXTURES, write_invoice
from anchor_ai.document_loader import load_document
from anchor_ai.llm_client import extract_document
from anchor_ai.ocr import Limits, character_error
from anchor_ai.vision import recorded_document_vision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tesseract', action='store_true')
    args = parser.parse_args()
    path = write_invoice()
    raw = path.read_bytes()
    reference = (FIXTURES / 'scanned_invoice_reference.txt').read_text()
    limits = Limits()
    engine = 'tesseract' if args.tesseract else 'fixture'
    local = extract_document(raw, engine=engine, limits=limits)
    vision = extract_document(raw, engine='vision_fixture', limits=limits,
                              vision=recorded_document_vision(raw, FIXTURES / 'ocr_responses.json'))
    records = []
    for result in (local, vision):
        text = '\n'.join(page.text for page in result.pages)
        error = character_error(text, reference)
        assert error <= limits.max_char_error, f'{result.engine}: CER {error} exceeds threshold'
        records.append({**result.as_dict(), 'character_error_rate': error,
                        'runtime_basis': 'measured local execution, including replay overhead'})
    document = load_document(path, ocr_engine=engine, allow_fixture=not args.tesseract)
    costs = json.loads((FIXTURES / 'ocr_responses.json').read_text())['hypothetical_cost']
    estimated = (costs['input_tokens'] * costs['input_usd_per_million'] +
                 costs['output_tokens'] * costs['output_usd_per_million']) / 1_000_000
    print(json.dumps({
        'before': 'scanned invoice: no searchable text layer',
        'after': {'text': document.text,
                  'page_evidence': [unit.metadata for unit in document.units]},
        'comparisons': records,
        'vision_estimated_provider_cost_usd': estimated,
        'cost_basis': 'invented planning inputs; recorded response; no billed provider usage',
        'production_evidence': args.tesseract,
        'not_verified': ['container isolation', 'held-out production invoice accuracy'] +
                        ([] if args.tesseract else ['Docker build', 'Tesseract execution and accuracy']),
    }, indent=2))


if __name__ == '__main__':
    main()
