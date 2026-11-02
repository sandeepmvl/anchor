import re
from html.parser import HTMLParser
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config/retrieval.yml').read_text())['ingestion']


def redact(text):
    for rule in CONFIG['pii_patterns']:
        text = re.sub(rule['pattern'], rule['replacement'], text, flags=re.I)
    return text


class Extractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def prepare(text, suffix='.txt'):
    if suffix not in {'.txt', '.md', '.html'}:
        raise ValueError('unsupported_document_type')
    if suffix == '.html':
        parser = Extractor()
        parser.feed(text)
        parser.close()
        text = ' '.join(parser.parts)
    text = ' '.join(redact(text).split())
    if not text:
        raise ValueError('empty_document')
    return text


# Evaluation data is redacted before any client call or report output.
# Apply configured detectors as well as this explicit email detector.
EMAIL = r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'


def redact_evaluation(value):
    if isinstance(value, str):
        return re.sub(EMAIL, '[REDACTED_EMAIL]', redact(value), flags=re.I)
    if isinstance(value, list):
        return [redact_evaluation(item) for item in value]
    if isinstance(value, dict):
        return {redact_evaluation(key): redact_evaluation(item)
                for key, item in value.items()}
    return value
