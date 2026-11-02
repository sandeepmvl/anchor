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


# Training exports use additional detectors without changing ingestion behavior.
FINETUNING_PATTERNS = (
    (EMAIL, '[REDACTED_EMAIL]'),
    (r'\b\d{3}[- ]\d{3}[- ]\d{4}\b', '[REDACTED_PHONE]'),
)


def finetuning_detected(text):
    patterns = [rule['pattern'] for rule in CONFIG['pii_patterns']]
    patterns += [pattern for pattern, _ in FINETUNING_PATTERNS]
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def redact_finetuning_text(text):
    # Apply training placeholders before overlapping ingestion rules.
    for pattern, replacement in FINETUNING_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.I)
    # Retain all configured protections for other detected sensitive values.
    return redact(text)


def redact_finetuning_batch(rows):
    """Return a complete safe batch, or raise without returning partial results."""
    try:
        safe = []
        for row in rows:
            if set(row) != {'input', 'output', 'label'}:
                raise ValueError()
            if row['label'] not in {'refund_review', 'refund_decline'}:
                raise ValueError()
            result = {'label': row['label']}
            for field in ('input', 'output'):
                value = row[field]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError()
                value = redact_finetuning_text(value)
                if not value.strip() or finetuning_detected(value):
                    raise ValueError()
                result[field] = value
            safe.append(result)
        return safe
    except Exception:
        # Never include source text or detector exception details in a log.
        raise ValueError('finetuning_redaction_failed') from None


def redact_memory(text):
    """Redact before memory persistence, buffering or episode export."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('invalid_memory_text')
    safe = ' '.join(redact_finetuning_text(text).split())
    if not safe or finetuning_detected(safe):
        raise ValueError('memory_redaction_failed')
    return safe


def redact_graph(value):
    """Prepare JSON context before graph storage, logs, or export."""
    if isinstance(value, str):
        safe = redact_finetuning_text(value)
        safe = re.sub(r'\+?\d[\d ()-]{8,}\d', '[REDACTED_PHONE]', safe)
        return safe
    if isinstance(value, list):
        return [redact_graph(item) for item in value]
    if isinstance(value, dict):
        return {redact_graph(str(key)): redact_graph(item)
                for key, item in value.items()}
    if value is None or type(value) in (int, float, bool):
        return value
    raise ValueError('unsupported_graph_value')


def redact_attachment_span(text):
    """Bounded sanitized span for attachment evidence persistence."""
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError('attachment_span_rejected')
    safe = redact_memory(text)
    if len(safe) > 4000:
        raise ValueError('attachment_span_rejected')
    return safe
