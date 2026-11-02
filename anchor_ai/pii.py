import hashlib
import hmac
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config/retrieval.yml').read_text())['ingestion']
EMAIL = r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
TOKEN = re.compile(r'\[REDACTED_[A-Z]+(?:_[0-9a-f]{64})?\]')
MAX_TEXT = 65536


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    kind: str


NER_TOKEN = re.compile(r'\[REDACTED_[A-Z]+(?:_[0-9a-f]{64})?\]|[A-Za-z]+|[0-9]+|[^\w\s]')


def normalized(value):
    if value.startswith('[REDACTED_'):
        return 'MASK'
    if value.isdigit():
        return 'DIGITS:' + str(len(value))
    if value.isalpha() and value[0].isupper():
        return 'TITLE'
    return value.lower()


def signature(tokens, index):
    return tuple(normalized(tokens[j].group()) if 0 <= j < len(tokens) else '<EDGE>'
                 for j in range(index - 2, index + 3))


def examples(given, surname, street, city):
    name = given + ' ' + surname
    address = '123 ' + street + ' Street, ' + city + ', WA 98109'
    billing = name + ', email paulo@example.test, card 4111 1111 1111 1111.'
    return [
        (billing, [(name, 'NAME')]),
        ('Policy: returns within 30 days. ' + billing, [(name, 'NAME')]),
        ('Policy: returns within 30 days. ' + name + ': reviewing card 4111 1111 1111 1111.',
         [(name, 'NAME')]),
        (name + ': reviewing card 4111 1111 1111 1111.', [(name, 'NAME')]),
        ('Customer: ' + name, [(name, 'NAME')]),
        ('Shipping address: ' + address, [(address, 'ADDRESS')]),
        ('Billing ticket: [REDACTED_NAME]\nContext: Shipping address: ' + address,
         [(address, 'ADDRESS')]),
    ]


class LocalNER:
    def __init__(self):
        counts = defaultdict(Counter)
        for row in [('Marta', 'Rossi', 'Cedar', 'Portland'),
                    ('Jonas', 'Meyer', 'Maple', 'Boston'),
                    ('Leila', 'Hassan', 'Birch', 'Denver')]:
            for text, annotations in examples(*row):
                gold = [(text.index(value), text.index(value) + len(value), kind)
                        for value, kind in annotations]
                tokens = list(NER_TOKEN.finditer(text))
                for i, token in enumerate(tokens):
                    label = next((kind for start, end, kind in gold
                                  if start <= token.start() and token.end() <= end), 'O')
                    counts[signature(tokens, i)][label] += 1
        self.labels = {feature: votes.most_common(1)[0][0]
                       for feature, votes in counts.items()}

    def entities(self, text):
        tokens = list(NER_TOKEN.finditer(text))
        pending = None
        for i, token in enumerate(tokens):
            kind = self.labels.get(signature(tokens, i), 'O')
            if token.group().startswith('[REDACTED_'):
                kind = 'O'
            if pending:
                start, end, previous = pending
                gap = text[end:token.start()]
                if kind == previous and not gap.strip() and '\n' not in gap:
                    pending = (start, token.end(), kind)
                    continue
                yield Entity(*pending)
                pending = None
            if kind != 'O':
                pending = (token.start(), token.end(), kind)
        if pending:
            yield Entity(*pending)


NER = LocalNER()


class RedactionUnavailable(ValueError):
    code = 'pii_redaction_unavailable'
    status = 503

    def __init__(self):
        super().__init__('pii_redaction_unavailable')


def luhn(value):
    digits = [int(c) for c in value if c in '0123456789']
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def unprotected(text):
    start = 0
    for match in TOKEN.finditer(text):
        yield start, text[start:match.start()]
        start = match.end()
    yield start, text[start:]


def detect(text, ner=None):
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise RedactionUnavailable()
    ner = NER if ner is None else ner
    spans = []
    patterns = (
        ('EMAIL', EMAIL),
        ('CARD', r'(?<!\w)(?:[0-9][ -]?){12,18}[0-9](?!\w)'),
        ('SSN', r'(?<!\w)[0-9]{3}-[0-9]{2}-[0-9]{4}(?!\w)'),
        ('PHONE', r'(?<!\w)(?:\+1[ -]?)?\(?[0-9]{3}\)?[ -][0-9]{3}[ -][0-9]{4}(?!\w)'),
    )
    try:
        for offset, part in unprotected(text):
            for item in ner.entities(part):
                if not (0 <= item.start < item.end <= len(part)
                        and item.kind in {'NAME', 'ADDRESS'}):
                    raise RedactionUnavailable()
                spans.append(Entity(offset + item.start, offset + item.end, item.kind))
            for kind, pattern in patterns:
                for match in re.finditer(pattern, part, re.I):
                    value = match.group()
                    if kind == 'CARD' and not luhn(value):
                        continue
                    if kind == 'SSN':
                        area, group, serial = value.split('-')
                        if (area in {'000', '666'} or int(area) >= 900
                                or group == '00' or serial == '0000'):
                            continue
                    spans.append(Entity(offset + match.start(), offset + match.end(), kind))
        return spans
    except Exception:
        raise RedactionUnavailable() from None


def secret_key():
    try:
        value = bytes.fromhex(os.environ['ANCHOR_PII_HMAC_KEY'])
        if len(value) < 32:
            raise ValueError()
        return value
    except Exception:
        raise RedactionUnavailable() from None


def rewrite(text, *, key=None, ner=None):
    try:
        groups = []
        for item in sorted(detect(text, ner), key=lambda item: (item.start, -item.end)):
            if groups and item.start < groups[-1][1]:
                start, end, kinds = groups[-1]
                groups[-1] = (start, max(end, item.end), kinds | {item.kind})
            else:
                groups.append((item.start, item.end, {item.kind}))
        result = text
        for start, end, kinds in reversed(groups):
            kind = next(iter(kinds)) if len(kinds) == 1 else 'PII'
            replacement = '[REDACTED_' + kind + ']'
            if kind in {'CARD', 'SSN'}:
                actual_key = secret_key() if key is None else key
                if not isinstance(actual_key, bytes) or len(actual_key) < 32:
                    raise RedactionUnavailable()
                canonical = ''.join(c for c in text[start:end] if c in '0123456789')
                digest = hmac.new(actual_key, (kind + ':' + canonical).encode(), hashlib.sha256).hexdigest()
                replacement = '[REDACTED_' + kind + '_' + digest + ']'
            result = result[:start] + replacement + result[end:]
        return result
    except Exception:
        raise RedactionUnavailable() from None


def configured_redaction(text):
    for rule in CONFIG['pii_patterns']:
        text = re.sub(rule['pattern'], rule['replacement'], text, flags=re.I)
    return text


def redact(text):
    safe = rewrite(text)
    pieces = []
    start = 0
    for match in TOKEN.finditer(safe):
        pieces.extend([configured_redaction(safe[start:match.start()]), match.group()])
        start = match.end()
    return ''.join(pieces) + configured_redaction(safe[start:])


def redact_value(value, depth=0):
    if depth > 20:
        raise RedactionUnavailable()
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        if len(value) > 1000:
            raise RedactionUnavailable()
        return [redact_value(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 1000:
            raise RedactionUnavailable()
        return {redact(str(key)): redact_value(item, depth + 1)
                for key, item in value.items()
                if str(key).lower() not in {'notes', 'private_notes'}}
    if value is None or type(value) in (int, float, bool):
        return value
    raise RedactionUnavailable()


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
    # Preserve existing document-ingestion placeholders and lineage fingerprints.
    text = ' '.join(redact(configured_redaction(text)).split())
    if not text:
        raise ValueError('empty_document')
    return text


def redact_evaluation(value):
    if isinstance(value, str):
        return re.sub(EMAIL, '[REDACTED_EMAIL]', redact(value), flags=re.I)
    if isinstance(value, list):
        return [redact_evaluation(item) for item in value]
    if isinstance(value, dict):
        return {redact_evaluation(key): redact_evaluation(item) for key, item in value.items()}
    return value


FINETUNING_PATTERNS = (
    (EMAIL, '[REDACTED_EMAIL]'),
    (r'\b\d{3}[- ]\d{3}[- ]\d{4}\b', '[REDACTED_PHONE]'),
)


def finetuning_detected(text):
    if detect(text):
        return True
    patterns = [rule['pattern'] for rule in CONFIG['pii_patterns']]
    patterns += [pattern for pattern, _ in FINETUNING_PATTERNS]
    return any(re.search(pattern, part, re.I)
               for _, part in unprotected(text) for pattern in patterns)


def redact_finetuning_text(text):
    text = redact(text)
    pieces = []
    start = 0
    for match in TOKEN.finditer(text):
        part = text[start:match.start()]
        for pattern, replacement in FINETUNING_PATTERNS:
            part = re.sub(pattern, replacement, part, flags=re.I)
        pieces.extend([part, match.group()])
        start = match.end()
    part = text[start:]
    for pattern, replacement in FINETUNING_PATTERNS:
        part = re.sub(pattern, replacement, part, flags=re.I)
    return ''.join(pieces) + part


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
        return re.sub(r'\+?\d[\d ()-]{8,}\d', '[REDACTED_PHONE]', safe)
    if isinstance(value, list):
        return [redact_graph(item) for item in value]
    if isinstance(value, dict):
        return {redact_graph(str(key)): redact_graph(item) for key, item in value.items()}
    if value is None or type(value) in (int, float, bool):
        return value
    raise ValueError('unsupported_graph_value')


def redact_attachment_span(text):
    """Bounded image or voice text, sanitized before evidence persistence or export."""
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError('attachment_span_rejected')
    safe = redact_memory(text)
    if len(safe) > 4000:
        raise ValueError('attachment_span_rejected')
    return safe


AUDIO_EVIDENCE_PREFIX = re.compile(
    r'^\[(?:(?P<start>\d{1,3}\.\d{3})-(?P<end>\d{1,3}\.\d{3})s; '
    r'role=unknown|MISSING (?P<missing_start>\d{1,3}\.\d{3})-'
    r'(?P<missing_end>\d{1,3}\.\d{3})s)\] ')


def redact_evidence_span(text, region):
    """Keep validated audio coordinates separate from untrusted transcript text."""
    if not isinstance(region, str) or not re.fullmatch(r'audio_(?:gap_)?\d+', region):
        return redact_attachment_span(text)
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError('attachment_span_rejected')
    match = AUDIO_EVIDENCE_PREFIX.match(text)
    if match is None:
        raise ValueError('audio_evidence_prefix_rejected')
    missing = match['missing_start'] is not None
    if missing != region.startswith('audio_gap_'):
        raise ValueError('audio_evidence_prefix_rejected')
    start = float(match['missing_start'] if missing else match['start'])
    end = float(match['missing_end'] if missing else match['end'])
    if not 0 <= start < end <= 720:
        raise ValueError('audio_evidence_interval_rejected')
    safe = match[0] + redact_attachment_span(text[match.end():])
    if len(safe) > 4000:
        raise ValueError('attachment_span_rejected')
    return safe
