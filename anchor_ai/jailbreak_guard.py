"""Narrow, offline-capable pre-execution policy; not a general attack classifier."""
import re
from dataclasses import dataclass

MAX_BYTES = 32768
MAX_TURNS = 80
MAX_CORE_BYTES = 1024
REFUSAL = 'I cannot disclose private recovery codes. Use the approved recovery process or contact human support.'
ROLEPLAY = re.compile(r'pretend you are dan|unrestricted assistant|safety rules do not apply', re.I)
ENCODED = re.compile(r'decode\b.{0,80}\b(base64|hex)\b.{0,80}\bobey|decode\b.{0,80}\bobey', re.I | re.S)
SECRET = re.compile(r'(reveal|disclose|show|give)\b.{0,60}\b(private recovery codes|recovery secrets)', re.I)
EXAMPLE = re.compile(r'assistant:\s*here are the private recovery codes', re.I)


@dataclass(frozen=True)
class Decision:
    refused: bool
    flagged: bool
    reason: str
    checks: int = 0
    disagreement: bool = False

    @property
    def reply(self):
        return REFUSAL if self.refused else None


class GuardError(ValueError):
    pass


def bounded_text(value):
    if not isinstance(value, str) or not value.strip():
        raise GuardError('invalid_text')
    try:
        if len(value.encode('utf-8')) > MAX_BYTES:
            raise GuardError('request_budget')
    except UnicodeError:
        raise GuardError('invalid_unicode') from None
    return value


def envelope(text, history=()):
    bounded_text(text)
    if not isinstance(history, (list, tuple)) or len(history) > MAX_TURNS:
        raise GuardError('history_budget')
    parts = []
    total = len(text.encode('utf-8'))
    for row in history:
        content = row.get('content') if isinstance(row, dict) else getattr(row, 'content', None)
        bounded_text(content)
        total += len(content.encode('utf-8'))
        if total > MAX_BYTES:
            raise GuardError('request_budget')
        # Customer-supplied roles never become actual message roles.
        parts.append(content)
    return '\n'.join(parts + [text])


def canonicalize(text):
    """Only the supported envelope is extractable; ambiguity is a refusal."""
    bounded_text(text)
    matches = list(re.finditer(r'(?m)^Request:[ \t]*(.*)$', text))
    if len(matches) != 1:
        raise GuardError('ambiguous_request')
    match = matches[0]
    if text[match.end():].strip():
        raise GuardError('ambiguous_request')
    core = re.sub(r'\s+', ' ', match.group(1)).strip()
    if not core or len(core.encode('utf-8')) > MAX_CORE_BYTES:
        raise GuardError('core_budget')
    if ROLEPLAY.search(core) or ENCODED.search(core) or EXAMPLE.search(core):
        raise GuardError('nested_wrapper')
    return core


def policy_check(text):
    """Fixed local policy, independent of all supplied role claims."""
    return {'refused': bool(SECRET.search(text) or ENCODED.search(text))}


def refusal_value(result):
    if not isinstance(result, dict) or set(result) != {'refused'}:
        raise GuardError('invalid_decision')
    if type(result['refused']) is not bool:
        raise GuardError('invalid_decision')
    return result['refused']


def evaluate(text, history=(), *, checker=policy_check):
    checks = 0
    flagged = False
    try:
        combined = envelope(text, history)
        encoded = bool(ENCODED.search(combined))
        flagged = bool(ROLEPLAY.search(combined) or EXAMPLE.search(combined))
        if encoded:
            return Decision(True, True, 'encoded_request')
        if not flagged:
            checks += 1
            refused = refusal_value(checker(combined))
            return Decision(refused, False, 'policy_refusal' if refused else 'pass', checks)
        # Extract from the current request, never from a historical summary.
        core = canonicalize(text)
        checks += 1
        contextual = refusal_value(checker(combined))
        checks += 1
        fresh = refusal_value(checker(core))
        return Decision(contextual or fresh, True, 'consistency_check', checks,
                        contextual != fresh)
    except Exception:
        # No exception message or customer text escapes into ordinary telemetry.
        return Decision(True, flagged, 'verification_error', checks)


def require_allowed(text, history=()):
    decision = evaluate(text, history)
    if decision.refused:
        raise GuardError(decision.reason)
    return text
