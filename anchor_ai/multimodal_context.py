import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass


class VisionContextRejected(ValueError):
    pass


@dataclass(frozen=True)
class VisionIdentity:
    tenant_id: int
    conversation_id: int


def sign_test_context(secret, tenant_id=1, conversation_id=42, *, now=None):
    """Local test helper. Production callers sign only after authorizing a ticket."""
    now = int(time.time()) if now is None else now
    payload = json.dumps(dict(aud='anchor-vision-v1', tenant_id=tenant_id,
                              conversation_id=conversation_id, exp=now + 60),
                         sort_keys=True, separators=(',', ':')).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip('=')
    signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return encoded + '.' + signature


def verify_context(token, secret, *, now=None):
    now = int(time.time()) if now is None else now
    try:
        if not secret or not isinstance(token, str) or len(token) > 1024:
            raise ValueError()
        encoded, signature = token.split('.')
        expected = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError()
        raw = base64.b64decode(encoded + '=' * (-len(encoded) % 4),
                               altchars=b'-_', validate=True)
        data = json.loads(raw)
        if set(data) != {'aud', 'tenant_id', 'conversation_id', 'exp'}:
            raise ValueError()
        if data['aud'] != 'anchor-vision-v1':
            raise ValueError()
        if any(type(data[k]) is not int or data[k] <= 0
               for k in ('tenant_id', 'conversation_id', 'exp')):
            raise ValueError()
        if not now < data['exp'] <= now + 120:
            raise ValueError()
        return VisionIdentity(data['tenant_id'], data['conversation_id'])
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise VisionContextRejected('vision_context_rejected') from None
