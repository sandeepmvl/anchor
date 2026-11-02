"""Authenticated request scope; importing this module does not authenticate anyone."""
import hmac
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar

_current = ContextVar('anchor_authenticated_tenant', default=None)


class ScopeError(PermissionError):
    pass


def enforcement_enabled():
    return (os.getenv('ANCHOR_ENV') == 'production'
            or 'ANCHOR_TENANT_CREDENTIALS' in os.environ
            or os.getenv('ANCHOR_LLM_PROVIDER', 'offline') not in {'offline', 'fake'})


def has_scope():
    return _current.get() is not None


def check_bound_tenant(requested):
    """Internal adapters retain offline contracts; deployed access requires scope."""
    if has_scope() or enforcement_enabled():
        return require_tenant(requested)
    return requested


def positive_tenant(value):
    if type(value) is not int or value <= 0:
        raise ScopeError('invalid_tenant_scope')
    return value


def require_tenant(requested=None):
    tenant = _current.get()
    if tenant is None:
        raise ScopeError('authenticated_tenant_required')
    if requested is not None and str(requested) != str(tenant):
        raise ScopeError('tenant_scope_conflict')
    return tenant


def verify_credential(credential, credentials=None):
    if credentials is None:
        try:
            credentials = json.loads(os.environ['ANCHOR_TENANT_CREDENTIALS'])
        except (KeyError, ValueError):
            raise ScopeError('tenant_credentials_unavailable') from None
    if not isinstance(credentials, dict) or not credentials:
        raise ScopeError('tenant_credentials_unavailable')
    if not isinstance(credential, str) or not credential:
        raise ScopeError('tenant_credential_required')
    matched = None
    for secret, tenant in credentials.items():
        if not isinstance(secret, str) or len(secret) < 32:
            raise ScopeError('invalid_tenant_credentials')
        positive_tenant(tenant)
        if hmac.compare_digest(secret.encode(), credential.encode()):
            matched = tenant
    if matched is None:
        raise ScopeError('tenant_credential_rejected')
    return matched


@contextmanager
def authenticated_scope(credential, credentials=None):
    tenant = verify_credential(credential, credentials)
    token = _current.set(tenant)
    try:
        yield tenant
    finally:
        _current.reset(token)


class TenantScopeMiddleware:
    """ASGI middleware: identity is established before endpoint execution.

    Install around an application that accepts tenant data. Health checks and
    global, stateless redaction do not need tenant-owned storage.
    """
    def __init__(self, app, credentials=None):
        self.app = app
        self.credentials = credentials

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') in {'/healthz', '/v1/pii/redact'}:
            return await self.app(scope, receive, send)
        if self.credentials is None and not enforcement_enabled():
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', ()))
        credential = headers.get(b'x-anchor-tenant-credential', b'').decode('utf-8', 'replace')
        try:
            tenant = verify_credential(credential, self.credentials)
        except ScopeError:
            return await self.reject(send)
        messages = []
        data = bytearray()
        while True:
            message = await receive()
            messages.append(message)
            if message['type'] == 'http.disconnect':
                return
            if message['type'] != 'http.request':
                return await self.reject(send)
            data.extend(message.get('body', b''))
            if len(data) > 1048576:
                return await self.reject(send, 413)
            if not message.get('more_body', False):
                break
        if data:
            try:
                body = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                return await self.reject(send, 422)
            if isinstance(body, dict) and 'tenant_id' in body:
                if type(body['tenant_id']) is not int or body['tenant_id'] != tenant:
                    return await self.reject(send)
        position = 0

        async def replay():
            nonlocal position
            if position < len(messages):
                message = messages[position]
                position += 1
                return message
            return await receive()

        token = _current.set(tenant)
        scope.setdefault('state', {})['authenticated_tenant_id'] = tenant
        try:
            await self.app(scope, replay, send)
        except ScopeError:
            # Storage errors must be handled before response headers are sent.
            # Streaming endpoints must perform authorization before streaming.
            raise
        finally:
            _current.reset(token)

    @staticmethod
    async def reject(send, status=403):
        body = b'{"detail":"tenant_scope_rejected"}'
        await send({'type': 'http.response.start', 'status': status,
                    'headers': [(b'content-type', b'application/json')]})
        await send({'type': 'http.response.body', 'body': body})


def scoped_key(kind, *parts):
    """Unambiguous identity for tenant-owned cache and checkpoint objects."""
    return json.dumps([require_tenant(), kind, *parts], separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)
