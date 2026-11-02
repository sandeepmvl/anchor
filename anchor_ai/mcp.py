"""Constrained stateless Streamable HTTP tools client for Anchor."""
import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid

import httpx
from jsonschema import Draft202012Validator
import yaml

from .observability import mcp_span, span

ROOT = Path(__file__).resolve().parents[1]
VERSION = '2025-06-18'
LOCAL_ENDPOINT = 'http://127.0.0.1:4000/mcp'


class McpRejected(ValueError):
    code = 'mcp_rejected'


@dataclass(frozen=True)
class Credential:
    tenant_id: int
    token: str


class McpClient:
    def __init__(self, server_id, *, config=None, credential_resolver=None):
        self.config = config if config is not None else yaml.safe_load(
            (ROOT / 'config/mcp.yml').read_text())
        servers = self.config.get('servers', {})
        if server_id not in servers:
            raise McpRejected('unapproved_server')
        self.server_id = server_id
        self.policy = servers[server_id]
        self.endpoint = self.policy['endpoint']
        parsed = urlsplit(self.endpoint)
        local = (self.policy.get('development_loopback') is True
                 and self.endpoint == LOCAL_ENDPOINT)
        if (parsed.username or parsed.password or parsed.query or parsed.fragment
                or not parsed.hostname or (parsed.scheme != 'https' and not local)):
            raise McpRejected('unapproved_endpoint')
        if self.policy.get('protocol_version') != VERSION:
            raise McpRejected('unsupported_protocol')
        self.resolve = credential_resolver or self._environment_credential

    def _environment_credential(self, tenant_id, reference):
        return Credential(tenant_id, os.environ.get(reference, ''))

    def credential(self, scope):
        tenant = self.policy.get('tenants', {}).get(scope.tenant_id)
        if tenant is None:
            raise McpRejected('tenant_not_configured')
        credential = self.resolve(scope.tenant_id, tenant['credential_env'])
        if (not isinstance(credential, Credential)
                or credential.tenant_id != scope.tenant_id or not credential.token):
            raise McpRejected('credential_scope_mismatch')
        return credential

    def wrap(self, registry):
        result = dict(registry)
        for local_name, remote_name in self.policy.get('tools', {}).items():
            if local_name not in registry:
                raise McpRejected('unapproved_tool_mapping')
            tool = registry[local_name]

            async def invoke(*, scope, call_id, _tool=tool,
                             _remote=remote_name, idempotency_key=None, **arguments):
                return await self.invoke(scope, call_id, _remote, _tool,
                                         arguments, idempotency_key)

            result[local_name] = replace(tool, run=invoke,
                                         external_server=self.server_id)
        return result

    async def invoke(self, scope, call_id, remote_name, tool, arguments,
                     idempotency_key=None):
        # No URLs, credentials, descriptions, arguments, or result text in spans.
        with mcp_span(scope, self.server_id, call_id) as operation:
            credential = self.credential(scope)
            headers = {
                'Authorization': 'Bearer ' + credential.token,
                'Accept': 'application/json, text/event-stream',
                'MCP-Protocol-Version': VERSION,
                'traceparent': '00-' + operation.record['trace_id'] + '-'
                               + operation.record['span_id'] + '-01',
                'X-Anchor-Tool-Call': hashlib.sha256(call_id.encode()).hexdigest(),
            }
            limit = int(self.policy.get('max_response_bytes', 65536))
            sequence = 0
            async with httpx.AsyncClient(
                timeout=float(self.policy.get('timeout_seconds', 2)),
                follow_redirects=False, trust_env=False,
            ) as http:
                async def exchange(method, params=None, notification=False):
                    nonlocal sequence
                    sequence += 1
                    packet = {'jsonrpc': '2.0', 'method': method}
                    if not notification:
                        packet['id'] = sequence
                    if params is not None:
                        packet['params'] = params
                    async with http.stream('POST', self.endpoint, headers=headers,
                                           json=packet) as response:
                        if notification:
                            if response.status_code != 202:
                                raise McpRejected('notification_rejected')
                            return None
                        if response.status_code != 200:
                            raise McpRejected('transport_rejected')
                        if response.headers.get('mcp-session-id'):
                            raise McpRejected('sessions_not_supported')
                        if response.headers.get('content-type', '').split(';')[0] != 'application/json':
                            raise McpRejected('unsupported_response_transport')
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data) > limit:
                                raise McpRejected('response_too_large')
                    reply = json.loads(data)
                    if (reply.get('jsonrpc') != '2.0'
                            or reply.get('id') != sequence or 'error' in reply
                            or 'result' not in reply):
                        raise McpRejected('invalid_protocol_response')
                    return reply['result']

                initialized = await exchange('initialize', {
                    'protocolVersion': VERSION, 'capabilities': {},
                    'clientInfo': {'name': 'anchor', 'version': '1.0'},
                })
                if (initialized.get('protocolVersion') != VERSION
                        or 'tools' not in initialized.get('capabilities', {})):
                    raise McpRejected('unsupported_capabilities')
                await exchange('notifications/initialized', notification=True)
                # This constrained client rejects pagination rather than truncating it.
                discovered = await exchange('tools/list', {})
                if discovered.get('nextCursor'):
                    raise McpRejected('pagination_not_supported')
                matches = [item for item in discovered.get('tools', [])
                           if item.get('name') == remote_name]
                expected = tool.arguments.model_json_schema()
                if len(matches) != 1 or matches[0].get('inputSchema') != expected:
                    raise McpRejected('schema_drift')
                Draft202012Validator.check_schema(expected)
                Draft202012Validator(expected).validate(arguments)
                params = {'name': remote_name, 'arguments': arguments}
                if tool.consequential:
                    if not idempotency_key:
                        raise McpRejected('idempotency_required')
                    params['_meta'] = {'anchor/idempotency_key': idempotency_key}
                result = await exchange('tools/call', params)
                if result.get('isError'):
                    raise McpRejected('tool_execution_failed')
                value = result.get('structuredContent')
                if value is None:
                    content = result.get('content', [])
                    if len(content) != 1 or content[0].get('type') != 'text':
                        raise McpRejected('unsupported_tool_content')
                    value = json.loads(content[0]['text'])
                # The guarded dispatcher validates and redacts this model again.
                return tool.result.model_validate(value).model_dump()


class OfflineOrderClient:
    """Deterministic proposal and answer; no provider or network model call."""
    def complete(self, prompt, system=None, max_tokens=256):
        observations = [m for m in prompt if m.get('role') == 'tool']
        if not observations:
            return {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'mcp_lookup_1', 'type': 'function', 'function': {
                    'name': 'lookup_order', 'arguments': '{"order_id":"A100"}'}}]}
        payload = json.loads(observations[-1]['content'])
        text = ('Order A100 is ' + payload['data']['status'] + '.') if payload['ok'] else (
            'I cannot obtain the order status.')
        return {'role': 'assistant', 'content': text, 'tool_calls': []}


@contextmanager
def local_fixture():
    """Own a real loopback server, including credentials and guaranteed cleanup."""
    names = ['ANCHOR_MCP_TENANT_1_TOKEN', 'ANCHOR_MCP_TENANT_2_TOKEN']
    saved = {name: os.environ.get(name) for name in names}
    process = None
    try:
        for name in names:
            os.environ[name] = secrets.token_urlsafe(24)
        env = dict(os.environ)
        env['ANCHOR_MCP_FIXTURE_TOKENS'] = json.dumps({
            os.environ[names[0]]: 1, os.environ[names[1]]: 2})
        process = subprocess.Popen(
            [sys.executable, str(ROOT / 'labs/mcp-interop/sample_server.py'), '--serve'],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('fixture_failed_to_start; port 4000 must be free')
            try:
                with httpx.Client(timeout=0.15, trust_env=False) as http:
                    response = http.post(LOCAL_ENDPOINT, headers={
                        'Authorization': 'Bearer ' + os.environ[names[0]],
                        'Accept': 'application/json, text/event-stream',
                    }, json={'jsonrpc': '2.0', 'id': 'ready', 'method': 'initialize',
                             'params': {'protocolVersion': VERSION}})
                if response.status_code == 200:
                    # Verify this is our child, not a service already using the port.
                    if response.json()['result']['serverInfo']['version'] == env[names[0]]:
                        break
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            time.sleep(0.025)
        else:
            raise RuntimeError('fixture_start_deadline')
        yield
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


async def order_demo(tenant_id=1):
    from .contracts import ToolScope
    from .tools import MemoryOrderBackend, registry_for, run_conversation
    registry = registry_for(MemoryOrderBackend(), mcp_client=McpClient('local_orders'))
    scope = ToolScope(tenant_id=tenant_id, conversation_id=7)
    with span('python.mcp_demo', tenant_id=tenant_id, conversation_id=7):
        history, batches = await run_conversation(
            OfflineOrderClient(), registry, scope, message='Where is order A100?')
    if not batches or not batches[0][0]['payload']['ok']:
        raise RuntimeError('external_lookup_failed')
    return history[-1]['content'], batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='mcp://localhost:4000')
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    if args.server not in {'mcp://localhost:4000', LOCAL_ENDPOINT}:
        raise McpRejected('unapproved_server')
    from . import observability
    records = []
    original = observability.SINK
    observability.SINK = records.append
    try:
        with local_fixture():
            answer, _ = asyncio.run(order_demo())
        print('Anchor (external MCP): ' + answer)
        if args.trace:
            for record in records:
                print(json.dumps(record, sort_keys=True))
    finally:
        observability.SINK = original


if __name__ == '__main__':
    main()
