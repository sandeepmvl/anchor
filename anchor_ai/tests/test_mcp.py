import asyncio
from copy import deepcopy
import hashlib
import json
import os

import httpx
import pytest

from anchor_ai import observability
from anchor_ai.contracts import ToolCall, ToolScope
from anchor_ai.mcp import (
    Credential, LOCAL_ENDPOINT, McpClient, McpRejected, VERSION,
    local_fixture, order_demo,
)
from anchor_ai.tools import MemoryOrderBackend, definitions, dispatch, registry_for


@pytest.fixture(scope='module')
def server():
    with local_fixture():
        yield


def test_external_lookup_through_bounded_agent_and_redacted_trace(server, monkeypatch):
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    answer, batches = asyncio.run(order_demo(1))
    assert answer == 'Order A100 is delivered.'
    assert batches[0][0]['payload']['data']['customer_contact'] == '[REDACTED]'
    mcp = next(r for r in records if r['name'] == 'python.mcp')
    tool = next(r for r in records if r['name'] == 'python.tool')
    assert mcp['status'] == 'ok'
    assert mcp['mcp_server_id'] == 'local_orders'
    assert mcp['parent_span_id'] == tool['span_id']
    assert mcp['trace_id'] == tool['trace_id']
    expected = hashlib.sha256(b'mcp_lookup_1').hexdigest()
    assert mcp['tool_call_fingerprint'] == tool['tool_call_fingerprint'] == expected
    rendered = json.dumps(records)
    for forbidden in ['alex@example.test', 'Bearer ', os.environ['ANCHOR_MCP_TENANT_1_TOKEN'],
                      os.environ['ANCHOR_MCP_TENANT_2_TOKEN'], 'customer_contact']:
        assert forbidden not in rendered


def test_allowlist_rejects_unknown_server_and_nonapproved_endpoint():
    with pytest.raises(McpRejected, match='unapproved_server'):
        McpClient('model-supplied-host')
    policy = deepcopy(McpClient('local_orders').config)
    policy['servers']['local_orders']['endpoint'] = 'http://169.254.169.254/mcp'
    with pytest.raises(McpRejected, match='unapproved_endpoint'):
        McpClient('local_orders', config=policy)


def test_tenants_use_distinct_credentials_and_order_rows(server):
    assert os.environ['ANCHOR_MCP_TENANT_1_TOKEN'] != os.environ['ANCHOR_MCP_TENANT_2_TOKEN']
    answer, batches = asyncio.run(order_demo(2))
    assert answer == 'Order A100 is processing.'
    assert batches[0][0]['payload']['data']['payment_id'] == 'pay_A200'


def test_cross_tenant_credential_binding_rejected_before_network(monkeypatch):
    client = McpClient('local_orders', credential_resolver=lambda tenant, reference:
                       Credential(2, 'wrong-tenant-secret'))
    async def network_tripwire(*args, **kwargs):
        pytest.fail('cross-tenant credential reached HTTP')
    monkeypatch.setattr(httpx.AsyncClient, 'send', network_tripwire)
    scope = ToolScope(tenant_id=1, conversation_id=7)
    with pytest.raises(McpRejected, match='credential_scope_mismatch'):
        client.credential(scope)
    registry = registry_for(MemoryOrderBackend(), mcp_client=client)
    call = ToolCall(call_id='wrong_binding', name='lookup_order',
                    arguments={'order_id': 'A100'})
    result = asyncio.run(dispatch(call, registry, scope))
    assert result['payload']['ok'] is False


def test_unconfirmed_write_never_reaches_mcp(monkeypatch):
    client = McpClient('local_orders')
    entered = []
    async def forbidden_invocation(*args, **kwargs):
        entered.append(True)
        raise AssertionError('unconfirmed write reached MCP')
    monkeypatch.setattr(client, 'invoke', forbidden_invocation)
    backend = MemoryOrderBackend()
    registry = registry_for(backend, mcp_client=client)
    original = registry_for(backend)['issue_refund']
    assert registry['issue_refund'].consequential is True
    assert registry['issue_refund'].authorize == original.authorize
    scope = ToolScope(tenant_id=1, conversation_id=7)
    call = ToolCall(call_id='unconfirmed_write', name='issue_refund',
                    arguments={'order_id': 'A100', 'amount_cents': 100})
    result = asyncio.run(dispatch(call, registry, scope))
    assert result['payload']['ok'] is False
    assert entered == []
    assert backend.refunds == []


def test_schema_adapter_preserves_full_existing_argument_schema():
    registry = registry_for(MemoryOrderBackend())
    adapted = {tool['name']: tool for tool in definitions(registry, api='mcp')}
    assert adapted['lookup_order']['inputSchema'] == (
        registry['lookup_order'].arguments.model_json_schema())
    assert 'parameters' not in adapted['lookup_order']


def test_schema_drift_rejected_before_tools_call(server, monkeypatch):
    send = httpx.AsyncClient.send
    methods = []
    async def changed_schema(self, request, *args, **kwargs):
        method = json.loads(request.content)['method']
        methods.append(method)
        response = await send(self, request, *args, **kwargs)
        if method == 'tools/list':
            await response.aread()
            packet = response.json()
            packet['result']['tools'][0]['inputSchema'] = {'type': 'object'}
            return httpx.Response(200, json=packet, request=request)
        return response
    monkeypatch.setattr(httpx.AsyncClient, 'send', changed_schema)
    scope = ToolScope(tenant_id=1, conversation_id=7)
    client = McpClient('local_orders')
    tool = registry_for(MemoryOrderBackend())['lookup_order']
    with pytest.raises(McpRejected, match='schema_drift'):
        asyncio.run(client.invoke(scope, 'drift', 'lookup_order', tool, {'order_id': 'A100'}))
    assert 'tools/list' in methods
    assert 'tools/call' not in methods
