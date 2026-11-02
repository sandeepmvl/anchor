"""Local, stateless MCP tools fixture. Never deploy this synthetic backend."""
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn
from anchor_ai.contracts import ToolScope
from anchor_ai.mcp import VERSION, local_fixture
from anchor_ai.tools import MemoryOrderBackend, definitions, registry_for

app = FastAPI()
backend = MemoryOrderBackend()
registry = registry_for(backend)
TOKENS = json.loads(os.environ.get('ANCHOR_MCP_FIXTURE_TOKENS', '{}'))


@app.post('/mcp')
async def mcp(request: Request):
    # Reject browser origins; this fixture has no browser-facing use case.
    if request.headers.get('origin'):
        return Response(status_code=403)
    if request.headers.get('host') not in {'127.0.0.1:4000', 'localhost:4000'}:
        return Response(status_code=403)
    token = request.headers.get('authorization', '').removeprefix('Bearer ')
    tenant_id = TOKENS.get(token)
    if tenant_id is None:
        return Response(status_code=401)
    if int(request.headers.get('content-length', '0')) > 65536:
        return Response(status_code=413)
    body = await request.json()
    if not isinstance(body, dict) or body.get('jsonrpc') != '2.0':
        return Response(status_code=400)
    method = body.get('method')
    params = body.get('params', {})
    if method == 'notifications/initialized' and 'id' not in body:
        return Response(status_code=202)
    if 'id' not in body:
        return Response(status_code=202)
    packet = {'jsonrpc': '2.0', 'id': body['id']}
    if method != 'initialize' and request.headers.get('mcp-protocol-version') != VERSION:
        return Response(status_code=400)
    if method == 'initialize':
        if params.get('protocolVersion') != VERSION:
            packet['error'] = {'code': -32602, 'message': 'Unsupported version'}
        else:
            packet['result'] = {
                'protocolVersion': VERSION, 'capabilities': {'tools': {}},
                'serverInfo': {'name': 'anchor-local-orders',
                               'version': os.environ.get('ANCHOR_MCP_TENANT_1_TOKEN', 'fixture')},
            }
    elif method == 'tools/list':
        approved = {name: registry[name] for name in ('lookup_order', 'issue_refund')}
        packet['result'] = {'tools': definitions(approved, api='mcp')}
    elif method == 'tools/call':
        try:
            name = params['name']
            if name != 'lookup_order':
                raise ValueError('fixture_writes_disabled')
            arguments = registry[name].arguments.model_validate(params['arguments'])
            scope = ToolScope(tenant_id=tenant_id, conversation_id=7)
            if not backend.owns(scope, arguments.model_dump()):
                raise ValueError('order_not_owned')
            value = await backend.lookup_order(scope=scope, **arguments.model_dump())
            packet['result'] = {
                'content': [{'type': 'text', 'text': json.dumps(value)}],
                'structuredContent': value, 'isError': False,
            }
        except Exception:
            packet['result'] = {'content': [{'type': 'text', 'text': 'Lookup failed'}],
                                'isError': True}
    else:
        packet['error'] = {'code': -32601, 'message': 'Method not supported'}
    return JSONResponse(packet)


if __name__ == '__main__':
    if '--serve' in sys.argv:
        if not TOKENS:
            raise SystemExit('fixture credentials required')
        uvicorn.run(app, host='127.0.0.1', port=4000, log_level='error',
                    access_log=False)
    else:
        with local_fixture():
            print('MCP endpoint listening on 127.0.0.1:4000/mcp (loopback only)')
        print('Fixture stopped; port 4000 released.')
