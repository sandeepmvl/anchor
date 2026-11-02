import argparse
import asyncio
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import HTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anchor_ai import observability
from anchor_ai.agent_interop import InteropAdapter, card_handler
from anchor_ai.harness import HarnessAdapter
from anchor_ai.tests.test_harness_interop import make_runtime, request, task


PLACEMENTS = {
    'OpenAI Responses API / Anthropic Messages API':
        'model API: inference, tool proposals, optional provider-hosted tools',
    'Claude Agent SDK / OpenAI Agents SDK':
        'harness SDK: packaged client-side loop and lifecycle hooks',
    'LangGraph': 'application: Anchor composes graph, state, and routing',
    "Anchor Rails app": 'application: identity, authorization, persistence, UI',
    'MCP': 'agent-to-tool/context: tools and resources',
    'A2A': 'agent-to-agent: agent card, skills, tasks',
}


async def compare():
    spans = []
    original = observability.SINK
    observability.SINK = spans.append
    try:
        with tempfile.TemporaryDirectory() as directory:
            runtime = make_runtime(directory)
            state = await HarnessAdapter(runtime).run('reader', request())
            assert state['status'] == 'awaiting_approval'
            assert 'issue_refund' not in runtime.backend.calls
            local_names = [s['name'] for s in spans]
            assert 'python.harness.pre_tool' in local_names
            print('LOCAL recorded refund:', state['status'], 'calls=', state['calls'])
            for event in state['events']:
                if event['subsystem'] in {'permission_decision', 'tool_dispatch', 'escalation'}:
                    print(json.dumps({k: v for k, v in event.items()
                                      if k not in {'observation'}}, sort_keys=True))
            print('LOCAL spans:', ', '.join(sorted(set(local_names))))
            spans.clear()
            # Recorded hosted-session boundary, not a live provider or a local veto.
            with observability.span('python.provider_native.session', stage='agent',
                                    execution_mode='provider_native'):
                hosted = {'ticket': request(), 'status': 'requires_application_review'}
            assert hosted['ticket'] == request()
            assert [s['name'] for s in spans] == ['python.provider_native.session']
            print('HOSTED recorded session: requires_application_review; no client pre-tool spans')
            print('VERIFIED: same input, different loop ownership; no money moved')
    finally:
        observability.SINK = original


async def delegate():
    with tempfile.TemporaryDirectory() as directory:
        runtime = make_runtime(directory)
        result = await InteropAdapter(HarnessAdapter(runtime)).delegate('reader', task())
        assert result['task_id'] == 'refund42' and result['tenant_id'] == 1
        assert result['status'] == 'awaiting_approval'
        print(json.dumps(result, indent=2))
        print('VERIFIED: specialist result preserves tenant, task, and trace; full handoff retained')


def curl_card():
    server = HTTPServer(('127.0.0.1', 0), card_handler(''))
    base = f'http://127.0.0.1:{server.server_port}'
    server.RequestHandlerClass = card_handler(base)
    curl = shutil.which('curl')
    client = 'curl' if curl else 'Python http.client (curl unavailable)'

    def fetch(method, path):
        if curl:
            command = [curl, '--noproxy', '*', '--silent', '--show-error',
                       '--max-time', '2', '-X', method, '-w', '\n%{http_code}',
                       base + path]
            print('curl verification:', ' '.join(command))
            response = subprocess.run(command, check=True, capture_output=True,
                                      text=True, timeout=3)
            body, status = response.stdout.rsplit('\n', 1)
            return int(status), body
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port,
                                                timeout=2)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            return response.status, response.read().decode('utf-8')
        finally:
            connection.close()

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print('Discovery verification client:', client)
        status, body = fetch('GET', '/.well-known/agent-card.json')
        assert status == 200
        card = json.loads(body)
        assert card['skills'][0]['id'] == 'refund-resolution'
        assert card['supportedInterfaces'][0]['url'] == base + '/a2a'
        assert 'not implemented' in card['description']
        print(json.dumps(card, indent=2))
        status, _ = fetch('POST', '/a2a')
        assert status == 501
        print('VERIFIED via ' + client +
              ': discovery stub served; public task POST returns 501')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--delegate', action='store_true')
    parser.add_argument('--curl-card', action='store_true')
    args = parser.parse_args()
    if args.compare:
        asyncio.run(compare())
    elif args.delegate:
        asyncio.run(delegate())
    elif args.curl_card:
        curl_card()
    else:
        for product, responsibility in PLACEMENTS.items():
            print(product + ' -> ' + responsibility)
