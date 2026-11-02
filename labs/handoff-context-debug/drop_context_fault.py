"""Test-only loopback fault injection. Never mount in anchor_ai.app."""
import argparse
import copy
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.contracts import RefundArguments, ToolCall
from anchor_ai.orchestration import dispatch_specialist


def trusted_fixture():
    # Stands in for authenticated order lookup output, not model extraction.
    return {'tenant_id': 1, 'conversation_id': 17, 'order_id': 'A104',
            'amount_cents': 300, 'provenance': 'verified-order-read:run17:v1',
            'messages': ['Review my order.'], 'evidence': ['Verified tenant order lookup.']}


def replay(request, mode):
    trusted = trusted_fixture()
    trace = []
    def record(event, state, **metadata):
        trace.append({'trace_id': 'handoff-run17', 'event': event,
                      'keys': sorted(state),
                      'order_id_present': 'order_id' in state, **metadata})
    record('memory.read', trusted, read_status='hit')
    record('triage.exit', trusted)
    mapped = copy.deepcopy(trusted)
    if mode == 'drop':
        mapped['order_number'] = mapped.pop('order_id')
    elif mode != 'fixed':
        raise ValueError('invalid_fault_mode')
    if request.get('incomplete'):
        mapped.pop('order_id', None)
    record('handoff.output', mapped)
    # This is attempted entry, not permission to invoke the specialist.
    record('specialist.entry', mapped, admitted=False)
    calls = []
    def specialist(context):
        record('specialist.admitted', context, admitted=True)
        arguments = RefundArguments(order_id=context['order_id'],
                                    amount_cents=context['amount_cents'])
        call = ToolCall(call_id='refund_run17', name='issue_refund',
                        arguments=arguments.model_dump())
        if request.get('dry_run') is not True or request.get('confirmed') is not True:
            return {'status': 'approval_required', 'tool_calls': []}
        # Offline consequential-tool double. Production dispatch and its
        # authorization, eligibility, approval and idempotency remain unchanged.
        calls.append({'name': call.name, 'arguments': call.arguments,
                      'dry_run': True, 'backend': 'offline_test_double'})
        return {'status': 'dry_run', 'specialist_state': context, 'tool_calls': calls}
    result = dispatch_specialist(mapped, trusted, specialist,
                                 boundary=request.get('boundary', 'dispatch'))
    return {**result, 'trace': trace, 'tool_executions': len(calls),
            'not_verified': ['live refund execution', 'order ownership infrastructure',
                             'sandbox isolation', 'production release readiness']}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.respond({'context_dropping_enabled': self.server.mode == 'drop',
                      'test_only': True})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 4096:
            self.send_error(400)
            return
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError('object_required')
            result = replay(request, self.server.mode)
        except (ValueError, TypeError):
            self.send_error(400)
            return
        self.respond(result)

    def respond(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def server(mode):
    instance = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    instance.daemon_threads = True
    instance.mode = mode
    return instance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['drop', 'fixed'], default='drop')
    parser.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    instance = server(args.mode)
    endpoint = 'http://127.0.0.1:' + str(instance.server_port)
    if args.serve:
        print(json.dumps({'endpoint': endpoint, 'context_dropping_enabled': args.mode == 'drop'}), flush=True)
        try:
            instance.serve_forever()
        finally:
            instance.server_close()
        return
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(endpoint, timeout=2) as response:
            print(response.read().decode())
        request = Request(endpoint, data=json.dumps({'dry_run': True, 'confirmed': True}).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=2) as response:
            result = json.load(response)
        print(json.dumps(result, indent=2))
        if args.mode == 'drop':
            assert result['status'] == 'human_escalation' and result['tool_executions'] == 0
            print('Expected regression: order_id at triage exit but absent at specialist entry; human handoff, zero tools.')
        else:
            assert result['specialist_state']['order_id'] == 'A104'
            assert result['tool_calls'][0]['name'] == 'issue_refund'
            print('Verified order A104 reaches the specialist and guarded refund dry-run; incomplete state goes to a human.')
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=2)


if __name__ == '__main__':
    main()
