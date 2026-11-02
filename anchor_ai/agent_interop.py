import json
from http.server import BaseHTTPRequestHandler

from anchor_ai.contracts import DelegatedTask, DelegatedResult
from anchor_ai.observability import span
from anchor_ai.orchestration import run_harness_support_turn
from anchor_ai.runtime import ContextError


class InteropAdapter:
    """Authenticated internal delegation, not a complete public A2A server."""
    def __init__(self, harness):
        self.harness = harness

    async def delegate(self, token, envelope):
        scope = self.harness.runtime.identity(token)
        if not isinstance(envelope, dict):
            raise ValueError('delegated_task_required')
        if 'content' in envelope or 'isError' in envelope:
            raise ValueError('mcp_result_is_not_a2a_task')
        task = DelegatedTask.model_validate(envelope)
        if (task.tenant_id != scope.tenant_id
                or task.conversation_id != scope.conversation_id):
            raise ContextError('delegation_scope_mismatch')
        # A delegation identifier is correlation, never proof of permission.
        with span('python.interop.delegate', stage='agent', task_id=task.task_id,
                  tenant_id=scope.tenant_id, conversation_id=scope.conversation_id,
                  interop_protocol='anchor-a2a-loopback-v1'):
            state = await run_harness_support_turn(
                self.harness, token, task.request.model_dump(), enabled=True)
        return DelegatedResult(
            task_id=task.task_id, tenant_id=state['tenant_id'],
            conversation_id=state['conversation_id'], trace_id=state['trace_id'],
            status=state['status'], handoff=state.get('escalation')).model_dump()


def agent_card(base):
    """Discovery stub only; the advertised public task interface is unimplemented."""
    return {
        'name': 'Anchor Refund Resolution',
        'description': 'Discovery stub. Public A2A task execution is not implemented.',
        'version': '0.1.0',
        'supportedInterfaces': [{
            'url': base + '/a2a', 'protocolBinding': 'JSONRPC',
            'protocolVersion': '1.0'}],
        'url': base + '/a2a', 'protocolVersion': '1.0',
        'capabilities': {},
        'defaultInputModes': ['text/plain'],
        'defaultOutputModes': ['text/plain'],
        'skills': [{
            'id': 'refund-resolution', 'name': 'Refund resolution',
            'description': 'Assess an Anchor refund request; public execution unavailable.',
            'tags': ['support', 'refunds']}],
    }


def card_handler(base):
    # Mounted only by the bounded loopback verification session in this lesson.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != '/.well-known/agent-card.json':
                self.send_error(404)
                return
            payload = json.dumps(agent_card(base)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            self.send_error(501, 'Public A2A task handling is not implemented')

    return Handler
