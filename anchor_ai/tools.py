import asyncio
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError
from .contracts import ToolCall, ToolScope
from .schemas import OrderToolArguments, OrderToolResult, RefundToolResult
from .observability import span, tool_span
from .llm_client import LlmClient


@dataclass(frozen=True)
class Tool:
    description: str
    arguments: type[BaseModel]
    result: type[BaseModel]
    run: Callable


def normalize(api, response):
    if api == 'chat':
        return [ToolCall(call_id=c['id'], name=c['function']['name'],
                         arguments=c['function']['arguments'])
                for c in response.get('tool_calls', [])]
    if api == 'responses':
        return [ToolCall(call_id=c['call_id'], name=c['name'], arguments=c['arguments'])
                for c in response['output'] if c['type'] == 'function_call']
    if api == 'anthropic':
        return [ToolCall(call_id=c['id'], name=c['name'], arguments=c['input'])
                for c in response['content'] if c['type'] == 'tool_use']
    raise ValueError('unsupported_api')


def result_messages(api, results):
    if api == 'chat':
        return [{'role': 'tool', 'tool_call_id': r['call_id'],
                 'content': json.dumps(r['payload'])} for r in results]
    if api == 'responses':
        return [{'type': 'function_call_output', 'call_id': r['call_id'],
                 'output': json.dumps(r['payload'])} for r in results]
    if api == 'anthropic':
        return [{'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': r['call_id'],
             'is_error': not r['payload']['ok'], 'content': json.dumps(r['payload'])}
            for r in results]}]
    raise ValueError('unsupported_api')


def definitions(registry, api='chat'):
    result = []
    for name, tool in registry.items():
        fields = {'name': name, 'description': tool.description,
                  'parameters': tool.arguments.model_json_schema()}
        if api == 'chat':
            result.append({'type': 'function', 'function': dict(fields, strict=True)})
        elif api == 'responses':
            result.append(dict(fields, type='function', strict=True))
        elif api == 'anthropic':
            result.append({'name': name, 'description': tool.description,
                           'input_schema': fields['parameters']})
        else:
            raise ValueError('unsupported_api')
    return result


class MemoryOrderBackend:
    """Injected deterministic backend for replay; production supplies tenant-scoped reads."""
    def __init__(self, reverse=False):
        self.reverse = reverse
        self.invocations = []
        self.completed = []
        self.active = 0
        self.max_active = 0
        self.rows = {
            (1, 'A100'): {'order_id': 'A100', 'payment_id': 'pay_A100',
                         'status': 'delivered', 'customer_contact': 'alex@example.test',
                         'eligible': True},
            (2, 'A100'): {'order_id': 'A100', 'payment_id': 'pay_A200',
                         'status': 'processing', 'customer_contact': '+1 202 555 0199',
                         'eligible': False},
        }

    async def _read(self, name, scope, order_id):
        self.invocations.append((name, scope.tenant_id, order_id))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        slow = (name == 'lookup_order') == self.reverse
        try:
            await asyncio.sleep(0.03 if slow else 0.001)
            row = dict(self.rows[(scope.tenant_id, order_id)])
            self.completed.append(name)
            return row
        finally:
            self.active -= 1

    async def lookup_order(self, *, scope, order_id):
        row = await self._read('lookup_order', scope, order_id)
        return {key: row[key] for key in
                ('order_id', 'payment_id', 'status', 'customer_contact')}

    async def check_refund(self, *, scope, order_id):
        row = await self._read('check_refund', scope, order_id)
        return {'order_id': row['order_id'], 'eligible': row['eligible'],
                'reason': 'within_window' if row['eligible'] else 'outside_window'}


def registry_for(backend):
    return {
        'lookup_order': Tool('Read order details; does not change the order.',
                             OrderToolArguments, OrderToolResult, backend.lookup_order),
        'check_refund': Tool('Read refund eligibility; never issue a refund.',
                             OrderToolArguments, RefundToolResult, backend.check_refund),
    }


def safe_result(model):
    # Structured identifiers and enums are constrained by result schemas.
    # The only free-text field is classified as PII and suppressed in full.
    data = model.model_dump()
    if 'customer_contact' in data:
        data['customer_contact'] = '[REDACTED]'
    return data


async def dispatch(call, registry, scope):
    with tool_span(scope, call) as operation:
        def failure(code):
            operation.fail(code)
            return {'call_id': call.call_id, 'payload': {'ok': False, 'error': code}}

        tool = registry.get(call.name)
        if tool is None:
            return failure('unknown_tool')
        try:
            raw = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
        except (ValueError, TypeError):
            return failure('invalid_json')
        try:
            arguments = tool.arguments.model_validate(raw)
        except ValidationError:
            return failure('invalid_arguments')
        try:
            raw_result = await tool.run(scope=scope, **arguments.model_dump())
            validated = tool.result.model_validate(raw_result)
            data = safe_result(validated)
        except Exception:
            return failure('execution_error')
        return {'call_id': call.call_id, 'payload': {'ok': True, 'data': data}}


async def dispatch_batch(calls, registry, scope):
    ids = [call.call_id for call in calls]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_call_id')
    # gather runs independent reads concurrently and retains input order.
    # Each result still carries its ID: no caller needs completion-order pairing.
    return await asyncio.gather(*(dispatch(call, registry, scope) for call in calls))


class ToolReplayClient(LlmClient):
    """Tool-aware offline fake using the existing client's complete interface."""
    def __init__(self):
        if os.getenv('ANCHOR_LLM_PROVIDER', 'fake') not in ('', 'fake', 'offline'):
            raise ValueError('tool_replay_requires_offline_provider')
        self.steps = json.loads((Path(__file__).parent / 'tests' / 'fixtures' /
                                 'tool_call_responses.json').read_text())
        self.position = 0
        self.prefix = None
        self.pending = []
        self.expected_tools = None

    def complete(self, prompt, system=None, max_tokens=256):
        history = prompt
        if self.expected_tools is None:
            self.expected_tools = deepcopy(system)
        assert system == self.expected_tools, 'changed tool definitions'
        if self.prefix is not None:
            assert history[:len(self.prefix)] == self.prefix, 'changed conversation history'
            tail = history[len(self.prefix):]
            assert [m.get('tool_call_id') for m in tail] == self.pending, 'incomplete batch'
            for message in tail:
                value = json.loads(message['content'])
                assert type(value['ok']) is bool
        else:
            assert history == [{'role': 'user', 'content': 'Can I refund order A100?'}]
        step = deepcopy(self.steps[self.position])
        self.position += 1
        calls = []
        for call_id, name, arguments in step.get('calls', []):
            if arguments == '$lookup_order_id':
                lookup = next(m for m in history if m.get('tool_call_id') == 'lookup_1')
                order_id = json.loads(lookup['content'])['data']['order_id']
                arguments = json.dumps({'order_id': order_id})
            calls.append({'id': call_id, 'type': 'function',
                          'function': {'name': name, 'arguments': arguments}})
        message = {'role': 'assistant', 'content': step.get('text'), 'tool_calls': calls}
        self.prefix = deepcopy(history) + [deepcopy(message)]
        self.pending = [c['id'] for c in calls]
        return message


async def run_conversation(client, registry, scope):
    history = [{'role': 'user', 'content': 'Can I refund order A100?'}]
    batches = []
    for _ in range(6):
        with span('python.tool_llm', stage='llm', tenant_id=scope.tenant_id,
                  conversation_id=scope.conversation_id):
            message = client.complete(history, system=definitions(registry))
        calls = normalize('chat', message)
        history.append(message)
        if not calls:
            return history, batches
        results = await dispatch_batch(calls, registry, scope)
        history.extend(result_messages('chat', results))
        batches.append(results)
    raise RuntimeError('tool_turn_limit')


def inspect_registry():
    registry = registry_for(MemoryOrderBackend())
    for name, tool in registry.items():
        print(name, 'arguments=', json.dumps(tool.arguments.model_json_schema()),
              'result=', json.dumps(tool.result.model_json_schema()))


async def demo(reverse=False):
    backend = MemoryOrderBackend(reverse=reverse)
    scope = ToolScope(tenant_id=1, conversation_id=7)
    with span('python.tool_conversation', tenant_id=1, conversation_id=7):
        history, batches = await run_conversation(ToolReplayClient(), registry_for(backend), scope)
    print('FIRST BATCH', json.dumps(batches[0]))
    print('COMPLETION ORDER', backend.completed[:2], 'max_active=', backend.max_active)
    print('CHAIN', json.dumps(batches[1]))
    print('ERRORS', json.dumps(batches[2]))
    print('CORRECTION', json.dumps(batches[3]))
    print('Anchor:', history[-1]['content'])


if __name__ == '__main__':
    import sys
    if '--inspect' in sys.argv:
        inspect_registry()
    else:
        asyncio.run(demo(reverse='--reverse' in sys.argv))
