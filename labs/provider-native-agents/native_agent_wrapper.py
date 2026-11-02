import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anchor_ai.contracts import ToolScope
from anchor_ai.native_agent_wrapper import NativeAgent
from anchor_ai.orchestration import run_native_support_turn
from anchor_ai.provider import configured_native_provider
from anchor_ai.storage import NativeSessionStore
from anchor_ai.tools import MemoryOrderBackend, native_definitions, registry_for


async def main(trace_only=False):
    provider = configured_native_provider()
    backend = MemoryOrderBackend()
    registry = registry_for(backend)
    scope = ToolScope(tenant_id=1, conversation_id=456)
    with tempfile.TemporaryDirectory() as directory:
        sessions = NativeSessionStore(Path(directory) / 'native_sessions.sqlite')
        try:
            agent = NativeAgent(provider, registry, sessions)
            if not trace_only:
                print('NATIVE DEFINITIONS', json.dumps(native_definitions(registry)))
            first = await run_native_support_turn(
                agent, scope, 'For ticket 456, look up order A100.', enabled=True)
            second = await run_native_support_turn(
                agent, scope, 'Which order did we discuss, and what was its status?',
                enabled=True, previous_response_id=first['response_id'])
            if not trace_only:
                print('FOLLOW-UP', json.dumps(dict(
                    previous_response_id=first['response_id'], **second,
                    prior_messages_sent=0)))
                print('LOCAL TOOL INVOCATIONS', backend.invocations)
                if hasattr(provider, 'responses'):
                    count = len(provider.responses[second['response_id']]['history'])
                    print('SEPARATE PROVIDER STORE', json.dumps({'history_items': count}))
                print('SIMULATION: authored fixtures; no billing or latency measurement')
            for event in agent.events:
                print('TRACE', json.dumps(event, sort_keys=True))
        finally:
            sessions.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-only', action='store_true')
    args = parser.parse_args()
    asyncio.run(main(args.trace_only))
