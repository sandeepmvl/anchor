import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.agentic_rag import RecordedRetrievalBoundary, fixture_data
from anchor_ai.rag import RagRuntime


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-tool-calls', type=int, default=None)
    args = parser.parse_args()
    data = fixture_data()
    for name in ('supplied-policy', 'known-policy', 'compound-policy'):
        events = []
        backend = RecordedRetrievalBoundary(data)
        runtime = RagRuntime(backend, client=None, emit=events.append,
                             evidence_budget=2400)
        answer = await runtime.answer(data['tickets'][name]['question'],
            request_id=name, trace_id='a' * 32, mode='agent',
            max_tool_calls=args.max_tool_calls)
        for record in runtime.last_trajectory:
            print(json.dumps({'ticket': name, 'decision': record}))
        for event in events:
            print(json.dumps({'ticket': name, **event}))
        # Queries below come only from authored fixtures, not customer telemetry.
        print(json.dumps({'ticket': name, 'fixture_retrieve_calls': [
            {'tool': 'retrieve', 'arguments': {'query': call[2]}}
            for call in backend.calls]}))
        print(json.dumps({'ticket': name, 'final_answer': answer}))


if __name__ == '__main__':
    asyncio.run(main())
