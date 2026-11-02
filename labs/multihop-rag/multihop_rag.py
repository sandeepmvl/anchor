import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from anchor_ai.app import app
from anchor_ai.multihop import fixture_data, fixture_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('single', 'multi'), default='single')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        from anchor_ai.prompts import MULTIHOP_PLAN_INSTRUCTION
        runtime = fixture_runtime()
        adapter = runtime.client
        print('Existing adapter:', type(adapter).__name__)
        print('Task client:', type(adapter.client).__name__)
        print('Model mode:', adapter.client.mode)
        print('Planner instruction:', MULTIHOP_PLAN_INSTRUCTION)
        print('Both tasks call adapter.complete(messages).')
        return
    data = fixture_data()
    captured = []

    def factory(tenant_id, emit=None):
        runtime = fixture_runtime(tenant_id, emit=emit)
        original = runtime.retriever.retrieve

        async def recording_retrieve(query, top_k=1):
            result = await original(query, top_k)
            captured.append(dict(query=query, tenant_id=result['tenant_id'],
                                 generation_id=result['generation_id'],
                                 passages=result['hits']))
            return result

        runtime.retriever.retrieve = recording_retrieve
        return runtime

    previous = getattr(app.state, 'bounded_rag_runtime_factory', None)
    app.state.bounded_rag_runtime_factory = factory
    os.environ.setdefault('ANCHOR_AI_SECRET', 'local-multihop-demo-only')
    try:
        with TestClient(app) as client:
            response = client.post('/v1/rag', headers={
                'X-Anchor-Secret': os.environ['ANCHOR_AI_SECRET']}, json=dict(
                tenant_id=1, conversation_id=10, message=data['ticket'],
                mode=args.mode, top_k=1))
        response.raise_for_status()
        answer = response.json()
        print(json.dumps(dict(model_mode=os.environ.get('ANCHOR_LLM_MODE', 'offline'),
                              mode=args.mode, question=data['ticket'],
                              hops=captured, answer=answer), indent=2))
        if os.environ.get('ANCHOR_LLM_MODE', 'offline') == 'offline':
            if args.mode == 'single':
                assert len(captured) == 1 and answer['abstained']
                print('Single-hop: the refund-eligibility connection is missing.')
            else:
                assert len(captured) == 2
                assert 'CustomCare' in captured[1]['query']
                assert answer['citations'] == [701, 702]
                print('Anchor joins order policy [701] and refund policy [702]: '
                      'not eligible at day 20; the applicable window is 14 days.')
    finally:
        if previous is None:
            del app.state.bounded_rag_runtime_factory
        else:
            app.state.bounded_rag_runtime_factory = previous


if __name__ == '__main__':
    main()
