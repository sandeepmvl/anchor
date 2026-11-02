"""Local profiling smoke path; no provider SDK, sockets, credentials, or ticket text."""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.observability import (
    ACTIVE, Profile, distribution, model_call, segment, LatencyBoundary, configuration)


class OfflineClient:
    async def complete(self):
        profile = ACTIVE.get()
        node = profile.nodes[-1]
        node['cached_tokens'] = 0
        node['injected_inference_ms'] = 12.0
        await asyncio.sleep(.008)
        profile.text(node)
        for _ in range(2):
            await asyncio.sleep(.002)
            profile.text(node)
        return 'offline reply'


async def query_branch():
    with segment('branch'):
        for name in ('embedding', 'search', 'rerank'):
            with segment(name):
                await asyncio.sleep(.001)


async def ticket(mode='rag', calls=1):
    if mode == 'rag':
        with segment('retrieval'):
            await asyncio.gather(query_branch(), query_branch())
    client = OfflineClient()
    for index in range(calls):
        if mode == 'rag' or index:
            with segment('tool'):
                await asyncio.sleep(.001)
        await model_call(client.complete)


async def profile_ticket(mode='rag', calls=1, trace_id=None):
    profile = Profile(trace_id)
    with profile.activate():
        with profile.span('request'):
            await ticket(mode, calls)
    result = profile.report()
    result['endpoint'] = mode
    result['mode'] = 'offline_synthetic_adapter'
    return result


async def boundary_probe(traceparent, mode):
    """ASGI contract smoke used by Rails. No production endpoint/auth claims."""
    records = []

    async def application(scope, receive, send):
        await ticket(mode)
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'offline'})

    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message):
        pass

    boundary = LatencyBoundary(application, sink=records.append)
    await boundary({'type': 'http', 'headers': [(b'traceparent', traceparent.encode())]}, receive, send)
    return records[0]


def report(rows):
    groups = defaultdict(list)
    for row in rows:
        for span in row['spans']:
            groups[span['name']].append(span['exclusive_ms'])
    ranking = sorted(({'name': name, 'exclusive_ms': distribution(values)}
                      for name, values in groups.items()),
                     key=lambda item: item['exclusive_ms']['p95'], reverse=True)
    return {
        'samples': len(rows), 'mode': 'in_process_offline',
        'total_ms': distribution([r['total_ms'] for r in rows]),
        'round_trip_counts': sorted(set(r['round_trips'] for r in rows)),
        'exclusive_segments': ranking,
        'dominant_exclusive_segment': ranking[0]['name'],
        'labels': {
            'measured': ['local elapsed spans', 'synthetic text-event timestamps',
                         'logical model-client invocation count'],
            'fixture_only': ['cached_tokens', 'injected_inference_ms'],
            'unknown': ['network', 'prefill', 'decode', 'provider_queueing',
                        'provider_cached_tokens', 'provider_output_tokens'],
            'not_verified': ['production HTTP', 'SSE delivery', 'hosted inference',
                             'Rails-to-Python network latency'],
            'transport': 'Client-boundary elapsed time is not pure network time. '
                         'The required run makes no network requests.',
            'retry_wait': 'Instrument actual client sleep as retry_wait; no retry is injected here.',
        },
        'cohorts': {
            str(count): distribution([r['total_ms'] for r in rows if r['round_trips'] == count])
            for count in sorted(set(r['round_trips'] for r in rows))
        },
        'max_remainder_fraction': max(r['application_remainder_ms'] / r['total_ms'] for r in rows),
        'traces': rows,
    }


async def semantic_cache_probe():
    from dataclasses import replace
    from types import SimpleNamespace
    import yaml
    from anchor_ai.app import public_faq_answer
    from anchor_ai.runtime import SemanticAnswerRuntime
    from anchor_ai.semantic_cache import Snapshot, Generated, SemanticCache
    settings = yaml.safe_load(Path('config/performance.yml').read_text())['semantic_cache']
    if settings['record_content'] is not False:
        raise ValueError('unsafe_cache_profile')
    cases = [json.loads(line) for line in Path('eval/datasets/semantic_cache_cases.jsonl').read_text().splitlines()]
    results = []
    for path in ('chat', 'rag'):
        events, calls = [], []
        state = {'kb': '1:1:0'}
        def verify(token):
            if token not in {'one', 'two'}:
                raise ValueError('fixture_token_rejected')
            return SimpleNamespace(tenant_id=1 if token == 'one' else 2, role='customer')
        def resolve(identity, query, answer_path):
            return Snapshot(identity.tenant_id, identity.role, 'public', 'en', 'p1',
                            answer_path, state['kb'], 'e1', 'prompt1', 'model1', 'r1', 'public_faq')
        def embed(query, snapshot):
            score = next(c['score'] for c in cases if c['query'] == query)
            return [score, (1 - score * score) ** .5]
        async def generate(identity, query, snapshot):
            calls.append(1)
            return Generated('Cancel from Billing.', public_faq=True)
        runtime = SemanticAnswerRuntime(verify=verify, resolve=resolve, embed=embed,
            generate=generate, cache=SemanticCache(emit=events.append))
        async def ask(token, query):
            return await public_faq_answer(token, query, path=path, runtime=runtime)
        await ask('one', cases[0]['query'])
        await ask('one', cases[1]['query'])
        await ask('one', cases[0]['query'])
        assert len(calls) == 1
        await ask('two', cases[0]['query'])
        assert len(calls) == 2
        state['kb'] = '1:1:1'
        await ask('one', cases[0]['query'])
        assert len(calls) == 3
        results.append(dict(path=path, cold='miss', paraphrase='hit', exact='hit',
                            repeat_answer_model_calls=1, other_tenant='miss', changed_kb='miss'))
    return dict(scope=settings['replay_scope'], results=results,
                not_verified=['live embedding quality', 'production authentication',
                              'HTTP integration', 'cross-worker consistency', 'hosted latency'])


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge-traceparent')
    parser.add_argument('--mode', choices=['chat', 'rag'], default='rag')
    parser.add_argument('--semantic-cache', action='store_true')
    args = parser.parse_args()
    if args.semantic_cache:
        print(json.dumps(await semantic_cache_probe(), indent=2))
        return
    config = configuration()
    if not config['enabled']:
        raise SystemExit('profiling_disabled')
    if args.bridge_traceparent:
        print(json.dumps(await boundary_probe(args.bridge_traceparent, args.mode)))
        return
    rows = [await profile_ticket(args.mode, 3 if i >= 16 else 1)
            for i in range(config['samples'])]
    result = report(rows)
    path = Path('tmp/latency_profile/report.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'traces'}, indent=2))
    print('Correlated timing waterfall: ' + rows[0]['trace_id'])
    for span in rows[0]['spans']:
        print(f"{span['id']:2} parent={str(span['parent']):4} {span['name']:10} "
              f"inclusive={span['inclusive_ms']:.3f}ms exclusive={span['exclusive_ms']:.3f}ms")


if __name__ == '__main__':
    asyncio.run(main())
