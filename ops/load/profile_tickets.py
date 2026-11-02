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


async def spike_profile(path, trace_id):
    """Reuse profiling workloads, not production endpoint/auth contracts."""
    if path not in {'/rag', '/bounded_agent'}:
        raise ValueError('unknown_spike_path')
    # chat with three bounded model calls is a local workflow timing fixture.
    # It does not execute the production bounded-agent policy or its tools.
    result = await profile_ticket('rag' if path == '/rag' else 'chat',
                                  1 if path == '/rag' else 3, trace_id)
    result['generation_ms'] = sum(s['inclusive_ms'] for s in result['spans']
                                  if s['name'] == 'model')
    result['adapter'] = 'offline_profile_workload_not_production_endpoint'
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


class FakeUsageProvider:
    def __init__(self, usage):
        self.usage = usage

    def key(self):
        return 'offline-only'

    def client(self):
        import httpx
        def send(request):
            return httpx.Response(200, json={'usage': self.usage, 'reply': 'fixture'})
        return httpx.AsyncClient(transport=httpx.MockTransport(send), trust_env=False)


def raw_usage(i, c, o, r):
    return dict(prompt_tokens=i, completion_tokens=o, total_tokens=i+o,
                prompt_tokens_details={'cached_tokens': c},
                completion_tokens_details={'reasoning_tokens': r})


async def perform(collector, *, feature, role, model, quantities, retry=False):
    from types import SimpleNamespace
    from anchor_ai.provider import accounted_provider
    from anchor_ai.runtime import verified_cost_scope
    def verify(token):
        if token != 'offline-execution':
            raise PermissionError('fixture_token_rejected')
        return SimpleNamespace(tenant_id=1, role='customer')
    provider = accounted_provider(FakeUsageProvider(raw_usage(*quantities)), collector)
    with verified_cost_scope(collector, verify=verify, token='offline-execution',
                             feature=feature, role=role, model=model, window='2026-01'):
        async with provider.client() as client:
            await client.post('https://offline.invalid/completions', json={'fixture': True})
            if retry:
                await client.post('https://offline.invalid/completions', json={'fixture': True})


async def replay():
    from anchor_ai.cost_accounting import Collector, DRIVERS, price
    from anchor_ai.observability import cost_measurement
    collector = Collector()
    for _ in range(15):
        for model, quantities in [('standard', (100, 0, 40, 30)),
                                  ('standard', (200, 50, 40, 30)),
                                  ('premium', (300, 100, 40, 30))]:
            await perform(collector, feature='chat', role='base_completion', model=model, quantities=quantities)
        for role in ('first_query_embedding', 'extra_query_embedding'):
            await perform(collector, feature='rag', role=role, model='embedding', quantities=(20, 0, 0, 0))
        await perform(collector, feature='rag', role='rerank', model='standard', quantities=(30, 0, 10, 0))
        await perform(collector, feature='rag', role='base_completion', model='standard',
                      quantities=(200, 50, 50, 10), retry=True)
    collector.require_release()
    totals = {feature: dict.fromkeys(DRIVERS, 0) for feature in ('chat', 'rag')}
    for event in collector.events:
        diagnostic = cost_measurement(event, collector.config)
        assert diagnostic['total'] == sum(price(event, collector.config).values())
        for key, value in price(event, collector.config).items():
            totals[event['feature']][key] += value
    assert list(totals['chat'].values()) == [19500, 3750, 3600, 10800, 0, 0]
    assert list(totals['rag'].values()) == [4500, 750, 3600, 900, 300, 11850]
    directory = Path('tmp/cost_profile')
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'events.json').write_text(json.dumps(collector.events, indent=2))
    # Exercise the real semantic runtime and cache with local fixture embeddings.
    from types import SimpleNamespace
    from anchor_ai.runtime import SemanticAnswerRuntime
    from anchor_ai.semantic_cache import Snapshot, Generated, SemanticCache
    comparison = Collector()
    async def generate(identity, query, snapshot):
        await perform(comparison, feature='chat', role='base_completion', model='standard',
                      quantities=(100, 0, 10, 0))
        return Generated('Cancel from Billing.', public_faq=True)
    runtime = SemanticAnswerRuntime(
        verify=lambda token: SimpleNamespace(tenant_id=1, role='customer'),
        resolve=lambda identity, query, path: Snapshot(1, 'customer', 'public', 'en', 'p1',
            path, '1:1:0', 'e1', 'prompt1', 'model1', 'r1', 'public_faq'),
        embed=lambda query, snapshot: [1.0, 0.0], generate=generate,
        cache=SemanticCache(emit=lambda record: None))
    await runtime.answer('fixture', 'How do I cancel my subscription?', path='chat')
    before = len(comparison.events)
    await runtime.answer('fixture', 'How do I cancel my subscription?', path='chat')
    semantic_attempts = len(comparison.events) - before
    assert semantic_attempts == 0
    await perform(comparison, feature='chat', role='base_completion', model='standard',
                  quantities=(100, 80, 10, 0))
    prefix_cost = sum(price(comparison.events[-1], comparison.config).values())
    assert prefix_cost == 180
    print(json.dumps({'tickets': 30, 'events': len(collector.events), 'totals': totals,
        'price_version': collector.config['price_version'], 'budget_version': collector.config['budget_version'],
        'semantic_hit_inference_attempts': semantic_attempts, 'prefix_hit_inference_attempts': 1,
        'prefix_hit_cost': prefix_cost,
        'not_verified': ['hosted inference', 'production authentication', 'network delivery', 'write capacity']}, indent=2))
    return collector


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge-traceparent')
    parser.add_argument('--mode', choices=['chat', 'rag'], default='rag')
    parser.add_argument('--semantic-cache', action='store_true')
    parser.add_argument('--costs', action='store_true')
    args = parser.parse_args()
    if args.costs:
        await replay()
        return
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
