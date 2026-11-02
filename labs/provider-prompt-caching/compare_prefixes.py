"""Run from Anchor's root. No network, sleeps, keys, or production evidence."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anchor_ai.observability import (
    Profile, break_even, choose_tenant_policy, prefix_configuration)
from anchor_ai.prompts import AuthorizedScope, layout
from anchor_ai.provider import SyntheticPrefixProvider
from anchor_ai.llm_client import PrefixClient
from anchor_ai.semantic_cache import Generated, SemanticCache, Snapshot, with_answer_cache


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def emit(label, value):
    print(label + ': ' + json.dumps(value, sort_keys=True))


async def main(section):
    print('OFFLINE SYNTHETIC: hosted cache eligibility, real tokenization, routing, prices,')
    print('and production TTFT are NOT verified. No production release evidence is emitted.')
    cfg = prefix_configuration()
    policy = cfg['policies']['default']
    clock = Clock()
    provider = SyntheticPrefixProvider(ttl_seconds=policy['ttl_seconds'], clock=clock)
    client = PrefixClient(provider, prices=policy['prices'],
                          ttl_seconds=policy['ttl_seconds'], synthetic=True)
    scope = AuthorizedScope(1, 'support', 'public', 'permissions:v1')
    tools = [{'name': 'lookup_order', 'parameters': {'type': 'object'}},
             {'name': 'check_refund', 'parameters': {'type': 'object'}}]

    def make(n, unstable=False):
        return layout(scope=scope, tools=tools, ticket=f'Invoice name {n}?',
                      retrieved_context='Public invoice FAQ', timestamp=f'fixture-{n}',
                      unstable=unstable)

    async def measured(prompt):
        profile = Profile(clock=clock)
        with profile.activate():
            with profile.span('request'):
                with profile.span('model') as node:
                    node['round_trips'] = 1
                    provider.first_text = lambda: profile.text(node)
                    text, evidence = await client.complete(prompt)
                    node['cached_tokens'] = evidence['cached_input_tokens']
        timing = profile.report()
        model = next(s for s in timing['spans'] if s['name'] == 'model')
        return text, dict(evidence, prefix_identity=prompt.prefix_identity,
                          ttft_ms=model['ttft_ms'], timing_basis='synthetic_clock',
                          profiling=timing)

    if section in {'all', 'layout'}:
        rows = [(await measured(make(n, unstable=True)))[1] for n in range(3)]
        assert all(r['cached_input_tokens'] == 0 for r in rows)
        emit('unstable_prefix_read_counts', [r['cached_input_tokens'] for r in rows])
        emit('unstable_prefix_identities', [r['prefix_identity'] for r in rows])

    if section in {'all', 'stable', 'usage', 'expiry'}:
        provider.entries.clear()
        rows = [(await measured(make(n)))[1] for n in range(20)]
        assert rows[0]['cached_input_tokens'] == 0
        assert all(r['cached_input_tokens'] > 0 for r in rows[1:])
        assert len({r['prefix_identity'] for r in rows}) == 1
        emit('stable_prefix_read_counts', [r['cached_input_tokens'] for r in rows])
        emit('cold_report', rows[0])
        emit('new_ticket_still_calls_model', rows[1])
        emit('lifetime_accounting', {
            'net_input_saving_usd': sum(r['estimated_input_saving_usd'] for r in rows),
            'cold_minus_warm_mean_ttft_ms': rows[0]['ttft_ms'] -
                sum(r['ttft_ms'] for r in rows[1:]) / 19,
            'default_break_even_reads_per_write': break_even(policy['prices']),
            'extended_break_even_reads_per_write': break_even(cfg['policies']['extended']['prices']),
            'basis': 'illustrative_prices_and_synthetic_usage_and_timing'})
        before, _ = await measured(make(21))
        clock.advance(policy['ttl_seconds'] + 1)
        after, expired = await measured(make(22))
        assert expired['cached_input_tokens'] == 0 and before == after
        emit('expired_prefix_correctness_unchanged', expired)

    if section in {'all', 'answers'}:
        cache = SemanticCache(clock=clock, emit=lambda event: None)
        snap = Snapshot(1, 'support', 'public', 'en', 'permissions:v1', 'faq',
                        'kb:v1', 'embedding:v1', 'triage-prefix:v1',
                        'gpt-4o-mini', 'retrieval:v1', eligibility='public_faq')
        async def generate():
            text, evidence = await measured(make(50))
            return Generated(text, public_faq=True), evidence
        start = len(provider.requests)
        a, first = await with_answer_cache(cache, snap, 'Invoice name?', [1, 0], generate)
        b, second = await with_answer_cache(cache, snap, 'Invoice name?', [1, 0], generate)
        assert a == b and len(provider.requests) - start == 1
        emit('first_faq_report', first)
        emit('repeated_faq_report', second)
        emit('provider_calls_for_two_faq_requests', len(provider.requests) - start)

    if section in {'all', 'decisions'}:
        choices = []
        for interval, count in [(7200, 10), (2400, 2), (2400, 10), (60, 10)]:
            choices.append(dict(interval_seconds=interval, requests=count,
                                choice=choose_tenant_policy(interval, count, cfg['policies'])))
        emit('tenant_ttl_decisions', choices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', default='all',
                        choices=['all', 'layout', 'stable', 'usage', 'expiry', 'answers', 'decisions'])
    asyncio.run(main(parser.parse_args().section))
