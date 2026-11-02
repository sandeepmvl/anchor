import asyncio
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest
import yaml

from anchor_ai.prompts import AuthorizedScope, layout, LAYOUT_VERSION
from anchor_ai.provider import SyntheticPrefixProvider
from anchor_ai.llm_client import PrefixClient
from anchor_ai.observability import (
    break_even, choose_tenant_policy, prefix_configuration, normalize_usage,
    report, require_production_evidence)
from anchor_ai.semantic_cache import SemanticCache, Snapshot, Generated, with_answer_cache


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


SCOPE = AuthorizedScope(1, 'support', 'public', 'permissions:v1')
PRICES = {'base': '1', 'read': '.1', 'write': '1.25'}
TOOLS = [{'name': 'lookup_order', 'parameters': {'type': 'object', 'properties': {}}},
         {'name': 'check_refund', 'parameters': {'properties': {}, 'type': 'object'}}]


def prompt(ticket='Invoice name?', **options):
    return layout(scope=SCOPE, ticket=ticket, tools=TOOLS, **options)


def snapshot():
    return Snapshot(1, 'support', 'public', 'en', 'permissions:v1', 'faq',
                    'kb:v1', 'embedding:v1', 'triage-prefix:v1',
                    'gpt-4o-mini', 'retrieval:v1', eligibility='public_faq')


def test_policy_tools_examples_precede_variable_content():
    a = prompt(retrieved_context='A', timestamp='first')
    b = layout(scope=SCOPE, ticket='Different invoice?', tools=list(reversed(TOOLS)),
               retrieved_context='B', timestamp='second')
    assert a.prefix_bytes == b.prefix_bytes
    assert a.prefix_identity == b.prefix_identity
    text = a.context_messages[0]['content']
    assert text.index('Classify a support ticket') < text.index('Tool definitions:')
    assert text.index('check_refund') < text.index('lookup_order') < text.index('Examples:')
    assert 'first' not in text and 'Different invoice?' not in text
    assert 'retrieved_context' in a.context_messages[-1]['content']
    assert a.shared_identity == b.shared_identity


def test_tenant_prefix_namespace_and_version_invalidation():
    opts = dict(tenant_prefix=True, tenant_policy='Public invoice policy',
                tenant_policy_version='tenant:v1')
    a = prompt(**opts)
    b = layout(scope=replace(SCOPE, tenant_id=2), ticket='Invoice name?', tools=TOOLS, **opts)
    permission = layout(scope=replace(SCOPE, permission_version='permissions:v2'),
                        ticket='Invoice name?', tools=TOOLS, **opts)
    assert len({a.prefix_identity, b.prefix_identity, permission.prefix_identity}) == 3
    assert a.shared_identity == b.shared_identity
    assert prompt(**dict(opts, tenant_policy_version='tenant:v2')).prefix_identity != a.prefix_identity
    assert prompt(version='triage-prefix:v2').prefix_identity != prompt().prefix_identity
    assert prompt(model='other-model').prefix_identity != prompt().prefix_identity
    assert prompt(policy='Changed platform policy').prefix_identity != prompt().prefix_identity
    assert layout(scope=SCOPE, ticket='Invoice name?', tools=TOOLS[:1]).prefix_identity != prompt().prefix_identity
    with pytest.raises(ValueError):
        layout(scope=None, ticket='Invoice name?')
    with pytest.raises(ValueError):
        prompt(tenant_prefix=True)


def test_explicit_usage_and_configured_prices_only():
    raw = {'prompt_tokens': 1200, 'completion_tokens': 20,
           'prompt_tokens_details': {'cached_tokens': 1000, 'cache_write_tokens': 0}}
    r = report(raw, style='chat', prices=PRICES, ttl_seconds=300)
    assert r['provider_cache_status'] == 'hit'
    assert r['uncached_input_tokens'] == 200
    assert r['estimated_input_saving_usd'] == pytest.approx(.0009)
    cold = dict(raw, prompt_tokens_details={'cached_tokens': 0, 'cache_write_tokens': 1000})
    c = report(cold, style='chat', prices=PRICES, ttl_seconds=300)
    assert c['provider_cache_status'] == 'miss'
    assert c['estimated_input_saving_usd'] == pytest.approx(-.00025)
    assert c['estimated_input_saving_usd'] + r['estimated_input_saving_usd'] == pytest.approx(.00065)
    assert break_even(PRICES) == pytest.approx(.25 / .9)
    assert break_even(dict(PRICES, write='2')) == pytest.approx(1 / .9)
    for missing in (None, {'prompt_tokens': 1200, 'completion_tokens': 20}):
        unknown = report(missing, style='chat', prices=PRICES, ttl_seconds=300)
        assert unknown['provider_cache_status'] == 'unknown'
        assert unknown['estimated_input_saving_usd'] is None
    no_write = dict(raw, prompt_tokens_details={'cached_tokens': 1000})
    assert report(no_write, style='chat', prices=PRICES,
                  ttl_seconds=300)['estimated_input_saving_usd'] is None
    with pytest.raises(ValueError):
        normalize_usage(dict(raw, prompt_tokens_details={'cached_tokens': 1201}), 'chat')
    with pytest.raises(ValueError):
        normalize_usage(dict(raw, prompt_tokens_details={'cached_tokens': True}), 'chat')


def test_api_usage_shapes_agree():
    chat = {'prompt_tokens': 1200, 'completion_tokens': 20,
            'prompt_tokens_details': {'cached_tokens': 100, 'cache_write_tokens': 900}}
    responses = {'input_tokens': 1200, 'output_tokens': 20,
                 'input_tokens_details': {'cached_tokens': 100, 'cache_write_tokens': 900}}
    messages = {'input_tokens': 200, 'output_tokens': 20,
                'cache_read_input_tokens': 100, 'cache_creation_input_tokens': 900}
    assert normalize_usage(chat, 'chat') == normalize_usage(responses, 'responses')
    assert normalize_usage(chat, 'chat') == normalize_usage(messages, 'messages')


def test_ttl_decisions_use_whole_run_economics():
    policies = prefix_configuration()['policies']
    assert choose_tenant_policy(7200, 10, policies) == 'shared_only'
    assert choose_tenant_policy(2400, 2, policies) == 'shared_only'
    assert choose_tenant_policy(2400, 10, policies) == 'extended'
    assert choose_tenant_policy(60, 10, policies) == 'default'
    assert choose_tenant_policy(300, 10, {'default': policies['default'],
                                       'extended': dict(policies['extended'], ttl_seconds=300)}) == 'shared_only'


def test_existing_client_cold_warm_expiry_and_fail_closed_evidence():
    async def run():
        clock = Clock()
        provider = SyntheticPrefixProvider(ttl_seconds=300, clock=clock)
        client = PrefixClient(provider, prices=PRICES, ttl_seconds=300)
        rows, replies = [], []
        for n in range(20):
            answer, evidence = await client.complete(prompt(ticket=f'Invoice name {n}?'))
            replies.append(answer)
            rows.append(evidence)
        assert rows[0]['cached_input_tokens'] == 0
        assert all(r['cached_input_tokens'] > 0 for r in rows[1:])
        assert len(provider.requests) == 20
        with pytest.raises(ValueError, match='real_provider'):
            require_production_evidence(rows)
        with pytest.raises(ValueError):
            require_production_evidence([])
        unknown = report(None, style='chat', prices=PRICES, ttl_seconds=300)
        with pytest.raises(ValueError):
            require_production_evidence([unknown])
        clock.advance(301)
        answer, evidence = await client.complete(prompt())
        assert evidence['provider_cache_status'] == 'miss'
        assert answer == replies[0]
    asyncio.run(run())


def test_whole_answer_hit_skips_provider_and_does_not_claim_prefix_hit():
    async def run():
        clock = Clock()
        provider = SyntheticPrefixProvider(ttl_seconds=300, clock=clock)
        client = PrefixClient(provider, prices=PRICES, ttl_seconds=300, synthetic=True)
        cache = SemanticCache(clock=clock, emit=lambda event: None)
        async def generate():
            text, evidence = await client.complete(prompt())
            return Generated(text, public_faq=True), evidence
        first, cold = await with_answer_cache(cache, snapshot(), 'Invoice name?', [1, 0], generate)
        second, hit = await with_answer_cache(cache, snapshot(), 'Invoice name?', [1, 0], generate)
        assert first == second
        assert len(provider.requests) == 1
        assert cold['provider_called'] is True
        assert hit['answer_cache_hit'] is True
        assert hit['provider_called'] is False
        assert hit['provider_cache_status'] == 'not_called'
        assert hit['cached_input_tokens'] is None
        assert hit['estimated_input_saving_usd'] is None
    asyncio.run(run())


def test_registry_layout_config_matches_python_and_rejects_unknown_version():
    source = Path('config/llm.yml').read_text()
    line = next(x for x in source.splitlines() if 'provider_prefix_layout_version:' in x)
    assert yaml.safe_load(line)['provider_prefix_layout_version'] == LAYOUT_VERSION
    ruby = '''
require_relative './app/services/prompt_registry'
module Rails
  def self.application = self
  def self.config_for(name) = {provider_prefix_layout_version: 'triage-prefix:v1'}
end
raise unless PromptRegistry.triage_prefix_layout == 'triage-prefix:v1'
module Rails
  def self.config_for(name) = {provider_prefix_layout_version: 'unknown'}
end
begin
  PromptRegistry.triage_prefix_layout
  raise 'accepted unknown layout'
rescue ArgumentError
end
'''
    subprocess.run(['ruby', '-e', ruby], check=True, timeout=5)
