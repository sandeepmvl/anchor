import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from anchor_ai.semantic_cache import SemanticCache, Snapshot, Generated, configuration
from anchor_ai.runtime import SemanticAnswerRuntime, ContextError
from anchor_ai.index_maintenance import scoped_generation_key

CASES = [json.loads(line) for line in Path('eval/datasets/semantic_cache_cases.jsonl').read_text().splitlines()]
BASE = Snapshot(1, 'customer', 'public', 'en', 'permissions1', 'chat',
                '1:1:0', 'embedding1', 'prompt1', 'model1', 'retrieval1', 'public_faq')
SAFE = Generated('Cancel from Billing.', public_faq=True)


def run(awaitable):
    return asyncio.run(awaitable)


class Fixture:
    def __init__(self, path='chat'):
        self.snapshot = replace(BASE, answer_path=path)
        self.events, self.calls = [], []
        self.now = 0
        self.cache = SemanticCache(clock=lambda: self.now, emit=self.events.append)
        self.runtime = SemanticAnswerRuntime(verify=self.verify, resolve=self.resolve,
            embed=self.embed, generate=self.generate, cache=self.cache)

    def verify(self, token):
        if token != 'signed-fixture':
            raise ContextError('invalid_fixture_token')
        return SimpleNamespace(tenant_id=self.snapshot.tenant_id, role=self.snapshot.role)

    def resolve(self, identity, query, path):
        case = next(c for c in CASES if c['query'] == query)
        return replace(self.snapshot, answer_path=path,
                       eligibility=case.get('bypass', 'public_faq'))

    def embed(self, query, snapshot):
        score = next(c['score'] for c in CASES if c['query'] == query)
        return [score, (1 - score ** 2) ** .5]

    async def generate(self, identity, query, snapshot):
        self.calls.append(snapshot.scope)
        case = next(c for c in CASES if c['query'] == query)
        return replace(SAFE, personalized=case.get('unsafe_answer', False))

    def ask(self, index=0):
        return run(self.runtime.answer('signed-fixture', CASES[index]['query'],
                                      path=self.snapshot.answer_path))


@pytest.mark.parametrize('path', ['chat', 'rag'])
def test_public_faq(path):
    f = Fixture(path)
    assert f.ask() == f.ask(1) == f.ask() == SAFE.text
    assert len(f.calls) == 1
    assert [e['cache_outcome'] for e in f.events] == ['miss', 'hit', 'hit']
    assert [e['cache_reason'] for e in f.events] == ['no_match', 'semantic', 'exact']
    print(f'{path}: cold=miss paraphrase=hit exact=hit answer_model_calls=1')


@pytest.mark.parametrize('field,value', [
    ('tenant_id', 2), ('role', 'admin'), ('audience', 'staff'), ('locale', 'fr'),
    ('permission_version', 'permissions2'), ('kb', '1:2:0'), ('kb', '1:1:1'),
    ('embedding_version', 'embedding2'), ('prompt_version', 'prompt2'),
    ('model_version', 'model2'), ('retrieval_version', 'retrieval2'),
    ('answer_path', 'rag')])
def test_scope_and_invalidation(field, value):
    f = Fixture()
    f.ask()
    f.snapshot = replace(f.snapshot, **{field: value})
    f.ask()
    assert len(f.calls) == 2
    assert f.events[-1]['cache_outcome'] == 'miss'


def test_scope_and_invalidation_ttl_and_generation_key():
    f = Fixture()
    f.ask()
    f.now = 60
    f.ask()
    assert len(f.calls) == 2
    gen = dict(tenant_id=1, id=2, epoch=3, status='ready')
    assert scoped_generation_key(gen) == '1:2:3'
    assert scoped_generation_key(dict(gen, epoch=4)) != scoped_generation_key(gen)
    with pytest.raises(ValueError):
        scoped_generation_key(dict(gen, status='building'))


def test_update_during_hit():
    f = Fixture()
    f.ask()
    original = f.cache.lookup
    def changed(*args):
        answer = original(*args)
        f.snapshot = replace(f.snapshot, kb='1:1:1')
        return answer
    f.cache.lookup = changed
    f.ask()
    assert len(f.calls) == 2
    assert f.calls[-1][6] == '1:1:1'


@pytest.mark.parametrize('index', range(4, 10))
def test_unsafe_or_filtered(index):
    f = Fixture()
    f.ask(index)
    f.ask(index)
    assert len(f.calls) == 2 and not f.cache.rows
    if index != 8:
        assert all(e['cache_outcome'] == 'bypass' for e in f.events)


@pytest.mark.parametrize('flag', ['personalized', 'action_bearing', 'permission_filtered'])
def test_unsafe_generated_flags(flag):
    f = Fixture()
    assert not f.cache.store(BASE, CASES[0]['query'], [1, 0], replace(SAFE, **{flag: True}))
    assert not f.cache.store(BASE, CASES[0]['query'], [1, 0], Generated('person@example.com', True))


def test_threshold():
    f = Fixture()
    f.ask()
    assert f.cache.lookup(BASE, CASES[2]['query'], [.96, .28]) is None
    f.cache.config['threshold'] = .95
    assert f.cache.lookup(BASE, CASES[2]['query'], [.96, .28]) == SAFE.text
    print('0.97 rejects close negative; deliberately unsafe 0.95 admits it')


def test_ann_switch():
    f = Fixture()
    # Synthetic records use the same public answer; this is not a quality dataset.
    for i in range(3001):
        assert f.cache.store(BASE, 'synthetic FAQ ' + str(i), [.96, .28], SAFE)
    foreign = replace(BASE, tenant_id=2)
    f.cache.store(foreign, 'foreign FAQ', [1, 0], SAFE)
    assert f.cache.lookup(BASE, 'probe', [1, 0]) is None
    assert f.events[-1]['cache_lookup_mode'] == 'ann'
    assert 0 < f.events[-1]['cache_candidates'] <= 16
    assert f.cache.lookup(BASE, 'paraphrase probe', [.96, .28]) == SAFE.text
    assert f.cache.lookup(replace(BASE, kb='1:1:1'), 'probe', [1, 0]) is None
    print('tenant_entries=3001 lookup=ann backend=FAISS-HNSW exact_verification=true')
    f.now = 60
    assert f.cache.lookup(BASE, 'probe', [1, 0]) is None
    assert not f.cache.rows


def test_bounds_auth_and_production_refusal():
    f = Fixture()
    f.cache.config.update(max_tenant_entries=2, max_entries=3)
    for i in range(4):
        f.cache.store(BASE, 'public FAQ ' + str(i), [1, 0], SAFE)
    assert len(f.cache.rows) == 2
    with pytest.raises(ContextError):
        run(f.runtime.answer('forged', CASES[0]['query'], path='chat'))
    for fixture, evidence in [(True, None), (False, None)]:
        with pytest.raises(ValueError, match='release_evidence_missing'):
            SemanticAnswerRuntime(verify=f.verify, resolve=f.resolve, embed=f.embed,
                generate=f.generate, cache=f.cache, production=True,
                fixture_adapters=fixture, release_evidence=evidence)
    assert configuration()['record_content'] is False


def test_telemetry_and_product_seam():
    from anchor_ai.app import public_faq_answer
    from anchor_ai.observability import emit_semantic_cache
    f = Fixture()
    assert run(public_faq_answer('signed-fixture', CASES[0]['query'],
                                path='chat', runtime=f.runtime)) == SAFE.text
    emit_semantic_cache(f.events[-1])
    with pytest.raises(ValueError):
        emit_semantic_cache(dict(f.events[-1], query='secret'))
