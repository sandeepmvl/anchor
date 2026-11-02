import asyncio
import copy
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from anchor_ai.app import app
from anchor_ai.multihop import (
    FixtureRetriever, TaskModelClient, fixture_data, fixture_runtime,
)
from anchor_ai.prompts import MULTIHOP_PLAN_INSTRUCTION, MULTIHOP_SYNTHESIS_INSTRUCTION
from anchor_ai.rag import ExistingClientAdapter, RagRuntime, published_bounded_rag_runtime
from anchor_ai.schemas import RagRequest


def run(runtime, **kwargs):
    return asyncio.run(runtime.answer(fixture_data()['ticket'], 1,
                       request_id='multihop-test', trace_id='multihop-trace', **kwargs))


def runtime_for(data=None, retriever=None):
    model = TaskModelClient(mode='offline', data=data)
    retriever = retriever or FixtureRetriever(data=data)
    return RagRuntime(retriever, ExistingClientAdapter(model), evidence_budget=2400), model


def test_planning_synthesis_and_dependent_trajectory():
    runtime, model = runtime_for()
    answer = run(runtime, mode='multi')
    assert answer['citations'] == [701, 702]
    assert not answer['abstained']
    assert '14 days' in answer['reply']
    assert [call['system'] for call in model.calls] == [
        MULTIHOP_PLAN_INSTRUCTION, MULTIHOP_SYNTHESIS_INSTRUCTION]
    assert all(call['max_tokens'] == 256 for call in model.calls)
    assert runtime.retriever.calls == ['engraved order policy', 'CustomCare refund eligibility']
    assert runtime.last_trajectory[1]['binding_source_id'] == 701
    evidence = json.loads(model.calls[1]['prompt'])['evidence']['sources']
    assert {source['id'] for source in evidence} == {701, 702}


def test_default_single_hop_is_unchanged():
    assert RagRequest(tenant_id=1, conversation_id=10, message='hello').mode == 'single'
    implicit, first_model = runtime_for()
    explicit, second_model = runtime_for()
    assert run(implicit) == run(explicit, mode='single')
    assert len(implicit.retriever.calls) == 1
    assert len(first_model.calls) == len(second_model.calls) == 1
    assert first_model.calls[0]['system'] != MULTIHOP_PLAN_INSTRUCTION
    assert run(runtime_for()[0])['abstained']


def test_hop_limit_and_oversized_plan_do_not_expand_execution():
    runtime, model = runtime_for()
    assert run(runtime, mode='multi', max_hops=1)['abstained']
    assert runtime.retriever.calls == []
    assert len(model.calls) == 1
    data = copy.deepcopy(fixture_data())
    data['plan']['queries'].append('one more search')
    runtime, model = runtime_for(data)
    assert run(runtime, mode='multi')['abstained']
    assert runtime.retriever.calls == []


def test_missing_binding_stops_without_extra_search():
    data = copy.deepcopy(fixture_data())
    data['documents'][0]['text'] = 'Engraved order policy: contact support.'
    runtime, model = runtime_for(data)
    assert run(runtime, mode='multi')['abstained']
    assert len(runtime.retriever.calls) == 1
    assert len(model.calls) == 1


def test_binding_is_read_from_the_retrieved_passage():
    data = copy.deepcopy(fixture_data())
    data['documents'][0]['text'] = 'Engraved order policy: eligibility class = ArtisanCare.'
    runtime, _ = runtime_for(data)
    answer = run(runtime, mode='multi')
    assert runtime.retriever.calls[1] == 'ArtisanCare refund eligibility'
    assert answer['abstained']


@pytest.mark.parametrize('corruption', ['generation', 'tenant', 'foreign_hit'])
def test_scope_cannot_change_between_hops(corruption):
    class ChangingRetriever(FixtureRetriever):
        async def retrieve(self, query, top_k=1):
            result = await super().retrieve(query, top_k)
            if len(self.calls) == 2:
                if corruption == 'generation':
                    result['generation_id'] = 'new-generation'
                elif corruption == 'tenant':
                    result['tenant_id'] = 2
                else:
                    result['hits'][0]['tenant_id'] = 2
            return result

    runtime, model = runtime_for(retriever=ChangingRetriever())
    with pytest.raises(ValueError):
        run(runtime, mode='multi')
    assert len(model.calls) == 1


def test_fixture_filters_foreign_policy():
    runtime, _ = runtime_for()
    assert run(runtime, mode='multi')['citations'] == [701, 702]
    ids = {p['id'] for h in runtime.last_trajectory for p in h['passages']}
    assert 704 not in ids


@pytest.mark.parametrize('citations', [[701], [702], [701, 704], [701, 701, 702]])
def test_synthesis_requires_both_hops_and_only_supplied_ids(citations):
    data = copy.deepcopy(fixture_data())
    data['synthesis']['citations'] = citations

    class BadSynthesis(TaskModelClient):
        async def complete(self, prompt, system=None, max_tokens=256):
            if system == MULTIHOP_SYNTHESIS_INSTRUCTION:
                return json.dumps(data['synthesis'])
            return await super().complete(prompt, system=system, max_tokens=max_tokens)

    runtime = RagRuntime(FixtureRetriever(), ExistingClientAdapter(BadSynthesis(mode='offline')),
                         evidence_budget=2400)
    assert run(runtime, mode='multi')['abstained']


def test_live_mode_uses_same_existing_adapter_without_network():
    delegate = TaskModelClient(mode='offline')
    live = TaskModelClient(mode='live', delegate=delegate)
    runtime = RagRuntime(FixtureRetriever(), ExistingClientAdapter(live), evidence_budget=2400)
    assert run(runtime, mode='multi')['citations'] == [701, 702]
    assert len(delegate.calls) == len(live.calls) == 2


def test_published_factory_captures_one_snapshot():
    acquisitions = []
    observed = []
    snapshot = object()

    def acquire(tenant_id):
        acquisitions.append(tenant_id)
        return snapshot

    def build(tenant_id, pinned, dense, sparse, **settings):
        class PinnedRetriever(FixtureRetriever):
            async def retrieve(self, query, top_k=1):
                observed.append(pinned(self.tenant_id))
                return await super().retrieve(query, top_k)
        return PinnedRetriever(tenant_id)

    with patch('anchor_ai.rag.published_hybrid_runtime', side_effect=build):
        runtime = published_bounded_rag_runtime(1, acquire, None, None,
            client=ExistingClientAdapter(TaskModelClient(mode='offline')))
        assert run(runtime, mode='multi')['citations'] == [701, 702]
    assert acquisitions == [1]
    assert observed == [snapshot, snapshot]


def test_endpoint_auth_fixture_scope_and_default(monkeypatch):
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'multihop-test-secret')
    monkeypatch.setenv('ANCHOR_MULTIHOP_FIXTURE', '1')
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    body = dict(tenant_id=1, conversation_id=10, message=fixture_data()['ticket'])
    headers = {'X-Anchor-Secret': 'multihop-test-secret'}
    with TestClient(app) as client:
        assert client.post('/v1/rag', json=body).status_code == 401
        single = client.post('/v1/rag', json=body, headers=headers)
        assert single.status_code == 200
        assert single.json()['abstained']
        multi = client.post('/v1/rag', json=dict(body, mode='multi'), headers=headers)
        assert multi.status_code == 200
        assert multi.json()['citations'] == [701, 702]
        foreign = client.post('/v1/rag', json=dict(body, tenant_id=2), headers=headers)
        assert foreign.status_code == 503
        monkeypatch.delenv('ANCHOR_MULTIHOP_FIXTURE')
        assert client.post('/v1/rag', json=body, headers=headers).status_code == 503
