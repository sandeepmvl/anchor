import asyncio
import copy

import pytest
from pydantic import ValidationError

from anchor_ai.answer_policy import AnswerPolicy, CitedAnswer, RecordedCitedClient, choose, settings


HITS = [{'id': 31, 'tenant_id': 1, 'text': 'Refunds require review.', 'score': .92}]
QUESTION = 'When will my refund arrive?'


def run(score=0.93, top=0.92, hits=None, fault=None, cfg=None):
    events = []
    client = RecordedCitedClient(fault)
    policy = AnswerPolicy(client, lambda answer, evidence: score, cfg=cfg, emit=events.append)
    result = asyncio.run(policy.generate(QUESTION, tenant_id=1,
        hits=copy.deepcopy(HITS if hits is None else hits), top_score=top))
    return result, events, client


def test_existing_signals_choose_all_three_states():
    cfg = settings()
    assert choose(HITS, .8, .85, cfg) == 'answer'
    assert choose(HITS, .82, .70, cfg) == 'hedge'
    assert choose(HITS, .39, .99, cfg) == 'refuse'
    assert choose(HITS, .9, .49, cfg) == 'refuse'
    assert choose([], 1, 1, cfg) == 'refuse'
    assert choose(HITS, float('nan'), .9, cfg) == 'refuse'
    assert run()[0]['decision'] == 'answer'
    assert run(score=.7)[0]['decision'] == 'hedge'
    assert run(hits=[])[0]['decision'] == 'refuse'
    assert run(score=None)[0]['decision'] == 'refuse'


def test_citation_membership_tenant_and_span_are_hard_checks():
    raw = {'answer': 'Refunds require review.', 'decision': 'answer',
           'citations': [{'chunk_id': 31, 'quoted_span': 'Refunds require review.'}]}
    context = {'tenant_id': 1, 'chunks': {31: HITS[0]}}
    assert CitedAnswer.model_validate(raw, context=context)
    for citation in ({'chunk_id': 99, 'quoted_span': 'Refunds require review.'},
                     {'chunk_id': 31, 'quoted_span': 'Guaranteed tomorrow.'}):
        with pytest.raises(ValidationError):
            CitedAnswer.model_validate(dict(raw, citations=[citation]), context=context)
    with pytest.raises(ValidationError):
        CitedAnswer.model_validate(raw, context={'tenant_id': 2, 'chunks': {31: HITS[0]}})
    foreign = [dict(HITS[0], tenant_id=2)]
    result, events, client = run(hits=foreign)
    assert result['decision'] == 'refuse' and client.calls == 0
    assert events[0]['reason'] == 'scope_rejected'


def test_retries_are_bounded_and_customer_never_gets_validation_error():
    for fault in ('unknown_chunk', 'span_absent'):
        result, events, client = run(fault=fault)
        assert client.calls == 2
        assert [event['reason'] for event in events] == [fault, fault]
        assert result['decision'] == 'refuse' and result['citations'] == []
        assert set(result) == {'answer', 'citations', 'decision'}
        assert all(set(event) == {'event', 'attempt', 'reason'} for event in events)


def test_repair_uses_same_evidence_and_scores_the_generated_text():
    class Repair(RecordedCitedClient):
        async def complete(self, **kwargs):
            assert kwargs['evidence'] == HITS
            self.fault = 'span_absent' if self.calls == 0 else None
            return await super().complete(**kwargs)
    seen = []
    client = Repair()
    def scorer(answer, evidence):
        seen.append((answer, evidence))
        return .7
    result = asyncio.run(AnswerPolicy(client, scorer).generate(
        QUESTION, tenant_id=1, hits=HITS, top_score=.92))
    assert client.calls == 2
    assert result['decision'] == 'hedge'
    assert seen == [('Refunds require review.', HITS)]
    assert HITS[0]['text'] == 'Refunds require review.'


def test_configuration_is_bounded():
    for value in (-1, True, 4):
        with pytest.raises(ValueError):
            AnswerPolicy(RecordedCitedClient(), lambda *_: .9,
                cfg=dict(settings(), max_regenerations=value))
    cfg = dict(settings(), max_regenerations=0)
    assert run(fault='unknown_chunk', cfg=cfg)[2].calls == 1


def test_cited_runtime_uses_retrieval_score_and_preserves_metadata():
    from anchor_ai.rag import CitedRagRuntime
    class Retriever:
        async def retrieve(self, question, top_k):
            assert question == QUESTION
            return dict(tenant_id=1, hits=copy.deepcopy(HITS))
    runtime = CitedRagRuntime(1, Retriever(),
        AnswerPolicy(RecordedCitedClient(), lambda *_: .7), 'refund-ux:v1')
    result = asyncio.run(runtime.answer(QUESTION))
    assert result['cited_answer']['decision'] == 'hedge'
    assert result['retrieved_chunk_ids'] == [31]
    assert result['prompt_version'] == 'refund-ux:v1'


def test_endpoint_requires_auth_and_configured_runtime(monkeypatch):
    from fastapi.testclient import TestClient
    from anchor_ai.app import app
    monkeypatch.setenv('ANCHOR_AI_SECRET', 'offline-test-secret')
    monkeypatch.delattr(app.state, 'cited_rag_runtime_factory', raising=False)
    client = TestClient(app)
    body = dict(tenant_id=1, conversation_id=1, message=QUESTION)
    assert client.post('/v1/rag/cited', json=body).status_code == 401
    response = client.post('/v1/rag/cited', json=body,
                           headers={'X-Anchor-Secret': 'offline-test-secret'})
    assert response.status_code == 503
    assert response.json()['detail'] == 'cited_generation_not_configured'


if __name__ == '__main__':
    supported, _, _ = run()
    weak, _, _ = run(score=.7)
    missing, _, _ = run(hits=[])
    print('Same refund question:', QUESTION)
    print('Supported:', supported['answer'], supported['citations'])
    print('Weak evidence:', weak['decision'], '—', weak['answer'])
    print('No supporting KB:', missing['decision'], '—', missing['answer'])
    print('Offline contract replay only: model quality and hosted latency NOT verified.')
