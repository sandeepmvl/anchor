import copy
import json

import pytest

from eval.faithfulness import (
    LLMJudge, RecordedFaithfulnessClient, persist_report, score_payload,
)
from eval.groundedness_scorer import evaluate_cases, load_dataset, run_gate
from anchor_ai.prompts import FAITHFULNESS_DECOMPOSE


def payload(answer='Refunds are available within 30 days.'):
    data = load_dataset()
    return dict(answer=answer, citations=[101], evidence=copy.deepcopy(data['evidence']),
                tenant_id=data['tenant_id'], generation_id=data['generation_id'])


def judge():
    return LLMJudge(RecordedFaithfulnessClient())


def test_recorded_claim_decomposition_flags_invented_promise():
    report = evaluate_cases(judge=judge())
    rows = {r['case']: r for r in report['reports']}
    assert rows['grounded']['score'] == 1.0
    bad = rows['invented_promise']
    assert bad['score'] == 0.5
    assert len(bad['claims']) == 2
    assert bad['unsupported_claims'] == ['Shipping insurance is included.']
    assert rows['contradiction']['score'] == 0.0
    assert rows['paraphrase']['score'] == 1.0


def test_answer_citation_must_have_been_retrieved():
    data = payload()
    data['citations'] = [999]
    result = score_payload(data, judge())
    assert result['status'] == 'scored'
    assert result['score'] == 0.0
    assert result['invalid_citations'] == [999]
    assert result['evidence_ids'] == [101]
    assert result['unsupported_claims'] == [data['answer']]


def test_missing_citation_does_not_earn_support():
    data = payload()
    data['citations'] = []
    assert score_payload(data, judge())['score'] == 0.0


def test_empty_claims_are_not_perfect():
    result = score_payload(payload(''), judge())
    assert result['status'] == 'no_claims'
    assert result['score'] is None
    assert result['claims'] == []


class OutputClient:
    def __init__(self, verification):
        self.verification = verification

    def complete(self, prompt, system=None, max_tokens=256):
        if system == FAITHFULNESS_DECOMPOSE:
            return '{"claims":["Refunds are available within 30 days."]}'
        return self.verification


@pytest.mark.parametrize('verification', [
    'not json',
    '{"verdicts":[]}',
    '{"verdicts":[{"claim_id":0,"supported":"true","evidence_ids":[101],"reason":"x"}]}',
    '{"verdicts":[{"claim_id":7,"supported":true,"evidence_ids":[101],"reason":"x"}]}',
    '{"verdicts":[{"claim_id":0,"supported":true,"evidence_ids":[999],"reason":"x"}]}',
    '{"verdicts":[{"claim_id":0,"supported":true,"evidence_ids":[],"reason":"x"}]}',
])
def test_malformed_verification_has_no_score(verification):
    result = score_payload(payload(), LLMJudge(OutputClient(verification)))
    assert result['status'] == 'evaluation_error'
    assert result['score'] is None


def test_malformed_decomposition_has_no_score():
    class BadClient:
        def complete(self, *args, **kwargs):
            return '{"claims":[42]}'
    assert score_payload(payload(), LLMJudge(BadClient()))['status'] == 'evaluation_error'


@pytest.mark.parametrize('field', ['tenant_id', 'generation_id'])
def test_wrong_snapshot_is_rejected_before_judge(field):
    data = payload()
    data['evidence'][0][field] = 'other'
    client = RecordedFaithfulnessClient()
    assert score_payload(data, LLMJudge(client))['status'] == 'evaluation_error'
    assert client.calls == []


def test_sensitive_text_redacted_before_both_calls_and_persistence(tmp_path):
    secret = 'customer@example.com'
    class SpyClient:
        def __init__(self):
            self.calls = []

        def complete(self, prompt, system=None, max_tokens=256):
            self.calls.append(prompt)
            assert secret not in prompt
            if system == FAITHFULNESS_DECOMPOSE:
                return json.dumps({'claims': ['Contact ' + secret]})
            return json.dumps({'verdicts': [dict(claim_id=0, supported=False,
                evidence_ids=[], reason='Cannot verify ' + secret)]})
    data = payload('Contact ' + secret)
    data['evidence'][0]['text'] += ' Contact ' + secret
    client = SpyClient()
    result = score_payload(data, LLMJudge(client))
    assert len(client.calls) == 2
    assert result['status'] == 'scored'
    assert secret not in json.dumps(result)
    path = tmp_path / 'report.json'
    persist_report(path, dict(result=result, extra=secret))
    assert secret not in path.read_text()


def test_offline_gate():
    assert run_gate()['dataset_version'] == 'groundedness-v1'
