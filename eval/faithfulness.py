import json
import os
from pathlib import Path
from typing import Protocol

from anchor_ai.pii import redact_evaluation
from anchor_ai.prompts import FAITHFULNESS_DECOMPOSE, FAITHFULNESS_VERIFY

ROOT = Path(__file__).resolve().parents[1]


class Judge(Protocol):
    def decompose(self, answer): ...
    def verify_many(self, claims, evidence): ...


class RecordedFaithfulnessClient:
    """Implements the existing LlmClient.complete boundary without a network."""
    def __init__(self):
        path = ROOT / 'anchor_ai/tests/fixtures/faithfulness_responses.json'
        self.responses = json.loads(path.read_text())
        self.calls = []

    def complete(self, prompt, system=None, max_tokens=256):
        # Only redacted inputs reach this client or its diagnostic records.
        self.calls.append(dict(prompt=prompt, system=system))
        data = json.loads(prompt)
        if system == FAITHFULNESS_DECOMPOSE:
            return json.dumps({'claims': self.responses['decompositions'][data['answer']]})
        if system != FAITHFULNESS_VERIFY:
            raise ValueError('unknown_judge_task')
        return json.dumps({'verdicts': [dict(claim_id=i,
            **self.responses['verifications'][claim])
            for i, claim in enumerate(data['claims'])]})


class LLMJudge:
    def __init__(self, client):
        self.client = client

    def request(self, system, data):
        prompt = json.dumps(redact_evaluation(data), sort_keys=True)
        raw = self.client.complete(prompt, system=system, max_tokens=2048)
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError('invalid_judge_object')
        return redact_evaluation(result)

    def decompose(self, answer):
        result = self.request(FAITHFULNESS_DECOMPOSE, {'answer': answer})
        claims = result.get('claims')
        if (set(result) != {'claims'} or not isinstance(claims, list)
                or any(not isinstance(c, str) or not c.strip() for c in claims)):
            raise ValueError('invalid_claims')
        return claims

    def verify_many(self, claims, evidence):
        result = self.request(FAITHFULNESS_VERIFY,
                              {'claims': claims, 'evidence': evidence})
        if set(result) != {'verdicts'}:
            raise ValueError('invalid_verdict_object')
        return result['verdicts']

    def verify(self, claim, evidence):
        return self.verify_many([claim], evidence)[0]


def make_judge():
    mode = os.getenv('ANCHOR_FAITHFULNESS_JUDGE', 'recorded')
    if mode == 'recorded':
        return LLMJudge(RecordedFaithfulnessClient())
    if mode == 'live':
        from anchor_ai.llm_client import LlmClient
        return LLMJudge(LlmClient())
    raise ValueError('unknown_faithfulness_judge')


def valid_ids(value):
    return isinstance(value, list) and all(type(i) is int for i in value)


def score_payload(payload, judge):
    payload = redact_evaluation(payload)
    try:
        answer = payload['answer']
        evidence = payload['evidence']
        citations = payload['citations']
        if not isinstance(answer, str) or not isinstance(evidence, list) or not valid_ids(citations):
            raise ValueError('invalid_payload')
        ids = set()
        for chunk in evidence:
            if (type(chunk['id']) is not int or chunk['id'] in ids
                    or not isinstance(chunk['text'], str)
                    or chunk['tenant_id'] != payload['tenant_id']
                    or chunk['generation_id'] != payload['generation_id']):
                raise ValueError('invalid_evidence_scope')
            ids.add(chunk['id'])
        claims = judge.decompose(answer)
        if (not isinstance(claims, list)
                or any(not isinstance(c, str) or not c.strip() for c in claims)):
            raise ValueError('invalid_claims')
        invalid_citations = sorted(set(citations) - ids)
        base = dict(tenant_id=payload['tenant_id'], generation_id=payload['generation_id'],
                    evidence_ids=sorted(ids), invalid_citations=invalid_citations)
        if not claims:
            return redact_evaluation(dict(base, status='no_claims', score=None,
                                          claims=[], unsupported_claims=[]))
        verdicts = judge.verify_many(redact_evaluation(claims), evidence)
        if not isinstance(verdicts, list) or len(verdicts) != len(claims):
            raise ValueError('invalid_verdict_count')
        indexed = {}
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                raise ValueError('invalid_verdict')
            i = verdict.get('claim_id')
            supporting = verdict.get('evidence_ids')
            if (type(i) is not int or i not in range(len(claims)) or i in indexed
                    or type(verdict.get('supported')) is not bool
                    or not valid_ids(supporting)
                    or not isinstance(verdict.get('reason'), str)):
                raise ValueError('invalid_verdict')
            if set(supporting) - ids:
                raise ValueError('invented_judge_evidence')
            if verdict['supported'] and not supporting:
                raise ValueError('missing_judge_evidence')
            indexed[i] = verdict
        rows = []
        for i, claim in enumerate(claims):
            verdict = indexed[i]
            citation_ok = bool(set(verdict['evidence_ids']) & set(citations))
            supported = verdict['supported'] and citation_ok and not invalid_citations
            reason = verdict['reason']
            if invalid_citations:
                reason = 'Answer cites identifiers absent from retrieved evidence.'
            elif verdict['supported'] and not citation_ok:
                reason = 'Answer does not cite the supporting retrieved evidence.'
            rows.append(dict(claim_id=i, claim=claim, supported=bool(supported),
                             evidence_ids=verdict['evidence_ids'], reason=reason))
        return redact_evaluation(dict(base, status='scored',
            score=sum(row['supported'] for row in rows) / len(rows), claims=rows,
            unsupported_claims=[row['claim'] for row in rows if not row['supported']]))
    except Exception:
        # Never persist raw provider errors: they may echo sensitive inputs.
        return dict(status='evaluation_error', score=None, claims=[],
                    unsupported_claims=[], error='invalid_input_or_judge_output')


def persist_report(path, report):
    Path(path).write_text(json.dumps(redact_evaluation(report), indent=2) + '\n')
