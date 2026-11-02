import copy
import inspect
import json
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator


REFUSAL = ('I searched the available support knowledge base but could not verify an answer. '
           'Please check your receipt and share the policy name or purchase details to narrow the search.')
HEDGE_PREFIX = 'I searched the available support knowledge base. '
HEDGE_SUFFIX = (' I cannot verify that this covers your situation. '
                'Check the policy conditions and your receipt before acting.')


class Citation(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    chunk_id: int = Field(gt=0)
    quoted_span: str = Field(min_length=1, max_length=4000)


class CitedAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    answer: str = Field(min_length=1, max_length=4000)
    citations: list[Citation] = Field(max_length=20)
    decision: Literal['answer', 'hedge', 'refuse']

    @model_validator(mode='after')
    def evidence(self, info: ValidationInfo):
        context = info.context or {}
        tenant = context.get('tenant_id')
        chunks = context.get('chunks', {})
        if not self.answer.strip():
            raise ValueError('empty_answer')
        if self.decision == 'refuse' and self.citations:
            raise ValueError('refusal_must_not_cite')
        if self.decision != 'refuse' and not self.citations:
            raise ValueError('support_required')
        ids = [citation.chunk_id for citation in self.citations]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate_citation')
        for citation in self.citations:
            chunk = chunks.get(citation.chunk_id)
            if chunk is None or chunk.get('tenant_id') != tenant:
                raise ValueError('unknown_chunk')
            quote = ' '.join(citation.quoted_span.split())
            if not quote or quote not in ' '.join(chunk['text'].split()):
                raise ValueError('span_absent')
        return self


CITED_FORMAT = {'type': 'json_schema', 'json_schema': {
    'name': 'anchor_cited_answer', 'strict': True,
    'schema': CitedAnswer.model_json_schema()}}


def settings():
    path = Path(__file__).resolve().parents[1] / 'config/answer_policy.yml'
    return validate_settings(yaml.safe_load(path.read_text()))


def validate_settings(cfg):
    cfg = dict(cfg)
    keys = {'refuse_top_below', 'refuse_groundedness_below',
            'answer_top_at_least', 'answer_groundedness_at_least', 'max_regenerations'}
    if set(cfg) != keys:
        raise ValueError('invalid_policy_configuration')
    for key in keys - {'max_regenerations'}:
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid_policy_threshold')
    if (type(cfg['max_regenerations']) is not int or
            not 0 <= cfg['max_regenerations'] <= 3):
        raise ValueError('invalid_retry_limit')
    if (cfg['refuse_top_below'] > cfg['answer_top_at_least'] or
            cfg['refuse_groundedness_below'] > cfg['answer_groundedness_at_least']):
        raise ValueError('unordered_thresholds')
    return cfg


def choose(chunks, top_score, groundedness, cfg):
    scores = (top_score, groundedness)
    if (not chunks or any(type(s) not in (int, float) or
            not math.isfinite(s) or not 0 <= s <= 1 for s in scores)):
        return 'refuse'
    if top_score < cfg['refuse_top_below'] or groundedness < cfg['refuse_groundedness_below']:
        return 'refuse'
    if top_score >= cfg['answer_top_at_least'] and groundedness >= cfg['answer_groundedness_at_least']:
        return 'answer'
    return 'hedge'


def refusal():
    return CitedAnswer(answer=REFUSAL, citations=[], decision='refuse')


def rejection(error):
    allowed = {'empty_answer', 'refusal_must_not_cite', 'support_required',
               'duplicate_citation', 'unknown_chunk', 'span_absent'}
    for item in error.errors(include_input=False):
        reason = item['msg'].removeprefix('Value error, ')
        if reason in allowed:
            return reason
    return 'invalid_shape'


async def resolve(value):
    return await value if inspect.isawaitable(value) else value


class AnswerPolicy:
    def __init__(self, client, scorer, cfg=None, emit=None):
        self.client = client
        self.scorer = scorer
        self.cfg = validate_settings(cfg) if cfg is not None else settings()
        self.emit = emit or (lambda event: None)

    async def generate(self, question, *, tenant_id, hits, top_score):
        # Pin a private evidence snapshot for all attempts. Never trust model metadata.
        snapshot = copy.deepcopy(hits)
        if any(hit.get('tenant_id') != tenant_id for hit in snapshot):
            self.emit({'event': 'answer_attempt', 'attempt': 0, 'reason': 'scope_rejected'})
            return refusal().model_dump()
        chunks = {hit['id']: hit for hit in snapshot}
        if len(chunks) != len(snapshot):
            return refusal().model_dump()
        if not chunks or choose(chunks, top_score, 1.0, self.cfg) == 'refuse':
            return refusal().model_dump()
        for attempt in range(self.cfg['max_regenerations'] + 1):
            try:
                raw = await resolve(self.client.complete(
                    question=question, evidence=copy.deepcopy(snapshot),
                    response_format=copy.deepcopy(CITED_FORMAT)))
                if isinstance(raw, str):
                    answer = CitedAnswer.model_validate_json(raw,
                        context={'tenant_id': tenant_id, 'chunks': chunks})
                else:
                    answer = CitedAnswer.model_validate(raw,
                        context={'tenant_id': tenant_id, 'chunks': chunks})
            except ValidationError as error:
                self.emit({'event': 'answer_attempt', 'attempt': attempt,
                           'reason': rejection(error)})
                continue
            except (TimeoutError, ConnectionError):
                self.emit({'event': 'answer_attempt', 'attempt': attempt,
                           'reason': 'generation_unavailable'})
                return refusal().model_dump()
            # Production injects Anchor's scorer for THIS generated answer.
            try:
                groundedness = await resolve(self.scorer(answer.answer, copy.deepcopy(snapshot)))
            except (TimeoutError, ConnectionError):
                groundedness = None
            decision = choose(chunks, top_score, groundedness, self.cfg)
            if answer.decision == 'refuse' or decision == 'refuse':
                result = refusal()
            else:
                text = answer.answer
                if decision == 'hedge':
                    text = HEDGE_PREFIX + text + HEDGE_SUFFIX
                result = CitedAnswer.model_validate({
                    'answer': text[:4000], 'citations': [c.model_dump() for c in answer.citations],
                    'decision': decision}, context={'tenant_id': tenant_id, 'chunks': chunks})
            self.emit({'event': 'answer_attempt', 'attempt': attempt, 'reason': 'accepted'})
            return result.model_dump()
        return refusal().model_dump()


class RecordedCitedClient:
    """Offline contract fake. Scores are supplied separately, not measured here."""
    def __init__(self, fault=None):
        self.fault = fault
        self.calls = 0

    async def complete(self, *, question, evidence, response_format):
        assert response_format == CITED_FORMAT
        self.calls += 1
        source = evidence[0]
        return {'answer': source['text'], 'decision': 'answer', 'citations': [{
            'chunk_id': 999 if self.fault == 'unknown_chunk' else source['id'],
            'quoted_span': 'Guaranteed refund in one day.' if self.fault == 'span_absent' else source['text']}]}


def answer_attachment_envelope(envelope, *, tenant_id, conversation_id):
    from anchor_ai.attachment_evidence import attachment_answer
    if (not isinstance(envelope, dict)
            or set(envelope) != {'label', 'sources'}
            or envelope['label'] != 'UNTRUSTED_EVIDENCE'):
        raise ValueError('attachment_envelope_rejected')
    return attachment_answer(copy.deepcopy(envelope['sources']),
                             tenant_id=tenant_id, conversation_id=conversation_id)
