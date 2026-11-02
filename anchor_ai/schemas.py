import json
import math

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

try:
    from .contracts import TriageLabels as ExistingTriageLabels
except ImportError:
    from contracts import TriageLabels as ExistingTriageLabels


class TriageLabels(ExistingTriageLabels):
    model_config = ConfigDict(strict=True, extra='forbid')


TRIAGE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'ticket_triage', 'strict': True,
        'schema': TriageLabels.model_json_schema(),
    },
}


class InvalidModelOutput(Exception):
    def __init__(self, reason, errors=None):
        self.reason = reason
        self.errors = errors
        super().__init__('invalid_model_output')

    def detail(self):
        result = {'code': 'invalid_model_output', 'reason': self.reason}
        if self.errors is not None:
            result['errors'] = self.errors
        return result


def validate_triage(raw):
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        raise InvalidModelOutput('invalid_json') from None
    try:
        return TriageLabels.model_validate(decoded)
    except ValidationError as error:
        safe = [
            {'loc': [e['loc'][0] if e['loc'] and
                     e['loc'][0] in ('category', 'urgency') else '$'],
             'type': e['type']}
            for e in error.errors()
        ]
        raise InvalidModelOutput('schema_violation', safe) from None


class ChunkEmbedRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    tenant_id: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=1000000)


class IngestionNode(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chunk_index: int = Field(ge=0)
    parent_index: int | None
    is_leaf: bool
    text: str
    window_text: str | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_revision: str | None = None
    embedding_dimensions: int | None = None

    @model_validator(mode='after')
    def payload(self):
        metadata = (self.embedding_model, self.embedding_revision, self.embedding_dimensions)
        if self.is_leaf:
            if self.parent_index is None or not self.window_text or not all(metadata):
                raise ValueError('invalid_leaf')
            if self.embedding is None or len(self.embedding) != self.embedding_dimensions:
                raise ValueError('invalid_embedding_dimension')
            if not all(math.isfinite(value) for value in self.embedding):
                raise ValueError('invalid_embedding_number')
        elif self.parent_index is not None or self.embedding is not None or any(metadata):
            raise ValueError('invalid_parent')
        return self


class DenseRagRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    tenant_id: int = Field(gt=0)
    conversation_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=1800)


class DenseRagResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    reply: str = Field(min_length=1, max_length=4000)
    citations: list[int]
    abstained: bool


class SparseHit(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    id: int = Field(gt=0)
    tenant_id: int = Field(gt=0)
    text: str
    score: float = Field(allow_inf_nan=False)


class HybridRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    tenant_id: int = Field(gt=0)
    conversation_id: int = Field(gt=0)
    query: str = Field(min_length=1, max_length=1800)
    top_k: int = Field(default=3, ge=1, le=20)


class HybridHit(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    id: int = Field(gt=0)
    tenant_id: int = Field(gt=0)
    text: str
    score: float = Field(ge=0, allow_inf_nan=False)
    dense: float = Field(ge=0, allow_inf_nan=False)
    sparse: float = Field(ge=0, allow_inf_nan=False)


class HybridResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    tenant_id: int = Field(gt=0)
    generation_id: str = Field(min_length=1)
    hits: list[HybridHit]


class QueryPlanPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    queries: list[str] = Field(min_length=1, max_length=3)

    @model_validator(mode='after')
    def query_lengths(self):
        if any(not q.strip() or len(q.encode('utf-8')) > 300 for q in self.queries):
            raise ValueError('invalid_subquery')
        return self


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    version: str
    queries: list[str]
    fallback: bool
    reason: str


class AnalyzedResponse(HybridResponse):
    plan: RetrievalPlan


from typing import Literal


class RagRequest(DenseRagRequest):
    mode: Literal['single', 'multi'] = 'single'
    top_k: int = Field(default=1, ge=1, le=20)


class MultiHopPlan(QueryPlanPayload):
    @model_validator(mode='after')
    def dependency(self):
        if len(self.queries) != 2:
            raise ValueError('expected_two_slots')
        first, second = self.queries
        if '{' in first or '}' in first:
            raise ValueError('invalid_first_slot')
        if second.count('{eligibility_class}') != 1:
            raise ValueError('missing_class_binding')
        remainder = second.replace('{eligibility_class}', '')
        if '{' in remainder or '}' in remainder:
            raise ValueError('unsupported_binding')
        return self
