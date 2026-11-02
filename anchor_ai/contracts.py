from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator, model_serializer


@dataclass(frozen=True)
class VerifiedExecution:
    """Returned by the trusted identity verifier, never deserialized from a ticket."""
    tenant_id: int
    conversation_id: int
    principal_id: str
    role: str


class AgentRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    run_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,80}$')
    operation: Literal['start', 'resume', 'approve', 'reject'] = 'start'
    ticket_type: Literal['single-lookup', 'multi-step refund'] = 'multi-step refund'
    message: str = Field(default='', max_length=1800)
    order_id: str = Field(default='A104', pattern=r'^[A-Z][0-9]{3}$')
    amount_cents: int = Field(default=300, gt=0, le=1000000)


ReasoningEffort = Literal['off', 'low', 'high']

class PromptContract(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    version: str = Field(pattern=r'^chat:v[1-9][0-9]*$')
    template: str = Field(min_length=1)
    model_id: str = Field(pattern=r'^[A-Za-z0-9.-]+-\d{4}-\d{2}-\d{2}$')
    temperature: float = Field(ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(gt=0)
    @field_validator('template')
    @classmethod
    def one_placeholder(cls, value):
        if value.count('{{ticket}}') != 1:
            raise ValueError('one ticket placeholder required')
        return value

class ChatMessage(str):
    """Internal string-compatible input; never stores contract in global state."""
    def __new__(cls, text, contract):
        value = super().__new__(cls, text)
        value.prompt_contract = contract
        return value

class HistoryTurn(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    position: int = Field(gt=0)
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=100000)

class SummaryState(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    summary: str = Field(default='', max_length=100000)
    prompt_version: str | None = Field(default=None, pattern=r'^summary:v[1-9][0-9]*$')
    folded_position: int = Field(default=0, ge=0)

class InferRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    version: Literal['v1']
    conversation_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=4000)
    tenant_id: int = Field(gt=0)
    prompt_contract: PromptContract | None = None
    history: list[HistoryTurn] | None = Field(default=None, max_length=1000)
    summary_state: SummaryState | None = None
    @field_validator('message')
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('message must not be blank')
        return value
    @model_validator(mode='after')
    def attach_chat_contract(self):
        if self.prompt_contract is not None:
            self.message = ChatMessage(self.message, self.prompt_contract)
        if self.history is not None or self.summary_state is not None:
            if self.prompt_contract is None:
                raise ValueError('context requires a pinned chat contract')
            positions = [row.position for row in self.history or []]
            if positions != sorted(set(positions)):
                raise ValueError('history positions must be strictly increasing')
            try:
                from .context import attach_context
            except ImportError:
                from context import attach_context
            attach_context(self)
        return self

class PromptDetails(BaseModel):
    model_config = ConfigDict(strict=True, extra='ignore')
    cached_tokens: int = Field(default=0, ge=0)

class CompletionDetails(BaseModel):
    model_config = ConfigDict(strict=True, extra='ignore')
    reasoning_tokens: int | None = Field(default=None, ge=0)

class Usage(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    prompt_tokens_details: PromptDetails | None = None
    completion_tokens_details: CompletionDetails | None = None
    @model_validator(mode='after')
    def valid_subsets(self):
        if self.prompt_tokens_details and self.prompt_tokens_details.cached_tokens > self.prompt_tokens:
            raise ValueError('invalid cached usage')
        reasoning = self.completion_tokens_details.reasoning_tokens if self.completion_tokens_details else None
        if reasoning is not None and reasoning > self.completion_tokens:
            raise ValueError('invalid reasoning usage')
        return self

class InferResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    reply: str
    tokens: int = Field(ge=0)
    usage: Usage
    finish_reason: str
    attempts: int = Field(ge=1)
    context_state: SummaryState | None = None
    @model_serializer(mode='wrap')
    def omit_absent_context(self, handler):
        result = handler(self)
        if self.context_state is None:
            result.pop('context_state', None)
        return result

class TriageLabels(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    category: Literal['billing', 'technical', 'account', 'order']
    urgency: Literal['P1', 'P2', 'P3']

# schemas.py inherits TriageLabels for strict requests and local output validation.
class ToolScope(BaseModel):
    """Construct only from authenticated application context, not model input."""
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    tenant_id: int = Field(gt=0)
    conversation_id: int = Field(gt=0)
    role: str = Field(default='support_reader', min_length=1)
    _execution: dict = PrivateAttr(default_factory=dict)

class ToolCall(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    call_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    arguments: str | dict

class RefundArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    order_id: str = Field(pattern=r'^[A-Z][0-9]{3}$')
    amount_cents: int = Field(gt=0, le=1000000)

class IssuedRefundResult(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    order_id: str = Field(pattern=r'^[A-Z][0-9]{3}$')
    amount_cents: int = Field(gt=0)
    status: Literal['refunded']

class ReviewIdentity(BaseModel):
    """Trusted service-boundary identity, never populated from LLM output."""
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    tenant_id: int = Field(gt=0)
    reviewer_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,80}$')
    role: Literal['refund_reviewer', 'support_reader']


class OrchestrationFlow(BaseModel):
    """Internal routing snapshot; never accepted from a customer request."""
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    pattern: Literal['single-agent', 'supervisor-worker']
    workers: dict[str, str] = Field(default_factory=dict)


class DelegatedTask(BaseModel):
    """Internal A2A-inspired envelope, not a public A2A wire implementation."""
    model_config = ConfigDict(strict=True, extra='forbid')
    protocol: Literal['anchor-a2a-loopback-v1']
    task_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,80}$')
    tenant_id: int = Field(gt=0)
    conversation_id: int = Field(gt=0)
    skill: Literal['refund-resolution']
    request: AgentRequest

    @model_validator(mode='after')
    def bind_run(self):
        if self.request.run_id != self.task_id:
            raise ValueError('task_run_mismatch')
        if self.request.operation in ('approve', 'reject'):
            raise ValueError('delegation_cannot_review')
        return self


class DelegatedResult(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    protocol: Literal['anchor-a2a-loopback-v1'] = 'anchor-a2a-loopback-v1'
    task_id: str
    tenant_id: int
    conversation_id: int
    trace_id: str
    status: str
    handoff: dict | None


HANDOFF_KEYS = ('tenant_id', 'conversation_id', 'order_id', 'amount_cents',
                'provenance', 'messages', 'evidence')


class HandoffViolation(ValueError):
    """Machine-readable diagnostics; never contains ticket text or identifiers."""
    def __init__(self, reason, keys=()):
        self.reason = reason
        self.keys = tuple(keys)
        super().__init__(reason + ':' + ','.join(self.keys))


def validate_specialist_handoff(context, trusted, *, boundary='dispatch'):
    """Trusted is application-held state, not another field in model output.

    Revalidate after persistence and on resume. Never fill missing fields from
    defaults, conversation prose, provider response IDs, or model guesses.
    """
    import copy
    import re
    if boundary not in ('dispatch', 'resume'):
        raise HandoffViolation('invalid_handoff_boundary')
    if not isinstance(context, dict) or not isinstance(trusted, dict):
        raise HandoffViolation('handoff_missing')
    for value in (trusted, context):
        missing = [key for key in HANDOFF_KEYS if key not in value]
        if missing:
            raise HandoffViolation('handoff_missing', missing)
        for key in ('tenant_id', 'conversation_id', 'amount_cents'):
            if type(value[key]) is not int or value[key] <= 0:
                raise HandoffViolation('handoff_invalid', [key])
        if (not isinstance(value['order_id'], str)
                or not re.fullmatch(r'[A-Z][A-Z0-9]{1,30}', value['order_id'])):
            raise HandoffViolation('handoff_invalid', ['order_id'])
        if not isinstance(value['provenance'], str) or not value['provenance'].strip():
            raise HandoffViolation('handoff_invalid', ['provenance'])
        for key in ('messages', 'evidence'):
            if (not isinstance(value[key], list) or not value[key]
                    or not all(isinstance(v, str) and v.strip() for v in value[key])):
                raise HandoffViolation('handoff_invalid', [key])
    changed = [key for key in HANDOFF_KEYS if context[key] != trusted[key]]
    if changed:
        raise HandoffViolation('handoff_conflict', changed)
    return copy.deepcopy(context)


class VisionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    image_base64: str = Field(min_length=1, max_length=2800000)
    ticket_text: str = Field(min_length=1, max_length=1800)


class AttachmentScope(BaseModel):
    """Shape of independently authenticated ticket identity, not credentials."""
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True)
    tenant_id: int = Field(gt=0)
    conversation_id: int = Field(gt=0)
