from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class InferRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    version: Literal['v1']
    conversation_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=4000)
    tenant_id: int = Field(gt=0)
    prompt_contract: PromptContract | None = None

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


class TriageLabels(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    category: Literal['billing', 'technical', 'account', 'order']
    urgency: Literal['P1', 'P2', 'P3']


# schemas.py inherits TriageLabels for strict requests and local output validation.
