from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InferRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: Literal["v1"]
    conversation_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=4000)
    tenant_id: int = Field(gt=0)

    @field_validator("message")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class InferResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    reply: str
    tokens: int = Field(ge=0)
