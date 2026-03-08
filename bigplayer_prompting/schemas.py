from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import MalformedProviderResponseError

PromptMode = Literal["simple", "split"]


class _BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _normalize_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class SimplePromptResult(_BaseSchema):
    positive_prompt: str
    negative_prompt: str
    comments: str = Field(min_length=1)


class SplitPromptResult(_BaseSchema):
    text_l_positive: str
    text_g_positive: str
    text_l_negative: str
    text_g_negative: str
    comments: str = Field(min_length=1)


def get_provider_schema(mode: PromptMode) -> dict[str, Any]:
    if mode == "simple":
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["positive_prompt", "negative_prompt", "comments"],
            "properties": {
                "positive_prompt": {"type": "string"},
                "negative_prompt": {"type": "string"},
                "comments": {"type": "string"},
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "text_l_positive",
            "text_g_positive",
            "text_l_negative",
            "text_g_negative",
            "comments",
        ],
        "properties": {
            "text_l_positive": {"type": "string"},
            "text_g_positive": {"type": "string"},
            "text_l_negative": {"type": "string"},
            "text_g_negative": {"type": "string"},
            "comments": {"type": "string"},
        },
    }


def validate_result(mode: PromptMode, payload: dict[str, Any]) -> SimplePromptResult | SplitPromptResult:
    try:
        if mode == "simple":
            return SimplePromptResult.model_validate(payload)
        return SplitPromptResult.model_validate(payload)
    except ValidationError as exc:
        raise MalformedProviderResponseError(f"Provider response failed schema validation: {exc}") from exc
