from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .schemas import PromptMode


class OperationKind(str, Enum):
    PROMPT_GENERATION = "prompt_generation"


@dataclass(frozen=True)
class PromptGenerationOperation:
    prose: str
    target_model_name: str
    style_policy: str
    output_mode: PromptMode
    response_schema_name: str
    response_schema: dict[str, Any]
    kind: OperationKind = field(init=False, default=OperationKind.PROMPT_GENERATION)
