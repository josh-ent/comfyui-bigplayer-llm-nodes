from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperationKind(str, Enum):
    PROMPT_GENERATION = "prompt_generation"


@dataclass(frozen=True)
class PromptGenerationOperation:
    prose: str
    context_blocks: tuple[tuple[str, str], ...]
    capability_instructions: tuple[str, ...]
    requested_capabilities: tuple[str, ...]
    capability_configs: dict[str, dict[str, Any]]
    response_schema_name: str
    response_schema: dict[str, Any]
    kind: OperationKind = field(init=False, default=OperationKind.PROMPT_GENERATION)
