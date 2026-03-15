from __future__ import annotations

import json
from typing import Any

from ..generation.capabilities import (
    BASIC_PROMPT_CAPABILITY,
    CHECKPOINT_PICKER_CAPABILITY,
    KSAMPLER_CONFIG_CAPABILITY,
    SPLIT_PROMPT_CAPABILITY,
)
from ..errors import ProviderError, UnsupportedOperationError
from ..generation.operations import OperationKind, PromptGenerationOperation
from .base import InvocationContext, ProviderConfig

NO_PROVIDER_ID = "No Provider"
NO_PROVIDER_BASE_URL = ""
NO_PROVIDER_MODELS = ("Positive", "Negative")
NO_PROVIDER_COMMENT = "Goes nowhere, does nothing"


class NoProvider:
    def invoke(
        self,
        operation: Any,
        config: ProviderConfig,
        context: InvocationContext | None = None,
    ) -> dict[str, Any]:
        context = context or InvocationContext()
        context.report_status("Running local No Provider passthrough.")
        self._validate_model(config.provider_model)
        kind = getattr(operation, "kind", None)
        if kind != OperationKind.PROMPT_GENERATION:
            raise UnsupportedOperationError(f"No Provider does not support operation `{kind}`.")
        context.set_request_text(self._render_request_text(operation, config))
        output = self._prompt_generation(operation, config)
        context.set_response_text(json.dumps(output, sort_keys=True, indent=2, ensure_ascii=True))
        return output

    def _prompt_generation(
        self,
        operation: PromptGenerationOperation,
        config: ProviderConfig,
    ) -> dict[str, Any]:
        prose = operation.prose.strip()
        output: dict[str, Any] = {}

        if BASIC_PROMPT_CAPABILITY in operation.requested_capabilities:
            if config.provider_model == "Positive":
                output[BASIC_PROMPT_CAPABILITY] = {
                    "positive_prompt": prose,
                    "negative_prompt": "",
                    "comments": NO_PROVIDER_COMMENT,
                }
            else:
                output[BASIC_PROMPT_CAPABILITY] = {
                    "positive_prompt": "",
                    "negative_prompt": prose,
                    "comments": NO_PROVIDER_COMMENT,
                }

        if SPLIT_PROMPT_CAPABILITY in operation.requested_capabilities:
            if config.provider_model == "Positive":
                output[SPLIT_PROMPT_CAPABILITY] = {
                    "text_l_positive": prose,
                    "text_g_positive": prose,
                    "text_l_negative": "",
                    "text_g_negative": "",
                    "comments": NO_PROVIDER_COMMENT,
                }
            else:
                output[SPLIT_PROMPT_CAPABILITY] = {
                    "text_l_positive": "",
                    "text_g_positive": "",
                    "text_l_negative": prose,
                    "text_g_negative": prose,
                    "comments": NO_PROVIDER_COMMENT,
                }

        if KSAMPLER_CONFIG_CAPABILITY in operation.requested_capabilities:
            sampler_config = operation.capability_configs[KSAMPLER_CONFIG_CAPABILITY]
            output[KSAMPLER_CONFIG_CAPABILITY] = {
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": sampler_config["sampler_names"][0],
                "scheduler": sampler_config["scheduler_names"][0],
                "denoise": 1.0,
                "comments": NO_PROVIDER_COMMENT,
            }

        if CHECKPOINT_PICKER_CAPABILITY in operation.requested_capabilities:
            checkpoint_config = operation.capability_configs[CHECKPOINT_PICKER_CAPABILITY]
            output[CHECKPOINT_PICKER_CAPABILITY] = {
                "checkpoint_name": checkpoint_config["available_checkpoints"][0],
                "comments": NO_PROVIDER_COMMENT,
            }

        return output

    def _validate_model(self, provider_model: str) -> None:
        if provider_model not in NO_PROVIDER_MODELS:
            raise ProviderError(f"No Provider does not support model `{provider_model}`.")

    def _render_request_text(self, operation: PromptGenerationOperation, config: ProviderConfig) -> str:
        sections = [
            f"Provider model: {config.provider_model}",
            f"User prose:\n{operation.prose.strip()}",
            f"Requested output capabilities:\n{', '.join(operation.requested_capabilities)}",
        ]
        if operation.context_blocks:
            context_lines = [f"{title}:\n{body.strip()}" for title, body in operation.context_blocks]
            sections.append("Context blocks:\n" + "\n\n".join(context_lines))
        return "\n\n".join(section for section in sections if section.strip())
