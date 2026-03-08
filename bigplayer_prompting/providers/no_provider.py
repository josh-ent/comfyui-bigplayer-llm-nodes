from __future__ import annotations

from typing import Any

from ..errors import ProviderError, UnsupportedOperationError
from ..operations import OperationKind, PromptGenerationOperation
from ..provider import ProviderConfig

NO_PROVIDER_ID = "No Provider"
NO_PROVIDER_BASE_URL = ""
NO_PROVIDER_MODELS = ("Positive", "Negative")
NO_PROVIDER_COMMENT = "Goes nowhere, does nothing"


class NoProvider:
    def invoke(self, operation: Any, config: ProviderConfig) -> dict[str, Any]:
        self._validate_model(config.provider_model)
        kind = getattr(operation, "kind", None)
        if kind != OperationKind.PROMPT_GENERATION:
            raise UnsupportedOperationError(f"No Provider does not support operation `{kind}`.")
        return self._prompt_generation(operation, config)

    def _prompt_generation(
        self,
        operation: PromptGenerationOperation,
        config: ProviderConfig,
    ) -> dict[str, Any]:
        prose = operation.prose.strip()
        if operation.output_mode == "simple":
            if config.provider_model == "Positive":
                return {
                    "positive_prompt": prose,
                    "negative_prompt": "",
                    "comments": NO_PROVIDER_COMMENT,
                }
            return {
                "positive_prompt": "",
                "negative_prompt": prose,
                "comments": NO_PROVIDER_COMMENT,
            }

        if config.provider_model == "Positive":
            return {
                "text_l_positive": prose,
                "text_g_positive": prose,
                "text_l_negative": "",
                "text_g_negative": "",
                "comments": NO_PROVIDER_COMMENT,
            }
        return {
            "text_l_positive": "",
            "text_g_positive": "",
            "text_l_negative": prose,
            "text_g_negative": prose,
            "comments": NO_PROVIDER_COMMENT,
        }

    def _validate_model(self, provider_model: str) -> None:
        if provider_model not in NO_PROVIDER_MODELS:
            raise ProviderError(f"No Provider does not support model `{provider_model}`.")
