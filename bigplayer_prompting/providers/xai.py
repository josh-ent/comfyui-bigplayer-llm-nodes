from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from ..errors import MalformedProviderResponseError, ProviderError, UnsupportedOperationError
from ..operations import OperationKind, PromptGenerationOperation
from ..provider import ProviderConfig, redact_secret

XAI_PROVIDER_ID = "xAI"
XAI_PROVIDER_BASE_URL = "https://api.x.ai/v1"
XAI_MODELS = (
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-latest",
)
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class RenderedXAIRequest:
    model: str
    system_prompt: str
    user_prompt: str
    schema_name: str
    schema: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "store": False,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": self.user_prompt}],
                },
            ],
            "tools": [{"type": "web_search"}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "schema": self.schema,
                    "strict": True,
                }
            },
        }


class XAIProvider:
    def __init__(self, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def invoke(self, operation: Any, config: ProviderConfig) -> dict[str, Any]:
        request = self.render_operation(operation, config)
        base_url = (config.provider_base_url or XAI_PROVIDER_BASE_URL).rstrip("/")
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(f"{base_url}/responses", headers=headers, json=request.as_payload())
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderError("Timed out while waiting for the xAI provider response.") from exc
        except httpx.HTTPStatusError as exc:
            message = exc.response.text.strip()
            raise ProviderError(
                f"xAI provider returned HTTP {exc.response.status_code}. "
                f"Body: {message.replace(config.api_key, redact_secret(config.api_key))}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("Failed to call the xAI provider.") from exc

        body = response.json()
        text = self._extract_text(body)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedProviderResponseError("Provider returned non-JSON structured output.") from exc

        if not isinstance(decoded, dict):
            raise MalformedProviderResponseError("Provider returned a non-object structured payload.")
        return decoded

    def render_operation(self, operation: Any, config: ProviderConfig) -> RenderedXAIRequest:
        self._validate_model(config.provider_model)
        kind = getattr(operation, "kind", None)
        if kind == OperationKind.PROMPT_GENERATION:
            return self._render_prompt_generation(operation, config)
        raise UnsupportedOperationError(f"xAI provider does not support operation `{kind}`.")

    def _render_prompt_generation(
        self,
        operation: PromptGenerationOperation,
        config: ProviderConfig,
    ) -> RenderedXAIRequest:
        if operation.output_mode == "simple":
            output_instructions = "Return `positive_prompt`, `negative_prompt`, and `comments`."
        else:
            output_instructions = (
                "Return `text_l_positive`, `text_g_positive`, `text_l_negative`, "
                "`text_g_negative`, and `comments`. Do not duplicate the full prompt into both "
                "positive channels unless you explicitly treat that as a fallback and say so in comments."
            )

        style_section = operation.style_policy.strip() or "No extra style policy supplied."
        system_prompt = """You convert user prose into production-ready ComfyUI prompts.

You are not allowed to invent deterministic local logic that the node does not have.
Return only the schema requested by the caller.
Use the provided model name to tailor the prompt. When useful, use web search to look up
that model on CivitAI or HuggingFace before deciding on prompt shape.
Keep prompt text practical and immediately usable in image or video workflows.
Comments must explain how the model name influenced the result and note any fallback.
"""
        user_prompt = f"""User prose:
{operation.prose.strip()}

Target model name:
{operation.target_model_name}

Style or policy guidance:
{style_section}

Output mode:
{operation.output_mode}

Requirements:
- Tailor the prompt to the target model name.
- Keep the output concise, specific, and workflow-ready.
- Negative prompts should omit content instead of explaining policy.
- {output_instructions}
- Return only data matching the requested schema.
"""

        return RenderedXAIRequest(
            model=config.provider_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=operation.response_schema_name,
            schema=operation.response_schema,
        )

    def _extract_text(self, body: dict[str, Any]) -> str:
        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        for item in body.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        raise MalformedProviderResponseError("Provider response did not contain structured output text.")

    def _validate_model(self, provider_model: str) -> None:
        if provider_model not in XAI_MODELS:
            raise ProviderError(f"xAI provider does not support model `{provider_model}`.")
