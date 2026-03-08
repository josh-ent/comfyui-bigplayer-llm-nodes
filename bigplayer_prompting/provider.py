from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .errors import MalformedProviderResponseError, ProviderError

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
XAI_PROVIDER_ID = "xAI"
XAI_MODELS = (
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-latest",
)


@dataclass(frozen=True)
class ProviderRequest:
    api_key: str
    provider_model: str
    system_prompt: str
    user_prompt: str
    schema_name: str
    schema: dict[str, Any]
    provider_base_url: str | None = None


def redact_secret(secret: str) -> str:
    if not secret:
        return "<empty>"
    return f"<redacted:{len(secret)}>"


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: str
    models: tuple[str, ...]
    default_base_url: str
    factory: Callable[[], "GrokProvider"]


class GrokProvider:
    def __init__(self, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds

    def generate_structured(self, request: ProviderRequest) -> dict[str, Any]:
        base_url = (request.provider_base_url or DEFAULT_BASE_URL).rstrip("/")
        payload = {
            "model": request.provider_model,
            "store": False,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": request.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request.user_prompt}],
                },
            ],
            "tools": [{"type": "web_search"}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.schema,
                    "strict": True,
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {request.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(f"{base_url}/responses", headers=headers, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderError("Timed out while waiting for the Grok provider response.") from exc
        except httpx.HTTPStatusError as exc:
            message = exc.response.text.strip()
            raise ProviderError(
                f"Grok provider returned HTTP {exc.response.status_code}. "
                f"Body: {message.replace(request.api_key, redact_secret(request.api_key))}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("Failed to call the Grok provider.") from exc

        body = response.json()
        text = self._extract_text(body)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedProviderResponseError("Provider returned non-JSON structured output.") from exc

        if not isinstance(decoded, dict):
            raise MalformedProviderResponseError("Provider returned a non-object structured payload.")
        return decoded

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


REGISTERED_PROVIDERS: dict[str, ProviderDefinition] = {
    XAI_PROVIDER_ID: ProviderDefinition(
        provider_id=XAI_PROVIDER_ID,
        models=XAI_MODELS,
        default_base_url=DEFAULT_BASE_URL,
        factory=GrokProvider,
    ),
}


def list_provider_ids() -> list[str]:
    return list(REGISTERED_PROVIDERS)


def list_models(provider_id: str | None = None) -> list[str]:
    if provider_id is not None:
        return list(REGISTERED_PROVIDERS[provider_id].models)

    all_models: list[str] = []
    for provider in REGISTERED_PROVIDERS.values():
        all_models.extend(provider.models)
    return all_models
