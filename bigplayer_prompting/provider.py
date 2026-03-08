from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


def redact_secret(secret: str) -> str:
    if not secret:
        return "<empty>"
    return f"<redacted:{len(secret)}>"


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: str
    models: tuple[str, ...]
    default_base_url: str
    factory: Callable[[], "OperationProvider"]

@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    provider_model: str
    api_key: str
    provider_base_url: str = ""


class OperationProvider(Protocol):
    def invoke(self, operation: Any, config: ProviderConfig) -> dict[str, Any]:
        ...


from .providers import XAIProvider, XAI_MODELS, XAI_PROVIDER_BASE_URL, XAI_PROVIDER_ID


REGISTERED_PROVIDERS: dict[str, ProviderDefinition] = {
    XAI_PROVIDER_ID: ProviderDefinition(
        provider_id=XAI_PROVIDER_ID,
        models=XAI_MODELS,
        default_base_url=XAI_PROVIDER_BASE_URL,
        factory=XAIProvider,
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
