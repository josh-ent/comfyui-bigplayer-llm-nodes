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
    requires_api_key: bool = True

@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    provider_model: str
    api_key: str
    provider_base_url: str = ""


@dataclass(frozen=True)
class InvocationContext:
    status_callback: Callable[[str], None] | None = None

    def report_status(self, message: str) -> None:
        if self.status_callback is not None and message:
            self.status_callback(message)


class OperationProvider(Protocol):
    def invoke(
        self,
        operation: Any,
        config: ProviderConfig,
        context: InvocationContext | None = None,
    ) -> dict[str, Any]:
        ...


from .providers import (
    NO_PROVIDER_BASE_URL,
    NO_PROVIDER_ID,
    NO_PROVIDER_MODELS,
    NoProvider,
    XAIProvider,
    XAI_MODELS,
    XAI_PROVIDER_BASE_URL,
    XAI_PROVIDER_ID,
)


REGISTERED_PROVIDERS: dict[str, ProviderDefinition] = {
    XAI_PROVIDER_ID: ProviderDefinition(
        provider_id=XAI_PROVIDER_ID,
        models=XAI_MODELS,
        default_base_url=XAI_PROVIDER_BASE_URL,
        factory=XAIProvider,
    ),
    NO_PROVIDER_ID: ProviderDefinition(
        provider_id=NO_PROVIDER_ID,
        models=NO_PROVIDER_MODELS,
        default_base_url=NO_PROVIDER_BASE_URL,
        factory=NoProvider,
        requires_api_key=False,
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


def provider_model_map() -> dict[str, list[str]]:
    return {provider_id: list(definition.models) for provider_id, definition in REGISTERED_PROVIDERS.items()}
