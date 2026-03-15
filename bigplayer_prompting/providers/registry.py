from __future__ import annotations

from .base import ProviderDefinition
from .no_provider import NO_PROVIDER_BASE_URL, NO_PROVIDER_ID, NO_PROVIDER_MODELS, NoProvider
from .xai import XAI_MODELS, XAI_PROVIDER_BASE_URL, XAI_PROVIDER_ID, XAIProvider


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
