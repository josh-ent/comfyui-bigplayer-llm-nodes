from __future__ import annotations

from ...generation.service import LLMProviderBundle
from ...providers.registry import REGISTERED_PROVIDERS, list_models, list_provider_ids, provider_model_map


def _validate_provider_inputs(api_key: str, provider: str, provider_model: str) -> bool | str:
    if provider not in REGISTERED_PROVIDERS:
        return f"Unsupported provider: {provider}"
    if REGISTERED_PROVIDERS[provider].requires_api_key and (api_key is None or not api_key.strip()):
        return "The api_key input cannot be empty."
    if provider_model not in REGISTERED_PROVIDERS[provider].models:
        return f"Unsupported model `{provider_model}` for provider `{provider}`."
    return True


class BigPlayerLLMProvider:
    CATEGORY = "BigPlayer/Prompting"
    DESCRIPTION = "Build reusable LLM provider configuration for a BigPlayer Natural Language Root."
    RETURN_TYPES = ("BIGPLAYER_LLM_PROVIDER",)
    RETURN_NAMES = ("provider_config",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Provider API key. This value is never echoed back by the node.",
                    },
                ),
                "provider": (
                    list_provider_ids(),
                    {
                        "default": list_provider_ids()[0],
                        "tooltip": "Registered LLM provider to call.",
                    },
                ),
                "provider_model": (
                    list_models(),
                    {
                        "default": list_models()[0],
                        "tooltip": "Registered provider model to call.",
                        "provider_models": provider_model_map(),
                    },
                ),
                "provider_base_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional override for the provider base URL.",
                    },
                ),
                "assume_determinism": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reuse deterministic results when the request is unchanged.",
                    },
                ),
            }
        }

    def build(self, api_key, provider, provider_model, provider_base_url="", assume_determinism=True):
        return (
            LLMProviderBundle(
                api_key=api_key,
                provider=provider,
                provider_model=provider_model,
                provider_base_url=provider_base_url,
                assume_determinism=assume_determinism,
            ),
        )

    @classmethod
    def VALIDATE_INPUTS(cls, api_key=None, provider=None, provider_model=None, **kwargs):
        del kwargs
        return _validate_provider_inputs(api_key, provider, provider_model)
