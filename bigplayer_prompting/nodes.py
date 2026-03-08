from __future__ import annotations

from .provider import REGISTERED_PROVIDERS, list_models, list_provider_ids, provider_model_map
from .errors import BigPlayerError
from .service import PromptGenerationRequest, PromptGenerationService


_SERVICE = PromptGenerationService()


def _validate_common_text_inputs(prose: str, api_key: str, provider: str, provider_model: str) -> bool | str:
    if prose is None or not prose.strip():
        return "The prose input cannot be empty."
    if provider not in REGISTERED_PROVIDERS:
        return f"Unsupported provider: {provider}"
    if REGISTERED_PROVIDERS[provider].requires_api_key and (api_key is None or not api_key.strip()):
        return "The api_key input cannot be empty."
    if provider_model not in REGISTERED_PROVIDERS[provider].models:
        return f"Unsupported model `{provider_model}` for provider `{provider}`."
    return True


class _BasePromptNode:
    CATEGORY = "BigPlayer/Prompting"
    DESCRIPTION = (
        "Convert freeform prose into validated prompts using an external LLM. "
        "The connected MODEL is used only to derive the model name."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prose": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Freeform intent that should be converted into prompts.",
                    },
                ),
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
                "target_model": (
                    "MODEL",
                    {
                        "tooltip": "Connected ComfyUI MODEL. The node derives only the model name from it.",
                    },
                ),
                "style_policy": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional style or policy guidance for the prompt transformation.",
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

    @classmethod
    def VALIDATE_INPUTS(cls, prose=None, api_key=None, provider=None, provider_model=None, **kwargs):
        return _validate_common_text_inputs(prose, api_key, provider, provider_model)


class BigPlayerPromptSimple(_BasePromptNode):
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "comments")
    OUTPUT_TOOLTIPS = (
        "Validated positive prompt text.",
        "Validated negative prompt text.",
        "Comments explaining how the prompt was shaped.",
    )
    FUNCTION = "generate"
    SEARCH_ALIASES = ["grok prompt", "llm prompt", "bigplayer simple prompt"]

    def generate(self, prose, api_key, provider, provider_model, target_model, style_policy="", provider_base_url="", assume_determinism=True):
        try:
            result = _SERVICE.generate(
                PromptGenerationRequest(
                    mode="simple",
                    prose=prose,
                    api_key=api_key,
                    provider=provider,
                    provider_model=provider_model,
                    target_model=target_model,
                    style_policy=style_policy,
                    provider_base_url=provider_base_url,
                    assume_determinism=assume_determinism,
                )
            )
        except BigPlayerError:
            raise
        return (result.positive_prompt, result.negative_prompt, result.comments)

    @classmethod
    def IS_CHANGED(
        cls,
        prose="",
        api_key="",
        provider="",
        provider_model="",
        style_policy="",
        provider_base_url="",
        assume_determinism=True,
        **kwargs,
    ):
        return _SERVICE.build_is_changed_token(
            prose=prose,
            api_key=api_key,
            provider=provider,
            provider_model=provider_model,
            style_policy=style_policy,
            provider_base_url=provider_base_url,
            assume_determinism=assume_determinism,
            mode="simple",
        )


class BigPlayerPromptSplit(_BasePromptNode):
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "text_l_positive",
        "text_g_positive",
        "text_l_negative",
        "text_g_negative",
        "comments",
    )
    OUTPUT_TOOLTIPS = (
        "Validated text-l positive prompt text.",
        "Validated text-g positive prompt text.",
        "Validated text-l negative prompt text.",
        "Validated text-g negative prompt text.",
        "Comments explaining how the prompt was shaped.",
    )
    FUNCTION = "generate"
    SEARCH_ALIASES = ["grok split prompt", "llm split prompt", "bigplayer split prompt"]

    def generate(self, prose, api_key, provider, provider_model, target_model, style_policy="", provider_base_url="", assume_determinism=True):
        result = _SERVICE.generate(
            PromptGenerationRequest(
                mode="split",
                prose=prose,
                api_key=api_key,
                provider=provider,
                provider_model=provider_model,
                target_model=target_model,
                style_policy=style_policy,
                provider_base_url=provider_base_url,
                assume_determinism=assume_determinism,
            )
        )
        return (
            result.text_l_positive,
            result.text_g_positive,
            result.text_l_negative,
            result.text_g_negative,
            result.comments,
        )

    @classmethod
    def IS_CHANGED(
        cls,
        prose="",
        api_key="",
        provider="",
        provider_model="",
        style_policy="",
        provider_base_url="",
        assume_determinism=True,
        **kwargs,
    ):
        return _SERVICE.build_is_changed_token(
            prose=prose,
            api_key=api_key,
            provider=provider,
            provider_model=provider_model,
            style_policy=style_policy,
            provider_base_url=provider_base_url,
            assume_determinism=assume_determinism,
            mode="split",
        )


NODE_CLASS_MAPPINGS = {
    "BigPlayerPromptSimple": BigPlayerPromptSimple,
    "BigPlayerPromptSplit": BigPlayerPromptSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BigPlayerPromptSimple": "BigPlayer Prompt (Simple)",
    "BigPlayerPromptSplit": "BigPlayer Prompt (Split)",
}
