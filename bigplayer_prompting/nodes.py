from __future__ import annotations

from .capabilities import (
    BASIC_PROMPT_CAPABILITY,
    CHECKPOINT_PICKER_CAPABILITY,
    KSAMPLER_CONFIG_CAPABILITY,
    SPLIT_PROMPT_CAPABILITY,
)
from .errors import BigPlayerError
from .provider import REGISTERED_PROVIDERS, list_models, list_provider_ids, provider_model_map
from .service import LLMProviderBundle, PromptGenerationService
from .status import ComfyStatusReporter


_SERVICE = PromptGenerationService()


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
        return _validate_provider_inputs(api_key, provider, provider_model)


class BigPlayerNaturalLanguageRoot:
    CATEGORY = "BigPlayer/Prompting"
    DESCRIPTION = "Discover attached BigPlayer modules, perform one LLM call, and publish a shared session."
    RETURN_TYPES = ("BIGPLAYER_LLM_SESSION",)
    RETURN_NAMES = ("session",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prose": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Freeform intent that should be transformed into structured workflow data.",
                    },
                ),
                "provider_config": (
                    "BIGPLAYER_LLM_PROVIDER",
                    {
                        "forceInput": True,
                        "tooltip": "Provider bundle produced by the BigPlayer LLM Provider node.",
                    },
                ),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def generate(self, prose, provider_config, dynprompt=None, unique_id=None):
        reporter = ComfyStatusReporter(unique_id)
        if prose is None or not str(prose).strip():
            raise BigPlayerError("The prose input cannot be empty.")
        return (
            _SERVICE.begin_session(
                prose=prose,
                provider_bundle=provider_config,
                dynprompt=dynprompt,
                root_node_id=str(unique_id or ""),
                invocation_context=reporter.as_invocation_context(),
            ),
        )


class _BaseSessionModule:
    CATEGORY = "BigPlayer/Prompting"

    @classmethod
    def _session_input(cls):
        return (
            "BIGPLAYER_LLM_SESSION",
            {
                "forceInput": True,
                "tooltip": "Shared session emitted by a BigPlayer Natural Language Root.",
            },
        )


class BigPlayerBasicPrompt(_BaseSessionModule):
    DESCRIPTION = "Read structured basic prompt output from the shared LLM session."
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "comments")
    FUNCTION = "read"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": cls._session_input()}}

    def read(self, session):
        result = _SERVICE.resolve_capability(session, BASIC_PROMPT_CAPABILITY)
        return (result.positive_prompt, result.negative_prompt, result.comments)


class BigPlayerSplitPrompt(_BaseSessionModule):
    DESCRIPTION = "Read structured split prompt output from the shared LLM session."
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "text_l_positive",
        "text_g_positive",
        "text_l_negative",
        "text_g_negative",
        "comments",
    )
    FUNCTION = "read"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": cls._session_input()}}

    def read(self, session):
        result = _SERVICE.resolve_capability(session, SPLIT_PROMPT_CAPABILITY)
        return (
            result.text_l_positive,
            result.text_g_positive,
            result.text_l_negative,
            result.text_g_negative,
            result.comments,
        )


class BigPlayerKSamplerConfig(_BaseSessionModule):
    DESCRIPTION = "Read structured KSampler settings from the shared LLM session."
    RETURN_TYPES = ("INT", "FLOAT", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("steps", "cfg", "sampler_name", "scheduler", "denoise", "comments")
    FUNCTION = "read"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": cls._session_input()}}

    def read(self, session):
        result = _SERVICE.resolve_capability(session, KSAMPLER_CONFIG_CAPABILITY)
        return (
            result.steps,
            result.cfg,
            result.sampler_name,
            result.scheduler,
            result.denoise,
            result.comments,
        )


class BigPlayerCheckpointPicker(_BaseSessionModule):
    DESCRIPTION = "Read structured checkpoint selection from the shared LLM session."
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("checkpoint_name", "comments")
    FUNCTION = "read"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": cls._session_input()}}

    def read(self, session):
        result = _SERVICE.resolve_capability(session, CHECKPOINT_PICKER_CAPABILITY)
        return (result.checkpoint_name, result.comments)


class BigPlayerModelContext(_BaseSessionModule):
    DESCRIPTION = "Contribute optional model-context text to a shared LLM session."
    RETURN_TYPES = ()
    FUNCTION = "register"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": cls._session_input(),
                "model_context": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional context string that the LLM can use when shaping all attached outputs.",
                    },
                ),
            }
        }

    def register(self, session, model_context=""):
        del session, model_context
        return {}


NODE_CLASS_MAPPINGS = {
    "BigPlayerLLMProvider": BigPlayerLLMProvider,
    "BigPlayerNaturalLanguageRoot": BigPlayerNaturalLanguageRoot,
    "BigPlayerBasicPrompt": BigPlayerBasicPrompt,
    "BigPlayerSplitPrompt": BigPlayerSplitPrompt,
    "BigPlayerKSamplerConfig": BigPlayerKSamplerConfig,
    "BigPlayerCheckpointPicker": BigPlayerCheckpointPicker,
    "BigPlayerModelContext": BigPlayerModelContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BigPlayerLLMProvider": "BigPlayer LLM Provider",
    "BigPlayerNaturalLanguageRoot": "BigPlayer Natural Language Root",
    "BigPlayerBasicPrompt": "BigPlayer Basic Prompt",
    "BigPlayerSplitPrompt": "BigPlayer Split Prompt",
    "BigPlayerKSamplerConfig": "BigPlayer KSampler Config",
    "BigPlayerCheckpointPicker": "BigPlayer Checkpoint Picker",
    "BigPlayerModelContext": "BigPlayer Model Context",
}
