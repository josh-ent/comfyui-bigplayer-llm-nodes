from __future__ import annotations

from .capabilities import (
    BASIC_PROMPT_CAPABILITY,
    CHECKPOINT_PICKER_CAPABILITY,
    KSAMPLER_CONFIG_CAPABILITY,
    SPLIT_PROMPT_CAPABILITY,
    list_available_checkpoints,
    list_sampler_names,
    list_scheduler_names,
)
from .errors import BigPlayerError
from .preset import (
    NONE_OPTION,
    normalize_preset_config,
    with_checkpoint_state,
    with_controlnet_state,
    with_lora_state,
)
from .provider import REGISTERED_PROVIDERS, list_models, list_provider_ids, provider_model_map
from .service import LLMProviderBundle, PromptGenerationService
from .status import ComfyStatusReporter


_SERVICE = PromptGenerationService()
_SAMPLER_NAME_OUTPUT_TYPE = list_sampler_names()
_SCHEDULER_OUTPUT_TYPE = list_scheduler_names()
_CHECKPOINT_NAME_OUTPUT_TYPE = list_available_checkpoints()
_CHECKPOINT_OPTIONS = [NONE_OPTION, *list_available_checkpoints()]
_CONTROLNET_TOOLTIP = (
    "Enter one ControlNet per line or comma-separated. "
    "You can also connect a string or string-list output."
)


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
            "optional": {
                "preset_config": (
                    "BIGPLAYER_PRESET_CONFIG",
                    {
                        "forceInput": True,
                        "tooltip": "Optional preset workflow state produced by BigPlayer state-indication nodes.",
                    },
                ),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def generate(self, prose, provider_config, preset_config=None, dynprompt=None, unique_id=None):
        reporter = ComfyStatusReporter(unique_id)
        if prose is None or not str(prose).strip():
            raise BigPlayerError("The prose input cannot be empty.")
        return (
            _SERVICE.begin_session(
                prose=prose,
                provider_bundle=provider_config,
                preset_config=preset_config,
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


class _BaseStateNode:
    CATEGORY = "BigPlayer/State Indication"
    RETURN_TYPES = ("BIGPLAYER_PRESET_CONFIG",)
    RETURN_NAMES = ("preset_config",)

    @classmethod
    def _preset_input(cls):
        return (
            "BIGPLAYER_PRESET_CONFIG",
            {
                "forceInput": True,
                "tooltip": "Optional preset config emitted by another BigPlayer state-indication node.",
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
    RETURN_TYPES = ("INT", "FLOAT", _SAMPLER_NAME_OUTPUT_TYPE, _SCHEDULER_OUTPUT_TYPE, "FLOAT", "STRING")
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
    RETURN_TYPES = (_CHECKPOINT_NAME_OUTPUT_TYPE, "STRING")
    RETURN_NAMES = ("checkpoint_name", "comments")
    FUNCTION = "read"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": cls._session_input()}}

    def read(self, session):
        result = _SERVICE.resolve_capability(session, CHECKPOINT_PICKER_CAPABILITY)
        return (result.checkpoint_name, result.comments)


class BigPlayerCheckpointState(_BaseStateNode):
    DESCRIPTION = "Indicate the currently chosen checkpoint and optional refiner checkpoint."
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_name": (
                    _CHECKPOINT_OPTIONS,
                    {
                        "default": NONE_OPTION,
                        "tooltip": "Selected base checkpoint, or <none> to leave unspecified.",
                    },
                ),
                "refiner_checkpoint_name": (
                    _CHECKPOINT_OPTIONS,
                    {
                        "default": NONE_OPTION,
                        "tooltip": "Optional refiner checkpoint, or <none> to leave unspecified.",
                    },
                ),
            },
            "optional": {
                "preset_config": cls._preset_input(),
            }
        }

    def build(self, checkpoint_name=NONE_OPTION, refiner_checkpoint_name=NONE_OPTION, preset_config=None):
        return (
            with_checkpoint_state(
                normalize_preset_config(preset_config),
                checkpoint_name=checkpoint_name,
                refiner_checkpoint_name=refiner_checkpoint_name,
            ),
        )


class BigPlayerLoRAState(_BaseStateNode):
    DESCRIPTION = "Indicate currently chosen LoRAs using LoRA Manager syntax or a linked LORA_STACK."
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_syntax": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Use LoRA Manager syntax like <lora:name:0.6> or <lora:name:0.6:0.4>.",
                    },
                ),
            },
            "optional": {
                "preset_config": cls._preset_input(),
                "lora_syntax_input": (
                    "*",
                    {
                        "forceInput": True,
                        "tooltip": "Optional linked LoRA syntax string or list of strings.",
                    },
                ),
                "lora_stack": (
                    "LORA_STACK",
                    {
                        "forceInput": True,
                        "tooltip": "Optional linked LORA_STACK compatible with ComfyUI-Lora-Manager.",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        del input_types, kwargs
        return True

    def build(self, lora_syntax="", preset_config=None, lora_syntax_input=None, lora_stack=None):
        return (
            with_lora_state(
                normalize_preset_config(preset_config),
                manual_syntax=str(lora_syntax or ""),
                linked_syntax=lora_syntax_input,
                lora_stack=lora_stack,
            ),
        )


class BigPlayerControlNetState(_BaseStateNode):
    DESCRIPTION = "Indicate currently chosen ControlNets using manual text or linked string inputs."
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnets": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": _CONTROLNET_TOOLTIP,
                    },
                ),
            },
            "optional": {
                "preset_config": cls._preset_input(),
                "controlnets_input": (
                    "*",
                    {
                        "forceInput": True,
                        "tooltip": "Optional linked ControlNet string or list of strings.",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        del input_types, kwargs
        return True

    def build(self, controlnets="", preset_config=None, controlnets_input=None):
        return (
            with_controlnet_state(
                normalize_preset_config(preset_config),
                manual_controlnets=str(controlnets or ""),
                linked_controlnets=controlnets_input,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "BigPlayerLLMProvider": BigPlayerLLMProvider,
    "BigPlayerNaturalLanguageRoot": BigPlayerNaturalLanguageRoot,
    "BigPlayerBasicPrompt": BigPlayerBasicPrompt,
    "BigPlayerSplitPrompt": BigPlayerSplitPrompt,
    "BigPlayerKSamplerConfig": BigPlayerKSamplerConfig,
    "BigPlayerCheckpointPicker": BigPlayerCheckpointPicker,
    "BigPlayerCheckpointState": BigPlayerCheckpointState,
    "BigPlayerLoRAState": BigPlayerLoRAState,
    "BigPlayerControlNetState": BigPlayerControlNetState,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BigPlayerLLMProvider": "BigPlayer LLM Provider",
    "BigPlayerNaturalLanguageRoot": "BigPlayer Natural Language Root",
    "BigPlayerBasicPrompt": "BigPlayer Basic Prompt",
    "BigPlayerSplitPrompt": "BigPlayer Split Prompt",
    "BigPlayerKSamplerConfig": "BigPlayer KSampler Config",
    "BigPlayerCheckpointPicker": "BigPlayer Checkpoint Picker",
    "BigPlayerCheckpointState": "BigPlayer Checkpoint State",
    "BigPlayerLoRAState": "BigPlayer LoRA State",
    "BigPlayerControlNetState": "BigPlayer ControlNet State",
}
