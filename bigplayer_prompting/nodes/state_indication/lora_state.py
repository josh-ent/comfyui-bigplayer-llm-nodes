from __future__ import annotations

from ...state.preset import normalize_preset_config, with_lora_state
from .._common import _BaseStateNode


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
