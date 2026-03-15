from __future__ import annotations

from ...state.preset import normalize_preset_config, with_controlnet_state
from .._common import _BaseStateNode


_CONTROLNET_TOOLTIP = (
    "Enter one ControlNet per line or comma-separated. "
    "You can also connect a string or string-list output."
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
                "controlnets_also": (
                    "*",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Takes a string or list of strings and concatenates it onto "
                            "`controlnets`; linked entries win ties where needed."
                        ),
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        del input_types, kwargs
        return True

    def build(self, controlnets="", preset_config=None, controlnets_also=None):
        return (
            with_controlnet_state(
                normalize_preset_config(preset_config),
                manual_controlnets=str(controlnets or ""),
                linked_controlnets=controlnets_also,
            ),
        )
