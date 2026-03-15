from __future__ import annotations

from ...generation.capabilities import list_available_checkpoints
from ...state.preset import NONE_OPTION, normalize_preset_config, with_checkpoint_state
from .._common import _BaseStateNode


_CHECKPOINT_OPTIONS = [NONE_OPTION, *list_available_checkpoints()]


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
            },
        }

    def build(self, checkpoint_name=NONE_OPTION, refiner_checkpoint_name=NONE_OPTION, preset_config=None):
        return (
            with_checkpoint_state(
                normalize_preset_config(preset_config),
                checkpoint_name=checkpoint_name,
                refiner_checkpoint_name=refiner_checkpoint_name,
            ),
        )
