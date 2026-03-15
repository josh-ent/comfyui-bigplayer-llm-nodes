from __future__ import annotations

from ...errors import BigPlayerError
from ...generation.status import ComfyStatusReporter
from .._common import _SERVICE


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
