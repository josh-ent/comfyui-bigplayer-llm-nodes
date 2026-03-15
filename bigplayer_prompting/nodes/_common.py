from __future__ import annotations

from ..generation.service import PromptGenerationService


_SERVICE = PromptGenerationService()


class _BaseSessionNode:
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
