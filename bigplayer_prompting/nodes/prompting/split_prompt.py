from __future__ import annotations

from ...generation.capabilities import SPLIT_PROMPT_CAPABILITY
from .._common import _BaseSessionNode, _SERVICE


class BigPlayerSplitPrompt(_BaseSessionNode):
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
