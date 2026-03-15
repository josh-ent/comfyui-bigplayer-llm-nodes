from __future__ import annotations

from ...generation.capabilities import BASIC_PROMPT_CAPABILITY
from .._common import _BaseSessionNode, _SERVICE


class BigPlayerBasicPrompt(_BaseSessionNode):
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
