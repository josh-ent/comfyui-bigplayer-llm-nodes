from __future__ import annotations

from .._common import _BaseSessionNode, _SERVICE


class BigPlayerPromptDebug(_BaseSessionNode):
    DESCRIPTION = "Read raw provider-owned request and response text from the shared LLM session."
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("request_text", "response_text")
    FUNCTION = "read"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": cls._session_input()}}

    def read(self, session):
        debug = _SERVICE.resolve_debug(session)
        return (debug.request_text, debug.response_text)
