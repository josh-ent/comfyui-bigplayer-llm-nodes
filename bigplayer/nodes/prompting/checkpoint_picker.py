from __future__ import annotations

from ...generation.capabilities import CHECKPOINT_PICKER_CAPABILITY, list_available_checkpoints
from .._common import _BaseSessionNode, _SERVICE


class BigPlayerCheckpointPicker(_BaseSessionNode):
    DESCRIPTION = "Read structured checkpoint selection from the shared LLM session."
    RETURN_TYPES = (list_available_checkpoints(), "STRING")
    RETURN_NAMES = ("checkpoint_name", "comments")
    FUNCTION = "read"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": cls._session_input()}}

    def read(self, session):
        result = _SERVICE.resolve_capability(session, CHECKPOINT_PICKER_CAPABILITY)
        return (result.checkpoint_name, result.comments)
