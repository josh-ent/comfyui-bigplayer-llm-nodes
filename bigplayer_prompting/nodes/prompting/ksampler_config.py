from __future__ import annotations

from ...generation.capabilities import KSAMPLER_CONFIG_CAPABILITY, list_sampler_names, list_scheduler_names
from .._common import _BaseSessionNode, _SERVICE


class BigPlayerKSamplerConfig(_BaseSessionNode):
    DESCRIPTION = "Read structured KSampler settings from the shared LLM session."
    RETURN_TYPES = ("INT", "FLOAT", list_sampler_names(), list_scheduler_names(), "FLOAT", "STRING")
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
