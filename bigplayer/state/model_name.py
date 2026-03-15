from __future__ import annotations

import os
from typing import Any, Iterable

from ..errors import ModelNameExtractionError


def _candidate_paths(model: Any) -> Iterable[str]:
    cached = getattr(model, "cached_patcher_init", None)
    if isinstance(cached, tuple) and len(cached) >= 2:
        args = cached[1]
        if isinstance(args, tuple) and args:
            first = args[0]
            if isinstance(first, str):
                yield first

    patcher = getattr(model, "patcher", None)
    if patcher is not None:
        cached = getattr(patcher, "cached_patcher_init", None)
        if isinstance(cached, tuple) and len(cached) >= 2:
            args = cached[1]
            if isinstance(args, tuple) and args:
                first = args[0]
                if isinstance(first, str):
                    yield first

    for attr in ("ckpt_name", "checkpoint_name", "model_name", "filename", "file_name", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            yield value

    model_options = getattr(model, "model_options", None)
    if isinstance(model_options, dict):
        for key in ("ckpt_name", "checkpoint_name", "model_name", "filename", "name"):
            value = model_options.get(key)
            if isinstance(value, str):
                yield value

    inner_model = getattr(model, "model", None)
    if inner_model is not None and inner_model is not model:
        for attr in ("ckpt_name", "checkpoint_name", "model_name", "filename", "file_name", "name"):
            value = getattr(inner_model, attr, None)
            if isinstance(value, str):
                yield value


def extract_model_name(model: Any) -> str:
    for candidate in _candidate_paths(model):
        cleaned = os.path.basename(candidate.strip())
        if cleaned:
            return cleaned
    raise ModelNameExtractionError(
        "Could not derive a model name from the connected MODEL input. "
        "Use a checkpoint-loaded MODEL and ensure it retains its source checkpoint path."
    )
