from __future__ import annotations

import pytest

from bigplayer.errors import ModelNameExtractionError
from bigplayer.state.model_name import extract_model_name


class ModelWithCachedInit:
    cached_patcher_init = (object(), ("/models/sdxl-base-1.0.safetensors",))


class ModelWithNestedPatcher:
    class Patcher:
        cached_patcher_init = (object(), ("/models/flux-dev.safetensors",))

    patcher = Patcher()


class ModelWithOptions:
    model_options = {"model_name": "wan-video-14b.safetensors"}


def test_extract_model_name_from_cached_patcher_init():
    assert extract_model_name(ModelWithCachedInit()) == "sdxl-base-1.0.safetensors"


def test_extract_model_name_from_nested_patcher():
    assert extract_model_name(ModelWithNestedPatcher()) == "flux-dev.safetensors"


def test_extract_model_name_from_model_options():
    assert extract_model_name(ModelWithOptions()) == "wan-video-14b.safetensors"


def test_extract_model_name_fails_clearly():
    with pytest.raises(ModelNameExtractionError):
        extract_model_name(object())

