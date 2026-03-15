from __future__ import annotations

import pytest

from bigplayer.errors import ProviderError
from bigplayer.state.preset import (
    PresetConfigBundle,
    with_controlnet_state,
    with_lora_state,
)


def test_lora_state_parses_single_and_dual_strength_syntax():
    bundle = with_lora_state(
        None,
        manual_syntax="<lora:first:0.5> <lora:second:0.8:0.3>",
    )

    assert [lora.name for lora in bundle.loras] == ["first", "second"]
    assert bundle.loras[0].model_strength == 0.5
    assert bundle.loras[0].clip_strength == 0.5
    assert bundle.loras[1].model_strength == 0.8
    assert bundle.loras[1].clip_strength == 0.3


def test_lora_state_preserves_relative_paths_from_lora_stack():
    bundle = with_lora_state(
        None,
        manual_syntax="",
        lora_stack=[("styles\\detail.safetensors", 0.8, 0.6)],
    )

    assert bundle.loras[0].relative_path == "styles\\detail.safetensors"
    assert bundle.loras[0].name == "detail"
    assert bundle.loras[0].model_strength == 0.8
    assert bundle.loras[0].clip_strength == 0.6


def test_lora_state_rejects_invalid_syntax():
    with pytest.raises(ProviderError):
        with_lora_state(None, manual_syntax="<lora:broken:not-a-number>")


def test_controlnet_state_accepts_strings_and_string_lists():
    bundle = with_controlnet_state(
        PresetConfigBundle(),
        manual_controlnets="depth.safetensors\ncanny.safetensors",
        linked_controlnets=["tile.safetensors", "depth.safetensors"],
    )

    assert bundle.controlnets == ("depth.safetensors", "canny.safetensors", "tile.safetensors")
