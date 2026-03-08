from __future__ import annotations

from bigplayer_prompting.nodes import BigPlayerLLMProvider
from bigplayer_prompting.provider import list_models, list_provider_ids, provider_model_map


def test_registered_xai_provider_models_match_expected_node_dropdown():
    assert list_provider_ids() == ["xAI", "No Provider"]
    assert list_models("xAI") == [
        "grok-4-1-fast-non-reasoning",
        "grok-4-1-fast-reasoning",
        "grok-4-latest",
    ]
    assert list_models("No Provider") == ["Positive", "Negative"]
    assert provider_model_map() == {
        "xAI": [
            "grok-4-1-fast-non-reasoning",
            "grok-4-1-fast-reasoning",
            "grok-4-latest",
        ],
        "No Provider": ["Positive", "Negative"],
    }


def test_no_provider_does_not_require_api_key():
    assert (
        BigPlayerLLMProvider.VALIDATE_INPUTS(
            api_key="",
            provider="No Provider",
            provider_model="Positive",
        )
        is True
    )
