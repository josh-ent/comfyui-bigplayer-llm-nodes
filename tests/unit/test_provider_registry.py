from __future__ import annotations

from bigplayer_prompting.provider import list_models, list_provider_ids


def test_registered_xai_provider_models_match_expected_node_dropdown():
    assert list_provider_ids() == ["xAI"]
    assert list_models("xAI") == [
        "grok-4-1-fast-non-reasoning",
        "grok-4-1-fast-reasoning",
        "grok-4-latest",
    ]
