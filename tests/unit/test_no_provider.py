from __future__ import annotations

from bigplayer_prompting.operations import PromptGenerationOperation
from bigplayer_prompting.provider import ProviderConfig
from bigplayer_prompting.providers.no_provider import NO_PROVIDER_COMMENT, NoProvider


def test_no_provider_positive_simple_routes_to_positive_prompt():
    result = NoProvider().invoke(
        PromptGenerationOperation(
            prose="bright red apple",
            target_model_name="test-model",
            style_policy="",
            output_mode="simple",
            response_schema_name="simple",
            response_schema={},
        ),
        ProviderConfig(provider="No Provider", provider_model="Positive", api_key=""),
    )
    assert result == {
        "positive_prompt": "bright red apple",
        "negative_prompt": "",
        "comments": NO_PROVIDER_COMMENT,
    }


def test_no_provider_negative_split_routes_to_negative_channels():
    result = NoProvider().invoke(
        PromptGenerationOperation(
            prose="washed out",
            target_model_name="test-model",
            style_policy="",
            output_mode="split",
            response_schema_name="split",
            response_schema={},
        ),
        ProviderConfig(provider="No Provider", provider_model="Negative", api_key=""),
    )
    assert result == {
        "text_l_positive": "",
        "text_g_positive": "",
        "text_l_negative": "washed out",
        "text_g_negative": "washed out",
        "comments": NO_PROVIDER_COMMENT,
    }
