from __future__ import annotations

from bigplayer_prompting.capabilities import BASIC_PROMPT_CAPABILITY, SPLIT_PROMPT_CAPABILITY
from bigplayer_prompting.operations import PromptGenerationOperation
from bigplayer_prompting.provider import ProviderConfig
from bigplayer_prompting.providers.no_provider import NO_PROVIDER_COMMENT, NoProvider


def test_no_provider_positive_routes_to_basic_prompt():
    result = NoProvider().invoke(
        PromptGenerationOperation(
            prose="bright red apple",
            context_blocks=(),
            capability_instructions=(),
            requested_capabilities=(BASIC_PROMPT_CAPABILITY,),
            capability_configs={BASIC_PROMPT_CAPABILITY: {}},
            response_schema_name="simple",
            response_schema={},
        ),
        ProviderConfig(provider="No Provider", provider_model="Positive", api_key=""),
    )
    assert result == {
        "basic_prompt": {
            "positive_prompt": "bright red apple",
            "negative_prompt": "",
            "comments": NO_PROVIDER_COMMENT,
        }
    }


def test_no_provider_negative_routes_to_split_prompt():
    result = NoProvider().invoke(
        PromptGenerationOperation(
            prose="washed out",
            context_blocks=(),
            capability_instructions=(),
            requested_capabilities=(SPLIT_PROMPT_CAPABILITY,),
            capability_configs={SPLIT_PROMPT_CAPABILITY: {}},
            response_schema_name="split",
            response_schema={},
        ),
        ProviderConfig(provider="No Provider", provider_model="Negative", api_key=""),
    )
    assert result == {
        "split_prompt": {
            "text_l_positive": "",
            "text_g_positive": "",
            "text_l_negative": "washed out",
            "text_g_negative": "washed out",
            "comments": NO_PROVIDER_COMMENT,
        }
    }
