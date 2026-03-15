from __future__ import annotations

from bigplayer.generation.capabilities import BASIC_PROMPT_CAPABILITY, SPLIT_PROMPT_CAPABILITY
from bigplayer.generation.operations import PromptGenerationOperation
from bigplayer.providers.base import InvocationContext, ProviderConfig, ProviderDebugRecord
from bigplayer.providers.no_provider import NO_PROVIDER_COMMENT, NoProvider


def test_no_provider_positive_routes_to_basic_prompt():
    debug = ProviderDebugRecord()
    result = NoProvider().invoke(
        PromptGenerationOperation(
            prose="bright red apple",
            context_blocks=(),
            requested_capabilities=(BASIC_PROMPT_CAPABILITY,),
            capability_configs={BASIC_PROMPT_CAPABILITY: {}},
        ),
        ProviderConfig(provider="No Provider", provider_model="Positive", api_key=""),
        InvocationContext(debug_record=debug),
    )
    assert result == {
        "basic_prompt": {
            "positive_prompt": "bright red apple",
            "negative_prompt": "",
            "comments": NO_PROVIDER_COMMENT,
        }
    }
    assert "bright red apple" in debug.request_text
    assert '"positive_prompt": "bright red apple"' in debug.response_text


def test_no_provider_negative_routes_to_split_prompt():
    debug = ProviderDebugRecord()
    result = NoProvider().invoke(
        PromptGenerationOperation(
            prose="washed out",
            context_blocks=(),
            requested_capabilities=(SPLIT_PROMPT_CAPABILITY,),
            capability_configs={SPLIT_PROMPT_CAPABILITY: {}},
        ),
        ProviderConfig(provider="No Provider", provider_model="Negative", api_key=""),
        InvocationContext(debug_record=debug),
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
    assert "washed out" in debug.request_text
    assert '"text_l_negative": "washed out"' in debug.response_text
