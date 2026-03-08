from __future__ import annotations

import os

import pytest

from bigplayer_prompting.operations import PromptGenerationOperation
from bigplayer_prompting.provider import ProviderConfig
from bigplayer_prompting.schemas import get_provider_schema, validate_result
from bigplayer_prompting.providers.xai import XAIProvider


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.environ.get("BIGPLAYER_GROK_LIVE_TEST") != "1" or not os.environ.get("BIGPLAYER_GROK_API_KEY"),
    reason="Live Grok tests are opt-in.",
)
def test_live_provider_accepts_structured_output():
    provider = XAIProvider()
    result = provider.invoke(
        PromptGenerationOperation(
            prose="A cinematic portrait of a cat sitting on a velvet chair.",
            target_model_name="sdxl-base-1.0.safetensors",
            style_policy="",
            output_mode="simple",
            response_schema_name="bigplayer_live_smoke",
            response_schema=get_provider_schema("simple"),
        ),
        ProviderConfig(
            provider="xAI",
            provider_model=os.environ.get("BIGPLAYER_GROK_MODEL", "grok-4-latest"),
            api_key=os.environ["BIGPLAYER_GROK_API_KEY"],
            provider_base_url=os.environ.get("BIGPLAYER_GROK_BASE_URL", ""),
        ),
    )
    validated = validate_result("simple", result)
    assert validated.positive_prompt
