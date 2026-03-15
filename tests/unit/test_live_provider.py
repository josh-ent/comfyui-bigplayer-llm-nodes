from __future__ import annotations

import os

import pytest

from bigplayer.generation.capabilities import BASIC_PROMPT_CAPABILITY
from bigplayer.generation.operations import PromptGenerationOperation
from bigplayer.generation.schemas import validate_result
from bigplayer.providers.base import ProviderConfig
from bigplayer.providers.xai import XAIProvider


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.environ.get("BIGPLAYER_GROK_LIVE_TEST") != "1" or not os.environ.get("BIGPLAYER_GROK_API_KEY"),
    reason="Live Grok tests are opt-in.",
)
def test_live_provider_accepts_structured_output():
    provider = XAIProvider()
    capability_configs = {BASIC_PROMPT_CAPABILITY: {}}
    result = provider.invoke(
        PromptGenerationOperation(
            prose="A cinematic portrait of a cat sitting on a velvet chair.",
            context_blocks=(),
            requested_capabilities=(BASIC_PROMPT_CAPABILITY,),
            capability_configs=capability_configs,
        ),
        ProviderConfig(
            provider="xAI",
            provider_model=os.environ.get("BIGPLAYER_GROK_MODEL", "grok-4-latest"),
            api_key=os.environ["BIGPLAYER_GROK_API_KEY"],
            provider_base_url=os.environ.get("BIGPLAYER_GROK_BASE_URL", ""),
        ),
    )
    validated = validate_result(capability_configs, result)
    assert validated["basic_prompt"]["positive_prompt"]
