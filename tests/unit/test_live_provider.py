from __future__ import annotations

import os

import pytest

from bigplayer_prompting.provider import GrokProvider, ProviderRequest


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.environ.get("BIGPLAYER_GROK_LIVE_TEST") != "1" or not os.environ.get("BIGPLAYER_GROK_API_KEY"),
    reason="Live Grok tests are opt-in.",
)
def test_live_provider_accepts_structured_output():
    provider = GrokProvider()
    result = provider.generate_structured(
        ProviderRequest(
            api_key=os.environ["BIGPLAYER_GROK_API_KEY"],
            provider_model=os.environ.get("BIGPLAYER_GROK_MODEL", "grok-4-latest"),
            system_prompt="Return only the requested schema.",
            user_prompt="Return a JSON object with a single key `value` set to `ok`.",
            schema_name="bigplayer_live_smoke",
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
            },
            provider_base_url=os.environ.get("BIGPLAYER_GROK_BASE_URL", ""),
        )
    )
    assert result["value"]
