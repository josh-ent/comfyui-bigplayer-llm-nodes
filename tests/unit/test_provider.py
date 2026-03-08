from __future__ import annotations

import json

import httpx
import pytest
import respx

from bigplayer_prompting.errors import MalformedProviderResponseError, ProviderError, UnsupportedOperationError
from bigplayer_prompting.operations import PromptGenerationOperation
from bigplayer_prompting.provider import ProviderConfig
from bigplayer_prompting.providers.xai import XAIProvider


def _config(provider_model: str = "grok-4-latest") -> ProviderConfig:
    return ProviderConfig(
        provider="xAI",
        provider_model=provider_model,
        api_key="secret-key",
        provider_base_url="https://example.test/v1",
    )


def _operation() -> PromptGenerationOperation:
    return PromptGenerationOperation(
        prose="A cat on a windowsill.",
        target_model_name="sdxl-base-1.0.safetensors",
        style_policy="Keep it practical.",
        output_mode="simple",
        response_schema_name="test_schema",
        response_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    )


def test_xai_provider_renders_prompt_generation_operation():
    rendered = XAIProvider().render_operation(_operation(), _config())
    assert rendered.model == "grok-4-latest"
    assert "production-ready ComfyUI prompts" in rendered.system_prompt
    assert "A cat on a windowsill." in rendered.user_prompt
    assert "sdxl-base-1.0.safetensors" in rendered.user_prompt
    payload = rendered.as_payload()
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["text"]["format"]["strict"] is True


@respx.mock
def test_xai_provider_posts_structured_responses_request():
    route = respx.post("https://example.test/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"value": "ok"}),
                            }
                        ]
                    }
                ]
            },
        )
    )
    provider = XAIProvider()
    payload = provider.invoke(_operation(), _config())
    assert payload == {"value": "ok"}
    sent = route.calls[0].request.read().decode("utf-8")
    assert '"tools":[{"type":"web_search"}]' in sent
    assert "secret-key" not in sent


@respx.mock
def test_xai_provider_raises_for_non_json_output():
    respx.post("https://example.test/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={"output": [{"content": [{"type": "output_text", "text": "not-json"}]}]},
        )
    )
    with pytest.raises(MalformedProviderResponseError):
        XAIProvider().invoke(_operation(), _config())


def test_xai_provider_redacts_secret_in_http_errors(monkeypatch):
    def fake_post(self, *args, **kwargs):
        request = httpx.Request("POST", "https://example.test/v1/responses")
        response = httpx.Response(401, request=request, text="api key secret-key is invalid")
        raise httpx.HTTPStatusError("bad status", request=request, response=response)

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(ProviderError) as exc:
        XAIProvider().invoke(_operation(), _config())
    assert "secret-key" not in str(exc.value)
    assert "<redacted:" in str(exc.value)


def test_xai_provider_rejects_unknown_operation():
    class UnknownOperation:
        kind = "unknown"

    with pytest.raises(UnsupportedOperationError):
        XAIProvider().render_operation(UnknownOperation(), _config())


def test_xai_provider_rejects_unknown_model():
    with pytest.raises(ProviderError):
        XAIProvider().render_operation(_operation(), _config(provider_model="not-a-real-model"))
