from __future__ import annotations

import json

import httpx
import pytest
import respx

from bigplayer_prompting.errors import MalformedProviderResponseError, ProviderError
from bigplayer_prompting.provider import GrokProvider, ProviderRequest


def _request() -> ProviderRequest:
    return ProviderRequest(
        api_key="secret-key",
        llm_model="grok-test",
        system_prompt="system",
        user_prompt="user",
        schema_name="test_schema",
        schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        provider_base_url="https://example.test/v1",
    )


@respx.mock
def test_provider_posts_structured_responses_request():
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
    provider = GrokProvider()
    payload = provider.generate_structured(_request())
    assert payload == {"value": "ok"}
    sent = route.calls[0].request.read().decode("utf-8")
    assert '"tools":[{"type":"web_search"}]' in sent
    assert "secret-key" not in sent


@respx.mock
def test_provider_raises_for_non_json_output():
    respx.post("https://example.test/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={"output": [{"content": [{"type": "output_text", "text": "not-json"}]}]},
        )
    )
    with pytest.raises(MalformedProviderResponseError):
        GrokProvider().generate_structured(_request())


def test_provider_redacts_secret_in_http_errors(monkeypatch):
    def fake_post(self, *args, **kwargs):
        request = httpx.Request("POST", "https://example.test/v1/responses")
        response = httpx.Response(401, request=request, text="api key secret-key is invalid")
        raise httpx.HTTPStatusError("bad status", request=request, response=response)

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    with pytest.raises(ProviderError) as exc:
        GrokProvider().generate_structured(_request())
    assert "secret-key" not in str(exc.value)
    assert "<redacted:" in str(exc.value)

