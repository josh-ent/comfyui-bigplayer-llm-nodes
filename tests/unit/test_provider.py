from __future__ import annotations

import json
from contextlib import nullcontext

import httpx
import pytest
import respx

from bigplayer_prompting.errors import MalformedProviderResponseError, ProviderError, UnsupportedOperationError
from bigplayer_prompting.operations import PromptGenerationOperation
from bigplayer_prompting.provider import InvocationContext, ProviderConfig
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


class FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        json_body: dict | None = None,
        lines: list[str] | None = None,
        error_text: str = "",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._json_body = json_body
        self._lines = lines or []
        self.text = error_text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test/v1/responses")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("bad status", request=request, response=response)

    def json(self) -> dict:
        assert self._json_body is not None
        return self._json_body

    def read(self) -> bytes:
        if self._json_body is None:
            return b""
        return json.dumps(self._json_body).encode("utf-8")

    def iter_lines(self):
        for line in self._lines:
            yield line


def test_xai_provider_renders_prompt_generation_operation():
    rendered = XAIProvider().render_operation(_operation(), _config())
    assert rendered.model == "grok-4-latest"
    assert "production-ready ComfyUI prompts" in rendered.system_prompt
    assert "A cat on a windowsill." in rendered.user_prompt
    assert "sdxl-base-1.0.safetensors" in rendered.user_prompt
    payload = rendered.as_payload()
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["text"]["format"]["strict"] is True
    assert payload["stream"] is True


@respx.mock
def test_xai_provider_posts_streaming_responses_request_and_accepts_json_fallback():
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
            headers={"content-type": "application/json"},
        )
    )
    provider = XAIProvider()
    payload = provider.invoke(_operation(), _config())
    assert payload == {"value": "ok"}
    sent = route.calls[0].request.read().decode("utf-8")
    assert '"tools":[{"type":"web_search"}]' in sent
    assert '"stream":true' in sent
    assert "secret-key" not in sent


def test_xai_provider_accumulates_sse_output_and_reports_status(monkeypatch):
    response = FakeStreamResponse(
        headers={"content-type": "text/event-stream"},
        lines=[
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"{\\"value\\": "}',
            "",
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"\\"ok\\"}"}',
            "",
            "data: [DONE]",
            "",
        ],
    )
    monkeypatch.setattr(httpx.Client, "stream", lambda self, *args, **kwargs: nullcontext(response))

    statuses: list[str] = []
    payload = XAIProvider().invoke(
        _operation(),
        _config(),
        InvocationContext(status_callback=statuses.append),
    )
    assert payload == {"value": "ok"}
    assert any("Connecting to xAI" in status for status in statuses)
    assert any("Receiving structured output from xAI" in status for status in statuses)


def test_xai_provider_prefers_completed_response_object_from_stream(monkeypatch):
    response = FakeStreamResponse(
        headers={"content-type": "text/event-stream"},
        lines=[
            'event: response.completed',
            'data: {"type":"response.completed","response":{"output_text":"{\\"value\\": \\"ok\\"}"}}',
            "",
            "data: [DONE]",
            "",
        ],
    )
    monkeypatch.setattr(httpx.Client, "stream", lambda self, *args, **kwargs: nullcontext(response))
    payload = XAIProvider().invoke(_operation(), _config())
    assert payload == {"value": "ok"}


def test_xai_provider_raises_for_non_json_output(monkeypatch):
    response = FakeStreamResponse(
        headers={"content-type": "text/event-stream"},
        lines=[
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"not-json"}',
            "",
            "data: [DONE]",
            "",
        ],
    )
    monkeypatch.setattr(httpx.Client, "stream", lambda self, *args, **kwargs: nullcontext(response))
    with pytest.raises(MalformedProviderResponseError):
        XAIProvider().invoke(_operation(), _config())


def test_xai_provider_redacts_secret_in_http_errors(monkeypatch):
    response = FakeStreamResponse(status_code=401, error_text="api key secret-key is invalid")
    monkeypatch.setattr(httpx.Client, "stream", lambda self, *args, **kwargs: nullcontext(response))
    with pytest.raises(ProviderError) as exc:
        XAIProvider().invoke(_operation(), _config())
    assert "secret-key" not in str(exc.value)
    assert "<redacted:" in str(exc.value)


def test_xai_provider_reports_timeout_as_idle_timeout(monkeypatch):
    def fake_stream(self, *args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    with pytest.raises(ProviderError) as exc:
        XAIProvider().invoke(_operation(), _config())
    assert "stopped sending data" in str(exc.value)


def test_xai_provider_strips_whitespace_base_url_override(monkeypatch):
    captured_urls: list[str] = []
    response = FakeStreamResponse(
        json_body={"output_text": json.dumps({"value": "ok"})},
        headers={"content-type": "application/json"},
    )

    def fake_stream(self, method, url, **kwargs):
        captured_urls.append(url)
        return nullcontext(response)

    monkeypatch.setattr(httpx.Client, "stream", fake_stream)
    payload = XAIProvider().invoke(
        _operation(),
        ProviderConfig(
            provider="xAI",
            provider_model="grok-4-latest",
            api_key="secret-key",
            provider_base_url="   ",
        ),
    )
    assert payload == {"value": "ok"}
    assert captured_urls == ["https://api.x.ai/v1/responses"]


def test_xai_provider_rejects_unknown_operation():
    class UnknownOperation:
        kind = "unknown"

    with pytest.raises(UnsupportedOperationError):
        XAIProvider().render_operation(UnknownOperation(), _config())


def test_xai_provider_rejects_unknown_model():
    with pytest.raises(ProviderError):
        XAIProvider().render_operation(_operation(), _config(provider_model="not-a-real-model"))
