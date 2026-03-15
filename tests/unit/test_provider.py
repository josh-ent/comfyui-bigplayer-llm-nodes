from __future__ import annotations

import json
from contextlib import nullcontext

import httpx
import pytest
import respx

from bigplayer.generation.capabilities import BASIC_PROMPT_CAPABILITY
from bigplayer.errors import MalformedProviderResponseError, ProviderError, UnsupportedOperationError
from bigplayer.generation.operations import PromptGenerationOperation
from bigplayer.providers.base import InvocationContext, ProviderConfig, ProviderDebugRecord
from bigplayer.providers.xai import XAIProvider


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
        context_blocks=(("Preset workflow config", "A photoreal SDXL workflow."),),
        requested_capabilities=(BASIC_PROMPT_CAPABILITY,),
        capability_configs={BASIC_PROMPT_CAPABILITY: {}},
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


def test_xai_provider_renders_generic_prompt_generation_operation():
    rendered = XAIProvider().render_operation(_operation(), _config())
    assert rendered.model == "grok-4-latest"
    assert "production-ready ComfyUI workflow data" in rendered.system_prompt
    assert "A cat on a windowsill." in rendered.user_prompt
    assert "A photoreal SDXL workflow." in rendered.user_prompt
    assert "basic_prompt" in rendered.user_prompt
    assert "positive_prompt" in rendered.user_prompt
    payload = rendered.as_payload()
    assert payload["tools"] == [
        {
            "type": "web_search",
            "filters": {
                "allowed_domains": ["civitai.com", "huggingface.co", "github.com"],
            },
        }
    ]
    assert payload["text"]["format"]["strict"] is True
    assert payload["stream"] is True
    assert "basic_prompt" in payload["text"]["format"]["schema"]["properties"]


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
                                "text": json.dumps({"basic_prompt": {"positive_prompt": "ok"}}),
                            }
                        ]
                    }
                ]
            },
            headers={"content-type": "application/json"},
        )
    )
    provider = XAIProvider()
    debug = ProviderDebugRecord()
    payload = provider.invoke(_operation(), _config(), InvocationContext(debug_record=debug))
    assert payload == {"basic_prompt": {"positive_prompt": "ok"}}
    assert "System prompt:" in debug.request_text
    assert "Response schema (bigplayer_modular_llm_result):" in debug.request_text
    assert debug.response_text == '{"basic_prompt": {"positive_prompt": "ok"}}'
    sent = route.calls[0].request.read().decode("utf-8")
    assert (
        '"tools":[{"type":"web_search","filters":{"allowed_domains":["civitai.com","huggingface.co","github.com"]}}]'
        in sent
    )
    assert '"stream":true' in sent
    assert "secret-key" not in sent


def test_xai_provider_accumulates_sse_output_and_reports_status(monkeypatch):
    response = FakeStreamResponse(
        headers={"content-type": "text/event-stream"},
        lines=[
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"{\\"basic_prompt\\": "}',
            "",
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"{\\"positive_prompt\\": \\"ok\\"}}"}',
            "",
            "data: [DONE]",
            "",
        ],
    )
    monkeypatch.setattr(httpx.Client, "stream", lambda self, *args, **kwargs: nullcontext(response))

    statuses: list[str] = []
    debug = ProviderDebugRecord()
    payload = XAIProvider().invoke(
        _operation(),
        _config(),
        InvocationContext(status_callback=statuses.append, debug_record=debug),
    )
    assert payload == {"basic_prompt": {"positive_prompt": "ok"}}
    assert any("Connecting to xAI" in status for status in statuses)
    assert any("Receiving structured output from xAI" in status for status in statuses)
    assert debug.response_text == '{"basic_prompt": {"positive_prompt": "ok"}}'


def test_xai_provider_prefers_completed_response_object_from_stream(monkeypatch):
    response = FakeStreamResponse(
        headers={"content-type": "text/event-stream"},
        lines=[
            'event: response.completed',
            'data: {"type":"response.completed","response":{"output_text":"{\\"basic_prompt\\": {\\"positive_prompt\\": \\"ok\\"}}"}}',
            "",
            "data: [DONE]",
            "",
        ],
    )
    monkeypatch.setattr(httpx.Client, "stream", lambda self, *args, **kwargs: nullcontext(response))
    payload = XAIProvider().invoke(_operation(), _config())
    assert payload == {"basic_prompt": {"positive_prompt": "ok"}}


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
        json_body={"output_text": json.dumps({"basic_prompt": {"positive_prompt": "ok"}})},
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
    assert payload == {"basic_prompt": {"positive_prompt": "ok"}}
    assert captured_urls == ["https://api.x.ai/v1/responses"]


def test_xai_provider_rejects_unknown_operation():
    class UnknownOperation:
        kind = "unknown"

    with pytest.raises(UnsupportedOperationError):
        XAIProvider().render_operation(UnknownOperation(), _config())


def test_xai_provider_rejects_unknown_model():
    with pytest.raises(ProviderError):
        XAIProvider().render_operation(_operation(), _config(provider_model="not-a-real-model"))
