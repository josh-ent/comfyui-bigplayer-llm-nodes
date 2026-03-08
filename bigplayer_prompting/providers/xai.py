from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from ..errors import MalformedProviderResponseError, ProviderError, UnsupportedOperationError
from ..operations import OperationKind, PromptGenerationOperation
from ..provider import InvocationContext, ProviderConfig, redact_secret

XAI_PROVIDER_ID = "xAI"
XAI_PROVIDER_BASE_URL = "https://api.x.ai/v1"
XAI_MODELS = (
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-latest",
)
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 300.0
WRITE_TIMEOUT_SECONDS = 30.0
POOL_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class RenderedXAIRequest:
    model: str
    system_prompt: str
    user_prompt: str
    schema_name: str
    schema: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "store": False,
            "stream": True,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": self.user_prompt}],
                },
            ],
            "tools": [{"type": "web_search"}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "schema": self.schema,
                    "strict": True,
                }
            },
        }


class XAIProvider:
    def __init__(
        self,
        *,
        connect_timeout_seconds: float = CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds: float = READ_TIMEOUT_SECONDS,
        write_timeout_seconds: float = WRITE_TIMEOUT_SECONDS,
        pool_timeout_seconds: float = POOL_TIMEOUT_SECONDS,
    ) -> None:
        self._timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=write_timeout_seconds,
            pool=pool_timeout_seconds,
        )

    def invoke(
        self,
        operation: Any,
        config: ProviderConfig,
        context: InvocationContext | None = None,
    ) -> dict[str, Any]:
        request = self.render_operation(operation, config)
        base_url = (config.provider_base_url or XAI_PROVIDER_BASE_URL).rstrip("/")
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        context = context or InvocationContext()

        try:
            context.report_status("Connecting to xAI...")
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream(
                    "POST",
                    f"{base_url}/responses",
                    headers=headers,
                    json=request.as_payload(),
                ) as response:
                    response.raise_for_status()
                    body = self._read_response_body(response, context)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "Timed out while waiting for the xAI provider response. "
                "The request uses a long idle read timeout, so this usually means the provider "
                "stopped sending data rather than simply taking time to think."
            ) from exc
        except httpx.HTTPStatusError as exc:
            message = self._response_text(exc.response).strip()
            raise ProviderError(
                f"xAI provider returned HTTP {exc.response.status_code}. "
                f"Body: {message.replace(config.api_key, redact_secret(config.api_key))}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("Failed to call the xAI provider.") from exc

        text = self._extract_text(body)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedProviderResponseError("Provider returned non-JSON structured output.") from exc

        if not isinstance(decoded, dict):
            raise MalformedProviderResponseError("Provider returned a non-object structured payload.")
        return decoded

    def render_operation(self, operation: Any, config: ProviderConfig) -> RenderedXAIRequest:
        self._validate_model(config.provider_model)
        kind = getattr(operation, "kind", None)
        if kind == OperationKind.PROMPT_GENERATION:
            return self._render_prompt_generation(operation, config)
        raise UnsupportedOperationError(f"xAI provider does not support operation `{kind}`.")

    def _render_prompt_generation(
        self,
        operation: PromptGenerationOperation,
        config: ProviderConfig,
    ) -> RenderedXAIRequest:
        if operation.output_mode == "simple":
            output_instructions = "Return `positive_prompt`, `negative_prompt`, and `comments`."
        else:
            output_instructions = (
                "Return `text_l_positive`, `text_g_positive`, `text_l_negative`, "
                "`text_g_negative`, and `comments`. Do not duplicate the full prompt into both "
                "positive channels unless you explicitly treat that as a fallback and say so in comments."
            )

        style_section = operation.style_policy.strip() or "No extra style policy supplied."
        system_prompt = """You convert user prose into production-ready ComfyUI prompts.

You are not allowed to invent deterministic local logic that the node does not have.
Return only the schema requested by the caller.
Use the provided model name to tailor the prompt. When useful, use web search to look up
that model on CivitAI or HuggingFace before deciding on prompt shape.
Keep prompt text practical and immediately usable in image or video workflows.
Comments must explain how the model name influenced the result and note any fallback.
"""
        user_prompt = f"""User prose:
{operation.prose.strip()}

Target model name:
{operation.target_model_name}

Style or policy guidance:
{style_section}

Output mode:
{operation.output_mode}

Requirements:
- Tailor the prompt to the target model name.
- Keep the output concise, specific, and workflow-ready.
- Negative prompts should omit content instead of explaining policy.
- {output_instructions}
- Return only data matching the requested schema.
"""

        return RenderedXAIRequest(
            model=config.provider_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=operation.response_schema_name,
            schema=operation.response_schema,
        )

    def _read_response_body(self, response: httpx.Response, context: InvocationContext) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            context.report_status("Receiving non-streamed xAI response...")
            response.read()
            return response.json()

        context.report_status("Waiting for first streamed xAI event...")
        stream = _StreamAccumulator(context)
        event_name: str | None = None
        data_lines: list[str] = []

        for raw_line in response.iter_lines():
            line = raw_line.strip()
            if not line:
                stream.consume_event(event_name, "\n".join(data_lines))
                event_name = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines:
            stream.consume_event(event_name, "\n".join(data_lines))
        return stream.final_body()

    def _extract_text(self, body: dict[str, Any]) -> str:
        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        for item in body.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        raise MalformedProviderResponseError("Provider response did not contain structured output text.")

    def _validate_model(self, provider_model: str) -> None:
        if provider_model not in XAI_MODELS:
            raise ProviderError(f"xAI provider does not support model `{provider_model}`.")

    def _response_text(self, response: httpx.Response) -> str:
        try:
            return response.text
        except httpx.ResponseNotRead:
            response.read()
            return response.text


class _StreamAccumulator:
    def __init__(self, context: InvocationContext) -> None:
        self._context = context
        self._text_fragments: list[str] = []
        self._final_response: dict[str, Any] | None = None
        self._event_count = 0

    def consume_event(self, event_name: str | None, payload: str) -> None:
        if not payload:
            return
        if payload == "[DONE]":
            self._context.report_status("xAI stream finished.")
            return

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return

        self._event_count += 1
        event_type = str(event.get("type") or event_name or "")

        if event_type.endswith("output_text.delta"):
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                self._text_fragments.append(delta)
                self._context.report_status(
                    f"Receiving structured output from xAI... {len(''.join(self._text_fragments))} chars"
                )
            return

        response_obj = event.get("response")
        if isinstance(response_obj, dict):
            self._final_response = response_obj
            if event_type.endswith("completed"):
                self._context.report_status("xAI stream completed. Finalizing response...")
            return

        output_text = event.get("output_text")
        if isinstance(output_text, str) and output_text:
            self._text_fragments.append(output_text)
            self._context.report_status(
                f"Receiving structured output from xAI... {len(''.join(self._text_fragments))} chars"
            )

    def final_body(self) -> dict[str, Any]:
        if self._final_response is not None:
            return self._final_response
        if self._text_fragments:
            return {"output_text": "".join(self._text_fragments)}
        raise MalformedProviderResponseError(
            "Provider response stream ended without any structured output text."
        )
