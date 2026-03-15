from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from ..generation.capabilities import (
    BASIC_PROMPT_CAPABILITY,
    CHECKPOINT_PICKER_CAPABILITY,
    KSAMPLER_CONFIG_CAPABILITY,
    SPLIT_PROMPT_CAPABILITY,
)
from ..errors import MalformedProviderResponseError, ProviderError, UnsupportedOperationError
from ..generation.operations import OperationKind, PromptGenerationOperation
from .base import InvocationContext, ProviderConfig, redact_secret

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


@dataclass(frozen=True)
class XAIFragmentDefinition:
    capability_id: str
    schema_name: str
    build_schema: Callable[[dict[str, Any]], dict[str, Any]]
    build_prompt: Callable[[dict[str, Any]], str]


def _basic_prompt_schema(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["positive_prompt", "negative_prompt", "comments"],
        "properties": {
            "positive_prompt": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "comments": {"type": "string"},
        },
    }


def _split_prompt_schema(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "text_l_positive",
            "text_g_positive",
            "text_l_negative",
            "text_g_negative",
            "comments",
        ],
        "properties": {
            "text_l_positive": {"type": "string"},
            "text_g_positive": {"type": "string"},
            "text_l_negative": {"type": "string"},
            "text_g_negative": {"type": "string"},
            "comments": {"type": "string"},
        },
    }


def _ksampler_schema(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["steps", "cfg", "sampler_name", "scheduler", "denoise", "comments"],
        "properties": {
            "steps": {"type": "integer", "minimum": 1, "maximum": 10000},
            "cfg": {"type": "number", "minimum": 0.0, "maximum": 100.0},
            "sampler_name": {"type": "string", "enum": config["sampler_names"]},
            "scheduler": {"type": "string", "enum": config["scheduler_names"]},
            "denoise": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "comments": {"type": "string"},
        },
    }


def _checkpoint_schema(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["checkpoint_name", "comments"],
        "properties": {
            "checkpoint_name": {"type": "string", "enum": config["available_checkpoints"]},
            "comments": {"type": "string"},
        },
    }


def _basic_prompt_fragment(_: dict[str, Any]) -> str:
    return (
        "Capability `basic_prompt`:\n"
        "- Return `positive_prompt`, `negative_prompt`, and `comments`.\n"
        "- Positive prompts should be concise, specific, and workflow-ready.\n"
        "- Negative prompts should list unwanted content rather than explain policy."
    )


def _split_prompt_fragment(_: dict[str, Any]) -> str:
    return (
        "Capability `split_prompt`:\n"
        "- Return `text_l_positive`, `text_g_positive`, `text_l_negative`, `text_g_negative`, and `comments`.\n"
        "- `text_l_*` should focus on local subject/content details.\n"
        "- `text_g_*` should focus on broader mood, composition, and global styling.\n"
        "- Do not duplicate the full positive prompt into both positive channels unless you explicitly note the fallback in comments."
    )


def _ksampler_fragment(config: dict[str, Any]) -> str:
    return (
        "Capability `ksampler_config`:\n"
        "- Return `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`, and `comments` for a standard ComfyUI KSampler.\n"
        f"- Allowed sampler names: {', '.join(config['sampler_names'])}\n"
        f"- Allowed scheduler names: {', '.join(config['scheduler_names'])}\n"
        "- Choose practical settings that match the user's request."
    )


def _checkpoint_fragment(config: dict[str, Any]) -> str:
    return (
        "Capability `checkpoint_picker`:\n"
        "- Return `checkpoint_name` and `comments`.\n"
        "- Choose exactly one checkpoint from the allowed list.\n"
        f"- Allowed checkpoints: {', '.join(config['available_checkpoints'])}"
    )


XAI_FRAGMENT_REGISTRY: dict[str, XAIFragmentDefinition] = {
    BASIC_PROMPT_CAPABILITY: XAIFragmentDefinition(
        capability_id=BASIC_PROMPT_CAPABILITY,
        schema_name="basic_prompt",
        build_schema=_basic_prompt_schema,
        build_prompt=_basic_prompt_fragment,
    ),
    SPLIT_PROMPT_CAPABILITY: XAIFragmentDefinition(
        capability_id=SPLIT_PROMPT_CAPABILITY,
        schema_name="split_prompt",
        build_schema=_split_prompt_schema,
        build_prompt=_split_prompt_fragment,
    ),
    KSAMPLER_CONFIG_CAPABILITY: XAIFragmentDefinition(
        capability_id=KSAMPLER_CONFIG_CAPABILITY,
        schema_name="ksampler_config",
        build_schema=_ksampler_schema,
        build_prompt=_ksampler_fragment,
    ),
    CHECKPOINT_PICKER_CAPABILITY: XAIFragmentDefinition(
        capability_id=CHECKPOINT_PICKER_CAPABILITY,
        schema_name="checkpoint_picker",
        build_schema=_checkpoint_schema,
        build_prompt=_checkpoint_fragment,
    ),
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
        override_base_url = config.provider_base_url.strip()
        base_url = (override_base_url or XAI_PROVIDER_BASE_URL).rstrip("/")
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        context = context or InvocationContext()
        context.set_request_text(self.render_request_text(request))

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
        except httpx.UnsupportedProtocol as exc:
            raise ProviderError(
                f"Invalid xAI provider base URL `{base_url}`. Expected an http:// or https:// URL."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("Failed to call the xAI provider.") from exc

        text = self._extract_text(body)
        context.set_response_text(text)
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

    def render_request_text(self, request: RenderedXAIRequest) -> str:
        schema_text = json.dumps(request.schema, sort_keys=True, indent=2, ensure_ascii=True)
        sections = [
            f"Model:\n{request.model}",
            f"System prompt:\n{request.system_prompt.strip()}",
            f"User prompt:\n{request.user_prompt.strip()}",
            f"Response schema ({request.schema_name}):\n{schema_text}",
        ]
        return "\n\n".join(section for section in sections if section.strip())

    def _render_prompt_generation(
        self,
        operation: PromptGenerationOperation,
        config: ProviderConfig,
    ) -> RenderedXAIRequest:
        response_schema = self._build_response_schema(operation)
        capability_instructions = self._build_capability_instructions(operation)
        system_prompt = """You convert user prose into structured, production-ready ComfyUI workflow data.

You are not allowed to invent deterministic local logic that the node does not have.
Return only the schema requested by the caller.
Use the supplied workflow context when it is present.
Keep outputs practical and immediately usable in image workflows.
Every capability object must include concise comments explaining the main decision or fallback.
"""
        sections = [f"User prose:\n{operation.prose.strip()}"]

        if operation.context_blocks:
            for title, body in operation.context_blocks:
                sections.append(f"{title}:\n{body.strip()}")

        sections.append(f"Requested output capabilities:\n{', '.join(operation.requested_capabilities)}")
        sections.append(
            "Capability requirements:\n"
            + "\n\n".join(instruction.strip() for instruction in capability_instructions if instruction.strip())
        )
        sections.append(
            "Global requirements:\n"
            "- Keep each capability response concise and workflow-ready.\n"
            "- Return only data matching the requested schema.\n"
            "- Do not add capabilities or fields that were not requested."
        )
        user_prompt = "\n\n".join(section for section in sections if section.strip())

        return RenderedXAIRequest(
            model=config.provider_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="bigplayer_modular_llm_result",
            schema=response_schema,
        )

    def _build_response_schema(self, operation: PromptGenerationOperation) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for capability_id in operation.requested_capabilities:
            fragment = self._fragment(capability_id)
            properties[fragment.schema_name] = fragment.build_schema(operation.capability_configs[capability_id])
            required.append(fragment.schema_name)
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }

    def _build_capability_instructions(self, operation: PromptGenerationOperation) -> list[str]:
        instructions: list[str] = []
        for capability_id in operation.requested_capabilities:
            fragment = self._fragment(capability_id)
            instructions.append(fragment.build_prompt(operation.capability_configs[capability_id]))
        return instructions

    def _fragment(self, capability_id: str) -> XAIFragmentDefinition:
        fragment = XAI_FRAGMENT_REGISTRY.get(capability_id)
        if fragment is None:
            raise UnsupportedOperationError(f"xAI provider does not support capability `{capability_id}`.")
        return fragment

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
