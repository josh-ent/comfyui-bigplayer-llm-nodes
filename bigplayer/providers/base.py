from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


def redact_secret(secret: str) -> str:
    if not secret:
        return "<empty>"
    return f"<redacted:{len(secret)}>"


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: str
    models: tuple[str, ...]
    default_base_url: str
    factory: Callable[[], "OperationProvider"]
    requires_api_key: bool = True


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    provider_model: str
    api_key: str
    provider_base_url: str = ""


@dataclass
class ProviderDebugRecord:
    request_text: str = ""
    response_text: str = ""


@dataclass(frozen=True)
class InvocationContext:
    status_callback: Callable[[str], None] | None = None
    debug_record: ProviderDebugRecord | None = None

    def report_status(self, message: str) -> None:
        if self.status_callback is not None and message:
            self.status_callback(message)

    def set_request_text(self, text: str) -> None:
        if self.debug_record is not None:
            self.debug_record.request_text = text

    def set_response_text(self, text: str) -> None:
        if self.debug_record is not None:
            self.debug_record.response_text = text


class OperationProvider(Protocol):
    def invoke(
        self,
        operation: Any,
        config: ProviderConfig,
        context: InvocationContext | None = None,
    ) -> dict[str, Any]:
        ...
