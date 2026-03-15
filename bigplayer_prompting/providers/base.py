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


@dataclass(frozen=True)
class InvocationContext:
    status_callback: Callable[[str], None] | None = None

    def report_status(self, message: str) -> None:
        if self.status_callback is not None and message:
            self.status_callback(message)


class OperationProvider(Protocol):
    def invoke(
        self,
        operation: Any,
        config: ProviderConfig,
        context: InvocationContext | None = None,
    ) -> dict[str, Any]:
        ...
