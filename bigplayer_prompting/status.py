from __future__ import annotations

from dataclasses import dataclass

from .provider import InvocationContext


@dataclass
class ComfyStatusReporter:
    node_id: str | None
    _last_message: str = ""

    def as_invocation_context(self) -> InvocationContext:
        return InvocationContext(status_callback=self.report)

    def report(self, message: str) -> None:
        text = message.strip()
        if not text or text == self._last_message or not self.node_id:
            return
        self._last_message = text
        try:
            from server import PromptServer
        except Exception:
            return

        server_instance = getattr(PromptServer, "instance", None)
        if server_instance is None:
            return
        try:
            server_instance.send_progress_text(text, self.node_id, server_instance.client_id)
        except Exception:
            return
