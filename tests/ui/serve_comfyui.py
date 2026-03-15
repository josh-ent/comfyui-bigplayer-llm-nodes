from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import threading

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.comfyui_runtime import comfy_server, ensure_test_image


DEFAULT_PORT = 18188
DEFAULT_CHECKPOINTS = ("sdxl-base-1.0.safetensors",)
UI_TEST_LABELS = {"bigplayer.ui_test": "1"}


def _watch_stdin_for_shutdown(shutdown: threading.Event) -> None:
    while not shutdown.is_set():
        chunk = sys.stdin.read(1)
        if chunk == "":
            shutdown.set()
            return


def main() -> None:
    port = int(os.environ.get("BIGPLAYER_COMFYUI_UI_TEST_PORT", str(DEFAULT_PORT)))
    shutdown = threading.Event()
    ensure_test_image()

    def handle_signal(signum, frame):
        del signum, frame
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    stdin_watcher = threading.Thread(
        target=_watch_stdin_for_shutdown,
        args=(shutdown,),
        daemon=True,
    )
    stdin_watcher.start()

    from tests.comfyui_runtime import remove_containers_by_labels

    remove_containers_by_labels(UI_TEST_LABELS)

    with comfy_server(port=port, checkpoints=DEFAULT_CHECKPOINTS, docker_labels=UI_TEST_LABELS):
        print(f"ComfyUI UI test server listening on http://127.0.0.1:{port}", flush=True)
        shutdown.wait()


if __name__ == "__main__":
    main()
