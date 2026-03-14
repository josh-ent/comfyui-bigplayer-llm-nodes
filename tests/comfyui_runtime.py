from __future__ import annotations

from contextlib import contextmanager
import hashlib
import http.server
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_NODE_DIR = REPO_ROOT / "tests" / "support_nodes" / "model_stub_pack"
DOCKERFILE_PATH = REPO_ROOT / "tests" / "integration" / "comfyui" / "Dockerfile"
COMFYUI_REF = "v0.16.4"
IMAGE_TAG = "bigplayer-comfyui-integration:" + hashlib.blake2b(
    (COMFYUI_REF + DOCKERFILE_PATH.read_text(encoding="utf-8")).encode("utf-8"),
    digest_size=8,
).hexdigest()
REQUIRED_NODES = {
    "BigPlayerLLMProvider",
    "BigPlayerNaturalLanguageRoot",
    "BigPlayerBasicPrompt",
    "BigPlayerSplitPrompt",
    "BigPlayerKSamplerConfig",
    "BigPlayerCheckpointPicker",
    "BigPlayerModelContext",
    "BigPlayerTestSink",
    "BigPlayerTestPairSink",
    "BigPlayerTestKSamplerSink",
}


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ProviderHandler(http.server.BaseHTTPRequestHandler):
    responder = None

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        response = type(self).responder(body)
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args, **kwargs):
        return


@contextmanager
def mock_provider_server(responder):
    port = find_free_port()
    _ProviderHandler.responder = responder
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _ProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://host.docker.internal:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def ensure_test_image() -> None:
    image_check = _docker("image", "inspect", IMAGE_TAG, check=False)
    if image_check.returncode == 0:
        return

    _docker(
        "build",
        "--build-arg",
        f"COMFYUI_REF={COMFYUI_REF}",
        "-t",
        IMAGE_TAG,
        "-f",
        str(DOCKERFILE_PATH),
        str(DOCKERFILE_PATH.parent),
    )


@contextmanager
def comfy_server(*, checkpoints: tuple[str, ...] = (), port: int | None = None):
    ensure_test_image()
    bound_port = port if port is not None else find_free_port()
    container_name = f"bigplayer-comfyui-{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="bigplayer-comfyui-runtime-") as runtime_root:
        runtime_path = Path(runtime_root)
        checkpoint_dir = runtime_path / "models" / "checkpoints"
        custom_nodes_dir = runtime_path / "custom_nodes"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        custom_nodes_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint_name in checkpoints:
            (checkpoint_dir / checkpoint_name).touch()

        _docker(
            "run",
            "--rm",
            "--detach",
            "--name",
            container_name,
            "--add-host",
            "host.docker.internal:host-gateway",
            "--publish",
            f"127.0.0.1:{bound_port}:8188",
            "--mount",
            f"type=bind,src={REPO_ROOT},dst=/opt/bigplayer/runtime/custom_nodes/comfyui-bigplayer-llm-nodes,readonly",
            "--mount",
            f"type=bind,src={SUPPORT_NODE_DIR},dst=/opt/bigplayer/runtime/custom_nodes/bigplayer-test-support,readonly",
            "--mount",
            f"type=bind,src={runtime_path},dst=/opt/bigplayer/runtime",
            IMAGE_TAG,
        )

        try:
            started = False
            deadline = time.time() + 90
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{bound_port}/object_info", timeout=2) as response:
                        payload = json.loads(response.read())
                        if REQUIRED_NODES.issubset(payload):
                            started = True
                            break
                except Exception:
                    time.sleep(1)

            if not started:
                logs = _docker("logs", container_name, check=False)
                raise RuntimeError(f"ComfyUI container did not start successfully.\n{logs.stdout}{logs.stderr}")

            yield bound_port
        finally:
            _docker("stop", "--time", "5", container_name, check=False)


def queue_prompt(port: int, prompt: dict) -> str:
    data = json.dumps({"prompt": prompt, "client_id": "bigplayer-tests"}).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", data=data)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())["prompt_id"]


def wait_for_history(port: int, prompt_id: str, timeout_seconds: float = 30.0) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=5) as response:
                history = json.loads(response.read())
                if prompt_id in history:
                    return history[prompt_id]
        except urllib.error.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("Timed out waiting for ComfyUI prompt history.")
