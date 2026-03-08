from __future__ import annotations

from contextlib import contextmanager
import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import pytest


pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
COMFY_ROOT = REPO_ROOT / ".integration" / "ComfyUI"
SUPPORT_NODE_DIR = REPO_ROOT / "tests" / "support_nodes" / "model_stub_pack"
SUPPORT_LINK = COMFY_ROOT / "custom_nodes" / "bigplayer-test-support"


def _find_free_port() -> int:
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
    port = _find_free_port()
    _ProviderHandler.responder = responder
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _ProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@contextmanager
def comfy_server():
    SUPPORT_LINK.unlink(missing_ok=True)
    SUPPORT_LINK.symlink_to(SUPPORT_NODE_DIR)
    port = _find_free_port()
    proc = subprocess.Popen(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--cpu",
            "--disable-auto-launch",
            "--disable-api-nodes",
            "--base-directory",
            str(COMFY_ROOT),
        ],
        cwd=COMFY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        started = False
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/object_info", timeout=2) as response:
                    payload = json.loads(response.read())
                    if "BigPlayerPromptSimple" in payload and "BigPlayerPromptSplit" in payload:
                        started = True
                        break
            except Exception:
                time.sleep(1)
        if not started:
            output = ""
            if proc.stdout is not None:
                output = proc.stdout.read()
            raise RuntimeError(f"ComfyUI server did not start successfully.\n{output}")

        yield port
    finally:
        proc.kill()
        proc.wait(timeout=10)
        SUPPORT_LINK.unlink(missing_ok=True)


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


def build_simple_workflow(provider_base_url: str) -> dict:
    return {
        "1": {
            "class_type": "BigPlayerTestModel",
            "inputs": {"model_name": "sdxl-base-1.0.safetensors"},
        },
        "2": {
            "class_type": "BigPlayerPromptSimple",
            "inputs": {
                "prose": "A cinematic portrait of a cat in a film still.",
                "api_key": "test-key",
                "provider": "xAI",
                "provider_model": "grok-4-latest",
                "target_model": ["1", 0],
                "style_policy": "Keep it practical.",
                "provider_base_url": provider_base_url,
                "assume_determinism": True,
            },
        },
        "3": {
            "class_type": "BigPlayerTestSink",
            "inputs": {
                "value_1": ["2", 0],
                "value_2": ["2", 1],
                "value_3": ["2", 2],
            },
        },
    }


def build_split_workflow(provider_base_url: str) -> dict:
    return {
        "1": {
            "class_type": "BigPlayerTestModel",
            "inputs": {"model_name": "sdxl-base-1.0.safetensors"},
        },
        "2": {
            "class_type": "BigPlayerPromptSplit",
            "inputs": {
                "prose": "A cinematic portrait of a cat in a film still.",
                "api_key": "test-key",
                "provider": "xAI",
                "provider_model": "grok-4-latest",
                "target_model": ["1", 0],
                "style_policy": "",
                "provider_base_url": provider_base_url,
                "assume_determinism": True,
            },
        },
        "3": {
            "class_type": "BigPlayerTestSplitSink",
            "inputs": {
                "value_1": ["2", 0],
                "value_2": ["2", 1],
                "value_3": ["2", 2],
                "value_4": ["2", 3],
                "value_5": ["2", 4],
            },
        },
    }


def build_no_provider_simple_workflow(provider_model: str, prose: str) -> dict:
    return {
        "1": {
            "class_type": "BigPlayerTestModel",
            "inputs": {"model_name": "sdxl-base-1.0.safetensors"},
        },
        "2": {
            "class_type": "BigPlayerPromptSimple",
            "inputs": {
                "prose": prose,
                "api_key": "",
                "provider": "No Provider",
                "provider_model": provider_model,
                "target_model": ["1", 0],
                "style_policy": "",
                "provider_base_url": "",
                "assume_determinism": True,
            },
        },
        "3": {
            "class_type": "BigPlayerTestSink",
            "inputs": {
                "value_1": ["2", 0],
                "value_2": ["2", 1],
                "value_3": ["2", 2],
            },
        },
    }


def test_simple_and_split_nodes_execute_in_comfyui():
    def responder(body):
        assert body["stream"] is True
        user_text = body["input"][1]["content"][0]["text"]
        if "Output mode:\nsimple" in user_text:
            data = {
                "positive_prompt": "cinematic cat portrait, shallow depth of field",
                "negative_prompt": "blurry, distorted, low quality",
                "comments": "Used the SDXL checkpoint name to bias toward cinematic phrasing.",
            }
        else:
            data = {
                "text_l_positive": "cat portrait",
                "text_g_positive": "cinematic film still, shallow depth of field",
                "text_l_negative": "blurry",
                "text_g_negative": "distorted, low quality",
                "comments": "Separated content and global style cues.",
            }
        return {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(data),
                        }
                    ]
                }
            ]
        }

    with mock_provider_server(responder) as provider_base_url:
        with comfy_server() as port:
            simple_id = queue_prompt(port, build_simple_workflow(provider_base_url))
            simple_history = wait_for_history(port, simple_id)
            simple_output = simple_history["outputs"]["3"]
            assert simple_output["value_1"][0] == "cinematic cat portrait, shallow depth of field"
            assert simple_output["value_3"][0].startswith("Used the SDXL checkpoint name")

            split_id = queue_prompt(port, build_split_workflow(provider_base_url))
            split_history = wait_for_history(port, split_id)
            split_output = split_history["outputs"]["3"]
            assert split_output["value_1"][0] == "cat portrait"
            assert split_output["value_5"][0] == "Separated content and global style cues."


def test_comfyui_surface_schema_failures():
    def responder(body):
        assert body["stream"] is True
        return {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"positive_prompt": "only-one-field"}),
                        }
                    ]
                }
            ]
        }

    with mock_provider_server(responder) as provider_base_url:
        with comfy_server() as port:
            prompt_id = queue_prompt(port, build_simple_workflow(provider_base_url))
            history = wait_for_history(port, prompt_id, timeout_seconds=10)
            assert history["status"]["status_str"] == "error"
            error_message = history["status"]["messages"][-1][1]["exception_message"]
            assert "schema validation" in error_message


def test_no_provider_executes_locally_without_network():
    with comfy_server() as port:
        positive_id = queue_prompt(
            port,
            build_no_provider_simple_workflow("Positive", "bright red apple"),
        )
        positive_history = wait_for_history(port, positive_id)
        positive_output = positive_history["outputs"]["3"]
        assert positive_output["value_1"][0] == "bright red apple"
        assert positive_output["value_2"][0] == ""
        assert positive_output["value_3"][0] == "Goes nowhere, does nothing"

        negative_id = queue_prompt(
            port,
            build_no_provider_simple_workflow("Negative", "washed out"),
        )
        negative_history = wait_for_history(port, negative_id)
        negative_output = negative_history["outputs"]["3"]
        assert negative_output["value_1"][0] == ""
        assert negative_output["value_2"][0] == "washed out"
        assert negative_output["value_3"][0] == "Goes nowhere, does nothing"
