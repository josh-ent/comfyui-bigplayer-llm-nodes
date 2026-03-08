from __future__ import annotations

from contextlib import contextmanager
import http.server
import json
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
                    required_nodes = {
                        "BigPlayerLLMProvider",
                        "BigPlayerLLMRoot",
                        "BigPlayerBasicPrompt",
                        "BigPlayerKSamplerConfig",
                        "BigPlayerCheckpointPicker",
                        "BigPlayerTestSink",
                        "BigPlayerTestPairSink",
                        "BigPlayerTestKSamplerSink",
                    }
                    if required_nodes.issubset(payload):
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


@contextmanager
def temp_checkpoint(name: str = "sdxl-base-1.0.safetensors"):
    checkpoint_dir = COMFY_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / name
    checkpoint_path.touch()
    try:
        yield checkpoint_path.name
    finally:
        checkpoint_path.unlink(missing_ok=True)


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


def build_modular_workflow(provider_base_url: str) -> dict:
    return {
        "1": {
            "class_type": "BigPlayerLLMProvider",
            "inputs": {
                "api_key": "test-key",
                "provider": "xAI",
                "provider_model": "grok-4-latest",
                "provider_base_url": provider_base_url,
                "assume_determinism": True,
            },
        },
        "2": {
            "class_type": "BigPlayerLLMRoot",
            "inputs": {
                "prose": "A cinematic portrait of a cat in a film still.",
                "provider_config": ["1", 0],
            },
        },
        "3": {
            "class_type": "BigPlayerBasicPrompt",
            "inputs": {
                "session": ["2", 0],
            },
        },
        "4": {
            "class_type": "BigPlayerKSamplerConfig",
            "inputs": {
                "session": ["2", 0],
            },
        },
        "5": {
            "class_type": "BigPlayerCheckpointPicker",
            "inputs": {
                "session": ["2", 0],
            },
        },
        "6": {
            "class_type": "BigPlayerTestSink",
            "inputs": {
                "value_1": ["3", 0],
                "value_2": ["3", 1],
                "value_3": ["3", 2],
            },
        },
        "7": {
            "class_type": "BigPlayerTestKSamplerSink",
            "inputs": {
                "value_1": ["4", 0],
                "value_2": ["4", 1],
                "value_3": ["4", 2],
                "value_4": ["4", 3],
                "value_5": ["4", 4],
                "value_6": ["4", 5],
            },
        },
        "8": {
            "class_type": "BigPlayerTestPairSink",
            "inputs": {
                "value_1": ["5", 0],
                "value_2": ["5", 1],
            },
        },
        "9": {
            "class_type": "BigPlayerModelContext",
            "inputs": {
                "session": ["2", 0],
                "model_context": "Use an SDXL-style workflow.",
            },
        },
    }


def build_no_provider_basic_workflow(provider_model: str, prose: str) -> dict:
    return {
        "1": {
            "class_type": "BigPlayerLLMProvider",
            "inputs": {
                "api_key": "",
                "provider": "No Provider",
                "provider_model": provider_model,
                "provider_base_url": "",
                "assume_determinism": True,
            },
        },
        "2": {
            "class_type": "BigPlayerLLMRoot",
            "inputs": {
                "prose": prose,
                "provider_config": ["1", 0],
            },
        },
        "3": {
            "class_type": "BigPlayerBasicPrompt",
            "inputs": {
                "session": ["2", 0],
            },
        },
        "4": {
            "class_type": "BigPlayerTestSink",
            "inputs": {
                "value_1": ["3", 0],
                "value_2": ["3", 1],
                "value_3": ["3", 2],
            },
        },
    }


def build_duplicate_prompt_workflow(provider_base_url: str) -> dict:
    return {
        "1": {
            "class_type": "BigPlayerLLMProvider",
            "inputs": {
                "api_key": "test-key",
                "provider": "xAI",
                "provider_model": "grok-4-latest",
                "provider_base_url": provider_base_url,
                "assume_determinism": True,
            },
        },
        "2": {
            "class_type": "BigPlayerLLMRoot",
            "inputs": {
                "prose": "A cinematic portrait of a cat in a film still.",
                "provider_config": ["1", 0],
            },
        },
        "3": {
            "class_type": "BigPlayerBasicPrompt",
            "inputs": {"session": ["2", 0]},
        },
        "4": {
            "class_type": "BigPlayerBasicPrompt",
            "inputs": {"session": ["2", 0]},
        },
        "5": {
            "class_type": "BigPlayerTestSink",
            "inputs": {
                "value_1": ["3", 0],
                "value_2": ["3", 1],
                "value_3": ["3", 2],
            },
        },
        "6": {
            "class_type": "BigPlayerTestSink",
            "inputs": {
                "value_1": ["4", 0],
                "value_2": ["4", 1],
                "value_3": ["4", 2],
            },
        },
    }


def build_multi_root_workflow(provider_base_url: str) -> dict:
    return {
        "1": {
            "class_type": "BigPlayerLLMProvider",
            "inputs": {
                "api_key": "test-key",
                "provider": "xAI",
                "provider_model": "grok-4-latest",
                "provider_base_url": provider_base_url,
                "assume_determinism": True,
            },
        },
        "2": {
            "class_type": "BigPlayerLLMRoot",
            "inputs": {
                "prose": "A cinematic portrait of a cat in a film still.",
                "provider_config": ["1", 0],
            },
        },
        "3": {
            "class_type": "BigPlayerBasicPrompt",
            "inputs": {"session": ["2", 0]},
        },
        "4": {
            "class_type": "BigPlayerTestSink",
            "inputs": {
                "value_1": ["3", 0],
                "value_2": ["3", 1],
                "value_3": ["3", 2],
            },
        },
        "5": {
            "class_type": "BigPlayerLLMRoot",
            "inputs": {
                "prose": "A dramatic portrait of a dog in rain.",
                "provider_config": ["1", 0],
            },
        },
        "6": {
            "class_type": "BigPlayerBasicPrompt",
            "inputs": {"session": ["5", 0]},
        },
        "7": {
            "class_type": "BigPlayerTestSink",
            "inputs": {
                "value_1": ["6", 0],
                "value_2": ["6", 1],
                "value_3": ["6", 2],
            },
        },
    }


def test_modular_nodes_execute_in_comfyui_with_one_provider_call():
    calls = {"count": 0}

    def responder(body):
        calls["count"] += 1
        assert body["stream"] is True
        user_text = body["input"][1]["content"][0]["text"]
        assert "Requested output capabilities" in user_text
        assert "sdxl-base-1.0.safetensors" in user_text
        data = {
            "basic_prompt": {
                "positive_prompt": "cinematic cat portrait, shallow depth of field",
                "negative_prompt": "blurry, distorted, low quality",
                "comments": "Used the model context to bias toward cinematic phrasing.",
            },
            "ksampler_config": {
                "steps": 28,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "comments": "Balanced detail and speed.",
            },
            "checkpoint_picker": {
                "checkpoint_name": "sdxl-base-1.0.safetensors",
                "comments": "Best fit for a cinematic SDXL portrait.",
            },
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

    with temp_checkpoint():
        with mock_provider_server(responder) as provider_base_url:
            with comfy_server() as port:
                prompt_id = queue_prompt(port, build_modular_workflow(provider_base_url))
                history = wait_for_history(port, prompt_id)
                assert calls["count"] == 1
                basic_output = history["outputs"]["6"]
                ksampler_output = history["outputs"]["7"]
                checkpoint_output = history["outputs"]["8"]
                assert basic_output["value_1"][0] == "cinematic cat portrait, shallow depth of field"
                assert basic_output["value_3"][0].startswith("Used the model context")
                assert ksampler_output["value_1"][0] == 28
                assert ksampler_output["value_3"][0] == "euler"
                assert checkpoint_output["value_1"][0] == "sdxl-base-1.0.safetensors"


def test_duplicate_prompt_modules_share_one_provider_call():
    calls = {"count": 0}

    def responder(body):
        calls["count"] += 1
        data = {
            "basic_prompt": {
                "positive_prompt": "cinematic cat portrait",
                "negative_prompt": "blurry",
                "comments": "Shared result.",
            }
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
            prompt_id = queue_prompt(port, build_duplicate_prompt_workflow(provider_base_url))
            history = wait_for_history(port, prompt_id)
            assert calls["count"] == 1
            first = history["outputs"]["5"]
            second = history["outputs"]["6"]
            assert first["value_1"][0] == second["value_1"][0]
            assert first["value_3"][0] == second["value_3"][0]


def test_multiple_roots_do_not_cross_contaminate():
    calls = {"count": 0}

    def responder(body):
        calls["count"] += 1
        user_text = body["input"][1]["content"][0]["text"]
        if "dog" in user_text:
            data = {
                "basic_prompt": {
                    "positive_prompt": "dramatic dog portrait",
                    "negative_prompt": "muddy",
                    "comments": "Dog result.",
                }
            }
        else:
            data = {
                "basic_prompt": {
                    "positive_prompt": "cinematic cat portrait",
                    "negative_prompt": "blurry",
                    "comments": "Cat result.",
                }
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
            prompt_id = queue_prompt(port, build_multi_root_workflow(provider_base_url))
            history = wait_for_history(port, prompt_id)
            assert calls["count"] == 2
            first = history["outputs"]["4"]
            second = history["outputs"]["7"]
            assert first["value_1"][0] == "cinematic cat portrait"
            assert second["value_1"][0] == "dramatic dog portrait"


def test_comfyui_surfaces_schema_failures_for_shared_sessions():
    def responder(body):
        assert body["stream"] is True
        return {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"basic_prompt": {"positive_prompt": "only-one-field"}}),
                        }
                    ]
                }
            ]
        }

    with mock_provider_server(responder) as provider_base_url:
        with comfy_server() as port:
            prompt_id = queue_prompt(port, build_duplicate_prompt_workflow(provider_base_url))
            history = wait_for_history(port, prompt_id, timeout_seconds=10)
            assert history["status"]["status_str"] == "error"
            error_message = history["status"]["messages"][-1][1]["exception_message"]
            assert "schema validation" in error_message


def test_no_provider_executes_locally_without_network():
    with comfy_server() as port:
        positive_id = queue_prompt(
            port,
            build_no_provider_basic_workflow("Positive", "bright red apple"),
        )
        positive_history = wait_for_history(port, positive_id)
        positive_output = positive_history["outputs"]["4"]
        assert positive_output["value_1"][0] == "bright red apple"
        assert positive_output["value_2"][0] == ""
        assert positive_output["value_3"][0] == "Goes nowhere, does nothing"

        negative_id = queue_prompt(
            port,
            build_no_provider_basic_workflow("Negative", "washed out"),
        )
        negative_history = wait_for_history(port, negative_id)
        negative_output = negative_history["outputs"]["4"]
        assert negative_output["value_1"][0] == ""
        assert negative_output["value_2"][0] == "washed out"
        assert negative_output["value_3"][0] == "Goes nowhere, does nothing"
