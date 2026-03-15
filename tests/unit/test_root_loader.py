from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

from bigplayer_prompting.nodes import BigPlayerControlNetState, BigPlayerLoRAState


def test_root_loader_supports_comfyui_style_import():
    repo_root = Path(__file__).resolve().parents[2]
    init_path = repo_root / "__init__.py"

    script = textwrap.dedent(
        f"""
        import importlib.util

        init_path = r"{init_path}"
        spec = importlib.util.spec_from_file_location("custom_node_under_test", init_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert "BigPlayerLLMProvider" in module.NODE_CLASS_MAPPINGS
        assert "BigPlayerCheckpointState" in module.NODE_CLASS_MAPPINGS
        assert "BigPlayerLoRAState" in module.NODE_CLASS_MAPPINGS
        assert "BigPlayerControlNetState" in module.NODE_CLASS_MAPPINGS
        assert "BigPlayerNaturalLanguageRoot" in module.NODE_CLASS_MAPPINGS
        assert "BigPlayerBasicPrompt" in module.NODE_CLASS_MAPPINGS
        assert module.WEB_DIRECTORY == "web"
        print("ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "ok"


def test_state_nodes_accept_wildcard_inputs():
    assert BigPlayerLoRAState.VALIDATE_INPUTS(input_types={"lora_syntax_input": "STRING"}) is True
    assert BigPlayerControlNetState.VALIDATE_INPUTS(input_types={"controlnets_input": "STRING"}) is True
