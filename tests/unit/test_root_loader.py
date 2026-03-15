from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

from bigplayer import LLMProviderBundle, LLMSessionHandle, PresetConfigBundle, PresetLora
from bigplayer.nodes import BigPlayerControlNetState, BigPlayerLoRAState
from bigplayer.nodes.prompting import (
    BigPlayerBasicPrompt,
    BigPlayerCheckpointPicker,
    BigPlayerKSamplerConfig,
    BigPlayerLLMProvider,
    BigPlayerNaturalLanguageRoot,
    BigPlayerSplitPrompt,
)


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
    assert BigPlayerLoRAState.VALIDATE_INPUTS(input_types={"lora_syntax_also": "STRING"}) is True
    assert BigPlayerControlNetState.VALIDATE_INPUTS(input_types={"controlnets_also": "STRING"}) is True


def test_structure_definitions_are_reexported():
    assert LLMProviderBundle.__name__ == "LLMProviderBundle"
    assert LLMSessionHandle.__name__ == "LLMSessionHandle"
    assert PresetConfigBundle.__name__ == "PresetConfigBundle"
    assert PresetLora.__name__ == "PresetLora"


def test_node_categories_match_the_intended_menu_structure():
    assert BigPlayerLLMProvider.CATEGORY == "BigPlayer"
    assert BigPlayerNaturalLanguageRoot.CATEGORY == "BigPlayer/Prompting"
    assert BigPlayerBasicPrompt.CATEGORY == "BigPlayer/Prompting/Capabilities"
    assert BigPlayerSplitPrompt.CATEGORY == "BigPlayer/Prompting/Capabilities"
    assert BigPlayerKSamplerConfig.CATEGORY == "BigPlayer/Prompting/Capabilities"
    assert BigPlayerCheckpointPicker.CATEGORY == "BigPlayer/Prompting/Capabilities"
    assert BigPlayerLoRAState.CATEGORY == "BigPlayer/State Indication"
    assert BigPlayerControlNetState.CATEGORY == "BigPlayer/State Indication"
