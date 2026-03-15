from .prompting import (
    BigPlayerBasicPrompt,
    BigPlayerCheckpointPicker,
    BigPlayerKSamplerConfig,
    BigPlayerLLMProvider,
    BigPlayerNaturalLanguageRoot,
    BigPlayerSplitPrompt,
)
from .state_indication import (
    BigPlayerCheckpointState,
    BigPlayerControlNetState,
    BigPlayerLoRAState,
)


NODE_CLASS_MAPPINGS = {
    "BigPlayerLLMProvider": BigPlayerLLMProvider,
    "BigPlayerNaturalLanguageRoot": BigPlayerNaturalLanguageRoot,
    "BigPlayerBasicPrompt": BigPlayerBasicPrompt,
    "BigPlayerSplitPrompt": BigPlayerSplitPrompt,
    "BigPlayerKSamplerConfig": BigPlayerKSamplerConfig,
    "BigPlayerCheckpointPicker": BigPlayerCheckpointPicker,
    "BigPlayerCheckpointState": BigPlayerCheckpointState,
    "BigPlayerLoRAState": BigPlayerLoRAState,
    "BigPlayerControlNetState": BigPlayerControlNetState,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BigPlayerLLMProvider": "BigPlayer LLM Provider",
    "BigPlayerNaturalLanguageRoot": "BigPlayer Natural Language Root",
    "BigPlayerBasicPrompt": "BigPlayer Basic Prompt",
    "BigPlayerSplitPrompt": "BigPlayer Split Prompt",
    "BigPlayerKSamplerConfig": "BigPlayer KSampler Config",
    "BigPlayerCheckpointPicker": "BigPlayer Checkpoint Picker",
    "BigPlayerCheckpointState": "BigPlayer Checkpoint State",
    "BigPlayerLoRAState": "BigPlayer LoRA State",
    "BigPlayerControlNetState": "BigPlayer ControlNet State",
}

__all__ = [
    "BigPlayerBasicPrompt",
    "BigPlayerCheckpointPicker",
    "BigPlayerCheckpointState",
    "BigPlayerControlNetState",
    "BigPlayerKSamplerConfig",
    "BigPlayerLLMProvider",
    "BigPlayerLoRAState",
    "BigPlayerNaturalLanguageRoot",
    "BigPlayerSplitPrompt",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
