from __future__ import annotations

from dataclasses import dataclass

import pytest

import bigplayer_prompting.capabilities as capabilities
from bigplayer_prompting.capabilities import (
    BASIC_PROMPT_CAPABILITY,
    CHECKPOINT_PICKER_CAPABILITY,
    KSAMPLER_CONFIG_CAPABILITY,
    MODEL_CONTEXT_CAPABILITY,
)
from bigplayer_prompting.errors import MalformedProviderResponseError, ProviderError
from bigplayer_prompting.operations import OperationKind
from bigplayer_prompting.provider import InvocationContext
from bigplayer_prompting.service import LLMProviderBundle, PromptGenerationService


class FakeDynPrompt:
    def __init__(self, prompt: dict):
        self._prompt = prompt

    def get_original_prompt(self) -> dict:
        return self._prompt


class FakeProvider:
    def __init__(self):
        self.operations = []
        self.configs = []
        self.contexts = []

    def invoke(self, operation, config, context=None):
        self.operations.append(operation)
        self.configs.append(config)
        self.contexts.append(context)
        payload: dict[str, dict] = {}
        if BASIC_PROMPT_CAPABILITY in operation.requested_capabilities:
            payload[BASIC_PROMPT_CAPABILITY] = {
                "positive_prompt": "cinematic cat portrait",
                "negative_prompt": "blurry",
                "comments": "Used cinematic phrasing.",
            }
        if KSAMPLER_CONFIG_CAPABILITY in operation.requested_capabilities:
            config_data = operation.capability_configs[KSAMPLER_CONFIG_CAPABILITY]
            payload[KSAMPLER_CONFIG_CAPABILITY] = {
                "steps": 24,
                "cfg": 7.5,
                "sampler_name": config_data["sampler_names"][0],
                "scheduler": config_data["scheduler_names"][0],
                "denoise": 1.0,
                "comments": "Balanced quality and speed.",
            }
        if CHECKPOINT_PICKER_CAPABILITY in operation.requested_capabilities:
            checkpoints = operation.capability_configs[CHECKPOINT_PICKER_CAPABILITY]["available_checkpoints"]
            payload[CHECKPOINT_PICKER_CAPABILITY] = {
                "checkpoint_name": checkpoints[0],
                "comments": "Best match for the prose.",
            }
        return payload


def _provider_bundle(assume_determinism: bool = True) -> LLMProviderBundle:
    return LLMProviderBundle(
        api_key="secret-key",
        provider="xAI",
        provider_model="grok-4-latest",
        assume_determinism=assume_determinism,
    )


def _prompt(*nodes: tuple[str, str, dict]) -> FakeDynPrompt:
    prompt = {}
    for node_id, class_type, inputs in nodes:
        prompt[node_id] = {"class_type": class_type, "inputs": inputs}
    return FakeDynPrompt(prompt)


def test_service_discovers_root_modules_and_builds_composed_operation(monkeypatch):
    monkeypatch.setattr(capabilities, "list_sampler_names", lambda: ["euler"])
    monkeypatch.setattr(capabilities, "list_scheduler_names", lambda: ["karras"])
    monkeypatch.setattr(capabilities, "list_available_checkpoints", lambda: ["sdxl-base.safetensors"])
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})

    session = service.begin_session(
        prose="A cinematic portrait of a cat.",
        provider_bundle=_provider_bundle(),
        dynprompt=_prompt(
            ("2", "BigPlayerBasicPrompt", {"session": ["root", 0]}),
            ("3", "BigPlayerKSamplerConfig", {"session": ["root", 0]}),
            ("4", "BigPlayerCheckpointPicker", {"session": ["root", 0]}),
            ("5", "BigPlayerModelContext", {"session": ["root", 0], "model_context": "Use an SDXL-style workflow."}),
            ("6", "BigPlayerBasicPrompt", {"session": ["other_root", 0]}),
        ),
        root_node_id="root",
    )

    operation = provider.operations[0]
    assert operation.kind is OperationKind.PROMPT_GENERATION
    assert operation.requested_capabilities == (
        BASIC_PROMPT_CAPABILITY,
        CHECKPOINT_PICKER_CAPABILITY,
        KSAMPLER_CONFIG_CAPABILITY,
    )
    assert any("Use an SDXL-style workflow." in block[1] for block in operation.context_blocks)
    assert isinstance(provider.contexts[0], InvocationContext)
    assert session.root_node_id == "root"


def test_service_caches_when_assume_determinism_is_enabled(monkeypatch):
    monkeypatch.setattr(capabilities, "list_sampler_names", lambda: ["euler"])
    monkeypatch.setattr(capabilities, "list_scheduler_names", lambda: ["karras"])
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})
    dynprompt = _prompt(("2", "BigPlayerBasicPrompt", {"session": ["root", 0]}))

    first = service.begin_session(
        prose="A cinematic portrait of a cat.",
        provider_bundle=_provider_bundle(assume_determinism=True),
        dynprompt=dynprompt,
        root_node_id="root",
    )
    second = service.begin_session(
        prose="A cinematic portrait of a cat.",
        provider_bundle=_provider_bundle(assume_determinism=True),
        dynprompt=dynprompt,
        root_node_id="root",
    )
    assert first.cache_key == second.cache_key
    assert len(provider.operations) == 1


def test_service_reexecutes_when_assume_determinism_is_disabled():
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})
    dynprompt = _prompt(("2", "BigPlayerBasicPrompt", {"session": ["root", 0]}))

    service.begin_session(
        prose="A cinematic portrait of a cat.",
        provider_bundle=_provider_bundle(assume_determinism=False),
        dynprompt=dynprompt,
        root_node_id="root",
    )
    service.begin_session(
        prose="A cinematic portrait of a cat.",
        provider_bundle=_provider_bundle(assume_determinism=False),
        dynprompt=dynprompt,
        root_node_id="root",
    )
    assert len(provider.operations) == 2


def test_duplicate_identical_modules_share_one_capability_request():
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})

    session = service.begin_session(
        prose="A cinematic portrait of a cat.",
        provider_bundle=_provider_bundle(),
        dynprompt=_prompt(
            ("2", "BigPlayerBasicPrompt", {"session": ["root", 0]}),
            ("3", "BigPlayerBasicPrompt", {"session": ["root", 0]}),
        ),
        root_node_id="root",
    )
    resolved = service.resolve_capability(session, BASIC_PROMPT_CAPABILITY)
    assert provider.operations[0].requested_capabilities == (BASIC_PROMPT_CAPABILITY,)
    assert resolved.positive_prompt == "cinematic cat portrait"


def test_conflicting_duplicate_modules_fail_predictably():
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})

    with pytest.raises(ProviderError) as exc:
        service.begin_session(
            prose="A cinematic portrait of a cat.",
            provider_bundle=_provider_bundle(),
            dynprompt=_prompt(
                ("2", "BigPlayerModelContext", {"session": ["root", 0], "model_context": "SDXL"}),
                ("3", "BigPlayerModelContext", {"session": ["root", 0], "model_context": "Flux"}),
                ("4", "BigPlayerBasicPrompt", {"session": ["root", 0]}),
            ),
            root_node_id="root",
        )
    assert "Conflicting `model_context` modules" in str(exc.value)


def test_no_output_modules_fail_before_provider_call():
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})

    with pytest.raises(ProviderError) as exc:
        service.begin_session(
            prose="A cinematic portrait of a cat.",
            provider_bundle=_provider_bundle(),
            dynprompt=_prompt(
                ("2", "BigPlayerModelContext", {"session": ["root", 0], "model_context": "SDXL"}),
            ),
            root_node_id="root",
        )
    assert "at least one attached output module" in str(exc.value)
    assert provider.operations == []


def test_invalid_sampler_response_surfaces_schema_failure(monkeypatch):
    monkeypatch.setattr(capabilities, "list_sampler_names", lambda: ["euler"])
    monkeypatch.setattr(capabilities, "list_scheduler_names", lambda: ["karras"])

    class BadSamplerProvider(FakeProvider):
        def invoke(self, operation, config, context=None):
            return {
                "ksampler_config": {
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "bad",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "comments": "Invalid sampler.",
                }
            }

    service = PromptGenerationService(providers={"xAI": BadSamplerProvider()})
    with pytest.raises(MalformedProviderResponseError):
        service.begin_session(
            prose="A cinematic portrait of a cat.",
            provider_bundle=_provider_bundle(),
            dynprompt=_prompt(("2", "BigPlayerKSamplerConfig", {"session": ["root", 0]})),
            root_node_id="root",
        )


def test_cache_key_changes_when_checkpoint_inventory_changes(monkeypatch):
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})
    dynprompt = _prompt(("2", "BigPlayerCheckpointPicker", {"session": ["root", 0]}))

    monkeypatch.setattr(capabilities, "list_available_checkpoints", lambda: ["one.safetensors"])
    first = service.begin_session(
        prose="A cinematic portrait of a cat.",
        provider_bundle=_provider_bundle(),
        dynprompt=dynprompt,
        root_node_id="root",
    )

    monkeypatch.setattr(capabilities, "list_available_checkpoints", lambda: ["two.safetensors"])
    second = service.begin_session(
        prose="A cinematic portrait of a cat.",
        provider_bundle=_provider_bundle(),
        dynprompt=dynprompt,
        root_node_id="root",
    )

    assert first.cache_key != second.cache_key
    assert len(provider.operations) == 2
